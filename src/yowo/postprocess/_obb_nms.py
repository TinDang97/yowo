"""OBB postprocessing: probiou NMS and OBBDetection assembly.

Implements probabilistic IoU (probiou) for rotated boxes using the Bhattacharyya
distance between Gaussian distributions, matching ultralytics OBB NMS exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from yowo.types import Frame, ModelSpec, OBBBox, OBBDetection, PreprocessedTensor

DOTA_CLASSES: list[str] = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]


def _get_covariance_matrix(boxes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Covariance matrix components from (N, 5) xywhr boxes.

    Args:
        boxes: ``(N, 5)`` — cx, cy, w, h, angle (radians).

    Returns:
        Tuple of ``(a, b, c)`` covariance components, each of shape ``(N,)``.
    """
    # Treat w, h as axis-aligned half-widths: variance = (w/2)^2 = w^2/12 * 3 — approximation
    # ultralytics: gbbs = cat(boxes[:, 2:4]**2 / 12, boxes[:, 4:5], dim=-1)
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:5]), dim=-1)
    a, b, c_angle = gbbs.unbind(-1)
    cos, sin = torch.cos(c_angle), torch.sin(c_angle)
    cos2, sin2, sincos = cos.pow(2), sin.pow(2), cos * sin
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * sincos


def probiou_matrix(obb1: Tensor, obb2: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute pairwise probabilistic IoU (probiou) between rotated boxes.

    Uses the Bhattacharyya distance between Gaussian distributions fitted to
    each rotated box. Matches ultralytics batch_probiou exactly.

    Args:
        obb1: ``(N, 5)`` — xywhr rotated boxes.
        obb2: ``(M, 5)`` — xywhr rotated boxes.
        eps: Numerical stability epsilon.

    Returns:
        ``(N, M)`` IoU matrix with values in [0, 1].
    """
    x1, y1 = obb1[:, 0:1], obb1[:, 1:2]  # (N, 1)
    x2, y2 = obb2[:, 0:1].T, obb2[:, 1:2].T  # (1, M)

    a1, b1, c1 = _get_covariance_matrix(obb1)  # each (N,)
    a2, b2, c2 = _get_covariance_matrix(obb2)  # each (M,)

    # Broadcast to (N, M) for pairwise computation
    a1, b1, c1 = a1[:, None], b1[:, None], c1[:, None]  # (N, 1)
    a2, b2, c2 = a2[None, :], b2[None, :], c2[None, :]  # (1, M)

    # Bhattacharyya distance components
    t1 = (
        ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2)
        / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
        * 0.25
    )
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
        / (4 * ((a1 * b1 - c1**2).clamp_(0) * (a2 * b2 - c2**2).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1.0 - hd


def _nms_rotated(
    boxes_xywhr: Tensor,
    scores: Tensor,
    iou_threshold: float,
) -> Tensor:
    """Greedy NMS using probiou for rotated boxes.

    Args:
        boxes_xywhr: ``(N, 5)`` — xywhr boxes.
        scores: ``(N,)`` — confidence scores.
        iou_threshold: Suppress boxes with IoU > this threshold.

    Returns:
        ``(K,)`` indices of kept boxes in original order.
    """
    order = scores.argsort(descending=True)
    kept: list[int] = []
    suppressed = torch.zeros(len(order), dtype=torch.bool, device=boxes_xywhr.device)

    for i in range(len(order)):
        idx = int(order[i].item())
        if suppressed[idx]:
            continue
        kept.append(idx)
        if i + 1 >= len(order):
            break
        rest_idx = order[i + 1 :]
        rest_idx = rest_idx[~suppressed[rest_idx]]
        if len(rest_idx) == 0:
            break
        iou = probiou_matrix(boxes_xywhr[idx : idx + 1], boxes_xywhr[rest_idx]).squeeze(0)
        suppress_mask = iou > iou_threshold
        suppressed[rest_idx[suppress_mask]] = True

    return torch.tensor(kept, dtype=torch.long, device=boxes_xywhr.device)


def postprocess_obb(
    raw: Tensor,
    frames: list[Frame],
    spec: ModelSpec,
    tensor_meta: PreprocessedTensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: list[str] | None = None,
) -> list[OBBDetection]:
    """Convert raw OBBHead output to a list of OBBDetection (one per frame in batch).

    Args:
        raw: ``(B, 4+nc+1, A)`` — raw OBBHead.forward() output.
        frames: Batch of source frames (length == B).
        spec: Model specification (provides num_classes).
        conf_threshold: Minimum class confidence for a detection to survive.
        iou_threshold: probiou threshold for NMS suppression.
        class_names: Override class name list; defaults to DOTA_CLASSES.

    Returns:
        List of ``OBBDetection``, one per frame in the batch.
    """
    names = class_names or DOTA_CLASSES
    nc = spec.num_classes or 15
    results: list[OBBDetection] = []

    for b in range(raw.shape[0]):
        pred = raw[b]  # (4+nc+1, A)

        # Inverse letterbox transform: remove padding, then divide by scale
        scale_h, _scale_w = tensor_meta.scale_factors[b]
        scale = scale_h  # uniform scale
        pad_top, pad_left = tensor_meta.pad_offsets[b]
        boxes_xywh = pred[:4].T  # (A, 4)
        cls_scores = pred[4 : 4 + nc].T  # (A, nc)
        angles = pred[4 + nc :].T  # (A, 1)

        # Per-anchor max class confidence
        conf, cls_ids = cls_scores.max(dim=1)  # (A,), (A,)
        mask = conf > conf_threshold

        if not mask.any():
            results.append(
                OBBDetection(
                    frame_index=frames[b].frame_index,
                    source_id=frames[b].source_id,
                    boxes=(),
                )
            )
            continue

        boxes_f = boxes_xywh[mask]  # (K, 4)
        angles_f = angles[mask, 0]  # (K,)
        conf_f = conf[mask]  # (K,)
        cls_ids_f = cls_ids[mask]  # (K,)

        # Class-aware offset trick: shift box centres by class_id * large_offset
        # so boxes from different classes never suppress each other
        class_offsets = cls_ids_f.float().unsqueeze(1) * 10000.0
        boxes_offset = boxes_f.clone()
        boxes_offset[:, :2] = boxes_offset[:, :2] + class_offsets
        xywhr = torch.cat([boxes_offset, angles_f.unsqueeze(1)], dim=1)  # (K, 5)

        kept = _nms_rotated(xywhr, conf_f, iou_threshold)

        obb_boxes: list[OBBBox] = []
        for k in kept.tolist():
            cid = int(cls_ids_f[k].item())

            raw_cx = float(boxes_f[k, 0].item())
            raw_cy = float(boxes_f[k, 1].item())
            raw_w = float(boxes_f[k, 2].item())
            raw_h = float(boxes_f[k, 3].item())

            # Inverse letterbox: remove pad offset, then remove scale
            cx_orig = (raw_cx - pad_left) / scale
            cy_orig = (raw_cy - pad_top) / scale
            w_orig = raw_w / scale
            h_orig = raw_h / scale

            obb_boxes.append(
                OBBBox(
                    cx=cx_orig,
                    cy=cy_orig,
                    w=w_orig,
                    h=h_orig,
                    angle=float(angles_f[k].item()),
                    confidence=float(conf_f[k].item()),
                    class_id=cid,
                    class_name=names[cid] if cid < len(names) else "",
                )
            )

        results.append(
            OBBDetection(
                frame_index=frames[b].frame_index,
                source_id=frames[b].source_id,
                boxes=tuple(obb_boxes),
            )
        )

    return results


__all__ = ["DOTA_CLASSES", "postprocess_obb", "probiou_matrix"]
