"""YOLO detection heads.

Implements the decoupled Detect head used by both YOLO11 and YOLO26.
YOLO11: ``reg_max=16``, ``end2end=False`` → requires NMS post-processing.
YOLO26: ``reg_max=1``, ``end2end=True``  → NMS-free top-k selection.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from yowo.arch._blocks import Conv, DWConv

__all__ = ["DFL", "Detect"]


# ---------------------------------------------------------------------------
# DFL — Distribution Focal Loss decode
# ---------------------------------------------------------------------------


class DFL(nn.Module):
    """Decode DFL distribution bins into expected distance values.

    Converts ``(B, 4*reg_max, N)`` → ``(B, 4, N)`` via weighted sum
    over a learned distribution.
    """

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x: Tensor) -> Tensor:
        b, _, a = x.shape
        # (B,4*c1,A) -> view -> transpose -> softmax(c1) -> conv -> (B,4,A)
        return self.conv(x.view(b, 4, self.c1, a).transpose(1, 2).contiguous().softmax(1)).view(
            b, 4, a
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_anchors(
    feats: list[Tensor],
    strides: Tensor,
    grid_cell_offset: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Generate anchor points and stride tensors for all feature map levels.

    Returns:
        anchor_points: ``(total_anchors, 2)`` — (x, y) centre of each grid cell.
        stride_tensor: ``(total_anchors, 1)`` — stride for each anchor.
    """
    anchor_points: list[Tensor] = []
    stride_tensor: list[Tensor] = []
    dtype, device = feats[0].dtype, feats[0].device

    for i, feat in enumerate(feats):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack([grid_x, grid_y], dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), strides[i].item(), dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(
    distance: Tensor,
    anchor_points: Tensor,
    xywh: bool = True,
) -> Tensor:
    """Convert distance predictions to bounding boxes.

    Args:
        distance: ``(B, 4, N)`` — left, top, right, bottom distances.
        anchor_points: ``(N, 2)`` — anchor centre coordinates.
        xywh: If True return ``(cx, cy, w, h)``; else ``(x1, y1, x2, y2)``.
    """
    lt, rb = distance.chunk(2, dim=1)
    anchors = anchor_points.transpose(0, 1).unsqueeze(0)  # (1, 2, N)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim=1)
    return torch.cat([x1y1, x2y2], dim=1)


# ---------------------------------------------------------------------------
# Detect — Decoupled detection head
# ---------------------------------------------------------------------------


class Detect(nn.Module):
    """Decoupled YOLO detection head for multi-scale feature maps.

    Handles both standard (YOLO11) and end-to-end (YOLO26) inference:

    - **YOLO11** (``reg_max=16, end2end=False``):
      Output ``(B, 4+nc, total_anchors)`` requiring external NMS.
    - **YOLO26** (``reg_max=1, end2end=True``):
      Output ``(B, max_det, 6)`` — NMS-free top-k selection.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.end2end = end2end
        self.max_det = max_det

        # Stride is set during the first forward pass
        self.stride = torch.zeros(self.nl)

        # Box regression branch (per scale)
        c2 = max(16, ch[0] // 4, reg_max * 4) if ch else 16
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1),
            )
            for c in ch
        )

        # Classification branch (per scale)
        c3 = max(ch[0], min(nc, 100)) if ch else nc
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for c in ch
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        # End-to-end: duplicate heads for one-to-one matching
        if end2end:
            self.one2one_cv2 = deepcopy(self.cv2)
            self.one2one_cv3 = deepcopy(self.cv3)

    def forward(self, x: list[Tensor]) -> Tensor:
        """Run detection head on multi-scale feature maps.

        Args:
            x: List of 3 tensors [P3, P4, P5].

        Returns:
            During inference:
              - YOLO11: ``(B, 4+nc, total_anchors)``
              - YOLO26: ``(B, max_det, 6)`` — ``[x, y, w, h, conf, class_id]``
        """
        if self.end2end:
            return self._forward_end2end(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: list[Tensor]) -> Tensor:
        """Standard detection: returns raw concatenated predictions."""
        box_feats, cls_feats = self._extract_features(x, self.cv2, self.cv3)
        return self._decode(box_feats, cls_feats, x)

    def _forward_end2end(self, x: list[Tensor]) -> Tensor:
        """End-to-end (NMS-free) detection: uses one-to-one heads + top-k."""
        box_feats, cls_feats = self._extract_features(x, self.one2one_cv2, self.one2one_cv3)
        decoded = self._decode(box_feats, cls_feats, x)
        # decoded: (B, 4+nc, total_anchors)
        return self._postprocess_end2end(decoded)

    def _extract_features(
        self,
        x: list[Tensor],
        cv2: nn.ModuleList,
        cv3: nn.ModuleList,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run box and class heads on each scale."""
        box_feats: list[Tensor] = []
        cls_feats: list[Tensor] = []
        for i in range(self.nl):
            box_feats.append(cv2[i](x[i]))
            cls_feats.append(cv3[i](x[i]))
        return box_feats, cls_feats

    def _decode(
        self,
        box_feats: list[Tensor],
        cls_feats: list[Tensor],
        x: list[Tensor],
    ) -> Tensor:
        """Decode box distances and class scores into predictions."""
        # Flatten spatial dims: (B, C, H, W) → (B, C, H*W)
        box_cat = torch.cat([b.flatten(2) for b in box_feats], dim=2)
        cls_cat = torch.cat([c.flatten(2) for c in cls_feats], dim=2)

        # Generate anchors on first pass or shape change
        if self.stride.device != x[0].device or self.stride.sum() == 0:
            self._init_strides(x)

        # Cache anchors — regenerate only when feature map shapes change
        shape_key = tuple(xi.shape[2:] for xi in x)
        if not hasattr(self, "_anchor_cache_key") or self._anchor_cache_key != shape_key:
            self._cached_anchors, self._cached_strides = make_anchors(x, self.stride)
            self._anchor_cache_key = shape_key

        # DFL decode: (B, 4*reg_max, N) → (B, 4, N)
        dfl_out = self.dfl(box_cat)

        # Decode to bounding boxes in pixel coords
        dbox = dist2bbox(
            dfl_out, self._cached_anchors, xywh=not self.end2end
        ) * self._cached_strides.transpose(0, 1)

        # Class scores via sigmoid
        scores = cls_cat.sigmoid()

        return torch.cat([dbox, scores], dim=1)  # (B, 4+nc, N)

    def _postprocess_end2end(self, preds: Tensor) -> Tensor:
        """Top-k selection for NMS-free inference.

        Args:
            preds: ``(B, 4+nc, N)``

        Returns:
            ``(B, max_det, 6)`` — ``[x1, y1, x2, y2, confidence, class_id]``
        """
        # Transpose to (B, N, 4+nc)
        preds = preds.transpose(1, 2)
        boxes, scores = preds.split([4, self.nc], dim=-1)

        # Get max class score and class index per anchor
        max_scores, class_ids = scores.max(dim=-1)  # (B, N)

        # Top-k selection
        topk_scores, topk_idx = max_scores.topk(self.max_det, dim=-1)  # (B, max_det)

        # Gather corresponding boxes and class ids
        topk_idx_expanded = topk_idx.unsqueeze(-1)
        topk_boxes = boxes.gather(1, topk_idx_expanded.expand(-1, -1, 4))
        topk_classes = class_ids.gather(1, topk_idx).unsqueeze(-1).float()
        topk_scores = topk_scores.unsqueeze(-1)

        return torch.cat([topk_boxes, topk_scores, topk_classes], dim=-1)

    def _init_strides(self, x: list[Tensor]) -> None:
        """Compute stride for each detection level from input shapes."""
        # Infer strides from feature map sizes relative to largest (P3).
        # P3 has stride 8 by convention; P4=16, P5=32 for 640 input.
        feat_sizes = torch.tensor([x[i].shape[2] for i in range(self.nl)], dtype=torch.float)
        max_feat = feat_sizes.max()
        self.stride = (max_feat / feat_sizes * 8.0).to(x[0].device)
