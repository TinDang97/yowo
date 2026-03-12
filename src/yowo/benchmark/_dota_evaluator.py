"""DOTA v1 dataset loader and OBB mAP evaluator for YOWO benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yowo.postprocess._obb_nms import DOTA_CLASSES

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # pyright: ignore[reportDeprecated]  # numpy <2.0 compat

if TYPE_CHECKING:
    from yowo.types import OBBDetection

_DOTA_CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(DOTA_CLASSES)}


def load_dota_dataset(
    data_path: str | Path,
    subset: int | None = None,
) -> tuple[list[Path], list[list[list[float]]], list[list[int]]]:
    """Load DOTA v1 val split ground truth.

    Expected layout::

        data_path/
          images/val/    -- image files (.jpg, .png, .tif, ...)
          labelTxt/val/  -- per-image .txt files with format:
                           x1 y1 x2 y2 x3 y3 x4 y4 class difficulty

    Args:
        data_path: Root directory of DOTA v1 dataset.
        subset: If provided, limit to first *subset* images.

    Returns:
        Tuple of (image_paths, gt_boxes_per_image, gt_classes_per_image).
        gt_boxes_per_image: list of lists; each inner list is 8 float coords
            [x1, y1, x2, y2, x3, y3, x4, y4] (quad corner format).
        gt_classes_per_image: list of lists; each inner list is class indices
            (int, 0-indexed matching DOTA_CLASSES).

    Raises:
        FileNotFoundError: If images/val or labelTxt/val directories are missing.
    """
    root = Path(data_path)
    images_dir = root / "images" / "val"
    labels_dir = root / "labelTxt" / "val"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"DOTA images/val not found: {images_dir}. "
            "Expected layout: data_path/images/val/ and data_path/labelTxt/val/"
        )
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"DOTA labelTxt/val not found: {labels_dir}. "
            "Expected layout: data_path/images/val/ and data_path/labelTxt/val/"
        )

    image_paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    )
    if subset is not None:
        image_paths = image_paths[:subset]

    gt_boxes_per_image: list[list[list[float]]] = []
    gt_classes_per_image: list[list[int]] = []

    for img_path in image_paths:
        label_file = labels_dir / (img_path.stem + ".txt")
        boxes: list[list[float]] = []
        classes: list[int] = []
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) < 9:
                    continue
                coords = [float(t) for t in tokens[:8]]
                class_name = tokens[8]
                class_idx = _DOTA_CLASS_INDEX.get(class_name, -1)
                if class_idx < 0:
                    continue
                boxes.append(coords)
                classes.append(class_idx)
        gt_boxes_per_image.append(boxes)
        gt_classes_per_image.append(classes)

    return image_paths, gt_boxes_per_image, gt_classes_per_image


def _quad_to_xywhr(quad: list[float]) -> list[float]:
    """Convert 4-corner quad [x1,y1,x2,y2,x3,y3,x4,y4] to [cx,cy,w,h,angle].

    Uses minimum bounding box: cx/cy = corner mean, w/h from edge lengths,
    angle from longest edge. Matches ultralytics DOTA preprocessing convention.
    """
    dx1 = quad[2] - quad[0]
    dy1 = quad[3] - quad[1]
    dx2 = quad[4] - quad[2]
    dy2 = quad[5] - quad[3]
    len1 = (dx1**2 + dy1**2) ** 0.5
    len2 = (dx2**2 + dy2**2) ** 0.5
    cx = (quad[0] + quad[2] + quad[4] + quad[6]) / 4
    cy = (quad[1] + quad[3] + quad[5] + quad[7]) / 4
    if len1 >= len2:
        angle = float(np.arctan2(dy1, dx1))
        w, h = len1, len2
    else:
        angle = float(np.arctan2(dy2, dx2))
        w, h = len2, len1
    return [cx, cy, w, h, angle]


def evaluate_obb_map(
    detections_per_image: list[OBBDetection],
    gt_boxes_per_image: list[list[list[float]]],
    gt_classes_per_image: list[list[int]],
    iou_thresholds: list[float] | None = None,
    num_classes: int = 15,
) -> dict[str, float]:
    """Compute COCO-style mAP50-95 for OBB using probiou rotated IoU.

    Sweeps IoU thresholds 0.50, 0.55, ..., 0.95 (10 values), computes
    per-class AP at each threshold via precision-recall curve, then averages
    over classes and thresholds.

    Args:
        detections_per_image: OBBDetection results, one per image.
        gt_boxes_per_image: Ground truth quads per image (DOTA quad format).
        gt_classes_per_image: Ground truth class indices per image.
        iou_thresholds: Thresholds to sweep. Defaults to COCO range 0.50-0.95.
        num_classes: Number of OBB classes (15 for DOTA v1).

    Returns:
        Dict with keys ``"mAP_50_95"`` (float) and ``"mAP_50"`` (float).
    """
    import torch

    from yowo.postprocess._obb_nms import probiou_matrix

    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95

    # Convert GT quads to xywhr once
    gt_xywhr_per_image: list[list[list[float]]] = [
        [_quad_to_xywhr(b) for b in gt_boxes] for gt_boxes in gt_boxes_per_image
    ]

    ap_by_threshold: list[float] = []

    for iou_thresh in iou_thresholds:
        ap_by_class: list[float] = []
        for cls_idx in range(num_classes):
            all_scores: list[float] = []
            all_tp: list[int] = []
            total_gt = 0

            for img_idx, det in enumerate(detections_per_image):
                gt_xywhr_cls = [
                    r
                    for r, c in zip(
                        gt_xywhr_per_image[img_idx],
                        gt_classes_per_image[img_idx],
                        strict=True,
                    )
                    if c == cls_idx
                ]
                total_gt += len(gt_xywhr_cls)

                pred_boxes = sorted(
                    (b for b in det.boxes if b.class_id == cls_idx),
                    key=lambda b: b.confidence,
                    reverse=True,
                )

                matched_gt: set[int] = set()
                for pred in pred_boxes:
                    all_scores.append(pred.confidence)
                    if not gt_xywhr_cls:
                        all_tp.append(0)
                        continue
                    pred_t = torch.tensor([[pred.cx, pred.cy, pred.w, pred.h, pred.angle]])
                    gt_t = torch.tensor(gt_xywhr_cls)
                    iou_mat = probiou_matrix(pred_t, gt_t)  # (1, M)
                    best_gt = int(iou_mat[0].argmax())
                    if float(iou_mat[0, best_gt]) >= iou_thresh and best_gt not in matched_gt:
                        all_tp.append(1)
                        matched_gt.add(best_gt)
                    else:
                        all_tp.append(0)

            if total_gt == 0:
                ap_by_class.append(0.0)
                continue

            order = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
            tp_sorted = [all_tp[i] for i in order]
            tp_cumsum = np.cumsum(tp_sorted)
            fp_cumsum = np.cumsum([1 - x for x in tp_sorted])
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
            recall = tp_cumsum / (total_gt + 1e-9)
            ap = float(_trapezoid(precision, recall))
            ap_by_class.append(ap)

        ap_by_threshold.append(float(np.mean(ap_by_class)))

    map_50_95 = float(np.mean(ap_by_threshold))
    map_50 = ap_by_threshold[0] if ap_by_threshold else 0.0

    return {"mAP_50_95": map_50_95, "mAP_50": map_50}


__all__ = [
    "DOTA_CLASSES",
    "evaluate_obb_map",
    "load_dota_dataset",
]
