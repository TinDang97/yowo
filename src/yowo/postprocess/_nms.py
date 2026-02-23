"""Detection postprocessing: decode raw model output -> Detection objects.

Handles two output formats:
- Standard YOLO (YOLO11, YOLO12): (B, 4+num_classes, num_anchors)
- NMS-free YOLO26: (B, num_detections, 6) - [x1,y1,x2,y2,confidence,class_id]

All NMS is implemented in pure numpy — no torch dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from yowo.types import (
    BackendType,
    BoundingBox,
    Detection,
    Frame,
    ModelFamily,
    ModelSpec,
    PreprocessedTensor,
)

# Standard COCO 80-class names (index 0..79).
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _iou(box: NDArray[np.float32], boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Vectorized IoU of one box against N boxes.

    Args:
        box: Shape (4,) in xyxy format.
        boxes: Shape (N, 4) in xyxy format.

    Returns:
        Shape (N,) float32 IoU values.
    """
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area + 1e-6

    return (inter_area / union_area).astype(np.float32)


def _greedy_nms(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
) -> NDArray[np.intp]:
    """Greedy NMS for a single class.

    Args:
        boxes: (N, 4) xyxy float32.
        scores: (N,) float32 confidence scores.
        iou_threshold: Suppress boxes with IoU >= this value.

    Returns:
        Indices of surviving boxes (sorted by descending score).
    """
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        iou_vals = _iou(boxes[idx], boxes[rest])
        suppressed = iou_vals >= iou_threshold
        order = rest[~suppressed]

    return np.array(keep, dtype=np.intp)


def _class_aware_nms(
    boxes_xyxy: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_ids: NDArray[np.intp],
    iou_threshold: float,
) -> NDArray[np.intp]:
    """Apply NMS independently for each class and return surviving indices.

    Args:
        boxes_xyxy: (N, 4) xyxy float32.
        scores: (N,) float32.
        class_ids: (N,) integer class indices.
        iou_threshold: IoU suppression threshold.

    Returns:
        Sorted (ascending) indices of kept boxes.
    """
    kept: list[int] = []
    for cls in np.unique(class_ids):
        mask = class_ids == cls
        indices = np.where(mask)[0]
        surviving = _greedy_nms(
            boxes_xyxy[mask],
            scores[mask],
            iou_threshold,
        )
        kept.extend(indices[surviving].tolist())
    return np.array(sorted(kept), dtype=np.intp)


def _inverse_letterbox(
    boxes_xyxy: NDArray[np.float32],
    scale: float,
    pad_top: int,
    pad_left: int,
    orig_h: int,
    orig_w: int,
) -> NDArray[np.float32]:
    """Map boxes from tensor space back to original frame pixel coordinates.

    Args:
        boxes_xyxy: (N, 4) in tensor pixel space.
        scale: Uniform scale applied during letterbox resize.
        pad_top: Top padding pixels added during letterbox.
        pad_left: Left padding pixels added during letterbox.
        orig_h: Original frame height.
        orig_w: Original frame width.

    Returns:
        (N, 4) clipped to [0, orig_w] / [0, orig_h].
    """
    out = boxes_xyxy.copy()
    out[:, 0] = (boxes_xyxy[:, 0] - pad_left) / scale  # x1
    out[:, 1] = (boxes_xyxy[:, 1] - pad_top) / scale   # y1
    out[:, 2] = (boxes_xyxy[:, 2] - pad_left) / scale  # x2
    out[:, 3] = (boxes_xyxy[:, 3] - pad_top) / scale   # y2

    out[:, 0] = np.clip(out[:, 0], 0.0, float(orig_w))
    out[:, 2] = np.clip(out[:, 2], 0.0, float(orig_w))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(orig_h))
    out[:, 3] = np.clip(out[:, 3], 0.0, float(orig_h))

    return out


def _decode_standard(
    raw: NDArray[np.float32],
    confidence_threshold: float,
    iou_threshold: float,
    scale: float,
    pad_top: int,
    pad_left: int,
    orig_h: int,
    orig_w: int,
    names: list[str],
) -> tuple[BoundingBox, ...]:
    """Decode one batch item from standard YOLO output.

    Args:
        raw: (num_anchors, 4+num_classes) float32, already transposed.
        confidence_threshold: Min score to keep.
        iou_threshold: NMS IoU threshold.
        scale: Letterbox uniform scale.
        pad_top: Top padding offset.
        pad_left: Left padding offset.
        orig_h: Original frame height.
        orig_w: Original frame width.
        names: Class name list.

    Returns:
        Tuple of BoundingBox objects.
    """
    # Confidence filter: max class score across all class columns.
    class_scores_all = raw[:, 4:]
    max_class_score = class_scores_all.max(axis=1)
    mask = max_class_score >= confidence_threshold

    if not mask.any():
        return ()

    filtered = raw[mask]
    class_scores = class_scores_all[mask]
    max_scores = max_class_score[mask]

    # xywh -> xyxy in tensor space.
    cx = filtered[:, 0]
    cy = filtered[:, 1]
    w = filtered[:, 2]
    h = filtered[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    class_ids = np.argmax(class_scores, axis=1).astype(np.intp)

    # Inverse letterbox transform.
    boxes_orig = _inverse_letterbox(boxes_xyxy, scale, pad_top, pad_left, orig_h, orig_w)

    # Class-aware NMS.
    kept = _class_aware_nms(boxes_orig, max_scores, class_ids, iou_threshold)

    result: list[BoundingBox] = []
    for i in kept:
        cid = int(class_ids[i])
        result.append(
            BoundingBox(
                x1=float(boxes_orig[i, 0]),
                y1=float(boxes_orig[i, 1]),
                x2=float(boxes_orig[i, 2]),
                y2=float(boxes_orig[i, 3]),
                confidence=float(max_scores[i]),
                class_id=cid,
                class_name=names[cid] if cid < len(names) else str(cid),
            )
        )
    return tuple(result)


def _decode_yolo26(
    raw: NDArray[np.float32],
    confidence_threshold: float,
    scale: float,
    pad_top: int,
    pad_left: int,
    orig_h: int,
    orig_w: int,
    names: list[str],
) -> tuple[BoundingBox, ...]:
    """Decode one batch item from YOLO26 NMS-free output.

    Expected format per detection row: [x1, y1, x2, y2, confidence, class_id].

    Args:
        raw: (num_detections, 6) float32.
        confidence_threshold: Min confidence to keep.
        scale: Letterbox uniform scale.
        pad_top: Top padding offset.
        pad_left: Left padding offset.
        orig_h: Original frame height.
        orig_w: Original frame width.
        names: Class name list.

    Returns:
        Tuple of BoundingBox objects.
    """
    confidences = raw[:, 4]
    mask = confidences >= confidence_threshold

    if not mask.any():
        return ()

    filtered = raw[mask]
    boxes_xyxy = filtered[:, :4].copy().astype(np.float32)
    conf = filtered[:, 4]
    class_ids = filtered[:, 5].astype(np.intp)

    boxes_orig = _inverse_letterbox(boxes_xyxy, scale, pad_top, pad_left, orig_h, orig_w)

    result: list[BoundingBox] = []
    for i in range(len(filtered)):
        cid = int(class_ids[i])
        result.append(
            BoundingBox(
                x1=float(boxes_orig[i, 0]),
                y1=float(boxes_orig[i, 1]),
                x2=float(boxes_orig[i, 2]),
                y2=float(boxes_orig[i, 3]),
                confidence=float(conf[i]),
                class_id=cid,
                class_name=names[cid] if cid < len(names) else str(cid),
            )
        )
    return tuple(result)


def postprocess(
    raw_output: NDArray[np.float32],
    tensor_meta: PreprocessedTensor,
    frames: list[Frame],
    *,
    model_spec: ModelSpec,
    backend: BackendType,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: list[str] | None = None,
    inference_time_ms: float = 0.0,
) -> list[Detection]:
    """Decode raw YOLO output into Detection objects.

    Handles two output shape conventions:
    - Standard YOLO (YOLO11/12): ``(B, 4+num_classes, num_anchors)`` or
      transposed ``(B, num_anchors, 4+num_classes)``.
    - YOLO26 (NMS-free): ``(B, num_detections, 6)`` —
      ``[x1, y1, x2, y2, confidence, class_id]``.

    Algorithm per batch item (standard path):
    1. Detect output orientation; transpose to ``(num_anchors, 4+num_classes)``
       if the second dimension is smaller than the third.
    2. Compute max class score; filter rows below ``confidence_threshold``.
    3. Decode ``cx,cy,w,h`` -> ``x1,y1,x2,y2``.
    4. Apply inverse letterbox to recover original-frame coordinates.
    5. Class-aware greedy NMS.
    6. Build ``BoundingBox`` objects with COCO names unless ``class_names``
       is provided.

    Args:
        raw_output: Raw float32 tensor from the inference backend.
        tensor_meta: Preprocessing metadata (scale/pad per image).
        frames: Original Frame objects, one per batch item.
        model_spec: Used to dispatch YOLO26 NMS-free path.
        backend: Backend that produced the output (attached to Detection).
        confidence_threshold: Minimum class confidence to keep a box.
        iou_threshold: NMS IoU suppression threshold.
        class_names: Override COCO class names. Must have length == num_classes.
        inference_time_ms: Wall-clock inference time (attached to Detection).

    Returns:
        List of ``Detection`` objects, one per frame in *frames*.
    """
    names = class_names if class_names is not None else COCO_CLASSES
    is_yolo26 = model_spec.family == ModelFamily.YOLO26

    batch_size = len(frames)
    detections: list[Detection] = []

    for b in range(batch_size):
        scale_h, _scale_w = tensor_meta.scale_factors[b]
        # Uniform scale: both components equal; use scale_h.
        scale = scale_h
        pad_top, pad_left = tensor_meta.pad_offsets[b]
        orig_h, orig_w = tensor_meta.original_shapes[b]
        frame = frames[b]

        item: NDArray[np.float32] = raw_output[b]

        if is_yolo26:
            # YOLO26 output: (num_detections, 6).
            # May arrive as (6, num_detections) — transpose if col count is 6.
            if item.ndim == 2 and item.shape[1] == 6:
                boxes = _decode_yolo26(
                    item, confidence_threshold, scale, pad_top, pad_left,
                    orig_h, orig_w, names,
                )
            elif item.ndim == 2 and item.shape[0] == 6:
                boxes = _decode_yolo26(
                    item.T, confidence_threshold, scale, pad_top, pad_left,
                    orig_h, orig_w, names,
                )
            else:
                boxes = ()
        else:
            # Standard YOLO: (4+num_classes, num_anchors) or transposed.
            if item.ndim != 2:
                boxes = ()
            else:
                rows, cols = item.shape
                # Orient to (num_anchors, 4+num_classes): more rows than cols
                # means it is already transposed; otherwise transpose.
                if cols > rows:
                    # shape is (4+num_classes, num_anchors) -> transpose
                    item = item.T
                boxes = _decode_standard(
                    item, confidence_threshold, iou_threshold,
                    scale, pad_top, pad_left, orig_h, orig_w, names,
                )

        detections.append(
            Detection(
                frame=frame,
                boxes=boxes,
                inference_time_ms=inference_time_ms,
                backend=backend,
                model_spec=model_spec,
            )
        )

    return detections


__all__ = [
    "COCO_CLASSES",
    "_iou",
    "postprocess",
]
