"""Detection postprocessing: decode raw model output -> Detection objects.

Handles two output formats:
- Standard YOLO (YOLO11): (B, 4+num_classes, num_anchors)
- NMS-free YOLO26: (B, num_detections, 6) - [x1,y1,x2,y2,confidence,class_id]

NMS uses ``cv2.dnn.NMSBoxes`` (C++ implementation) with the offset trick
for class-aware suppression — no torch dependency.
"""

from __future__ import annotations

import cv2
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
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class PostprocessBuffer:
    """Reusable scratch arrays for postprocessing.

    Eliminates np.empty() allocation in _inverse_letterbox hot path.
    Single-writer pattern: must not be shared across concurrent detect() calls.
    """

    __slots__ = ("_inverse_out", "_max_detections")

    def __init__(self, max_detections: int = 8400) -> None:
        self._max_detections = max_detections
        self._inverse_out = np.empty((max_detections, 4), dtype=np.float32)

    @property
    def memory_bytes(self) -> int:
        return self._max_detections * 4 * 4  # max_detections * 4 coords * 4 bytes

    def get_inverse_out(self, n: int) -> NDArray[np.float32]:
        """Get writable view of first n rows of pre-allocated inverse output.

        Falls back to a fresh allocation when n exceeds the pre-allocated capacity
        rather than silently truncating, which would corrupt coordinate outputs.
        """
        if n <= self._max_detections:
            return self._inverse_out[:n]
        return np.empty((n, 4), dtype=np.float32)


def _class_aware_nms(
    boxes_xyxy: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_ids: NDArray[np.intp],
    iou_threshold: float,
) -> NDArray[np.intp]:
    """Apply class-aware NMS via ``cv2.dnn.NMSBoxes`` with the offset trick.

    Shifts box coordinates by ``class_id * (max_coord + 1)`` so boxes from
    different classes never overlap spatially, then runs a single C++
    ``NMSBoxes`` call instead of per-class Python loops.

    Args:
        boxes_xyxy: (N, 4) xyxy float32.
        scores: (N,) float32.
        class_ids: (N,) integer class indices.
        iou_threshold: IoU suppression threshold.

    Returns:
        Sorted (ascending) indices of kept boxes.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.intp)

    # Offset trick: shift coordinates by class so different classes
    # never overlap. This lets us run a single NMS pass.
    max_coord = float(boxes_xyxy.max())
    offsets = class_ids.astype(np.float32) * np.float32(max_coord + 1.0)

    # Convert xyxy -> xywh (required by cv2.dnn.NMSBoxes) with offsets.
    shifted_xywh = np.empty_like(boxes_xyxy)
    shifted_xywh[:, 0] = boxes_xyxy[:, 0] + offsets  # x + offset
    shifted_xywh[:, 1] = boxes_xyxy[:, 1] + offsets  # y + offset
    shifted_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # w = x2 - x1
    shifted_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # h = y2 - y1

    indices = cv2.dnn.NMSBoxes(
        shifted_xywh,  # type: ignore[arg-type]  # cv2 stubs expect Sequence, ndarray works at runtime
        scores,  # type: ignore[arg-type]
        score_threshold=0.0,  # already pre-filtered by confidence
        nms_threshold=iou_threshold,
    )

    # cv2.dnn.NMSBoxes returns () when empty, Sequence[int] otherwise.
    if isinstance(indices, tuple) or len(indices) == 0:
        return np.array([], dtype=np.intp)

    return np.sort(np.asarray(indices, dtype=np.intp).ravel())


def _inverse_letterbox(
    boxes_xyxy: NDArray[np.float32],
    scale: float,
    pad_top: int,
    pad_left: int,
    orig_h: int,
    orig_w: int,
    *,
    scratch: PostprocessBuffer | None = None,
) -> NDArray[np.float32]:
    """Map boxes from tensor space back to original frame pixel coordinates.

    Uses in-place numpy operations (``out=`` parameter) to avoid creating
    temporary arrays. Each column reuses the output buffer directly.

    Args:
        boxes_xyxy: (N, 4) in tensor pixel space.
        scale: Uniform scale applied during letterbox resize.
        pad_top: Top padding pixels added during letterbox.
        pad_left: Left padding pixels added during letterbox.
        orig_h: Original frame height.
        orig_w: Original frame width.
        scratch: Optional pre-allocated buffer to eliminate np.empty() call.

    Returns:
        (N, 4) clipped to [0, orig_w] / [0, orig_h].
    """
    # Vectorised across all 4 columns: 3 ufunc dispatches instead of 12.
    inv_scale = np.float32(1.0 / scale)
    pad = np.array([pad_left, pad_top, pad_left, pad_top], dtype=np.float32)
    clip_max = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)

    n = boxes_xyxy.shape[0]
    # Use scratch buffer to avoid allocation; fall back to np.empty when absent.
    out: NDArray[np.float32] = (
        scratch.get_inverse_out(n) if scratch is not None else np.empty((n, 4), dtype=np.float32)
    )

    np.subtract(boxes_xyxy, pad, out=out)
    np.multiply(out, inv_scale, out=out)
    np.clip(out, 0.0, clip_max, out=out)

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
    scratch: PostprocessBuffer | None = None,
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
    max_scores = max_class_score[mask]

    # xywh -> xyxy in tensor space: pre-allocate once, write columns in-place.
    # Avoids 4 intermediate scalar arrays + np.stack allocation.
    boxes_xyxy = np.empty((len(filtered), 4), dtype=np.float32)
    half_w = filtered[:, 2] * 0.5
    half_h = filtered[:, 3] * 0.5
    boxes_xyxy[:, 0] = filtered[:, 0] - half_w  # x1
    boxes_xyxy[:, 1] = filtered[:, 1] - half_h  # y1
    boxes_xyxy[:, 2] = filtered[:, 0] + half_w  # x2
    boxes_xyxy[:, 3] = filtered[:, 1] + half_h  # y2

    # Use filtered[:, 4:] (view) instead of class_scores_all[mask] (copy)
    # to avoid a redundant (N, num_classes) allocation.
    class_ids = np.argmax(filtered[:, 4:], axis=1).astype(np.intp)

    # Inverse letterbox transform.
    boxes_orig = _inverse_letterbox(
        boxes_xyxy, scale, pad_top, pad_left, orig_h, orig_w, scratch=scratch
    )

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
    scratch: PostprocessBuffer | None = None,
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
    # astype(copy=False): avoids an extra allocation when data is already
    # float32; the preceding .copy() was redundant since astype copies anyway.
    boxes_xyxy = filtered[:, :4].astype(np.float32, copy=False)
    conf = filtered[:, 4]
    class_ids = filtered[:, 5].astype(np.intp)

    boxes_orig = _inverse_letterbox(
        boxes_xyxy, scale, pad_top, pad_left, orig_h, orig_w, scratch=scratch
    )

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
    scratch: PostprocessBuffer | None = None,
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
                    item,
                    confidence_threshold,
                    scale,
                    pad_top,
                    pad_left,
                    orig_h,
                    orig_w,
                    names,
                    scratch,
                )
            elif item.ndim == 2 and item.shape[0] == 6:
                boxes = _decode_yolo26(
                    item.T,
                    confidence_threshold,
                    scale,
                    pad_top,
                    pad_left,
                    orig_h,
                    orig_w,
                    names,
                    scratch,
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
                # Valid for standard YOLO anchor grids where num_anchors (>=1000) >> 4+num_classes.
                if cols > rows:
                    # shape is (4+num_classes, num_anchors) -> transpose
                    item = item.T
                boxes = _decode_standard(
                    item,
                    confidence_threshold,
                    iou_threshold,
                    scale,
                    pad_top,
                    pad_left,
                    orig_h,
                    orig_w,
                    names,
                    scratch,
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
    "PostprocessBuffer",
    "postprocess",
]
