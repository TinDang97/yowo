# postprocess — Output Decoding and NMS

Decodes the raw float32 tensor produced by an inference backend into `Detection` objects containing bounding boxes with class names, class IDs, and confidence scores.

---

## Module Structure

```
postprocess/
├── __init__.py    — public surface: postprocess()
└── _nms.py        — tensor reshape, confidence filter, box decode, NMS, Detection assembly
```

---

## Public Interface

```python
def postprocess(
    raw_output:           NDArray[np.float32],
    tensor_meta:          TensorMeta,
    frames:               list[Frame],
    model_spec:           ModelSpec,
    confidence_threshold: float = 0.25,
    iou_threshold:        float = 0.45,
    class_names:          list[str] | None = None,
    inference_time_ms:    float | None = None,
) -> list[Detection]:
    """
    Convert raw model output to Detection objects.

    raw_output:           NDArray shape varies by model family (see below).
    tensor_meta:          TensorMeta from io.preprocess() — carries scale/pad for coord recovery.
    frames:               original Frame list (one per batch item).
    model_spec:           used to dispatch YOLO26 NMS-free path vs standard path.
    confidence_threshold: discard boxes with max class confidence below this value.
    iou_threshold:        NMS suppression threshold.
    class_names:          list of length num_classes. Defaults to COCO 80-class names.
    inference_time_ms:    attached to Detection.inference_time_ms if provided.

    Returns list[Detection] of length == len(frames).
    Boxes are in original frame pixel coordinates (xyxy, top-left origin).
    """
```

`TensorMeta` is imported from `io._decode` — the type is defined there because it is created during preprocessing and consumed here during postprocessing. This avoids defining it in `types.py` (which would create an `io` → `postprocess` dependency).

---

## Algorithm

### Standard Path (YOLO11)

```
raw_output shape: (B, 4 + num_classes, num_anchors)
                   └── per-anchor: [cx, cy, w, h, class_scores...]

Step 1 — Reshape
    Transpose to (B, num_anchors, 4 + num_classes) for row-wise processing.

Step 2 — Confidence filter
    max_class_score = raw[:, :, 4:].max(axis=-1)        shape: (B, num_anchors)
    mask = max_class_score >= confidence_threshold
    Discard rows where mask is False (vectorized boolean index).

Step 3 — Box decode: xywh → xyxy in tensor space
    cx, cy, w, h = raw[mask, 0], raw[mask, 1], raw[mask, 2], raw[mask, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

Step 4 — Inverse letterbox transform (tensor space → original frame space)
    Using TensorMeta.scale_factors = (scale, scale) and TensorMeta.pad_offsets = (pad_top, pad_left):
    x1_orig = (x1 - pad_left)  / scale
    y1_orig = (y1 - pad_top)   / scale
    x2_orig = (x2 - pad_left)  / scale
    y2_orig = (y2 - pad_top)   / scale
    Clip to [0, frame_width] and [0, frame_height] respectively.

Step 5 — Class-aware NMS (vectorized)
    For each batch item independently:
      For each unique class_id present:
        Apply torchvision-style NMS to boxes of that class using iou_threshold.
        (Implemented in numpy: sort by score, compute IoU matrix, greedy suppress.)
    Concatenate surviving boxes.

Step 6 — Assemble Detection objects
    class_id  = argmax of class scores for surviving rows
    class_name = class_names[class_id]
    confidence = class_scores[class_id]
    BoundingBox(x1=x1_orig, y1=y1_orig, x2=x2_orig, y2=y2_orig,
                class_id=class_id, class_name=class_name, confidence=confidence)
    Detection(frame=frames[b], boxes=[...], inference_time_ms=inference_time_ms)
```

### YOLO26 Path (NMS-free)

YOLO26 uses a differentiable NMS head — boxes in raw output are already post-NMS. The standard NMS step is skipped.

```
raw_output shape: (B, num_detections, 6)
                   └── per detection: [x1, y1, x2, y2, confidence, class_id]
                   (already in tensor coordinate space, already filtered)

Step 1 — Confidence filter (still applied)
Step 2 — Inverse letterbox (same as standard path)
Step 3 — Skip NMS, proceed directly to Detection assembly
```

---

## Box Format Conventions

| Stage | Format | Coordinate space |
|-------|--------|-----------------|
| Raw model output | `xywh` (center + dimensions) | tensor pixel space (0 to target_size) |
| After decode | `xyxy` (corners) | tensor pixel space |
| After inverse letterbox | `xyxy` (corners) | original frame pixel space |
| `BoundingBox` fields | `x1, y1, x2, y2` | original frame pixel space, top-left origin |

All values are floats at intermediate stages. `BoundingBox` stores them as `float` — callers that need integer pixel indices should apply `int(round(...))`.

---

## NMS Implementation

NMS is implemented in pure numpy (no torchvision dependency) using the standard greedy algorithm:

```
1. Sort boxes by confidence descending.
2. Mark the highest-confidence box as kept.
3. Compute IoU between it and all remaining boxes.
4. Suppress boxes with IoU >= iou_threshold.
5. Repeat from step 2 with remaining unsuppressed boxes.
```

IoU computation is vectorized:

```python
intersection_area = max(0, min(x2, x2s) - max(x1, x1s)) * max(0, min(y2, y2s) - max(y1, y1s))
union_area = area + areas - intersection_area
iou = intersection_area / (union_area + 1e-6)
```

---

## Dependencies

- **numpy**: all tensor operations — no torch dependency
- **yowo imports**: `types.py` (`Detection`, `BoundingBox`, `Frame`, `ModelSpec`), `io._decode.TensorMeta`

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `types.py` | `Detection`, `BoundingBox`, `Frame`, `ModelSpec` |
| Upstream | `io/_decode.py` | `TensorMeta` (scale_factors, pad_offsets) |
| Downstream | `engine.py` | calls `postprocess()` after each `backend.infer()` call |
