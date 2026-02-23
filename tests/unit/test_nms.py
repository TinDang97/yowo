"""Unit tests for yowo.postprocess._nms."""

from __future__ import annotations

import numpy as np
import pytest

from yowo.postprocess._nms import COCO_CLASSES, _class_aware_nms, postprocess
from yowo.types import (
    BackendType,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
    PreprocessedTensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(height: int = 480, width: int = 640) -> Frame:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0, timestamp_ms=0.0)


def _make_tensor_meta(
    batch: int,
    orig_h: int = 480,
    orig_w: int = 640,
    scale: float = 1.0,
    pad_top: int = 0,
    pad_left: int = 0,
) -> PreprocessedTensor:
    """Build a minimal PreprocessedTensor for postprocess()."""
    data = np.zeros((batch, 3, 640, 640), dtype=np.float32)
    return PreprocessedTensor(
        data=data,
        original_shapes=tuple((orig_h, orig_w) for _ in range(batch)),
        input_shape=(640, 640),
        scale_factors=tuple((scale, scale) for _ in range(batch)),
        pad_offsets=tuple((pad_top, pad_left) for _ in range(batch)),
    )


def _make_model_spec(family: ModelFamily = ModelFamily.YOLO11) -> ModelSpec:
    return ModelSpec(family=family, size=ModelSize.NANO)


def _make_raw_standard(
    batch: int = 1,
    num_classes: int = 80,
    num_anchors: int = 8400,
) -> np.ndarray:
    """Return a zeroed (B, 4+num_classes, num_anchors) float32 array."""
    return np.zeros((batch, 4 + num_classes, num_anchors), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests for _class_aware_nms()
# ---------------------------------------------------------------------------


class TestClassAwareNms:
    """Tests for _class_aware_nms using cv2.dnn.NMSBoxes + offset trick."""

    def test_empty_input_returns_empty(self) -> None:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        class_ids = np.empty(0, dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(result) == 0
        assert result.dtype == np.intp

    def test_single_box_kept(self) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert result.tolist() == [0]

    def test_overlapping_same_class_suppressed(self) -> None:
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert result.tolist() == [0]

    def test_overlapping_different_classes_both_kept(self) -> None:
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert sorted(result.tolist()) == [0, 1]

    def test_non_overlapping_same_class_both_kept(self) -> None:
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert sorted(result.tolist()) == [0, 1]

    def test_result_sorted_ascending(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 100.0, 110.0, 110.0],
                [200.0, 200.0, 210.0, 210.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.7, 0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1, 2], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert list(result) == sorted(result.tolist())

    def test_result_dtype_is_intp(self) -> None:
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert result.dtype == np.intp

    def test_nmsboxes_empty_tuple_return_handled(self) -> None:
        """When cv2.dnn.NMSBoxes returns () internally, result is empty."""
        from unittest.mock import patch

        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.intp)

        with patch("yowo.postprocess._nms.cv2.dnn.NMSBoxes", return_value=()):
            result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)

        assert len(result) == 0
        assert result.dtype == np.intp

    def test_high_class_id_offset_trick(self) -> None:
        """Large class IDs produce correct offsets without cross-class suppression."""
        boxes = np.array(
            [[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 79], dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert sorted(result.tolist()) == [0, 1]


# ---------------------------------------------------------------------------
# Tests for postprocess()
# ---------------------------------------------------------------------------


class TestPostprocessStandard:
    def _make_raw_with_detection(
        self,
        cx: float = 320.0,
        cy: float = 240.0,
        w: float = 100.0,
        h: float = 80.0,
        class_id: int = 0,
        score: float = 0.9,
    ) -> np.ndarray:
        """(1, 85, 8400) array with one strong detection."""
        raw = _make_raw_standard(batch=1, num_classes=80, num_anchors=8400)
        raw[0, 0, 0] = cx
        raw[0, 1, 0] = cy
        raw[0, 2, 0] = w
        raw[0, 3, 0] = h
        raw[0, 4 + class_id, 0] = score
        return raw

    def test_single_detection_returned(self) -> None:
        raw = self._make_raw_with_detection(score=0.9)
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        frames = [_make_frame()]
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            frames,
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.25,
        )

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, Detection)
        assert det.num_boxes == 1
        assert det.has_detections

    def test_detection_class_id(self) -> None:
        raw = self._make_raw_with_detection(class_id=0, score=0.9)
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        assert results[0].boxes[0].class_id == 0
        assert results[0].boxes[0].class_name == "person"

    def test_detection_confidence(self) -> None:
        raw = self._make_raw_with_detection(score=0.85)
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        assert results[0].boxes[0].confidence == pytest.approx(0.85, rel=1e-4)

    def test_confidence_threshold_filters_weak_detections(self) -> None:
        raw = _make_raw_standard()
        # Place two detections at different confidence levels.
        raw[0, 4, 0] = 0.8  # strong (class 0)
        raw[0, 0, 0], raw[0, 1, 0] = 320.0, 240.0
        raw[0, 2, 0], raw[0, 3, 0] = 100.0, 80.0

        raw[0, 4, 1] = 0.1  # weak (class 0)
        raw[0, 0, 1], raw[0, 1, 1] = 100.0, 100.0
        raw[0, 2, 1], raw[0, 3, 1] = 50.0, 50.0

        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        high_threshold = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.5,
        )
        low_threshold = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.05,
        )

        assert high_threshold[0].num_boxes == 1
        assert low_threshold[0].num_boxes >= 1

    def test_no_detection_when_all_below_threshold(self) -> None:
        raw = _make_raw_standard()  # all zeros -> all scores = 0
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.25,
        )

        assert results[0].num_boxes == 0
        assert not results[0].has_detections

    def test_coordinate_inverse_transform_no_padding(self) -> None:
        # Detection centered at (320, 240) with size 100x80 in tensor space.
        # scale=1.0, no padding -> original coords identical.
        raw = self._make_raw_with_detection(cx=320.0, cy=240.0, w=100.0, h=80.0, score=0.9)
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0, pad_top=0, pad_left=0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        box = results[0].boxes[0]
        assert box.x1 == pytest.approx(270.0, abs=1.0)  # cx - w/2
        assert box.y1 == pytest.approx(200.0, abs=1.0)  # cy - h/2
        assert box.x2 == pytest.approx(370.0, abs=1.0)  # cx + w/2
        assert box.y2 == pytest.approx(280.0, abs=1.0)  # cy + h/2

    def test_coordinate_inverse_transform_with_padding(self) -> None:
        # scale=0.5, pad_top=80, pad_left=160
        # Detection in tensor space: cx=320, cy=240, w=100, h=80
        # x1_tensor=270, y1_tensor=200, x2_tensor=370, y2_tensor=280
        # x1_orig=(270-160)/0.5=220, y1_orig=(200-80)/0.5=240
        # x2_orig=(370-160)/0.5=420, y2_orig=(280-80)/0.5=400
        raw = self._make_raw_with_detection(cx=320.0, cy=240.0, w=100.0, h=80.0, score=0.9)
        tensor_meta = _make_tensor_meta(
            batch=1,
            orig_h=480,
            orig_w=640,
            scale=0.5,
            pad_top=80,
            pad_left=160,
        )
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        box = results[0].boxes[0]
        assert box.x1 == pytest.approx(220.0, abs=1.0)
        assert box.y1 == pytest.approx(240.0, abs=1.0)
        assert box.x2 == pytest.approx(420.0, abs=1.0)
        assert box.y2 == pytest.approx(400.0, abs=1.0)

    def test_batch_size_matches_frames(self) -> None:
        # Two frames, two batch items.
        raw = np.zeros((2, 85, 8400), dtype=np.float32)
        # Detection in batch item 0.
        raw[0, 4, 0] = 0.9
        raw[0, 0, 0], raw[0, 1, 0] = 320.0, 240.0
        raw[0, 2, 0], raw[0, 3, 0] = 100.0, 80.0

        tensor_meta = _make_tensor_meta(batch=2, scale=1.0)
        frames = [_make_frame(), _make_frame()]
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            frames,
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        assert len(results) == 2
        assert results[0].num_boxes == 1
        assert results[1].num_boxes == 0

    def test_custom_class_names(self) -> None:
        raw = self._make_raw_with_detection(class_id=0, score=0.9)
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()
        custom_names = [f"cls_{i}" for i in range(80)]

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            class_names=custom_names,
        )

        assert results[0].boxes[0].class_name == "cls_0"

    def test_backend_attached_to_detection(self) -> None:
        raw = _make_raw_standard()
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.ONNX,
        )

        assert results[0].backend == BackendType.ONNX

    def test_inference_time_ms_attached(self) -> None:
        raw = _make_raw_standard()
        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            inference_time_ms=42.5,
        )

        assert results[0].inference_time_ms == pytest.approx(42.5)

    def test_transposed_input_shape_handled(self) -> None:
        # Input shape (B, num_anchors, 4+num_classes) — already transposed.
        raw = np.zeros((1, 8400, 85), dtype=np.float32)
        raw[0, 0, 0] = 320.0  # cx
        raw[0, 0, 1] = 240.0  # cy
        raw[0, 0, 2] = 100.0  # w
        raw[0, 0, 3] = 80.0  # h
        raw[0, 0, 4] = 0.9  # class 0 score

        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = _make_model_spec()

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        assert results[0].num_boxes == 1


class TestPostprocessYolo26:
    def test_yolo26_nms_free_path(self) -> None:
        # YOLO26 output: (B, num_detections, 6) = [x1,y1,x2,y2,conf,class_id]
        raw = np.zeros((1, 100, 6), dtype=np.float32)
        raw[0, 0] = [100.0, 100.0, 200.0, 200.0, 0.85, 2.0]  # class 2 = car

        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
        frames = [_make_frame()]

        results = postprocess(
            raw,
            tensor_meta,
            frames,
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.25,
        )

        assert results[0].num_boxes == 1
        box = results[0].boxes[0]
        assert box.class_id == 2
        assert box.class_name == "car"
        assert box.confidence == pytest.approx(0.85, rel=1e-4)

    def test_yolo26_confidence_filter(self) -> None:
        raw = np.zeros((1, 10, 6), dtype=np.float32)
        raw[0, 0] = [10.0, 10.0, 50.0, 50.0, 0.9, 0.0]  # above threshold
        raw[0, 1] = [60.0, 60.0, 100.0, 100.0, 0.1, 1.0]  # below threshold

        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.5,
        )

        assert results[0].num_boxes == 1
        assert results[0].boxes[0].class_id == 0

    def test_yolo26_transposed_input_handled(self) -> None:
        """YOLO26 output with shape (B, 6, num_detections) is auto-transposed."""
        raw = np.zeros((1, 6, 100), dtype=np.float32)
        raw[0, :, 0] = [100.0, 100.0, 200.0, 200.0, 0.85, 2.0]

        tensor_meta = _make_tensor_meta(batch=1, scale=1.0)
        spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)

        results = postprocess(
            raw,
            tensor_meta,
            [_make_frame()],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            confidence_threshold=0.25,
        )

        assert results[0].num_boxes == 1
        assert results[0].boxes[0].class_id == 2


class TestCocoClasses:
    def test_has_80_classes(self) -> None:
        assert len(COCO_CLASSES) == 80

    def test_first_class_is_person(self) -> None:
        assert COCO_CLASSES[0] == "person"

    def test_last_class_is_toothbrush(self) -> None:
        assert COCO_CLASSES[79] == "toothbrush"

    def test_car_is_class_2(self) -> None:
        assert COCO_CLASSES[2] == "car"


class TestNmsOptimizations:
    """Regression tests for NMS hot-path allocation and contiguity optimizations."""

    def test_inverse_letterbox_numerical_identity(self) -> None:
        from yowo.postprocess._nms import _inverse_letterbox

        boxes = np.array(
            [[160.0, 0.0, 480.0, 320.0], [0.0, 160.0, 640.0, 480.0]],
            dtype=np.float32,
        )
        # Reference: original algorithm (copy + separate subtract/divide).
        scale, pad_top, pad_left, orig_h, orig_w = 0.5, 10, 20, 480, 640
        ref = boxes.copy()
        ref[:, 0] = (boxes[:, 0] - pad_left) / scale
        ref[:, 1] = (boxes[:, 1] - pad_top) / scale
        ref[:, 2] = (boxes[:, 2] - pad_left) / scale
        ref[:, 3] = (boxes[:, 3] - pad_top) / scale
        ref[:, 0] = np.clip(ref[:, 0], 0.0, float(orig_w))
        ref[:, 2] = np.clip(ref[:, 2], 0.0, float(orig_w))
        ref[:, 1] = np.clip(ref[:, 1], 0.0, float(orig_h))
        ref[:, 3] = np.clip(ref[:, 3], 0.0, float(orig_h))

        result = _inverse_letterbox(boxes, scale, pad_top, pad_left, orig_h, orig_w)
        assert np.allclose(result, ref, atol=1e-6), f"max delta: {np.abs(result - ref).max()}"

    def test_decode_standard_boxes_unchanged(self) -> None:
        from yowo.postprocess._nms import _decode_standard

        # 3 anchors: 2 above confidence threshold, 1 below.
        # Format: cx, cy, w, h, class0, class1
        raw = np.array(
            [
                [320.0, 240.0, 100.0, 80.0, 0.9, 0.1],  # high confidence
                [200.0, 150.0, 50.0, 40.0, 0.05, 0.1],  # below threshold
                [400.0, 300.0, 60.0, 50.0, 0.1, 0.85],  # high confidence
            ],
            dtype=np.float32,
        )
        boxes = _decode_standard(
            raw,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_h=480,
            orig_w=640,
            names=["cat", "dog"],
        )
        # 2 boxes should survive; low-confidence row filtered out.
        assert len(boxes) == 2
        confidences = [b.confidence for b in boxes]
        assert all(c >= 0.25 for c in confidences)

    def test_decode_yolo26_no_redundant_copy(self) -> None:
        from yowo.postprocess._nms import _decode_yolo26

        # Verify the function returns correct boxes without crashing.
        raw = np.array(
            [
                [10.0, 10.0, 50.0, 50.0, 0.9, 0.0],
                [100.0, 100.0, 200.0, 200.0, 0.1, 1.0],  # below threshold
            ],
            dtype=np.float32,
        )
        boxes = _decode_yolo26(
            raw,
            confidence_threshold=0.5,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_h=480,
            orig_w=640,
            names=["cat"],
        )
        assert len(boxes) == 1
        assert boxes[0].confidence == pytest.approx(0.9, abs=1e-5)

    def test_inverse_letterbox_output_is_c_contiguous(self) -> None:
        from yowo.postprocess._nms import _inverse_letterbox

        boxes = np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float32)
        result = _inverse_letterbox(
            boxes, scale=0.5, pad_top=10, pad_left=20, orig_h=480, orig_w=640
        )
        assert result.flags["C_CONTIGUOUS"]

    def test_inverse_letterbox_non_contiguous_input_c_contiguous_output(self) -> None:
        from yowo.postprocess._nms import _inverse_letterbox

        # Build a non-C-contiguous 2D array via Fortran order (requires >=2 rows).
        data = np.asfortranarray(
            np.array(
                [[50.0, 50.0, 150.0, 150.0], [10.0, 10.0, 60.0, 60.0]],
                dtype=np.float32,
            )
        )
        assert not data.flags["C_CONTIGUOUS"]
        result = _inverse_letterbox(data, scale=1.0, pad_top=0, pad_left=0, orig_h=480, orig_w=640)
        # np.empty(shape, dtype) must produce C-contiguous output.
        assert result.flags["C_CONTIGUOUS"]
        assert np.allclose(result, data, atol=1e-6)

    def test_decode_standard_coordinate_values(self) -> None:
        from yowo.postprocess._nms import _decode_standard

        # Single anchor: cx=320, cy=240, w=100, h=80 -> x1=270,y1=200,x2=370,y2=280.
        raw = np.array([[320.0, 240.0, 100.0, 80.0, 0.9, 0.0]], dtype=np.float32)
        boxes = _decode_standard(
            raw,
            confidence_threshold=0.5,
            iou_threshold=0.5,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_h=480,
            orig_w=640,
            names=["person", "car"],
        )
        assert len(boxes) == 1
        assert boxes[0].x1 == pytest.approx(270.0, abs=1e-3)
        assert boxes[0].y1 == pytest.approx(200.0, abs=1e-3)
        assert boxes[0].x2 == pytest.approx(370.0, abs=1e-3)
        assert boxes[0].y2 == pytest.approx(280.0, abs=1e-3)

    def test_inverse_letterbox_clips_negative_and_overflow(self) -> None:
        """Coordinates outside [0, orig_w/h] are clipped to bounds."""
        from yowo.postprocess._nms import _inverse_letterbox

        # After inverse: x1=(50-100)/1=-50 -> 0, x2=(700-100)/1=600 -> 500
        # y1=(30-50)/1=-20 -> 0, y2=(600-50)/1=550 -> 400
        boxes = np.array([[50.0, 30.0, 700.0, 600.0]], dtype=np.float32)
        result = _inverse_letterbox(
            boxes, scale=1.0, pad_top=50, pad_left=100, orig_h=400, orig_w=500
        )
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(500.0)
        assert result[0, 3] == pytest.approx(400.0)

    def test_decode_standard_class_id_exceeds_names_fallback(self) -> None:
        """When class_id >= len(names), class_name falls back to str(class_id)."""
        from yowo.postprocess._nms import _decode_standard

        # 4 class columns, but names list has only 2 entries. Class 3 highest.
        raw = np.array(
            [[320.0, 240.0, 100.0, 80.0, 0.1, 0.1, 0.1, 0.9]],
            dtype=np.float32,
        )
        boxes = _decode_standard(
            raw,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_h=480,
            orig_w=640,
            names=["cat", "dog"],
        )
        assert len(boxes) == 1
        assert boxes[0].class_id == 3
        assert boxes[0].class_name == "3"

    def test_decode_yolo26_class_id_exceeds_names_fallback(self) -> None:
        """YOLO26: class_id beyond names list uses str(class_id) as class_name."""
        from yowo.postprocess._nms import _decode_yolo26

        raw = np.array(
            [[10.0, 10.0, 50.0, 50.0, 0.9, 5.0]],
            dtype=np.float32,
        )
        boxes = _decode_yolo26(
            raw,
            confidence_threshold=0.25,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_h=480,
            orig_w=640,
            names=["cat", "dog"],
        )
        assert len(boxes) == 1
        assert boxes[0].class_id == 5
        assert boxes[0].class_name == "5"
