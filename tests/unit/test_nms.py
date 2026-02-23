"""Unit tests for yowo.postprocess._nms."""

from __future__ import annotations

import numpy as np
import pytest

from yowo.postprocess._nms import COCO_CLASSES, _iou, postprocess
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


def _make_model_spec(family: ModelFamily = ModelFamily.YOLO12) -> ModelSpec:
    return ModelSpec(family=family, size=ModelSize.NANO)


def _make_raw_standard(
    batch: int = 1,
    num_classes: int = 80,
    num_anchors: int = 8400,
) -> np.ndarray:
    """Return a zeroed (B, 4+num_classes, num_anchors) float32 array."""
    return np.zeros((batch, 4 + num_classes, num_anchors), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests for _iou()
# ---------------------------------------------------------------------------


class TestIou:
    def test_identical_boxes_iou_is_one(self) -> None:
        box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        result = _iou(box, boxes)
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_non_overlapping_boxes_iou_is_zero(self) -> None:
        box = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
        result = _iou(box, boxes)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_half_overlap_iou(self) -> None:
        # box: [0,0,10,10], area=100
        # other: [5,0,15,10], area=100
        # intersection: [5,0,10,10], area=50
        # union: 100+100-50=150; iou=50/150≈0.333
        box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
        boxes = np.array([[5.0, 0.0, 15.0, 10.0]], dtype=np.float32)
        result = _iou(box, boxes)
        assert result[0] == pytest.approx(50.0 / 150.0, rel=1e-4)

    def test_vectorized_multiple_boxes(self) -> None:
        box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],  # iou = 1.0
                [10.0, 10.0, 20.0, 20.0],  # iou = 0.0
                [5.0, 0.0, 15.0, 10.0],  # iou ≈ 0.333
            ],
            dtype=np.float32,
        )
        result = _iou(box, boxes)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0, abs=1e-5)
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] == pytest.approx(50.0 / 150.0, rel=1e-4)

    def test_output_dtype_is_float32(self) -> None:
        box = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        boxes = np.array([[0.0, 0.0, 5.0, 5.0]], dtype=np.float32)
        assert _iou(box, boxes).dtype == np.float32


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


class TestCocoClasses:
    def test_has_80_classes(self) -> None:
        assert len(COCO_CLASSES) == 80

    def test_first_class_is_person(self) -> None:
        assert COCO_CLASSES[0] == "person"

    def test_last_class_is_toothbrush(self) -> None:
        assert COCO_CLASSES[79] == "toothbrush"

    def test_car_is_class_2(self) -> None:
        assert COCO_CLASSES[2] == "car"
