"""Unit tests for Detection/BoundingBox serialization.

Tests BoundingBox.to_dict(), Detection.to_dict(), Detection.to_json(),
and the write_json refactor in io/_sink.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from yowo.types import (
    BackendType,
    BoundingBox,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(source_id: str = "test.jpg", frame_index: int = 0) -> Frame:
    return Frame(
        pixels=np.zeros((480, 640, 3), dtype=np.uint8),
        source_id=source_id,
        frame_index=frame_index,
        timestamp_ms=100.0,
    )


def _make_spec() -> ModelSpec:
    return ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)


def _make_detection(
    boxes: tuple[BoundingBox, ...] = (),
    inference_time_ms: float = 5.0,
) -> Detection:
    return Detection(
        frame=_make_frame(),
        boxes=boxes,
        inference_time_ms=inference_time_ms,
        backend=BackendType.ONNX,
        model_spec=_make_spec(),
    )


# ---------------------------------------------------------------------------
# BoundingBox.to_dict()
# ---------------------------------------------------------------------------


class TestBoundingBoxToDict:
    def test_all_fields_present(self) -> None:
        box = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=0.9, class_id=0)
        d = box.to_dict()
        assert set(d.keys()) == {"x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"}

    def test_coordinates_exact(self) -> None:
        box = BoundingBox(x1=10.5, y1=20.5, x2=100.5, y2=200.5, confidence=0.85, class_id=3)
        d = box.to_dict()
        assert d["x1"] == pytest.approx(10.5)
        assert d["y1"] == pytest.approx(20.5)
        assert d["x2"] == pytest.approx(100.5)
        assert d["y2"] == pytest.approx(200.5)

    def test_confidence_preserved(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.123, class_id=0)
        assert box.to_dict()["confidence"] == pytest.approx(0.123)

    def test_class_id_integer(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.5, class_id=7)
        d = box.to_dict()
        assert d["class_id"] == 7
        assert isinstance(d["class_id"], int)

    def test_class_name_default_empty(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.5, class_id=0)
        assert box.to_dict()["class_name"] == ""

    def test_class_name_preserved(self) -> None:
        box = BoundingBox(
            x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.5, class_id=0, class_name="person"
        )
        assert box.to_dict()["class_name"] == "person"

    def test_json_serializable(self) -> None:
        box = BoundingBox(
            x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=0.9, class_id=0, class_name="car"
        )
        d = box.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        roundtrip = json.loads(serialized)
        assert roundtrip["x1"] == pytest.approx(10.0)
        assert roundtrip["class_name"] == "car"

    def test_no_numpy_types(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.5, class_id=0)
        d = box.to_dict()
        for v in d.values():
            assert not isinstance(v, np.generic), f"numpy type leaked: {type(v)}"


# ---------------------------------------------------------------------------
# Detection.to_dict()
# ---------------------------------------------------------------------------


class TestDetectionToDict:
    def test_required_keys_present(self) -> None:
        det = _make_detection()
        d = det.to_dict()
        required = {
            "source_id",
            "frame_index",
            "timestamp_ms",
            "inference_time_ms",
            "backend",
            "model",
            "boxes",
        }
        assert required.issubset(d.keys())

    def test_source_id_from_frame(self) -> None:
        frame = Frame(
            pixels=np.zeros((100, 100, 3), dtype=np.uint8),
            source_id="rtsp://camera/1",
            frame_index=42,
            timestamp_ms=999.0,
        )
        det = Detection(
            frame=frame,
            boxes=(),
            inference_time_ms=5.0,
            backend=BackendType.PYTORCH,
            model_spec=_make_spec(),
        )
        d = det.to_dict()
        assert d["source_id"] == "rtsp://camera/1"
        assert d["frame_index"] == 42
        assert d["timestamp_ms"] == pytest.approx(999.0)

    def test_backend_is_string(self) -> None:
        det = _make_detection()
        d = det.to_dict()
        assert isinstance(d["backend"], str)
        assert d["backend"] == "onnx"

    def test_model_string_format(self) -> None:
        det = _make_detection()
        d = det.to_dict()
        assert d["model"] == "yolo26n"

    def test_model_string_yolo11_xlarge(self) -> None:
        spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.XLARGE)
        det = Detection(
            frame=_make_frame(),
            boxes=(),
            inference_time_ms=5.0,
            backend=BackendType.TENSORRT,
            model_spec=spec,
        )
        assert det.to_dict()["model"] == "yolo11x"

    def test_boxes_list_serialized(self) -> None:
        boxes = (
            BoundingBox(
                x1=0.0, y1=0.0, x2=10.0, y2=10.0, confidence=0.8, class_id=0, class_name="dog"
            ),
            BoundingBox(
                x1=5.0, y1=5.0, x2=50.0, y2=50.0, confidence=0.6, class_id=1, class_name="cat"
            ),
        )
        det = _make_detection(boxes=boxes)
        d = det.to_dict()
        assert len(d["boxes"]) == 2
        assert d["boxes"][0]["class_name"] == "dog"
        assert d["boxes"][1]["class_name"] == "cat"

    def test_empty_boxes_list(self) -> None:
        det = _make_detection(boxes=())
        d = det.to_dict()
        assert d["boxes"] == []

    def test_no_pixels_in_dict(self) -> None:
        """Frame.pixels (numpy) must NOT appear in the dict."""
        det = _make_detection()
        d = det.to_dict()
        assert "pixels" not in d
        assert "frame" not in d

    def test_no_weights_path_in_dict(self) -> None:
        """ModelSpec.weights_path (Path) must NOT appear in the dict."""
        from pathlib import Path

        spec = ModelSpec(
            family=ModelFamily.YOLO26, size=ModelSize.NANO, weights_path=Path("/tmp/w.pt")
        )
        det = Detection(
            frame=_make_frame(),
            boxes=(),
            inference_time_ms=5.0,
            backend=BackendType.PYTORCH,
            model_spec=spec,
        )
        d = det.to_dict()
        assert "weights_path" not in d
        assert "model_spec" not in d

    def test_json_serializable(self) -> None:
        boxes = (BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0, confidence=0.9, class_id=0),)
        det = _make_detection(boxes=boxes)
        d = det.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["model"] == "yolo26n"

    def test_inference_time_preserved(self) -> None:
        det = _make_detection(inference_time_ms=12.345)
        assert det.to_dict()["inference_time_ms"] == pytest.approx(12.345)


# ---------------------------------------------------------------------------
# Detection.to_json()
# ---------------------------------------------------------------------------


class TestDetectionToJson:
    def test_returns_valid_json_string(self) -> None:
        det = _make_detection()
        result = det.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_roundtrip_preserves_values(self) -> None:
        boxes = (
            BoundingBox(
                x1=1.0, y1=2.0, x2=10.0, y2=20.0, confidence=0.95, class_id=2, class_name="bike"
            ),
        )
        det = _make_detection(boxes=boxes)
        parsed = json.loads(det.to_json())
        assert parsed["boxes"][0]["class_name"] == "bike"
        assert parsed["boxes"][0]["confidence"] == pytest.approx(0.95)

    def test_indent_parameter(self) -> None:
        det = _make_detection()
        compact = det.to_json()
        indented = det.to_json(indent=2)
        assert "\n" in indented
        assert len(indented) > len(compact)
        # Both parse to same content
        assert json.loads(compact) == json.loads(indented)

    def test_no_custom_encoder_needed(self) -> None:
        """to_json() must not raise TypeError for any non-primitive type."""
        det = _make_detection()
        try:
            det.to_json()
        except (TypeError, ValueError) as exc:
            pytest.fail(f"to_json() raised unexpectedly: {exc}")

    def test_multiple_backends_all_serialize(self) -> None:
        for backend in BackendType:
            det = Detection(
                frame=_make_frame(),
                boxes=(),
                inference_time_ms=1.0,
                backend=backend,
                model_spec=_make_spec(),
            )
            parsed = json.loads(det.to_json())
            assert parsed["backend"] == backend.value
