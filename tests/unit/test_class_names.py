"""Class names flow from the user API down to postprocess.

Sibling of test_custom_num_classes.py. num_classes already threaded through the
engine, but the labels did not: DetectionEngine._postprocess called postprocess
without class_names, so every detection came back with a COCO name regardless of
what the model was trained on. A model that outputs class 2 = "Bus" was reported
as "car", the COCO name at index 2 - a wrong answer that still reads like a
right one. These tests pin the label path the way the num_classes tests pin the
count path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.config import InferenceConfig
from yowo.errors import ConfigError

VEHICLES = ["motorcycle", "car", "bus", "truck", "transporter", "container", "big_transporter"]


def _mock_backend(shape=(640, 640)):
    b = MagicMock()
    b.backend_type = MagicMock()
    b.is_loaded = True
    b.input_shape = shape
    return b


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------


class TestConfigClassNames:
    def test_default_is_none(self) -> None:
        assert InferenceConfig().class_names is None

    def test_set_value(self) -> None:
        cfg = InferenceConfig(class_names=VEHICLES)
        assert cfg.class_names == VEHICLES

    def test_a_names_list_fills_in_num_classes(self) -> None:
        """A names list IS a class count, so the two can never disagree."""
        cfg = InferenceConfig(class_names=VEHICLES)
        assert cfg.num_classes == 7

    def test_explicit_matching_num_classes_ok(self) -> None:
        cfg = InferenceConfig(class_names=VEHICLES, num_classes=7)
        assert cfg.num_classes == 7

    def test_mismatched_num_classes_raises(self) -> None:
        with pytest.raises(ConfigError, match="does not match"):
            InferenceConfig(class_names=VEHICLES, num_classes=5)

    def test_empty_names_raises(self) -> None:
        with pytest.raises(ConfigError, match="non-empty"):
            InferenceConfig(class_names=[])


# ---------------------------------------------------------------------------
# DetectionEngine stores and forwards the names
# ---------------------------------------------------------------------------


class TestDetectionEngineClassNames:
    def test_stored_on_engine(self) -> None:
        from yowo.engine import DetectionEngine

        engine = DetectionEngine(backend_instance=_mock_backend(), class_names=VEHICLES)
        assert engine._class_names == VEHICLES
        engine.close()

    def test_none_default(self) -> None:
        from yowo.engine import DetectionEngine

        engine = DetectionEngine(backend_instance=_mock_backend())
        assert engine._class_names is None
        engine.close()

    def test_names_set_num_classes_on_spec(self) -> None:
        from yowo.engine import DetectionEngine

        engine = DetectionEngine(backend_instance=_mock_backend(), class_names=VEHICLES)
        assert engine._spec.num_classes == 7
        engine.close()

    def test_postprocess_receives_class_names(self) -> None:
        """The whole point: the names reach postprocess, not COCO.

        Without the fix _postprocess called postprocess with no class_names and
        this assertion fails - postprocess is invoked with class_names=None.
        """
        from yowo.engine import DetectionEngine

        engine = DetectionEngine(backend_instance=_mock_backend(), class_names=VEHICLES)
        raw = np.zeros((1, 4 + 7, 10), dtype=np.float32)
        tensor = MagicMock()
        frames = [MagicMock()]

        with patch("yowo.engine.postprocess", return_value=[MagicMock()]) as pp:
            engine._postprocess_and_emit(raw, tensor, frames, 1.0, None)

        assert pp.call_args.kwargs["class_names"] == VEHICLES
        engine.close()

    def test_postprocess_gets_none_when_unset(self) -> None:
        from yowo.engine import DetectionEngine

        engine = DetectionEngine(backend_instance=_mock_backend())
        raw = np.zeros((1, 4 + 80, 10), dtype=np.float32)

        with patch("yowo.engine.postprocess", return_value=[MagicMock()]) as pp:
            engine._postprocess_and_emit(raw, MagicMock(), [MagicMock()], 1.0, None)

        assert pp.call_args.kwargs["class_names"] is None
        engine.close()


# ---------------------------------------------------------------------------
# End to end through postprocess: names actually land on the boxes
# ---------------------------------------------------------------------------


class TestNamesReachTheBoxes:
    def test_custom_labels_not_coco(self) -> None:
        """Run real postprocess via the engine; a class-2 box must read 'bus'."""
        from yowo.engine import DetectionEngine
        from yowo.io import preprocess
        from yowo.types import Frame, ModelFamily, ModelSize, ModelSpec

        engine = DetectionEngine(backend_instance=_mock_backend(), class_names=VEHICLES)
        engine._spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=7)

        # one detection at anchor 0, class id 2, high score. A full 8400-anchor
        # grid (the rest zero) so postprocess reads the (4+nc, anchors)
        # orientation unambiguously instead of guessing on a 1-wide array.
        raw = np.zeros((1, 4 + 7, 8400), dtype=np.float32)
        raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0], raw[0, 3, 0] = 320, 320, 40, 40
        raw[0, 4 + 2, 0] = 0.9
        frame = Frame(pixels=np.zeros((640, 640, 3), dtype=np.uint8), source_id="x")
        tensor = preprocess([frame], (640, 640))

        result = engine._postprocess_and_emit(raw, tensor, [frame], 1.0, None)[0]
        assert len(result.boxes) == 1
        assert result.boxes[0].class_id == 2
        assert result.boxes[0].class_name == "bus"  # not COCO's "car"
        engine.close()


# ---------------------------------------------------------------------------
# Convenience detect() passes class_names through **engine_kwargs
# ---------------------------------------------------------------------------


class TestConvenienceClassNames:
    def test_detect_forwards_class_names(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = []

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source"),
        ):
            from yowo._convenience import detect

            detect("img.jpg", class_names=VEHICLES)

        assert eng_cls.call_args.kwargs["class_names"] == VEHICLES
