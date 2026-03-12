"""Unit tests for OBBEngine and OBBConfig."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.errors import ConfigError, InferenceError, ShutdownError
from yowo.types import BackendType, Frame, ModelFamily, ModelSize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend(nc: int = 15) -> MagicMock:
    """MagicMock satisfying InferenceBackend protocol for OBB detection."""
    backend = MagicMock(spec=InferenceBackend)
    backend.backend_type = BackendType.PYTORCH
    backend.is_loaded = False
    backend.input_shape = (640, 640)

    # OBB head output: (1, 4+nc+1, 8400) — all zeros (below conf threshold)
    raw = np.zeros((1, 4 + nc + 1, 8400), dtype=np.float32)
    backend.infer.return_value = raw

    def _load(*args: object, **kwargs: object) -> None:
        backend.is_loaded = True

    backend.load.side_effect = _load
    backend.warmup.return_value = None
    backend.unload.return_value = None
    backend.clear_kv_cache = MagicMock()
    return backend


def _make_frame(index: int = 0) -> Frame:
    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=index)


# ---------------------------------------------------------------------------
# OBBConfig tests
# ---------------------------------------------------------------------------


class TestOBBConfig:
    def test_default_construction(self) -> None:
        """OBBConfig can be constructed with defaults."""
        from yowo.config import OBBConfig

        cfg = OBBConfig()
        assert cfg.model_family == ModelFamily.YOLO11
        assert cfg.model_size == ModelSize.NANO
        assert cfg.confidence_threshold == 0.25
        assert cfg.iou_threshold == 0.45
        assert cfg.num_classes is None

    def test_num_classes_zero_raises(self) -> None:
        """OBBConfig raises ConfigError when num_classes < 1."""
        from yowo.config import OBBConfig

        with pytest.raises(ConfigError, match="num_classes"):
            OBBConfig(num_classes=0)

    def test_num_classes_negative_raises(self) -> None:
        """OBBConfig raises ConfigError when num_classes is negative."""
        from yowo.config import OBBConfig

        with pytest.raises(ConfigError, match="num_classes"):
            OBBConfig(num_classes=-1)

    def test_num_classes_one_valid(self) -> None:
        """OBBConfig accepts num_classes=1."""
        from yowo.config import OBBConfig

        cfg = OBBConfig(num_classes=1)
        assert cfg.num_classes == 1

    def test_batch_size_zero_raises(self) -> None:
        """OBBConfig raises ConfigError when batch_size < 1."""
        from yowo.config import OBBConfig

        with pytest.raises(ConfigError, match="batch_size"):
            OBBConfig(batch_size=0)

    def test_invalid_log_level_raises(self) -> None:
        """OBBConfig raises ConfigError for invalid log_level."""
        from yowo.config import OBBConfig

        with pytest.raises(ConfigError, match="log_level"):
            OBBConfig(log_level="VERBOSE")

    def test_no_top_k_field(self) -> None:
        """OBBConfig does not have a top_k field (unlike ClassificationConfig)."""
        import dataclasses

        from yowo.config import OBBConfig

        field_names = {f.name for f in dataclasses.fields(OBBConfig)}
        assert "top_k" not in field_names

    def test_has_confidence_and_iou(self) -> None:
        """OBBConfig has confidence_threshold and iou_threshold fields."""
        import dataclasses

        from yowo.config import OBBConfig

        field_names = {f.name for f in dataclasses.fields(OBBConfig)}
        assert "confidence_threshold" in field_names
        assert "iou_threshold" in field_names


# ---------------------------------------------------------------------------
# OBBEngine construction tests
# ---------------------------------------------------------------------------


class TestOBBEngineInit:
    def test_init_with_kwargs(self) -> None:
        """OBBEngine can be constructed via keyword args without error."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        engine = OBBEngine(
            backend_instance=backend,
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.NANO,
        )
        assert engine is not None

    def test_init_with_config(self) -> None:
        """OBBEngine can be constructed via OBBConfig without error."""
        from yowo.config import OBBConfig
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        config = OBBConfig(
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.NANO,
        )
        engine = OBBEngine(config, backend_instance=backend)
        assert engine is not None

    def test_spec_task_is_obb(self) -> None:
        """OBBEngine._spec.task == 'obb'."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        engine = OBBEngine(backend_instance=backend)
        assert engine._spec.task == "obb"

    def test_result_event_name(self) -> None:
        """OBBEngine._result_event_name returns 'obb_detection'."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        engine = OBBEngine(backend_instance=backend)
        assert engine._result_event_name == "obb_detection"


# ---------------------------------------------------------------------------
# OBBEngine lifecycle tests
# ---------------------------------------------------------------------------


class TestOBBEngineLifecycle:
    def test_detect_obb_not_loaded_raises(self) -> None:
        """detect_obb() before load() raises InferenceError."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        engine = OBBEngine(backend_instance=backend)

        with pytest.raises(InferenceError, match="not loaded"):
            engine.detect_obb([_make_frame()])

    def test_detect_obb_after_close_raises_shutdown(self) -> None:
        """detect_obb() after close() raises ShutdownError."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/yolo11n-obb.pt")):
            engine = OBBEngine(backend_instance=backend)
            engine.load()
            engine.close()

        with pytest.raises(ShutdownError):
            engine.detect_obb([_make_frame()])

    def test_load_and_detect_returns_obb_detections(self) -> None:
        """load() + detect_obb() returns a list of OBBDetection."""
        from yowo.obb_engine import OBBEngine
        from yowo.types import OBBDetection

        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/yolo11n-obb.pt")):
            engine = OBBEngine(backend_instance=backend)
            engine.load()
            results = engine.detect_obb([_make_frame()])
            engine.close()

        assert len(results) == 1
        assert isinstance(results[0], OBBDetection)

    def test_detect_obb_zero_boxes_on_all_zeros_output(self) -> None:
        """All-zeros backend output (below conf) returns OBBDetection with 0 boxes."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/yolo11n-obb.pt")):
            engine = OBBEngine(backend_instance=backend)
            engine.load()
            results = engine.detect_obb([_make_frame()])
            engine.close()

        assert results[0].boxes == ()

    def test_context_manager(self) -> None:
        """Context manager calls load() on enter, close() on exit."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        with (
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/yolo11n-obb.pt")),
            OBBEngine(backend_instance=backend) as engine,
        ):
            assert engine.is_loaded
            results = engine.detect_obb([_make_frame()])

        assert len(results) == 1
        assert not engine.is_loaded

    def test_events_fired_on_detect_obb(self) -> None:
        """on('obb_detection', cb) fires; 'detection' and 'classification' do NOT fire."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()
        obb_received: list[object] = []
        det_received: list[object] = []
        cls_received: list[object] = []

        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/yolo11n-obb.pt")):
            engine = OBBEngine(backend_instance=backend)
            engine.on("obb_detection", lambda results: obb_received.extend(results))
            engine.on("detection", lambda results: det_received.extend(results))
            engine.on("classification", lambda results: cls_received.extend(results))
            engine.load()
            engine.detect_obb([_make_frame()])
            engine.close()

        assert len(obb_received) == 1
        assert len(det_received) == 0, "OBBEngine must not emit 'detection' events"
        assert len(cls_received) == 0, "OBBEngine must not emit 'classification' events"


# ---------------------------------------------------------------------------
# OBBEngine _process_batch tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OBBEngine tune profile tests (INT-P1 regression)
# ---------------------------------------------------------------------------


class TestOBBEngineTuneProfile:
    def test_obb_engine_loads_tune_profile_when_no_backend_instance(self) -> None:
        """OBBEngine calls _load_tune_profile when backend_instance is None."""
        from yowo.obb_engine import OBBEngine

        mock_hw = MagicMock()
        mock_cfg_updated = MagicMock()

        with (
            patch("yowo.obb_engine._load_tune_profile", return_value=mock_cfg_updated) as mock_load,
            patch("yowo.obb_engine.get_hardware_profile", return_value=mock_hw),
            # Prevent BaseEngine.__init__ from doing real work
            patch("yowo.engine.BaseEngine.__init__", return_value=None),
        ):
            engine = OBBEngine.__new__(OBBEngine)
            # Manually run just __init__ logic up to super().__init__
            # by calling the full __init__ but with a patched BaseEngine
            OBBEngine.__init__(engine, backend_instance=None)

        mock_load.assert_called_once()
        call_args = mock_load.call_args
        # First arg is spec, second is cfg, third is hw
        assert call_args.args[2] is mock_hw

    def test_obb_engine_skips_tune_profile_when_backend_instance_provided(self) -> None:
        """OBBEngine does NOT call _load_tune_profile when backend_instance is given."""
        from yowo.obb_engine import OBBEngine

        backend = _make_mock_backend()

        with (
            patch("yowo.obb_engine._load_tune_profile") as mock_load,
            patch("yowo.obb_engine.get_hardware_profile"),
        ):
            OBBEngine(backend_instance=backend)

        mock_load.assert_not_called()


class TestOBBEngineProcessBatch:
    def test_process_batch_zero_output_returns_empty_boxes(self) -> None:
        """_process_batch with all-zeros output returns OBBDetection with no boxes."""
        from yowo.obb_engine import OBBEngine
        from yowo.types import OBBDetection

        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/yolo11n-obb.pt")):
            engine = OBBEngine(backend_instance=backend)
            engine.load()

            # Simulate raw output (1, 20, 8400) nc=15 -> 4+15+1=20
            raw = np.zeros((1, 20, 8400), dtype=np.float32)
            frame = _make_frame()
            from yowo.types import PreprocessedTensor

            tensor = PreprocessedTensor(
                data=np.zeros((1, 3, 640, 640), dtype=np.float32),
                original_shapes=((640, 640),),
                input_shape=(640, 640),
                scale_factors=((1.0, 1.0),),
                pad_offsets=((0, 0),),
            )
            results = engine._process_batch(raw, tensor, [frame], elapsed_ms=10.0, scratch=None)
            engine.close()

        assert len(results) == 1
        assert isinstance(results[0], OBBDetection)
        assert results[0].boxes == ()
        assert results[0].inference_time_ms == 10.0
