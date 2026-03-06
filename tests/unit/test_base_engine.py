"""Unit tests for BaseEngine shared lifecycle via DetectionEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import DetectionEngine
from yowo.types import BackendType, Frame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)

    def _load(*args: object, **kwargs: object) -> None:
        mock.is_loaded = True

    mock.load.side_effect = _load
    mock.warmup.return_value = None
    mock.unload.return_value = None
    mock.clear_kv_cache = MagicMock()
    return mock


def _make_frame(index: int = 0) -> Frame:
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseEngineLifecycle:
    def test_health_starts_at_starting(self) -> None:
        """Health is STARTING before load() is called."""
        from yowo.types import HealthStatus

        backend = _make_mock_backend()
        engine = DetectionEngine(backend_instance=backend)
        assert engine.health == HealthStatus.STARTING

    def test_health_ready_after_load(self) -> None:
        """Health transitions to READY after load()."""
        from yowo.types import HealthStatus

        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")):
            engine = DetectionEngine(backend_instance=backend)
            engine.load()

        assert engine.health == HealthStatus.READY
        engine.close()

    def test_is_loaded_false_before_load(self) -> None:
        """is_loaded is False before load()."""
        backend = _make_mock_backend()
        engine = DetectionEngine(backend_instance=backend)
        assert not engine.is_loaded

    def test_is_loaded_true_after_load(self) -> None:
        """is_loaded is True after load()."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")):
            engine = DetectionEngine(backend_instance=backend)
            engine.load()

        assert engine.is_loaded
        engine.close()

    def test_metrics_record_inference(self) -> None:
        """After detect(), metrics.total_frames > 0."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")):
            engine = DetectionEngine(backend_instance=backend)
            engine.load()
            engine.detect([_make_frame()])

        assert engine.metrics.frames_total > 0
        engine.close()

    def test_reset_metrics(self) -> None:
        """After reset_metrics(), frame count is 0."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")):
            engine = DetectionEngine(backend_instance=backend)
            engine.load()
            engine.detect([_make_frame()])
            engine.reset_metrics()

        assert engine.metrics.frames_total == 0
        engine.close()

    def test_events_dropped_starts_at_zero(self) -> None:
        """events_dropped starts at 0."""
        backend = _make_mock_backend()
        engine = DetectionEngine(backend_instance=backend)
        assert engine.events_dropped == 0


class TestAutoLetterboxValidation:
    def test_auto_letterbox_pytorch_backend_ok(self) -> None:
        """auto_letterbox with PyTorch backend loads successfully."""
        backend = _make_mock_backend()
        backend.backend_type = BackendType.PYTORCH
        engine = DetectionEngine(backend_instance=backend, auto_letterbox=True)
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")):
            engine.load()
        assert engine.is_loaded

    def test_auto_letterbox_non_pytorch_raises_config_error(self) -> None:
        """auto_letterbox with non-PyTorch backend raises ConfigError and unloads backend."""
        from yowo.errors import ConfigError

        backend = _make_mock_backend()
        backend.backend_type = BackendType.ONNX

        engine = DetectionEngine(backend_instance=backend, auto_letterbox=True)
        fake_weights = patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt"))
        with fake_weights, pytest.raises(ConfigError, match="auto_letterbox"):
            engine.load()

        # Backend must be unloaded to prevent resource leak.
        backend.unload.assert_called_once()
        assert not engine.is_loaded

    def test_auto_letterbox_non_pytorch_close_does_not_double_unload(self) -> None:
        """close() after a ConfigError from load() skips backend.unload() (not loaded)."""
        import contextlib

        from yowo.errors import ConfigError

        backend = _make_mock_backend()
        backend.backend_type = BackendType.ONNX

        engine = DetectionEngine(backend_instance=backend, auto_letterbox=True)
        fake_weights = patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt"))
        with fake_weights, contextlib.suppress(ConfigError):
            engine.load()

        engine.close()
        # unload called once in _finalize_load cleanup, NOT again in close()
        backend.unload.assert_called_once()
