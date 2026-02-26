"""Unit tests for engine health state machine.

Tests HealthStatus enum and InferenceEngine.health property transitions.
No real hardware — all backends are mocked.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import InferenceEngine
from yowo.metrics import EngineMetrics
from yowo.types import (
    BackendType,
    HealthStatus,
)

_RESOLVE_PATCH = "yowo.engine.resolve_weights"


# ---------------------------------------------------------------------------
# Helpers (mirrors test_engine.py patterns)
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    """Return a MagicMock satisfying the InferenceBackend protocol."""
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)

    def _load(*args: object, **kwargs: object) -> None:
        mock.is_loaded = True

    def _unload(*args: object, **kwargs: object) -> None:
        mock.is_loaded = False

    mock.load.side_effect = _load
    mock.unload.side_effect = _unload
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.warmup.return_value = None
    return mock


def _engine(mock_backend: MagicMock, **kwargs: object) -> InferenceEngine:
    """Return an engine using an injected backend — no hardware mocks needed."""
    return InferenceEngine(backend_instance=mock_backend, **kwargs)


def _loaded_engine(mock_backend: MagicMock, **kwargs: object) -> InferenceEngine:
    """Return a loaded engine with resolve_weights patched."""
    engine = _engine(mock_backend, **kwargs)
    with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


# ---------------------------------------------------------------------------
# HealthStatus enum
# ---------------------------------------------------------------------------


class TestHealthStatusEnum:
    def test_all_values_are_strings(self) -> None:
        for status in HealthStatus:
            assert isinstance(status, str)

    def test_expected_values(self) -> None:
        assert HealthStatus.STARTING == "starting"
        assert HealthStatus.READY == "ready"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.SHUTTING_DOWN == "shutting_down"
        assert HealthStatus.CLOSED == "closed"

    def test_public_export(self) -> None:
        import yowo

        assert hasattr(yowo, "HealthStatus")
        assert yowo.HealthStatus.READY == "ready"

    def test_str_enum_comparison(self) -> None:
        assert HealthStatus.READY == "ready"
        assert HealthStatus.DEGRADED != "ready"


# ---------------------------------------------------------------------------
# Engine.health transitions
# ---------------------------------------------------------------------------


class TestEngineHealthTransitions:
    def test_starting_before_load(self) -> None:
        engine = _engine(_make_mock_backend())
        assert engine.health == HealthStatus.STARTING

    def test_ready_after_load(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        assert engine.health == HealthStatus.READY
        engine.close()

    def test_closed_after_close(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        assert engine.health == HealthStatus.CLOSED

    def test_closed_is_terminal(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        engine.close()  # idempotent
        assert engine.health == HealthStatus.CLOSED

    def test_degraded_on_error_threshold(self) -> None:
        engine = _loaded_engine(_make_mock_backend(), error_threshold=3)
        engine._metrics.record_error()
        engine._metrics.record_error()
        engine._metrics.record_error()
        assert engine.health == HealthStatus.DEGRADED
        engine.close()

    def test_not_degraded_below_error_threshold(self) -> None:
        engine = _loaded_engine(_make_mock_backend(), error_threshold=5)
        engine._metrics.record_error()
        engine._metrics.record_error()
        assert engine.health == HealthStatus.READY
        engine.close()

    def test_degraded_on_stale_stream(self) -> None:
        """If no frame in >30s after processing started, health = DEGRADED."""
        engine = _loaded_engine(_make_mock_backend())
        engine._metrics.record_inference(5.0, batch_size=1)
        engine._metrics._last_frame_time = time.monotonic() - 31.0
        assert engine.health == HealthStatus.DEGRADED
        engine.close()

    def test_not_degraded_with_recent_frame(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine._metrics.record_inference(5.0, batch_size=1)
        assert engine.health == HealthStatus.READY
        engine.close()

    def test_not_degraded_with_zero_frames(self) -> None:
        """Engine with no frames processed should stay READY, not DEGRADED."""
        engine = _loaded_engine(_make_mock_backend())
        assert engine.health == HealthStatus.READY
        engine.close()

    def test_custom_error_threshold_one(self) -> None:
        engine = _loaded_engine(_make_mock_backend(), error_threshold=1)
        engine._metrics.record_error()
        assert engine.health == HealthStatus.DEGRADED
        engine.close()

    def test_health_via_context_manager(self) -> None:
        mock_be = _make_mock_backend()
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")), _engine(mock_be) as engine:
            assert engine.health == HealthStatus.READY
        assert engine.health == HealthStatus.CLOSED

    def test_metrics_property_returns_snapshot(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        snap = engine.metrics
        assert isinstance(snap, EngineMetrics)
        assert snap.frames_total == 0
        engine.close()

    def test_reset_metrics_zeroes_counts(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine._metrics.record_inference(5.0, batch_size=5)
        engine._metrics.record_error()
        engine.reset_metrics()
        snap = engine.metrics
        assert snap.frames_total == 0
        assert snap.errors_total == 0
        engine.close()

    def test_starting_state_ignores_staleness(self) -> None:
        """Before load(), STARTING takes priority over staleness check."""
        engine = _engine(_make_mock_backend())
        assert engine.health == HealthStatus.STARTING

    def test_error_threshold_config_validation(self) -> None:
        """error_threshold < 1 raises ConfigError."""
        from yowo.errors import ConfigError

        mock_be = _make_mock_backend()
        with pytest.raises(ConfigError, match="error_threshold"):
            _engine(mock_be, error_threshold=0)
