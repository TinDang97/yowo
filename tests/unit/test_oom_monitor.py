"""Unit tests for OOM monitor daemon and three-tier recovery ladder.

All tests use mocked torch.cuda and direct _apply_oom_recovery() calls
to avoid threading race conditions.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import DetectionEngine
from yowo.types import BackendType, DeviceType, HealthStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.load.return_value = None
    mock.warmup.return_value = None
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.unload.return_value = None
    return mock


def _make_cuda_engine(mock_backend: MagicMock) -> DetectionEngine:
    """Create a DetectionEngine injected with a mock backend, reporting CUDA device."""
    from yowo.types import BackendSelection, Precision

    engine = DetectionEngine(backend_instance=mock_backend)
    # Override selection to report CUDA device so _is_cuda is True
    engine._selection = BackendSelection(
        backend=BackendType.PYTORCH,
        device_type=DeviceType.CUDA,
        precision=Precision.FP32,
        device_index=0,
        reason="Test CUDA override",
    )
    return engine


def _load_engine(engine: DetectionEngine) -> None:
    """Load engine without starting OOM monitor (no weights needed)."""
    with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
        engine.load()


# ---------------------------------------------------------------------------
# _is_cuda property
# ---------------------------------------------------------------------------


class TestIsCudaProperty:
    def test_cuda_engine_returns_true(self) -> None:
        mock = _make_mock_backend()
        engine = _make_cuda_engine(mock)
        assert engine._is_cuda is True

    def test_cpu_engine_returns_false(self) -> None:
        mock = _make_mock_backend()
        engine = DetectionEngine(backend_instance=mock)
        assert engine._is_cuda is False


# ---------------------------------------------------------------------------
# _start_oom_monitor / close
# ---------------------------------------------------------------------------


class TestOomMonitorLifecycle:
    def test_oom_monitor_starts_on_cuda_load(self) -> None:
        mock = _make_mock_backend()
        engine = _make_cuda_engine(mock)

        with (
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")),
            patch.object(engine, "_oom_monitor_loop"),
        ):
            engine.load()

        assert engine._oom_thread is not None
        assert engine._oom_thread.daemon is True
        engine._oom_stop.set()

    def test_oom_monitor_not_started_on_cpu(self) -> None:
        mock = _make_mock_backend()
        engine = DetectionEngine(backend_instance=mock)  # CPU by default
        _load_engine(engine)

        assert engine._oom_thread is None

    def test_oom_stop_event_set_on_close(self) -> None:
        mock = _make_mock_backend()
        engine = _make_cuda_engine(mock)

        with (
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")),
            patch.object(engine, "_oom_monitor_loop"),
        ):
            engine.load()

        engine.close()
        assert engine._oom_stop.is_set()


# ---------------------------------------------------------------------------
# _apply_oom_recovery — tier thresholds
# ---------------------------------------------------------------------------


class TestApplyOomRecovery:
    def _get_cpu_engine(self, batch_size: int = 4) -> DetectionEngine:
        from yowo.config import InferenceConfig

        mock = _make_mock_backend()
        engine = DetectionEngine(InferenceConfig(batch_size=batch_size), backend_instance=mock)
        _load_engine(engine)
        return engine

    def test_tier1_halves_batch_size(self) -> None:
        engine = self._get_cpu_engine(batch_size=4)
        assert engine._batch_size == 4

        engine._apply_oom_recovery(0.82)  # >= 0.80, < 0.90

        assert engine._batch_size == 2
        assert engine._oom_recovering is True
        assert engine._health_state == HealthStatus.DEGRADED

    def test_tier1_clamps_to_1(self) -> None:
        engine = self._get_cpu_engine(batch_size=1)
        engine._apply_oom_recovery(0.82)
        assert engine._batch_size == 1

    def test_tier1_saves_original_batch_size(self) -> None:
        engine = self._get_cpu_engine(batch_size=8)
        engine._apply_oom_recovery(0.82)
        assert engine._original_batch_size == 8

    def test_tier1_does_not_overwrite_original_on_second_call(self) -> None:
        engine = self._get_cpu_engine(batch_size=8)
        engine._apply_oom_recovery(0.82)  # 8 -> 4, saves original=8
        engine._apply_oom_recovery(0.82)  # 4 -> 2, should not overwrite original=8
        assert engine._original_batch_size == 8
        assert engine._batch_size == 2

    def test_tier2_attempts_precision_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        engine = self._get_cpu_engine(batch_size=4)
        # backend does NOT have set_precision — should log debug
        with caplog.at_level(logging.DEBUG):
            engine._apply_oom_recovery(0.92)  # >= 0.90, < 0.95

        assert engine._health_state == HealthStatus.DEGRADED

    def test_tier3_logs_eviction_attempt(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        engine = self._get_cpu_engine(batch_size=4)
        with caplog.at_level(logging.WARNING):
            engine._apply_oom_recovery(0.96)  # >= 0.95

        # BaseEngine stub: no FrameCollector — warning logged
        assert any(
            "evict" in r.message.lower() or "collector" in r.message.lower() for r in caplog.records
        )

    def test_clear_restores_batch_size(self) -> None:
        engine = self._get_cpu_engine(batch_size=4)
        # Simulate prior recovery
        engine._apply_oom_recovery(0.82)
        assert engine._batch_size == 2
        assert engine._oom_recovering is True

        # Drop below clear threshold
        engine._apply_oom_recovery(0.70)  # < 0.75

        assert engine._batch_size == 4
        assert engine._oom_recovering is False
        assert engine._health_state == HealthStatus.READY

    def test_below_tier1_does_nothing(self) -> None:
        engine = self._get_cpu_engine(batch_size=4)
        engine._apply_oom_recovery(0.70)  # < 0.80 and not recovering
        assert engine._batch_size == 4
        assert engine._oom_recovering is False

    def test_health_change_event_emitted_on_tier1(self) -> None:
        engine = self._get_cpu_engine(batch_size=4)
        received = threading.Event()
        events: list[HealthStatus] = []

        def _cb(data: HealthStatus) -> None:
            events.append(data)
            received.set()

        engine.on("health_change", _cb)

        engine._apply_oom_recovery(0.82)
        received.wait(timeout=1.0)

        assert HealthStatus.DEGRADED in events

    def test_health_change_event_emitted_on_clear(self) -> None:
        engine = self._get_cpu_engine(batch_size=4)
        received = threading.Event()
        events: list[HealthStatus] = []

        def _cb(data: HealthStatus) -> None:
            events.append(data)
            received.set()

        engine.on("health_change", _cb)

        # First: trigger DEGRADED and wait for it
        engine._apply_oom_recovery(0.82)
        received.wait(timeout=1.0)
        received.clear()
        events.clear()

        # Then: trigger clear (READY) and wait for it
        engine._apply_oom_recovery(0.70)
        received.wait(timeout=1.0)

        assert HealthStatus.READY in events


# ---------------------------------------------------------------------------
# _halve_batch_size — PreprocessBuffer reallocation
# ---------------------------------------------------------------------------


class TestHalveBatchSize:
    def test_reallocates_preprocess_buffer(self) -> None:
        from yowo.config import InferenceConfig

        mock = _make_mock_backend()
        engine = DetectionEngine(InferenceConfig(batch_size=4), backend_instance=mock)
        _load_engine(engine)

        # Buffer exists after load
        assert engine._preprocess_buf is not None
        old_capacity = engine._preprocess_buf.capacity

        engine._halve_batch_size()

        assert engine._preprocess_buf is not None
        assert engine._preprocess_buf.capacity == max(1, old_capacity // 2)

    def test_no_buffer_realloc_when_preprocess_buf_is_none(self) -> None:
        """auto_letterbox mode: _preprocess_buf is None; halve should not crash."""
        from yowo.config import InferenceConfig

        mock = _make_mock_backend()
        engine = DetectionEngine(
            InferenceConfig(batch_size=4, auto_letterbox=True), backend_instance=mock
        )
        # Don't call load — manually set state to avoid letterbox ConfigError
        engine._batch_size = 4
        engine._original_batch_size = 4
        engine._preprocess_buf = None

        # Should not raise
        engine._halve_batch_size()
        assert engine._batch_size == 2
