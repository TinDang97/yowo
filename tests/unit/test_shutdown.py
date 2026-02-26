"""Unit tests for graceful shutdown behavior."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import yowo
from yowo.backends import InferenceBackend
from yowo.engine import InferenceEngine
from yowo.errors import ShutdownError, YowoError
from yowo.types import BackendType, HealthStatus

_RESOLVE_PATCH = "yowo.engine.resolve_weights"


# ---------------------------------------------------------------------------
# Helpers (mirrors test_health.py patterns)
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
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
    return InferenceEngine(backend_instance=mock_backend, **kwargs)


def _loaded_engine(mock_backend: MagicMock, **kwargs: object) -> InferenceEngine:
    engine = _engine(mock_backend, **kwargs)
    with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


# ---------------------------------------------------------------------------
# ShutdownError hierarchy and export
# ---------------------------------------------------------------------------


class TestShutdownError:
    def test_ShutdownError_inherits_YowoError(self) -> None:
        err = ShutdownError("test")
        assert isinstance(err, YowoError)

    def test_ShutdownError_exported(self) -> None:
        assert hasattr(yowo, "ShutdownError")
        assert yowo.ShutdownError is ShutdownError

    def test_ShutdownError_is_exception(self) -> None:
        with pytest.raises(ShutdownError, match="shutting down"):
            raise ShutdownError("shutting down")


# ---------------------------------------------------------------------------
# detect() / stream() rejection after shutdown
# ---------------------------------------------------------------------------


class TestShutdownGuards:
    def test_detect_after_shutdown_raises_ShutdownError(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        with pytest.raises(ShutdownError):
            engine.detect([])

    def test_stream_after_shutdown_raises_ShutdownError(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False
        with pytest.raises(ShutdownError):
            list(engine.stream(mock_source))

    def test_detect_raises_ShutdownError_not_InferenceError(self) -> None:
        """Shutdown takes priority over the 'not loaded' check."""
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        with pytest.raises(ShutdownError):
            engine.detect([])


# ---------------------------------------------------------------------------
# HealthStatus transitions during close
# ---------------------------------------------------------------------------


class TestCloseHealthTransitions:
    def test_close_sets_health_to_closed(self) -> None:
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        assert engine.health == HealthStatus.CLOSED

    def test_close_idempotent(self) -> None:
        """Second close() is a no-op — unload called exactly once."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        engine.close()
        engine.close()
        assert engine.health == HealthStatus.CLOSED
        # close() is idempotent: second call returns immediately without unloading again.
        assert mock_be.unload.call_count == 1

    def test_close_calls_backend_unload(self) -> None:
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        engine.close()
        mock_be.unload.assert_called()

    def test_close_before_load_noop(self) -> None:
        """close() on a never-loaded engine raises no exception and sets CLOSED."""
        mock_be = _make_mock_backend()
        engine = _engine(mock_be)
        # Engine is STARTING (not loaded) — close() must not raise.
        engine.close()
        assert engine.health == HealthStatus.CLOSED
        # unload must not be called — _loaded is False so we skip the unload call.
        mock_be.unload.assert_not_called()

    def test_health_shutting_down_set_before_closed(self) -> None:
        """_health_state transitions through SHUTTING_DOWN before CLOSED."""
        engine = _loaded_engine(_make_mock_backend())
        observed: list[HealthStatus] = []

        # Monkey-patch event bus close to observe intermediate state
        original_close = engine._event_bus.close

        def _spy_close(**kwargs: object) -> None:
            observed.append(engine._health_state)
            original_close(**kwargs)  # type: ignore[call-arg]

        engine._event_bus.close = _spy_close  # type: ignore[method-assign]
        engine.close()
        # During event bus close, state should have been SHUTTING_DOWN
        assert HealthStatus.SHUTTING_DOWN in observed
        assert engine.health == HealthStatus.CLOSED


# ---------------------------------------------------------------------------
# Active stream tracking
# ---------------------------------------------------------------------------


class TestActiveStreamTracking:
    def test_active_streams_cleared_on_close(self) -> None:
        """After close(), _active_streams set is empty."""
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        assert len(engine._active_streams) == 0

    def test_close_closes_event_bus(self) -> None:
        """After close(), the event bus is_closed."""
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        assert engine._event_bus.is_closed is True

    def test_close_signals_in_progress_stream(self) -> None:
        """close() signals stop-events registered by active stream paths."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)

        # Manually inject a stop-event (simulating what _stream_* does)
        stop = threading.Event()
        with engine._shutdown_lock:
            engine._active_streams.add(stop)

        # close() should signal the stop-event
        engine.close()

        assert stop.is_set(), "close() did not signal active stream stop-event"

    def test_unload_raises_state_still_cleaned_up(self) -> None:
        """backend.unload() raising must not prevent _loaded=False and health=CLOSED."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)

        # Override side_effect AFTER engine is loaded to make unload() raise
        mock_be.unload.side_effect = RuntimeError("unload boom")

        assert engine.is_loaded
        assert engine.health == HealthStatus.READY

        engine.close()  # unload() will raise internally

        assert not engine.is_loaded, "_loaded must be False even if unload() raised"
        assert engine.health == HealthStatus.CLOSED, "health must be CLOSED even if unload() raised"


# ---------------------------------------------------------------------------
# Event bus delegation on engine
# ---------------------------------------------------------------------------


class TestEngineEventDelegation:
    def test_on_method_registers_callback(self) -> None:
        """engine.on() registers a callback that fires on emit."""
        import time

        engine = _loaded_engine(_make_mock_backend())
        calls: list[object] = []
        engine.on("detection", calls.append)
        engine._event_bus.emit("detection", ["det1"])
        deadline = time.monotonic() + 2.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(calls) == 1
        engine.close()

    def test_events_dropped_property(self) -> None:
        """engine.events_dropped delegates to event bus."""
        engine = _loaded_engine(_make_mock_backend())
        assert engine.events_dropped == 0
        engine.close()


# ---------------------------------------------------------------------------
# Eager shutdown guard + _streams_drained event
# ---------------------------------------------------------------------------


class TestEagerShutdownAndDrainedEvent:
    def test_stream_shutdown_error_raised_eagerly(self) -> None:
        """stream() raises ShutdownError before generator iteration (eager guard)."""
        from unittest.mock import MagicMock

        from yowo.errors import ShutdownError

        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        mock_source = MagicMock()
        mock_source.total_frames = 5
        mock_source.is_live = False
        # stream() is now a regular function — ShutdownError raised on call, not on next()
        with pytest.raises(ShutdownError):
            engine.stream(mock_source)  # must raise HERE, not on iteration

    def test_streams_drained_set_by_production_stream_sync(self) -> None:
        """_stream_sync sets _streams_drained via production code after completing."""
        import numpy as np

        from yowo.types import Frame

        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        frame = Frame(
            pixels=np.zeros((480, 640, 3), dtype=np.uint8),
            source_id="test",
            frame_index=0,
        )
        mock_source = MagicMock()
        mock_source.__iter__ = MagicMock(return_value=iter([frame]))
        mock_source.close = MagicMock()

        assert engine._streams_drained.is_set()  # empty at start

        # Consume the production _stream_sync generator — exercises add/clear/discard/set
        list(engine._stream_sync(mock_source))

        assert engine._streams_drained.is_set()  # set by production code after drain
        assert not engine._active_streams
        engine.close()
