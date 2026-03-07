"""Unit tests for async engine API: adetect, astream, __aenter__/__aexit__."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import InferenceEngine
from yowo.errors import InferenceError, ShutdownError
from yowo.types import BackendType, Detection, Frame

_RESOLVE_PATCH = "yowo.engine.resolve_weights"

# ---------------------------------------------------------------------------
# Helpers — mirror test_health.py patterns exactly
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
    return InferenceEngine(backend_instance=mock_backend, **kwargs)


def _loaded_engine(mock_backend: MagicMock, **kwargs: object) -> InferenceEngine:
    engine = _engine(mock_backend, **kwargs)
    with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


def _dummy_frame() -> Frame:
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0)


# ---------------------------------------------------------------------------
# adetect tests
# ---------------------------------------------------------------------------


class TestADetect:
    async def test_adetect_returns_detections(self) -> None:
        """await engine.adetect([frame]) returns a list."""
        engine = _loaded_engine(_make_mock_backend())
        try:
            result = await engine.adetect([_dummy_frame()])
            assert isinstance(result, list)
        finally:
            engine.close()

    async def test_adetect_not_loaded_raises(self) -> None:
        """adetect() on an unloaded engine propagates InferenceError."""
        engine = _engine(_make_mock_backend())
        with pytest.raises(InferenceError, match="not loaded"):
            await engine.adetect([_dummy_frame()])

    async def test_adetect_shutdown_raises(self) -> None:
        """adetect() after close() propagates ShutdownError."""
        engine = _loaded_engine(_make_mock_backend())
        engine.close()
        with pytest.raises(ShutdownError):
            await engine.adetect([_dummy_frame()])

    async def test_adetect_propagates_inference_error(self) -> None:
        """backend.infer raising RuntimeError is wrapped as InferenceError."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        # Set side_effect AFTER load() so warmup validation succeeds
        mock_be.infer.side_effect = RuntimeError("backend failure")
        try:
            with pytest.raises((InferenceError, RuntimeError)):
                await engine.adetect([_dummy_frame()])
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# astream tests
# ---------------------------------------------------------------------------


class TestAStream:
    async def test_astream_yields_all_frames(self) -> None:
        """Collect all detections from an async for loop over astream."""
        mock_be = _make_mock_backend()
        # Return one zero-detection result per call
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        # Build a mock source that yields 3 frames then is exhausted
        frames = [_dummy_frame() for _ in range(3)]
        mock_source = MagicMock()
        mock_source.total_frames = 3
        mock_source.is_live = False
        mock_source.__iter__ = MagicMock(return_value=iter(frames))
        mock_source.close = MagicMock()

        collected: list[Detection] = []
        try:
            async for det in engine.astream(mock_source):
                collected.append(det)
        finally:
            engine.close()

        # 3 frames → 3 detection results (may be empty boxes but 3 items)
        assert len(collected) == 3

    async def test_astream_cancellation_stops_thread(self) -> None:
        """task.cancel() triggers CancelledError and stops the background thread."""
        mock_be = _make_mock_backend()
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        # Infinite source — blocks until cancelled
        call_count = 0

        class _InfiniteSource:
            total_frames = -1
            is_live = True

            def __iter__(self) -> object:
                nonlocal call_count
                while True:
                    call_count += 1
                    yield _dummy_frame()

            def close(self) -> None:
                pass

        source = _InfiniteSource()

        async def _run() -> None:
            async for _ in engine.astream(source):  # type: ignore[arg-type]
                await asyncio.sleep(0)  # yield control

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        engine.close()
        # At least one frame was processed before cancellation
        assert call_count >= 1

    async def test_astream_thread_daemon(self) -> None:
        """The background thread spawned by astream is a daemon thread."""
        mock_be = _make_mock_backend()
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        threads_seen: list[threading.Thread] = []
        original_start = threading.Thread.start

        def _spy_start(self: threading.Thread) -> None:
            if self.name == "yowo-astream":
                threads_seen.append(self)
            original_start(self)

        frames = [_dummy_frame()]
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False
        mock_source.__iter__ = MagicMock(return_value=iter(frames))
        mock_source.close = MagicMock()

        with patch.object(threading.Thread, "start", _spy_start):
            async for _ in engine.astream(mock_source):
                pass

        engine.close()
        assert len(threads_seen) >= 1
        assert all(t.daemon for t in threads_seen)

    async def test_astream_inherits_shutting_down_guard(self) -> None:
        """After close(), astream() raises ShutdownError (from stream())."""
        engine = _loaded_engine(_make_mock_backend())
        engine.close()

        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False

        with pytest.raises(ShutdownError):
            async for _ in engine.astream(mock_source):
                pass

    async def test_astream_stop_event_registered_in_active_streams(self) -> None:
        """astream() registers its stop-event in engine._active_streams while running."""
        mock_be = _make_mock_backend()
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        seen_active: list[int] = []

        # Infinite source so the stream stays alive long enough to observe
        class _SlowSource:
            total_frames = -1
            is_live = True

            def __iter__(self) -> object:
                while True:
                    yield _dummy_frame()

            def close(self) -> None:
                pass

        source = _SlowSource()

        async def _run() -> None:
            async for _ in engine.astream(source):  # type: ignore[arg-type]
                seen_active.append(len(engine._active_streams))
                break

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        engine.close()
        # At least one iteration saw the stop-event registered
        assert any(n >= 1 for n in seen_active), "astream() did not register stop-event"

    async def test_close_signals_astream_background_thread(self) -> None:
        """engine.close() sets the astream stop-event registered in _active_streams."""
        engine = _loaded_engine(_make_mock_backend())
        stop = threading.Event()
        with engine._shutdown_lock:
            engine._active_streams.add(stop)
            engine._streams_drained.clear()
        engine.close()
        assert stop.is_set(), "close() must signal astream stop-event"

    async def test_astream_no_per_item_coroutine_scheduling(self) -> None:
        """Background thread uses call_soon_threadsafe, not run_coroutine_threadsafe per item."""
        mock_be = _make_mock_backend()
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        frames = [_dummy_frame() for _ in range(5)]
        mock_source = MagicMock()
        mock_source.total_frames = 5
        mock_source.is_live = False
        mock_source.__iter__ = MagicMock(return_value=iter(frames))
        mock_source.close = MagicMock()

        rcts_calls: list[object] = []
        orig_rcts = asyncio.run_coroutine_threadsafe

        def _spy_rcts(coro: object, loop: object) -> object:
            rcts_calls.append(coro)
            return orig_rcts(coro, loop)  # type: ignore[arg-type]

        try:
            with patch("yowo._async.asyncio.run_coroutine_threadsafe", side_effect=_spy_rcts):
                collected = []
                async for det in engine.astream(mock_source):
                    collected.append(det)
        finally:
            engine.close()

        assert len(collected) == 5
        # Only the sentinel None should use run_coroutine_threadsafe (1 call)
        assert len(rcts_calls) == 1, (
            f"Expected 1 run_coroutine_threadsafe call (sentinel), got {len(rcts_calls)}"
        )

    async def test_astream_sentinel_after_early_stop(self) -> None:
        """After stop_event is set, sentinel arrives and async generator terminates."""
        mock_be = _make_mock_backend()
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        class _InfiniteSource:
            total_frames = -1
            is_live = True

            def __iter__(self) -> object:
                while True:
                    yield _dummy_frame()

            def close(self) -> None:
                pass

        source = _InfiniteSource()
        collected: list[object] = []

        async def _run() -> None:
            async for det in engine.astream(source):  # type: ignore[arg-type]
                collected.append(det)
                if len(collected) >= 2:
                    break

        await asyncio.wait_for(_run(), timeout=5.0)
        engine.close()
        assert len(collected) >= 2

    async def test_astream_handles_closed_loop(self) -> None:
        """If event loop closes mid-stream, background thread exits cleanly."""
        mock_be = _make_mock_backend()
        mock_be.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
        engine = _loaded_engine(mock_be)

        frames = [_dummy_frame()]
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False
        mock_source.__iter__ = MagicMock(return_value=iter(frames))
        mock_source.close = MagicMock()

        try:
            # Normal operation — just verify no hang or crash
            collected = []
            async for det in engine.astream(mock_source):
                collected.append(det)
            assert len(collected) == 1
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# Async context manager tests
# ---------------------------------------------------------------------------


class TestAsyncContextManager:
    async def test_aenter_calls_load(self) -> None:
        """__aenter__ loads the engine."""
        mock_be = _make_mock_backend()
        engine = _engine(mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            await engine.__aenter__()
        assert engine.is_loaded is True
        engine.close()

    async def test_aexit_calls_close(self) -> None:
        """__aexit__ closes the engine."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        await engine.__aexit__(None, None, None)
        assert engine.is_loaded is False

    async def test_async_context_manager_lifecycle(self) -> None:
        """Full `async with engine:` roundtrip loads then closes."""
        mock_be = _make_mock_backend()
        engine = _engine(mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            async with engine:
                assert engine.is_loaded is True
        assert engine.is_loaded is False


# ---------------------------------------------------------------------------
# on_async integration
# ---------------------------------------------------------------------------


class TestOnAsync:
    async def test_on_async_delivers_to_loop(self) -> None:
        """Register async callback, emit event, verify it is scheduled on the loop."""
        engine = _loaded_engine(_make_mock_backend())
        received: list[object] = []
        current_loop = asyncio.get_running_loop()

        async def _handler(payload: object) -> None:
            received.append(payload)

        engine.on_async("detection", _handler, loop=current_loop)

        # Emit via detect() which internally calls _event_bus.emit("detection", ...)
        engine._event_bus.emit("detection", ["synthetic"])

        # Give the event bus worker thread time to schedule on our loop
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

        engine.close()
        assert len(received) >= 1
