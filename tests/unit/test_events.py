"""Unit tests for src/yowo/events/__init__.py EventBus."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

from yowo.events import EventBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for(condition: object, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll *condition* callable until truthy or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():  # type: ignore[operator]
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Delivery tests
# ---------------------------------------------------------------------------


class TestEventBusDelivery:
    def test_sync_callback_delivered(self) -> None:
        """emit() fires sync callback with correct payload."""
        bus = EventBus()
        received: list[object] = []
        bus.on("detection", received.append)
        bus.emit("detection", [1, 2, 3])
        assert _wait_for(lambda: len(received) == 1)
        assert received[0] == [1, 2, 3]
        bus.close()

    def test_multiple_callbacks_same_event(self) -> None:
        """Two callbacks registered for the same event both fire."""
        bus = EventBus()
        calls_a: list[object] = []
        calls_b: list[object] = []
        bus.on("detection", calls_a.append)
        bus.on("detection", calls_b.append)
        bus.emit("detection", "payload")
        assert _wait_for(lambda: len(calls_a) == 1 and len(calls_b) == 1)
        assert calls_a[0] == calls_b[0] == "payload"
        bus.close()

    def test_callbacks_on_different_events_are_isolated(self) -> None:
        """Emitting eventA does not trigger eventB callbacks."""
        bus = EventBus()
        a_calls: list[object] = []
        b_calls: list[object] = []
        bus.on("eventA", a_calls.append)
        bus.on("eventB", b_calls.append)
        bus.emit("eventA", "hello")
        assert _wait_for(lambda: len(a_calls) == 1)
        time.sleep(0.05)  # give worker a chance to deliver B (should not happen)
        assert len(b_calls) == 0
        bus.close()

    def test_remove_callback_stops_delivery(self) -> None:
        """After remove(), the callback is not called on subsequent emits."""
        bus = EventBus()
        calls: list[object] = []
        cb = calls.append
        bus.on("evt", cb)
        bus.emit("evt", "first")
        assert _wait_for(lambda: len(calls) == 1)
        bus.remove("evt", cb)
        bus.emit("evt", "second")
        time.sleep(0.05)
        assert len(calls) == 1  # only the first emit was delivered
        bus.close()

    def test_remove_nonexistent_callback_is_noop(self) -> None:
        """Removing a callback that was never registered raises no error."""
        bus = EventBus()
        bus.on("evt", lambda x: None)
        bus.remove("evt", lambda x: None)  # different object — should not raise
        bus.remove("noevent", lambda x: None)
        bus.close()


# ---------------------------------------------------------------------------
# Overflow / capacity
# ---------------------------------------------------------------------------


class TestEventBusOverflow:
    def test_overflow_increments_events_dropped(self) -> None:
        """When the queue is full, excess emits increment events_dropped."""
        bus = EventBus()
        # Replace the semaphore with capacity=1 to force overflow easily.
        bus._slots = threading.Semaphore(1)

        # Block the worker so the queue doesn't drain
        gate = threading.Event()
        bus.on("slow", lambda _: gate.wait())

        # First emit fills the slot; subsequent emits should be dropped.
        for i in range(5):
            bus.emit("slow", i)

        assert bus.events_dropped > 0

        gate.set()  # unblock worker
        bus.close()

    def test_events_dropped_starts_zero(self) -> None:
        """A freshly created EventBus has events_dropped == 0."""
        bus = EventBus()
        assert bus.events_dropped == 0
        bus.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestEventBusLifecycle:
    def test_close_idempotent(self) -> None:
        """Calling close() twice raises no error."""
        bus = EventBus()
        bus.on("evt", lambda x: None)
        bus.close()
        bus.close()  # should not raise

    def test_emit_after_close_is_noop(self) -> None:
        """emit() after close() does not deliver the callback."""
        bus = EventBus()
        calls: list[object] = []
        bus.on("evt", calls.append)
        bus.close()
        bus.emit("evt", "after-close")
        time.sleep(0.05)
        assert len(calls) == 0

    def test_is_closed_property(self) -> None:
        """is_closed is False before close() and True after."""
        bus = EventBus()
        bus.on("evt", lambda x: None)
        assert bus.is_closed is False
        bus.close()
        assert bus.is_closed is True

    def test_close_drains_pending_events(self) -> None:
        """Events emitted before close() are delivered before the bus shuts down."""
        bus = EventBus()
        calls: list[object] = []
        bus.on("evt", calls.append)
        bus.emit("evt", "before-close")
        bus.close(timeout=2.0)
        # After close() returns the pending event should have been dispatched.
        assert len(calls) == 1

    def test_exception_in_callback_doesnt_crash_worker(self) -> None:
        """A callback that raises must not kill the worker thread."""
        bus = EventBus()
        second_calls: list[object] = []

        def bad_cb(_: object) -> None:
            raise RuntimeError("intentional failure")

        bus.on("evt", bad_cb)
        bus.on("evt", second_calls.append)
        bus.emit("evt", "trigger")
        assert _wait_for(lambda: len(second_calls) == 1)
        bus.close()


# ---------------------------------------------------------------------------
# Thread / worker tests
# ---------------------------------------------------------------------------


class TestEventBusThreading:
    def test_lazy_worker_start(self) -> None:
        """No worker thread exists before the first on() call."""
        bus = EventBus()
        # Worker not started yet
        assert bus._worker is None
        bus.on("evt", lambda x: None)
        # Worker should now be alive
        assert bus._worker is not None
        assert bus._worker.is_alive()
        bus.close()

    def test_daemon_thread(self) -> None:
        """Worker thread is a daemon so it does not block interpreter exit."""
        bus = EventBus()
        bus.on("evt", lambda x: None)
        assert bus._worker is not None
        assert bus._worker.daemon is True
        bus.close()

    def test_register_after_close_does_not_restart_worker(self) -> None:
        """on() after close() must not restart the daemon worker thread."""
        bus = EventBus()
        received: list[object] = []
        bus.on("x", received.append)  # starts worker
        bus.close(timeout=1.0)
        assert bus.is_closed

        # Register after close — should NOT start a new thread
        bus.on("x", received.append)

        # Verify: no live worker thread with name "yowo-events"
        worker_alive = any(t.name == "yowo-events" and t.is_alive() for t in threading.enumerate())
        assert not worker_alive, "Worker restarted after close()"

        # Emit after close is a noop — callback never fires
        bus.emit("x", "payload")
        time.sleep(0.05)
        assert received == []  # nothing delivered after close


# ---------------------------------------------------------------------------
# Async callbacks
# ---------------------------------------------------------------------------


class TestEventBusAsyncCallback:
    def test_async_callback_delivered(self) -> None:
        """on_async() schedules the coroutine on the provided loop."""
        bus = EventBus()
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)

        async def my_handler(payload: object) -> None:
            pass

        bus.on_async("evt", my_handler, loop=mock_loop)
        bus.emit("evt", "data")

        assert _wait_for(lambda: mock_loop.call_soon_threadsafe.called)
        bus.close()
        # call_soon_threadsafe was called with a lambda — verify at least once
        assert mock_loop.call_soon_threadsafe.call_count >= 1

    def test_on_async_without_running_loop_raises(self) -> None:
        """on_async(loop=None) raises RuntimeError when called outside an async context."""
        bus = EventBus()

        async def cb(payload: object) -> None:
            pass

        # Not inside an event loop — asyncio.get_running_loop() should raise RuntimeError
        import pytest

        with pytest.raises(RuntimeError):
            bus.on_async("x", cb)  # loop=None, no running loop

        bus.close(timeout=0.1)
