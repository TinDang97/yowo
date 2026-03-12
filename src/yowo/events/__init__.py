"""Lightweight async-capable event bus for yowo.

Background daemon thread dispatches callbacks from a SimpleQueue.
Sync and async callbacks are both supported. Non-blocking emit().

Well-known event constants:

    EVENT_DETECTION      - payload: list[Detection]
    EVENT_CLASSIFICATION - payload: list[ClassificationResult]
    EVENT_ERROR          - payload: Exception
    EVENT_HEALTH_CHANGE  - payload: HealthStatus
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from typing import Any, Optional

# Type alias for a listener entry: (callback, is_async, loop_or_None)
_ListenerEntry = tuple[Callable[..., Any], bool, Optional[asyncio.AbstractEventLoop]]

# ---------------------------------------------------------------------------
# Well-known event name constants
# ---------------------------------------------------------------------------

EVENT_DETECTION = "detection"
"""Emitted after each successful detection batch. Payload: list[Detection]."""

EVENT_CLASSIFICATION = "classification"
"""Emitted after each successful classification batch. Payload: list[ClassificationResult]."""

EVENT_ERROR = "error"
"""Emitted on inference or backend error. Payload: Exception."""

EVENT_HEALTH_CHANGE = "health_change"
"""Emitted when engine health transitions. Payload: HealthStatus."""

# ---------------------------------------------------------------------------
# Sentinel for stopping the worker thread
# ---------------------------------------------------------------------------

_SENTINEL = object()


# ---------------------------------------------------------------------------
# Async dispatch helper (avoids untyped lambda in worker thread)
# ---------------------------------------------------------------------------


def _schedule_async(
    cb: Callable[..., Any],
    payload: Any,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Schedule *cb(payload)* as a coroutine on *loop* from any thread."""

    def _thunk() -> None:
        loop.create_task(cb(payload))

    loop.call_soon_threadsafe(_thunk)


class EventBus:
    """Non-blocking async-capable event bus.

    Callbacks are dispatched on a background daemon thread so ``emit()``
    never blocks the caller.  Both synchronous and async callbacks are
    supported.

    Thread-safety:
        - ``on()``, ``on_async()``, ``remove()`` are protected by a lock.
        - ``emit()`` uses a Semaphore for queue-slot budgeting — no lock needed
          on the common path.
        - ``_has_listeners`` is read lock-free in emit(); written under lock in
          _register()/_remove(). The tiny TOCTOU is acceptable: worst case is
          one wasted emit when the last listener unregisters concurrently.
        - ``_snapshot`` is replaced atomically (Python dict assignment) under
          lock in _register()/remove(); read lock-free in _dispatch_loop().
        - ``close()`` is idempotent.

    Usage::

        bus = EventBus()
        bus.on("detection", lambda dets: print(len(dets)))
        bus.emit("detection", detections)
        bus.close()
    """

    _MAX_QUEUE = 1000

    def __init__(self) -> None:
        # Listener registry: event -> list of (callback, is_async, loop|None)
        self._listeners: dict[str, list[_ListenerEntry]] = {}
        self._lock = threading.Lock()

        # Dispatch queue and worker bookkeeping.
        # Semaphore tracks available queue slots — acquire to emit, release
        # after dispatch.  This is atomic and lock-free on the common path.
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._slots = threading.Semaphore(self._MAX_QUEUE)
        self._events_dropped = 0
        self._closed = False
        self._worker: threading.Thread | None = None

        # P0-1: zero-listener fast path — read without lock in emit().
        self._has_listeners: bool = False

        # P1-5: lock-free snapshot for _dispatch_loop — replaced atomically
        # under _lock in _register()/remove(); read lock-free in worker thread.
        self._snapshot: dict[str, list[_ListenerEntry]] = {}

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a synchronous callback for *event*.

        The callback is invoked in the worker thread with the event payload
        as the sole positional argument.  Exceptions are caught and silently
        discarded so a bad callback cannot crash the bus.
        """
        self._register(event, callback, is_async=False, loop=None)

    def on_async(
        self,
        event: str,
        callback: Callable[..., Any],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Register an async callback for *event*.

        The coroutine is scheduled on *loop* via
        ``loop.call_soon_threadsafe``.  If *loop* is ``None``, the running
        event loop at registration time is used.

        Raises:
            RuntimeError: If *loop* is ``None`` and no event loop is running.
                Callers from synchronous context must pass ``loop=`` explicitly.
        """
        if loop is None:
            loop = asyncio.get_running_loop()  # raises RuntimeError if no loop
        self._register(event, callback, is_async=True, loop=loop)

    def remove(self, event: str, callback: Callable[..., Any]) -> None:
        """Unregister *callback* from *event*.  No-op if not registered."""
        with self._lock:
            entries = self._listeners.get(event)
            if not entries:
                return
            self._listeners[event] = [e for e in entries if e[0] is not callback]
            # Rebuild snapshot atomically (dict assignment is atomic in CPython).
            self._snapshot = {k: list(v) for k, v in self._listeners.items()}
            # Reset fast-path flag when all lists are empty.
            self._has_listeners = any(self._listeners.values())

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: str, payload: Any) -> None:
        """Non-blocking enqueue of ``(event, payload)``.

        Dropped silently if the bus is closed, no listeners are registered,
        or the queue is full.
        """
        if self._closed:
            return
        # P0-1: skip Semaphore + SimpleQueue cost entirely when no listeners.
        # Read without lock — tiny TOCTOU is acceptable (see class docstring).
        if not self._has_listeners:
            return
        if not self._slots.acquire(blocking=False):
            # Queue full — count the drop under lock (rare path, acceptable cost).
            with self._lock:
                self._events_dropped += 1
            return
        self._queue.put_nowait((event, payload))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self, timeout: float = 2.0) -> None:
        """Drain the queue, stop the worker thread.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._worker is not None and self._worker.is_alive():
            self._queue.put_nowait(_SENTINEL)
            self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def events_dropped(self) -> int:
        """Number of events dropped due to queue overflow."""
        return self._events_dropped

    @property
    def is_closed(self) -> bool:
        """``True`` after :meth:`close` has been called."""
        return self._closed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(
        self,
        event: str,
        callback: Callable[..., Any],
        *,
        is_async: bool,
        loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        with self._lock:
            self._listeners.setdefault(event, []).append((callback, is_async, loop))
            # Rebuild snapshot atomically (dict assignment is atomic in CPython).
            self._snapshot = {k: list(v) for k, v in self._listeners.items()}
            self._has_listeners = True
            self._ensure_worker()

    def _ensure_worker(self) -> None:
        """Start the dispatch thread lazily on first registration.

        No-op if the bus has already been closed — prevents a new thread
        from blocking forever on the queue after ``close()`` is called.
        """
        if self._closed:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._dispatch_loop,
                name="yowo-events",
                daemon=True,
            )
            self._worker.start()

    def _dispatch_loop(self) -> None:
        """Worker thread: pull items from queue and dispatch to callbacks."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            event, payload = item
            # Release the slot so the next emit() can proceed.
            self._slots.release()
            # P1-5: lock-free snapshot read — no lock needed.
            # _snapshot is replaced atomically under _lock in _register()/remove().
            entries = self._snapshot.get(event, [])
            for cb, is_async, loop in entries:
                try:
                    if is_async and loop is not None:
                        _schedule_async(cb, payload, loop)
                    else:
                        cb(payload)
                except Exception:
                    pass


__all__ = [
    "EVENT_CLASSIFICATION",
    "EVENT_DETECTION",
    "EVENT_ERROR",
    "EVENT_HEALTH_CHANGE",
    "EventBus",
]
