"""Async helpers for InferenceEngine.

Extracted from engine.py to keep that module under 700 lines.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from yowo.io import FrameSource


def _enqueue_or_drop(
    q: asyncio.Queue[Any],
    item: Any,
    on_drop: Callable[[], None] | None = None,
) -> None:
    """Put an item on the async queue, dropping it if full.

    Runs on the event loop thread (via ``call_soon_threadsafe``).
    Explicitly handles ``QueueFull`` instead of letting it propagate
    as an unhandled callback exception.  Calls *on_drop* when a frame
    is discarded so callers can record the event in metrics.
    """
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        logger.debug(
            "astream: result dropped — async queue full (maxsize=%d); "
            "consumer is slower than the source. Consider increasing max_queue_size.",
            q.maxsize,
        )
        if on_drop is not None:
            on_drop()


async def astream(
    stream_fn: Callable[..., Any],
    source: FrameSource,
    emit_fn: Callable[[str, object], None],
    stop_event: threading.Event | None = None,
    on_drop: Callable[[], None] | None = None,
) -> AsyncIterator[Any]:
    """Yield results asynchronously from any source.

    Background thread runs the sync *stream_fn*. Cancellation sets stop_event.

    Args:
        stream_fn: Bound ``engine.stream`` method.
        source: Any :class:`FrameSource` (image, video, RTSP, ...).
        emit_fn: Bound ``engine._event_bus.emit`` callable for error reporting.
        stop_event: Optional external stop event; when set the background
            thread exits its iteration loop early (used by engine.close()).
        on_drop: Optional zero-argument callable invoked each time a result
            is silently dropped because the async queue is full.

    Yields:
        One result per frame (type depends on the engine).
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    _internal_stop = threading.Event()

    def _background() -> None:
        # Capture the generator explicitly so gen.close() triggers _stream_*
        # finally blocks immediately (reader.stop(), source.close()) on early exit
        # rather than waiting for CPython GC.
        gen = stream_fn(source)
        try:
            for detection in gen:
                if _internal_stop.is_set():
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    # Fire-and-forget: O(1) cross-thread overhead instead of
                    # one event-loop round-trip per frame.
                    loop.call_soon_threadsafe(_enqueue_or_drop, q, detection, on_drop)
                except RuntimeError:
                    break  # event loop is closed
        except Exception as exc:
            emit_fn("error", exc)
        finally:
            gen.close()  # ensures _stream_live reader thread stops immediately
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result(timeout=2.0)

    thread = threading.Thread(target=_background, name="yowo-astream", daemon=True)
    thread.start()
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            yield item
    except (asyncio.CancelledError, GeneratorExit):
        _internal_stop.set()
        raise
    finally:
        _internal_stop.set()
        thread.join(timeout=5.0)
