"""Async helpers for InferenceEngine.

Extracted from engine.py to keep that module under 700 lines.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yowo.io import FrameSource
    from yowo.types import Detection


async def astream(
    stream_fn: Callable[..., Any],
    source: FrameSource,
    emit_fn: Callable[[str, object], None],
    stop_event: threading.Event | None = None,
) -> AsyncIterator[Detection]:
    """Yield detections asynchronously from any source.

    Background thread runs the sync *stream_fn*. Cancellation sets stop_event.

    Args:
        stream_fn: Bound ``engine.stream`` method.
        source: Any :class:`FrameSource` (image, video, RTSP, ...).
        emit_fn: Bound ``engine._event_bus.emit`` callable for error reporting.
        stop_event: Optional external stop event; when set the background
            thread exits its iteration loop early (used by engine.close()).

    Yields:
        One :class:`Detection` per frame.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Detection | None] = asyncio.Queue(maxsize=64)
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
                future = asyncio.run_coroutine_threadsafe(q.put(detection), loop)
                try:
                    future.result(timeout=1.0)
                except Exception:
                    break
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
