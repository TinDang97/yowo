"""Async helpers for InferenceEngine.

Extracted from engine.py to keep that module under 700 lines.
"""

from __future__ import annotations

import asyncio
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
) -> AsyncIterator[Detection]:
    """Yield detections asynchronously from any source.

    Background thread runs the sync *stream_fn*. Cancellation sets stop_event.

    Args:
        stream_fn: Bound ``engine.stream`` method.
        source: Any :class:`FrameSource` (image, video, RTSP, ...).
        emit_fn: Bound ``engine._event_bus.emit`` callable for error reporting.

    Yields:
        One :class:`Detection` per frame.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Detection | None] = asyncio.Queue(maxsize=64)
    stop_event = threading.Event()

    def _background() -> None:
        try:
            for detection in stream_fn(source):
                if stop_event.is_set():
                    break
                future = asyncio.run_coroutine_threadsafe(q.put(detection), loop)
                try:
                    future.result(timeout=10.0)
                except Exception:
                    break
        except Exception as exc:
            emit_fn("error", exc)
        finally:
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
        stop_event.set()
        raise
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
