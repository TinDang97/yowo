"""Streaming strategy mixin for BaseEngine.

Extracted from ``engine.py`` to keep file size manageable.
These are private implementation details — not part of the public API.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

from yowo.errors import ShutdownError
from yowo.io import FrameSource, PreparedItem, ThreadedFrameReader, preprocess, preprocess_into
from yowo.types import Frame, FrameDropPolicy

logger = logging.getLogger(__name__)


class StreamingMixin:
    """Streaming strategy methods for BaseEngine.

    Must only be used as a mixin with BaseEngine — relies on its
    instance attributes (``_backend``, ``_batch_size``, etc.).
    """

    def _stream_dispatch(self, source: FrameSource) -> Iterator[Any]:
        self._backend.clear_kv_cache()  # type: ignore[attr-defined]
        if not self._prefetch:  # type: ignore[attr-defined]
            yield from self._stream_sync(source)
        elif source.total_frames == 1:
            yield from self._stream_single(source)
        elif source.is_live:
            yield from self._stream_live(source)
        else:
            yield from self._stream_pipeline(source)

    def _stream_single(self, source: FrameSource) -> Iterator[Any]:
        """Fast path for single-image sources — no threading overhead."""
        _stop = threading.Event()
        with self._shutdown_lock:  # type: ignore[attr-defined]
            if self._shutting_down.is_set():  # type: ignore[attr-defined]
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)  # type: ignore[attr-defined]
            self._streams_drained.clear()  # type: ignore[attr-defined]
        try:
            for frame in source:
                if _stop.is_set():
                    break
                yield from self._dispatch_frames([frame])  # type: ignore[attr-defined]
        finally:
            self._active_streams.discard(_stop)  # type: ignore[attr-defined]
            if not self._active_streams:  # type: ignore[attr-defined]
                self._streams_drained.set()  # type: ignore[attr-defined]
            source.close()

    def _stream_live(self, source: FrameSource) -> Iterator[Any]:
        """Live source path: threaded reader, batch=1, 30 s idle timeout."""
        _MAX_IDLE_S = 30.0
        _POLL_TIMEOUT = 1.0
        if self._batch_size > 1:  # type: ignore[attr-defined]
            logger.debug(
                "Live source: using batch=1 (configured batch_size=%d ignored)",
                self._batch_size,  # type: ignore[attr-defined]
            )
        target_size = (
            self._model_meta.input_height,  # type: ignore[attr-defined]
            self._model_meta.input_width,  # type: ignore[attr-defined]
        )
        reader = ThreadedFrameReader(
            source,
            max_queue_size=self._max_queue_size,  # type: ignore[attr-defined]
            policy=self._frame_drop_policy,  # type: ignore[attr-defined]
            preprocess_fn=preprocess,
            target_size=target_size,
        )
        _stop = threading.Event()
        with self._shutdown_lock:  # type: ignore[attr-defined]
            if self._shutting_down.is_set():  # type: ignore[attr-defined]
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)  # type: ignore[attr-defined]
            self._streams_drained.clear()  # type: ignore[attr-defined]
        reader.start()
        try:
            idle_since: float | None = None
            while not _stop.is_set():
                item = reader.get(timeout=_POLL_TIMEOUT)
                if item is not None:
                    idle_since = None
                    assert isinstance(item, PreparedItem)
                    yield from self._infer_from_tensor(  # type: ignore[attr-defined]
                        item.tensor,
                        [item.frame],
                        scratch=self._postprocess_buf,  # type: ignore[attr-defined]
                    )
                elif reader.is_exhausted:
                    break
                else:
                    now = time.monotonic()
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since >= _MAX_IDLE_S:
                        logger.warning(
                            "Live source idle for %.0fs, terminating stream", now - idle_since
                        )
                        break
        finally:
            self._active_streams.discard(_stop)  # type: ignore[attr-defined]
            if not self._active_streams:  # type: ignore[attr-defined]
                self._streams_drained.set()  # type: ignore[attr-defined]
            reader.stop()
            source.close()

    def _stream_pipeline(self, source: FrameSource) -> Iterator[Any]:
        """Offline source: threaded prefetch + pipeline overlap."""
        from collections import deque as Deque
        from concurrent.futures import Future, ThreadPoolExecutor

        from yowo.io._decode import PreprocessBufferPool

        _stop = threading.Event()
        with self._shutdown_lock:  # type: ignore[attr-defined]
            if self._shutting_down.is_set():  # type: ignore[attr-defined]
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)  # type: ignore[attr-defined]
            self._streams_drained.clear()  # type: ignore[attr-defined]
        reader = ThreadedFrameReader(
            source,
            max_queue_size=self._batch_size * 2,  # type: ignore[attr-defined]
            policy=FrameDropPolicy.NONE,
        )
        reader.start()
        concurrent = self._pipeline_workers > 1  # type: ignore[attr-defined]
        infer_lock = threading.Lock() if concurrent else None
        target = (
            self._model_meta.input_height,  # type: ignore[attr-defined]
            self._model_meta.input_width,  # type: ignore[attr-defined]
        )
        buffer_pool: PreprocessBufferPool | None = None
        if concurrent:
            buffer_pool = PreprocessBufferPool(
                self._pipeline_workers,  # type: ignore[attr-defined]
                self._batch_size,  # type: ignore[attr-defined]
                target,
            )

        def _infer_batch(frames: list[Frame]) -> list[Any]:
            if concurrent:
                assert buffer_pool is not None
                buf = buffer_pool.acquire()
                try:
                    tensor = preprocess_into(frames, target, buf)
                    with infer_lock:  # type: ignore[union-attr]
                        return self._infer_from_tensor(tensor, frames, scratch=None)  # type: ignore[attr-defined]
                except Exception:
                    self._metrics.record_error()  # type: ignore[attr-defined]
                    raise
                finally:
                    buffer_pool.release(buf)
            return self._run_batch(frames)  # type: ignore[attr-defined]

        pending: Deque[Future[list[Any]]] = Deque()
        try:
            batch: list[Frame] = []
            max_pending = self._pipeline_workers  # type: ignore[attr-defined]
            with ThreadPoolExecutor(max_workers=self._pipeline_workers) as pool:  # type: ignore[attr-defined]
                while not _stop.is_set():
                    raw = reader.get(timeout=5.0)
                    frame: Frame | None = raw  # type: ignore[assignment]
                    if frame is not None:
                        batch.append(frame)
                    if len(batch) >= self._batch_size or (frame is None and batch):  # type: ignore[attr-defined]
                        if len(pending) >= max_pending:
                            yield from pending.popleft().result()
                        pending.append(pool.submit(_infer_batch, batch))
                        batch = []
                    if frame is None and reader.is_exhausted:
                        break
                while pending:
                    yield from pending.popleft().result()
        finally:
            for fut in pending:
                fut.cancel()
            self._active_streams.discard(_stop)  # type: ignore[attr-defined]
            if not self._active_streams:  # type: ignore[attr-defined]
                self._streams_drained.set()  # type: ignore[attr-defined]
            reader.stop()
            source.close()

    def _stream_sync(self, source: FrameSource) -> Iterator[Any]:
        """Legacy sequential streaming path (prefetch=False)."""
        _stop = threading.Event()
        with self._shutdown_lock:  # type: ignore[attr-defined]
            if self._shutting_down.is_set():  # type: ignore[attr-defined]
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)  # type: ignore[attr-defined]
            self._streams_drained.clear()  # type: ignore[attr-defined]
        batch: list[Frame] = []
        try:
            for frame in source:
                if _stop.is_set():
                    break
                batch.append(frame)
                if len(batch) >= self._batch_size:  # type: ignore[attr-defined]
                    yield from self._dispatch_frames(batch)  # type: ignore[attr-defined]
                    batch.clear()
            if batch and not _stop.is_set():
                yield from self._dispatch_frames(batch)  # type: ignore[attr-defined]
        finally:
            self._active_streams.discard(_stop)  # type: ignore[attr-defined]
            if not self._active_streams:  # type: ignore[attr-defined]
                self._streams_drained.set()  # type: ignore[attr-defined]
            source.close()


__all__: list[str] = []
