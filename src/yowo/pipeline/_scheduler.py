"""Batch scheduler for the multi-stream inference pipeline.

Wraps an upstream ``Iterable[TaggedFrame]`` and yields fixed-size batches,
flushing either when the batch reaches ``max_batch_size`` or when
``timeout_ms`` milliseconds have elapsed since the first frame in the
current batch -- whichever comes first.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yowo.types import TaggedFrame

__all__ = ["BatchScheduler"]


class BatchScheduler:
    """Synchronous iterator that accumulates tagged frames into batches.

    The scheduler pulls frames one at a time from the upstream iterator
    and groups them into ``list[TaggedFrame]`` batches. A batch is yielded
    (flushed) when either condition is met:

    1. The batch contains ``max_batch_size`` frames (capacity flush).
    2. ``timeout_ms`` milliseconds have passed since the first frame was
       added to the current batch (timeout flush).

    When the upstream is exhausted, any partial (non-empty) batch
    is flushed before the scheduler itself raises ``StopIteration``.

    Pass ``stop_event`` to allow external cancellation (checked between
    each upstream pull).

    Example::

        collector = FrameCollector(...)
        scheduler = BatchScheduler(collector, max_batch_size=4, timeout_ms=50.0)
        for batch in scheduler:
            results = engine.detect([t.frame for t in batch])

    Args:
        upstream: Iterable producing ``TaggedFrame`` objects.
        max_batch_size: Maximum frames per batch (>= 1).
        timeout_ms: Maximum milliseconds to wait before flushing a
            partial batch (> 0).
        stop_event: Optional threading.Event — when set, the scheduler
            flushes any partial batch and stops.
    """

    __slots__ = (
        "_batch",
        "_batch_start_time",
        "_iter",
        "_max_batch_size",
        "_stop_event",
        "_timeout_s",
        "_upstream",
        "_upstream_exhausted",
    )

    def __init__(
        self,
        upstream: Iterable[TaggedFrame],
        *,
        max_batch_size: int = 1,
        timeout_ms: float = 100.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be > 0, got {timeout_ms}")

        self._upstream: Iterable[TaggedFrame] = upstream
        self._iter: Iterator[TaggedFrame] | None = None
        self._max_batch_size = max_batch_size
        self._timeout_s = timeout_ms / 1000.0
        self._stop_event = stop_event
        self._batch: list[TaggedFrame] = []
        self._batch_start_time: float = 0.0
        self._upstream_exhausted = False

    def __iter__(self) -> BatchScheduler:
        # Derive a fresh iterator each time __iter__ is called, allowing reuse.
        self._iter = iter(self._upstream)
        self._upstream_exhausted = False
        self._batch = []
        self._batch_start_time = 0.0
        return self

    def __next__(self) -> list[TaggedFrame]:
        """Return the next batch of tagged frames.

        Raises:
            StopIteration: When the upstream is exhausted and no frames
                remain to flush, or when stop_event is set and the batch
                is empty.
        """
        if self._iter is None:
            self._iter = iter(self._upstream)

        # Fast path: upstream already exhausted and nothing buffered.
        if self._upstream_exhausted:
            raise StopIteration

        while True:
            # Check stop event.
            if self._stop_event is not None and self._stop_event.is_set():
                self._upstream_exhausted = True
                if self._batch:
                    return self._flush()
                raise StopIteration

            # Try to pull one frame from upstream.
            try:
                frame = next(self._iter)
            except StopIteration:
                self._upstream_exhausted = True
                if self._batch:
                    return self._flush()
                raise

            # First frame in a new batch -- record start time.
            if not self._batch:
                self._batch_start_time = time.monotonic()

            self._batch.append(frame)

            # Capacity flush.
            if len(self._batch) >= self._max_batch_size:
                return self._flush()

            # Timeout flush.
            elapsed = time.monotonic() - self._batch_start_time
            if elapsed >= self._timeout_s:
                return self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> list[TaggedFrame]:
        """Return the current batch and reset internal state."""
        batch = self._batch
        self._batch = []
        self._batch_start_time = 0.0
        return batch

    def set_stop_event(self, event: threading.Event) -> None:
        """Attach an external stop signal.

        When *event* is set, the scheduler flushes any partial batch and
        raises ``StopIteration`` on the next ``__next__`` call.
        """
        self._stop_event = event

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def max_batch_size(self) -> int:
        """Configured maximum frames per batch."""
        return self._max_batch_size

    @property
    def timeout_ms(self) -> float:
        """Configured flush timeout in milliseconds."""
        return self._timeout_s * 1000.0

    @property
    def pending(self) -> int:
        """Number of frames accumulated in the current (unflushed) batch."""
        return len(self._batch)
