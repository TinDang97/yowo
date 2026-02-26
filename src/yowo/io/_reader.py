"""Threaded frame reader for decoupling I/O from inference.

ThreadedFrameReader wraps any FrameSource and provides a bounded queue with
configurable frame drop policies. All shared state is protected by explicit
threading primitives — correct on both GIL and free-threaded Python 3.14t.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yowo.types import Frame, FrameDropPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from yowo.io._source import FrameSource
    from yowo.types import PreprocessedTensor

logger = logging.getLogger(__name__)

__all__ = ["PreparedItem", "ThreadedFrameReader"]


@dataclass(slots=True)
class PreparedItem:
    """A frame that has been preprocessed in the reader thread.

    Bundles the ``PreprocessedTensor`` (ready for ``backend.infer()``) with the
    original ``Frame`` (needed by ``postprocess()`` for ``Detection.frame``).
    """

    tensor: PreprocessedTensor
    frame: Frame


class ThreadedFrameReader:
    """Thread-safe frame reader with bounded queue and drop policies.

    Spawns a background daemon thread that reads frames from the source into
    a bounded deque. The inference thread consumes frames via ``get()``.

    Drop policy behaviour:
    - ``NONE``: backpressure — reader blocks when queue is full.
    - ``LATEST``: evict all queued frames, keep only the newest one (best for
      live sources where stale frames are worthless).
    - ``SKIP_OLDEST``: evict the oldest frame when the queue is at capacity.

    All shared state is protected by ``threading.Lock`` + ``threading.Condition``.
    Safe on both GIL and free-threaded Python 3.14t (PEP 703).

    When ``preprocess_fn`` and ``target_size`` are both provided, the reader
    thread applies preprocessing (resize + blobFromImages) **before** enqueueing,
    overlapping CPU work with inference on the main thread.

    Example::

        with ThreadedFrameReader(source, max_queue_size=2, policy=FrameDropPolicy.LATEST) as reader:
            reader.start()
            while (frame := reader.get(timeout=1.0)) is not None:
                process(frame)
    """

    __slots__ = (
        "_deque",
        "_error",
        "_exhausted",
        "_frames_dropped",
        "_frames_read",
        "_lock",
        "_max_size",
        "_not_empty",
        "_not_full",
        "_policy",
        "_preprocess_fn",
        "_source",
        "_stop_event",
        "_target_size",
        "_thread",
    )

    def __init__(
        self,
        source: FrameSource,
        max_queue_size: int = 2,
        policy: FrameDropPolicy = FrameDropPolicy.NONE,
        *,
        preprocess_fn: Callable[[list[Frame], tuple[int, int]], PreprocessedTensor] | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")
        self._source = source
        self._max_size = max_queue_size
        self._policy = policy
        self._preprocess_fn = preprocess_fn
        self._target_size = target_size
        self._deque: deque[Frame | PreparedItem] = deque()
        self._lock = threading.Lock()
        self._not_empty: threading.Condition = threading.Condition(self._lock)
        self._not_full: threading.Condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._error: BaseException | None = None
        self._exhausted = False
        self._frames_read = 0
        self._frames_dropped = 0
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Background reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread body: read source frames into bounded deque."""
        try:
            for frame in self._source:
                if self._stop_event.is_set():
                    break

                # Optional preprocessing: runs OUTSIDE the lock to avoid
                # blocking the consumer while CPU-bound resize executes.
                item: Frame | PreparedItem
                if self._preprocess_fn is not None and self._target_size is not None:
                    tensor = self._preprocess_fn([frame], self._target_size)
                    item = PreparedItem(tensor=tensor, frame=frame)
                else:
                    item = frame

                with self._not_empty:
                    self._frames_read += 1
                    if self._policy == FrameDropPolicy.LATEST:
                        # Drop everything queued; keep only this latest frame.
                        dropped = len(self._deque)
                        self._deque.clear()
                        self._frames_dropped += dropped
                        self._deque.append(item)
                        self._not_empty.notify()
                    elif self._policy == FrameDropPolicy.SKIP_OLDEST:
                        if len(self._deque) >= self._max_size:
                            self._deque.popleft()
                            self._frames_dropped += 1
                        self._deque.append(item)
                        self._not_empty.notify()
                    else:  # FrameDropPolicy.NONE — backpressure
                        while len(self._deque) >= self._max_size and not self._stop_event.is_set():
                            self._not_full.wait(timeout=0.05)
                        if not self._stop_event.is_set():
                            self._deque.append(item)
                            self._not_empty.notify()
        except Exception as exc:
            with self._not_empty:
                self._error = exc
                logger.error("ThreadedFrameReader error: %s", exc)
        finally:
            with self._not_empty:
                self._exhausted = True
                self._not_empty.notify_all()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background reader thread.

        Must be called before the first ``get()``.
        """
        if self._thread is not None:
            return  # idempotent
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="yowo-frame-reader",
            daemon=True,
        )
        self._thread.start()

    def get(self, timeout: float = 1.0) -> Frame | PreparedItem | None:
        """Get the next item from the queue.

        Blocks until an item is available, the source is exhausted, or
        ``timeout`` seconds pass. Returns ``None`` only when the source is
        fully consumed (``is_exhausted`` is True and the queue is empty).
        Returns ``None`` on timeout as well --- callers that need to distinguish
        timeout from exhaustion should check ``is_exhausted`` after receiving
        ``None``.

        When the reader was created with ``preprocess_fn`` and ``target_size``,
        items are ``PreparedItem`` instances; otherwise plain ``Frame`` objects.

        Args:
            timeout: Maximum seconds to wait for the next item.

        Returns:
            A ``Frame``, a ``PreparedItem``, or ``None`` when exhausted / timed out.

        Raises:
            Exception: Any exception raised by the underlying ``FrameSource``
                or by ``preprocess_fn``.
        """
        with self._not_empty:
            deadline = time.monotonic() + timeout
            while not self._deque and not self._exhausted and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._not_empty.wait(timeout=min(remaining, 0.1))
            if self._error is not None:
                raise self._error
            if self._deque:
                item = self._deque.popleft()
                self._not_full.notify()
                return item
            return None  # exhausted or timed out

    def stop(self) -> None:
        """Signal the reader to stop and wait for the thread to join."""
        self._stop_event.set()
        with self._not_empty:
            # Count remaining queued frames as dropped on shutdown.
            self._frames_dropped += len(self._deque)
            self._deque.clear()
            self._not_empty.notify_all()
            self._not_full.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> ThreadedFrameReader:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def is_exhausted(self) -> bool:
        """True when the source has been fully consumed or errored."""
        with self._lock:
            return self._exhausted

    @property
    def frames_read(self) -> int:
        """Total frames read from the source (thread-safe snapshot)."""
        with self._lock:
            return self._frames_read

    @property
    def frames_dropped(self) -> int:
        """Total frames dropped due to queue policy (thread-safe snapshot)."""
        with self._lock:
            return self._frames_dropped

    @property
    def drop_rate(self) -> float:
        """Fraction of frames dropped (0.0-1.0)."""
        with self._lock:
            total = self._frames_read
            return self._frames_dropped / total if total > 0 else 0.0
