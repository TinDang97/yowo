"""Multi-stream frame collector with shared-queue dispatch.

FrameCollector manages N concurrent video/RTSP streams, wrapping each in
a ThreadedFrameReader. Per-stream bridge threads drain readers into a single
shared queue, giving O(1) dispatch in ``__iter__`` regardless of stream count.
Streams can be added or removed while iteration is active; all shared state
is protected by an explicit threading.Lock.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from yowo.io._reader import ThreadedFrameReader
from yowo.types import Frame, FrameDropPolicy, StreamState, TaggedFrame

if TYPE_CHECKING:
    from yowo.io._source import FrameSource

logger = logging.getLogger(__name__)

__all__ = ["FrameCollector"]

# Sentinel posted to the shared queue when the collector is closed.
_CLOSED: None = None


class _StreamEntry:
    """Internal bookkeeping for a single managed stream."""

    __slots__ = ("bridge", "error", "reader", "source", "state", "stop_flag")

    def __init__(self, reader: ThreadedFrameReader, source: FrameSource) -> None:
        self.reader: ThreadedFrameReader = reader
        self.source: FrameSource = source
        self.state: StreamState = StreamState.RUNNING
        self.error: BaseException | None = None
        self.bridge: threading.Thread | None = None
        self.stop_flag: bool = False


def _run_bridge(
    stream_id: str,
    entry: _StreamEntry,
    shared_q: queue.Queue[tuple[str, Frame | None] | None],
) -> None:
    """Bridge daemon: read from one ThreadedFrameReader, push to shared_q.

    Posts ``(stream_id, None)`` as exhaustion sentinel before exiting.
    """
    try:
        while not entry.stop_flag:
            try:
                item = entry.reader.get(timeout=0.1)
            except Exception as exc:
                logger.exception("Stream %r bridge error", stream_id)
                entry.state = StreamState.ERROR
                entry.error = exc
                break

            if isinstance(item, Frame):
                shared_q.put((stream_id, item))
            elif item is None:
                if entry.reader.is_exhausted:
                    entry.state = StreamState.STOPPED
                    logger.debug(
                        "Stream %r exhausted (%d frames read)",
                        stream_id,
                        entry.reader.frames_read,
                    )
                    break
                # Live stream momentary gap — keep polling.
            else:
                entry.state = StreamState.ERROR
                entry.error = TypeError(
                    f"FrameCollector expects Frame from reader, got {type(item).__name__}. "
                    "Do not pass preprocess_fn to readers used by FrameCollector."
                )
                break
    finally:
        # Always post exhaustion sentinel so __iter__ tracks active count.
        shared_q.put((stream_id, None))


class FrameCollector:
    """Manages N concurrent streams, yielding TaggedFrame via shared queue.

    Each stream is backed by a ThreadedFrameReader drained by a bridge
    daemon thread. All bridges push to a single shared queue, so
    ``__iter__`` is O(1) per frame regardless of stream count.

    Usage::

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", cam_source)
        collector.add_stream("rtsp-1", rtsp_source)

        for tagged in collector:
            print(tagged.stream_id, tagged.frame.shape_hw)

        collector.close()

    Or as a context manager::

        with FrameCollector() as collector:
            collector.add_stream("cam-0", cam_source)
            for tagged in collector:
                process(tagged)

    Args:
        max_queue_size: Per-stream bounded queue depth. Defaults to 2.
    """

    __slots__ = ("_closed", "_lock", "_max_queue_size", "_shared_q", "_streams")

    def __init__(self, max_queue_size: int = 2) -> None:
        if max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")
        self._max_queue_size: int = max_queue_size
        self._streams: dict[str, _StreamEntry] = {}
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._shared_q: queue.Queue[tuple[str, Frame | None] | None] = queue.Queue()

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def add_stream(
        self,
        stream_id: str,
        source: FrameSource,
        *,
        policy: FrameDropPolicy | None = None,
    ) -> None:
        """Register and start a new stream.

        If ``policy`` is ``None``, the collector auto-selects
        ``FrameDropPolicy.LATEST`` for live sources and
        ``FrameDropPolicy.NONE`` for offline sources.

        Args:
            stream_id: Unique identifier for this stream.
            source: A FrameSource instance to read from.
            policy: Override frame drop policy. Auto-selected when None.

        Raises:
            RuntimeError: If the collector is closed.
            ValueError: If ``stream_id`` is already registered.
        """
        if policy is None:
            policy = FrameDropPolicy.LATEST if source.is_live else FrameDropPolicy.NONE

        reader = ThreadedFrameReader(
            source=source,
            max_queue_size=self._max_queue_size,
            policy=policy,
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("FrameCollector is closed")
            if stream_id in self._streams:
                raise ValueError(f"Stream already registered: {stream_id!r}")
            entry = _StreamEntry(reader=reader, source=source)
            self._streams[stream_id] = entry
            reader.start()
            bridge = threading.Thread(
                target=_run_bridge,
                args=(stream_id, entry, self._shared_q),
                name=f"yowo-bridge-{stream_id}",
                daemon=True,
            )
            entry.bridge = bridge
            bridge.start()

        logger.info("Added stream %r (policy=%s)", stream_id, policy.value)

    def remove_stream(self, stream_id: str) -> None:
        """Stop and remove a stream by its identifier.

        Idempotent: silently returns if ``stream_id`` is not registered.

        Args:
            stream_id: The identifier of the stream to remove.
        """
        with self._lock:
            entry = self._streams.pop(stream_id, None)

        if entry is None:
            return

        _stop_entry(entry)
        logger.info("Removed stream %r", stream_id)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TaggedFrame]:
        """O(1) per-frame dispatch from shared queue across all streams.

        Terminates when all active streams post their exhaustion sentinel,
        or when ``close()`` posts the closed sentinel.

        Yields:
            TaggedFrame for every frame from every active stream.
        """
        with self._lock:
            if self._closed:
                return
            active = set(self._streams.keys())

        if not active:
            return

        while active:
            try:
                item = self._shared_q.get(timeout=5.0)
            except queue.Empty:
                # Timeout — check if we should still be waiting.
                with self._lock:
                    if self._closed:
                        break
                continue

            if item is None:
                break  # close() sentinel

            stream_id, frame = item
            if frame is None:
                # Exhaustion sentinel from bridge thread
                active.discard(stream_id)
                continue
            yield TaggedFrame(stream_id=stream_id, frame=frame)

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def stream_states(self) -> dict[str, StreamState]:
        """Snapshot of per-stream health states.

        Returns:
            Mapping from stream_id to its current StreamState.
        """
        with self._lock:
            return {sid: e.state for sid, e in self._streams.items()}

    @property
    def stream_errors(self) -> dict[str, BaseException]:
        """Snapshot of per-stream errors for streams in ERROR state.

        Returns:
            Mapping from stream_id to the exception that caused the error.
        """
        with self._lock:
            return {sid: e.error for sid, e in self._streams.items() if e.error is not None}

    @property
    def stream_count(self) -> int:
        """Number of currently registered streams (any state)."""
        with self._lock:
            return len(self._streams)

    @property
    def active_count(self) -> int:
        """Number of streams in RUNNING or RECONNECTING state."""
        with self._lock:
            return sum(
                1
                for e in self._streams.values()
                if e.state in (StreamState.RUNNING, StreamState.RECONNECTING)
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop all readers and close all sources. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._streams.values())

        for entry in entries:
            _stop_entry(entry)

        # Unblock __iter__ if it's waiting on shared_q.get()
        self._shared_q.put(None)

        logger.info("FrameCollector closed (%d streams)", len(entries))

    def __enter__(self) -> FrameCollector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _stop_entry(entry: _StreamEntry) -> None:
    """Stop a reader and close its source, suppressing cleanup errors."""
    entry.stop_flag = True
    try:
        entry.reader.stop()
    except Exception:
        logger.exception("Error stopping reader")
    try:
        entry.source.close()
    except Exception:
        logger.exception("Error closing source")
