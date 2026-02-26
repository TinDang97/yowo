"""Multi-stream frame collector with round-robin dispatch.

FrameCollector manages N concurrent video/RTSP streams, wrapping each in
a ThreadedFrameReader and yielding TaggedFrame objects (frame + stream_id)
via round-robin iteration. Streams can be added or removed while iteration
is active; all shared state is protected by an explicit threading.Lock.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from yowo.io._reader import ThreadedFrameReader
from yowo.types import Frame, FrameDropPolicy, StreamState, TaggedFrame

if TYPE_CHECKING:
    from yowo.io._source import FrameSource

logger = logging.getLogger(__name__)

__all__ = ["FrameCollector"]

# Short poll timeout so round-robin doesn't stall on one empty stream.
_POLL_TIMEOUT: float = 0.05


class _StreamEntry:
    """Internal bookkeeping for a single managed stream."""

    __slots__ = ("error", "reader", "source", "state")

    def __init__(self, reader: ThreadedFrameReader, source: FrameSource) -> None:
        self.reader: ThreadedFrameReader = reader
        self.source: FrameSource = source
        self.state: StreamState = StreamState.RUNNING
        self.error: BaseException | None = None


class FrameCollector:
    """Manages N concurrent streams, yielding TaggedFrame via round-robin.

    Each stream is backed by a ThreadedFrameReader. The collector provides
    thread-safe ``add_stream`` / ``remove_stream`` for dynamic membership
    and tracks per-stream health via ``StreamState``.

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

    __slots__ = ("_closed", "_lock", "_max_queue_size", "_streams")

    def __init__(self, max_queue_size: int = 2) -> None:
        if max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")
        self._max_queue_size: int = max_queue_size
        self._streams: dict[str, _StreamEntry] = {}
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

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
            self._streams[stream_id] = _StreamEntry(reader=reader, source=source)
            reader.start()

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
        """Round-robin across active streams, yielding TaggedFrame objects.

        Terminates when all streams reach a terminal state (STOPPED or ERROR)
        and no frames remain in any queue. Also terminates immediately if
        ``close()`` is called.

        Yields:
            TaggedFrame for every frame from every active stream.
        """
        while True:
            snapshot = self._active_snapshot()
            if not snapshot:
                break

            yielded_any = False
            for stream_id, entry in snapshot:
                frame = self._try_get(stream_id, entry)
                if frame is not None:
                    yielded_any = True
                    yield frame

            if not yielded_any and self._all_terminal():
                break

    def _active_snapshot(self) -> list[tuple[str, _StreamEntry]]:
        """Return a snapshot of (stream_id, entry) pairs under the lock.

        Used by ``__iter__`` to iterate without holding the lock.
        """
        with self._lock:
            if self._closed:
                return []
            return list(self._streams.items())

    def _try_get(self, stream_id: str, entry: _StreamEntry) -> TaggedFrame | None:
        """Attempt a non-blocking read from one stream.

        Handles exhaustion (marks STOPPED) and exceptions (marks ERROR).

        Returns:
            A TaggedFrame on success, None otherwise.
        """
        if entry.state in (StreamState.STOPPED, StreamState.ERROR):
            return None

        try:
            item = entry.reader.get(timeout=_POLL_TIMEOUT)
        except Exception as exc:
            logger.exception("Stream %r error during get()", stream_id)
            entry.state = StreamState.ERROR
            entry.error = exc
            return None

        if isinstance(item, Frame):
            return TaggedFrame(stream_id=stream_id, frame=item)

        # No frame returned — check if exhausted.
        if entry.reader.is_exhausted:
            entry.state = StreamState.STOPPED
            logger.debug(
                "Stream %r exhausted (%d frames read)", stream_id, entry.reader.frames_read
            )
        return None

    def _all_terminal(self) -> bool:
        """True when every registered stream is STOPPED or ERROR."""
        with self._lock:
            if not self._streams:
                return True
            return all(
                e.state in (StreamState.STOPPED, StreamState.ERROR) for e in self._streams.values()
            )

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
    try:
        entry.reader.stop()
    except Exception:
        logger.exception("Error stopping reader")
    try:
        entry.source.close()
    except Exception:
        logger.exception("Error closing source")
