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
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from yowo.io._reader import ThreadedFrameReader
from yowo.pipeline._ids import safe_stream_id
from yowo.types import Frame, FrameDropPolicy, StreamConfig, StreamState, TaggedFrame

if TYPE_CHECKING:
    from yowo.io._source import FrameSource

logger = logging.getLogger(__name__)

__all__ = ["FrameCollector"]


class _StreamEntry:
    """Internal bookkeeping for a single managed stream."""

    __slots__ = (
        "auto_remove",
        "bridge",
        "consecutive_errors",
        "error",
        "frames_dropped",
        "frames_processed",
        "last_frame_time",
        "reader",
        "source",
        "state",
        "stop_event",
    )

    def __init__(self, reader: ThreadedFrameReader, source: FrameSource) -> None:
        self.reader: ThreadedFrameReader = reader
        self.source: FrameSource = source
        self.state: StreamState = StreamState.RUNNING
        self.error: BaseException | None = None
        self.bridge: threading.Thread | None = None
        self.stop_event: threading.Event = threading.Event()
        # Per-stream stats (added in v2.3.0 for STRM-01 to STRM-05)
        self.auto_remove: bool = False
        self.consecutive_errors: int = 0
        self.frames_dropped: int = 0
        self.frames_processed: int = 0
        self.last_frame_time: float = 0.0


def _put_or_stop(
    shared_q: queue.Queue[tuple[str, Frame | None] | None],
    item: tuple[str, Frame | None],
    stop: threading.Event,
) -> bool:
    """Put *item* on the bounded queue, retrying until success or *stop* is set.

    Returns ``True`` if the item was enqueued, ``False`` if *stop* fired first.
    """
    while not stop.is_set():
        try:
            shared_q.put(item, timeout=0.5)
            return True
        except queue.Full:
            pass
    return False


def _run_bridge(
    stream_id: str,
    entry: _StreamEntry,
    shared_q: queue.Queue[tuple[str, Frame | None] | None],
    max_consecutive_errors: int = 3,
) -> None:
    """Bridge daemon: read from one ThreadedFrameReader, push to shared_q.

    Posts ``(stream_id, None)`` as exhaustion sentinel before exiting.

    Tracks per-stream stats on *entry*:
    - ``frames_processed``: increments on each successful Frame enqueue.
    - ``frames_dropped``: increments when ``_put_or_stop`` returns False.
    - ``consecutive_errors``: increments on each read error, resets to 0
      on any successful item (Frame or None gap).
    - ``last_frame_time``: updated to ``time.monotonic()`` on each Frame.
    - ``auto_remove``: set to True when consecutive_errors reaches
      *max_consecutive_errors*. The bridge then exits WITHOUT calling
      remove_stream() directly — the iterator thread handles removal to
      avoid the self-join deadlock (RESEARCH.md pitfall 2).
    """
    try:
        while not entry.stop_event.is_set():
            try:
                item = entry.reader.get(timeout=0.1)
            except Exception as exc:
                entry.consecutive_errors += 1
                if entry.consecutive_errors >= max_consecutive_errors:
                    logger.warning(
                        "Stream %r auto-removing after %d consecutive errors",
                        stream_id,
                        entry.consecutive_errors,
                    )
                    entry.state = StreamState.ERROR
                    entry.error = exc
                    entry.auto_remove = True
                    break
                logger.debug(
                    "Stream %r bridge error (%d/%d): %s",
                    stream_id,
                    entry.consecutive_errors,
                    max_consecutive_errors,
                    exc,
                )
                entry.state = StreamState.ERROR
                entry.error = exc
                continue

            if isinstance(item, Frame):
                entry.consecutive_errors = 0
                entry.last_frame_time = time.monotonic()
                if _put_or_stop(shared_q, (stream_id, item), entry.stop_event):
                    entry.frames_processed += 1
                else:
                    entry.frames_dropped += 1
                    break
            elif item is None:
                entry.consecutive_errors = 0
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
        # Always deliver the exhaustion sentinel so __iter__ can decrement
        # active-stream count. Do NOT use _put_or_stop here — stop_event is
        # already set on normal shutdown, which would cause an immediate
        # False return and drop the sentinel.
        try:
            shared_q.put((stream_id, None), timeout=5.0)
        except queue.Full:
            logger.warning("Bridge %s: sentinel dropped (queue full after 5 s)", stream_id)


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

    __slots__ = (
        "_auto_removed_errors",
        "_closed",
        "_lock",
        "_max_queue_size",
        "_shared_q",
        "_streams",
    )

    def __init__(self, max_queue_size: int = 2) -> None:
        if max_queue_size < 1:
            raise ValueError(f"max_queue_size must be >= 1, got {max_queue_size}")
        self._max_queue_size: int = max_queue_size
        self._streams: dict[str, _StreamEntry] = {}
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._shared_q: queue.Queue[tuple[str, Frame | None] | None] = queue.Queue(
            maxsize=max_queue_size * 8,
        )
        # Tracks (stream_id -> exception) for streams auto-removed due to errors.
        # These are no longer in _streams but must be visible via stream_errors.
        self._auto_removed_errors: dict[str, BaseException] = {}

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def add_stream(
        self,
        stream_id: str,
        source: FrameSource,
        *,
        policy: FrameDropPolicy | None = None,
        stream_config: StreamConfig | None = None,
    ) -> None:
        """Register and start a new stream.

        If ``policy`` is ``None``, the collector auto-selects
        ``FrameDropPolicy.LATEST`` for live sources and
        ``FrameDropPolicy.NONE`` for offline sources.

        Args:
            stream_id: Unique identifier for this stream. Any credential embedded in
                it (an RTSP URL's ``user:password@``) is stripped before anything else
                sees it, so logs, errors, dict keys and callbacks never carry it. The
                stripped form becomes the canonical id -- pass the same value to
                :meth:`DetectionRouter.register`, which normalises it identically.
            source: A FrameSource instance to read from.
            policy: Override frame drop policy. Auto-selected when None.
            stream_config: Per-stream failure handling configuration. Uses
                ``StreamConfig()`` defaults when None.

        Raises:
            RuntimeError: If the collector is closed.
            ValueError: If ``stream_id`` is already registered.
        """
        # Boundary: nothing else in this package ever sees the raw identifier, so
        # every sink below -- the log lines, the ValueError, the bridge thread name,
        # the dict keys and the routing callbacks -- is safe by construction.
        stream_id = safe_stream_id(stream_id)

        if policy is None:
            policy = FrameDropPolicy.LATEST if source.is_live else FrameDropPolicy.NONE

        cfg = stream_config if stream_config is not None else StreamConfig()

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
                args=(stream_id, entry, self._shared_q, cfg.max_consecutive_errors),
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
        stream_id = safe_stream_id(stream_id)

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
                    # If a bridge thread has exited without delivering its
                    # exhaustion sentinel (dropped due to a full queue), the
                    # stream would never leave `active`.  Detect dead bridges
                    # and retire them so __iter__ terminates naturally.
                    dead = {
                        sid
                        for sid in active
                        if sid in self._streams
                        and self._streams[sid].bridge is not None
                        and not self._streams[sid].bridge.is_alive()  # type: ignore[union-attr]
                    }
                active -= dead
                continue

            if item is None:
                break  # close() sentinel

            stream_id, frame = item
            if frame is None:
                # Exhaustion sentinel from bridge thread.
                # Check if this was an auto-remove triggered by consecutive errors.
                with self._lock:
                    entry = self._streams.get(stream_id)
                    should_auto_remove = entry is not None and entry.auto_remove
                if should_auto_remove:
                    # Record the error before removing the stream so stream_errors
                    # remains visible to _check_stream_errors after removal.
                    with self._lock:
                        exc = entry.error if entry is not None else None
                        if exc is not None:
                            self._auto_removed_errors[stream_id] = exc
                    # Safe: iterator thread is distinct from all bridge threads.
                    self.remove_stream(stream_id)
                    logger.warning(
                        "Stream %r auto-removed after consecutive errors",
                        stream_id,
                    )
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

        Includes errors from streams that were auto-removed due to consecutive
        failures — those streams are no longer in ``_streams`` but their errors
        are tracked in ``_auto_removed_errors``.

        Returns:
            Mapping from stream_id to the exception that caused the error.
        """
        with self._lock:
            result = {sid: e.error for sid, e in self._streams.items() if e.error is not None}
            result.update(self._auto_removed_errors)
        return result

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
        try:
            self._shared_q.put(None, timeout=2.0)
        except queue.Full:
            logger.warning("FrameCollector.close: close sentinel dropped (queue full)")

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
    """Stop a reader, join bridge thread, and close source."""
    entry.stop_event.set()
    try:
        entry.reader.stop()
    except Exception:
        logger.exception("Error stopping reader")
    if entry.bridge is not None:
        entry.bridge.join(timeout=5.0)
        if entry.bridge.is_alive():
            logger.warning(
                "Bridge thread '%s' did not exit within 5 s after stop; "
                "it may still be running and holding a reference to the shared queue.",
                entry.bridge.name,
            )
    try:
        entry.source.close()
    except Exception:
        logger.exception("Error closing source")
