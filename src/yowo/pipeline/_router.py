"""Detection routing for multi-stream batched inference.

Routes :class:`~yowo.types.Detection` results back to per-stream
callbacks after batched inference completes.  Uses positional index
correlation: ``detections[i]`` corresponds to ``batch[i]`` (guaranteed
by ``engine.detect()`` which returns one Detection per input frame, in
order).

Callbacks are invoked on a snapshot taken before dispatch begins, so
a callback that calls ``register``/``unregister`` will not affect the
current dispatch round.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING

from yowo.pipeline._ids import safe_stream_id
from yowo.types import Detection, TaggedFrame

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["DetectionRouter"]


class DetectionRouter:
    """Thread-safe router that dispatches detections to per-stream callbacks.

    Typical usage::

        router = DetectionRouter()
        router.register("cam-1", on_cam1_detections)
        router.register("cam-2", on_cam2_detections)

        # After batched inference:
        router.route(detections, batch)

    Detections whose stream has no registered callback are silently
    dropped (logged at DEBUG level).
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[[str, list[Detection]], None]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        stream_id: str,
        callback: Callable[[str, list[Detection]], None],
    ) -> None:
        """Register *callback* for *stream_id*.

        Overwrites any previously registered callback for the same stream.

        Args:
            stream_id: Unique identifier for the source stream. Normalised the same
                way as :meth:`FrameCollector.add_stream`: any embedded credential is
                stripped, so the value the callback receives is safe to emit.
            callback: Invoked as ``callback(stream_id, detections)`` when
                detections are available for this stream.
        """
        # Boundary: the collector redacts the same identifier on the way in, so both
        # sides must normalise or a credentialed id would never match its callback.
        stream_id = safe_stream_id(stream_id)

        with self._lock:
            self._callbacks[stream_id] = callback
            logger.debug("Registered callback for stream '%s'", stream_id)

    def unregister(self, stream_id: str) -> None:
        """Remove the callback for *stream_id*.

        No-op if the stream was never registered.

        Args:
            stream_id: Stream whose callback should be removed.
        """
        stream_id = safe_stream_id(stream_id)

        with self._lock:
            removed = self._callbacks.pop(stream_id, None)
        if removed is not None:
            logger.debug("Unregistered callback for stream '%s'", stream_id)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        detections: list[Detection],
        batch: list[TaggedFrame],
    ) -> None:
        """Match detections to streams by positional index and dispatch.

        ``engine.detect(frames)`` returns one ``Detection`` per input frame
        in the same order as the input list.  This method uses that 1:1
        positional correspondence: ``detections[i]`` maps to ``batch[i]``.

        Args:
            detections: Results from batched inference (same length as *batch*).
            batch: The :class:`TaggedFrame` entries that produced *detections*.
        """
        # Group detections by stream_id using positional correspondence.
        grouped: dict[str, list[Detection]] = defaultdict(list)
        for det, tagged in zip(detections, batch):
            grouped[tagged.stream_id].append(det)

        # Snapshot callbacks under lock, dispatch outside.
        with self._lock:
            callbacks_snapshot = dict(self._callbacks)

        for stream_id, stream_detections in grouped.items():
            cb = callbacks_snapshot.get(stream_id)
            if cb is None:
                logger.debug(
                    "No callback registered for stream '%s'; dropping %d detection(s)",
                    stream_id,
                    len(stream_detections),
                )
                continue
            cb(stream_id, stream_detections)
