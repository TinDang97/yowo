"""Thread-safe object counter with zone and line-cross counting."""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

from yowo.counter._geometry import box_center, cross_sign, point_in_polygon
from yowo.counter._types import (
    CountLine,
    CountResult,
    CountZone,
    CrossDirection,
    LineCrossEvent,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasTrackId(Protocol):
    """Structural protocol satisfied by any object with a track_id int attribute."""

    track_id: int


class ObjectCounter:
    """Thread-safe accumulator for detection counting.

    Supports three composable counting modes:
    1. **Per-class counts** — live (current frame) and cumulative (since reset).
    2. **Zone-based counting** — count detections whose box center falls inside
       a polygonal ROI (CountZone).
    3. **Line-cross counting** — detect when tracked objects cross a directed
       line segment (CountLine). Requires TrackedDetection input with track_id.

    Thread safety: all public methods acquire an internal Lock. Safe for
    concurrent calls from multi-stream pipeline callbacks (DetectionRouter).

    Example::

        counter = ObjectCounter(
            zones=[CountZone("entrance", vertices=((0,0),(100,0),(100,200),(0,200)))],
            lines=[CountLine("gate", p1=(50,0), p2=(50,200))],
        )
        for detection in engine.stream(source):
            result = counter.update(detection)
            print(result.cumulative_counts)
    """

    __slots__ = (
        "_cumulative",
        "_line_totals",
        "_lines",
        "_lock",
        "_prev_centers",
        "_zone_cumulative",
        "_zones",
    )

    def __init__(
        self,
        *,
        zones: list[CountZone] | None = None,
        lines: list[CountLine] | None = None,
    ) -> None:
        self._zones: tuple[CountZone, ...] = tuple(zones) if zones else ()
        self._lines: tuple[CountLine, ...] = tuple(lines) if lines else ()
        self._cumulative: dict[str, int] = {}
        self._zone_cumulative: dict[str, dict[str, int]] = {z.zone_id: {} for z in self._zones}
        self._line_totals: dict[str, dict[CrossDirection, int]] = {
            ln.line_id: {CrossDirection.IN: 0, CrossDirection.OUT: 0} for ln in self._lines
        }
        # track_id → center from previous frame (for line-cross detection)
        self._prev_centers: dict[int, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def update(self, detection: object) -> CountResult:
        """Process one detection frame and return an immutable counting snapshot.

        Accepts both Detection and TrackedDetection (or any object with
        .frame and .boxes attributes). Line-cross counting requires boxes
        that expose a ``track_id`` attribute (TrackedDetection); plain
        Detection boxes are silently skipped for line events.

        Args:
            detection: A Detection or TrackedDetection instance.

        Returns:
            CountResult snapshot with live and cumulative counts.
        """
        with self._lock:
            frame = detection.frame  # type: ignore[attr-defined]
            boxes = detection.boxes  # type: ignore[attr-defined]
            frame_index: int = getattr(frame, "frame_index", 0)
            timestamp_ms: float = getattr(frame, "timestamp_ms", 0.0)

            # --- 1. Live per-class counts + cumulative ---
            live: dict[str, int] = {}
            for box in boxes:
                name: str = box.class_name  # type: ignore[attr-defined]
                live[name] = live.get(name, 0) + 1
                self._cumulative[name] = self._cumulative.get(name, 0) + 1

            # --- 2. Zone-based counting ---
            zone_live: dict[str, dict[str, int]] = {}
            for zone in self._zones:
                zc: dict[str, int] = {}
                for box in boxes:
                    name = box.class_name  # type: ignore[attr-defined]
                    if zone.class_filter and name not in zone.class_filter:
                        continue
                    center = box_center(box)
                    if point_in_polygon(center, zone.vertices):
                        zc[name] = zc.get(name, 0) + 1
                        self._zone_cumulative[zone.zone_id][name] = (
                            self._zone_cumulative[zone.zone_id].get(name, 0) + 1
                        )
                zone_live[zone.zone_id] = zc

            # --- 3. Line-cross counting (tracked boxes only) ---
            events = self._detect_crossings(boxes, frame_index, timestamp_ms)

            # --- 4. Build immutable result snapshot ---
            return CountResult(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                live_counts=dict(live),
                cumulative_counts=dict(self._cumulative),
                zone_counts=zone_live,
                cumulative_zone_counts={zid: dict(zc) for zid, zc in self._zone_cumulative.items()},
                line_events=tuple(events),
                line_totals={lid: dict(lt) for lid, lt in self._line_totals.items()},
            )

    def _detect_crossings(
        self,
        boxes: tuple[object, ...],
        frame_index: int,
        timestamp_ms: float,
    ) -> list[LineCrossEvent]:
        """Detect line crossings for tracked boxes. Called under lock."""
        if not self._lines:
            return []

        # Check type once at frame level — all boxes in a detection share the same type.
        # Avoids O(N x L) runtime-checkable Protocol isinstance calls per frame.
        if not boxes or not isinstance(boxes[0], _HasTrackId):
            # Prune stale prev_centers when tracking is absent
            self._prev_centers.clear()
            return []

        events: list[LineCrossEvent] = []
        current_ids: set[int] = set()

        for box in boxes:
            tid: int = box.track_id  # type: ignore[attr-defined]
            current_ids.add(tid)
            curr = box_center(box)
            prev = self._prev_centers.get(tid)

            if prev is not None:
                for line in self._lines:
                    class_name: str = box.class_name  # type: ignore[attr-defined]
                    if line.class_filter and class_name not in line.class_filter:
                        continue
                    sign = cross_sign(line.p1, line.p2, prev, curr)
                    if sign != 0:
                        direction = CrossDirection.IN if sign > 0 else CrossDirection.OUT
                        events.append(
                            LineCrossEvent(
                                line_id=line.line_id,
                                track_id=tid,
                                direction=direction,
                                class_name=class_name,
                                frame_index=frame_index,
                                timestamp_ms=timestamp_ms,
                            )
                        )
                        self._line_totals[line.line_id][direction] += 1

            self._prev_centers[tid] = curr

        # Prune dead tracks — bounds memory, prevents stale crossings on track reuse
        dead = set(self._prev_centers.keys()) - current_ids
        for dead_id in dead:
            del self._prev_centers[dead_id]

        return events

    def reset(self) -> None:
        """Reset all cumulative counts, line totals, and tracking state."""
        with self._lock:
            self._cumulative.clear()
            for zc in self._zone_cumulative.values():
                zc.clear()
            for lt in self._line_totals.values():
                lt[CrossDirection.IN] = 0
                lt[CrossDirection.OUT] = 0
            self._prev_centers.clear()

    @property
    def counts(self) -> dict[str, int]:
        """Current cumulative per-class counts (snapshot copy)."""
        with self._lock:
            return dict(self._cumulative)

    @property
    def zone_counts(self) -> dict[str, dict[str, int]]:
        """Current cumulative per-zone, per-class counts (snapshot copy)."""
        with self._lock:
            return {zid: dict(zc) for zid, zc in self._zone_cumulative.items()}

    @property
    def line_totals(self) -> dict[str, dict[CrossDirection, int]]:
        """Current cumulative per-line crossing totals (snapshot copy)."""
        with self._lock:
            return {lid: dict(lt) for lid, lt in self._line_totals.items()}


__all__ = ["ObjectCounter"]
