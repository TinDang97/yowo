"""Data models for yowo object counting."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CrossDirection(enum.StrEnum):
    """Direction of a line-crossing event relative to the line's direction vector."""

    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class CountZone:
    """Polygonal region of interest for zone-based counting.

    Vertices define a closed polygon in pixel coordinates. The last vertex
    connects back to the first automatically. Minimum 3 vertices required.

    Attributes:
        zone_id: Unique identifier for this zone.
        vertices: Sequence of (x, y) points. Must contain at least 3 points.
        class_filter: If non-empty, only count detections whose class_name
            is in this set. Empty frozenset means count all classes.
    """

    zone_id: str
    vertices: tuple[tuple[float, float], ...]
    class_filter: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError(
                f"CountZone '{self.zone_id}' requires >= 3 vertices, got {len(self.vertices)}"
            )


@dataclass(frozen=True, slots=True)
class CountLine:
    """Directed line segment for crossing-based counting.

    The direction p1 → p2 determines the positive side. Objects crossing
    from the left side of this vector to the right are counted as IN;
    right-to-left crossings are counted as OUT.

    Attributes:
        line_id: Unique identifier for this counting line.
        p1: Start point (x, y) in pixel coordinates.
        p2: End point (x, y) in pixel coordinates.
        class_filter: If non-empty, only count crossings for these classes.
    """

    line_id: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    class_filter: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.p1 == self.p2:
            raise ValueError(
                f"CountLine '{self.line_id}' requires p1 != p2, got p1 == p2 == {self.p1}"
            )


@dataclass(frozen=True, slots=True)
class LineCrossEvent:
    """A single line-crossing event.

    Attributes:
        line_id: Which counting line was crossed.
        track_id: Tracker-assigned ID of the crossing object.
        direction: Whether the crossing was IN or OUT.
        class_name: Class label of the crossing object.
        frame_index: Frame index at which the crossing occurred.
        timestamp_ms: Source timestamp when the crossing occurred.
    """

    line_id: str
    track_id: int
    direction: CrossDirection
    class_name: str
    frame_index: int
    timestamp_ms: float


@dataclass(frozen=True, slots=True)
class CountResult:
    """Immutable snapshot returned by ObjectCounter.update().

    Attributes:
        frame_index: Frame index of the processed detection.
        timestamp_ms: Source timestamp of the processed detection.
        live_counts: Per-class counts for the current frame only.
        cumulative_counts: Total per-class counts since last reset().
        zone_counts: Per-zone, per-class counts for the current frame.
        cumulative_zone_counts: Per-zone cumulative counts since last reset().
        line_events: Line-crossing events detected in this frame.
        line_totals: Per-line cumulative crossing totals since last reset().
    """

    frame_index: int
    timestamp_ms: float
    live_counts: dict[str, int]
    cumulative_counts: dict[str, int]
    zone_counts: dict[str, dict[str, int]]
    cumulative_zone_counts: dict[str, dict[str, int]]
    line_events: tuple[LineCrossEvent, ...]
    line_totals: dict[str, dict[CrossDirection, int]]


__all__ = [
    "CountLine",
    "CountResult",
    "CountZone",
    "CrossDirection",
    "LineCrossEvent",
]
