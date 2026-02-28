"""Unit tests for yowo.counter — ObjectCounter, geometry, zone/line counting."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pytest

from yowo.counter._counter import ObjectCounter, _HasTrackId
from yowo.counter._geometry import box_center, cross_sign, point_in_polygon, segments_intersect
from yowo.counter._types import (
    CountLine,
    CountZone,
    CrossDirection,
)
from yowo.types import BackendType, BoundingBox, Detection, Frame, ModelFamily, ModelSize, ModelSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)


def _make_frame(frame_index: int = 0, timestamp_ms: float = 0.0) -> Frame:
    return Frame(
        pixels=np.zeros((480, 640, 3), dtype=np.uint8),
        source_id="test",
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )


def _make_box(
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 200.0,
    conf: float = 0.9,
    cls_id: int = 0,
    cls_name: str = "person",
) -> BoundingBox:
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
    )


def _make_detection(
    boxes: tuple[BoundingBox, ...] = (),
    frame_index: int = 0,
) -> Detection:
    return Detection(
        frame=_make_frame(frame_index=frame_index),
        boxes=boxes,
        inference_time_ms=5.0,
        backend=BackendType.ONNX,
        model_spec=_SPEC,
    )


@dataclass
class _MockTrackedBox:
    """Minimal tracked box satisfying _HasTrackId Protocol (no yowo.tracking import)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.9
    class_id: int = 0
    class_name: str = "person"
    track_id: int = 1
    is_confirmed: bool = True


@dataclass
class _MockTrackedDet:
    """Minimal TrackedDetection-like for counter tests."""

    frame: Frame
    boxes: tuple
    inference_time_ms: float = 5.0
    tracking_time_ms: float = 0.1
    backend: BackendType = BackendType.ONNX
    model_spec: ModelSpec = field(default_factory=lambda: _SPEC)


def _make_tracked_det(
    boxes: tuple = (),
    frame_index: int = 0,
) -> _MockTrackedDet:
    return _MockTrackedDet(frame=_make_frame(frame_index=frame_index), boxes=boxes)


# ---------------------------------------------------------------------------
# Section 1: Geometry
# ---------------------------------------------------------------------------

_SQUARE: tuple[tuple[float, float], ...] = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


class TestBoxCenter:
    def test_square_box(self) -> None:
        box = _make_box(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        assert box_center(box) == (50.0, 50.0)

    def test_asymmetric_box(self) -> None:
        box = _make_box(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
        cx, cy = box_center(box)
        assert abs(cx - 30.0) < 1e-9
        assert abs(cy - 50.0) < 1e-9


class TestPointInPolygon:
    def test_center_inside_square(self) -> None:
        assert point_in_polygon((50.0, 50.0), _SQUARE) is True

    def test_outside_square(self) -> None:
        assert point_in_polygon((150.0, 50.0), _SQUARE) is False

    def test_far_outside(self) -> None:
        assert point_in_polygon((-10.0, -10.0), _SQUARE) is False

    def test_triangle_inside(self) -> None:
        tri: tuple[tuple[float, float], ...] = ((0.0, 0.0), (10.0, 0.0), (5.0, 10.0))
        assert point_in_polygon((5.0, 3.0), tri) is True

    def test_triangle_outside(self) -> None:
        tri: tuple[tuple[float, float], ...] = ((0.0, 0.0), (10.0, 0.0), (5.0, 10.0))
        assert point_in_polygon((15.0, 5.0), tri) is False

    def test_concave_l_shape_inside(self) -> None:
        # L-shape: bottom-left part of the concave region
        l_shape: tuple[tuple[float, float], ...] = (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 50.0),
            (50.0, 50.0),
            (50.0, 100.0),
            (0.0, 100.0),
        )
        assert point_in_polygon((25.0, 75.0), l_shape) is True

    def test_concave_l_shape_void(self) -> None:
        l_shape: tuple[tuple[float, float], ...] = (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 50.0),
            (50.0, 50.0),
            (50.0, 100.0),
            (0.0, 100.0),
        )
        # Point in the concave void (top-right quadrant)
        assert point_in_polygon((75.0, 75.0), l_shape) is False


class TestSegmentsIntersect:
    def test_x_cross(self) -> None:
        # Horizontal and vertical crossing at (50,50)
        assert segments_intersect((0.0, 50.0), (100.0, 50.0), (50.0, 0.0), (50.0, 100.0)) is True

    def test_parallel_horizontal(self) -> None:
        assert segments_intersect((0.0, 0.0), (100.0, 0.0), (0.0, 10.0), (100.0, 10.0)) is False

    def test_collinear_no_overlap(self) -> None:
        assert segments_intersect((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)) is False

    def test_t_intersection(self) -> None:
        # T-shape: endpoint of B lies on A
        assert segments_intersect((0.0, 0.0), (100.0, 0.0), (50.0, 0.0), (50.0, 50.0)) is True


class TestCrossSign:
    def test_in_left_to_right(self) -> None:
        # Vertical line x=100 pointing UPWARD (p1 bottom, p2 top) so that
        # left-to-right screen movement (x:50→150) = IN (+1).
        # Looking upward: left=west(-x), right=east(+x); x:50→150 = left→right = IN.
        sign = cross_sign((100.0, 200.0), (100.0, 0.0), (50.0, 100.0), (150.0, 100.0))
        assert sign == 1

    def test_out_right_to_left(self) -> None:
        # Same upward line, right-to-left movement → OUT (-1)
        sign = cross_sign((100.0, 200.0), (100.0, 0.0), (150.0, 100.0), (50.0, 100.0))
        assert sign == -1

    def test_no_crossing_same_side(self) -> None:
        # Both points on same side — no crossing
        sign = cross_sign((100.0, 200.0), (100.0, 0.0), (50.0, 100.0), (80.0, 100.0))
        assert sign == 0

    def test_stationary_no_crossing(self) -> None:
        # prev == curr (stationary object)
        sign = cross_sign((100.0, 200.0), (100.0, 0.0), (50.0, 100.0), (50.0, 100.0))
        assert sign == 0


# ---------------------------------------------------------------------------
# Section 2: Type validation
# ---------------------------------------------------------------------------


class TestTypeValidation:
    def test_count_zone_too_few_vertices(self) -> None:
        with pytest.raises(ValueError, match="entrance"):
            CountZone("entrance", vertices=((0.0, 0.0), (100.0, 0.0)))

    def test_count_zone_exactly_three_vertices_ok(self) -> None:
        zone = CountZone("z", vertices=((0.0, 0.0), (100.0, 0.0), (50.0, 100.0)))
        assert len(zone.vertices) == 3

    def test_count_line_same_endpoints(self) -> None:
        with pytest.raises(ValueError, match="gate"):
            CountLine("gate", p1=(50.0, 0.0), p2=(50.0, 0.0))


# ---------------------------------------------------------------------------
# Section 3: Basic counting
# ---------------------------------------------------------------------------


class TestBasicCounting:
    def test_live_counts_empty_detection(self) -> None:
        counter = ObjectCounter()
        result = counter.update(_make_detection())
        assert result.live_counts == {}
        assert result.cumulative_counts == {}

    def test_live_counts_single_class(self) -> None:
        counter = ObjectCounter()
        boxes = (_make_box(), _make_box(), _make_box())
        result = counter.update(_make_detection(boxes=boxes))
        assert result.live_counts == {"person": 3}

    def test_live_counts_multiple_classes(self) -> None:
        counter = ObjectCounter()
        boxes = (
            _make_box(cls_name="person"),
            _make_box(cls_name="car"),
            _make_box(cls_name="person"),
        )
        result = counter.update(_make_detection(boxes=boxes))
        assert result.live_counts == {"person": 2, "car": 1}

    def test_cumulative_across_frames(self) -> None:
        counter = ObjectCounter()
        counter.update(_make_detection(boxes=(_make_box(),), frame_index=0))
        result = counter.update(_make_detection(boxes=(_make_box(),), frame_index=1))
        assert result.cumulative_counts["person"] == 2

    def test_reset_clears_cumulative(self) -> None:
        counter = ObjectCounter()
        counter.update(_make_detection(boxes=(_make_box(),)))
        counter.reset()
        result = counter.update(_make_detection(boxes=(_make_box(),)))
        assert result.cumulative_counts == {"person": 1}

    def test_counts_property_returns_current(self) -> None:
        counter = ObjectCounter()
        counter.update(_make_detection(boxes=(_make_box(),)))
        assert counter.counts == {"person": 1}


# ---------------------------------------------------------------------------
# Section 4: Zone counting
# ---------------------------------------------------------------------------

_ZONE = CountZone("zone_a", vertices=((0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)))


class TestZoneCounting:
    def test_box_center_inside(self) -> None:
        counter = ObjectCounter(zones=[_ZONE])
        # box(50,50,100,100) → center (75,75) inside zone
        result = counter.update(_make_detection(boxes=(_make_box(50, 50, 100, 100),)))
        assert result.zone_counts["zone_a"].get("person", 0) == 1

    def test_box_center_outside(self) -> None:
        counter = ObjectCounter(zones=[_ZONE])
        # box(400,400,500,500) → center (450,450) outside zone
        result = counter.update(_make_detection(boxes=(_make_box(400, 400, 500, 500),)))
        assert result.zone_counts["zone_a"].get("person", 0) == 0

    def test_class_filter_excludes(self) -> None:
        zone = CountZone(
            "z",
            vertices=((0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)),
            class_filter=frozenset({"car"}),
        )
        counter = ObjectCounter(zones=[zone])
        # person box inside zone, but filter only allows car
        result = counter.update(_make_detection(boxes=(_make_box(50, 50, 100, 100),)))
        assert result.zone_counts["z"].get("person", 0) == 0

    def test_class_filter_empty_counts_all(self) -> None:
        zone = CountZone(
            "z",
            vertices=((0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)),
            class_filter=frozenset(),
        )
        counter = ObjectCounter(zones=[zone])
        boxes = (
            _make_box(50, 50, 100, 100, cls_name="person"),
            _make_box(60, 60, 110, 110, cls_name="car"),
        )
        result = counter.update(_make_detection(boxes=boxes))
        assert result.zone_counts["z"].get("person", 0) == 1
        assert result.zone_counts["z"].get("car", 0) == 1

    def test_zone_cumulative_across_frames(self) -> None:
        counter = ObjectCounter(zones=[_ZONE])
        box = _make_box(50, 50, 100, 100)
        counter.update(_make_detection(boxes=(box,), frame_index=0))
        counter.update(_make_detection(boxes=(box,), frame_index=1))
        assert counter.zone_counts["zone_a"]["person"] == 2

    def test_multiple_zones_independent(self) -> None:
        zone_a = CountZone("a", vertices=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
        zone_b = CountZone(
            "b", vertices=((200.0, 200.0), (400.0, 200.0), (400.0, 400.0), (200.0, 400.0))
        )
        counter = ObjectCounter(zones=[zone_a, zone_b])
        boxes = (
            _make_box(10, 10, 50, 50),  # center (30,30) → in zone_a
            _make_box(210, 210, 350, 350),  # center (280,280) → in zone_b
        )
        result = counter.update(_make_detection(boxes=boxes))
        assert result.zone_counts["a"].get("person", 0) == 1
        assert result.zone_counts["b"].get("person", 0) == 1


# ---------------------------------------------------------------------------
# Section 5: Line-cross counting
# ---------------------------------------------------------------------------

# Vertical line at x=100, pointing UPWARD (p1 at bottom, p2 at top)
# so that left-to-right screen movement = IN, right-to-left = OUT.
_LINE = CountLine("gate", p1=(100.0, 480.0), p2=(100.0, 0.0))


def _tracked_det_at(x: float, track_id: int = 1, frame_index: int = 0) -> _MockTrackedDet:
    """Tracked detection with box centered at (x, 100)."""
    box = _MockTrackedBox(x1=x - 25, y1=75.0, x2=x + 25, y2=125.0, track_id=track_id)
    return _make_tracked_det(boxes=(box,), frame_index=frame_index)


class TestLineCross:
    def test_first_frame_no_event(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        result = counter.update(_tracked_det_at(x=50.0, frame_index=0))
        assert result.line_events == ()

    def test_cross_in(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, frame_index=0))  # left side
        result = counter.update(_tracked_det_at(x=150.0, frame_index=1))  # right side → IN
        assert len(result.line_events) == 1
        assert result.line_events[0].direction == CrossDirection.IN
        assert result.line_events[0].line_id == "gate"
        assert result.line_events[0].track_id == 1

    def test_cross_out(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=150.0, frame_index=0))  # right side
        result = counter.update(_tracked_det_at(x=50.0, frame_index=1))  # left side → OUT
        assert len(result.line_events) == 1
        assert result.line_events[0].direction == CrossDirection.OUT

    def test_no_cross_same_side(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, frame_index=0))
        result = counter.update(_tracked_det_at(x=80.0, frame_index=1))
        assert result.line_events == ()

    def test_multiple_crossings_same_track_same_line(self) -> None:
        """Each actual crossing fires an independent event (IN then OUT then IN...)."""
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, frame_index=0))  # left of line
        counter.update(_tracked_det_at(x=150.0, frame_index=1))  # crosses → IN
        result = counter.update(_tracked_det_at(x=50.0, frame_index=2))  # crosses back → OUT
        assert len(result.line_events) == 1
        assert result.line_events[0].direction == CrossDirection.OUT
        assert counter.line_totals["gate"][CrossDirection.IN] == 1
        assert counter.line_totals["gate"][CrossDirection.OUT] == 1

    def test_track_death_prunes_state(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, track_id=1, frame_index=0))
        # Frame with different track_id — track 1 disappears
        box2 = _MockTrackedBox(x1=200, y1=75, x2=250, y2=125, track_id=2)
        counter.update(_make_tracked_det(boxes=(box2,), frame_index=1))
        # Internal _prev_centers should no longer contain track_id=1
        assert 1 not in counter._prev_centers  # type: ignore[attr-defined]

    def test_class_filter_on_line(self) -> None:
        line = CountLine(
            "gate", p1=(100.0, 480.0), p2=(100.0, 0.0), class_filter=frozenset({"car"})
        )
        counter = ObjectCounter(lines=[line])
        counter.update(_tracked_det_at(x=50.0, frame_index=0))  # class_name="person"
        result = counter.update(_tracked_det_at(x=150.0, frame_index=1))
        assert result.line_events == ()

    def test_plain_detection_no_events(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        # Plain Detection boxes have no track_id
        counter.update(_make_detection(boxes=(_make_box(x1=75, y1=75, x2=125, y2=125),)))
        result = counter.update(_make_detection(boxes=(_make_box(x1=125, y1=75, x2=175, y2=125),)))
        assert result.line_events == ()

    def test_multiple_lines_both_crossed(self) -> None:
        line_a = CountLine("a", p1=(100.0, 480.0), p2=(100.0, 0.0))
        line_b = CountLine("b", p1=(200.0, 480.0), p2=(200.0, 0.0))
        counter = ObjectCounter(lines=[line_a, line_b])
        # Start at x=50, jump to x=250 — crosses both lines in one step
        counter.update(_tracked_det_at(x=50.0, frame_index=0))
        result = counter.update(_tracked_det_at(x=250.0, frame_index=1))
        assert len(result.line_events) == 2
        line_ids = {e.line_id for e in result.line_events}
        assert line_ids == {"a", "b"}

    def test_line_totals_accumulate(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, track_id=1, frame_index=0))
        counter.update(_tracked_det_at(x=150.0, track_id=1, frame_index=1))  # track 1 IN
        # Track 2 crosses independently
        counter.update(
            _make_tracked_det(
                boxes=(_MockTrackedBox(x1=25, y1=75, x2=75, y2=125, track_id=2),),
                frame_index=0,
            )
        )
        counter.update(
            _make_tracked_det(
                boxes=(_MockTrackedBox(x1=125, y1=75, x2=175, y2=125, track_id=2),),
                frame_index=1,
            )
        )
        assert counter.line_totals["gate"][CrossDirection.IN] == 2


# ---------------------------------------------------------------------------
# Section 6: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_updates_no_race(self) -> None:
        counter = ObjectCounter()
        n_threads = 10
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                barrier.wait()
                counter.update(_make_detection(boxes=(_make_box(),)))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert counter.counts.get("person", 0) == n_threads

    def test_counts_property_is_copy(self) -> None:
        counter = ObjectCounter()
        counter.update(_make_detection(boxes=(_make_box(),)))
        snapshot = counter.counts
        snapshot["person"] = 999  # mutate snapshot
        # Internal state unchanged
        assert counter.counts["person"] == 1


# ---------------------------------------------------------------------------
# Section 7: CountResult immutability
# ---------------------------------------------------------------------------


class TestCountResult:
    def test_result_is_frozen(self) -> None:
        counter = ObjectCounter()
        result = counter.update(_make_detection())
        with pytest.raises((AttributeError, TypeError)):
            result.frame_index = 99  # type: ignore[misc]

    def test_line_events_is_tuple(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        result = counter.update(_make_detection())
        assert isinstance(result.line_events, tuple)

    def test_result_frame_metadata(self) -> None:
        counter = ObjectCounter()
        result = counter.update(_make_detection(frame_index=42))
        assert result.frame_index == 42

    def test_has_track_id_protocol_check(self) -> None:
        box = _MockTrackedBox(x1=0, y1=0, x2=10, y2=10)
        assert isinstance(box, _HasTrackId)

    def test_plain_box_not_has_track_id(self) -> None:
        box = _make_box()
        assert not isinstance(box, _HasTrackId)


# ---------------------------------------------------------------------------
# Section 8: Coverage gap tests
# ---------------------------------------------------------------------------


class TestCounterPropertySnapshots:
    def test_zone_counts_returns_snapshot(self) -> None:
        counter = ObjectCounter(zones=[_ZONE])
        counter.update(_make_detection(boxes=(_make_box(50, 50, 100, 100),)))
        snapshot = counter.zone_counts
        snapshot["zone_a"]["person"] = 999
        assert counter.zone_counts["zone_a"]["person"] == 1

    def test_line_totals_returns_snapshot(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, frame_index=0))
        counter.update(_tracked_det_at(x=150.0, frame_index=1))
        snapshot = counter.line_totals
        snapshot["gate"][CrossDirection.IN] = 999
        assert counter.line_totals["gate"][CrossDirection.IN] == 1

    def test_reset_clears_line_totals(self) -> None:
        counter = ObjectCounter(lines=[_LINE])
        counter.update(_tracked_det_at(x=50.0, frame_index=0))
        counter.update(_tracked_det_at(x=150.0, frame_index=1))
        assert counter.line_totals["gate"][CrossDirection.IN] == 1
        counter.reset()
        assert counter.line_totals["gate"][CrossDirection.IN] == 0
        assert counter.line_totals["gate"][CrossDirection.OUT] == 0

    def test_cumulative_zone_counts_in_result(self) -> None:
        counter = ObjectCounter(zones=[_ZONE])
        box = _make_box(50, 50, 100, 100)
        counter.update(_make_detection(boxes=(box,), frame_index=0))
        result = counter.update(_make_detection(boxes=(box,), frame_index=1))
        assert result.cumulative_zone_counts["zone_a"]["person"] == 2


class TestSegmentsIntersectEdge:
    def test_collinear_with_overlap(self) -> None:
        assert segments_intersect((0.0, 0.0), (20.0, 0.0), (10.0, 0.0), (30.0, 0.0)) is True


class TestLineCrossEventDirect:
    def test_fields(self) -> None:
        from yowo.counter._types import LineCrossEvent

        evt = LineCrossEvent(
            line_id="gate",
            track_id=5,
            direction=CrossDirection.IN,
            class_name="car",
            frame_index=10,
            timestamp_ms=1234.5,
        )
        assert evt.line_id == "gate"
        assert evt.track_id == 5
        assert evt.direction == CrossDirection.IN
        assert evt.class_name == "car"
        assert evt.frame_index == 10
        assert evt.timestamp_ms == 1234.5


class TestCountLineValid:
    def test_valid_construction(self) -> None:
        line = CountLine("gate", p1=(0.0, 0.0), p2=(100.0, 100.0))
        assert line.line_id == "gate"
        assert line.p1 == (0.0, 0.0)
        assert line.p2 == (100.0, 100.0)
        assert line.class_filter == frozenset()
