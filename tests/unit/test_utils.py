"""Unit tests for yowo.utils — drawing utilities and factory helpers."""

from __future__ import annotations

import numpy as np

from yowo.counter._types import CountLine, CountZone
from yowo.tracking._strack import TrackedBox, TrackedDetection
from yowo.types import BackendType, BoundingBox, Frame, ModelFamily, ModelSize, ModelSpec
from yowo.utils import (
    CLASS_PALETTE,
    TRACK_PALETTE,
    color_for_class,
    color_for_track,
    draw_bounding_boxes,
    draw_count_lines,
    draw_text_panel,
    draw_tracked_boxes,
    draw_zones,
    make_center_line,
    make_half_zones,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)


def _frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_frame(h: int = 480, w: int = 640) -> Frame:
    return Frame(
        pixels=np.zeros((h, w, 3), dtype=np.uint8),
        source_id="test",
        frame_index=0,
        timestamp_ms=0.0,
    )


def _bbox(
    x1: float = 10,
    y1: float = 20,
    x2: float = 100,
    y2: float = 120,
    class_id: int = 0,
    class_name: str = "car",
    confidence: float = 0.9,
) -> BoundingBox:
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


def _tracked_box(
    x1: float = 10,
    y1: float = 20,
    x2: float = 100,
    y2: float = 120,
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "car",
    confidence: float = 0.9,
    is_confirmed: bool = True,
) -> TrackedBox:
    return TrackedBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        track_id=track_id,
        is_confirmed=is_confirmed,
    )


def _tracked_detection(
    boxes: tuple[TrackedBox, ...] | None = None,
) -> TrackedDetection:
    if boxes is None:
        boxes = (_tracked_box(),)
    return TrackedDetection(
        frame=_make_frame(),
        boxes=boxes,
        inference_time_ms=5.0,
        tracking_time_ms=0.3,
        backend=BackendType.ONNX,
        model_spec=_SPEC,
    )


# ---------------------------------------------------------------------------
# Palette tests
# ---------------------------------------------------------------------------
class TestPalettes:
    def test_track_palette_length(self) -> None:
        assert len(TRACK_PALETTE) == 10

    def test_class_palette_length(self) -> None:
        assert len(CLASS_PALETTE) == 20

    def test_track_palette_bgr_tuples(self) -> None:
        for color in TRACK_PALETTE:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_class_palette_bgr_tuples(self) -> None:
        for color in CLASS_PALETTE:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


# ---------------------------------------------------------------------------
# Color helper tests
# ---------------------------------------------------------------------------
class TestColorHelpers:
    def test_color_for_track_deterministic(self) -> None:
        assert color_for_track(0) == TRACK_PALETTE[0]
        assert color_for_track(5) == TRACK_PALETTE[5]

    def test_color_for_track_wraps(self) -> None:
        assert color_for_track(10) == TRACK_PALETTE[0]
        assert color_for_track(15) == TRACK_PALETTE[5]

    def test_color_for_class_deterministic(self) -> None:
        assert color_for_class(0) == CLASS_PALETTE[0]
        assert color_for_class(19) == CLASS_PALETTE[19]

    def test_color_for_class_wraps(self) -> None:
        assert color_for_class(20) == CLASS_PALETTE[0]


# ---------------------------------------------------------------------------
# draw_bounding_boxes
# ---------------------------------------------------------------------------
class TestDrawBoundingBoxes:
    def test_no_exception_on_valid_input(self) -> None:
        frame = _frame()
        boxes = [_bbox(), _bbox(x1=200, y1=200, x2=300, y2=300, class_id=3)]
        draw_bounding_boxes(frame, boxes)

    def test_shape_unchanged(self) -> None:
        frame = _frame(720, 1280)
        draw_bounding_boxes(frame, [_bbox()])
        assert frame.shape == (720, 1280, 3)

    def test_mutates_frame(self) -> None:
        frame = _frame()
        original_sum = frame.sum()
        draw_bounding_boxes(frame, [_bbox()])
        assert frame.sum() > original_sum

    def test_empty_boxes(self) -> None:
        frame = _frame()
        draw_bounding_boxes(frame, [])
        assert frame.sum() == 0


# ---------------------------------------------------------------------------
# draw_tracked_boxes
# ---------------------------------------------------------------------------
class TestDrawTrackedBoxes:
    def test_no_exception(self) -> None:
        frame = _frame()
        tracked = _tracked_detection()
        draw_tracked_boxes(frame, tracked)

    def test_shape_unchanged(self) -> None:
        frame = _frame()
        tracked = _tracked_detection()
        draw_tracked_boxes(frame, tracked)
        assert frame.shape == (480, 640, 3)

    def test_unconfirmed_track(self) -> None:
        frame = _frame()
        box = _tracked_box(is_confirmed=False)
        tracked = _tracked_detection(boxes=(box,))
        draw_tracked_boxes(frame, tracked)

    def test_empty_boxes(self) -> None:
        frame = _frame()
        tracked = _tracked_detection(boxes=())
        draw_tracked_boxes(frame, tracked)
        assert frame.sum() == 0


# ---------------------------------------------------------------------------
# draw_zones
# ---------------------------------------------------------------------------
class TestDrawZones:
    def test_no_exception(self) -> None:
        frame = _frame()
        zones = list(make_half_zones(640, 480))
        draw_zones(frame, zones)

    def test_shape_unchanged(self) -> None:
        frame = _frame()
        zones = list(make_half_zones(640, 480))
        draw_zones(frame, zones)
        assert frame.shape == (480, 640, 3)

    def test_empty_zones(self) -> None:
        frame = _frame()
        draw_zones(frame, [])
        assert frame.sum() == 0


# ---------------------------------------------------------------------------
# draw_count_lines
# ---------------------------------------------------------------------------
class TestDrawCountLines:
    def test_no_exception(self) -> None:
        frame = _frame()
        line = make_center_line(640, 480)
        draw_count_lines(frame, [line])

    def test_shape_unchanged(self) -> None:
        frame = _frame()
        line = make_center_line(640, 480)
        draw_count_lines(frame, [line])
        assert frame.shape == (480, 640, 3)

    def test_empty_lines(self) -> None:
        frame = _frame()
        draw_count_lines(frame, [])
        assert frame.sum() == 0


# ---------------------------------------------------------------------------
# draw_text_panel
# ---------------------------------------------------------------------------
class TestDrawTextPanel:
    def test_no_exception(self) -> None:
        frame = _frame(720, 1280)
        draw_text_panel(frame, ["FPS: 30.0", "Active: 5"])

    def test_shape_unchanged(self) -> None:
        frame = _frame(720, 1280)
        draw_text_panel(frame, ["hello"])
        assert frame.shape == (720, 1280, 3)

    def test_empty_lines_noop(self) -> None:
        frame = _frame()
        original_sum = frame.sum()
        draw_text_panel(frame, [])
        assert frame.sum() == original_sum

    def test_top_left_position(self) -> None:
        frame = _frame(720, 1280)
        draw_text_panel(frame, ["test"], position="top-left")

    def test_top_right_position(self) -> None:
        frame = _frame(720, 1280)
        draw_text_panel(frame, ["test"], position="top-right")


# ---------------------------------------------------------------------------
# make_half_zones
# ---------------------------------------------------------------------------
class TestMakeHalfZones:
    def test_returns_two_zones(self) -> None:
        top, bottom = make_half_zones(640, 480)
        assert isinstance(top, CountZone)
        assert isinstance(bottom, CountZone)

    def test_zone_ids(self) -> None:
        top, bottom = make_half_zones(640, 480)
        assert top.zone_id == "top_zone"
        assert bottom.zone_id == "bottom_zone"

    def test_top_zone_vertices(self) -> None:
        top, _ = make_half_zones(640, 480)
        assert top.vertices == (
            (0.0, 0.0),
            (640.0, 0.0),
            (640.0, 240.0),
            (0.0, 240.0),
        )

    def test_bottom_zone_vertices(self) -> None:
        _, bottom = make_half_zones(640, 480)
        assert bottom.vertices == (
            (0.0, 240.0),
            (640.0, 240.0),
            (640.0, 480.0),
            (0.0, 480.0),
        )

    def test_zones_cover_full_frame(self) -> None:
        w, h = 1280, 720
        top, bottom = make_half_zones(w, h)
        # Top goes from y=0 to y=mid
        assert top.vertices[0][1] == 0.0
        assert top.vertices[2][1] == float(h // 2)
        # Bottom goes from y=mid to y=h
        assert bottom.vertices[0][1] == float(h // 2)
        assert bottom.vertices[2][1] == float(h)


# ---------------------------------------------------------------------------
# make_center_line
# ---------------------------------------------------------------------------
class TestMakeCenterLine:
    def test_horizontal_default(self) -> None:
        line = make_center_line(640, 480)
        assert line.line_id == "gate"
        assert line.p1 == (0.0, 240.0)
        assert line.p2 == (640.0, 240.0)

    def test_vertical(self) -> None:
        line = make_center_line(640, 480, direction="vertical")
        assert line.p1 == (320.0, 480.0)
        assert line.p2 == (320.0, 0.0)

    def test_custom_line_id(self) -> None:
        line = make_center_line(640, 480, line_id="custom_gate")
        assert line.line_id == "custom_gate"

    def test_returns_count_line(self) -> None:
        line = make_center_line(100, 200)
        assert isinstance(line, CountLine)
