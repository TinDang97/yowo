"""Reusable drawing / annotation utilities for video frames.

All functions mutate *frame* in-place and return ``None``.
Colour values are BGR tuples (OpenCV convention).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from yowo.counter._types import CountLine, CountZone
from yowo.tracking._strack import TrackedBox, TrackedDetection
from yowo.types import BoundingBox

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

#: 10-colour palette cycled by ``track_id``.
TRACK_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 50, 50),
    (50, 255, 50),
    (50, 50, 255),
    (255, 255, 50),
    (255, 50, 255),
    (50, 255, 255),
    (255, 150, 50),
    (50, 255, 150),
    (150, 50, 255),
    (255, 200, 100),
)

#: 20-colour palette cycled by ``class_id``.
CLASS_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
)

#: Zone overlay colours cycled per zone index.
_ZONE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (200, 150, 50),
    (50, 150, 100),
    (150, 50, 200),
    (50, 200, 150),
)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for *track_id*."""
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for *class_id*."""
    return CLASS_PALETTE[class_id % len(CLASS_PALETTE)]


# ---------------------------------------------------------------------------
# Box drawing
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bounding_boxes(
    frame: np.ndarray,
    boxes: Sequence[BoundingBox],
    *,
    font_scale: float = 0.5,
    thickness: int = 2,
) -> None:
    """Draw class-coloured bounding boxes with *"class conf"* labels."""
    for box in boxes:
        color = color_for_class(box.class_id)
        x1, y1, x2, y2 = round(box.x1), round(box.y1), round(box.x2), round(box.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{box.class_name} {box.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)
        label_y = max(y1 - baseline, th)
        cv2.rectangle(
            frame,
            (x1, label_y - th - baseline),
            (x1 + tw, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            _FONT,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_tracked_boxes(
    frame: np.ndarray,
    tracked: TrackedDetection,
    *,
    font_scale: float = 0.45,
) -> None:
    """Draw track-coloured boxes with *"ID:N class conf"* labels."""
    box: TrackedBox
    for box in tracked.boxes:
        color = color_for_track(box.track_id)
        x1, y1 = int(box.x1), int(box.y1)
        x2, y2 = int(box.x2), int(box.y2)
        thick = 2 if box.is_confirmed else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = f"ID:{box.track_id} {box.class_name} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, font_scale, 1)
        cv2.rectangle(
            frame,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            _FONT,
            font_scale,
            (255, 255, 255),
            1,
        )


# ---------------------------------------------------------------------------
# Zone / line drawing
# ---------------------------------------------------------------------------


def draw_zones(
    frame: np.ndarray,
    zones: Sequence[CountZone],
    *,
    alpha: float = 0.15,
) -> None:
    """Draw semi-transparent zone polygons with borders and labels."""
    overlay = frame.copy()
    for i, zone in enumerate(zones):
        color = _ZONE_PALETTE[i % len(_ZONE_PALETTE)]
        pts = np.array(
            [(int(x), int(y)) for x, y in zone.vertices],
            np.int32,
        )
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
        lx = int(zone.vertices[0][0]) + 5
        ly = int(zone.vertices[0][1]) + 25
        cv2.putText(
            frame,
            zone.zone_id.upper(),
            (lx, ly),
            _FONT,
            0.6,
            color,
            2,
        )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_count_lines(
    frame: np.ndarray,
    lines: Sequence[CountLine],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw counting lines in *color* (default red) with labels."""
    for line in lines:
        p1 = (int(line.p1[0]), int(line.p1[1]))
        p2 = (int(line.p2[0]), int(line.p2[1]))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        cv2.putText(
            frame,
            line.line_id.upper(),
            (mx - 40, my - 10),
            _FONT,
            0.5,
            color,
            2,
        )


# ---------------------------------------------------------------------------
# Text panel
# ---------------------------------------------------------------------------


def draw_text_panel(
    frame: np.ndarray,
    lines: Sequence[str],
    *,
    position: str = "top-right",
    font_scale: float = 0.45,
) -> None:
    """Draw a translucent text panel on *frame*.

    Args:
        frame: BGR image (HWC uint8).
        lines: Lines of text to render.
        position: ``"top-right"`` or ``"top-left"``.
        font_scale: OpenCV font scale.
    """
    if not lines:
        return
    _, fw = frame.shape[:2]
    max_tw = max(cv2.getTextSize(t, _FONT, font_scale, 1)[0][0] for t in lines)
    line_h = 20
    box_h = len(lines) * line_h + 10

    x0 = 10 if position == "top-left" else fw - max_tw - 20

    cv2.rectangle(frame, (x0 - 5, 5), (x0 + max_tw + 10, box_h + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0 - 5, 5), (x0 + max_tw + 10, box_h + 5), (180, 180, 180), 1)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x0, line_h + i * line_h),
            _FONT,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
