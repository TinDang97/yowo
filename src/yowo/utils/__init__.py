"""Reusable drawing, annotation, and factory utilities.

Exports:
    TRACK_PALETTE, CLASS_PALETTE — BGR colour palettes.
    color_for_track, color_for_class — deterministic colour selectors.
    draw_bounding_boxes — class-coloured detection boxes.
    draw_tracked_boxes — track-coloured boxes with IDs.
    draw_zones — semi-transparent zone polygon overlays.
    draw_count_lines — counting line overlays.
    draw_text_panel — translucent stats / info panel.
    make_half_zones — top/bottom zone factory.
    make_center_line — horizontal/vertical line factory.
"""

from yowo.utils._draw import (
    CLASS_PALETTE,
    TRACK_PALETTE,
    color_for_class,
    color_for_track,
    draw_bounding_boxes,
    draw_count_lines,
    draw_text_panel,
    draw_tracked_boxes,
    draw_zones,
)
from yowo.utils._factory import make_center_line, make_half_zones

__all__ = [
    "CLASS_PALETTE",
    "TRACK_PALETTE",
    "color_for_class",
    "color_for_track",
    "draw_bounding_boxes",
    "draw_count_lines",
    "draw_text_panel",
    "draw_tracked_boxes",
    "draw_zones",
    "make_center_line",
    "make_half_zones",
]
