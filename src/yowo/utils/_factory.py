"""Factory helpers for common zone / line configurations."""

from __future__ import annotations

from yowo.counter._types import CountLine, CountZone


def make_half_zones(width: int, height: int) -> tuple[CountZone, CountZone]:
    """Create top-half and bottom-half counting zones for a frame."""
    mid_y = float(height // 2)
    top = CountZone(
        "top_zone",
        vertices=(
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), mid_y),
            (0.0, mid_y),
        ),
    )
    bottom = CountZone(
        "bottom_zone",
        vertices=(
            (0.0, mid_y),
            (float(width), mid_y),
            (float(width), float(height)),
            (0.0, float(height)),
        ),
    )
    return top, bottom


def make_center_line(
    width: int,
    height: int,
    *,
    direction: str = "horizontal",
    line_id: str = "gate",
) -> CountLine:
    """Create a counting line at the frame center.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        direction: ``"horizontal"`` or ``"vertical"``.
        line_id: Identifier for the line.
    """
    if direction == "vertical":
        mid_x = float(width // 2)
        return CountLine(line_id, p1=(mid_x, float(height)), p2=(mid_x, 0.0))
    mid_y = float(height // 2)
    return CountLine(line_id, p1=(0.0, mid_y), p2=(float(width), mid_y))
