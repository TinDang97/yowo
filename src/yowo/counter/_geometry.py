"""Pure geometric utility functions for object counting.

All functions operate on plain float tuples — no numpy dependency.
Designed for small N (< 20 zones, < 10 lines, < 100 detections per frame).
"""

from __future__ import annotations

from typing import Tuple

Point = Tuple[float, float]


def box_center(box: object) -> Point:
    """Return (cx, cy) center of any object with x1, y1, x2, y2 attributes."""
    x1: float = box.x1  # type: ignore[attr-defined]
    y1: float = box.y1  # type: ignore[attr-defined]
    x2: float = box.x2  # type: ignore[attr-defined]
    y2: float = box.y2  # type: ignore[attr-defined]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Test if a point lies inside a polygon using the ray-casting algorithm.

    Casts a horizontal ray from the point to +infinity and counts edge
    crossings. Odd count = inside (Jordan curve theorem).

    Args:
        point: (x, y) test point.
        polygon: Sequence of (x, y) vertices forming a closed polygon.
            The last vertex implicitly connects back to the first.

    Returns:
        True if the point is inside (or on the boundary of) the polygon.

    Complexity:
        O(V) where V is the number of vertices.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi < y) != (yj < y):
            x_intersect = xi + (y - yi) / (yj - yi) * (xj - xi)
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Return True if point q lies on segment p-r (all collinear, pre-checked)."""
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def _orientation(p: Point, q: Point, r: Point) -> int:
    """Return orientation of ordered triplet (p, q, r).

    Returns:
        0 if collinear, 1 if clockwise, 2 if counter-clockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-10:
        return 0
    return 1 if val > 0 else 2


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Test if line segment a1-a2 and b1-b2 intersect.

    Uses the four-orientation method. Returns True for proper intersections
    and endpoint-touching intersections.

    Args:
        a1, a2: Endpoints of segment A.
        b1, b2: Endpoints of segment B.

    Returns:
        True if the segments share at least one point.
    """
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True

    # Collinear cases
    if o1 == 0 and _on_segment(a1, b1, a2):
        return True
    if o2 == 0 and _on_segment(a1, b2, a2):
        return True
    if o3 == 0 and _on_segment(b1, a1, b2):
        return True
    return bool(o4 == 0 and _on_segment(b1, a2, b2))


def cross_sign(line_p1: Point, line_p2: Point, prev: Point, curr: Point) -> int:
    """Determine crossing direction of a movement vector across a directed line.

    Uses the cross product of the line direction vector (p1→p2) with
    the vectors from p1 to prev and curr. If signs differ, a crossing occurred.

    Convention: standing at p1 looking toward p2 —
        object moving from left to right = IN (+1)
        object moving from right to left = OUT (-1)
        no crossing = 0

    Args:
        line_p1: First endpoint of the counting line.
        line_p2: Second endpoint of the counting line.
        prev: Previous center position of the tracked object.
        curr: Current center position of the tracked object.

    Returns:
        +1 (IN), -1 (OUT), or 0 (no crossing / stationary).
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]

    cross_prev = dx * (prev[1] - line_p1[1]) - dy * (prev[0] - line_p1[0])
    cross_curr = dx * (curr[1] - line_p1[1]) - dy * (curr[0] - line_p1[0])

    if cross_prev < 0.0 and cross_curr > 0.0:
        return 1  # IN
    if cross_prev > 0.0 and cross_curr < 0.0:
        return -1  # OUT
    return 0


__all__ = [
    "Point",
    "box_center",
    "cross_sign",
    "point_in_polygon",
    "segments_intersect",
]
