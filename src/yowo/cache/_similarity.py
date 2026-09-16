"""How the feature cache decides two frames are the same scene.

This module used to publish `frame_similarity`, which compared two full
tensors and returned 1.0 when their shapes differed. It had ZERO callers in
`src/` and was tested at `tests/unit/test_cache.py`, so its shape guard was
covered, passing, and unreachable, while the path a user actually hits --
`FeatureCache.check_and_load` -- compared `tensor.mean(axis=(2, 3))` with no
shape check at all.

Measured 2026-09-16 against that fingerprint, at the shipped 0.01 threshold:

    a 90x90 px object at max contrast    invisible   (1.98% of a 640x640 frame)
    165x165 at a realistic 0.15 contrast invisible   (6.65%)
    a 640x640 entry answering 320x320    cache HIT, 80x80 features for a 40x40 need
    a B=1 entry answering B=4            cache HIT, the fingerprints broadcast
    black-over-white vs uniform grey     cache HIT, both average exactly 0.5

Three numbers cannot describe where anything is, so none of those failures is
small. The fingerprint here is pooled over a coarse grid and compared by the
WORST cell rather than the frame average, which bounds the blind spot instead
of removing it -- a difference spread thinly enough to leave every cell mean
intact is invisible to any pooled fingerprint, and claiming otherwise would be
the same overclaim in a new size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["DEFAULT_GRID", "fingerprint_distance", "spatial_fingerprint"]

#: Cells per side. 8 gives 80x80-pixel cells on a 640 frame, which bounds the
#: invisible object at roughly 11 px aligned and 23 px straddling a corner, at
#: maximum contrast. Raising it shrinks the blind spot and costs hit rate.
DEFAULT_GRID = 8


def spatial_fingerprint(
    tensor: NDArray[np.float32],
    grid: int = DEFAULT_GRID,
) -> NDArray[np.float32]:
    """Per-cell means of a BCHW tensor, shaped ``(B, C, gy, gx)``.

    Cell edges come from ``np.linspace`` over the real extent, so every pixel
    falls in exactly one cell whatever ``H`` and ``W`` are -- a 641-pixel side
    at grid 8 gives seven 80-pixel cells and one of 81, not a dropped column.
    A side shorter than ``grid`` gets one cell per pixel rather than an empty
    cell or a division by zero.

    Computed from an integral image, so the cost is one pass over the tensor
    regardless of how many cells are asked for.
    """
    _, _, height, width = tensor.shape
    gy, gx = min(grid, height), min(grid, width)

    ys = np.linspace(0, height, gy + 1).round().astype(np.intp)
    xs = np.linspace(0, width, gx + 1).round().astype(np.intp)

    integral = np.pad(
        tensor.cumsum(axis=2, dtype=np.float64).cumsum(axis=3, dtype=np.float64),
        ((0, 0), (0, 0), (1, 0), (1, 0)),
    )

    lo_y, hi_y = ys[:-1, None], ys[1:, None]
    lo_x, hi_x = xs[None, :-1], xs[None, 1:]
    sums = (
        integral[:, :, hi_y, hi_x]
        - integral[:, :, lo_y, hi_x]
        - integral[:, :, hi_y, lo_x]
        + integral[:, :, lo_y, lo_x]
    )
    areas = (ys[1:] - ys[:-1])[:, None] * (xs[1:] - xs[:-1])[None, :]
    return (sums / areas).astype(np.float32)


def fingerprint_distance(
    current: NDArray[np.float32],
    previous: NDArray[np.float32],
) -> float:
    """Worst per-cell absolute difference, or 1.0 when the two do not align.

    The maximum, not the mean. A mean over the frame is what let a 90x90
    object hide inside 409,600 pixels: dividing a real local change by the
    whole frame is how it stopped clearing a threshold.

    Returns 1.0 -- a guaranteed miss -- on any shape mismatch, rather than
    letting numpy broadcast a ``(1, C, g, g)`` entry into agreement with a
    ``(4, C, g, g)`` query. That broadcast is not hypothetical: ``cache=True``
    ships in a preset that also sets ``batch_size=4``.
    """
    if current.shape != previous.shape:
        return 1.0
    return float(np.abs(current - previous).max())
