"""Lightweight frame similarity check for feature map caching."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def frame_similarity(
    current: NDArray[np.float32],
    previous: NDArray[np.float32],
) -> float:
    """Compute mean absolute difference between two preprocessed tensors.

    Both tensors are expected to be BCHW float32 in [0, 1] range.
    Returns a value in [0, 1] where 0 = identical, 1 = maximally different.

    Args:
        current: Current frame as BCHW float32 array.
        previous: Previously cached frame as BCHW float32 array.

    Returns:
        Mean absolute pixel difference (lower = more similar).
    """
    return float(np.abs(current - previous).mean())
