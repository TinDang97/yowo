"""Frame preprocessing pipeline.

Converts list[Frame] -> PreprocessedTensor for backend inference.
Implements letterbox resize, gray padding, BGR->RGB, HWC->CHW, and
float32 normalization.

TensorMeta is defined here because it is created during preprocessing
and consumed by postprocess/_nms.py to recover original-frame coordinates.
This avoids a circular dependency through types.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from yowo.types import Frame, PreprocessedTensor

_LETTERBOX_FILL = (114, 114, 114)


@dataclass(frozen=True, slots=True)
class TensorMeta:
    """Transform parameters needed to invert the letterbox preprocessing.

    Attributes:
        scale_factors: Per-image ``(scale_h, scale_w)`` applied during resize.
            Both components are equal (uniform scale).
        pad_offsets: Per-image ``(pad_top, pad_left)`` pixel offsets added
            during letterbox padding.
        original_sizes: Per-image ``(height, width)`` before any preprocessing.
    """

    scale_factors: tuple[tuple[float, float], ...]
    pad_offsets: tuple[tuple[int, int], ...]
    original_sizes: tuple[tuple[int, int], ...]


def preprocess(
    frames: list[Frame],
    target_size: tuple[int, int],
) -> PreprocessedTensor:
    """Letterbox resize + normalize frames to a batched BCHW tensor.

    Steps per frame:
    1. Compute ``scale = min(target_h / frame_h, target_w / frame_w)``.
    2. Compute ``new_h = int(frame_h * scale)``, ``new_w = int(frame_w * scale)``.
    3. Ensure C-contiguous memory layout, then resize with ``cv2.INTER_LINEAR``.
    4. Pad to *target_size* with (114, 114, 114) gray fill.
       ``pad_y = (target_h - new_h) // 2``, ``pad_x = (target_w - new_w) // 2``.
    5. Batch-convert BGR -> RGB, HWC -> BCHW, normalize to float32 in one
       ``cv2.dnn.blobFromImages`` C++ call (replaces per-frame cvtColor +
       transpose + divide).

    All frames stacked on axis 0 -> shape ``(B, 3, H_target, W_target)``.

    Args:
        frames: List of ``Frame`` objects in BGR uint8 HWC format.
        target_size: ``(height, width)`` — the model input spatial dimensions.

    Returns:
        ``PreprocessedTensor`` with ``data`` of shape
        ``(B, 3, target_h, target_w)`` and transform metadata.
    """
    if not frames:
        raise ValueError("frames list must not be empty")

    target_h, target_w = target_size
    padded_frames: list[cv2.typing.MatLike] = []
    scale_factors: list[tuple[float, float]] = []
    pad_offsets: list[tuple[int, int]] = []
    original_shapes: list[tuple[int, int]] = []

    for frame in frames:
        frame_h, frame_w = frame.height, frame.width
        original_shapes.append((frame_h, frame_w))

        scale = min(target_h / frame_h, target_w / frame_w)
        new_h = int(frame_h * scale)
        new_w = int(frame_w * scale)

        # Ensure C-contiguous input before cv2.resize. Guard the fast path:
        # cv2.VideoCapture and cv2.imdecode always return contiguous arrays, so
        # the check is a no-op overhead on the common case.
        pixels = (
            frame.pixels
            if frame.pixels.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray(frame.pixels)
        )
        resized = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_y = (target_h - new_h) // 2
        pad_x = (target_w - new_w) // 2
        pad_bottom = target_h - new_h - pad_y
        pad_right = target_w - new_w - pad_x

        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            pad_bottom,
            pad_x,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=_LETTERBOX_FILL,
        )

        padded_frames.append(padded)  # type: ignore[arg-type]
        scale_factors.append((scale, scale))
        pad_offsets.append((pad_y, pad_x))

    # BGR -> RGB, HWC -> BCHW, normalize: single C++ call replaces per-frame
    # cvtColor + np.transpose + astype + /255.0 + np.stack.
    # type: ignore is needed because cv2 stubs return MatLike, not NDArray.
    batch: NDArray[np.float32] = cv2.dnn.blobFromImages(  # type: ignore[assignment]
        padded_frames, scalefactor=1.0 / 255.0, swapRB=True
    )

    return PreprocessedTensor(
        data=batch,
        original_shapes=tuple(original_shapes),
        input_shape=target_size,
        scale_factors=tuple(scale_factors),
        pad_offsets=tuple(pad_offsets),
    )


def make_tensor_meta(tensor: PreprocessedTensor) -> TensorMeta:
    """Extract TensorMeta from a PreprocessedTensor.

    Convenience helper so postprocess can obtain TensorMeta without
    importing implementation details from _decode directly.
    """
    return TensorMeta(
        scale_factors=tensor.scale_factors,
        pad_offsets=tensor.pad_offsets,
        original_sizes=tensor.original_shapes,
    )


__all__ = [
    "TensorMeta",
    "make_tensor_meta",
    "preprocess",
]
