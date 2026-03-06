"""Frame preprocessing pipeline.

Converts list[Frame] -> PreprocessedTensor for backend inference.
Implements letterbox resize, gray padding, BGR->RGB, HWC->CHW, and
float32 normalization.

TensorMeta is defined here because it is created during preprocessing
and consumed by postprocess/_nms.py to recover original-frame coordinates.
This avoids a circular dependency through types.py.
"""

from __future__ import annotations

import threading
from collections import deque
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


class PreprocessBuffer:
    """Reusable staging buffers for letterbox preprocessing.

    Eliminates per-frame cv2.copyMakeBorder allocation by pre-allocating
    per-slot staging arrays filled with the letterbox fill value.

    Tracks last-used resize dimensions per slot so that full memset can be
    skipped when consecutive frames share the same resolution (the common
    case for video streams).
    """

    __slots__ = ("_capacity", "_last_dims", "_staging", "_target_size")

    def __init__(self, max_batch: int, target_size: tuple[int, int]) -> None:
        h, w = target_size
        self._staging: list[NDArray[np.uint8]] = [
            np.full((h, w, 3), _LETTERBOX_FILL[0], dtype=np.uint8) for _ in range(max_batch)
        ]
        self._capacity = max_batch
        self._target_size = target_size
        # (new_h, new_w, pad_y, pad_x) per slot; None means first use.
        self._last_dims: list[tuple[int, int, int, int] | None] = [None] * max_batch

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def target_size(self) -> tuple[int, int]:
        return self._target_size

    @property
    def memory_bytes(self) -> int:
        h, w = self._target_size
        return self._capacity * h * w * 3

    def get_staging(self, index: int) -> NDArray[np.uint8]:
        return self._staging[index]

    def needs_reset(self, index: int, new_h: int, new_w: int, pad_y: int, pad_x: int) -> bool:
        """Return True if the staging slot must be reset before use."""
        dims = (new_h, new_w, pad_y, pad_x)
        if self._last_dims[index] == dims:
            return False
        self._last_dims[index] = dims
        return True


class PreprocessBufferPool:
    """Thread-safe pool of PreprocessBuffer instances for concurrent workers.

    Each worker acquires exclusive access to a buffer via acquire/release.
    Blocks if all buffers are in use (backpressure).
    """

    __slots__ = ("_buffers", "_in_use_count", "_lock", "_sem")

    def __init__(self, pool_size: int, max_batch: int, target_size: tuple[int, int]) -> None:
        self._sem = threading.Semaphore(pool_size)
        self._buffers: deque[PreprocessBuffer] = deque(
            PreprocessBuffer(max_batch, target_size) for _ in range(pool_size)
        )
        self._lock = threading.Lock()
        self._in_use_count: int = 0

    def acquire(self) -> PreprocessBuffer:
        """Acquire a buffer, blocking if none available."""
        self._sem.acquire()
        with self._lock:
            buf = self._buffers.popleft()
            self._in_use_count += 1
            return buf

    def release(self, buf: PreprocessBuffer) -> None:
        """Return a buffer to the pool."""
        with self._lock:
            if self._in_use_count <= 0:
                raise ValueError("Buffer was not acquired from this pool or already released")
            self._in_use_count -= 1
            self._buffers.append(buf)
        self._sem.release()


def _align_to_stride(size: int, stride: int = 32) -> int:
    """Round *size* up to the nearest multiple of *stride*."""
    return ((size + stride - 1) // stride) * stride


def preprocess(
    frames: list[Frame],
    target_size: tuple[int, int],
    *,
    auto_letterbox: bool = False,
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

    When *auto_letterbox* is ``True``, target dimensions are replaced with
    stride-aligned (divisible by 32) scaled dimensions, minimizing padding.
    For 16:9 input this reduces pixel count by ~40%, yielding ~1.4-1.6x
    faster inference.

    Args:
        frames: List of ``Frame`` objects in BGR uint8 HWC format.
        target_size: ``(height, width)`` — the model input spatial dimensions.
        auto_letterbox: Use stride-aligned non-square dimensions instead of
            always padding to *target_size*.

    Returns:
        ``PreprocessedTensor`` with ``data`` of shape
        ``(B, 3, actual_h, actual_w)`` and transform metadata.
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

        if auto_letterbox:
            # Stride-aligned dimensions: minimal padding, non-square tensor.
            actual_h = _align_to_stride(new_h)
            actual_w = _align_to_stride(new_w)
            # Guard: at least one stride cell.
            actual_h = max(actual_h, 32)
            actual_w = max(actual_w, 32)
        else:
            actual_h, actual_w = target_h, target_w

        # Ensure C-contiguous input before cv2.resize. Guard the fast path:
        # cv2.VideoCapture and cv2.imdecode always return contiguous arrays, so
        # the check is a no-op overhead on the common case.
        pixels = (
            frame.pixels
            if frame.pixels.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray(frame.pixels)
        )
        resized = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_y = (actual_h - new_h) // 2
        pad_x = (actual_w - new_w) // 2
        pad_bottom = actual_h - new_h - pad_y
        pad_right = actual_w - new_w - pad_x

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


def preprocess_into(
    frames: list[Frame],
    target_size: tuple[int, int],
    buffer: PreprocessBuffer,
) -> PreprocessedTensor:
    """Like preprocess(), but reuses pre-allocated staging buffers.

    Eliminates cv2.copyMakeBorder allocation by writing into pre-filled
    staging arrays. Output metadata is identical to preprocess().

    Args:
        frames: List of Frame objects (same contract as preprocess()).
        target_size: (height, width) model input size.
        buffer: Pre-allocated buffer with capacity >= len(frames).

    Returns:
        PreprocessedTensor with identical structure to preprocess() output.

    Raises:
        ValueError: If frames is empty or exceeds buffer.capacity.
    """
    if not frames:
        raise ValueError("frames list must not be empty")
    if len(frames) > buffer.capacity:
        raise ValueError(f"batch size {len(frames)} exceeds buffer capacity {buffer.capacity}")
    if buffer.target_size != target_size:
        raise ValueError(
            f"buffer target_size {buffer.target_size} does not match target_size {target_size}"
        )

    target_h, target_w = target_size
    staging_views: list[cv2.typing.MatLike] = []
    scale_factors: list[tuple[float, float]] = []
    pad_offsets: list[tuple[int, int]] = []
    original_shapes: list[tuple[int, int]] = []

    for i, frame in enumerate(frames):
        frame_h, frame_w = frame.height, frame.width
        original_shapes.append((frame_h, frame_w))

        scale = min(target_h / frame_h, target_w / frame_w)
        new_h = int(frame_h * scale)
        new_w = int(frame_w * scale)

        pad_y = (target_h - new_h) // 2
        pad_x = (target_w - new_w) // 2

        # Get pre-allocated staging slot (already filled with 114)
        staging = buffer.get_staging(i)
        # Only reset when frame dimensions changed — skip the 1.2 MB memset
        # on the hot path where consecutive video frames share resolution.
        if buffer.needs_reset(i, new_h, new_w, pad_y, pad_x):
            staging[:] = _LETTERBOX_FILL[0]

        # Ensure C-contiguous input
        pixels = (
            frame.pixels
            if frame.pixels.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray(frame.pixels)
        )
        resized = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Write resized content into pre-allocated staging area
        staging[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        staging_views.append(staging)  # type: ignore[arg-type]
        scale_factors.append((scale, scale))
        pad_offsets.append((pad_y, pad_x))

    batch: NDArray[np.float32] = cv2.dnn.blobFromImages(  # type: ignore[assignment]
        staging_views, scalefactor=1.0 / 255.0, swapRB=True
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
    "PreprocessBuffer",
    "PreprocessBufferPool",
    "TensorMeta",
    "_align_to_stride",
    "make_tensor_meta",
    "preprocess",
    "preprocess_into",
]
