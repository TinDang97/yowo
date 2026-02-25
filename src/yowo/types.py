"""Core data primitives for yowo.

These types flow through every module boundary. No business logic here.
All types are immutable (frozen=True) with slots for memory efficiency.

The one exception is Frame and PreprocessedTensor: numpy arrays are not
hashable and cannot participate in frozen dataclasses. Both are documented
as logically immutable — callers must not mutate their contents.
"""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BackendType(enum.StrEnum):
    """Inference backend identifier."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class DeviceType(enum.StrEnum):
    """Compute device family."""

    CUDA = "cuda"
    CPU = "cpu"


class CPUArch(enum.StrEnum):
    """CPU instruction set architecture."""

    X86_64 = "x86_64"
    AARCH64 = "aarch64"


class GPUArch(enum.StrEnum):
    """NVIDIA GPU compute capability (SM version)."""

    TURING = "sm_75"
    AMPERE = "sm_80"
    AMPERE_GA10X = "sm_86"
    ORIN = "sm_87"
    ADA = "sm_89"
    HOPPER = "sm_90"
    UNKNOWN = "unknown"


class ModelFamily(enum.StrEnum):
    """YOLO model family."""

    YOLO11 = "yolo11"
    YOLO26 = "yolo26"


class ModelSize(enum.StrEnum):
    """YOLO model size variant."""

    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"


class ExportFormat(enum.StrEnum):
    """Target format for model export."""

    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class Precision(enum.StrEnum):
    """Numerical precision for inference or export."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class FrameDropPolicy(enum.StrEnum):
    """Frame backlog policy for live streaming.

    Controls how ThreadedFrameReader handles a full queue.
    """

    NONE = "none"  # Backpressure — process every frame (offline default)
    LATEST = "latest"  # Keep only newest frame (live default)
    SKIP_OLDEST = "skip_oldest"  # Evict oldest when queue full


class StreamState(enum.StrEnum):
    """Health state of a stream managed by FrameCollector."""

    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Frozen dataclasses (pure data, no numpy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Fully qualified model identity.

    Attributes:
        family: YOLO model family (YOLO11 or YOLO26).
        size: Size variant (nano … xlarge).
        task: Ultralytics task string; "detect" for object detection.
        weights_path: Path to a pre-downloaded .pt file, or None to use
            the registry default for this family/size combination.
    """

    family: ModelFamily
    size: ModelSize
    task: str = "detect"
    weights_path: Path | None = None


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned detection bounding box in pixel coordinates.

    Coordinates are in XYXY format relative to the original frame dimensions.

    Attributes:
        x1: Left edge (pixels).
        y1: Top edge (pixels).
        x2: Right edge (pixels).
        y2: Bottom edge (pixels).
        confidence: Detection confidence in [0, 1].
        class_id: Integer class index.
        class_name: Human-readable class label (empty string if unknown).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def area(self) -> float:
        """Box area in square pixels. Returns 0.0 for degenerate boxes."""
        width = max(0.0, self.x2 - self.x1)
        height = max(0.0, self.y2 - self.y1)
        return width * height

    @property
    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Coordinates as a plain (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class Detection:
    """Inference result for a single frame.

    Attributes:
        frame: The source frame that was processed.
        boxes: Tuple of bounding boxes produced by the model.
        inference_time_ms: Wall-clock time for the inference call only,
            excluding preprocessing and postprocessing.
        backend: Backend that produced this result.
        model_spec: Model that produced this result.
    """

    frame: Frame
    boxes: tuple[BoundingBox, ...]
    inference_time_ms: float
    backend: BackendType
    model_spec: ModelSpec

    @property
    def num_boxes(self) -> int:
        """Number of detected bounding boxes."""
        return len(self.boxes)

    @property
    def has_detections(self) -> bool:
        """True when at least one bounding box is present."""
        return len(self.boxes) > 0


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Record of a completed model export operation.

    Attributes:
        model_name: Human-readable model identifier (e.g. ``"yolo26n"``).
        format: Export format that was produced.
        precision: Numerical precision of the exported artifact.
        output_path: Filesystem path to the exported artifact.
        file_size_bytes: Size of the exported artifact on disk.
        export_time_s: Wall-clock seconds taken by the export operation.
        created_at: ISO-8601 timestamp string (UTC) of export completion.
    """

    model_name: str
    format: ExportFormat
    precision: Precision
    output_path: Path
    file_size_bytes: int
    export_time_s: float
    created_at: str


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """Result of hardware-aware backend selection.

    Attributes:
        backend: Chosen backend type.
        device_type: Compute device family.
        precision: Recommended precision for this backend/device pair.
        device_index: CUDA device index; 0 for CPU or single-GPU systems.
        reason: Human-readable explanation of why this backend was chosen.
    """

    backend: BackendType
    device_type: DeviceType
    precision: Precision
    device_index: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Mutable dataclasses (contain numpy arrays — logically immutable)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Frame:
    """A single video/image frame from any input source.

    The ``pixels`` array is BGR uint8 in HWC layout (OpenCV convention).
    This class is *logically immutable*: callers must not modify ``pixels``
    or any other field after construction. It is not ``frozen=True`` solely
    because numpy arrays are not hashable.

    Attributes:
        pixels: HWC BGR uint8 numpy array.
        source_id: Opaque string identifying the input source (path, URL, …).
        frame_index: Zero-based sequential index within the source.
        timestamp_ms: Milliseconds since the source epoch; 0.0 if unknown.
    """

    pixels: NDArray[np.uint8]
    source_id: str = ""
    frame_index: int = 0
    timestamp_ms: float = 0.0

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.pixels.shape[1])

    @property
    def shape_hw(self) -> tuple[int, int]:
        """(height, width) convenience accessor."""
        return (self.height, self.width)


@dataclass(slots=True)
class TaggedFrame:
    """A frame annotated with its owning stream identifier.

    Used by the multi-stream pipeline to track which stream produced
    each frame through batched inference and result routing.

    This class is *logically immutable*: callers must not modify fields
    after construction. It is not ``frozen=True`` because it contains a
    ``Frame`` reference (which itself is not frozen due to numpy arrays).

    Attributes:
        stream_id: Unique identifier for the source stream.
        frame: The underlying video/image frame.
    """

    stream_id: str
    frame: Frame


@dataclass(slots=True)
class PreprocessedTensor:
    """Preprocessed model input ready for inference.

    Produced by the preprocessor from one or more ``Frame`` objects.
    BCHW float32 layout, values normalized to [0, 1].
    This class is *logically immutable*: callers must not modify fields
    after construction. It is not ``frozen=True`` because it contains
    numpy arrays.

    Attributes:
        data: BCHW float32 numpy array.
        original_shapes: Per-image (height, width) before preprocessing.
        input_shape: Model input spatial dimensions (height, width).
        scale_factors: Per-image (scale_h, scale_w) applied during resize.
        pad_offsets: Per-image (pad_top, pad_left) applied during letterbox.
    """

    data: NDArray[np.float32]
    original_shapes: tuple[tuple[int, int], ...]
    input_shape: tuple[int, int]
    scale_factors: tuple[tuple[float, float], ...]
    pad_offsets: tuple[tuple[int, int], ...]

    @property
    def batch_size(self) -> int:
        """Number of images in the batch."""
        return int(self.data.shape[0])


def is_free_threaded() -> bool:
    """Detect if Python is running in free-threaded (no-GIL) mode.

    Returns True on Python 3.13t+ with GIL disabled (PEP 703 Phase II).
    Returns False on standard GIL builds and Python <3.13.
    """
    try:
        return not sys._is_gil_enabled()  # type: ignore[attr-defined]
    except AttributeError:
        return False


# Suppress "unused import" warnings — field is re-exported for subpackages.
__all__ = [
    "BackendSelection",
    "BackendType",
    "BoundingBox",
    "CPUArch",
    "Detection",
    "DeviceType",
    "ExportFormat",
    "ExportResult",
    "Frame",
    "FrameDropPolicy",
    "GPUArch",
    "ModelFamily",
    "ModelSize",
    "ModelSpec",
    "Precision",
    "PreprocessedTensor",
    "StreamState",
    "TaggedFrame",
    "is_free_threaded",
]

# Silence F401 for field — used in submodules that import from types.
_field = field
