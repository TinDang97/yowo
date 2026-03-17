"""Core data primitives for yowo.

These types flow through every module boundary. No business logic here.
All types are immutable (frozen=True) with slots for memory efficiency.

The one exception is Frame and PreprocessedTensor: numpy arrays are not
hashable and cannot participate in frozen dataclasses. Both are documented
as logically immutable — callers must not mutate their contents.
"""

from __future__ import annotations

import enum
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# StrEnum compatibility shim (Python 3.9 / 3.10 do not have enum.StrEnum)
# ---------------------------------------------------------------------------

if sys.version_info >= (3, 11):
    StrEnumBase = enum.StrEnum
else:

    class StrEnumBase(str, enum.Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python < 3.11."""

        def __str__(self) -> str:  # match StrEnum: str(v) returns the value
            return self.value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BackendType(StrEnumBase):
    """Inference backend identifier."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"


class DeviceType(StrEnumBase):
    """Compute device family."""

    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class CPUArch(StrEnumBase):
    """CPU instruction set architecture."""

    X86_64 = "x86_64"
    AARCH64 = "aarch64"


class GPUArch(StrEnumBase):
    """NVIDIA GPU compute capability (SM version)."""

    TURING = "sm_75"
    AMPERE = "sm_80"
    AMPERE_GA10X = "sm_86"
    ORIN = "sm_87"
    ADA = "sm_89"
    HOPPER = "sm_90"
    UNKNOWN = "unknown"


class ModelFamily(StrEnumBase):
    """YOLO model family."""

    YOLO11 = "yolo11"
    YOLO26 = "yolo26"


class ModelSize(StrEnumBase):
    """YOLO model size variant."""

    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"


class ExportFormat(StrEnumBase):
    """Target format for model export."""

    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"


class Precision(StrEnumBase):
    """Numerical precision for inference or export."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class FrameDropPolicy(StrEnumBase):
    """Frame backlog policy for live streaming.

    Controls how ThreadedFrameReader handles a full queue.
    """

    NONE = "none"  # Backpressure — process every frame (offline default)
    LATEST = "latest"  # Keep only newest frame (live default)
    SKIP_OLDEST = "skip_oldest"  # Evict oldest when queue full


class StreamState(StrEnumBase):
    """Health state of a stream managed by FrameCollector."""

    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


class HealthStatus(StrEnumBase):
    """Engine health state, derived from runtime metrics and lifecycle.

    Transitions:
        STARTING → READY (after load())
        READY → DEGRADED (error rate ≥ threshold or stream idle >30s)
        READY | DEGRADED → SHUTTING_DOWN (during close())
        SHUTTING_DOWN → CLOSED (after close() completes)
    """

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Stream configuration (defined here alongside HealthStatus for co-location)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamConfig:
    """Per-stream configuration for FrameCollector.

    Controls automatic failure handling and (future) reconnect behaviour.
    All fields are optional and have production-safe defaults.

    Attributes:
        auto_reconnect: Whether to attempt reconnection after stream failure.
            Currently stubbed — field is defined for API stability; reconnect
            loop is not implemented in this release.
        max_consecutive_errors: Number of consecutive read errors before the
            stream is automatically removed from the collector. Default: 3.
        reconnect_backoff_base_s: Initial backoff duration in seconds for
            reconnect attempts. Doubles on each retry up to the max.
            Stubbed for future use. Default: 1.0.
        reconnect_backoff_max_s: Maximum backoff duration in seconds.
            Stubbed for future use. Default: 30.0.
    """

    auto_reconnect: bool = False
    max_consecutive_errors: int = 3
    reconnect_backoff_base_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0


class SourceCategory(StrEnumBase):
    """Input source classification for preset selection."""

    IMAGE = "image"
    VIDEO = "video"
    LIVE_STREAM = "live"


class DeviceCategory(StrEnumBase):
    """Hardware classification for preset selection."""

    CUDA_HIGH = "cuda_high"
    CUDA_LOW = "cuda_low"
    JETSON = "jetson"
    APPLE_SILICON = "apple_silicon"
    CPU_X86 = "cpu_x86"
    CPU_ARM = "cpu_arm"


# ---------------------------------------------------------------------------
# Media format constants (shared by config, io, and export modules)
# ---------------------------------------------------------------------------

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".ts"})
RTSP_SCHEMES: tuple[str, ...] = ("rtsp://", "rtsps://")


# ---------------------------------------------------------------------------
# Frozen dataclasses (pure data, no numpy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Fully qualified model identity.

    Attributes:
        family: YOLO model family (YOLO11 or YOLO26).
        size: Size variant (nano … xlarge).
        task: Ultralytics task string; "detect" for object detection.
        weights_path: Path to a pre-downloaded .pt file, or None to use
            the registry default for this family/size combination.
        num_classes: Override number of output classes. When ``None`` the
            registry default is used (80 for COCO detection, 1000 for
            ImageNet classification).
    """

    family: ModelFamily
    size: ModelSize
    task: str = "detect"
    weights_path: Path | None = None
    num_classes: int | None = None


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize to a plain dict with JSON-safe primitive values."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict.

        Pixel data (``frame.pixels``) and local paths
        (``model_spec.weights_path``) are excluded — only portable metadata
        is included so the result is safe for logging and wire transport.
        """
        return {
            "source_id": self.frame.source_id,
            "frame_index": self.frame.frame_index,
            "timestamp_ms": self.frame.timestamp_ms,
            "inference_time_ms": self.inference_time_ms,
            "backend": str(self.backend),
            "model": f"{self.model_spec.family.value}{self.model_spec.size.value}",
            "boxes": [box.to_dict() for box in self.boxes],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string.

        All values in ``to_dict()`` are JSON-safe primitives, so no custom
        encoder is required.
        """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class OBBBox:
    """Oriented bounding box in (cx, cy, w, h, angle) format.

    angle: rotation in radians, range [-pi/4, 3pi/4] (ultralytics convention).

    Attributes:
        cx: Centre x coordinate (pixels).
        cy: Centre y coordinate (pixels).
        w: Box width in pixels.
        h: Box height in pixels.
        angle: Rotation angle in radians, range [-pi/4, 3pi/4].
        confidence: Detection confidence in [0, 1].
        class_id: Integer class index.
        class_name: Human-readable class label (empty string if unknown).
    """

    cx: float
    cy: float
    w: float
    h: float
    angle: float  # radians, [-pi/4, 3pi/4]
    confidence: float
    class_id: int
    class_name: str = ""

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize to a plain dict with JSON-safe primitive values."""
        return {
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
            "angle": self.angle,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(frozen=True)
class OBBDetection:
    """Inference result for one frame with oriented bounding boxes.

    Attributes:
        frame_index: Zero-based sequential index of the frame in its source.
        source_id: Opaque identifier of the input source; empty if unknown.
        boxes: Tuple of oriented bounding boxes produced by the model.
        frame: The source frame that was processed (``None`` when unavailable).
        inference_time_ms: Wall-clock time for the inference call only,
            excluding preprocessing and postprocessing.
    """

    frame_index: int
    source_id: str
    boxes: tuple[OBBBox, ...]
    frame: Frame | None = None
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict."""
        return {
            "frame_index": self.frame_index,
            "source_id": self.source_id,
            "inference_time_ms": self.inference_time_ms,
            "boxes": [box.to_dict() for box in self.boxes],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class ClassificationResult:
    """Inference result for a single frame from a classification model.

    Attributes:
        top1_class_id: Integer class index with the highest probability.
        top1_score: Probability of the top-1 class in [0, 1].
        topk_class_ids: Top-k class indices sorted descending by score.
        topk_scores: Corresponding probabilities for top-k classes.
        all_probs: Full probability vector (length == num_classes).
        source_id: Opaque identifier of the input source; empty if unknown.
        frame_index: Zero-based sequential index of the frame in its source.
        inference_time_ms: Wall-clock time for the inference call only,
            excluding preprocessing and postprocessing.
        backend: Backend that produced this result.
        model_spec: Model that produced this result.
    """

    top1_class_id: int
    top1_score: float
    topk_class_ids: tuple[int, ...]
    topk_scores: tuple[float, ...]
    all_probs: tuple[float, ...]
    source_id: str
    frame_index: int
    inference_time_ms: float
    backend: BackendType
    model_spec: ModelSpec

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict.

        ``all_probs`` is excluded by default (1000 floats for ImageNet) to
        keep the serialized size manageable for logging and wire transport.
        """
        return {
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "inference_time_ms": self.inference_time_ms,
            "backend": str(self.backend),
            "model": f"{self.model_spec.family.value}{self.model_spec.size.value}-cls",
            "top1_class_id": self.top1_class_id,
            "top1_score": self.top1_score,
            "topk": [
                {"class_id": int(cid), "score": float(sc)}
                for cid, sc in zip(self.topk_class_ids, self.topk_scores)
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass()
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


@dataclass()
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


@dataclass()
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
    "IMAGE_EXTS",
    "RTSP_SCHEMES",
    "VIDEO_EXTS",
    "BackendSelection",
    "BackendType",
    "BoundingBox",
    "CPUArch",
    "ClassificationResult",
    "Detection",
    "DeviceCategory",
    "DeviceType",
    "ExportFormat",
    "ExportResult",
    "Frame",
    "FrameDropPolicy",
    "GPUArch",
    "HealthStatus",
    "ModelFamily",
    "ModelSize",
    "ModelSpec",
    "OBBBox",
    "OBBDetection",
    "Precision",
    "PreprocessedTensor",
    "SourceCategory",
    "StreamConfig",
    "StreamState",
    "TaggedFrame",
    "is_free_threaded",
]

# Silence F401 for field — used in submodules that import from types.
_field = field
