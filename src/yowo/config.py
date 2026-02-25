"""Configuration dataclasses for inference and export sessions.

Load order (last wins): defaults -> YAML file -> environment variables.
Environment variable prefix: YOWO_

Environment variable mapping (all uppercase, prefix YOWO_)::

    YOWO_MODEL_FAMILY        -> InferenceConfig.model_family
    YOWO_MODEL_SIZE          -> InferenceConfig.model_size
    YOWO_BACKEND             -> InferenceConfig.backend
    YOWO_DEVICE              -> InferenceConfig.device
    YOWO_PRECISION           -> InferenceConfig.precision
    YOWO_CONFIDENCE          -> InferenceConfig.confidence_threshold
    YOWO_IOU                 -> InferenceConfig.iou_threshold
    YOWO_BATCH_SIZE          -> InferenceConfig.batch_size
    YOWO_CACHE               -> InferenceConfig.cache
    YOWO_CACHE_DIR           -> InferenceConfig.cache_dir
    YOWO_KV_CACHE            -> InferenceConfig.kv_cache
    YOWO_FRAME_DROP_POLICY   -> InferenceConfig.frame_drop_policy
    YOWO_MAX_QUEUE_SIZE      -> InferenceConfig.max_queue_size
    YOWO_PREFETCH            -> InferenceConfig.prefetch
    YOWO_PIPELINE_WORKERS    -> InferenceConfig.pipeline_workers
"""

from __future__ import annotations

import dataclasses as _dc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from yowo.errors import ConfigError
from yowo.types import (
    BackendType,
    CPUArch,
    DeviceCategory,
    ExportFormat,
    FrameDropPolicy,
    ModelFamily,
    ModelSize,
    Precision,
    SourceCategory,
)

if TYPE_CHECKING:
    from yowo.hardware import HardwareProfile

# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Runtime configuration for the inference engine.

    Validation is performed in ``__post_init__``. All fields may be
    overridden via environment variables after loading from YAML.

    Attributes:
        model_family: YOLO model family to use.
        model_size: Size variant of the model.
        weights_path: Optional path to a local .pt weights file. When
            ``None`` the registry resolves the path automatically.
        backend: Inference backend. ``None`` triggers automatic selection
            based on available hardware and installed packages.
        device: Device string (``"auto"``, ``"cuda"``, ``"cpu"``,
            ``"cuda:0"``, …). ``"auto"`` delegates to the hardware module.
        precision: Numerical precision. ``None`` triggers automatic
            selection (FP16 on CUDA, FP32 on CPU).
        confidence_threshold: Minimum confidence score to keep a detection.
            Must be in [0.0, 1.0].
        iou_threshold: NMS IoU threshold. Must be in [0.0, 1.0].
        batch_size: Number of frames per inference batch. Must be >= 1.
        cache: Enable in-memory feature map caching (PyTorch backend only).
        cache_dir: Feature map mmap cache directory. When set, enables
            mmap-backed caching regardless of ``cache``.
        kv_cache: Enable attention KV cache for streaming inference
            (PyTorch backend only).
        frame_drop_policy: Backlog policy for ThreadedFrameReader when the
            queue is full. ``NONE`` applies backpressure (offline default);
            ``LATEST`` keeps only the newest frame (live default).
        max_queue_size: Bounded queue depth for ThreadedFrameReader. Must
            be >= 1.
        prefetch: Enable threaded frame prefetch in ``stream()``.
        pipeline_workers: Worker thread count for the pipeline. ``0`` means
            auto-detect (2 on free-threaded Python, 1 otherwise).
    """

    model_family: ModelFamily = ModelFamily.YOLO26
    model_size: ModelSize = ModelSize.NANO
    weights_path: Path | None = None
    backend: BackendType | None = None
    device: str = "auto"
    precision: Precision | None = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    batch_size: int = 1
    cache: bool = False
    cache_dir: Path | None = None
    kv_cache: bool = False
    frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST
    max_queue_size: int = 2
    prefetch: bool = True
    pipeline_workers: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ConfigError(f"iou_threshold must be in [0.0, 1.0], got {self.iou_threshold}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_queue_size < 1:
            raise ConfigError(f"max_queue_size must be >= 1, got {self.max_queue_size}")
        if self.pipeline_workers < 0:
            raise ConfigError(f"pipeline_workers must be >= 0, got {self.pipeline_workers}")


# ---------------------------------------------------------------------------
# ExportConfig
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for a model export operation.

    Validation is performed in ``__post_init__``.

    Attributes:
        model_family: YOLO model family to export.
        model_size: Size variant to export.
        weights_path: Optional path to a local .pt weights file.
        target_format: Output format for the exported artifact.
        precision: Numerical precision of the exported artifact.
        dynamic_batch: Enable dynamic batch dimension in the ONNX graph.
            Has no effect for TensorRT or OpenVINO exports.
        output_dir: Directory where exported artifacts are written.
            Defaults to ``~/.yowo/models``.
        imgsz: Input image size (square). Must match training configuration.
        calibration_data: Path to calibration dataset required for INT8
            quantization. Must be provided when ``precision == INT8``.
    """

    model_family: ModelFamily = ModelFamily.YOLO26
    model_size: ModelSize = ModelSize.NANO
    weights_path: Path | None = None
    target_format: ExportFormat = ExportFormat.ONNX
    precision: Precision = Precision.FP16
    dynamic_batch: bool = False
    output_dir: Path = field(default_factory=lambda: Path.home() / ".yowo" / "models")
    imgsz: int = 640
    calibration_data: str | None = None

    def __post_init__(self) -> None:
        if self.precision == Precision.INT8 and self.calibration_data is None:
            raise ConfigError(
                "calibration_data must be provided when precision is INT8. "
                "Pass the path to a directory of calibration images."
            )
        if self.imgsz <= 0:
            raise ConfigError(f"imgsz must be > 0, got {self.imgsz}")


# ---------------------------------------------------------------------------
# YAML loader and env-var override
# ---------------------------------------------------------------------------


def _apply_env_overrides(cfg: InferenceConfig) -> None:
    """Mutate *cfg* in-place using YOWO_ environment variables."""
    env = os.environ

    if (v := env.get("YOWO_MODEL_FAMILY")) is not None:
        cfg.model_family = ModelFamily(v)
    if (v := env.get("YOWO_MODEL_SIZE")) is not None:
        cfg.model_size = ModelSize(v)
    if (v := env.get("YOWO_WEIGHTS_PATH")) is not None:
        cfg.weights_path = Path(v)
    if (v := env.get("YOWO_BACKEND")) is not None:
        cfg.backend = BackendType(v)
    if (v := env.get("YOWO_DEVICE")) is not None:
        cfg.device = v
    if (v := env.get("YOWO_PRECISION")) is not None:
        cfg.precision = Precision(v)
    if (v := env.get("YOWO_CONFIDENCE")) is not None:
        cfg.confidence_threshold = float(v)
    if (v := env.get("YOWO_IOU")) is not None:
        cfg.iou_threshold = float(v)
    if (v := env.get("YOWO_BATCH_SIZE")) is not None:
        cfg.batch_size = int(v)
    if (v := env.get("YOWO_CACHE")) is not None:
        cfg.cache = v.lower() in ("true", "1", "yes")
    if (v := env.get("YOWO_CACHE_DIR")) is not None:
        cfg.cache_dir = Path(v)
    if (v := env.get("YOWO_KV_CACHE")) is not None:
        cfg.kv_cache = v.lower() in ("true", "1", "yes")
    if (v := env.get("YOWO_FRAME_DROP_POLICY")) is not None:
        cfg.frame_drop_policy = FrameDropPolicy(v)
    if (v := env.get("YOWO_MAX_QUEUE_SIZE")) is not None:
        cfg.max_queue_size = int(v)
    if (v := env.get("YOWO_PREFETCH")) is not None:
        cfg.prefetch = v.lower() in ("true", "1", "yes")
    if (v := env.get("YOWO_PIPELINE_WORKERS")) is not None:
        cfg.pipeline_workers = int(v)


def _dict_to_inference_config(data: dict[str, Any]) -> InferenceConfig:
    """Convert a raw YAML dict to an InferenceConfig, coercing enum strings."""
    kwargs: dict[str, Any] = {}

    if "model_family" in data:
        kwargs["model_family"] = ModelFamily(data["model_family"])
    if "model_size" in data:
        kwargs["model_size"] = ModelSize(data["model_size"])
    if "weights_path" in data and data["weights_path"] is not None:
        kwargs["weights_path"] = Path(data["weights_path"])
    if "backend" in data and data["backend"] is not None:
        kwargs["backend"] = BackendType(data["backend"])
    if "device" in data:
        kwargs["device"] = str(data["device"])
    if "precision" in data and data["precision"] is not None:
        kwargs["precision"] = Precision(data["precision"])

    if "frame_drop_policy" in data and data["frame_drop_policy"] is not None:
        kwargs["frame_drop_policy"] = FrameDropPolicy(data["frame_drop_policy"])

    # Numeric scalars — copy as-is with type coercion for safety.
    for int_field in (
        "batch_size",
        "max_queue_size",
        "pipeline_workers",
    ):
        if int_field in data and data[int_field] is not None:
            kwargs[int_field] = int(data[int_field])

    for float_field in ("confidence_threshold", "iou_threshold"):
        if float_field in data:
            kwargs[float_field] = float(data[float_field])

    for bool_field in ("prefetch", "cache", "kv_cache"):
        if bool_field in data:
            kwargs[bool_field] = bool(data[bool_field])

    if "cache_dir" in data and data["cache_dir"] is not None:
        kwargs["cache_dir"] = Path(data["cache_dir"])

    return InferenceConfig(**kwargs)


def load_config(path: Path | None = None) -> InferenceConfig:
    """Build an InferenceConfig from optional YAML file and env vars.

    Load order (later entries win):
    1. Hard-coded dataclass defaults.
    2. YAML file at *path* (if provided and exists).
    3. ``YOWO_*`` environment variables.

    Args:
        path: Path to a YAML configuration file. Ignored when ``None``
            or when the file does not exist.

    Returns:
        Validated InferenceConfig instance.

    Raises:
        ConfigError: When the YAML file contains invalid values or the
            resulting configuration fails validation.
    """
    if path is not None and path.exists():
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse config file {path}: {exc}") from exc

        try:
            cfg = _dict_to_inference_config(raw)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"Invalid value in config file {path}: {exc}") from exc
    else:
        cfg = InferenceConfig()

    try:
        _apply_env_overrides(cfg)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid YOWO_ environment variable: {exc}") from exc

    # Re-validate after env overrides may have changed values.
    try:
        cfg.__post_init__()
    except ConfigError:
        raise

    return cfg


# ---------------------------------------------------------------------------
# Preset inference: source classification
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".ts"})
_RTSP_SCHEMES = ("rtsp://", "rtsps://")


def classify_source(source: str | Path) -> SourceCategory:
    """Classify a source string into a SourceCategory without opening it.

    Uses the same dispatch rules as ``open_source()`` in ``yowo.io``.

    Args:
        source: File path, URL string, webcam index string, or directory.

    Returns:
        The matching SourceCategory.

    Raises:
        ConfigError: If the source type cannot be determined.
    """
    source_str = str(source)

    # Webcam: pure digit string -> live
    if isinstance(source, str) and source.isdigit():
        return SourceCategory.LIVE_STREAM

    # RTSP -> live
    if source_str.startswith(_RTSP_SCHEMES):
        return SourceCategory.LIVE_STREAM

    path = Path(source_str)
    suffix = path.suffix.lower()

    # Existing directory -> image batch
    if path.is_dir():
        return SourceCategory.IMAGE

    if suffix in _IMAGE_EXTS:
        return SourceCategory.IMAGE

    if suffix in _VIDEO_EXTS:
        return SourceCategory.VIDEO

    raise ConfigError(
        f"Cannot classify source type for: {source!r}. "
        f"Supported: image files {sorted(_IMAGE_EXTS)}, "
        f"video files {sorted(_VIDEO_EXTS)}, "
        f'RTSP URLs (rtsp://), webcam indices ("0", "1", ...).'
    )


# ---------------------------------------------------------------------------
# Preset inference: device classification
# ---------------------------------------------------------------------------

_VRAM_HIGH_THRESHOLD_MB = 8192


def classify_device(hw: HardwareProfile) -> DeviceCategory:
    """Classify hardware into a DeviceCategory for preset selection.

    Priority order: Jetson > CUDA_HIGH > CUDA_LOW > APPLE_SILICON > CPU_X86 > CPU_ARM.

    Args:
        hw: Hardware profile snapshot.

    Returns:
        The matching DeviceCategory.
    """
    if hw.is_jetson:
        return DeviceCategory.JETSON

    if hw.has_nvidia_gpu:
        gpu = hw.primary_gpu
        assert gpu is not None  # guarded by has_nvidia_gpu
        if gpu.memory_total_mb >= _VRAM_HIGH_THRESHOLD_MB:
            return DeviceCategory.CUDA_HIGH
        return DeviceCategory.CUDA_LOW

    if hw.libraries.onnxruntime_has_coreml and hw.cpu.cpu_arch == CPUArch.AARCH64:
        return DeviceCategory.APPLE_SILICON

    if hw.cpu.cpu_arch == CPUArch.X86_64:
        return DeviceCategory.CPU_X86

    return DeviceCategory.CPU_ARM


# ---------------------------------------------------------------------------
# Preset inference: overrides dataclass + lookup table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PresetOverrides:
    """Pipeline/caching knobs that differ from InferenceConfig defaults."""

    batch_size: int | None = None
    cache: bool | None = None
    kv_cache: bool | None = None
    prefetch: bool | None = None
    frame_drop_policy: FrameDropPolicy | None = None
    max_queue_size: int | None = None
    pipeline_workers: int | None = None
    confidence_threshold: float | None = None


_DC = DeviceCategory
_SC = SourceCategory
_P = _PresetOverrides
_LATEST = FrameDropPolicy.LATEST

_PRESET_TABLE: dict[tuple[DeviceCategory, SourceCategory], _PresetOverrides] = {
    # -- CUDA HIGH (>= 8 GB VRAM) --
    (_DC.CUDA_HIGH, _SC.IMAGE): _P(batch_size=1, prefetch=False),
    (_DC.CUDA_HIGH, _SC.VIDEO): _P(
        batch_size=4,
        cache=True,
        prefetch=True,
    ),
    (_DC.CUDA_HIGH, _SC.LIVE_STREAM): _P(
        batch_size=1,
        kv_cache=True,
        frame_drop_policy=_LATEST,
        max_queue_size=4,
    ),
    # -- CUDA LOW (< 8 GB VRAM) --
    (_DC.CUDA_LOW, _SC.IMAGE): _P(batch_size=1, prefetch=False),
    (_DC.CUDA_LOW, _SC.VIDEO): _P(batch_size=2, prefetch=True),
    (_DC.CUDA_LOW, _SC.LIVE_STREAM): _P(
        batch_size=1,
        kv_cache=True,
        frame_drop_policy=_LATEST,
        max_queue_size=2,
    ),
    # -- JETSON --
    (_DC.JETSON, _SC.IMAGE): _P(batch_size=1, prefetch=False),
    (_DC.JETSON, _SC.VIDEO): _P(batch_size=1, prefetch=True),
    (_DC.JETSON, _SC.LIVE_STREAM): _P(
        batch_size=1,
        kv_cache=True,
        frame_drop_policy=_LATEST,
        max_queue_size=2,
    ),
    # -- APPLE SILICON (CoreML) --
    (_DC.APPLE_SILICON, _SC.IMAGE): _P(batch_size=1, prefetch=False),
    (_DC.APPLE_SILICON, _SC.VIDEO): _P(
        batch_size=2,
        cache=True,
        prefetch=True,
    ),
    (_DC.APPLE_SILICON, _SC.LIVE_STREAM): _P(
        batch_size=1,
        kv_cache=True,
        frame_drop_policy=_LATEST,
        max_queue_size=2,
    ),
    # -- CPU x86 --
    (_DC.CPU_X86, _SC.IMAGE): _P(batch_size=1, prefetch=False),
    (_DC.CPU_X86, _SC.VIDEO): _P(batch_size=2, prefetch=True),
    (_DC.CPU_X86, _SC.LIVE_STREAM): _P(
        batch_size=1,
        frame_drop_policy=_LATEST,
        max_queue_size=2,
    ),
    # -- CPU ARM (no CoreML) --
    (_DC.CPU_ARM, _SC.IMAGE): _P(batch_size=1, prefetch=False),
    (_DC.CPU_ARM, _SC.VIDEO): _P(batch_size=1, prefetch=True),
    (_DC.CPU_ARM, _SC.LIVE_STREAM): _P(
        batch_size=1,
        frame_drop_policy=_LATEST,
        max_queue_size=2,
    ),
}


# ---------------------------------------------------------------------------
# Preset inference: factory function
# ---------------------------------------------------------------------------

_INFERENCE_CONFIG_FIELDS = frozenset(f.name for f in _dc.fields(InferenceConfig))


def preset_config(
    hw: HardwareProfile,
    source_type: SourceCategory,
    **overrides: Any,
) -> InferenceConfig:
    """Build a device+source-optimized InferenceConfig.

    Looks up optimal pipeline/caching knobs for the ``(device, source)``
    combination, then applies any explicit ``**overrides`` on top.

    Backend, device, and precision are **not** set by presets -- those
    are resolved later by ``select_backend()``.

    Args:
        hw: Hardware profile snapshot.
        source_type: Classified source category.
        **overrides: Any ``InferenceConfig`` field name. Values replace
            preset values. Unknown keys raise ``ConfigError``.

    Returns:
        A validated ``InferenceConfig`` with preset + override values.

    Raises:
        ConfigError: If an unknown override key is passed.
    """
    bad_keys = set(overrides) - _INFERENCE_CONFIG_FIELDS
    if bad_keys:
        raise ConfigError(f"Unknown InferenceConfig fields: {sorted(bad_keys)}")

    device_cat = classify_device(hw)
    preset = _PRESET_TABLE[(device_cat, source_type)]

    # Start from preset values (only non-None fields)
    kwargs: dict[str, Any] = {}
    for f in _dc.fields(preset):
        val = getattr(preset, f.name)
        if val is not None:
            kwargs[f.name] = val

    # Explicit overrides win
    kwargs.update(overrides)

    return InferenceConfig(**kwargs)


__all__ = [
    "ExportConfig",
    "InferenceConfig",
    "classify_device",
    "classify_source",
    "load_config",
    "preset_config",
]
