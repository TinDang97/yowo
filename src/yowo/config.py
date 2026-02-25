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
    YOWO_MAX_MEMORY_MB       -> InferenceConfig.max_memory_mb
    YOWO_RECONNECT_TIMEOUT   -> InferenceConfig.reconnect_timeout_s
    YOWO_FRAME_SKIP          -> InferenceConfig.frame_skip
    YOWO_MAX_FRAMES          -> InferenceConfig.max_frames
    YOWO_FRAME_DROP_POLICY   -> InferenceConfig.frame_drop_policy
    YOWO_MAX_QUEUE_SIZE      -> InferenceConfig.max_queue_size
    YOWO_PREFETCH            -> InferenceConfig.prefetch
    YOWO_PIPELINE_WORKERS    -> InferenceConfig.pipeline_workers
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from yowo.errors import ConfigError
from yowo.types import BackendType, ExportFormat, FrameDropPolicy, ModelFamily, ModelSize, Precision

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
        max_memory_mb: Maximum GPU memory budget in megabytes. ``None``
            means no limit is enforced.
        reconnect_timeout_s: Seconds to wait for stream reconnection before
            raising ``SourceTimeoutError``.
        frame_skip: Skip every N frames; 0 disables skipping.
        max_frames: Stop after processing this many frames. ``None`` runs
            until the source is exhausted.
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
    max_memory_mb: int | None = None
    reconnect_timeout_s: float = 30.0
    frame_skip: int = 0
    max_frames: int | None = None
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
        if self.frame_skip < 0:
            raise ConfigError(f"frame_skip must be >= 0, got {self.frame_skip}")
        if self.reconnect_timeout_s <= 0:
            raise ConfigError(f"reconnect_timeout_s must be > 0, got {self.reconnect_timeout_s}")
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
    if (v := env.get("YOWO_MAX_MEMORY_MB")) is not None:
        cfg.max_memory_mb = int(v)
    if (v := env.get("YOWO_RECONNECT_TIMEOUT")) is not None:
        cfg.reconnect_timeout_s = float(v)
    if (v := env.get("YOWO_FRAME_SKIP")) is not None:
        cfg.frame_skip = int(v)
    if (v := env.get("YOWO_MAX_FRAMES")) is not None:
        cfg.max_frames = int(v)
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
        "max_memory_mb",
        "frame_skip",
        "max_frames",
        "max_queue_size",
        "pipeline_workers",
    ):
        if int_field in data and data[int_field] is not None:
            kwargs[int_field] = int(data[int_field])
        elif int_field in data:
            kwargs[int_field] = None

    for float_field in ("confidence_threshold", "iou_threshold", "reconnect_timeout_s"):
        if float_field in data:
            kwargs[float_field] = float(data[float_field])

    for bool_field in ("prefetch",):
        if bool_field in data:
            kwargs[bool_field] = bool(data[bool_field])

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


__all__ = [
    "ExportConfig",
    "InferenceConfig",
    "load_config",
]
