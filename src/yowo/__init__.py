"""yowo - Production YOLO inference and export.

Quick start::

    from yowo import InferenceEngine, open_source

    with InferenceEngine(confidence_threshold=0.35) as engine:
        for detection in engine.stream(open_source("image.jpg")):
            for box in detection.boxes:
                print(f"{box.class_name}: {box.confidence:.2f}")
"""

from yowo.config import (
    ExportConfig,
    InferenceConfig,
    classify_device,
    classify_source,
    load_config,
    preset_config,
)
from yowo.engine import InferenceEngine
from yowo.errors import (
    BackendError,
    BackendLoadError,
    ConfigError,
    DependencyError,
    DeviceError,
    ExportError,
    ExportUnsupportedError,
    InferenceError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    SourceError,
    SourceTimeoutError,
    YowoError,
)
from yowo.export import ExportMetadata, export_model
from yowo.io import open_source
from yowo.pipeline import BatchScheduler, DetectionRouter, FrameCollector, run_pipeline
from yowo.types import (
    IMAGE_EXTS,
    RTSP_SCHEMES,
    VIDEO_EXTS,
    BackendSelection,
    BackendType,
    BoundingBox,
    CPUArch,
    Detection,
    DeviceCategory,
    DeviceType,
    ExportFormat,
    ExportResult,
    Frame,
    FrameDropPolicy,
    GPUArch,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
    PreprocessedTensor,
    SourceCategory,
    StreamState,
    TaggedFrame,
    is_free_threaded,
)

__version__ = "1.3.1"

__all__ = [
    "IMAGE_EXTS",
    "RTSP_SCHEMES",
    "VIDEO_EXTS",
    "BackendError",
    "BackendLoadError",
    "BackendSelection",
    "BackendType",
    "BatchScheduler",
    "BoundingBox",
    "CPUArch",
    "ConfigError",
    "DependencyError",
    "Detection",
    "DetectionRouter",
    "DeviceCategory",
    "DeviceError",
    "DeviceType",
    "ExportConfig",
    "ExportError",
    "ExportFormat",
    "ExportMetadata",
    "ExportResult",
    "ExportUnsupportedError",
    "Frame",
    "FrameCollector",
    "FrameDropPolicy",
    "GPUArch",
    "InferenceConfig",
    "InferenceEngine",
    "InferenceError",
    "ModelError",
    "ModelFamily",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelSize",
    "ModelSpec",
    "Precision",
    "PreprocessedTensor",
    "SourceCategory",
    "SourceError",
    "SourceTimeoutError",
    "StreamState",
    "TaggedFrame",
    "YowoError",
    "__version__",
    "classify_device",
    "classify_source",
    "export_model",
    "is_free_threaded",
    "load_config",
    "open_source",
    "preset_config",
    "run_pipeline",
]
