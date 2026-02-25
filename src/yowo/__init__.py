"""yowo - Production YOLO inference and export.

Quick start::

    from yowo import InferenceEngine, open_source

    with InferenceEngine(confidence_threshold=0.35) as engine:
        for detection in engine.stream(open_source("image.jpg")):
            for box in detection.boxes:
                print(f"{box.class_name}: {box.confidence:.2f}")
"""

from yowo.config import ExportConfig, InferenceConfig, load_config
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
    BackendSelection,
    BackendType,
    BoundingBox,
    CPUArch,
    Detection,
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
    StreamState,
    TaggedFrame,
    is_free_threaded,
)

__version__ = "0.1.0"

__all__ = [
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
    "SourceError",
    "SourceTimeoutError",
    "StreamState",
    "TaggedFrame",
    "YowoError",
    "__version__",
    "export_model",
    "is_free_threaded",
    "load_config",
    "open_source",
    "run_pipeline",
]
