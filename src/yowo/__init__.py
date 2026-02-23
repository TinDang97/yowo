"""yowo - Production YOLO inference and export.

Quick start::

    from yowo import InferenceEngine, ModelSpec, ModelFamily, ModelSize, open_source

    spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
    with InferenceEngine(spec) as engine:
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
from yowo.io._source import open_source
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
    GPUArch,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
    PreprocessedTensor,
)

__version__ = "0.1.0"

__all__ = [
    "BackendError",
    "BackendLoadError",
    "BackendSelection",
    "BackendType",
    "BoundingBox",
    "CPUArch",
    "ConfigError",
    "DependencyError",
    "Detection",
    "DeviceError",
    "DeviceType",
    "ExportConfig",
    "ExportError",
    "ExportFormat",
    "ExportMetadata",
    "ExportResult",
    "ExportUnsupportedError",
    "Frame",
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
    "YowoError",
    "__version__",
    "export_model",
    "load_config",
    "open_source",
]
