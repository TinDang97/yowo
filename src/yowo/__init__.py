"""yowo - Production YOLO inference and export.

Quick start::

    from yowo import InferenceEngine, ModelSpec, ModelFamily, ModelSize, open_source

    spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
    with InferenceEngine(spec) as engine:
        for detection in engine.stream(open_source("image.jpg")):
            for box in detection.boxes:
                print(f"{box.class_name}: {box.confidence:.2f}")

At this stage only core types, errors, and config are available.
InferenceEngine and open_source will be exported here once implemented.
"""

from yowo.config import ExportConfig, InferenceConfig, load_config
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
    # ---- types ----
    "BackendSelection",
    "BackendType",
    "BoundingBox",
    "CPUArch",
    "ConfigError",
    "DependencyError",
    "Detection",
    "DeviceError",
    "DeviceType",
    # ---- config ----
    "ExportConfig",
    "ExportError",
    "ExportFormat",
    "ExportResult",
    "ExportUnsupportedError",
    "Frame",
    "GPUArch",
    "InferenceConfig",
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
    # ---- errors ----
    "YowoError",
    # Version
    "__version__",
    "load_config",
    # Future exports (not yet implemented):
    # "InferenceEngine",
    # "open_source",
]
