"""yowo - Production YOLO inference and export.

Quick start::

    from yowo import detect
    for det in detect("image.jpg"):
        for box in det.boxes:
            print(f"{box.class_name}: {box.confidence:.2f}")
"""

from yowo._convenience import classify, detect, detect_obb, parse_model_name
from yowo.backends import ModelBuilder
from yowo.classify_engine import ClassificationEngine
from yowo.config import (
    ClassificationConfig,
    InferenceConfig,
    OBBConfig,
    load_config,
    preset_config,
)
from yowo.config import (
    ExportConfig as ExportConfig,
)
from yowo.config import (
    classify_device as classify_device,
)
from yowo.config import (
    classify_source as classify_source,
)
from yowo.counter import (
    CountLine,
    CountResult,
    CountZone,
    CrossDirection,
    LineCrossEvent,
    ObjectCounter,
)
from yowo.engine import DetectionEngine, HealthReport, InferenceEngine
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
    ShutdownError,
    SourceError,
    SourceTimeoutError,
    TrackingError,
    WarmupValidationError,
    YowoError,
)
from yowo.events import EventBus as EventBus
from yowo.export import ExportMetadata, export_model
from yowo.io import open_source
from yowo.metrics import EngineMetrics as EngineMetrics
from yowo.metrics import MetricsCollector as MetricsCollector
from yowo.obb_engine import OBBEngine
from yowo.pipeline import BatchScheduler, DetectionRouter, FrameCollector, run_pipeline
from yowo.tracking import (
    ByteTracker,
    TrackedBox,
    TrackedDetection,
    TrackState,
    track_detections,
    track_stream,
)
from yowo.types import (
    IMAGE_EXTS as IMAGE_EXTS,
)
from yowo.types import (
    RTSP_SCHEMES as RTSP_SCHEMES,
)
from yowo.types import (
    VIDEO_EXTS as VIDEO_EXTS,
)
from yowo.types import (
    BackendSelection,
    BackendType,
    BoundingBox,
    ClassificationResult,
    Detection,
    ExportFormat,
    ExportResult,
    Frame,
    FrameDropPolicy,
    HealthStatus,
    ModelFamily,
    ModelSize,
    ModelSpec,
    OBBBox,
    OBBDetection,
    Precision,
    StreamConfig,
    StreamState,
    TaggedFrame,
    is_free_threaded,
)
from yowo.types import (
    CPUArch as CPUArch,
)
from yowo.types import (
    DeviceCategory as DeviceCategory,
)
from yowo.types import (
    DeviceType as DeviceType,
)
from yowo.types import (
    GPUArch as GPUArch,
)
from yowo.types import (
    PreprocessedTensor as PreprocessedTensor,
)
from yowo.types import (
    SourceCategory as SourceCategory,
)

__version__ = "2.4.1"

__all__ = [
    "BackendError",
    "BackendLoadError",
    "BackendSelection",
    "BackendType",
    "BatchScheduler",
    "BoundingBox",
    "ByteTracker",
    "ClassificationConfig",
    "ClassificationEngine",
    "ClassificationResult",
    "ConfigError",
    "CountLine",
    "CountResult",
    "CountZone",
    "CrossDirection",
    "DependencyError",
    "Detection",
    "DetectionEngine",
    "DetectionRouter",
    "DeviceError",
    "ExportError",
    "ExportFormat",
    "ExportMetadata",
    "ExportResult",
    "ExportUnsupportedError",
    "Frame",
    "FrameCollector",
    "FrameDropPolicy",
    "HealthReport",
    "HealthStatus",
    "InferenceConfig",
    "InferenceEngine",
    "InferenceError",
    "LineCrossEvent",
    "ModelBuilder",
    "ModelError",
    "ModelFamily",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelSize",
    "ModelSpec",
    "OBBBox",
    "OBBConfig",
    "OBBDetection",
    "OBBEngine",
    "ObjectCounter",
    "Precision",
    "ShutdownError",
    "SourceError",
    "SourceTimeoutError",
    "StreamConfig",
    "StreamState",
    "TaggedFrame",
    "TrackState",
    "TrackedBox",
    "TrackedDetection",
    "TrackingError",
    "WarmupValidationError",
    "YowoError",
    "__version__",
    "classify",
    "detect",
    "detect_obb",
    "export_model",
    "is_free_threaded",
    "load_config",
    "open_source",
    "parse_model_name",
    "preset_config",
    "run_pipeline",
    "track_detections",
    "track_stream",
]
