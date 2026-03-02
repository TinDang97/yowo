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
from yowo.counter import (
    CountLine,
    CountResult,
    CountZone,
    CrossDirection,
    LineCrossEvent,
    ObjectCounter,
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
    ShutdownError,
    SourceError,
    SourceTimeoutError,
    TrackingError,
    YowoError,
)
from yowo.events import EventBus
from yowo.export import ExportMetadata, export_model
from yowo.io import open_source
from yowo.metrics import EngineMetrics, MetricsCollector
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
    HealthStatus,
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

__version__ = "2.2.1"

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
    "ByteTracker",
    "CPUArch",
    "ConfigError",
    "CountLine",
    "CountResult",
    "CountZone",
    "CrossDirection",
    "DependencyError",
    "Detection",
    "DetectionRouter",
    "DeviceCategory",
    "DeviceError",
    "DeviceType",
    "EngineMetrics",
    "EventBus",
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
    "HealthStatus",
    "InferenceConfig",
    "InferenceEngine",
    "InferenceError",
    "LineCrossEvent",
    "MetricsCollector",
    "ModelError",
    "ModelFamily",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelSize",
    "ModelSpec",
    "ObjectCounter",
    "Precision",
    "PreprocessedTensor",
    "ShutdownError",
    "SourceCategory",
    "SourceError",
    "SourceTimeoutError",
    "StreamState",
    "TaggedFrame",
    "TrackState",
    "TrackedBox",
    "TrackedDetection",
    "TrackingError",
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
    "track_detections",
    "track_stream",
]
