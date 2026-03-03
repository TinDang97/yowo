# yowo — Library Architecture

Production YOLO inference and export library. Implements native YOLO11 and YOLO26 architectures with edge optimization, multi-backend runtime, automatic hardware selection, and production hardening.

---

## Quick Start

```python
# One-liner: detect objects in an image
from yowo import detect
for det in detect("photo.jpg"):
    for box in det.boxes:
        print(f"{box.class_name}: {box.confidence:.2f} at {box.x1},{box.y1},{box.x2},{box.y2}")

# Full control with InferenceEngine
from yowo import InferenceEngine, open_source

with InferenceEngine(confidence_threshold=0.35) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        process(detection)

# RTSP stream with explicit backend
with InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.MEDIUM,
    backend=BackendType.TENSORRT,
    precision=Precision.FP16,
) as engine:
    for detection in engine.stream(open_source("rtsp://192.168.1.10/live")):
        process(detection)

# Export weights to ONNX
from yowo import export_model, ModelSpec, ExportFormat, Precision
result = export_model(
    ModelSpec(ModelFamily.YOLO26, ModelSize.NANO),
    fmt=ExportFormat.ONNX,
    precision=Precision.FP16,
)
print(result.path)
```

---

## Module Dependency Graph

```
types.py    errors.py          (leaf modules — no yowo imports)
    │            │
    └────────────┴──────────────────────────┐
                                             │
              hardware/                  models/
          (HardwareProfile,          (ModelMeta registry,
           device detection)          weight resolution)
                │                          │
                └─────────────┬────────────┘
                              │
                           backends/
                    (InferenceBackend Protocol,
                     TRT / ONNX / OV / PyTorch)
                              │
                 ┌────────────┴────────────┐
                 │                         │
               io/                   postprocess/
         (FrameSource,            (NMS, box decode,
          preprocess,              coord inverse)
          output sinks)
                 │                         │
                 └────────────┬────────────┘
                              │
                           engine.py
                    (orchestration, batching,
                     lifecycle management)
                              │
                           export/
                    (torch.onnx.export,
                     calibration, metadata)
                              │
                            cli/
                    (Click commands:
                     detect, export, info, models)
```

Arrows represent "imports from". A module only imports from layers strictly above it.

---

## Data Flow

```
Input (path | URL | int)
        │
        ▼
  io/_source.py: open_source()
        │  yields
        ▼
  Frame
    dtype:   uint8
    shape:   (H, W, 3)  — BGR channel order
    source:  original capture
        │
        ▼
  io/_decode.py: preprocess()
        │  returns
        ▼
  PreprocessedTensor
    dtype:   float32
    shape:   (B, 3, H_target, W_target)  — RGB, normalized [0,1]
    meta:    scale_factors, pad_offsets   — for coord recovery
        │
        ▼
  backends/<impl>.infer()
        │  returns
        ▼
  NDArray[float32]
    shape:   (B, 4 + num_classes, num_anchors)  — raw logits
        │
        ▼
  postprocess/_nms.py: postprocess()
        │  returns
        ▼
  list[Detection]
    frame:   original Frame
    boxes:   list[BoundingBox]  — xyxy absolute pixels, original frame space
```

---

## Core Primitives

| Type | Location | Dtype / Shape | Description |
|------|----------|--------------|-------------|
| `Frame` | `types.py` | `uint8 (H, W, 3)` | Raw captured frame, BGR channel order |
| `PreprocessedTensor` | `types.py` | `float32 (B, 3, H, W)` | Letterboxed, normalized, BCHW — ready for inference |
| `Detection` | `types.py` | dataclass | Pairs a `Frame` with its `list[BoundingBox]` |
| `BoundingBox` | `types.py` | dataclass | `x1 y1 x2 y2` absolute pixels, top-left origin, plus `class_name`, `confidence`, `class_id` |
| `ModelSpec` | `types.py` | dataclass | `family` + `size` + optional `weights_path` — model identity |
| `BackendType` | `types.py` | `StrEnum` | `TENSORRT \| ONNX \| OPENVINO \| PYTORCH \| COREML` |
| `Precision` | `types.py` | `StrEnum` | `FP32 \| FP16 \| INT8` |

---

## Public API

Everything importable from `yowo` directly (i.e., exported in `__init__.py`):

```python
# Convenience
from yowo import detect, parse_model_name

# Core engine
from yowo import InferenceEngine, open_source

# Export pipeline
from yowo import export_model, ExportMetadata, ExportResult

# Type primitives
from yowo import (
    Frame, Detection, BoundingBox, ModelSpec,
    BackendType, ExportFormat, Precision,
    ModelFamily, ModelSize, FrameDropPolicy,
)

# Tracking
from yowo import (
    ByteTracker, TrackedBox, TrackedDetection, TrackState,
    track_stream, track_detections,
)

# Counting
from yowo import ObjectCounter, CountLine, CountZone, CountResult

# Multi-stream pipeline
from yowo import BatchScheduler, DetectionRouter, FrameCollector, run_pipeline

# Error hierarchy
from yowo import (
    YowoError,          # base
    ConfigError,        # invalid configuration
    DependencyError,    # missing SDK
    DeviceError,        # GPU/device failure
    ModelNotFoundError, # unknown family/size
    BackendError,       # inference failure
    ExportError,        # export failure
    SourceError,        # bad source path/URL
    InferenceError,     # runtime inference failure
    TrackingError,      # tracking failure
)
```

---

## Error Handling Strategy

All errors inherit from `YowoError`. The hierarchy maps to exit codes in the CLI.

```
YowoError
├── ConfigError         — invalid configuration values
├── DependencyError     — required SDK not installed; message includes pip install command
├── DeviceError         — GPU enumeration or memory allocation failure
├── ModelError          — model-related errors (base)
│   ├── ModelNotFoundError — family/size not in registry or weights not resolvable
│   └── ModelLoadError     — weight loading failure
├── BackendError        — backend lifecycle errors (base)
│   └── BackendLoadError   — backend initialization failure
├── InferenceError      — runtime inference failure
├── ExportError         — export failure or post-export validation failure
│   └── ExportUnsupportedError — format/precision combination not supported
├── SourceError         — source path not found, unreadable stream, unsupported format
│   └── SourceTimeoutError     — RTSP reconnect timeout
├── TrackingError       — object tracking failure
└── ShutdownError       — engine shutdown failure
```

Callers should catch `YowoError` for all-in-one handling or specific subclasses for fine-grained recovery. The engine never swallows errors silently.

---

## Module READMEs

| Module | Responsibility |
|--------|----------------|
| [arch/](arch/README.md) | Native YOLO11 and YOLO26 PyTorch implementations — backbone, FPN-PAN neck, detection head, scaling, weight loading |
| [hardware/](hardware/README.md) | GPU/CPU detection, installed library probing, `HardwareProfile` singleton |
| [models/](models/README.md) | Model family/size registry, weight download and cache |
| [backends/](backends/README.md) | Inference backend Protocol, TensorRT/ONNX/OpenVINO/PyTorch impls, auto-selection |
| [io/](io/README.md) | Source opening (file/RTSP/webcam), letterbox preprocessing, output sinks |
| [postprocess/](postprocess/README.md) | Raw tensor → `Detection` list, NMS, coordinate inverse transform |
| [export/](export/README.md) | PT → ONNX/TensorRT/OpenVINO conversion, calibration, metadata sidecar |
| [cli/](cli/README.md) | Click commands: `detect`, `export`, `info`, `models` |
