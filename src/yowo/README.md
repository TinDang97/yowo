# yowo — Library Architecture

Production YOLO inference and export library. Wraps ultralytics to add edge optimization, multi-backend runtime, automatic hardware selection, and production hardening.

---

## Quick Start

```python
from yowo import Engine, ModelSpec, BackendType

# Detect on a single image
engine = Engine(ModelSpec(family="yolo26", size="n"))
detections = engine.run("photo.jpg")
for det in detections:
    for box in det.boxes:
        print(f"{box.class_name}: {box.confidence:.2f} at {box.x1},{box.y1},{box.x2},{box.y2}")

# Detect on an RTSP stream with explicit backend
engine = Engine(
    ModelSpec(family="yolo26", size="m"),
    backend=BackendType.TENSORRT,
    precision="fp16",
)
for detection in engine.stream("rtsp://192.168.1.10/live"):
    process(detection)

# Export weights to TensorRT FP16
from yowo import export_model
result = export_model(
    ModelSpec(family="yolo26", size="n"),
    target_format="tensorrt",
    precision="fp16",
    output_dir="~/.yowo/engines/",
)
print(result.file_path)
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
                    (ultralytics wrapper,
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
| `BackendType` | `types.py` | `StrEnum` | `TENSORRT \| ONNX \| OPENVINO \| PYTORCH` |
| `Precision` | `types.py` | `StrEnum` | `FP32 \| FP16 \| INT8` |

---

## Public API

Everything importable from `yowo` directly (i.e., exported in `__init__.py`):

```python
# Core engine
from yowo import Engine

# Export pipeline
from yowo import export_model, ExportResult

# Type primitives
from yowo import (
    Frame,
    PreprocessedTensor,
    Detection,
    BoundingBox,
    ModelSpec,
    BackendType,
    Precision,
)

# Hardware introspection
from yowo import get_hardware_profile, HardwareProfile

# Model registry
from yowo import list_models, get_model_meta

# Error hierarchy
from yowo import (
    YowoError,          # base
    DependencyError,    # missing SDK
    HardwareError,      # GPU/device failure
    ModelNotFoundError, # unknown family/size
    BackendError,       # inference failure
    ExportError,        # export failure
    InputError,         # bad source path/URL
)
```

---

## Error Handling Strategy

All errors inherit from `YowoError`. The hierarchy maps to exit codes in the CLI.

```
YowoError
├── DependencyError    — required SDK not installed; message includes pip install command
├── HardwareError      — GPU enumeration or memory allocation failure
├── ModelNotFoundError — family/size not in registry or weights not resolvable
├── BackendError       — load, infer, or unload failure; wraps underlying SDK exception
├── ExportError        — ultralytics export failure or post-export validation failure
└── InputError         — source path not found, unreadable stream, unsupported format
```

Callers should catch `YowoError` for all-in-one handling or specific subclasses for fine-grained recovery. The engine never swallows errors silently.

---

## Module READMEs

| Module | Responsibility |
|--------|----------------|
| [hardware/](hardware/README.md) | GPU/CPU detection, installed library probing, `HardwareProfile` singleton |
| [models/](models/README.md) | Model family/size registry, weight download and cache |
| [backends/](backends/README.md) | Inference backend Protocol, TensorRT/ONNX/OpenVINO/PyTorch impls, auto-selection |
| [io/](io/README.md) | Source opening (file/RTSP/webcam), letterbox preprocessing, output sinks |
| [postprocess/](postprocess/README.md) | Raw tensor → `Detection` list, NMS, coordinate inverse transform |
| [export/](export/README.md) | PT → ONNX/TensorRT/OpenVINO conversion, calibration, metadata sidecar |
| [cli/](cli/README.md) | Click commands: `detect`, `export`, `info`, `models` |
