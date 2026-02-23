# yowo

> Production YOLO inference and export — hardware-aware, multi-backend, edge-ready.

yowo wraps [ultralytics](https://github.com/ultralytics/ultralytics) for inference and export while adding what production deployments need: automatic hardware detection, transparent backend selection, graceful degradation, and stream resilience.

---

## Install

```bash
# Core (PyTorch backend, CPU inference)
pip install yowo

# ONNX Runtime — CPU inference (ARM, x86)
pip install yowo[onnx]

# ONNX Runtime — CUDA inference (NVIDIA GPU)
pip install yowo[onnx-gpu]

# OpenVINO — Intel CPU/iGPU
pip install yowo[openvino]

# Everything (ONNX GPU + OpenVINO)
pip install yowo[all]

# TensorRT — requires Linux + NVIDIA GPU (manual step)
pip install tensorrt>=10.0 --extra-index-url https://pypi.nvidia.com
```

**Requirements**: Python >=3.11, Linux (production) / macOS (development)

---

## Quick Start

### CLI

```bash
# Auto-detect hardware and run inference
yowo detect image.jpg

# Use a specific model
yowo detect video.mp4 --model yolo12n

# Use a local weights file (skips download)
yowo detect image.jpg --model yolo26n --weights /path/to/YOLO26.pt

# RTSP stream
yowo detect rtsp://camera-ip:554/stream --model yolo26n --confidence 0.4

# Save detections to JSON
yowo detect ./images/ --model yolo11s --output detections.json

# Show hardware and installed backends
yowo info

# List all registered model variants
yowo models
```

### Python API

```python
from yowo import InferenceEngine, ModelSpec, ModelFamily, ModelSize, open_source

# Minimal: auto-select everything
spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
with InferenceEngine(spec) as engine:
    for detection in engine.stream(open_source("image.jpg")):
        for box in detection.boxes:
            print(f"{box.class_name}: {box.confidence:.2f} @ {box.as_xyxy()}")
```

---

## Real-World Example — Hanoi Traffic Surveillance

Detection run on a 965×539 Hanoi traffic surveillance screenshot using YOLO26 on CPU (Apple M4 Pro):

```bash
yowo detect "Hanoi AI Cameras Traffic Violations.webp" \
  --model yolo26n \
  --weights "Ultralytics YOLO26.pt" \
  --backend pytorch \
  --confidence 0.25 \
  --output detections.json
```

```
Frame 0: 29 detections (582.2ms)
Saved detections to detections.json
```

**Detection results** (sorted by confidence):

| Class | Confidence | Bounding Box (x1,y1,x2,y2) |
|-------|-----------|----------------------------|
| car | 0.888 | (387, 422, 622, 537) |
| car | 0.884 | (418, 151, 567, 300) |
| car | 0.839 | (250, 190, 402, 339) |
| car | 0.820 | (415, 269, 598, 447) |
| car | 0.685 | (427, 89, 555, 197) |
| motorcycle | 0.680 | (879, 384, 945, 499) |
| motorcycle | 0.679 | (713, 407, 781, 527) |
| car | 0.668 | (171, 251, 357, 451) |
| motorcycle | 0.573 | (777, 373, 839, 476) |
| motorcycle | 0.525 | (823, 449, 899, 536) |
| person | 0.500 | (759, 449, 844, 539) |
| … 18 more | 0.26–0.47 | motorcycles, persons, trucks, bus |

**Summary**: 29 objects — 9 cars, 9 persons, 6 motorcycles, 2 trucks, 1 bus, 2 overlapping detections — in **582ms** on CPU. YOLO26's NMS-free head eliminates the NMS step; detections are post-filtered by confidence only.

The full JSON output per detection:

```json
{
  "frame_index": 0,
  "source_id": "Hanoi AI Cameras Traffic Violations.webp",
  "inference_time_ms": 582.2,
  "backend": "pytorch",
  "model": "yolo26n",
  "boxes": [
    {
      "x1": 387.0, "y1": 422.0, "x2": 622.0, "y2": 537.0,
      "confidence": 0.888,
      "class_id": 2,
      "class_name": "car"
    }
  ]
}
```

---

## Models

| Name | Alias | Notes |
|------|-------|-------|
| `yolo11n/s/m/l/x` | YOLO11 | Stable, best production baseline |
| `yolo12n/s/m/l/x` | YOLO12 | Attention-based, better accuracy |
| `yolo26n/s/m/l/x` | YOLO26 | NMS-free, best CPU and INT8 speed |

Weights are downloaded automatically to `~/.cache/yowo/weights/` on first use.

---

## Backends

yowo selects the best available backend automatically. You can override.

| Backend | Format | When used |
|---------|--------|-----------|
| TensorRT | `.engine` | NVIDIA GPU + TensorRT installed |
| ONNX Runtime (CUDA) | `.onnx` | NVIDIA GPU + onnxruntime-gpu |
| OpenVINO | `_openvino_model/` | Intel CPU/iGPU + openvino |
| ONNX Runtime (CPU) | `.onnx` | Any CPU + onnxruntime |
| PyTorch | `.pt` | Universal fallback |

**Priority chain**: TensorRT → ONNX (CUDA) → OpenVINO → ONNX (CPU) → PyTorch

If a backend fails to load, yowo falls back to the next in chain and logs a warning — it never crashes.

---

## Detect

### Single image

```python
from yowo import InferenceEngine, ModelSpec, ModelFamily, ModelSize, open_source

spec = ModelSpec(ModelFamily.YOLO12, ModelSize.SMALL)
with InferenceEngine(spec, confidence=0.3) as engine:
    src = open_source("photo.jpg")
    for detection in engine.stream(src):
        print(f"{detection.num_boxes} objects in {detection.inference_time_ms:.1f}ms")
        for box in detection.boxes:
            print(f"  {box.class_name}: {box.confidence:.2f}")
```

### Video file

```python
with InferenceEngine(spec, batch_size=4) as engine:
    src = open_source("recording.mp4")
    for detection in engine.stream(src):
        # detection.frame.frame_index is the video frame number
        pass
```

### RTSP stream (auto-reconnect)

```python
with InferenceEngine(spec) as engine:
    src = open_source("rtsp://192.168.1.10:554/live")
    for detection in engine.stream(src):
        # Reconnects automatically on disconnect
        pass
```

### Batch of frames

```python
from yowo import InferenceEngine, ModelSpec, ModelFamily, ModelSize

spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
engine = InferenceEngine(spec, batch_size=8)
engine.load()

import cv2, numpy as np
from yowo.types import Frame

frames = [
    Frame(data=cv2.imread(f"frame_{i:04d}.jpg"), frame_index=i)
    for i in range(8)
]
detections = engine.detect(frames)
engine.close()
```

### Override backend and precision

```python
from yowo import BackendType, Precision

with InferenceEngine(spec, backend=BackendType.ONNX, precision=Precision.FP16) as engine:
    ...
```

---

## Export

Export `.pt` weights to an optimized format for your target hardware.

### CLI

```bash
# Export to ONNX (FP16) — downloads weights automatically
yowo export yolo12n --format onnx --precision fp16

# Export using a local weights file (skips download)
yowo export yolo26n --weights /path/to/YOLO26.pt --format onnx --precision fp32

# Export to TensorRT engine (FP16)
yowo export yolo26s --format tensorrt --precision fp16 --output-dir ./engines/

# Export to ONNX with INT8 quantization (requires calibration images)
yowo export yolo11m --format onnx --precision int8 --calibration-data ./cal_images/

# Export with dynamic batch support
yowo export yolo12n --format onnx --dynamic-batch --imgsz 1280
```

### Python API

```python
from yowo import export_model, ModelSpec, ModelFamily, ModelSize, ExportFormat, Precision
from pathlib import Path

meta = export_model(
    ModelSpec(ModelFamily.YOLO12, ModelSize.NANO),
    ExportFormat.ONNX,
    output_dir=Path("./exported/"),
    precision=Precision.FP16,
)

print(meta.file_path)          # Path to exported model file
print(meta.file_size_bytes)    # Size in bytes
print(meta.export_duration_sec)  # How long it took
```

Each export produces a `.yowo.json` sidecar file recording the model family, precision, export date, and hardware used.

### INT8 quantization

INT8 requires a calibration dataset of at least 300 representative images.

```bash
yowo export yolo26n --format tensorrt --precision int8 \
    --calibration-data /datasets/coco_val/images/
```

```python
meta = export_model(
    spec, ExportFormat.TENSORRT, Path("./engines/"),
    precision=Precision.INT8,
    calibration_data="/datasets/coco_val/images/",
)
```

---

## Hardware Info

```bash
yowo info
```

Output example:
```
=== Hardware ===
CPU: Device(type=cpu, name=AMD EPYC 7763, cpu_arch=x86_64)
GPU 0: Device(type=cuda, index=0, name=NVIDIA A100, arch=ampere)
CPU features: avx2

=== Libraries ===
torch:        2.3.0+cu121
cuda:         12.1
tensorrt:     10.0.1
onnxruntime:  1.18.0 (CUDA)
openvino:     not installed
```

---

## Configuration

### Via Python

```python
from yowo import InferenceConfig, InferenceEngine

cfg = InferenceConfig(
    confidence=0.35,
    iou_threshold=0.5,
    batch_size=4,
    max_det=100,
)
with InferenceEngine(spec, **cfg.__dict__) as engine:
    ...
```

### Via YAML file

```yaml
# yowo.yaml
confidence: 0.35
iou_threshold: 0.50
batch_size: 4
max_det: 100
```

```python
from yowo import load_config
cfg = load_config("yowo.yaml")
```

### Via environment variables

```bash
export YOWO_CONFIDENCE=0.35
export YOWO_BATCH_SIZE=4
export YOWO_IOU_THRESHOLD=0.5
```

Precedence: environment variables > YAML file > defaults.

---

## Error Handling

All exceptions inherit from `yowo.YowoError`.

```python
from yowo import (
    YowoError,
    DependencyError,   # SDK not installed
    BackendLoadError,  # Model file corrupt / wrong format
    InferenceError,    # Runtime inference failure
    SourceError,       # Input stream unreachable
    ConfigError,       # Invalid configuration values
)

try:
    with InferenceEngine(spec) as engine:
        ...
except DependencyError as e:
    print(f"Missing package: {e.package}")
    print(f"Install with: {e.install_cmd}")
except BackendLoadError as e:
    print(f"Backend failed: {e}")
    # Engine already tried all fallback backends before raising
except YowoError as e:
    print(f"yowo error: {e}")
```

---

## Platform Notes

| Platform | Backend | Notes |
|----------|---------|-------|
| NVIDIA GPU (server) | TensorRT or ONNX (CUDA) | Install `yowo[onnx-gpu]`; TensorRT is manual |
| NVIDIA Jetson | TensorRT | `JetPack >= 5.0`; CUDA and TensorRT pre-installed |
| Intel CPU/iGPU | OpenVINO | Install `yowo[openvino]` |
| x86 CPU (Linux) | ONNX | Install `yowo[onnx]`; AVX2 gives ~2x speedup |
| ARM CPU (Raspberry Pi, Graviton) | ONNX | Install `yowo[onnx]` |

---

## Architecture

| Module | Path | Responsibility |
|--------|------|----------------|
| core | [`src/yowo/`](src/yowo/README.md) | `InferenceEngine`, public API surface, `engine.py`, `config.py`, `types.py`, `errors.py` |
| backends | [`src/yowo/backends/`](src/yowo/backends/README.md) | Inference backend implementations (TensorRT, ONNX, OpenVINO, PyTorch) and automatic priority-chain selection |
| cli | [`src/yowo/cli/`](src/yowo/cli/README.md) | Click-based CLI — `detect`, `export`, `info`, `models` commands |
| export | [`src/yowo/export/`](src/yowo/export/README.md) | Export `.pt` weights to ONNX / TensorRT / OpenVINO with calibration, metadata sidecar, and output validation |
| hardware | [`src/yowo/hardware/`](src/yowo/hardware/README.md) | One-time hardware detection (GPU, CPU arch, installed libs), cached for session lifetime |
| io | [`src/yowo/io/`](src/yowo/io/README.md) | Frame sources (image, video, RTSP, directory), batch preprocessing, output sinks |
| models | [`src/yowo/models/`](src/yowo/models/README.md) | Model family / size registry, weight download, and `~/.cache/yowo/weights/` cache management |
| postprocess | [`src/yowo/postprocess/`](src/yowo/postprocess/README.md) | Decode raw backend tensors into `Detection` objects; NMS for backends that return raw proposals |

---

## Development

```bash
# Clone and install with dev deps
git clone https://github.com/your-org/yowo
cd yowo
uv sync --group dev

# Quality gates (run before every commit)
uv run ruff check src/ tests/
uv run pyright src/yowo/
uv run pytest tests/unit/ --cov=yowo --cov-report=term-missing

# CLI from source
uv run yowo info
```

Architecture and module contracts are documented in:
- [`CONTEXT.md`](CONTEXT.md) — project scope, principles, dependency graph
- [`src/yowo/README.md`](src/yowo/README.md) — library architecture overview
- Each module directory has its own `README.md`

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
