# yowo

> Production YOLO inference and export — hardware-aware, multi-backend, edge-ready.

yowo implements native YOLO11 and YOLO26 architectures for inference and export, adding what production deployments need: automatic hardware detection, transparent backend selection, graceful degradation, and stream resilience.

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
yowo detect video.mp4 --model yolo26n

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

## Example

```bash
yowo detect "input.jpg" \
  --model yolo26n \
  --weights "yolo26n.pt" \
  --backend pytorch \
  --confidence 0.25 \
  --output detections.json
```

The full JSON output per detection:

```json
{
  "frame_index": 0,
  "source_id": "input.jpg",
  "inference_time_ms": 582.2,
  "backend": "pytorch",
  "model": "yolo26n",
  "boxes": [
    {
      "x1": 387.0, "y1": 422.0, "x2": 622.0, "y2": 537.0,
      "confidence": 0.888,
      "class_id": 2,
      "class_name": "car"
    },
    ....
  ]
}
```

---

## Models

| Name | Alias | Notes |
|------|-------|-------|
| `yolo11n/s/m/l/x` | YOLO11 | Stable, best production baseline |
| `yolo26n/s/m/l/x` | YOLO26 | NMS-free, best CPU and INT8 speed |

Weights are downloaded automatically to `~/.cache/yowo/weights/` on first use.

---

## Backends

yowo selects the best available backend automatically. You can override.

| Backend | Format | When used |
|---------|--------|-----------|
| TensorRT | `.engine` | NVIDIA GPU + TensorRT installed |
| ONNX Runtime (CUDA) | `.onnx` | NVIDIA GPU + onnxruntime-gpu |
| ONNX Runtime (CoreML) | `.onnx` | macOS + Apple Silicon (auto-detected) |
| OpenVINO | `_openvino_model/` | Intel CPU/iGPU + openvino |
| ONNX Runtime (CPU) | `.onnx` | Any CPU + onnxruntime |
| PyTorch | `.pt` | Universal fallback |

**Priority chain**: TensorRT → ONNX (CUDA) → CoreML → OpenVINO → ONNX (CPU) → PyTorch

> **Apple Silicon**: CoreML EP is auto-detected and offloads inference to the Neural Engine — **4-5x faster** than PyTorch CPU. No configuration needed.

If a backend fails to load, yowo falls back to the next in chain and logs a warning — it never crashes.

---

## Detect

### Single image

```python
from yowo import InferenceEngine, ModelSpec, ModelFamily, ModelSize, open_source

spec = ModelSpec(ModelFamily.YOLO11, ModelSize.SMALL)
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

### Feature map cache (sequential video inference)

Skip backbone + neck on similar consecutive frames — 60–85% compute savings for slow-moving scenes.

```python
# In-memory cache (default)
with InferenceEngine(spec, cache=True) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        ...

# mmap-backed cache (OS manages memory pressure)
from pathlib import Path
with InferenceEngine(spec, cache_dir=Path("/tmp/yowo-cache")) as engine:
    for detection in engine.stream(open_source("rtsp://camera/stream")):
        ...
```

### KV cache (attention state across frames)

Reuse Attention K,V tensors and skip C2PSA/C3k2PSA blocks on similar frames. Best for PyTorch CPU/MPS; no benefit on ONNX runtimes.

```python
with InferenceEngine(spec, kv_cache=True) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        ...
```

Export a KV-cache-enabled ONNX model (K,V as explicit I/O for stateless runtimes):

```python
from yowo import export_model, ExportFormat, Precision

meta = export_model(
    spec, ExportFormat.ONNX, output_dir=Path("./exported/"),
    kv_cache=True,
)
```

---

## Export

Export `.pt` weights to an optimized format for your target hardware.

### CLI

```bash
# Export to ONNX (FP16) — downloads weights automatically
yowo export yolo11n --format onnx --precision fp16

# Export using a local weights file (skips download)
yowo export yolo26n --weights /path/to/YOLO26.pt --format onnx --precision fp32

# Export to TensorRT engine (FP16)
yowo export yolo26s --format tensorrt --precision fp16 --output-dir ./engines/

# Export to ONNX with INT8 quantization (requires calibration images)
yowo export yolo11m --format onnx --precision int8 --calibration-data ./cal_images/

# Export with dynamic batch support
yowo export yolo11n --format onnx --dynamic-batch --imgsz 1280
```

### Python API

```python
from yowo import export_model, ModelSpec, ModelFamily, ModelSize, ExportFormat, Precision
from pathlib import Path

meta = export_model(
    ModelSpec(ModelFamily.YOLO26, ModelSize.NANO),
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
| Apple Silicon (M1–M4) | ONNX (CoreML) | Install `yowo[onnx]`; auto-detects Neural Engine, 4-5x vs CPU |
| Apple Silicon (MPS) | PyTorch | MPS GPU via `--device mps`; 1.3x vs ultralytics |
| Intel CPU/iGPU | OpenVINO | Install `yowo[openvino]` |
| x86 CPU (Linux) | ONNX | Install `yowo[onnx]`; AVX2 gives ~2x speedup |
| ARM CPU (Raspberry Pi, Graviton) | ONNX | Install `yowo[onnx]` |

---

## Architecture

| Module | Path | Responsibility |
|--------|------|----------------|
| core | [`src/yowo/`](src/yowo/README.md) | `InferenceEngine`, public API surface, `engine.py`, `config.py`, `types.py`, `errors.py` |
| arch | [`src/yowo/arch/`](src/yowo/arch/README.md) | Native YOLO11 and YOLO26 PyTorch — backbone, FPN-PAN neck, detection head, scaling, weight loading |
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
- [`src/yowo/arch/README.md`](src/yowo/arch/README.md) — native YOLO backbone/neck/head, scaling, weight loading
- Each module directory has its own `README.md`

### Experiments

| Report | Summary |
|--------|---------|
| [Vehicle Detection Benchmark — YOLO11s vs YOLO26m](docs/experiments/2026-02-23-vehicle-detection-benchmark.md) | PyTorch FP32 vs ONNX FP32/FP16/INT8 on Apple M4 Pro. YOLO11s ONNX FP16 achieves 18.1 FPS (2.62x PyTorch). YOLO26m ONNX FP32 achieves 6.9 FPS. |
| [Native Architecture Inference Optimization — all 10 variants](docs/experiments/2026-02-24-arch-inference-optimization-benchmark.md) | DFL buffer, in-place sigmoid, stride flag, anchor cache applied to `arch/`. YOLO26 family 10-17% faster than ultralytics baseline; YOLO11 family 1-4% faster. Box IoU vs ultralytics: 0.967-0.995. 9/10 variants faster, avg 1.07x. |
| [ONNX + CoreML EP + MPS Optimization](docs/experiments/2026-02-24-onnx-coreml-optimization-benchmark.md) | CoreML EP auto-detection for Apple Neural Engine: **4.36x avg faster** than PyTorch across all 10 variants (nano 140-188 FPS, XL 27-29 FPS). MPS (Metal GPU): 1.32x avg faster than ultralytics. KV cache analysis: +12% on CPU PyTorch (block cache), negligible on GPU/CoreML. |

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
