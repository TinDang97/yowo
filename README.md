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

**Requirements**: Python >=3.8, Linux (production) / macOS (development)

> **Jetson (aarch64)**: NVIDIA provides custom `onnxruntime-gpu` wheels via JetPack.
> Install from NVIDIA's index instead of PyPI.

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

# Auto-tune config for your device and source type
yowo detect video.mp4 --model yolo26n --preset

# RTSP stream
yowo detect rtsp://camera-ip:554/stream --model yolo26n --confidence 0.4

# Save detections to JSON
yowo detect ./images/ --model yolo11s --output detections.json

# Classify images
yowo classify image.jpg --model yolo11n-cls

# Track objects with persistent IDs (ByteTrack)
yowo track video.mp4 --model yolo26n

# Count objects crossing a line or occupying zones
yowo count video.mp4 --model yolo26n --line --zone --json

# Show hardware and installed backends
yowo info

# List all registered model variants
yowo models
```

### Python API

```python
from yowo import InferenceEngine, open_source

# Detection: auto-select everything (defaults to YOLO26 Nano)
with InferenceEngine(confidence_threshold=0.35) as engine:
    for detection in engine.stream(open_source("image.jpg")):
        for box in detection.boxes:
            print(f"{box.class_name}: {box.confidence:.2f} @ {box.as_xyxy()}")

# Classification: one-liner
from yowo import classify
results = classify("photo.jpg", model="yolo11n-cls", top_k=3)
print(results[0].top1_class_id, results[0].top1_score)
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
from yowo import InferenceEngine, ModelFamily, ModelSize, open_source

with InferenceEngine(
    model_family=ModelFamily.YOLO11,
    model_size=ModelSize.SMALL,
    confidence_threshold=0.3,
) as engine:
    src = open_source("photo.jpg")
    for detection in engine.stream(src):
        print(f"{detection.num_boxes} objects in {detection.inference_time_ms:.1f}ms")
        for box in detection.boxes:
            print(f"  {box.class_name}: {box.confidence:.2f}")
```

### Video file

```python
# prefetch=True (default) overlaps decode and inference for offline sources
with InferenceEngine(prefetch=True, batch_size=4) as engine:
    src = open_source("recording.mp4")
    for detection in engine.stream(src):
        # detection.frame.frame_index is the video frame number
        pass
```

### RTSP stream (auto-reconnect, live pipeline)

```python
from yowo.types import FrameDropPolicy

# Live source → _stream_live path is selected automatically
# ThreadedFrameReader decouples network I/O from inference
with InferenceEngine(
    frame_drop_policy=FrameDropPolicy.LATEST,  # always-current frame
    max_queue_size=2,
) as engine:
    src = open_source("rtsp://192.168.1.10:554/live")
    for detection in engine.stream(src):
        # Reconnects automatically on disconnect
        print(f"lag: {detection.frame.frame_index}")
```

`FrameDropPolicy` controls what happens when inference is slower than frame delivery:

| Policy | Behaviour | Use case |
|--------|-----------|----------|
| `NONE` | Block until queue has space | Offline analysis — no frame skipping |
| `LATEST` | Evict oldest, insert newest | Real-time display — always-current view |
| `SKIP_OLDEST` | Pop back of queue | Ordered processing with bounded latency |

### Batch of frames

```python
from yowo import InferenceEngine
from yowo.types import Frame
import cv2

engine = InferenceEngine(batch_size=8)
engine.load()

frames = [
    Frame(pixels=cv2.imread(f"frame_{i:04d}.jpg"), source_id="batch", frame_index=i)
    for i in range(8)
]
detections = engine.detect(frames)
engine.close()
```

### Free-threaded Python (GIL=OFF)

On Python 3.13+ with the free-threaded build (`python3.13t`), inference workers
run truly in parallel. `pipeline_workers` is auto-detected:

```python
from yowo.types import is_free_threaded

print(is_free_threaded())  # True on python3.13t

# pipeline_workers=2 is set automatically on GIL=OFF builds
with InferenceEngine(prefetch=True) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        ...
# ~1.5x throughput vs GIL Python on CPU inference (YOLO26n: 39 → 58 FPS)
```

### Custom fine-tuned model (num_classes override)

```python
# Model trained on 7 vehicle classes instead of 80 COCO classes
with InferenceEngine(
    model_family=ModelFamily.YOLO11,
    model_size=ModelSize.SMALL,
    weights_path=Path("./best.pt"),
    num_classes=7,  # override default 80
) as engine:
    for detection in engine.stream(open_source("traffic.mp4")):
        ...
```

```bash
# CLI equivalent
yowo detect traffic.mp4 --model yolo11s --weights best.pt --num-classes 7
```

### Custom architecture (ModelBuilder)

Plug a non-YOLO model into yowo's inference pipeline:

```python
from yowo import InferenceEngine, ModelBuilder

class MyDetector:
    """Implements ModelBuilder protocol."""
    def build(self, num_classes: int, device: str):
        model = MyCustomArch(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load("custom.pt"))
        model.eval()
        return model

    @property
    def input_shape(self) -> tuple[int, int]:
        return (640, 640)

with InferenceEngine(model_builder=MyDetector(), num_classes=10) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        ...
```

### Using InferenceConfig

```python
from yowo import InferenceConfig, InferenceEngine

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    confidence_threshold=0.35,
    batch_size=4,
)
with InferenceEngine(config) as engine:
    ...
```

### Preset config (auto-tuning)

Auto-select pipeline knobs (batch size, caching, prefetch, frame drop policy) based on detected hardware and source type:

```python
from yowo import classify_source, preset_config
from yowo.hardware import get_hardware_profile

hw = get_hardware_profile()
source_cat = classify_source("rtsp://192.168.1.10/stream")
config = preset_config(hw, source_cat)

with InferenceEngine(config) as engine:
    ...
```

CLI equivalent — `--preset` auto-tunes, explicit flags override preset values:

```bash
yowo detect video.mp4 --preset                    # fully automatic
yowo detect video.mp4 --preset --batch 8           # override batch size
yowo detect rtsp://cam/stream --preset --confidence 0.4
```

### Override backend and precision

```python
from yowo import BackendType, Precision

with InferenceEngine(backend=BackendType.ONNX, precision=Precision.FP16) as engine:
    ...
```

### Feature map cache (sequential video inference)

Skip backbone + neck on similar consecutive frames — 60–85% compute savings for slow-moving scenes.

```python
# In-memory cache (default)
with InferenceEngine(cache=True) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        ...

# mmap-backed cache (OS manages memory pressure)
from pathlib import Path
with InferenceEngine(cache_dir=Path("/tmp/yowo-cache")) as engine:
    for detection in engine.stream(open_source("rtsp://camera/stream")):
        ...
```

### KV cache (attention state across frames)

Reuse Attention K,V tensors and skip C2PSA/C3k2PSA blocks on similar frames. Best for PyTorch CPU/MPS; no benefit on ONNX runtimes.

```python
with InferenceEngine(kv_cache=True) as engine:
    for detection in engine.stream(open_source("video.mp4")):
        ...
```

Export a KV-cache-enabled ONNX model (K,V as explicit I/O for stateless runtimes):

```python
from pathlib import Path
from yowo import export_model, ExportFormat, ModelSpec, ModelFamily, ModelSize, Precision

spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
meta = export_model(
    spec, ExportFormat.ONNX, output_dir=Path("./exported/"),
    kv_cache=True,
)
```

---

## Track

ByteTrack multi-object tracking with persistent IDs across frames. Two-stage IoU association with Kalman filter prediction.

```python
from yowo import InferenceEngine, open_source
from yowo.tracking import ByteTracker, track_stream

tracker = ByteTracker(
    track_high_thresh=0.3,   # stage-1 confidence threshold
    track_low_thresh=0.1,    # stage-2 low-confidence recovery
    match_thresh=0.8,        # IoU matching threshold
    max_age=30,              # frames before track removal
    min_hits=3,              # hits before track confirmation
)

with InferenceEngine() as engine:
    for tracked in track_stream(engine, open_source("video.mp4"), tracker=tracker):
        for box in tracked.boxes:
            print(f"ID:{box.track_id} {box.class_name} {box.confidence:.2f} confirmed={box.is_confirmed}")
        print(f"Active: {tracker.active_track_count}, Lost: {tracker.lost_track_count}")
```

Optional scipy acceleration for the Hungarian algorithm:

```bash
pip install yowo[tracking]  # installs scipy
```

### ReID-Enhanced Tracking

When a `ReIDExtractor` is provided, ByteTracker uses appearance features to resolve ambiguous IoU assignments and recover long-lost tracks:

```python
from yowo.tracking import ByteTracker, CLIPExtractor, track_stream

reid = CLIPExtractor("path/to/clip-vit-b16.onnx")
tracker = ByteTracker(reid_extractor=reid)

with InferenceEngine() as engine:
    for tracked in track_stream(engine, open_source("video.mp4"), tracker=tracker):
        for box in tracked.boxes:
            print(f"ID:{box.track_id} {box.class_name}")
```

The `needs_reid()` gate skips ReID extraction when IoU assignments are unambiguous — achieving 99.8% skip rate on typical surveillance footage with zero FPS impact.

---

## Cross-Camera Tracking

`CrossCameraTracker` unifies per-camera ByteTrackers with a shared embedding gallery for cross-camera identity matching. Each vehicle/person gets a `global_id` that persists across cameras.

```python
from yowo import InferenceEngine, open_source
from yowo.tracking import CrossCameraTracker, CLIPReIDExtractor, CameraLinkModel, CameraLink

# ReID model (CLIP-ReID fine-tuned on VeRi-776: mAP=82.28%, Rank-1=96.66%)
reid = CLIPReIDExtractor("path/to/clip-reid-veri-vit-b16.onnx")

# Optional: spatial-temporal transit constraints between cameras
links = CameraLinkModel(links=[
    CameraLink(src_camera="cam-entrance", dst_camera="cam-exit",
               min_transit_sec=10.0, max_transit_sec=60.0),
])

tracker = CrossCameraTracker(
    reid_extractor=reid,
    camera_link_model=links,
    match_threshold=0.35,
)

with InferenceEngine() as engine:
    # Process frames from multiple cameras
    for det in engine.stream(open_source("cam1.mp4")):
        results = tracker.update("cam-entrance", det)
        for box in results:
            print(f"Global:{box.global_id} Local:{box.local_track_id} "
                  f"{box.box.class_name} cam={box.camera_id}")
```

Built-in ReID extractors:

| Extractor | Architecture | Dim | Domain | Install |
|-----------|-------------|-----|--------|---------|
| `CLIPExtractor` | CLIP ViT-B/16 | 512 | Zero-shot general | `yowo[tracking]` |
| `CLIPReIDExtractor` | CLIP-ReID VeRi | 1280 | Vehicle (fine-tuned) | `yowo[tracking]` |
| `FastReIDExtractor` | ResNet-50 SBS | 256 | Person ReID | `yowo[tracking]` |
| `VehicleReIDExtractor` | ResNet-50 | 256 | Vehicle general | `yowo[tracking]` |

Any class implementing the `ReIDExtractor` Protocol can be used as a drop-in replacement.

---

## Count

Zone occupancy and line-crossing counting built on top of tracking.

```python
from yowo import InferenceEngine, open_source
from yowo.counter import ObjectCounter, CrossDirection
from yowo.tracking import ByteTracker, track_stream
from yowo.utils import make_half_zones, make_center_line

# Create counting geometry
zones = list(make_half_zones(1280, 720))    # top/bottom halves
line = make_center_line(1280, 720)          # horizontal center line

tracker = ByteTracker()
counter = ObjectCounter(zones=zones, lines=[line])

with InferenceEngine() as engine:
    for tracked in track_stream(engine, open_source("video.mp4"), tracker=tracker):
        counter.update(tracked)

# Results
for zone_id, counts in counter.zone_counts.items():
    print(f"{zone_id}: {dict(counts)}")

for line_id, dirs in counter.line_totals.items():
    in_count = dirs.get(CrossDirection.IN, 0)
    out_count = dirs.get(CrossDirection.OUT, 0)
    print(f"{line_id}: IN={in_count} OUT={out_count}")
```

---

## Classify

YOLO image classification — top-k class probabilities, no bounding boxes.

### CLI

```bash
yowo classify image.jpg --model yolo11n-cls
yowo classify video.mp4 --model yolo11s-cls --top-k 3
yowo classify image.jpg --model yolo11n-cls --weights ./best-cls.pt
```

### Python API

```python
from yowo import ClassificationEngine, open_source

with ClassificationEngine(model_family="yolo11", model_size="n") as engine:
    for result in engine.stream(open_source("image.jpg")):
        print(f"Top-1: class {result.top1_class_id} ({result.top1_score:.3f})")
        for class_id, score in result.top_k:
            print(f"  {class_id}: {score:.3f}")
```

One-liner convenience:

```python
from yowo import classify
results = classify("photo.jpg", model="yolo11n-cls", top_k=5)
```

Classification models use the `-cls` suffix (e.g. `yolo11n-cls`, `yolo26s-cls`). Weights are downloaded automatically.

---

## Annotate

Reusable drawing utilities for detection, tracking, and counting overlays.

```python
from yowo.utils import (
    draw_bounding_boxes,     # class-colored detection boxes
    draw_tracked_boxes,      # track-colored boxes with IDs
    draw_zones,              # semi-transparent zone polygons
    draw_count_lines,        # counting line overlays
    draw_text_panel,         # translucent stats panel
    make_half_zones,         # zone factory
    make_center_line,        # line factory
)
```

See [`examples/annotated_video.py`](examples/annotated_video.py) for a complete annotated video pipeline.

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

# Option A: Pass config object
config = InferenceConfig(
    confidence_threshold=0.35,
    iou_threshold=0.5,
    batch_size=4,
)
with InferenceEngine(config) as engine:
    ...

# Option B: Pass kwargs directly
with InferenceEngine(confidence_threshold=0.35, batch_size=4) as engine:
    ...
```

### Via YAML file

```yaml
# yowo.yaml
confidence_threshold: 0.35
iou_threshold: 0.50
batch_size: 4
```

```python
from yowo import load_config, InferenceEngine

config = load_config("yowo.yaml")
with InferenceEngine(config) as engine:
    ...
```

### Via environment variables

```bash
export YOWO_CONFIDENCE=0.35
export YOWO_BATCH_SIZE=4
export YOWO_IOU=0.5
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
    with InferenceEngine() as engine:
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
| core | [`src/yowo/`](src/yowo/README.md) | `InferenceEngine` (detection), `ClassificationEngine`, public API surface, `engine.py`, `classify_engine.py`, `config.py`, `types.py`, `errors.py` |
| arch | [`src/yowo/arch/`](src/yowo/arch/README.md) | Native YOLO11 and YOLO26 PyTorch — backbone, FPN-PAN neck, detection head, classification head, scaling, weight loading |
| backends | [`src/yowo/backends/`](src/yowo/backends/README.md) | Inference backend implementations (TensorRT, ONNX, OpenVINO, PyTorch) and automatic priority-chain selection |
| cli | [`src/yowo/cli/`](src/yowo/cli/README.md) | Click-based CLI — `detect`, `classify`, `export`, `info`, `models`, `track`, `count` commands |
| counter | `src/yowo/counter/` | Zone occupancy (ray-casting PIP) and line-crossing counting (cross-product sign test) |
| export | [`src/yowo/export/`](src/yowo/export/README.md) | Export `.pt` weights to ONNX / TensorRT / OpenVINO with calibration, metadata sidecar, and output validation |
| hardware | [`src/yowo/hardware/`](src/yowo/hardware/README.md) | One-time hardware detection (GPU, CPU arch, installed libs), cached for session lifetime |
| io | [`src/yowo/io/`](src/yowo/io/README.md) | Frame sources (image, video, RTSP, directory), batch preprocessing, output sinks |
| models | [`src/yowo/models/`](src/yowo/models/README.md) | Model family / size registry, weight download, and `~/.cache/yowo/weights/` cache management |
| postprocess | [`src/yowo/postprocess/`](src/yowo/postprocess/README.md) | Decode raw backend tensors into `Detection` objects; NMS for backends that return raw proposals |
| tracking | `src/yowo/tracking/` | ByteTrack multi-object tracking, cross-camera ReID, embedding gallery, appearance-gated fusion, camera link constraints |
| utils | `src/yowo/utils/` | Reusable drawing/annotation utilities — palettes, bounding boxes, zones, lines, text panels, factories |

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
| [Phase 3 Source-Aware Pipeline](docs/experiments/2026-02-25-phase3-source-aware-pipeline-benchmark.md) | Source-aware dispatch (`_stream_live` / `_stream_pipeline`), `ThreadedFrameReader`, `FrameDropPolicy`, pre-allocated I/O buffers. CoreML 3.5–3.75× faster than PyTorch on offline video. Free-threaded Python 3.13t: `pipeline_workers=2` auto-selected → **58.4 FPS vs 39.3 FPS (1.49×)** on YOLO26n. |
| [ByteTrack + ObjectCounter Annotated Video](docs/experiments/2026-02-28-bytetrack-counter-annotated-video-benchmark.md) | Full detect→track→count→annotate pipeline on 928-frame traffic video. ONNX+CoreML: YOLO26n **82 FPS**, YOLO26x 21 FPS. PyTorch+MPS: YOLO26s **63 FPS** (9% faster than CoreML for small models). CoreML wins 4/5 variants (up to 1.32×). ByteTrack overhead 0.3–0.7ms (1.3–5.2%). |
| [VeRi-776 Cross-Camera Vehicle ReID](docs/experiments/2026-03-01-veri-776-cross-camera-reid-benchmark.md) | CLIP zero-shot mAP=9.32%, FastReID SBS-S50 mAP=8.43%, CLIP-ReID VeRi mAP=**82.28%** (Rank-1=**96.66%**). CoreML EP: 15.5 img/s. |
| [ReID Method Comparison](docs/experiments/2026-03-01-reid-method-comparison.md) | `needs_reid()` gate achieves **99.8% skip rate** (2 ReID calls per 928 frames). Zero FPS impact: no-ReID 107 FPS, CLIP 106 FPS, FastReID 108 FPS. |

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
