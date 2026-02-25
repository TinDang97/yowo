# yowo User Guide

**yowo** — native YOLO11 and YOLO26 inference without ultralytics. Hardware auto-detection, backend fallback chain, stream resilience, and a production-grade CLI and Python API.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [CLI Reference](#3-cli-reference)
   - [yowo info](#31-yowo-info)
   - [yowo models](#32-yowo-models)
   - [yowo detect](#33-yowo-detect)
   - [yowo export](#34-yowo-export)
4. [Python API Reference](#4-python-api-reference)
   - [InferenceEngine](#41-inferenceengine)
   - [open_source](#42-open_source)
   - [Core Types](#43-core-types)
   - [Configuration](#44-configuration)
   - [IO Utilities](#45-io-utilities)
5. [Backend Selection](#5-backend-selection)
6. [Streaming Pipeline](#6-streaming-pipeline)
7. [Model Export](#7-model-export)
8. [Environment Variables](#8-environment-variables)
9. [YAML Configuration](#9-yaml-configuration)
10. [Error Reference](#10-error-reference)
11. [Use Cases](#11-use-cases)
    - [Traffic Surveillance on Apple Silicon](#111-traffic-surveillance-on-apple-silicon)
    - [Live RTSP Camera Stream](#112-live-rtsp-camera-stream)
    - [Custom Fine-Tuned Vehicle Detector](#113-custom-fine-tuned-vehicle-detector)
    - [Batch Video Processing](#114-batch-video-processing)
    - [Free-Threaded Python Parallelism](#115-free-threaded-python-parallelism)
    - [ONNX CoreML EP vs PyTorch MPS on Apple Silicon](#116-onnx-coreml-ep-vs-pytorch-mps-on-apple-silicon)
12. [Performance Reference](#12-performance-reference)

---

## 1. Installation

```bash
pip install yowo
```

**Optional backend extras:**

```bash
pip install yowo[onnx]       # ONNX Runtime (+ CoreML EP on macOS automatically)
pip install yowo[tensorrt]   # TensorRT (NVIDIA GPU)
pip install yowo[openvino]   # OpenVINO (Intel CPU/iGPU)
pip install yowo[all]        # All backends
```

**From source (development):**

```bash
git clone https://github.com/tind-repo/yowo
cd yowo
uv sync --group dev
```

**Weight cache:** model weights are cached at `~/.cache/yowo/weights/`. Pass `--weights` to use a local file and skip download.

---

## 2. Quick Start

### CLI — single image

```bash
yowo detect image.jpg --model yolo26n
```

### CLI — save annotated output

```bash
yowo detect image.jpg --model yolo11s --weights best.pt -o out.jpg
```

### CLI — RTSP stream

```bash
yowo detect rtsp://camera.local:8554/live --model yolo26n --save-frames /tmp/frames
```

### Python — single image

```python
import numpy as np
from yowo.engine import InferenceEngine
from yowo.io import open_source

with InferenceEngine(model_family="yolo26", model_size="n") as eng:
    src = open_source("image.jpg")
    for det in eng.stream(src):
        for box in det.boxes:
            print(f"{box.class_name}: {box.confidence:.2f}  [{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}]")
```

### Python — video file

```python
from yowo.engine import InferenceEngine
from yowo.io import open_source, write_annotated_frames
from yowo.types import ModelFamily, ModelSize

with InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
) as eng:
    src = open_source("video.mp4")
    results = list(eng.stream(src))

write_annotated_frames(results, "output/")
print(f"Processed {len(results)} frames")
```

---

## 3. CLI Reference

### 3.1 `yowo info`

Print detected hardware, available backends, and installed library versions.

```bash
yowo info
```

**Example output:**

```
=== Hardware ===
CPU: Apple M4 Pro
CPU features: avx2, neon
GPU 0: Apple M4 Pro (Metal)

=== Libraries ===
torch:        2.10.0
cuda:         not available
tensorrt:     not installed
onnxruntime:  1.24.2 (CoreML)
openvino:     not installed
```

---

### 3.2 `yowo models`

List all registered model variants.

```bash
yowo models
yowo models --family yolo26    # filter by family
yowo models --family yolo11
```

**Example output:**

```
Model        Input      Classes    URL
--------------------------------------------------------------------------------
yolo11n      640x640    80         https://github.com/ultralytics/assets/...
yolo11s      640x640    80         ...
yolo26n      640x640    80         ...
```

---

### 3.3 `yowo detect`

Run object detection on any source type.

```
yowo detect SOURCE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `SOURCE` | Image file, video file, directory, `rtsp://…` URL, or webcam index (`"0"`, `"1"`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | `yolo26n` | Model name: `yolo11n`, `yolo26s`, etc. |
| `--weights`, `-w` | auto | Path to local `.pt` weights file (skips download) |
| `--backend` | `auto` | `auto`, `pytorch`, `onnx`, `tensorrt`, `openvino` |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0` |
| `--precision` | `auto` | `auto`, `fp32`, `fp16`, `int8` |
| `--confidence` | `0.25` | Minimum confidence score `[0.0, 1.0]` |
| `--iou` | `0.45` | NMS IoU threshold `[0.0, 1.0]` |
| `--batch` | `1` | Frames per inference batch |
| `--output`, `-o` | none | Output path. Image extension → annotated frame; `.json` → detections JSON |
| `--output-format` | `auto` | `auto`, `json`, `image` (overrides extension inference) |
| `--save-frames` | none | Directory to save all annotated frames |

**Examples:**

```bash
# Single image, save annotated result
yowo detect photo.jpg --model yolo26m -o result.jpg

# Video with explicit ONNX backend, save JSON
yowo detect video.mp4 --backend onnx -o detections.json

# Webcam (device index 0)
yowo detect 0 --model yolo11n --device cpu

# RTSP with explicit confidence threshold
yowo detect rtsp://192.168.1.10:8554/stream --confidence 0.35 --save-frames /tmp/frames

# Directory of images
yowo detect /path/to/images/ --model yolo11s --output-format json -o results.json

# Local weights, no download
yowo detect image.jpg --model yolo11s --weights ./best.pt --backend pytorch
```

---

### 3.4 `yowo export`

Export a PyTorch model to ONNX, TensorRT, or OpenVINO.

```
yowo export MODEL --format FORMAT [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `MODEL` | Model name: `yolo11n`, `yolo26m`, etc. |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--weights`, `-w` | auto | Path to local `.pt` weights file |
| `--format`, `-f` | required | `onnx`, `tensorrt`, `openvino` |
| `--precision`, `-p` | `fp16` | `fp32`, `fp16`, `int8` |
| `--calibration-data` | none | Path to calibration images (required for INT8) |
| `--output-dir`, `-o` | `~/.yowo/models/MODEL/FORMAT_PRECISION` | Output directory |
| `--dynamic-batch` / `--no-dynamic-batch` | `--no-dynamic-batch` | Enable dynamic batch dimension in ONNX |
| `--imgsz` | `640` | Input image size (square) |

**Examples:**

```bash
# Export YOLO26n to ONNX FP16 with dynamic batch
yowo export yolo26n --format onnx --precision fp16 --dynamic-batch

# Export custom fine-tuned model
yowo export yolo11s --weights ./best.pt --format onnx --precision fp32 -o ./exports/

# Export to TensorRT FP16
yowo export yolo26m --format tensorrt --precision fp16

# Export ONNX INT8 with calibration data
yowo export yolo11n --format onnx --precision int8 --calibration-data ./calib_images/
```

**Output:** prints the exported file path, size in MB, and export duration.

---

## 4. Python API Reference

### 4.1 `InferenceEngine`

The central orchestrator. Lifecycle: `__init__` → `load()` → `detect()`/`stream()` → `close()`. Use as a context manager for automatic resource cleanup.

```python
from yowo.engine import InferenceEngine
from yowo.types import ModelFamily, ModelSize, BackendType, Precision, FrameDropPolicy
```

**Constructor:**

```python
InferenceEngine(
    config: InferenceConfig | None = None,
    *,
    # Model selection
    model_family: ModelFamily = ModelFamily.YOLO26,
    model_size: ModelSize = ModelSize.NANO,
    weights_path: Path | None = None,

    # Backend
    backend: BackendType | None = None,   # None = auto-select
    device: str = "auto",                 # "auto", "cpu", "cuda", "cuda:0"
    precision: Precision | None = None,   # None = auto (fp16 on CUDA, fp32 on CPU)

    # Detection thresholds
    batch_size: int = 1,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,

    # Caching (PyTorch backend only)
    cache: bool = False,           # in-memory feature map cache
    cache_dir: Path | None = None, # mmap-backed feature map cache
    kv_cache: bool = False,        # attention KV cache for streaming

    # Streaming pipeline
    frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST,
    max_queue_size: int = 2,
    prefetch: bool = True,
    pipeline_workers: int = 0,     # 0 = auto (2 on free-threaded Python, 1 otherwise)
)
```

**Methods:**

```python
engine.load() -> None
```
Load and warm up the backend. Called automatically by the context manager.

```python
engine.detect(frames: list[Frame]) -> list[Detection]
```
Run detection on a list of frames. Returns one `Detection` per frame. Raises `InferenceError` if not loaded.

```python
engine.stream(source: FrameSource) -> Iterator[Detection]
```
Yield detections from a `FrameSource`. Batches internally. Dispatches based on source type:
- Single image → direct inference (no thread overhead)
- Live source (RTSP/webcam) → `ThreadedFrameReader` + frame drop policy
- Offline video + `prefetch=True` → background prefetch overlaps I/O with inference
- `prefetch=False` → legacy sequential path

```python
engine.close() -> None
engine.selection  # -> BackendSelection (read-only)
engine.is_loaded  # -> bool (read-only)
```

**Context manager (recommended):**

```python
with InferenceEngine(model_family=ModelFamily.YOLO11, model_size=ModelSize.SMALL) as eng:
    # eng.load() called automatically
    results = list(eng.stream(open_source("video.mp4")))
# eng.close() called automatically
```

**Manual lifecycle:**

```python
eng = InferenceEngine(model_family=ModelFamily.YOLO26, model_size=ModelSize.NANO)
eng.load()
try:
    results = list(eng.stream(open_source("video.mp4")))
finally:
    eng.close()
```

**Using `InferenceConfig`:**

```python
from yowo.config import InferenceConfig
from yowo.types import ModelFamily, ModelSize, BackendType

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.MEDIUM,
    backend=BackendType.ONNX,
    confidence_threshold=0.3,
    batch_size=4,
)
with InferenceEngine(config) as eng:
    ...
```

---

### 4.2 `open_source`

Factory that returns the appropriate `FrameSource` by inspecting the input.

```python
from yowo.io import open_source

open_source(
    source: str | Path,
    *,
    loop: bool = False,               # repeat video when exhausted
    frame_skip: int = 0,              # skip N frames between yields
    max_frames: int | None = None,    # stop after N frames
    reconnect_timeout_s: float = 30.0 # RTSP reconnect budget in seconds
) -> FrameSource
```

**Dispatch rules:**

| Input | Source type |
|-------|-------------|
| `"0"`, `"1"`, … (digit string) | WebcamSource |
| `"rtsp://…"` or `"rtsps://…"` | RTSPStreamSource (auto-reconnect) |
| Image file (`.jpg`, `.png`, `.bmp`, `.webp`, `.tiff`) | ImageFileSource |
| Existing directory | ImageDirectorySource |
| Video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.ts`) | VideoFileSource |

**Examples:**

```python
src = open_source("photo.jpg")
src = open_source("traffic.mp4", frame_skip=2, max_frames=300)
src = open_source("rtsp://192.168.1.10:8554/stream", reconnect_timeout_s=60.0)
src = open_source("0")            # webcam 0
src = open_source("/images/dir/") # all images in directory
```

`FrameSource` protocol attributes:

```python
src.total_frames   # int: total frame count; -1 if unknown (live streams)
src.is_live        # bool: True for RTSP and webcam sources
```

---

### 4.3 Core Types

All public types are in `yowo.types`.

#### `Frame`

A single video or image frame.

```python
@dataclass(slots=True)
class Frame:
    pixels: NDArray[np.uint8]  # HWC BGR uint8 (OpenCV convention)
    source_id: str = ""        # opaque string identifying the input source
    frame_index: int = 0       # zero-based index within the source
    timestamp_ms: float = 0.0  # milliseconds since source epoch

    @property def height(self) -> int: ...
    @property def width(self) -> int: ...
    @property def shape_hw(self) -> tuple[int, int]: ...
```

#### `Detection`

Inference result for one frame.

```python
@dataclass(frozen=True, slots=True)
class Detection:
    frame: Frame                    # source frame
    boxes: tuple[BoundingBox, ...]  # all detected boxes
    inference_time_ms: float        # wall-clock time for the infer() call only
    backend: BackendType            # backend that produced this result
    model_spec: ModelSpec           # model that produced this result

    @property def num_boxes(self) -> int: ...
    @property def has_detections(self) -> bool: ...
```

#### `BoundingBox`

Axis-aligned detection box in pixel coordinates (XYXY format).

```python
@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float         # left edge (pixels)
    y1: float         # top edge (pixels)
    x2: float         # right edge (pixels)
    y2: float         # bottom edge (pixels)
    confidence: float # detection confidence [0, 1]
    class_id: int     # integer class index
    class_name: str   # human-readable class label

    @property def area(self) -> float: ...
    @property def as_xyxy(self) -> tuple[float, float, float, float]: ...
```

#### Enums

```python
from yowo.types import (
    ModelFamily,     # YOLO11 = "yolo11", YOLO26 = "yolo26"
    ModelSize,       # NANO="n", SMALL="s", MEDIUM="m", LARGE="l", XLARGE="x"
    BackendType,     # PYTORCH, ONNX, TENSORRT, OPENVINO
    Precision,       # FP32, FP16, INT8
    FrameDropPolicy, # NONE, LATEST, SKIP_OLDEST
    ExportFormat,    # ONNX, TENSORRT, OPENVINO
)
```

---

### 4.4 Configuration

#### `InferenceConfig`

Structured configuration dataclass for the inference engine.

```python
from yowo.config import InferenceConfig
from yowo.types import ModelFamily, ModelSize

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    weights_path=None,           # None → auto-resolve from registry
    backend=None,                # None → auto-select
    device="auto",
    precision=None,              # None → fp16 on CUDA, fp32 on CPU
    confidence_threshold=0.25,
    iou_threshold=0.45,
    batch_size=1,
    cache=False,
    cache_dir=None,
    kv_cache=False,
    frame_drop_policy=FrameDropPolicy.LATEST,
    max_queue_size=2,
    prefetch=True,
    pipeline_workers=0,          # 0 → auto
)
```

**Validation:**
- `confidence_threshold` and `iou_threshold` must be in `[0.0, 1.0]`
- `batch_size >= 1`
- `max_queue_size >= 1`
- `pipeline_workers >= 0`

#### `load_config`

Load from YAML file with environment variable overrides.

```python
from pathlib import Path
from yowo.config import load_config

config = load_config(Path("yowo.yaml"))
```

Load order (later entries win):
1. Dataclass defaults
2. YAML file values
3. `YOWO_*` environment variables

---

### 4.5 IO Utilities

```python
from yowo.io import (
    write_json,             # serialize Detection list → JSON file
    write_annotated_frame,  # draw boxes on single frame → image file
    write_annotated_frames, # draw boxes on all frames → directory of images
)
```

**`write_json`**

```python
from yowo.io import write_json
from pathlib import Path

write_json(detections, Path("results.json"))
```

**`write_annotated_frame`**

```python
from yowo.io import write_annotated_frame

for det in results:
    write_annotated_frame(det, Path("output.jpg"))
```

**`write_annotated_frames`**

```python
from yowo.io import write_annotated_frames

write_annotated_frames(results, Path("output/"))
# writes output/frame_000000.jpg, output/frame_000001.jpg, …
```

---

## 5. Backend Selection

yowo auto-selects the fastest available backend for your hardware:

```
TensorRT EP  →  ONNX (CUDA EP)  →  ONNX (CoreML EP)  →  ONNX (CPU EP)  →  PyTorch
   (NVIDIA)       (NVIDIA)          (Apple Silicon)       (any CPU)        (fallback)
```

Detection is one-time at startup. Run `yowo info` to see what is available on your system.

### Override backend

```bash
# CLI
yowo detect image.jpg --backend pytorch
yowo detect image.jpg --backend onnx
```

```python
# Python
from yowo.types import BackendType
eng = InferenceEngine(backend=BackendType.ONNX)
```

### Backend capabilities summary

| Backend | Hardware | Notes |
|---------|----------|-------|
| `pytorch` | Any CPU, CUDA, MPS (Apple GPU) | Always available if torch installed. FP16 autocast on CUDA. |
| `onnx` | CPU (CoreML EP on macOS, AVX/VNNI on Intel) | 4–5x faster than PyTorch on Apple Silicon via Neural Engine. |
| `tensorrt` | NVIDIA GPU only | Requires TensorRT install. Best GPU throughput. |
| `openvino` | Intel CPU/iGPU | Competitive on Intel hardware. |

### CoreML EP — ONNX on Apple Neural Engine

When `onnxruntime` is installed on macOS, CoreML EP is auto-detected and enabled. The Neural Engine handles 95–98% of ONNX nodes, delivering **4–5x faster inference** than PyTorch CPU. No user configuration required — just use `--backend onnx`.

**Step 1 — export your model to ONNX (one-time):**

```bash
yowo export yolo26n --format onnx --precision fp16 --dynamic-batch
# output: ~/.yowo/models/yolo26n/onnx_fp16/yolo26n.onnx
```

**Step 2 — detect (CoreML EP is auto-selected):**

```bash
yowo detect image.jpg --model yolo26n \
  --weights ~/.yowo/models/yolo26n/onnx_fp16/yolo26n.onnx \
  --backend onnx

# Verify CoreML EP is active
yowo info
# onnxruntime:  1.24.2 (CoreML)
```

```python
from pathlib import Path
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source
from yowo.types import ModelFamily, ModelSize, BackendType, Precision

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    weights_path=Path("~/.yowo/models/yolo26n/onnx_fp16/yolo26n.onnx").expanduser(),
    backend=BackendType.ONNX,   # CoreML EP selected automatically on macOS
    precision=Precision.FP16,
)

with InferenceEngine(config) as eng:
    for det in eng.stream(open_source("image.jpg")):
        print(f"{det.num_boxes} detections in {det.inference_time_ms:.1f}ms")
        for box in det.boxes:
            print(f"  {box.class_name}: {box.confidence:.2f}")
```

**Provider chain on macOS:** `CoreMLExecutionProvider` → `CPUExecutionProvider`. The Neural Engine (16 TOPS on M4 Pro) handles convolutions, matrix multiplications, and activations; the remaining reshape/transpose ops fall back to CPU.

**Do not use KV-cache ONNX models with CoreML EP** — the Neural Engine re-executes the full graph on every call, so KV cache adds I/O overhead with no compute savings (~3% slower). Use standard ONNX exports.

### MPS — PyTorch on Apple GPU (Metal)

MPS uses the PyTorch backend with `device="mps"`. No ONNX export needed — loads `.pt` weights directly onto the Metal GPU. Delivers ~1.3x faster inference than ultralytics on M-series chips.

```bash
yowo detect image.jpg --model yolo26n \
  --weights ./yolo26n.pt \
  --backend pytorch \
  --device mps
```

```python
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source
from yowo.types import ModelFamily, ModelSize, BackendType

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    backend=BackendType.PYTORCH,
    device="mps",
)

with InferenceEngine(config) as eng:
    for det in eng.stream(open_source("video.mp4")):
        print(f"[{det.frame.frame_index}] {det.num_boxes} detections")
```

yowo includes a workaround for a PyTorch MPS SDPA shape bug (`F.scaled_dot_product_attention` returns wrong output shape when `key_dim != head_dim`). Detection results are numerically correct.

### Choosing between CoreML EP and MPS

| | ONNX + CoreML EP | PyTorch + MPS |
|---|---|---|
| Speed (YOLO26n) | **170 FPS** | 131 FPS |
| Speedup vs PyTorch CPU | **4–5x** | 3x |
| Requires export step | Yes (`.onnx`) | No (`.pt` direct) |
| KV cache support | No (use standard ONNX) | Yes |
| Precision options | FP32, FP16 | FP32, FP16 (autocast) |
| Recommended for | Production throughput | Development / no-export workflow |

---

## 6. Streaming Pipeline

`engine.stream(source)` auto-dispatches based on source type. All behavior is controlled through `InferenceEngine` constructor parameters.

### Source-type dispatch

| Source | Dispatch | Behavior |
|--------|----------|----------|
| Single image | `_stream_single` | No threading overhead |
| RTSP / webcam | `_stream_live` | `ThreadedFrameReader` + `FrameDropPolicy` |
| Video file + `prefetch=True` | `_stream_pipeline` | Background prefetch overlaps I/O with inference |
| Any source + `prefetch=False` | `_stream_sync` | Legacy sequential (zero regression) |

### `FrameDropPolicy`

Controls what happens when the inference thread falls behind the live frame rate:

| Policy | Behavior | When to use |
|--------|----------|-------------|
| `NONE` | Block (backpressure) | Offline video, all frames required |
| `LATEST` | Drop oldest, keep newest | Live streams — stay temporally current |
| `SKIP_OLDEST` | Pop back of queue | Live streams — similar to `LATEST` |

```python
from yowo.types import FrameDropPolicy

eng = InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    frame_drop_policy=FrameDropPolicy.LATEST,
    max_queue_size=4,
)
```

### `ThreadedFrameReader`

Used internally by `_stream_live` but also available directly for custom pipelines:

```python
from yowo.io import ThreadedFrameReader
from yowo.types import FrameDropPolicy

src = open_source("rtsp://camera.local:8554/live")
reader = ThreadedFrameReader(
    src,
    max_queue_size=4,
    drop_policy=FrameDropPolicy.LATEST,
    idle_timeout_s=5.0,
)
reader.start()
try:
    while True:
        frame = reader.get(timeout_s=1.0)
        if frame is None:
            break
        # process frame
finally:
    reader.stop()

print(f"Read {reader.frames_read}, dropped {reader.frames_dropped}")
```

### KV Cache (streaming, PyTorch only)

Enable attention KV cache for YOLO11 streaming sequences. Reuses K, V tensors from the previous frame; recomputes only Q from the current frame. Also applies block-level cache for C2PSA/C3k2PSA blocks when spatial content is stable.

```python
eng = InferenceEngine(
    model_family=ModelFamily.YOLO11,
    model_size=ModelSize.NANO,
    kv_cache=True,
)
```

**When to use KV cache:**
- PyTorch backend on CPU or MPS: +10–12% throughput on video sequences
- ONNX backend (CoreML/CUDA): no benefit — disable for standard ONNX models

### Feature Map Cache (PyTorch only)

Skip backbone+neck for frames with similar visual content:

```python
# In-memory cache
eng = InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    cache=True,
)

# mmap-backed cache (persists across restarts)
eng = InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    cache_dir=Path("/tmp/yowo_cache"),
)
```

### Free-Threaded Python (Python 3.13t)

On Python 3.13+ with the GIL disabled (`python3.13t`), `pipeline_workers` auto-sets to `2`, enabling true CPU parallelism between the I/O decode thread and the inference thread:

```python
from yowo.types import is_free_threaded

print(is_free_threaded())  # True on python3.13t

# Auto-detection — no manual config needed
eng = InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    # pipeline_workers=0 → auto → 2 on 3.13t, 1 on GIL Python
)
```

Measured throughput gain on Apple M4 Pro: **+49%** (58.4 FPS vs 39.3 FPS) with `pipeline_workers=2` on free-threaded Python.

---

## 7. Model Export

### From CLI

```bash
# ONNX FP16, dynamic batch (recommended for Apple Silicon)
yowo export yolo26n --format onnx --precision fp16 --dynamic-batch

# ONNX FP32 for a custom fine-tuned model
yowo export yolo11s --weights ./best.pt --format onnx --precision fp32 --dynamic-batch -o ./exports/

# TensorRT FP16 on NVIDIA
yowo export yolo26m --format tensorrt --precision fp16

# INT8 (requires calibration data)
yowo export yolo11n --format onnx --precision int8 --calibration-data ./calib_images/
```

### From Python

```python
from pathlib import Path
from yowo.export import export_model
from yowo.types import ModelFamily, ModelSize, ModelSpec, ExportFormat, Precision

spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO, weights_path=Path("./yolo26n.pt"))

meta = export_model(
    spec,
    ExportFormat.ONNX,
    Path("./exports/"),
    precision=Precision.FP16,
    dynamic_batch=True,
    imgsz=640,
)
print(f"Exported to {meta.file_path} ({meta.file_size_bytes / 1e6:.1f} MB)")
```

**Export notes:**
- ONNX uses the dynamo path (opset 18). Legacy TorchScript exporter is not supported.
- `onnxslim` graph simplification is applied automatically.
- A `.yowo.json` sidecar file is written alongside the export with model metadata.
- INT8 dynamic quantization is **not recommended on Apple Silicon** — CoreML EP rejects it; use FP16 ONNX instead.

---

## 8. Environment Variables

All `InferenceConfig` fields can be set via `YOWO_*` environment variables. Env vars override YAML values.

| Variable | Config field |
|----------|-------------|
| `YOWO_MODEL_FAMILY` | `model_family` (e.g. `yolo26`) |
| `YOWO_MODEL_SIZE` | `model_size` (e.g. `n`, `s`, `m`, `l`, `x`) |
| `YOWO_WEIGHTS_PATH` | `weights_path` |
| `YOWO_BACKEND` | `backend` (e.g. `onnx`, `pytorch`) |
| `YOWO_DEVICE` | `device` (e.g. `cpu`, `cuda`, `auto`) |
| `YOWO_PRECISION` | `precision` (e.g. `fp32`, `fp16`) |
| `YOWO_CONFIDENCE` | `confidence_threshold` (float) |
| `YOWO_IOU` | `iou_threshold` (float) |
| `YOWO_BATCH_SIZE` | `batch_size` (int) |
| `YOWO_CACHE` | `cache` (`true`/`false`) |
| `YOWO_CACHE_DIR` | `cache_dir` (path) |
| `YOWO_KV_CACHE` | `kv_cache` (`true`/`false`) |
| `YOWO_FRAME_DROP_POLICY` | `frame_drop_policy` (`none`, `latest`, `skip_oldest`) |
| `YOWO_MAX_QUEUE_SIZE` | `max_queue_size` (int) |
| `YOWO_PREFETCH` | `prefetch` (`true`/`false`) |
| `YOWO_PIPELINE_WORKERS` | `pipeline_workers` (int, `0` = auto) |

**Example:**

```bash
export YOWO_BACKEND=onnx
export YOWO_CONFIDENCE=0.3
yowo detect traffic.mp4 --model yolo26n
```

---

## 9. YAML Configuration

```yaml
# yowo.yaml
model_family: yolo26
model_size: n
backend: onnx
device: auto
precision: fp16
confidence_threshold: 0.25
iou_threshold: 0.45
batch_size: 1
prefetch: true
frame_drop_policy: latest
max_queue_size: 4
pipeline_workers: 0
```

Load in Python:

```python
from pathlib import Path
from yowo.config import load_config
from yowo.engine import InferenceEngine

config = load_config(Path("yowo.yaml"))
with InferenceEngine(config) as eng:
    ...
```

---

## 10. Error Reference

All exceptions inherit from `YowoError` in `yowo.errors`.

| Exception | When raised |
|-----------|-------------|
| `BackendError` | Base for backend failures |
| `BackendLoadError` | Backend failed to load (missing weights, OOM) |
| `InferenceError` | `detect()` or `stream()` called before `load()`; runtime infer failure |
| `DeviceError` | Requested device unavailable |
| `ModelError` | Base for model failures |
| `ModelNotFoundError` | Model name not in registry |
| `ModelLoadError` | Weights file missing or corrupt |
| `ExportError` | Export operation failed |
| `ExportUnsupportedError` | Backend/format combination not supported |
| `SourceError` | Input source unavailable or unrecognized |
| `SourceTimeoutError` | RTSP reconnect budget exhausted |
| `ConfigError` | Invalid `InferenceConfig` / `ExportConfig` field value |
| `DependencyError` | Required package not installed (e.g., `onnxruntime`) |

**Example error handling:**

```python
from yowo.errors import YowoError, BackendLoadError, SourceError

try:
    with InferenceEngine(backend=BackendType.TENSORRT) as eng:
        src = open_source("rtsp://camera.local:8554/live")
        for det in eng.stream(src):
            process(det)
except BackendLoadError as e:
    print(f"Backend failed: {e}")
except SourceError as e:
    print(f"Source unavailable: {e}")
except YowoError as e:
    print(f"Unexpected error: {e}")
```

---

## 11. Use Cases

### 11.1 Traffic Surveillance on Apple Silicon

**Scenario:** Process a 2560×1440 traffic surveillance image. Maximize throughput on Apple M-series.

**Recommendation:** YOLO26n or YOLO11s via ONNX FP16 (CoreML EP). Both achieve 100–170 FPS. YOLO11s with domain-specific fine-tuning provides higher per-class confidence. YOLO26m adds 80-class COCO detection for non-vehicle objects.

```bash
# Export first (one-time)
yowo export yolo11s --weights ./best.pt --format onnx --precision fp16 --dynamic-batch -o ./models/

# Detect
yowo detect traffic.jpg --model yolo11s --weights ./models/yolo11s/onnx_fp16/best.pt.onnx \
    --backend onnx --confidence 0.25 -o result.jpg
```

```python
from pathlib import Path
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source, write_annotated_frame
from yowo.types import ModelFamily, ModelSize, BackendType, Precision

config = InferenceConfig(
    model_family=ModelFamily.YOLO11,
    model_size=ModelSize.SMALL,
    weights_path=Path("./models/best_fp16.onnx"),
    backend=BackendType.ONNX,      # auto-selects CoreML EP on macOS
    precision=Precision.FP16,
    confidence_threshold=0.25,
)

with InferenceEngine(config) as eng:
    src = open_source("traffic.jpg")
    for det in eng.stream(src):
        print(f"Detected {det.num_boxes} objects in {det.inference_time_ms:.1f}ms")
        for box in det.boxes:
            print(f"  {box.class_name} ({box.confidence:.2f})")
        write_annotated_frame(det, Path("result.jpg"))
```

**Observed performance (Apple M4 Pro, batch=1):**
- YOLO11s ONNX FP16 CoreML: **~17 FPS** (single image)
- YOLO11s ONNX FP16 CoreML batch=8: **~18 FPS**
- YOLO26n ONNX CoreML: **~170 FPS**

---

### 11.2 Live RTSP Camera Stream

**Scenario:** Process a live IP camera stream. Stay current — drop stale frames; don't queue up.

```python
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source
from yowo.types import ModelFamily, ModelSize, BackendType, FrameDropPolicy

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    backend=BackendType.ONNX,          # CoreML EP on macOS, CUDA EP on NVIDIA
    confidence_threshold=0.3,
    frame_drop_policy=FrameDropPolicy.LATEST,  # drop stale frames
    max_queue_size=4,
    prefetch=True,
)

with InferenceEngine(config) as eng:
    src = open_source(
        "rtsp://camera.local:8554/stream",
        reconnect_timeout_s=60.0,  # retry for up to 60s on disconnect
    )
    for det in eng.stream(src):
        if det.has_detections:
            print(f"[{det.frame.frame_index}] {det.num_boxes} detections @ {det.inference_time_ms:.1f}ms")
```

**Notes on RTSP performance:**
- Throughput is bounded by the stream delivery rate. If the camera delivers 10 FPS, yowo processes at 10 FPS regardless of inference speed.
- `FrameDropPolicy.LATEST` ensures you always process the current frame, not a backlog.
- CoreML EP speedup over PyTorch (3.5–3.75x offline) becomes visible only when the stream delivers frames faster than PyTorch inference (~40ms/frame for YOLO26n).

---

### 11.3 Custom Fine-Tuned Vehicle Detector

**Scenario:** Run a YOLO11s model fine-tuned on 7 vehicle classes (car, motorcycle, bus, truck, transporter, container, big_transporter).

```python
from pathlib import Path
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source, write_annotated_frames
from yowo.types import ModelFamily, ModelSize, BackendType

config = InferenceConfig(
    model_family=ModelFamily.YOLO11,
    model_size=ModelSize.SMALL,
    weights_path=Path("./best.pt"),
    backend=BackendType.PYTORCH,
    confidence_threshold=0.25,
)

with InferenceEngine(config) as eng:
    src = open_source("./surveillance_footage/", max_frames=500)
    results = list(eng.stream(src))

write_annotated_frames(results, Path("./annotated/"))
print(f"Processed {len(results)} frames")

# Summary
all_boxes = [box for det in results for box in det.boxes]
from collections import Counter
class_counts = Counter(box.class_name for box in all_boxes)
for name, count in class_counts.most_common():
    print(f"  {name}: {count}")
```

**Tip:** For production throughput on Apple Silicon, export the fine-tuned weights to ONNX FP16 first:

```bash
yowo export yolo11n --weights ./best.pt --format onnx --precision fp16 --dynamic-batch
```

Then point `weights_path` at the exported `.onnx` file with `backend=BackendType.ONNX`.

---

### 11.4 Batch Video Processing

**Scenario:** Process a long video file and save per-frame JSON results. Use prefetch for maximum throughput.

```python
import json
from pathlib import Path
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source, write_json
from yowo.types import ModelFamily, ModelSize, BackendType, Precision

config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    backend=BackendType.ONNX,
    precision=Precision.FP16,
    confidence_threshold=0.25,
    batch_size=4,
    prefetch=True,        # overlaps decode with inference
    pipeline_workers=0,   # auto: 2 on free-threaded Python, 1 on GIL Python
)

with InferenceEngine(config) as eng:
    src = open_source("video.mp4", frame_skip=1)  # process every other frame
    results = list(eng.stream(src))

write_json(results, Path("detections.json"))
total_boxes = sum(d.num_boxes for d in results)
avg_ms = sum(d.inference_time_ms for d in results) / len(results)
print(f"{len(results)} frames, {total_boxes} total detections, avg {avg_ms:.1f}ms/frame")
```

---

### 11.5 Free-Threaded Python Parallelism

**Scenario:** Maximize CPU throughput using Python 3.13 free-threaded build (`cp313t`, GIL disabled). Enables true thread parallelism between I/O decode and inference — no GPU required.

#### Known constraints before you start

| Package | Status on cp313t |
|---------|-----------------|
| `torch` | Available (2.10.0+) |
| `numpy` | Available (2.4.2+) |
| `opencv-python-headless` | **No cp313t wheel** — cannot use `open_source()` with real video/image files |
| `onnxruntime` | **No macOS arm64 cp313 wheel** — PyTorch backend only |

The benchmark uses **synthetic frames** (pre-generated NumPy arrays) to isolate the parallelism gain without needing cv2.

---

#### Step 1 — Install Python 3.13t

##### Option A: pyenv (recommended)

```bash
# Install pyenv if not present
brew install pyenv

# Install free-threaded Python 3.13
pyenv install 3.13t

# Verify: GIL should be OFF
pyenv exec 3.13t python -c "import sys; print(sys._is_gil_enabled())"
# False
```

##### Option B: python.org installer

Download the macOS `3.13.x freethreaded` pkg from `python.org/downloads/`. After install:

```bash
python3.13t -c "import sys; print(sys._is_gil_enabled())"
# False
```

---

#### Step 2 — Create an isolated venv

```bash
# Using pyenv
pyenv exec 3.13t python -m venv /tmp/yowo-3t

# Or using direct binary
python3.13t -m venv /tmp/yowo-3t

# Activate
source /tmp/yowo-3t/bin/activate

# Verify interpreter
python --version
# Python 3.13.x experimental free-threaded build
python -c "import sys; print(sys._is_gil_enabled())"
# False
```

---

#### Step 3 — Install dependencies

```bash
# Inside the activated /tmp/yowo-3t venv
pip install torch numpy

# Install yowo from source (editable, so src/ is on the path)
pip install -e /path/to/yowo
```

`onnxruntime` and `opencv-python` are intentionally skipped — no cp313t wheels exist yet.

---

#### Step 4 — Pre-extract model weights

Standard `.pt` checkpoints contain ultralytics-pickled objects, which require the ultralytics package and are blocked by `weights_only=True` in Python 3.13. Extract a plain state dict once using the GIL interpreter:

```bash
# Run this on your normal Python (3.11 or 3.12) with ultralytics installed
uv run python - <<'EOF'
import torch
from pathlib import Path

src = Path("tmp/weights/yolo26n.pt")
ckpt = torch.load(src, map_location="cpu", weights_only=False)

# Extract state dict from ultralytics checkpoint
model = ckpt.get("model") or ckpt
state = model.state_dict() if hasattr(model, "state_dict") else model

out = src.with_name(src.stem + "_statedict.pt")
torch.save(state, out)
print(f"Saved: {out}")
EOF
# Saved: tmp/weights/yolo26n_statedict.pt
```

The resulting `_statedict.pt` file is a plain dict — `weights_only=True` safe on any Python version.

---

#### Step 5 — Run the parallelism benchmark

```bash
source /tmp/yowo-3t/bin/activate
python tmp/bench_freethreaded.py
```

Expected output on free-threaded Python:

```text
Python 3.13.x  GIL=OFF (free-threaded)

Loading YOLO26n... done in 0.8s

────────────────────────────────────────────────────────────────
  1. Single-thread inference throughput
────────────────────────────────────────────────────────────────
  200 inferences in 4.90s → 40.8 FPS (24.5 ms/frame)

────────────────────────────────────────────────────────────────
  2. Two-thread concurrent inference
────────────────────────────────────────────────────────────────
  200 inferences (2×100) wall=3.34s → 59.9 FPS
  Thread times: 3.31s, 3.33s
  Speedup vs single-thread: 1.47x
  [GIL=OFF] True parallelism — expected ~2.0x speedup

────────────────────────────────────────────────────────────────
  3. Phase 3 _stream_pipeline with synthetic FrameSource
────────────────────────────────────────────────────────────────
  pipeline_workers=1  → 39.3 FPS  (200 frames in 5.09s)
  pipeline_workers=2  → 58.4 FPS  (200 frames in 3.43s)
  Parallel speedup: 1.49x  (expected ~1.5-2.0x on free-threaded Python)
```

Compare against GIL Python in the same shell:

```bash
deactivate
uv run python tmp/bench_freethreaded.py
# GIL=ON  → 2-thread speedup: 1.07x  (threads serialized)
```

---

#### Step 6 — Use the engine on 3.13t

```python
from pathlib import Path
from yowo.types import is_free_threaded, ModelFamily, ModelSize, Frame, FrameDropPolicy
from yowo.engine import InferenceEngine
from yowo.io import ThreadedFrameReader
import numpy as np

print(f"Free-threaded: {is_free_threaded()}")  # True

# Synthetic source — no cv2 required
class SyntheticSource:
    is_live = False
    total_frames = 200

    def __init__(self) -> None:
        self._frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    def __iter__(self):
        for i in range(self.total_frames):
            yield Frame(pixels=self._frame, source_id="synthetic", frame_index=i)

    def close(self) -> None:
        pass

# pipeline_workers=0 → auto-detects GIL=OFF → sets workers=2
with InferenceEngine(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    weights_path=Path("tmp/weights/yolo26n_statedict.pt"),
    prefetch=True,
    pipeline_workers=0,   # 2 on 3.13t, 1 on GIL Python
) as eng:
    results = list(eng.stream(SyntheticSource()))

print(f"Processed {len(results)} frames")
print(f"is_free_threaded: {is_free_threaded()}")
```

---

#### Observed performance (Apple M4 Pro, YOLO26n)

| Setup | FPS | Notes |
|-------|-----|-------|
| GIL Python 3.11, single-thread | 39.5 | Baseline |
| Free-threaded 3.13t, single-thread | 40.8 | ~same — GIL removal has no single-thread effect |
| GIL Python 3.11, 2 threads | 42.3 | +7% — threads serialize at GIL |
| Free-threaded 3.13t, 2 threads | **59.9** | **+51%** — true CPU parallelism |
| Free-threaded 3.13t, pipeline workers=2 | **58.4** | **+49%** — auto-enabled by `is_free_threaded()` |

Speedup caps at ~1.5x (not 2.0x theoretical) due to L3 cache contention between threads on the same die.

---

### 11.6 ONNX CoreML EP vs PyTorch MPS on Apple Silicon

**Scenario:** You have an Apple Silicon Mac and want to use GPU-class acceleration. Two paths are available — ONNX via CoreML EP (Neural Engine) and PyTorch via Metal (GPU). This example runs both and compares.

**Prerequisites:**

```bash
pip install yowo[onnx]   # enables CoreML EP

# Export once (for CoreML path)
yowo export yolo26n --format onnx --precision fp16 --dynamic-batch
```

**CoreML EP path (ONNX, Neural Engine):**

```python
from pathlib import Path
from yowo.config import InferenceConfig
from yowo.engine import InferenceEngine
from yowo.io import open_source
from yowo.types import ModelFamily, ModelSize, BackendType, Precision

# CoreML EP: fastest on Apple Silicon — 4-5x vs PyTorch CPU
coreml_config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    weights_path=Path("~/.yowo/models/yolo26n/onnx_fp16/yolo26n.onnx").expanduser(),
    backend=BackendType.ONNX,       # CoreML EP auto-selected on macOS
    precision=Precision.FP16,
    confidence_threshold=0.25,
)

with InferenceEngine(coreml_config) as eng:
    src = open_source("video.mp4")
    results = list(eng.stream(src))
    avg_ms = sum(d.inference_time_ms for d in results) / len(results)
    print(f"CoreML EP: {avg_ms:.1f}ms/frame  ({1000/avg_ms:.0f} FPS)")
```

**MPS path (PyTorch, Metal GPU):**

```python
# MPS: no export needed, loads .pt directly onto Metal GPU
mps_config = InferenceConfig(
    model_family=ModelFamily.YOLO26,
    model_size=ModelSize.NANO,
    weights_path=Path("./yolo26n.pt"),
    backend=BackendType.PYTORCH,
    device="mps",
    confidence_threshold=0.25,
)

with InferenceEngine(mps_config) as eng:
    src = open_source("video.mp4")
    results = list(eng.stream(src))
    avg_ms = sum(d.inference_time_ms for d in results) / len(results)
    print(f"MPS:       {avg_ms:.1f}ms/frame  ({1000/avg_ms:.0f} FPS)")
```

**Side-by-side results (YOLO26n, Apple M4 Pro, batch=1):**

| Backend | Latency | FPS | Requires export |
|---------|---------|-----|-----------------|
| PyTorch CPU | 23ms | 43 | No |
| PyTorch MPS | 7.6ms | 131 | No |
| ONNX CoreML EP | **5.9ms** | **170** | Yes (`.onnx`) |

**Decision guide:**

- Use **ONNX CoreML EP** for production — highest throughput, Neural Engine runs 24/7 at low power.
- Use **PyTorch MPS** for development or when you cannot export (custom ops, rapid iteration).
- Do not combine: ONNX Runtime has no MPS execution provider; Metal is PyTorch-only.

**Verify which execution provider is active:**

```python
with InferenceEngine(coreml_config) as eng:
    print(eng.selection)
    # BackendSelection(backend=<BackendType.ONNX>, device='cpu', ...)
    # actual EP shown via: yowo info → onnxruntime: 1.24.2 (CoreML)
```

---

## 12. Performance Reference

### Backend comparison (YOLO26n, Apple M4 Pro, batch=1)

| Backend | Latency | FPS | Notes |
|---------|---------|-----|-------|
| PyTorch CPU (FP32) | 23ms | 43 | Baseline |
| ONNX CoreML EP (FP32) | 6ms | **170** | 4x faster, auto-selected on macOS |
| PyTorch MPS | 7.6ms | 131 | Apple GPU; manual `device="mps"` |
| ONNX CPU EP | 32ms | 31 | Slower than PyTorch on Apple Silicon |

### Model selection guide

| Scenario | Recommended | FPS (CoreML) | Notes |
|----------|-------------|--------------|-------|
| Max throughput, any scene | YOLO26n | **170** | NMS-free head |
| Balanced accuracy/speed | YOLO11s or YOLO26s | 100–113 | |
| High accuracy, real-time | YOLO26m | 57 | |
| High accuracy, offline | YOLO26x | 27 | Still real-time on Neural Engine |
| Custom fine-tuned | Your YOLO11s | 100–113 | Export to ONNX FP16 for production |

### Batch size trade-offs (YOLO11s ONNX, Apple M4 Pro)

| Batch | FPS FP32 | FPS FP16 | Notes |
|-------|----------|----------|-------|
| 1 | 16.8 | 16.7 | Low latency |
| 4 | 12.3 | 12.1 | Diminishing return |
| 8 | 16.2 | **18.1** | FP16 wins at batch=8 |

### Do not use INT8 on Apple Silicon

Dynamic INT8 quantization (ORT `quantize_dynamic`) is **0.64–0.81× of PyTorch baseline** on Apple Silicon — slower than all alternatives. CoreML EP rejects dynamically quantized models; they fall back to a scalar CPU path. Use FP16 ONNX instead.

### `torch.compile` on CPU: avoid

`compile_for_inference()` causes 30–43% regression on CPU (inductor overhead exceeds savings for YOLO11/26 sizes). Only use on CUDA where it provides 20–40% gain.

### KV cache: PyTorch only

| Backend | KV cache effect |
|---------|-----------------|
| PyTorch CPU | +10–12% on video sequences |
| PyTorch MPS | +10% |
| ONNX CoreML | ≈-3% (I/O overhead, no benefit) |
| ONNX CPU | ≈-2% |

Enable KV cache only with the PyTorch backend on video sequences with stable frames.
