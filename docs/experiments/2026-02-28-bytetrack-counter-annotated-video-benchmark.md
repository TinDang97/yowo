# Experiment Report: ByteTrack + ObjectCounter — Real-Video Annotated Benchmark

**Date**: 2026-02-28
**Author**: Tin Dang
**Hardware**: Apple M4 Pro (12-core CPU, 16-core GPU, 16-core Neural Engine)
**Platform**: macOS 25.3.0, Python 3.11.11
**Branch**: `perf/inference-optimization`
**Version**: yowo v2.1.0

---

## Objective

Validate the v2.1.0 ByteTrack tracker and ObjectCounter end-to-end on real surveillance video across all 5 YOLO26 model variants and 2 accelerated backends:

1. **ONNX + CoreML EP** — ONNX Runtime with CoreML Execution Provider (Neural Engine)
2. **PyTorch + MPS** — Native PyTorch with Metal Performance Shaders (GPU compute)

Measure: inference FPS, tracking overhead, unique object detection quality, and line-crossing accuracy.

**Video under test**: `MCT 1.2.mp4` — 928 frames, 1280x720, 30fps real-world traffic surveillance
**Models**: YOLO26 n/s/m/l/x (COCO 80-class)

---

## Pipeline Architecture

```
VideoSource → YOLO Detect → ByteTrack → ObjectCounter → Annotated Video
   (decode)     (infer)      (track)      (zone+line)    (cv2 draw+write)
```

Each frame goes through:
1. **Decode**: `cv2.VideoCapture` frame read
2. **Inference**: YOLO26 detection (backend-dependent)
3. **Tracking**: ByteTrack 2-stage association (Kalman predict → IoU match → update)
4. **Counting**: Zone occupancy (top/bottom halves) + horizontal line crossing (IN/OUT)
5. **Annotation**: Bounding boxes colored by track_id, zone overlays, counting line, stats panel

---

## Tracker & Counter Configuration

```python
ByteTracker(
    track_high_thresh=0.3,    # stage-1 confidence threshold
    track_low_thresh=0.1,     # stage-2 low-confidence recovery
    match_thresh=0.8,         # IoU matching threshold
    max_age=30,               # frames before track removal
    min_hits=3,               # confirmations before track is confirmed
)

ObjectCounter(
    zones=[ZONE_TOP, ZONE_BOTTOM],  # top/bottom halves of frame
    lines=[LINE_MID],               # horizontal line at y=360
)
```

---

## Results: ONNX + CoreML EP (Neural Engine)

| Model | FPS | Infer (ms) | P95 (ms) | Track (ms) | Track % | Unique Objects |
|---------|------|------------|----------|------------|---------|----------------|
| YOLO26n | **82.3** | 6.41 | 7.07 | 0.333 | 5.2% | 96 |
| YOLO26s | 58.2 | 10.98 | 12.37 | 0.404 | 3.7% | 127 |
| YOLO26m | 36.6 | 20.04 | 22.34 | 0.543 | 2.7% | 191 |
| YOLO26l | 32.8 | 22.55 | 23.97 | 0.505 | 2.2% | 156 |
| YOLO26x | 21.1 | 37.46 | 38.68 | 0.688 | 1.8% | 202 |

### Detection Breakdown (CoreML, per-class unique tracked objects)

| Class | n | s | m | l | x |
|-------|---|---|---|---|---|
| car | 75 | 94 | 133 | 107 | 115 |
| motorcycle | 11 | 6 | 18 | 12 | 29 |
| truck | 6 | 14 | 16 | 16 | 26 |
| person | 1 | 10 | 11 | 9 | 22 |
| bus | 1 | 2 | 3 | 5 | 6 |
| stop sign | - | - | 5 | 5 | 4 |
| traffic light | 2 | 1 | 5 | 2 | - |
| fire hydrant | - | 1 | - | - | - |

### Line Crossing (CoreML)

| Model | IN | OUT |
|---------|-----|------|
| YOLO26n | 1 | 18 |
| YOLO26s | 1 | 18 |
| YOLO26m | 1 | 18 |
| YOLO26l | 1 | 18 |
| YOLO26x | 1 | 19 |

---

## Results: PyTorch + MPS (Metal GPU)

| Model | FPS | Infer (ms) | P95 (ms) | Track (ms) | Track % | Unique Objects |
|---------|------|------------|----------|------------|---------|----------------|
| YOLO26n | 71.8 | 8.56 | 9.91 | 0.333 | 3.9% | 94 |
| YOLO26s | **63.4** | 10.67 | 11.00 | 0.405 | 3.8% | 124 |
| YOLO26m | 35.2 | 23.15 | 24.24 | 0.540 | 2.3% | 186 |
| YOLO26l | 28.8 | 29.27 | 31.21 | 0.532 | 1.8% | 155 |
| YOLO26x | 16.0 | 56.26 | 61.29 | 0.726 | 1.3% | 204 |

---

## Cross-Backend Comparison

| Model | CoreML FPS | MPS FPS | Winner | Speedup |
|---------|-----------|---------|---------|---------|
| YOLO26n | **82.3** | 71.8 | CoreML | **1.15x** |
| YOLO26s | 58.2 | **63.4** | MPS | **1.09x** |
| YOLO26m | **36.6** | 35.2 | CoreML | 1.04x |
| YOLO26l | **32.8** | 28.8 | CoreML | **1.14x** |
| YOLO26x | **21.1** | 16.0 | CoreML | **1.32x** |

### Inference Latency Comparison

| Model | CoreML (ms) | MPS (ms) | Delta |
|---------|------------|----------|-------|
| YOLO26n | 6.41 | 8.56 | CoreML 25% faster |
| YOLO26s | 10.98 | 10.67 | MPS 3% faster |
| YOLO26m | 20.04 | 23.15 | CoreML 13% faster |
| YOLO26l | 22.55 | 29.27 | CoreML 23% faster |
| YOLO26x | 37.46 | 56.26 | CoreML 33% faster |

---

## Analysis

### Backend Selection

- **CoreML EP wins on 4 of 5 variants.** The Neural Engine is especially efficient for larger models — 1.32x faster on YOLO26x (37.46ms vs 56.26ms). ORT fuses Sigmoid+Mul into QuickGelu and offloads 95%+ of nodes to the Neural Engine.
- **MPS wins only on YOLO26s** by 9%. Metal GPU shaders have lower dispatch overhead for small models where the Neural Engine compilation overhead is proportionally larger.
- **Recommendation**: Use ONNX+CoreML EP as default on Apple Silicon. Only consider MPS for small models in latency-sensitive scenarios.

### Tracking Overhead

- ByteTrack adds **0.3-1.8ms per frame** (1.3-5.2% of inference time).
- Overhead scales with detection count, not model size — larger models produce more boxes.
- The 2-stage Kalman + IoU matching is well under the 10% overhead budget.

### Detection Quality vs Model Size

- **Nano (n)**: 96 unique objects — good for vehicles, misses small objects (people, signs).
- **Small (s)**: 127 objects — picks up more people and trucks.
- **Medium (m)**: 191 objects — best balance of speed (36 FPS real-time) and detection quality.
- **Large (l)**: 156 objects — slightly fewer than medium (different tracking dynamics with more confident detections).
- **XLarge (x)**: 202 objects — most comprehensive detection but 16-21 FPS, below real-time for 30fps video.

### Line Crossing Consistency

All variants produce consistent line crossing counts (IN=1, OUT=17-19), demonstrating stable tracking across model sizes. The minor OUT variation (17-19) is due to detection confidence differences near the counting line boundary.

---

## Annotated Video Outputs

| File | Size | Backend | Model |
|------|------|---------|-------|
| `MCT_1.2_yolo26n_onnx.mp4` | 37MB | CoreML | Nano |
| `MCT_1.2_yolo26s_onnx.mp4` | 38MB | CoreML | Small |
| `MCT_1.2_yolo26m_onnx.mp4` | 39MB | CoreML | Medium |
| `MCT_1.2_yolo26l_onnx.mp4` | 38MB | CoreML | Large |
| `MCT_1.2_yolo26x_onnx.mp4` | 42MB | CoreML | XLarge |
| `MCT_1.2_yolo26n_mps.mp4` | 37MB | MPS | Nano |
| `MCT_1.2_yolo26s_mps.mp4` | 38MB | MPS | Small |
| `MCT_1.2_yolo26m_mps.mp4` | 38MB | MPS | Medium |
| `MCT_1.2_yolo26l_mps.mp4` | 38MB | MPS | Large |
| `MCT_1.2_yolo26x_mps.mp4` | 42MB | MPS | XLarge |

Each annotated video includes:
- Colored bounding boxes per unique track_id
- Track ID + class + confidence labels
- Semi-transparent zone overlays (top/bottom)
- Red counting line with label
- Real-time stats panel: model name, FPS, inference/tracking latency, active/lost tracks, unique object counts, line crossing IN/OUT

---

## Reproduction

```bash
# All variants, both backends
uv run python tmp/experiment_annotated_video.py --backend both

# Single variant
uv run python tmp/experiment_annotated_video.py --size n --backend onnx
uv run python tmp/experiment_annotated_video.py --size x --backend pytorch

# CoreML only (fastest on Apple Silicon)
uv run python tmp/experiment_annotated_video.py --backend onnx
```

### Prerequisites

- ONNX exports: `tmp/exports/yolo26{n,s,m,l,x}.onnx`
- PyTorch state dicts: `tmp/weights/yolo26{n,s,m,l,x}_statedict.pt`
- Video: `tmp/MCT 1.2.mp4` (1280x720, 30fps, 928 frames)

---

## Companion Experiments

| Script | Purpose | Tests |
|--------|---------|-------|
| `tmp/experiment_tracker.py` | ByteTrack validation on animated video (Big Buck Bunny) | 11/11 PASS |
| `tmp/experiment_counter.py` | ObjectCounter validation (zones, lines, reset) | 18/18 PASS |
| `tmp/experiment_annotated_video.py` | Full pipeline benchmark + annotated video output | This report |

---

## Quality Gates

- 1028 unit tests pass
- 0 pyright errors
- 0 ruff errors
- ByteTrack stage-2 semantic fix applied (update vs re_activate for TRACKED tracks)
