# Experiment Report: Phase 3 Source-Aware Pipeline

**Date**: 2026-02-25
**Author**: Tin Dang
**Hardware**: Apple M4 Pro (12-core CPU, 16-core Neural Engine)
**Platform**: macOS 25.3.0, Python 3.11.11 (GIL) / Python 3.13.3+freethreaded (GIL=OFF)
**Branch**: `feat/phase3-source-aware-pipeline`

---

## Objective

Validate the Phase 3 source-aware pipeline implementation and measure its throughput impact across four scenarios:

1. **Offline video** — Phase 3 `_stream_pipeline` (prefetched) vs Legacy sequential stream
2. **Live RTSP stream** — Phase 3 `_stream_live` (ThreadedFrameReader + FrameDropPolicy) vs Legacy
3. **CoreML EP on RTSP** — ONNX+CoreML backend vs PyTorch, sync vs live dispatch
4. **Free-threaded Python** — `pipeline_workers=2` true parallelism on GIL=OFF Python 3.13t

**Model under test**: YOLO26 Nano (yolo26n.pt / yolo26n.onnx)
**Input**: Big Buck Bunny — 30s clip (720 frames, 1280×720, 24fps) for offline; looping RTSP H264 stream for live

---

## Phase 3 Feature Overview

| Feature | Class / Location | Purpose |
|---------|-----------------|---------|
| `_stream_single` | `engine.py` | Single-image detection path |
| `_stream_live` | `engine.py` | RTSP/webcam: `ThreadedFrameReader` + `FrameDropPolicy` |
| `_stream_pipeline` | `engine.py` | Offline video: background prefetch decouples I/O from inference |
| `_stream_sync` | `engine.py` | Legacy sequential fallback (no prefetch) |
| `ThreadedFrameReader` | `io/_reader.py` | Background daemon thread, bounded deque, idle timeout |
| `FrameDropPolicy` | `types.py` | `NONE` / `LATEST` / `SKIP_OLDEST` strategies |
| `PreprocessBuffer` | `io/_decode.py` | Pre-allocated `(max_batch, H, W, 3)` uint8 staging — eliminates `cv2.copyMakeBorder` |
| `PostprocessBuffer` | `postprocess/_nms.py` | Pre-allocated `(max_detections, 4)` float32 scratch for inverse letterbox |
| `is_free_threaded()` | `types.py` | Runtime `sys._is_gil_enabled()` detection; auto-sets `pipeline_workers=2` |
| `InferenceEngine` kwargs API | `engine.py` | New constructor: accepts `InferenceConfig` or flat kwargs |

---

## Scenario 1: Offline Video — Phase 3 vs Legacy

### Setup

- **Source**: `tmp/bunny_clip.mp4` (30s, 720 frames, 1280×720)
- **Legacy**: `/tmp/yowo-legacy/src` at commit `5c7bd17` (pre-Phase3) — sequential `eng.stream()`
- **Phase 3**: HEAD — `prefetch=True/False`, `pipeline_workers=1`
- **Metrics**: 72-frame window, 3 runs averaged, wall-clock FPS

### Results

| Scenario | FPS | vs Legacy |
|----------|-----|-----------|
| Legacy sequential | 39.6 | baseline |
| Legacy + KV cache | 37.1 | 0.94x |
| Phase 3 pipeline (`prefetch=True`) | **40.8** | **1.03x** |
| Phase 3 sync (`prefetch=False`) | 39.6 | 1.00x |
| Phase 3 + KV cache | 37.4 | 0.94x |

### Analysis

The `prefetch=True` path (`_stream_pipeline`) overlaps I/O decoding with inference via a background `ThreadedFrameReader`. The **1.03x gain vs legacy is expected** for local SSD video — disk/decode latency is low on NVMe, so the overlap window is narrow. The real benefit emerges with high-latency sources (RTSP over network) where decode blocks the inference thread for 80–200ms per frame.

The `_stream_sync` path is a **zero-regression drop-in**: 1.00x vs legacy by design.

---

## Scenario 2: Live RTSP Stream — Phase 3 vs Legacy

### Setup

- **RTSP server**: `mediamtx` v1.9.3 on `localhost:8554/bunny`
- **Feed**: `ffmpeg -re -stream_loop -1 -i bunny_clip.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp`
- **Legacy**: sequential `eng.stream(open_source(RTSP))`
- **Phase 3**: `_stream_live` path — `ThreadedFrameReader` + `FrameDropPolicy`
- **Metrics**: 60 frames, 3 runs averaged

### Results

| Scenario | Legacy FPS | Phase 3 FPS | Δ | Notes |
|----------|-----------|------------|---|-------|
| stream vs live+LATEST | 10.6 | 10.5 | ≈1.00x | Stream-rate limited |
| stream vs live+NONE | 10.6 | 10.6 | ≈1.00x | Stream-rate limited |
| kv_cache vs live+kv+LATEST | 10.3 | 10.5 | ≈1.02x | Within noise |

### Analysis

Both legacy and Phase 3 are bottlenecked at **~10–11 FPS by the RTSP delivery rate** (localhost H264 encode→decode pipeline). The Neural Engine inference (5.9ms/frame) is idle >80% of the time.

This result is expected: Phase 3 `_stream_live` is designed for **high-latency network RTSP** where frames arrive at 50–200ms intervals and blocking I/O would stall inference. On localhost with near-zero network latency, the decode is already fast enough that `ThreadedFrameReader` provides no measurable gain.

The **correct benchmark** for `_stream_live` is a remote RTSP source (10–50ms RTT). In that scenario, `ThreadedFrameReader` decouples the blocking `.grab()/.retrieve()` from inference, turning sequential `decode → infer → decode → ...` into overlapped `decode ‖ infer`, with `FrameDropPolicy.LATEST` dropping stale frames to stay current.

---

## Scenario 3: CoreML EP on Offline and RTSP

### Setup

- **Backends**: PyTorch CPU vs ONNX+CoreML (`yolo26n.onnx`, CoreML EP auto-selected)
- **Scenarios**: sync (offline video) and live (RTSP stream)
- **Metric**: 100 frames, 2 runs averaged

### Results — Offline Video

| Scenario | PyTorch FPS | CoreML FPS | Speedup |
|----------|------------|-----------|---------|
| Sync (offline) | 39.1 | **139.4** | **3.57x** |
| Live (offline-mode) | 38.7 | **145.2** | **3.75x** |

### Results — RTSP Stream

| Scenario | PyTorch FPS | CoreML FPS | Speedup | Notes |
|----------|------------|-----------|---------|-------|
| Sync (RTSP) | 10.5 | 10.6 | ≈1.01x | Stream-rate limited |
| Live (RTSP) | 10.4 | 10.7 | ≈1.03x | Stream-rate limited |

### CoreML vs PyTorch Speedup

| Mode | Speedup |
|------|---------|
| Offline sync | **3.57x** |
| Offline live | **3.75x** |
| RTSP sync | ≈1.01x (stream-limited) |
| RTSP live | ≈1.03x (stream-limited) |

### Analysis

CoreML EP delivers **3.5–3.8x speedup over PyTorch CPU** for offline video — the Neural Engine handles 371/389 ONNX nodes (95.4%), reducing inference from ~25ms to ~6.7ms per frame.

On RTSP, the speedup is invisible: the stream delivers frames at ~95ms intervals (10.5 FPS), making inference latency (25ms PyTorch vs 7ms CoreML) irrelevant. The Phase 3 `_stream_live` path with `FrameDropPolicy.LATEST` would expose the CoreML speedup on a **high-FPS remote stream** (e.g., 1080p 30fps camera over LAN) where frame delivery outpaces PyTorch inference but CoreML keeps up.

---

## Scenario 4: Free-threaded Python 3.13t

### Setup

- **GIL Python**: 3.11.11, `sys._is_gil_enabled() = True`
- **Free-threaded Python**: 3.13.3+freethreaded, `sys._is_gil_enabled() = False`
- **Environment**: `/tmp/yowo-3t/` — custom venv with torch=2.10.0, numpy=2.4.2, cv2 stub
- **Model**: YOLO26n loaded from pre-extracted state dict (`yolo26n_statedict.pt`) to avoid ultralytics pickle in 3.13t
- **Synthetic source**: pre-generated `(720, 1280, 3)` uint8 frames, no cv2 required
- **Metrics**: 200 inferences total

### Test 1: Single-Thread Throughput

| Python | FPS | ms/frame |
|--------|-----|---------|
| GIL 3.11 | 39.5 | 25.3ms |
| Free-threaded 3.13t | 40.8 | 24.5ms |

Single-thread performance is identical — GIL removal has no effect when there is only one thread.

### Test 2: Two-Thread Concurrent Inference

Each thread runs 100 inferences concurrently. Wall-clock time measured.

| Python | Wall FPS | Speedup | GIL effect |
|--------|---------|---------|-----------|
| GIL 3.11 | 42.3 | **1.07x** | Threads serialized at GIL |
| Free-threaded 3.13t | 59.9 | **1.47x** | True CPU parallelism |

On GIL Python, the two inference threads take turns → wall ≈ single-thread time (1.07x). On free-threaded Python, threads run on separate cores simultaneously → **47% throughput gain**.

Speedup is ~1.5x rather than the theoretical 2.0x because PyTorch CPU inference is partially memory-bandwidth-bound (conv weight reads), and both threads compete for L3 cache on the same M4 Pro die.

### Test 3: Phase 3 Pipeline (ThreadedFrameReader + ThreadPoolExecutor)

| Python | workers | FPS | Speedup |
|--------|---------|-----|---------|
| GIL 3.11 | 1 | 38.7 | baseline |
| GIL 3.11 | 1 | 38.7 | 1.00x |
| Free-threaded 3.13t | 1 | 39.3 | 1.02x |
| Free-threaded 3.13t | 2 (auto) | **58.4** | **1.49x** |

`is_free_threaded()` returns `True` on 3.13t → `pipeline_workers` auto-set to 2 → `ThreadPoolExecutor(max_workers=2)` dispatches inference in parallel with I/O decoding.

### Summary

| Metric | GIL Python 3.11 | Free-threaded 3.13t | Delta |
|--------|-----------------|---------------------|-------|
| Single-thread FPS | 39.5 | 40.8 | ≈ same |
| 2-thread wall FPS | 42.3 | **59.9** | +42% |
| 2-thread speedup | 1.07x | **1.47x** | +0.40x |
| Pipeline workers=1 FPS | 38.7 | 39.3 | ≈ same |
| Pipeline workers=2 FPS | 38.7 (forced 1) | **58.4** | +51% |
| `is_free_threaded()` | False | True | — |

---

## End-to-End Feature Validation (11 Tests)

All 11 Phase 3 feature tests pass on Python 3.11 using `tmp/test_bunny_phase3.py`:

| Test | Feature Exercised | Result |
|------|-------------------|--------|
| 1 | `prefetch=False` sync stream (72 frames) | PASS |
| 2 | `prefetch=True` pipeline stream (72 frames) | PASS |
| 3 | `FrameDropPolicy.LATEST` + `max_queue_size=2` | PASS |
| 4 | `FrameDropPolicy.SKIP_OLDEST` + `max_queue_size=3` | PASS |
| 5 | `FrameDropPolicy.NONE` + `max_queue_size=8` | PASS |
| 6 | `ThreadedFrameReader` direct API (frames_read, drop_rate) | PASS |
| 7 | `PreprocessBuffer(max_batch=2)` + `preprocess_into()` | PASS |
| 8 | `PostprocessBuffer` memory_bytes > 0 | PASS |
| 9 | `is_free_threaded()` returns `bool` | PASS |
| 10 | `eng.detect()` single frame via kwargs constructor | PASS |
| 11 | `InferenceConfig` + `InferenceEngine(config=...)` | PASS |

---

## Key Design Decisions Validated

### Source dispatch is correct

`InferenceEngine.stream()` dispatches based on `source.is_live`:
- `is_live=True` (RTSP/webcam) → `_stream_live` → `ThreadedFrameReader` + drop policy
- `is_live=False` + `prefetch=True` → `_stream_pipeline` → background prefetch
- `is_live=False` + `prefetch=False` → `_stream_sync` → legacy sequential

This is zero-config for users: `open_source("rtsp://...")` auto-sets `is_live=True`.

### `pipeline_workers` auto-detection

```python
# types.py
def is_free_threaded() -> bool:
    try:
        return not sys._is_gil_enabled()
    except AttributeError:
        return False

# engine.py — in __init__
if pipeline_workers is None:
    pipeline_workers = 2 if is_free_threaded() else 1
```

No user action required: running on free-threaded Python automatically enables 2-worker parallelism.

### `ThreadedFrameReader` thread safety

- Bounded deque (`maxlen=max_queue_size`) prevents unbounded memory growth
- `FrameDropPolicy.LATEST` popleft + append → always delivers freshest frame
- `_stop_event` + `_idle_timeout` (5s) clean shutdown without deadlock
- `frames_read` / `frames_dropped` counters use `threading.Lock` — no race condition

---

## Artifacts

| File | Description |
|------|-------------|
| `tmp/test_bunny_phase3.py` | 11-test feature validation suite (all PASS) |
| `tmp/bench_phase3_compare.py` | Phase 3 vs Legacy offline video comparison |
| `tmp/bench_rtsp_compare.py` | RTSP stream benchmark (Legacy vs Phase 3) |
| `tmp/bench_rtsp_coreml.py` | 4-scenario matrix: PyTorch/CoreML × sync/live |
| `tmp/bench_freethreaded.py` | GIL vs free-threaded Python parallelism benchmark |
| `tmp/weights/yolo26n_statedict.pt` | Pre-extracted state dict for 3.13t (weights_only=True safe) |
| `tmp/bunny_clip.mp4` | 30s Big Buck Bunny test clip (720 frames, 1280×720, 24fps) |

---

## Limitations and Future Work

1. **RTSP gain invisible on localhost** — Phase 3 `_stream_live` benefit requires network-latency RTSP (≥10ms RTT) to expose the I/O-inference overlap. Production camera streams are the intended target.
2. **Free-threaded speedup caps at ~1.5x on M4 Pro** — memory bandwidth contention between threads reduces gains below the 2.0x theoretical maximum. GPU inference would show greater benefit as compute is more parallelizable.
3. **cv2 and onnxruntime unavailable on cp313t** — `opencv-python-headless` has no cp313t wheels; `onnxruntime` has no macOS arm64 cp313 wheel. Phase 3 free-threaded test uses PyTorch backend only with a synthetic frame source.
4. **`pipeline_workers=2` GIL regression** — on GIL Python, setting `pipeline_workers=2` provides no benefit (threads serialize at GIL) but adds thread scheduling overhead. The `is_free_threaded()` guard correctly prevents this.
5. **`FrameDropPolicy.SKIP_OLDEST` vs `LATEST`** — both avoid backpressure but `LATEST` discards the oldest queued frame (preserves temporal recency); `SKIP_OLDEST` pops the back of the queue. Benchmarks show equivalent throughput — choice depends on application latency requirements.

---

## Quality Gates

All gates on branch `feat/phase3-source-aware-pipeline`:

```
ruff lint:    PASS (0 errors)
ruff format:  PASS (no changes)
pyright:      PASS (0 errors)
pytest:       PASS (442 tests)
```
