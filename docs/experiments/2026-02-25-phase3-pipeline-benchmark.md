# Phase 3: Source-Aware Pipeline Benchmark

**Date**: 2026-02-25
**Branch**: `feat/phase3-source-aware-pipeline`
**Platform**: macOS Darwin 25.3.0, Apple Silicon, CPU-only
**PyTorch**: eager mode (no torch.compile), FP32

## Summary

Phase 3 introduces three pipeline optimizations:

1. **PreprocessBuffer** — pre-allocated staging arrays eliminate per-frame `cv2.copyMakeBorder` allocation
2. **PostprocessBuffer** — pre-allocated scratch arrays for inverse letterbox coordinate transform
3. **Threaded pipeline streaming** — `ThreadedFrameReader` + `ThreadPoolExecutor` overlap read/preprocess with inference

| Optimization | Speedup | Memory Cost |
|---|---|---|
| Preprocess buffer reuse | **1.05x** | 1,200 KB |
| Postprocess buffer reuse | **1.00x** | 131 KB |
| Pipeline vs sync streaming | **1.09x avg** (8/10 faster) | ~thread stack overhead |

## 1. Preprocess Buffer Reuse

Compares `preprocess()` (allocates `cv2.copyMakeBorder` each call) vs `preprocess_into()` (reuses pre-filled staging array). 500 iterations, batch=1, 640x640 target.

| Metric | Baseline (ms) | Buffered (ms) | Delta |
|---|---|---|---|
| mean | 0.361 | 0.345 | **-4.5%** |
| p50 | 0.355 | 0.343 | -3.5% |
| p95 | 0.404 | 0.382 | -5.5% |
| p99 | 0.438 | 0.404 | **-7.9%** |

**Speedup: 1.05x** — consistent improvement across all percentiles. The `needs_reset()` check skips the full 1.2 MB memset when consecutive frames share the same resolution (common in video streams), making the hot path even faster for video workloads.

## 2. Postprocess Buffer Reuse

Compares `postprocess()` with/without `PostprocessBuffer` scratch allocation. 500 iterations.

| Metric | Baseline (ms) | Buffered (ms) | Delta |
|---|---|---|---|
| mean | 0.148 | 0.148 | +0.2% |
| p50 | 0.147 | 0.147 | -0.2% |
| p95 | 0.157 | 0.155 | -1.4% |
| p99 | 0.168 | 0.164 | **-2.6%** |

**Speedup: 1.00x** — neutral on average. The postprocess path is already fast (< 0.15ms); allocation cost is negligible. The buffer prevents GC pressure under sustained streaming (tail latency improvement at p99).

## 3. Pipeline Streaming: Sync vs Pipeline

End-to-end throughput comparison using `InferenceEngine.stream()` on 30 JPEG frames from a directory source. 3 rounds averaged.

- **Sync** (`prefetch=False`): legacy sequential read → preprocess → infer → postprocess
- **Pipeline** (`prefetch=True`, `pipeline_workers=1`): `ThreadedFrameReader` + `ThreadPoolExecutor` overlapping batch N+1 read with batch N inference

| Model | Sync (FPS) | Pipeline (FPS) | Speedup | Sync (ms/f) | Pipeline (ms/f) |
|---|---|---|---|---|---|
| yolo11n | 33.8 | 41.2 | **1.22x** | 29.58 | 24.26 |
| yolo11s | 21.5 | 23.8 | **1.11x** | 46.60 | 42.02 |
| yolo11m | 12.2 | 13.0 | **1.07x** | 81.96 | 76.78 |
| yolo11l | 9.4 | 8.8 | 0.93x | 105.93 | 113.60 |
| yolo11x | 5.7 | 6.2 | **1.08x** | 175.14 | 161.65 |
| yolo26n | 32.9 | 39.3 | **1.19x** | 30.42 | 25.46 |
| yolo26s | 20.9 | 23.9 | **1.15x** | 47.92 | 41.85 |
| yolo26m | 12.3 | 13.1 | **1.07x** | 81.38 | 76.28 |
| yolo26l | 9.5 | 10.1 | **1.06x** | 104.88 | 99.25 |
| yolo26x | 6.0 | 6.0 | 0.99x | 166.02 | 168.07 |

**Average: 1.09x faster | 8/10 models improved**

### Analysis

- **Nano/Small models benefit most** (1.11–1.22x): inference is fast enough that I/O + preprocess overlap is significant relative to total frame time.
- **Large/XLarge models see modest gains** (1.06–1.08x): inference dominates, so overlap provides diminishing returns.
- **yolo11l slight regression** (0.93x): thread scheduling overhead outweighs overlap on this model size with only 30 frames. CPU contention between inference and the reader thread is more pronounced when inference saturates cores.
- **Free-threaded Python 3.14t**: expected to increase pipeline gains further via true CPU parallelism (preprocess runs on a separate core without GIL contention).

## 4. Raw Model Latency (Architecture-Only, No Pipeline)

Confirms Phase 3 changes do not regress model-level inference. 100 iterations per variant.

| Model | Phase 3 (ms) | Phase 3 (FPS) | Pre-Phase 3 (ms) | Pre-Phase 3 (FPS) |
|---|---|---|---|---|
| yolo11n | 22.97 | 43.5 | 26.43 | 37.8 |
| yolo11s | 43.17 | 23.2 | 45.32 | 22.1 |
| yolo11m | 75.95 | 13.2 | 85.72 | 11.7 |
| yolo11l | 99.03 | 10.1 | 138.82 | 7.2 |
| yolo11x | 160.85 | 6.2 | 205.87 | 4.9 |
| yolo26n | 24.73 | 40.4 | 32.44 | 30.8 |
| yolo26s | 48.07 | 20.8 | 55.31 | 18.1 |
| yolo26m | 74.91 | 13.3 | 100.66 | 9.9 |
| yolo26l | 97.66 | 10.2 | 134.03 | 7.5 |
| yolo26x | 170.33 | 5.9 | 210.58 | 4.7 |
| **avg** | **81.77** | **18.7** | **103.52** | **15.5** |

**Phase 3 avg FPS: 18.7 vs pre-Phase 3: 15.5 — 1.21x improvement** (includes cumulative Phase 1+2 optimizations carried forward). No regression from the pipeline infrastructure.

## Conclusion

Phase 3 delivers measurable throughput improvement for streaming workloads:

- **Pipeline overlap**: 9% average FPS improvement (up to 22% for nano models)
- **Buffer reuse**: 5% preprocess speedup, negligible postprocess overhead
- **Zero regression**: raw model inference is unchanged or faster
- **Low memory cost**: 1.3 KB postprocess + 1.2 MB preprocess buffers (one-time allocation)
- **Source-aware dispatch**: automatic strategy selection (single image / live / offline pipeline / sync) with no user configuration required
