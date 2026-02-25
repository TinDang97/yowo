# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Releases after `v0.1.0` are generated automatically by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
from [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

---

## [1.2.0] — 2026-02-25

### Added

- **io, engine**: Source-aware pipeline dispatch — Phase 3.
  `InferenceEngine.stream()` now inspects `source.is_live` to select the
  optimal execution path automatically:
  - `_stream_live` — live sources (RTSP, webcam): `ThreadedFrameReader` runs
    in a background daemon thread, decoupling network I/O from inference.
    `FrameDropPolicy` controls queue behaviour under backpressure.
  - `_stream_pipeline` — offline video with `prefetch=True`: background
    prefetch overlaps decode and inference to hide I/O latency.
  - `_stream_sync` — offline video with `prefetch=False`: legacy sequential
    path, zero regression guarantee.
  - `_stream_single` — image sources: single-frame fast path.
  No user configuration required — `open_source("rtsp://...")` sets
  `is_live=True` and the engine dispatches accordingly.

- **io**: `ThreadedFrameReader` — bounded background frame reader.
  `io/_reader.py`: daemon thread fills a `deque(maxlen=max_queue_size)` from
  any `FrameSource`. Exposes `frames_read`, `frames_dropped`, `drop_rate`
  counters. Idle timeout (5 s) triggers clean shutdown without deadlock.
  Thread-safe via `threading.Lock` on all shared state.

- **types**: `FrameDropPolicy(StrEnum)` — three drop strategies for live
  sources under backpressure:
  - `NONE` — block until space is available (no drops, bounded latency risk)
  - `LATEST` — evict oldest frame, insert newest (always-current view)
  - `SKIP_OLDEST` — pop back of queue to make room (FIFO order preserved)

- **io, postprocess**: Pre-allocated I/O buffers eliminate per-frame heap
  allocation on the hot path.
  - `PreprocessBuffer(max_batch, target_size)` — pre-allocated
    `(max_batch, H, W, 3)` uint8 staging array. `preprocess_into()` writes
    directly into the buffer, replacing `cv2.copyMakeBorder` allocation.
  - `PostprocessBuffer(max_detections)` — pre-allocated `(max_detections, 4)`
    float32 scratch for inverse letterbox. `get_inverse_out(n)` returns a view
    with a fresh-allocation fallback for oversized batches.

- **types, engine**: `is_free_threaded()` — runtime GIL detection.
  `not sys._is_gil_enabled()` with `AttributeError` guard for Python < 3.13.
  `InferenceEngine` uses this to auto-set `pipeline_workers=2` on free-threaded
  Python (GIL=OFF) and `pipeline_workers=1` otherwise.

- **engine**: New `InferenceEngine` kwargs for Phase 3 pipeline control.
  `prefetch: bool` (default `True`), `pipeline_workers: int | None` (default
  `None`, auto), `frame_drop_policy: FrameDropPolicy` (default `NONE`),
  `max_queue_size: int` (default `4`). All reflected in `InferenceConfig`.

### Refactored

- **engine, backends**: Black box boundary fixes from architecture audit.
  - All cross-package imports now use public `__init__.py` surfaces instead of
    private `_module` paths.
  - `InferenceBackend` Protocol gains `clear_kv_cache()` and `set_source_id()`.
    Removes all `hasattr`/`type: ignore[attr-defined]` from engine.py.
  - `InferenceEngine` constructor flattened: accepts `InferenceConfig` or
    individual kwargs (all optional, defaults to YOLO26 Nano). `ModelSpec` is
    no longer accepted as a positional argument.

### Performance

- **io**: `PreprocessBuffer` eliminates `cv2.copyMakeBorder` allocation on
  every frame — saves one `(H, W, 3)` uint8 copy (~2.8 MB for 1280×720 input).
- **postprocess**: `PostprocessBuffer` reuses inverse-letterbox scratch array
  across frames — saves one `(max_detections, 4)` float32 allocation per frame.
- **engine**: `pipeline_workers=2` on free-threaded Python (GIL=OFF) delivers
  ~1.5× throughput on YOLO26n CPU inference via true thread parallelism
  (1.47× measured on Apple M4 Pro, Python 3.13.3+freethreaded).

### Fixes

- **engine, io**: Thread safety hardening (P0/P1 review).
  `ThreadedFrameReader` uses `threading.Lock` on `frames_dropped` counter
  increment to prevent lost updates under concurrent reads. `_stop_event` is
  checked before each `deque.append` to avoid post-stop writes.

### Experiments

- Phase 3 source-aware pipeline: offline video 1.03× vs legacy; RTSP
  stream-capped at ~11 FPS on localhost (expected — gains visible on
  high-latency remote cameras); CoreML 3.5–3.75× faster than PyTorch on
  offline video; free-threaded Python 3.13t `pipeline_workers=2` → 58.4 FPS
  vs 39.3 FPS (1.49×).
  See [`docs/experiments/2026-02-25-phase3-source-aware-pipeline-benchmark.md`](docs/experiments/2026-02-25-phase3-source-aware-pipeline-benchmark.md).

### Breaking Changes

- `InferenceEngine(spec, ...)` no longer accepts `ModelSpec` as the first
  positional argument. Use `InferenceEngine(model_family=..., model_size=...)`
  or `InferenceEngine(InferenceConfig(...))` instead.
- `confidence` kwarg renamed to `confidence_threshold` (matches `InferenceConfig`).
- `InferenceConfig` removes 4 dead fields: `max_memory_mb`,
  `reconnect_timeout_s`, `frame_skip`, `max_frames`. Adds 3 fields: `cache`,
  `cache_dir`, `kv_cache`.

---

## [1.1.0] — 2026-02-24

### Features

- **arch, cache**: Feature map caching for sequential inference.
  New `src/yowo/cache/` module with `FeatureCache` coordinator, bounded `FeatureStore`
  (in-memory default, mmap opt-in), and L1 frame-similarity check. On cache hits,
  backbone + neck are skipped entirely — only the detection head runs (~60–85% compute
  savings for slow-moving scenes). Integrated into `PyTorchBackend` via a forward hook
  and exposed via `InferenceEngine(cache=True, cache_dir=Path(...))`.

- **arch, backends, export**: KV cache as explicit ONNX I/O for all runtimes.
  PyTorch backend: fused QKV conv with K,V cached as Python state across frames;
  C2PSA/C3k2PSA blocks skip on spatial-mean fingerprint similarity (< 0.01).
  ONNX export: K,V tensors exposed as explicit model inputs/outputs via `YOLOKVWrapper`
  with a `use_cache` scalar flag — enables stateful streaming on ONNX Runtime, TensorRT,
  and OpenVINO. `export_model(kv_cache=True)` produces KV-capable ONNX models.
  `InferenceEngine(kv_cache=True)` wires the full pipeline.

- **backends, hardware**: CoreML EP auto-detection for Apple Neural Engine.
  `OnnxBackend._select_providers()` adds CoreML EP when `onnxruntime_has_coreml=True`
  (macOS). Provider priority: TensorRT → CUDA → CoreML → CPU → PyTorch.
  Hardware probe adds `onnxruntime_has_coreml: bool` to `InstalledLibraries`.
  No user configuration required — 4–5× speedup vs PyTorch on Apple Silicon.

- **backends, cli**: MPS (Metal GPU) support and CLI image output.
  `PyTorchBackend` accepts `device="mps"`. Manual SDPA fallback in `_attention.py`
  when `q.is_mps` (workaround for PyTorch MPS SDPA shape bug with `key_dim != head_dim`).
  CLI `-o` flag now dispatches on extension: `.jpg`/`.png` → annotated frame via
  `io/_sink.py:write_annotated_frame()`, otherwise → JSON detections file.

### Performance

- **arch**: Phase 2 architecture micro-optimizations (cumulative ~3–6% CPU gain).
  DFL `register_buffer` pre-shaped to `(1,1,c1,1)` — eliminates per-frame reshape.
  `Detect._decode()` caches pre-transposed anchors/strides (`_cached_anchors_t`,
  `_cached_strides_t`) — avoids per-frame `.T` allocation.
  `Conv.forward_fuse_no_act()` bypasses `nn.Identity` dispatch after `fuse()`.
  `Bottleneck` forward specialization post-`fuse()` eliminates per-call branch.
  Results: YOLO26 10–17% faster, YOLO11 1–4% faster vs ultralytics (CPU, FP32, batch=1).

- **backends**: ONNX thread tuning for Apple Silicon.
  `intra_op=cpu_count//2`, `inter_op=cpu_count//4` — avoids E-core scheduling
  overhead that caused thread migration slowdowns with all-core defaults.

### Fixes

- **arch**: `Detect.stride` registered as `persistent=False` buffer with `copy_()`
  to prevent Dynamo recompilation guard failures on stride mutation under
  `torch.compile`.

- **arch**: `Attention` reshape uses explicit `.contiguous().view()` for
  `fullgraph=False` compile path compatibility.

- **backends**: Silence `set_num_interop_threads` error on repeated `load()` calls
  (PyTorch raises if called more than once per process).

- **export**: Internalize external data for KV ONNX exports.
  `torch.onnx.export` dynamo path creates `.onnx.data` sidecar files;
  `_export_onnx_kv()` now loads with `onnx.load(load_external_data=True)` and
  re-saves to internalize all tensors. Fixes CoreML EP load failure:
  `"model_path must not be empty"`.

### Experiments

- CPU arch optimizations: 9/10 variants faster vs ultralytics. Avg 1.07×.
  YOLO26 family 1.10–1.17×; YOLO11 family 1.00–1.04×. Box IoU 0.967–0.995.
  See [`docs/experiments/2026-02-24-arch-inference-optimization-benchmark.md`](docs/experiments/2026-02-24-arch-inference-optimization-benchmark.md).
- CoreML EP: avg **4.36× faster** than PyTorch across all 10 variants.
  Nano models 147–188 FPS; XL models 27–29 FPS on Apple M4 Pro Neural Engine.
  MPS (Metal GPU): avg 1.32× faster than ultralytics (yowo-eager), 1.36× with KV cache.
  See [`docs/experiments/2026-02-24-onnx-coreml-optimization-benchmark.md`](docs/experiments/2026-02-24-onnx-coreml-optimization-benchmark.md).

---

## [0.1.0] — 2026-02-24

### Features

- **arch**: Remove ultralytics dependency — native YOLO11 and YOLO26 architectures
  (`Backbone + FPNPANNeck + Detect`) implemented from published specifications.
  `build_model()` + `load_weights()` public API. Supports all 10 variants:
  `yolo11{n,s,m,l,x}` and `yolo26{n,s,m,l,x}`.

### Fixes

- **arch**: Correct YOLO11/26 weight mapping for all 10 model variants.
  `C3k` now correctly inherits from `C3` (not `C2f`). Added `C3k2PSA`
  for YOLO26 neck layer 22. `backbone_c3k` and `neck_c3k` are now
  size-based (True for m/l/x) instead of family-based — matching
  actual ultralytics checkpoint structure.

- **arch, backend**: Eliminate non-leaf parameter warning in PyTorch backend.
  `fuse_conv_and_bn` wrapped in `torch.no_grad()` to prevent `CopyBackwards`
  grad_fn on fused Conv2d parameters. EMA checkpoint tensors detached before
  `load_state_dict`. CPU thread pool capped at `cpu_count // 2` to prevent
  memory-bandwidth saturation on many-core machines.

### Performance

- **postprocess, io**: Optimize inference hot path with C++ NMS and zero-alloc ops.
  `preprocess()` uses `cv2.dnn.blobFromImages` (single C++ call).
  NMS replaced with `cv2.dnn.NMSBoxes` + class-offset trick.
  Anchor caching in `Detect._decode()`, DFL `.contiguous()` before softmax,
  `non_blocking=True` H2D transfer.

---

## [0.0.1] — 2026-02-23

Initial beta release.

[1.1.0]: https://github.com/TinDang97/yowo/compare/v1.0.2...v1.1.0
[1.2.0]: https://github.com/TinDang97/yowo/compare/v1.1.1...v1.2.0
[0.1.0]: https://github.com/TinDang97/yowo/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/TinDang97/yowo/releases/tag/v0.0.1
