# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Releases after `v0.1.0` are generated automatically by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
from [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

### Refactored

- **engine, backends**: Black box boundary fixes from architecture audit.
  - All cross-package imports now use public `__init__.py` surfaces instead of
    private `_module` paths.
  - `InferenceBackend` Protocol gains `clear_kv_cache()` and `set_source_id()`.
    Removes all `hasattr`/`type: ignore[attr-defined]` from engine.py.
  - `InferenceEngine` constructor flattened: accepts `InferenceConfig` or
    individual kwargs (all optional, defaults to YOLO26 Nano). `ModelSpec` is
    no longer accepted as a positional argument.

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
[0.1.0]: https://github.com/TinDang97/yowo/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/TinDang97/yowo/releases/tag/v0.0.1
