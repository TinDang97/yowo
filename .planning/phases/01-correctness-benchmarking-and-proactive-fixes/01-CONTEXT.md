# Phase 1: Correctness, Benchmarking, and Proactive Fixes - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate inference accuracy vs ultralytics (mAP parity for detection and classification), build benchmark CLI tooling, and fix known production issues (RTSP leak, thread safety, deterministic NMS, export compatibility, graceful fallback). This phase establishes the correctness baseline that all subsequent phases depend on.

</domain>

<decisions>
## Implementation Decisions

### Benchmark CLI design
- New `yowo benchmark` CLI subcommand with rich colored terminal table output (columns: Format, mAP, FPS, Model Size, Device)
- `--json` flag for machine-readable output
- Metrics per format: mAP@0.5:0.95, average FPS, model file size
- If ultralytics is importable, automatically run same model through it and show side-by-side comparison; if not, show YOWO-only results with a note
- Default: test all available backends on current hardware; `--format onnx,trt` flag to filter specific formats; skips unavailable backends with a note
- `--data /path/to/dataset` flag for dataset path (required for mAP)
- `--subset N` flag to run on first N images for quick dev iteration; full dataset by default
- Benchmark results persistable to JSON (`BENCH-04`)

### Validation dataset handling
- User downloads datasets manually; `yowo benchmark` accepts `--data /path/to/coco/val2017` (detection) or `--data /path/to/imagenet/val` (classification)
- If dataset path missing or invalid, print clear download instructions (URL + expected directory structure)
- No auto-download (COCO val2017 ~6GB, ImageNet val ~6.3GB)
- Same pattern for both detection (COCO) and classification (ImageNet)
- mAP computation uses pycocotools (official COCO evaluator) as optional dependency

### Warmup validation
- Warmup validates output tensor shape AND confidence value range [0, 1] on dummy forward pass
- Same validation logic for all backends (no backend-specific checks)
- Applies to both detection and classification engines
- Classification validation: check softmax output sums to ~1.0 and values in [0, 1]
- On validation failure: raise `WarmupValidationError` with details (shape mismatch, bad value range); engine stays unloaded; fail-fast, refuse to serve

### Error messaging (PFIX-05)
- Structured error messages with install command for missing backends
- Format: show what's missing (checked/unchecked package list), install command (`uv add ...`), and list of available backends
- Applied to all 5 backends (PyTorch, ONNX, TensorRT, OpenVINO, CoreML)

### Export compatibility (PFIX-04)
- Both CLI command (`yowo info --compat`) and documentation
- CLI version auto-detects current system versions (CUDA, TensorRT, OpenVINO, etc.) and shows compatibility matrix
- Documentation in README/docs for reference

### Thread safety (PFIX-02)
- Engine is thread-safe by default using internal lock for inference calls
- Users can safely call detect()/classify() from multiple threads without external coordination
- Small locking overhead accepted for safety

### RTSP memory leak (PFIX-01)
- Periodic automatic reconnect of VideoCapture every N minutes (configurable) to reset leaked memory
- Transparent to user; reconnect happens between frames
- Built into existing ThreadedFrameReader

### Claude's Discretion
- Deterministic NMS implementation approach (PFIX-03)
- Exact benchmark FPS measurement methodology (warmup runs, averaging strategy)
- Internal locking mechanism for thread safety (threading.Lock vs RLock)
- Reconnect interval default value for RTSP leak prevention
- Export compatibility matrix content and format details

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `warmup()` method exists on all 5 backends — currently runs dummy forward pass, needs validation logic added
- `metrics/_collector.py` — thread-safety notes exist, GIL-based atomicity for counters
- `tmp/compare_arch.py` — architecture equivalence validation (10/10 pass), pattern for comparison testing
- CLI module (`cli/_main.py`) — existing subcommands (detect, classify, track, count, info, models, export) to pattern-match for benchmark
- `export/` module — exporter, INT8, calibration, KV wrapper already exist
- `io/_reader.py` ThreadedFrameReader — has lock-based frame counting, will receive RTSP reconnect logic

### Established Patterns
- Backend Protocol in `backends/__init__.py` — `load()`, `infer()`, `warmup()` interface
- `BaseEngine(StreamingMixin)` pattern — shared lifecycle across detection/classification engines
- Event-driven architecture via EventBus — can emit benchmark progress events
- Config pattern: `InferenceConfig` / `ClassificationConfig` with validation

### Integration Points
- New `benchmark` CLI subcommand in `cli/_main.py`
- Warmup validation in `backends/__init__.py` Protocol or in engine `load()` method
- Thread safety lock in `BaseEngine` or `DetectionEngine`/`ClassificationEngine`
- RTSP reconnect in `io/_reader.py` ThreadedFrameReader
- Error messages in backend `load()` methods
- Compatibility matrix in `yowo info` CLI subcommand

</code_context>

<specifics>
## Specific Ideas

- Benchmark table should feel like pytest output — rich, colored, easy to scan
- Error messages for missing backends should show the exact `uv add` command (not pip)
- ultralytics comparison is opportunistic — compare if installed, don't require it

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-correctness-benchmarking-and-proactive-fixes*
*Context gathered: 2026-03-07*
