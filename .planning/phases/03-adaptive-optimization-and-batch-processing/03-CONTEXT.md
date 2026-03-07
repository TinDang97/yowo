# Phase 3: Adaptive Optimization and Batch Processing - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Two new CLI tools built on existing engines and backends:
1. `yowo tune --model MODEL` — auto-detect optimal backend, batch size, and precision for current hardware via calibration sweep; persist profile for instant startup on subsequent runs
2. `yowo batch SOURCE_DIR --model MODEL` — offline high-throughput processing of image/video directories at maximum GPU utilization with resume-on-interruption support

No new inference capability added — tune and batch use existing backends, engines, and preprocessing pipeline.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation decisions for this phase are delegated to Claude. Decisions below are Claude's transparent choices with rationale — they are locked for planning purposes.

### Calibration sweep strategy
- Sweep dimensions: available backends × batch sizes [1, 2, 4, 8, 16, 32] × precision [FP32, FP16 (GPU only), INT8 (TensorRT only)]
- Each configuration: 50 warmup frames + 200 synthetic measurement frames (dummy tensors — no real dataset required for tune)
- OOM guard: wrap each config in try/except for `torch.cuda.OutOfMemoryError` + `RuntimeError`; skip remaining batch sizes for that backend/precision combo and continue
- INT8: skipped if TensorRT unavailable; note in output that INT8 requires `yowo export --format trt --int8` first
- Ranking criterion: FPS (higher is better); ties broken by lower batch size (more headroom for OOM recovery)
- Total sweep time target: ~2–5 minutes on typical GPU hardware
- Output: rich colored table showing all tested configs with FPS, then a highlighted "Best config" row
- `--json` flag for machine-readable output (consistent with benchmark CLI from Phase 1)
- `--dry-run` flag: shows what would be swept without running calibration

### Tune profile persistence
- Location: `~/.cache/yowo/profiles/{fingerprint}/{model_name}.yaml`
- Device fingerprint: 8-char hex of SHA-256(`{gpu_name}|{vram_bytes}|{cuda_version}|{driver_version}`) — computed via existing `HardwareProfile`; CPU-only devices use `cpu|{cpu_count}|{platform}`
- Profile fields: `model`, `backend`, `batch_size`, `precision`, `fps_achieved`, `tuned_at` (ISO8601), `fingerprint`
- On engine load: if profile exists for current device+model, load it silently and log at DEBUG level
- Fingerprint mismatch (device changed): warn at INFO level — "Hardware changed since last tune. Run `yowo tune --model {model}` to update profile." — then fall back to config defaults
- Multi-device: natural isolation via fingerprint directory (no collision)
- Multi-model: separate YAML file per model name within same fingerprint dir
- `--output PATH` flag on `yowo tune` to override default profile location
- `--force` flag to re-tune even if profile exists for current device

### Batch output design
- `--output DIR` is required (no default); directory is created if it doesn't exist
- Default output structure:
  - Annotated frames → `{output}/frames/` (images: `{stem}_annotated{ext}`; videos: `{output}/frames/{video_stem}/{frame_idx:06d}.jpg`)
  - Detection results → `{output}/results.jsonl` (JSONL: one JSON object per source file, containing path + detections list)
- `--no-annotate` flag: skip frame writing entirely (results.jsonl only) — for GPU-memory-constrained or headless batch jobs
- `--format json|jsonl` flag: default JSONL (streaming-friendly for large batches); `json` produces single array file
- Mixed source types (images + videos in same dir): processed together transparently; videos are expanded frame-by-frame internally, but results.jsonl groups by source file
- Progress counts source files (not frames) — a 1000-frame video = 1 file for ETA purposes; frame throughput shown separately
- Supported image extensions: reuse `IMAGE_EXTS` constant; video extensions: reuse `VIDEO_EXTS` — no new format handling
- Files that fail to open: logged as warnings, skipped (written to `{output}/errors.log`), batch continues
- `--recursive` flag: descend into subdirectories of SOURCE_DIR

### Batch resume and progress reporting
- Checkpoint file: `{output}/.yowo_checkpoint.json` — written atomically (write to `.tmp`, then rename)
- Checkpoint contents: `{"completed": ["rel/path/to/file1.jpg", ...], "stats": {"files_done": N, "frames_done": M, "started_at": "ISO8601"}}`
- Checkpoint frequency: every 100 completed files OR every 30 seconds (whichever comes first)
- Resume detection: on `yowo batch` start, check for `{output}/.yowo_checkpoint.json`; if found, print "Resuming: {N}/{total} files done" and skip completed files
- `--no-resume` flag: ignore checkpoint and reprocess all files
- Progress display: rich `Progress` with columns: bar, percentage, files count (done/total), FPS (rolling 5s window), ETA
- Final summary on completion: total files, total frames, elapsed time, average FPS, output location
- Exit code 0 on full success; exit code 1 if any files had errors (with count); exit code 2 on fatal error

### CLI integration
- `yowo tune` and `yowo batch` added as new subcommands in `src/yowo/cli/_main.py` (consistent with existing pattern)
- Tune: `yowo tune --model yolo11n [--weights PATH] [--output PATH] [--force] [--dry-run] [--json]`
- Batch: `yowo batch SOURCE_DIR --model yolo11n [--weights PATH] [--output DIR] [--no-annotate] [--format jsonl|json] [--recursive] [--no-resume] [--workers N]`
- Profile auto-loading integrates into `engine.load()` via optional `TuneProfile` lookup — no changes to public API signature

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `HardwareProfile` (`src/yowo/hardware/__init__.py`) — provides GPU name, VRAM, CUDA version for device fingerprint; `get_hardware_profile()` singleton
- `InferenceConfig` (`src/yowo/config.py`) — has `backend`, `batch_size`, `precision` fields; tune writes these; existing env var overrides still apply
- `BaseEngine` + `DetectionEngine` (`src/yowo/engine.py`) — `load()` / `detect()` lifecycle; tune reuses engine for sweep measurement
- `PreprocessBuffer` + `PreprocessBufferPool` (`src/yowo/io/_decode.py`) — buffer pooling reusable for batch preprocessing
- `IMAGE_EXTS` / `VIDEO_EXTS` constants (`src/yowo/types.py`) — reuse for source file filtering in batch
- `ThreadedFrameReader` (`src/yowo/io/_reader.py`) — existing video frame reader reused inside batch video processing
- `cli/_main.py` — Click group; `yowo benchmark` pattern from Phase 1 to follow for `yowo tune` and `yowo batch`
- `MetricsCollector` + `EngineMetrics` (`src/yowo/metrics/`) — FPS measurement for sweep ranking and batch progress
- `select_backend` / `get_fallback_backends` (`src/yowo/backends/_selector.py`) — enumerates available backends for sweep

### Established Patterns
- Config validation in `__post_init__` (dataclass pattern from `InferenceConfig`)
- Click subcommand pattern: `@cli.command("name")` with explicit option definitions
- Rich table output: Phase 1 benchmark uses rich tables — tune output follows same style
- `--json` flag for machine-readable output (Phase 1 established)
- Optional deps raise `click.UsageError` with `uv add` instruction (Phase 1)

### Integration Points
- `src/yowo/cli/_main.py` — add `tune` and `batch` subcommands
- `src/yowo/engine.py` — `BaseEngine.load()` optionally reads tune profile before setting backend/batch/precision
- New modules: `src/yowo/tune/_sweep.py` (calibration sweep logic), `src/yowo/tune/_profile.py` (profile read/write)
- New module: `src/yowo/batch/_runner.py` (batch processing loop, checkpoint management)
- `src/yowo/hardware/__init__.py` — fingerprint computation function added here or in tune/_profile.py

</code_context>

<specifics>
## Specific Ideas

- Tune output table should mirror benchmark table style from Phase 1 (rich, colored, easy to scan)
- Profile loading should be completely silent in normal operation — operators shouldn't notice it; only visible in DEBUG logs and `yowo info` output
- Batch JSONL format chosen over JSON for streaming-friendliness: can `tail -f results.jsonl` or process incrementally without loading full file into memory
- Checkpoint written atomically (write + rename) to prevent corruption on interruption

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-adaptive-optimization-and-batch-processing*
*Context gathered: 2026-03-07*
