# Phase 3: Adaptive Optimization and Batch Processing - Research

**Researched:** 2026-03-07
**Domain:** Auto-tuning calibration sweep, device profile persistence, offline batch inference, checkpoint resume, CLI extension
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

All implementation decisions for this phase are delegated to Claude. Decisions below are Claude's transparent choices with rationale — they are locked for planning purposes.

**Calibration sweep strategy:**
- Sweep dimensions: available backends × batch sizes [1, 2, 4, 8, 16, 32] × precision [FP32, FP16 (GPU only), INT8 (TensorRT only)]
- Each configuration: 50 warmup frames + 200 synthetic measurement frames (dummy tensors — no real dataset required for tune)
- OOM guard: wrap each config in try/except for `torch.cuda.OutOfMemoryError` + `RuntimeError`; skip remaining batch sizes for that backend/precision combo and continue
- INT8: skipped if TensorRT unavailable; note in output that INT8 requires `yowo export --format trt --int8` first
- Ranking criterion: FPS (higher is better); ties broken by lower batch size (more headroom for OOM recovery)
- Total sweep time target: ~2–5 minutes on typical GPU hardware
- Output: rich colored table showing all tested configs with FPS, then a highlighted "Best config" row
- `--json` flag for machine-readable output (consistent with benchmark CLI from Phase 1)
- `--dry-run` flag: shows what would be swept without running calibration

**Tune profile persistence:**
- Location: `~/.cache/yowo/profiles/{fingerprint}/{model_name}.yaml`
- Device fingerprint: 8-char hex of SHA-256(`{gpu_name}|{vram_bytes}|{cuda_version}|{driver_version}`) — computed via existing `HardwareProfile`; CPU-only devices use `cpu|{cpu_count}|{platform}`
- Profile fields: `model`, `backend`, `batch_size`, `precision`, `fps_achieved`, `tuned_at` (ISO8601), `fingerprint`
- On engine load: if profile exists for current device+model, load it silently and log at DEBUG level
- Fingerprint mismatch (device changed): warn at INFO level — "Hardware changed since last tune. Run `yowo tune --model {model}` to update profile." — then fall back to config defaults
- Multi-device: natural isolation via fingerprint directory (no collision)
- Multi-model: separate YAML file per model name within same fingerprint dir
- `--output PATH` flag on `yowo tune` to override default profile location
- `--force` flag to re-tune even if profile exists for current device

**Batch output design:**
- `--output DIR` is required (no default); directory is created if it doesn't exist
- Default output structure: annotated frames → `{output}/frames/`; detection results → `{output}/results.jsonl`
- `--no-annotate` flag: skip frame writing entirely (results.jsonl only)
- `--format json|jsonl` flag: default JSONL; `json` produces single array file
- Mixed source types (images + videos) processed together transparently
- Progress counts source files (not frames); frame throughput shown separately
- Supported extensions: reuse `IMAGE_EXTS` / `VIDEO_EXTS` constants from `types.py`
- Files that fail to open: logged as warnings, written to `{output}/errors.log`, batch continues
- `--recursive` flag: descend into subdirectories

**Batch resume and progress reporting:**
- Checkpoint file: `{output}/.yowo_checkpoint.json` — written atomically (write to `.tmp`, then rename)
- Checkpoint contents: `{"completed": [...], "stats": {"files_done": N, "frames_done": M, "started_at": "ISO8601"}}`
- Checkpoint frequency: every 100 completed files OR every 30 seconds (whichever comes first)
- Resume detection: check for checkpoint on start, print "Resuming: {N}/{total} files done" and skip completed files
- `--no-resume` flag: ignore checkpoint and reprocess all files
- Progress display: rich `Progress` with: bar, percentage, files count (done/total), FPS (rolling 5s window), ETA
- Final summary: total files, total frames, elapsed time, average FPS, output location
- Exit code 0 (success), 1 (partial errors), 2 (fatal error)

**CLI integration:**
- `yowo tune` and `yowo batch` added as new subcommands in `src/yowo/cli/_main.py`
- Tune: `yowo tune --model yolo11n [--weights PATH] [--output PATH] [--force] [--dry-run] [--json]`
- Batch: `yowo batch SOURCE_DIR --model yolo11n [--weights PATH] [--output DIR] [--no-annotate] [--format jsonl|json] [--recursive] [--no-resume] [--workers N]`
- Profile auto-loading integrates into `engine.load()` via optional `TuneProfile` lookup — no changes to public API signature

### Claude's Discretion

All implementation decisions are Claude's — none are deferred to user discretion.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TUNE-01 | User can run `yowo tune --model MODEL` to auto-detect optimal backend, batch size, and precision for current hardware | Calibration sweep logic in `tune/_sweep.py`; CLI subcommand in `cli/_main.py` |
| TUNE-02 | Auto-tune runs calibration sweep across available backends and precision levels | `select_backend` + `get_fallback_backends` enumerate candidates; `BaseEngine` + synthetic frames run each config |
| TUNE-03 | Auto-tune results persist to device-specific profile file for instant startup on subsequent runs | `tune/_profile.py` handles SHA-256 fingerprint, YAML read/write; `BaseEngine.load()` reads profile pre-selection |
| TUNE-04 | Auto-tune respects device memory constraints (does not OOM during calibration) | OOM guard: `torch.cuda.OutOfMemoryError` + `RuntimeError` caught per config; skip larger batch sizes on OOM |
| BATC-01 | User can run `yowo batch SOURCE_DIR --model MODEL` for offline high-throughput processing | `batch/_runner.py` implements directory walk, engine loop; CLI subcommand in `cli/_main.py` |
| BATC-02 | Batch mode maximizes GPU utilization with larger batch sizes than streaming mode | Profile auto-load provides tuned batch_size; `FrameDropPolicy.NONE` (backpressure) for offline; no frame dropping |
| BATC-03 | Batch processing supports resume from checkpoint on interruption | Atomic checkpoint write (tmp + rename); checkpoint read on start; completed-file skip set |
| BATC-04 | Batch mode reports progress (processed/total, ETA, throughput) | rich `Progress` with custom columns; rolling 5-second FPS window; ETA from rich's built-in |
</phase_requirements>

---

## Summary

Phase 3 adds two self-contained CLI tools — `yowo tune` and `yowo batch` — that layer on top of existing Phase 1/2 infrastructure without adding new inference capability. The calibration sweep in `tune` iterates over backend × batch × precision combinations using synthetic dummy tensors (no dataset required), measures FPS via the existing `MetricsCollector`, and writes a device-fingerprinted YAML profile. Subsequent engine loads silently read that profile to set optimal config. The `batch` runner walks a source directory, feeds files through the engine in JSONL-output mode, writes atomically checkpointed progress for resume-on-interruption, and reports a rich progress bar with rolling FPS and ETA.

All code builds on verified existing components: `HardwareProfile` for device detection, `select_backend` / `get_fallback_backends` for backend enumeration, `BaseEngine` + `InferenceConfig` for sweep measurement, `IMAGE_EXTS` / `VIDEO_EXTS` / `ThreadedFrameReader` for source handling, and `MetricsCollector` for FPS aggregation. No new dependencies are required beyond `rich` (already in the `benchmark` optional group) and stdlib `hashlib`, `yaml`, `json`, `pathlib`.

**Primary recommendation:** Implement in three new modules (`tune/_sweep.py`, `tune/_profile.py`, `batch/_runner.py`) plus two CLI subcommands. The engine profile integration point is in `BaseEngine.__init__` (before `select_backend` is called), not in `load()`, because selection happens at construction time.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rich` | >=13.0 (already in `benchmark` optional dep) | Progress bars, colored tables for tune output and batch progress | Already used in Phase 1 benchmark; consistent CLI experience |
| `pyyaml` | Already in core deps (used in `config.py`) | Profile YAML read/write | Existing usage in `load_config()`; no new dep |
| `hashlib` | stdlib | SHA-256 device fingerprint | Zero dep, deterministic, already available |
| `json` | stdlib | Checkpoint file, JSONL output | Already used throughout |
| `pathlib` | stdlib | File system operations | Project-wide pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.cuda.OutOfMemoryError` | PyTorch (already dep) | OOM guard in calibration sweep | Catch on GPU sweep configs |
| `threading.Event` | stdlib | Checkpoint timer (30s periodic flush) | Batch runner periodic checkpoint |
| `time.monotonic` | stdlib | Rolling FPS window (5s) in batch progress | Replace wall-clock with monotonic for accuracy |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| YAML for profiles | TOML or JSON | YAML already used in `load_config()` — consistency wins; TOML requires `tomllib` (Python 3.11+) |
| SHA-256 8-char hex fingerprint | UUID or full hash | 8-char is short enough to be directory-safe, long enough to be collision-resistant at this scale |
| Atomic write (tmp + rename) | Direct write | Direct write corrupts checkpoint on SIGKILL; rename is atomic on POSIX filesystems |
| rich Progress | tqdm | rich already in benchmark dep group; visual consistency with tune table output |

**Installation:**
```bash
# No new deps required — rich already in [benchmark] group
# Profile persistence uses stdlib hashlib + pyyaml (core dep)
# Verify rich is available:
uv add yowo[benchmark]
```

---

## Architecture Patterns

### Recommended Project Structure
```
src/yowo/
├── tune/
│   ├── __init__.py          # exports: TuneProfile, run_sweep, load_profile, save_profile
│   ├── _sweep.py            # calibration sweep logic: run_sweep() -> list[SweepResult]
│   └── _profile.py          # fingerprint, YAML read/write: TuneProfile dataclass
├── batch/
│   ├── __init__.py          # exports: run_batch, BatchConfig
│   └── _runner.py           # batch loop, checkpoint management, progress reporting
├── cli/
│   └── _main.py             # add tune_command() and batch_command() subcommands
└── engine.py                # minor: _load_tune_profile() helper called in __init__ before select_backend
```

### Pattern 1: Calibration Sweep Loop
**What:** Enumerate all backend × batch_size × precision combinations; measure FPS on synthetic frames; catch OOM and skip higher batch sizes for that combo.
**When to use:** `yowo tune` command execution.
**Example:**
```python
# src/yowo/tune/_sweep.py
import time
from dataclasses import dataclass
import numpy as np

@dataclass
class SweepResult:
    backend: str
    batch_size: int
    precision: str
    fps: float
    skipped: bool = False
    skip_reason: str = ""

def run_sweep(
    model_spec: ModelSpec,
    hw: HardwareProfile,
    warmup_frames: int = 50,
    measure_frames: int = 200,
) -> list[SweepResult]:
    """Sweep backends × batch_sizes × precisions. Returns results sorted by FPS desc."""
    from yowo.backends import select_backend, get_fallback_backends, create_backend

    results: list[SweepResult] = []
    available_backends = _enumerate_backends(hw)
    batch_sizes = [1, 2, 4, 8, 16, 32]

    for backend_type in available_backends:
        for precision in _precisions_for_backend(backend_type, hw):
            oom_hit = False
            for batch_size in batch_sizes:
                if oom_hit:
                    results.append(SweepResult(
                        backend=backend_type.value,
                        batch_size=batch_size,
                        precision=precision.value,
                        fps=0.0,
                        skipped=True,
                        skip_reason="OOM at smaller batch",
                    ))
                    continue
                try:
                    fps = _measure_config(
                        model_spec, hw, backend_type, precision, batch_size,
                        warmup_frames, measure_frames,
                    )
                    results.append(SweepResult(
                        backend=backend_type.value,
                        batch_size=batch_size,
                        precision=precision.value,
                        fps=fps,
                    ))
                except (MemoryError, RuntimeError) as exc:
                    if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                        oom_hit = True
                        results.append(SweepResult(..., skipped=True, skip_reason="OOM"))
                    else:
                        results.append(SweepResult(..., skipped=True, skip_reason=str(exc)))

    return sorted(
        [r for r in results if not r.skipped],
        key=lambda r: (-r.fps, r.batch_size),  # highest FPS, tie-break lower batch
    )
```

### Pattern 2: Device Fingerprint
**What:** Stable 8-char hex identifier for current hardware, used as profile directory name.
**When to use:** `_profile.py` on both save and load paths.
**Example:**
```python
# src/yowo/tune/_profile.py
import hashlib
import platform

def _compute_fingerprint(hw: HardwareProfile) -> str:
    """Compute 8-char hex SHA-256 fingerprint of current hardware."""
    gpu = hw.primary_gpu
    if gpu is not None:
        # GPU device: use GPU name + VRAM + CUDA version
        raw = f"{gpu.name}|{gpu.memory_total_mb * 1024 * 1024}|{hw.libraries.cuda_version or 'none'}"
    else:
        # CPU-only: use cpu_count + platform
        import os
        raw = f"cpu|{os.cpu_count()}|{platform.platform()}"

    digest = hashlib.sha256(raw.encode()).hexdigest()
    return digest[:8]
```

### Pattern 3: Atomic Checkpoint Write
**What:** Write checkpoint JSON atomically to prevent corruption on interruption.
**When to use:** `batch/_runner.py` every 100 files or 30 seconds.
**Example:**
```python
# src/yowo/batch/_runner.py
import json
import os
from pathlib import Path

def _write_checkpoint(checkpoint_path: Path, completed: list[str], stats: dict) -> None:
    """Atomically write checkpoint by writing to .tmp then renaming."""
    data = {"completed": completed, "stats": stats}
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, checkpoint_path)  # POSIX atomic rename
```

### Pattern 4: TuneProfile Integration in BaseEngine
**What:** Before calling `select_backend()`, check if a profile exists for current device + model. If found, apply backend/batch_size/precision from profile, bypassing auto-selection.
**When to use:** `BaseEngine.__init__()`.
**Example:**
```python
# src/yowo/engine.py  (modification to BaseEngine.__init__)
from yowo.tune._profile import load_profile, TuneProfile

def __init__(self, spec, *, ...):
    ...
    # Try to load tune profile before backend selection
    profile: TuneProfile | None = None
    if not self._user_provided_backend:
        profile = _load_tune_profile(spec.family, spec.size)
        if profile is not None:
            logger.debug("Loaded tune profile: backend=%s batch=%d precision=%s",
                         profile.backend, profile.batch_size, profile.precision)
            # Override: apply profile values (only if not explicitly overridden by caller)
            if backend_override is None:
                backend_override = profile.backend
            if batch_size == 1:  # default — apply profile
                batch_size = profile.batch_size
            if precision is None:
                precision = Precision(profile.precision)
    ...
```

### Anti-Patterns to Avoid
- **Importing rich at module top-level:** Use lazy import inside command function (same pattern as benchmark) — prevents CLI startup failure when rich not installed.
- **Writing checkpoint on every file:** Every 100 files or 30s reduces I/O significantly on large batches.
- **Using `json` single-array for results.jsonl:** JSONL (one object per line) is default — enables `tail -f` and incremental processing without loading full file.
- **Blocking video frame iteration inside `batch` with synchronous I/O:** Use `ThreadedFrameReader` (already available) inside the video processing loop for each video file.
- **Catching bare `Exception` in OOM guard:** Catch `torch.cuda.OutOfMemoryError` first (specific), then `RuntimeError` with OOM string check. Bare `Exception` would swallow legitimate bugs.
- **Modifying `InferenceConfig` fields after construction:** `InferenceConfig` uses `__post_init__` validation. Use `dataclasses.replace()` when constructing sweep configs (same pattern as Phase 2 OOM recovery).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend enumeration | Custom discovery loop | `_run_priority_chain()` + `get_fallback_backends()` in `_selector.py` | Already handles all 5 backends in correct priority order |
| FPS measurement | Custom timer | `MetricsCollector.snapshot().fps` | Already thread-safe, rolling histogram, correct uptime calculation |
| Progress bar with ETA | Custom progress display | `rich.progress.Progress` with `BarColumn`, `TaskProgressColumn`, `TimeRemainingColumn` | ETA calculation is non-trivial with variable file sizes |
| Device detection | Raw `nvidia-smi` subprocess | `get_hardware_profile()` | Already handles CUDA, CPU, Jetson, Apple Silicon |
| YAML profile serialization | Custom format | `yaml.safe_dump` / `yaml.safe_load` | Already used in `load_config()`; handles Path types |
| Batch size for high-throughput | Manual tuning | Profile from `yowo tune` | The entire point of TUNE phase |
| Video frame iteration | Custom OpenCV loop | `ThreadedFrameReader` in `io/_reader.py` | Already handles daemon thread, idle timeout, bounded deque |
| File extension filtering | Custom suffix check | `IMAGE_EXTS` / `VIDEO_EXTS` from `types.py` | Constants already defined and tested |

**Key insight:** This phase is 90% orchestration of existing components. The sweep logic, file handling, and engine lifecycle are all present — the implementation connects them with new control flow and persistence.

---

## Common Pitfalls

### Pitfall 1: OOM During Calibration Leaves GPU Memory Dirty
**What goes wrong:** `torch.cuda.OutOfMemoryError` is caught but CUDA memory is not explicitly freed, causing subsequent sweep configs to fail even at lower batch sizes.
**Why it happens:** PyTorch does not automatically release OOM-failed allocations from the Python exception path.
**How to avoid:** After catching OOM, call `torch.cuda.empty_cache()` before continuing to the next config. Also call `backend.close()` / `engine.close()` to release model weights.
**Warning signs:** Configs that should fit VRAM are also marked OOM; all configs after first OOM fail.

### Pitfall 2: Fingerprint Changes on Warmup/Driver Update
**What goes wrong:** Profile becomes stale (fingerprint mismatch) after a routine driver update, triggering the "Hardware changed" warning even on the same GPU.
**Why it happens:** Including driver version in fingerprint ties profile validity to driver, not hardware.
**How to avoid:** The decided fingerprint uses `gpu_name|vram_bytes|cuda_version` (not driver version for CPU path). This is already a good tradeoff — CUDA runtime version (from `hw.libraries.cuda_version`) changes infrequently. Ensure the fingerprint inputs map to `InstalledLibraries.cuda_version` (torch-reported, not nvidia-smi). The CONTEXT.md decision mentions "driver_version" but `HardwareProfile` exposes `cuda_version` via `InstalledLibraries` — use `cuda_version` consistently.
**Warning signs:** Frequent fingerprint mismatches reported; profiles re-created after minor updates.

### Pitfall 3: JSONL Batch Results Desync from Checkpoint
**What goes wrong:** On resume, `results.jsonl` contains entries from the previous run that duplicate files re-processed after `--no-resume`.
**Why it happens:** JSONL append mode adds to existing file without awareness of checkpoint state.
**How to avoid:** On fresh batch (no checkpoint or `--no-resume`), open `results.jsonl` in write mode (`"w"`). On resume, open in append mode (`"a"`) and only write entries for newly completed files (those not in checkpoint's `completed` set).
**Warning signs:** `results.jsonl` has duplicate file entries; line count exceeds total source files.

### Pitfall 4: Sweep Uses Real Model Load Time in FPS
**What goes wrong:** FPS measurement includes model load time (weights loading, CUDA kernel compilation), making first-config FPS artificially low.
**Why it happens:** Backend load happens before warmup frames in the naive implementation.
**How to avoid:** Structure sweep as: (1) `engine.load()` — not measured; (2) 50 warmup frames — not measured; (3) 200 measurement frames — only these contribute to FPS. Use `time.monotonic()` start only after warmup completes.
**Warning signs:** First backend config always reports much lower FPS than subsequent configs.

### Pitfall 5: Batch Progress ETA Wrong for Mixed Image/Video Sources
**What goes wrong:** ETA estimates assume uniform file processing time, but videos have 1000x more frames than images.
**Why it happens:** rich Progress counts task steps (files) uniformly; one video = same weight as one image.
**How to avoid:** As decided in CONTEXT.md, progress counts source files (not frames). This is correct for ETA purposes — videos are not expanded. Frame throughput is shown separately in a custom column. Document this clearly in `--help` text.
**Warning signs:** ETA wildly incorrect when batch contains few large videos.

### Pitfall 6: Profile Auto-Load Overrides Explicit CLI Args
**What goes wrong:** `yowo detect --backend pytorch` is ignored when a tune profile specifies `backend: tensorrt`.
**Why it happens:** Profile loading in `BaseEngine.__init__` applies before caller's `backend_override` is honored.
**How to avoid:** Profile values are applied ONLY when the corresponding arg is at its default (e.g., `backend_override is None`). Explicit caller args always win. Document in `BaseEngine` docstring.
**Warning signs:** CLI `--backend` option appears to have no effect.

---

## Code Examples

Verified patterns from existing codebase:

### Enumerating Available Backends
```python
# Source: src/yowo/backends/_selector.py — _run_priority_chain() pattern
from yowo.backends._selector import get_fallback_backends
from yowo.types import BackendType

# All backends to sweep (from priority chain logic)
_SWEEP_BACKENDS = [
    BackendType.TENSORRT,   # GPU only, skipped if TensorRT not installed
    BackendType.ONNX,       # GPU (CUDA EP) or CPU
    BackendType.PYTORCH,    # Universal fallback
    BackendType.OPENVINO,   # Intel CPU/GPU
    BackendType.COREML,     # Apple Silicon only
]

def _enumerate_backends(hw: HardwareProfile) -> list[BackendType]:
    """Return backends available on current hardware, in sweep priority order."""
    from yowo.backends._selector import _check_backend_available
    from yowo.errors import BackendError
    available = []
    for bt in _SWEEP_BACKENDS:
        try:
            _check_backend_available(bt, hw)
            available.append(bt)
        except BackendError:
            pass
    return available
```

### Using rich Progress for Batch
```python
# Source: rich library pattern (already in benchmark dep group)
from rich.progress import (
    Progress, BarColumn, TaskProgressColumn,
    TimeRemainingColumn, TextColumn, SpinnerColumn,
)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} files"),
    TextColumn("[cyan]{task.fields[fps]:.1f} FPS"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Processing", total=total_files, fps=0.0)
    for file in source_files:
        process_file(file)
        progress.update(task, advance=1, fps=current_fps)
```

### Loading YAML Profile
```python
# Source: src/yowo/config.py — load_config() pattern
import yaml
from pathlib import Path

def load_profile(profile_path: Path) -> TuneProfile | None:
    """Load tune profile from YAML, return None if missing or invalid."""
    if not profile_path.exists():
        return None
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        return TuneProfile(
            model=raw["model"],
            backend=raw["backend"],
            batch_size=int(raw["batch_size"]),
            precision=raw["precision"],
            fps_achieved=float(raw["fps_achieved"]),
            tuned_at=raw["tuned_at"],
            fingerprint=raw["fingerprint"],
        )
    except (KeyError, ValueError, yaml.YAMLError):
        return None  # Corrupt profile — silently fall back to defaults
```

### Dataclasses.replace Pattern for Sweep Configs
```python
# Source: src/yowo/engine.py — OOM recovery uses replace()
from dataclasses import replace
from yowo.config import InferenceConfig

# Build sweep config without mutating shared config
sweep_config = replace(
    base_config,
    backend=BackendType(backend_type),
    batch_size=batch_size,
    precision=precision,
    metrics_enabled=True,
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual `--batch` tuning via trial-and-error | Automated calibration sweep with OOM guard | This phase | Operators no longer need to know safe batch sizes |
| No persistent optimization state | Device-fingerprinted YAML profiles in `~/.cache/yowo/` | This phase | Cold start instant after first tune |
| JSON output only | JSONL (default) for streaming-friendly large batch results | This phase | Can `tail -f` results during processing |
| No resume capability | Atomic checkpoint every 100 files / 30s | This phase | Interrupted batch jobs recoverable |

**Deprecated/outdated:**
- None in this phase — new surface area only.

---

## Open Questions

1. **`driver_version` in fingerprint vs `cuda_version`**
   - What we know: CONTEXT.md mentions `{driver_version}` in the fingerprint formula, but `HardwareProfile` / `InstalledLibraries` does not expose `driver_version` — only `cuda_version` (from PyTorch's torch.version.cuda).
   - What's unclear: Should we add `driver_version` detection (requires `nvidia-smi` subprocess or `pynvml`) or substitute `cuda_version`?
   - Recommendation: Use `cuda_version` as the 4th field (it is already available as `hw.libraries.cuda_version`). The fingerprint formula becomes `{gpu_name}|{vram_bytes}|{cuda_version}|none` or add a `driver_version` field to `_detect.py` via `nvidia-smi --query-gpu=driver_version --format=csv,noheader`. The simpler path (no new subprocess) is to use `cuda_version` only — the plan should decide.

2. **Profile auto-load scope: detection only or also classification?**
   - What we know: CONTEXT.md references `InferenceConfig` fields (detection) for the sweep. Classification has `ClassificationConfig`.
   - What's unclear: Should `ClassificationEngine` also check tune profiles? The sweep only sweeps detection configs.
   - Recommendation: Phase 3 scope is detection-only. `ClassificationEngine` does not load profiles (no regression risk).

3. **`--workers N` in batch CLI vs `pipeline_workers` in config**
   - What we know: `yowo batch` has `--workers N` CLI option (from CONTEXT.md). `InferenceConfig.pipeline_workers` controls threading.
   - What's unclear: Does `--workers` map to `pipeline_workers` or to a separate batch-level worker pool?
   - Recommendation: Map `--workers N` directly to `InferenceConfig.pipeline_workers` for simplicity. Document that it controls preprocessing threads, not GPU workers.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | `pyproject.toml` ([tool.pytest.ini_options]) |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TUNE-01 | `yowo tune --model yolo11n` CLI command exists and runs | unit (CLI invoke) | `uv run pytest tests/unit/test_tune_cli.py -x -q` | ❌ Wave 0 |
| TUNE-02 | `run_sweep()` returns results for all available backends | unit | `uv run pytest tests/unit/test_sweep.py -x -q` | ❌ Wave 0 |
| TUNE-03 | Profile saved to correct path; loaded on next engine init | unit | `uv run pytest tests/unit/test_tune_profile.py -x -q` | ❌ Wave 0 |
| TUNE-04 | OOM configs are skipped; subsequent configs still measured | unit | `uv run pytest tests/unit/test_sweep.py::test_oom_skip -x -q` | ❌ Wave 0 |
| BATC-01 | `yowo batch SOURCE_DIR` CLI command exists and processes files | unit (CLI invoke) | `uv run pytest tests/unit/test_batch_cli.py -x -q` | ❌ Wave 0 |
| BATC-02 | Batch engine uses profile batch_size when profile exists | unit | `uv run pytest tests/unit/test_batch_runner.py::test_uses_profile_batch_size -x -q` | ❌ Wave 0 |
| BATC-03 | Checkpoint written atomically; resume skips completed files | unit | `uv run pytest tests/unit/test_batch_runner.py::test_checkpoint_resume -x -q` | ❌ Wave 0 |
| BATC-04 | Progress reports files/total, FPS, ETA | unit | `uv run pytest tests/unit/test_batch_runner.py::test_progress_columns -x -q` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_tune_cli.py tests/unit/test_sweep.py tests/unit/test_tune_profile.py tests/unit/test_batch_cli.py tests/unit/test_batch_runner.py -x -q`
- **Per wave merge:** `uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_tune_cli.py` — CLI invoke tests for `yowo tune` subcommand (TUNE-01)
- [ ] `tests/unit/test_sweep.py` — sweep loop unit tests with mock backends (TUNE-02, TUNE-04)
- [ ] `tests/unit/test_tune_profile.py` — fingerprint computation, YAML save/load, fingerprint mismatch warn (TUNE-03)
- [ ] `tests/unit/test_batch_cli.py` — CLI invoke tests for `yowo batch` subcommand (BATC-01)
- [ ] `tests/unit/test_batch_runner.py` — checkpoint atomic write, resume, progress columns (BATC-02, BATC-03, BATC-04)

---

## Sources

### Primary (HIGH confidence)
- Codebase direct read: `src/yowo/backends/_selector.py` — backend enumeration, `_check_backend_available`, `get_fallback_backends`
- Codebase direct read: `src/yowo/hardware/__init__.py` + `_capabilities.py` + `_detect.py` — `HardwareProfile`, `InstalledLibraries`, `cuda_version`, `memory_total_mb`
- Codebase direct read: `src/yowo/config.py` — `InferenceConfig`, `dataclasses.replace` pattern, `load_config` YAML pattern
- Codebase direct read: `src/yowo/engine.py` — `BaseEngine.__init__`, `select_backend` call point, `_original_batch_size`
- Codebase direct read: `src/yowo/metrics/_collector.py` — `MetricsCollector`, `EngineMetrics.fps`, `snapshot()`
- Codebase direct read: `src/yowo/cli/_main.py` — existing subcommand pattern (benchmark, detect, classify, health)
- Codebase direct read: `pyproject.toml` — `benchmark = ["pycocotools>=2.0.4", "rich>=13.0"]` optional dep group

### Secondary (MEDIUM confidence)
- Python stdlib docs: `os.replace()` for atomic file rename on POSIX (guaranteed atomic within same filesystem)
- Python stdlib docs: `hashlib.sha256()` for fingerprint computation
- `rich` library: `Progress` API with `BarColumn`, `TimeRemainingColumn`, `TaskProgressColumn`

### Tertiary (LOW confidence)
- None — all critical claims verified from codebase source.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified present in pyproject.toml or stdlib
- Architecture: HIGH — new modules mirror existing `benchmark/`, `tracking/` patterns in codebase
- Pitfalls: HIGH — OOM behavior from Phase 2 OOM monitor implementation; JSONL/checkpoint from CONTEXT.md design; profile override from engine.__init__ code reading
- Integration points: HIGH — exact file paths and method signatures verified from source

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable internal codebase — no external API drift risk)
