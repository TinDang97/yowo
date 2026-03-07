# Phase 1: Correctness, Benchmarking, and Proactive Fixes - Research

**Researched:** 2026-03-07
**Domain:** Inference correctness validation, benchmark CLI tooling, production reliability fixes
**Confidence:** HIGH

## Summary

Phase 1 establishes the correctness and reliability baseline for YOWO by: (1) validating detection/classification mAP parity with ultralytics across all model variants and export formats, (2) building a `yowo benchmark` CLI for automated accuracy and performance measurement, and (3) fixing known production pitfalls (RTSP memory leak, thread safety, deterministic NMS, export compatibility, graceful fallback errors).

The codebase is well-structured for these changes. Warmup validation (CORR-08) slots into existing `warmup()` methods on all 5 backends. Thread safety (PFIX-02) adds a lock to `BaseEngine`. The benchmark CLI follows the established click subcommand pattern in `cli/_main.py`. RTSP reconnect (PFIX-01) integrates into `ThreadedFrameReader`. NMS determinism (PFIX-03) requires a post-sort on `_class_aware_nms` output. Error messaging (PFIX-05) enhances existing `DependencyError` and `_check_backend_available()`.

**Primary recommendation:** Tackle correctness validation (mAP computation + warmup) first, then benchmark CLI (which depends on mAP), then proactive fixes (independent of each other, parallelizable).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New `yowo benchmark` CLI subcommand with rich colored terminal table output (columns: Format, mAP, FPS, Model Size, Device)
- `--json` flag for machine-readable output
- Metrics per format: mAP@0.5:0.95, average FPS, model file size
- If ultralytics is importable, automatically run same model through it and show side-by-side comparison; if not, show YOWO-only results with a note
- Default: test all available backends on current hardware; `--format onnx,trt` flag to filter specific formats; skips unavailable backends with a note
- `--data /path/to/dataset` flag for dataset path (required for mAP)
- `--subset N` flag to run on first N images for quick dev iteration; full dataset by default
- Benchmark results persistable to JSON (BENCH-04)
- User downloads datasets manually; `yowo benchmark` accepts `--data /path/to/coco/val2017` (detection) or `--data /path/to/imagenet/val` (classification)
- If dataset path missing or invalid, print clear download instructions (URL + expected directory structure)
- No auto-download (COCO val2017 ~6GB, ImageNet val ~6.3GB)
- Same pattern for both detection (COCO) and classification (ImageNet)
- mAP computation uses pycocotools (official COCO evaluator) as optional dependency
- Warmup validates output tensor shape AND confidence value range [0, 1] on dummy forward pass
- Same validation logic for all backends (no backend-specific checks)
- Applies to both detection and classification engines
- Classification validation: check softmax output sums to ~1.0 and values in [0, 1]
- On validation failure: raise `WarmupValidationError` with details (shape mismatch, bad value range); engine stays unloaded; fail-fast, refuse to serve
- Structured error messages with install command for missing backends
- Format: show what's missing (checked/unchecked package list), install command (`uv add ...`), and list of available backends
- Applied to all 5 backends (PyTorch, ONNX, TensorRT, OpenVINO, CoreML)
- Both CLI command (`yowo info --compat`) and documentation for export compatibility
- CLI version auto-detects current system versions (CUDA, TensorRT, OpenVINO, etc.) and shows compatibility matrix
- Engine is thread-safe by default using internal lock for inference calls
- Users can safely call detect()/classify() from multiple threads without external coordination
- Small locking overhead accepted for safety
- Periodic automatic reconnect of VideoCapture every N minutes (configurable) to reset leaked memory
- Transparent to user; reconnect happens between frames
- Built into existing ThreadedFrameReader

### Claude's Discretion
- Deterministic NMS implementation approach (PFIX-03)
- Exact benchmark FPS measurement methodology (warmup runs, averaging strategy)
- Internal locking mechanism for thread safety (threading.Lock vs RLock)
- Reconnect interval default value for RTSP leak prevention
- Export compatibility matrix content and format details

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORR-01 | Detection mAP matches ultralytics within 0.5% on COCO val2017 for all 10 detection variants | pycocotools COCOeval for official mAP; benchmark module runs both YOWO and ultralytics side-by-side |
| CORR-02 | Classification top-1 accuracy matches ultralytics within 0.5% on ImageNet val | ImageNet val loader + top-1 accuracy computation; same benchmark module |
| CORR-03 | Per-frame inference latency within 10% of ultralytics on same hardware | FPS measurement in benchmark module with warmup + averaging |
| CORR-04 | ONNX exported models produce correct results on CUDA server | Export + load + mAP validation via benchmark on onnxruntime-gpu |
| CORR-05 | TensorRT exported engines produce correct results on Jetson | Export + load + mAP validation via benchmark on TensorRT backend |
| CORR-06 | OpenVINO exported models produce correct results on Intel NUC | Export + load + mAP validation via benchmark on OpenVINO backend |
| CORR-07 | Export accuracy delta vs PyTorch baseline < 1% mAP per format | Benchmark comparison table: PyTorch mAP vs each exported format mAP |
| CORR-08 | Warmup validates output shape and value range before accepting requests | Warmup validation added to engine load() after backend.warmup(); WarmupValidationError on failure |
| BENCH-01 | `yowo benchmark --model MODEL` produces mAP + FPS + model size per format | New benchmark CLI subcommand with rich table output |
| BENCH-02 | Benchmark includes comparison table: format, mAP, FPS, model size, device | Rich Table with columns; optional ultralytics comparison column |
| BENCH-03 | Benchmark supports all backends (PyTorch, ONNX, TensorRT, OpenVINO) | Iterate available backends, skip unavailable with note |
| BENCH-04 | Benchmark results persistable to JSON | `--json` flag writes structured JSON to stdout or file |
| PFIX-01 | RTSP memory leak prevention | Periodic VideoCapture reconnect in ThreadedFrameReader |
| PFIX-02 | Thread safety for concurrent engine access | threading.Lock in BaseEngine wrapping _run_gpu() |
| PFIX-03 | Deterministic NMS output ordering | Stable sort by (confidence desc, class_id asc, x1 asc) after NMS |
| PFIX-04 | Export compatibility matrix | `yowo info --compat` CLI + docs |
| PFIX-05 | Graceful fallback with clear error messages | Enhanced DependencyError with structured install instructions using `uv add` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pycocotools | >=2.0.4 | Official COCO mAP evaluation (AP@0.5:0.95) | The canonical COCO evaluator; used by ultralytics, detectron2, mmdetection. No alternative produces identical numbers. |
| rich | >=13.0 | Colored terminal tables for benchmark output | De-facto standard for styled CLI output. Already used by click ecosystem. Tables, progress bars, colors. |
| click | >=8.1 (existing) | CLI framework for `yowo benchmark` subcommand | Already the CLI framework; benchmark follows established pattern |
| threading (stdlib) | -- | Lock for thread-safe engine access | Standard library; no external dependency needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ultralytics | >=8.4.21 (dev dep, already present) | Side-by-side mAP comparison in benchmark | Only when importable; benchmark degrades gracefully without it |
| json (stdlib) | -- | Benchmark result persistence | Always, for `--json` output |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pycocotools | Custom mAP implementation | Custom would not produce official COCO numbers; ultralytics uses pycocotools internally |
| rich | click.echo with ANSI codes | Fragile, no table formatting, no Windows support |
| threading.Lock | asyncio.Lock | Engine API is sync; threading.Lock is correct for thread safety |

**Installation:**
```bash
# New optional dependency for benchmarking
uv add --group dev pycocotools rich

# pycocotools also available as optional dep for users
# Add to pyproject.toml: benchmark = ["pycocotools>=2.0.4", "rich>=13.0"]
```

## Architecture Patterns

### Recommended Project Structure
```
src/yowo/
├── benchmark/              # NEW: benchmark module
│   ├── __init__.py         # Public API: run_benchmark()
│   ├── _evaluator.py       # mAP computation via pycocotools, top-1 accuracy
│   ├── _runner.py          # Per-backend benchmark execution (load, infer N images, measure)
│   ├── _comparison.py      # Ultralytics side-by-side comparison (optional)
│   └── _report.py          # Rich table rendering + JSON serialization
├── cli/
│   └── _main.py            # Add benchmark subcommand + info --compat
├── backends/
│   └── (existing)          # warmup() enhanced with validation
├── engine.py               # BaseEngine: thread lock, warmup validation
├── errors.py               # Add WarmupValidationError
└── io/
    └── _reader.py          # ThreadedFrameReader: RTSP periodic reconnect
```

### Pattern 1: Warmup Validation (CORR-08)
**What:** After backend.warmup() runs a dummy inference pass, validate the output tensor shape and value ranges before marking engine as loaded.
**When to use:** Every engine load(), for all backends, both detection and classification.
**Example:**
```python
# In engine.py BaseEngine.load(), after backend.warmup():
def _validate_warmup_output(self) -> None:
    """Validate backend output shape and value range on dummy input."""
    h, w = self._model_meta.input_height, self._model_meta.input_width
    dummy_data = np.zeros((1, 3, h, w), dtype=np.float32)
    dummy_tensor = PreprocessedTensor(
        data=dummy_data, batch_size=1,
        original_shapes=[(h, w)], scale_factors=[(1.0, 1.0)],
        pad_offsets=[(0, 0)],
    )
    output = self._backend.infer(dummy_tensor)

    # Shape validation: must be 3D (B, ..., ...)
    if output.ndim < 2:
        raise WarmupValidationError(
            f"Expected output ndim >= 2, got {output.ndim}. Shape: {output.shape}"
        )

    # Detection: values should contain confidence scores in [0, 1]
    # Classification: softmax output should sum to ~1.0
    # Subclass implements _validate_output_values()
    self._validate_output_values(output)
```

### Pattern 2: Thread-Safe Engine (PFIX-02)
**What:** Internal `threading.Lock` around GPU inference path to prevent concurrent backend.infer() calls.
**When to use:** Always (default safe behavior).
**Example:**
```python
# In engine.py BaseEngine.__init__:
self._infer_lock = threading.Lock()

# In _run_gpu():
def _run_gpu(self, tensor, frames):
    with self._infer_lock:
        # ... existing infer logic ...
```
**Recommendation:** Use `threading.Lock` (not RLock). RLock allows reentrant acquisition which could mask bugs. The inference path should never be reentrant -- detect() calls _run_batch() calls _run_gpu(), a linear chain. No recursion risk.

### Pattern 3: Deterministic NMS (PFIX-03)
**What:** Sort NMS output by a deterministic key to ensure identical ordering across runs.
**When to use:** After cv2.dnn.NMSBoxes returns indices.
**Example:**
```python
# In postprocess/_nms.py _class_aware_nms():
# Current: return np.sort(np.asarray(indices, dtype=np.intp).ravel())
# Problem: np.sort on indices gives position-based ordering, which is
# deterministic for identical inputs but cv2.dnn.NMSBoxes output order
# is NOT guaranteed to be deterministic for boxes with equal IoU.
#
# Fix: Sort by (confidence DESC, class_id ASC, x1 ASC) as tiebreaker
kept = np.asarray(indices, dtype=np.intp).ravel()
# Build sort key: primary=confidence (desc), secondary=class_id, tertiary=x1
sort_key = np.lexsort((
    boxes_xyxy[kept, 0],    # tertiary: x1 ascending (leftmost first)
    class_ids[kept],         # secondary: class_id ascending
    -scores[kept],           # primary: confidence descending (negate for ascending sort)
))
return kept[sort_key]
```

### Pattern 4: RTSP Periodic Reconnect (PFIX-01)
**What:** ThreadedFrameReader periodically releases and re-creates the underlying VideoCapture to reset leaked native memory.
**When to use:** Only for RTSP/network stream sources (not files or webcams).
**Example:**
```python
# In io/_reader.py ThreadedFrameReader:
# Add reconnect_interval_sec parameter (default: 300 = 5 minutes)
# In _reader_loop(), track elapsed time since last reconnect.
# When interval exceeded, release old capture, create new one, continue reading.
# Reconnect BETWEEN frames (after successful read, before next iteration).
def _maybe_reconnect(self) -> None:
    elapsed = time.monotonic() - self._last_reconnect
    if elapsed < self._reconnect_interval:
        return
    if not self._source.is_rtsp:  # only reconnect streams, not files
        return
    self._source.reconnect()  # release + re-open VideoCapture
    self._last_reconnect = time.monotonic()
```
**Recommendation:** Default reconnect interval of 300 seconds (5 minutes). The research shows OpenCV leaks ~1MB per 2-3 hours per stream. At 5-minute intervals, memory growth is bounded to a few MB before reset. Frequent enough to prevent accumulation, rare enough to not cause visible frame gaps (reconnect takes <1 second for local RTSP).

### Pattern 5: Benchmark CLI Architecture
**What:** `yowo benchmark` iterates available backends, runs mAP evaluation via pycocotools, measures FPS, and renders a rich table.
**When to use:** User wants to measure accuracy and performance.
**Example:**
```python
# CLI signature:
@cli.command("benchmark")
@click.option("--model", "-m", required=True)
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--format", "formats", default=None)  # comma-separated: onnx,trt
@click.option("--subset", default=None, type=int)
@click.option("--json", "json_output", is_flag=True)
@click.option("--output", "-o", default=None, type=click.Path())

# Execution flow:
# 1. Parse model spec, detect task (detection vs classification)
# 2. Load dataset (COCO annotations or ImageNet val directory)
# 3. For each available backend:
#    a. Export model if needed (ONNX, TensorRT, OpenVINO)
#    b. Load engine with backend
#    c. Warmup (N passes, discard)
#    d. Run inference on all/subset images, collect predictions
#    e. Compute mAP via pycocotools (detection) or top-1 accuracy (classification)
#    f. Compute average FPS (excluding warmup)
#    g. Record model file size
# 4. If ultralytics importable, run same model through ultralytics.YOLO.val()
# 5. Render rich Table or JSON output
```

### Anti-Patterns to Avoid
- **Warmup validation in individual backends:** Validation logic must be in engine.py (shared), not per-backend. Context says "same validation logic for all backends."
- **RLock for thread safety:** Use plain Lock. RLock masks reentrant bugs. The inference path is strictly linear.
- **Auto-downloading datasets in benchmark:** User decision -- no auto-download. Print download instructions instead.
- **Hardcoded mAP thresholds in tests:** Use configurable tolerance. The 0.5% threshold is for validation, not for breaking tests during development.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| COCO mAP computation | Custom IoU/AP calculation | pycocotools COCOeval | 80-class averaging, 10 IoU thresholds (0.5:0.05:0.95), area-based splits (small/medium/large). Hundreds of edge cases. Ultralytics uses this internally -- same evaluator ensures comparable numbers. |
| Terminal table rendering | ANSI escape code strings | rich.table.Table | Cross-platform (Windows included), Unicode box characters, column alignment, color styles, automatic terminal width detection |
| ImageNet accuracy computation | Manual top-k matching | Simple loop + argmax comparison | Top-1 accuracy IS simple enough to hand-roll (argmax == label). No library needed. |
| COCO annotation parsing | Custom JSON parser | pycocotools.coco.COCO | Handles annotation format, category mapping, image-id resolution |

**Key insight:** mAP computation is the critical "don't hand-roll" item. A custom implementation will produce different numbers than the official evaluator, making the 0.5% parity claim unverifiable. pycocotools is the only way to get numbers that are directly comparable to ultralytics.

## Common Pitfalls

### Pitfall 1: COCO Category ID Mapping
**What goes wrong:** COCO uses non-contiguous category IDs (1-90 with gaps). YOLO uses contiguous class IDs (0-79). If benchmark predictions use YOLO class IDs directly in COCO format, pycocotools computes wrong mAP (categories don't match).
**Why it happens:** YOLO training datasets remap COCO's 91 categories to 80 contiguous IDs. The postprocess module outputs YOLO IDs (0-79) which must be mapped back to COCO IDs (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, ...) for evaluation.
**How to avoid:** Maintain a YOLO-to-COCO category mapping array. Apply it when converting predictions to COCO result format before passing to COCOeval.
**Warning signs:** mAP is suspiciously low (< 10%) or zero for many categories.

### Pitfall 2: Warmup Skewing FPS Measurements
**What goes wrong:** First N inference passes are significantly slower (CUDA context init, JIT compilation, TensorRT tactic selection). Including them in FPS average gives misleadingly low numbers.
**Why it happens:** GPU backends have cold-start overhead. TensorRT can be 10-100x slower on first pass.
**How to avoid:** Run 5-10 warmup passes (discard results), then measure N passes for FPS. Report warmup time separately if relevant.
**Warning signs:** FPS much lower than expected; first-pass latency 10x+ slower than subsequent.

**Recommendation for FPS methodology:** 10 warmup passes (discarded), then measure all dataset images. Report: average FPS, p50/p95/p99 latency. This is standard in inference benchmarking.

### Pitfall 3: cv2.dnn.NMSBoxes Non-Determinism
**What goes wrong:** When multiple boxes have identical IoU overlap with the selected box, the order of suppression is not deterministic. This means the set of surviving boxes can differ between runs, making results non-reproducible.
**Why it happens:** OpenCV's NMS implementation uses greedy selection. When scores are equal or IoU values are exactly at the threshold boundary, floating-point comparison order depends on memory layout which can vary.
**How to avoid:** Apply a deterministic tiebreaker sort AFTER NMS. Sort by (confidence desc, class_id asc, x1 asc). This ensures identical output ordering regardless of NMS internal ordering.
**Warning signs:** Same image producing different detection count or different box ordering between runs.

### Pitfall 4: ThreadedFrameReader Reconnect Race Condition
**What goes wrong:** If reconnect happens while a frame is being read, the old VideoCapture may be released while read() is in progress, causing a segfault or undefined behavior in OpenCV's C++ layer.
**Why it happens:** OpenCV VideoCapture is not thread-safe. Release + re-open must not overlap with read().
**How to avoid:** Reconnect logic must be in the SAME thread as the reader loop (the background daemon thread), never from an external timer thread. Reconnect after a successful read(), before the next iteration.

### Pitfall 5: Lock Granularity for Thread Safety
**What goes wrong:** Locking too broadly (entire detect() call including preprocessing and postprocessing) unnecessarily serializes CPU work. Locking too narrowly (only backend.infer()) leaves shared state unprotected.
**Why it happens:** Temptation to wrap the highest-level method in a lock.
**How to avoid:** Lock only `_run_gpu()` (backend.infer + metrics recording). Preprocessing (CPU) and postprocessing (CPU NMS) happen outside the lock. This maximizes parallelism: thread A can preprocess while thread B infers.

## Code Examples

### COCO mAP Evaluation with pycocotools
```python
# Source: pycocotools official API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# YOLO class ID (0-79) to COCO category ID mapping
YOLO_TO_COCO = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

def evaluate_coco_map(
    gt_ann_path: str,
    predictions: list[dict],  # [{image_id, category_id, bbox, score}, ...]
) -> dict[str, float]:
    """Run official COCO evaluation, return mAP metrics."""
    coco_gt = COCO(gt_ann_path)
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP_50_95": coco_eval.stats[0],  # AP@[0.5:0.95]
        "mAP_50": coco_eval.stats[1],      # AP@0.5
        "mAP_75": coco_eval.stats[2],      # AP@0.75
    }
```

### Rich Benchmark Table
```python
# Source: rich.readthedocs.io
from rich.console import Console
from rich.table import Table

def render_benchmark_table(results: list[dict], ultralytics_results: dict | None) -> None:
    console = Console()
    table = Table(title="YOWO Benchmark Results")
    table.add_column("Format", style="cyan", no_wrap=True)
    table.add_column("mAP@0.5:0.95", justify="right", style="green")
    table.add_column("FPS", justify="right", style="yellow")
    table.add_column("Model Size", justify="right")
    table.add_column("Device", style="blue")
    if ultralytics_results:
        table.add_column("Ultralytics mAP", justify="right", style="magenta")

    for r in results:
        row = [
            r["format"], f"{r['map']:.4f}", f"{r['fps']:.1f}",
            f"{r['size_mb']:.1f} MB", r["device"],
        ]
        if ultralytics_results:
            delta = r["map"] - ultralytics_results.get("map", 0)
            row.append(f"{ultralytics_results['map']:.4f} ({delta:+.4f})")
        table.add_row(*row)
    console.print(table)
```

### WarmupValidationError
```python
# In errors.py:
class WarmupValidationError(BackendError):
    """Backend warmup output failed shape or value range validation.

    Raised when a dummy inference produces unexpected output, indicating
    the model is corrupt, misconfigured, or incompatible with the backend.
    """
    def __init__(self, detail: str) -> None:
        super().__init__(
            f"Warmup validation failed: {detail}. "
            f"The model may be corrupt or incompatible with this backend."
        )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom mAP computation | pycocotools COCOeval | Standard since COCO 2017 | Official numbers, directly comparable to all published results |
| print() for CLI tables | rich.Table | rich v10+ (2021) | Cross-platform colored tables, progress bars |
| No warmup validation | Shape + value range validation | Best practice since TensorRT 8+ | Catches corrupt models before serving; prevents silent wrong results |
| Per-call locking | GIL-only thread safety | Python 3.13 free-threaded removes GIL | Must add explicit locks now; GIL-reliance is deprecated pattern |

**Deprecated/outdated:**
- Relying on Python GIL for thread safety: Python 3.13+ free-threaded builds remove GIL. YOWO already has notes about PEP 703 in ThreadedFrameReader.
- `pip install` commands in error messages: User convention is `uv add` per CLAUDE.md.

## Open Questions

1. **COCO val2017 annotations path convention**
   - What we know: COCO val2017 has `instances_val2017.json` annotations file
   - What's unclear: Whether to expect `--data /path/to/coco/val2017` (images dir) + separate `--annotations` flag, or a single path to the COCO root containing both `val2017/` and `annotations/`
   - Recommendation: Accept `--data /path/to/coco` where coco root contains `val2017/` and `annotations/instances_val2017.json`. This is the standard COCO directory layout. Error message shows expected structure if not found.

2. **ImageNet val label format**
   - What we know: ImageNet val has 50,000 images in 1,000 class folders
   - What's unclear: Whether to use folder-based labels (standard torchvision ImageFolder layout) or a separate label file
   - Recommendation: Use folder-based layout (`val/n01440764/*.JPEG`). This is the standard torchvision convention. Map folder names to class indices.

3. **Export step in benchmark**
   - What we know: Benchmarking ONNX/TRT/OpenVINO requires exported models
   - What's unclear: Should benchmark auto-export before benchmarking, or require pre-exported models?
   - Recommendation: Auto-export to a temp directory if no pre-exported model exists. Show export time separately. This gives users a single-command experience.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 (existing) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest tests/unit/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORR-08 | Warmup validates shape and value range | unit | `uv run pytest tests/unit/test_warmup_validation.py -x` | Wave 0 |
| PFIX-01 | RTSP periodic reconnect resets memory | unit | `uv run pytest tests/unit/test_reader.py::test_rtsp_reconnect -x` | Wave 0 |
| PFIX-02 | Thread-safe concurrent engine access | unit | `uv run pytest tests/unit/test_thread_safety.py -x` | Wave 0 |
| PFIX-03 | Deterministic NMS output ordering | unit | `uv run pytest tests/unit/test_nms.py::test_deterministic_ordering -x` | Wave 0 |
| PFIX-04 | Export compatibility matrix CLI | unit | `uv run pytest tests/unit/test_cli.py::test_info_compat -x` | Wave 0 |
| PFIX-05 | Graceful error messages for missing backends | unit | `uv run pytest tests/unit/test_selector.py::test_error_messages -x` | Partial (test_selector.py exists) |
| BENCH-01 | Benchmark CLI produces output | unit | `uv run pytest tests/unit/test_benchmark_cli.py -x` | Wave 0 |
| BENCH-04 | Benchmark JSON persistence | unit | `uv run pytest tests/unit/test_benchmark_cli.py::test_json_output -x` | Wave 0 |
| CORR-01 | Detection mAP parity | integration | `uv run pytest tests/unit/test_benchmark_evaluator.py -x` | Wave 0 |
| CORR-02 | Classification accuracy parity | integration | `uv run pytest tests/unit/test_benchmark_evaluator.py::test_classification -x` | Wave 0 |
| CORR-04-07 | Export format correctness | integration | Requires real hardware + datasets; manual validation | manual-only (hardware-dependent) |

### Sampling Rate
- **Per task commit:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Per wave merge:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_warmup_validation.py` -- covers CORR-08
- [ ] `tests/unit/test_thread_safety.py` -- covers PFIX-02
- [ ] `tests/unit/test_benchmark_cli.py` -- covers BENCH-01, BENCH-02, BENCH-03, BENCH-04
- [ ] `tests/unit/test_benchmark_evaluator.py` -- covers CORR-01, CORR-02 (unit-testable parts with mock data)
- [ ] Add `test_deterministic_ordering` to existing `tests/unit/test_nms.py` -- covers PFIX-03
- [ ] Add `test_rtsp_reconnect` to existing `tests/unit/test_reader.py` -- covers PFIX-01
- [ ] Add `test_info_compat` to existing `tests/unit/test_cli.py` -- covers PFIX-04

## Sources

### Primary (HIGH confidence)
- YOWO codebase analysis: `src/yowo/engine.py`, `src/yowo/backends/__init__.py`, `src/yowo/postprocess/_nms.py`, `src/yowo/io/_reader.py`, `src/yowo/cli/_main.py`
- pycocotools official repository: [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) -- COCOeval API, COCO format
- Rich documentation: [rich.readthedocs.io](https://rich.readthedocs.io/en/stable/introduction.html) -- Table API
- YOWO `.planning/research/PITFALLS.md` -- OpenCV RTSP memory leak details, TensorRT compatibility

### Secondary (MEDIUM confidence)
- [OpenCV NMSBoxes inconsistency issue #26269](https://github.com/opencv/opencv/issues/26269) -- confirms non-deterministic output ordering
- [PyImageSearch COCO mAP guide](https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/) -- pycocotools usage patterns

### Tertiary (LOW confidence)
- RTSP reconnect interval (5 minutes) -- derived from pitfalls research (~1MB/2-3hrs leak rate); needs validation on real streams

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- pycocotools is the canonical COCO evaluator; rich is standard for CLI tables
- Architecture: HIGH -- all integration points exist in codebase; patterns follow established conventions
- Pitfalls: HIGH -- COCO category mapping is well-documented; NMS non-determinism confirmed by OpenCV issues
- Proactive fixes: MEDIUM -- RTSP reconnect interval is estimated; thread lock granularity recommendation based on code analysis

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable domain, no fast-moving dependencies)
