# Phase 7: OBB Tune Sweep Dispatch Fix — Research

**Researched:** 2026-03-08
**Domain:** Auto-tuning sweep engine dispatch — `yowo/tune/_sweep.py`
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TUNE-01 | User can run `yowo tune --model MODEL` to auto-detect optimal backend, batch size, and precision for current hardware | INT-C1 gap: `_measure_config` always dispatches DetectionEngine; must add OBB branch |
| TUNE-02 | Auto-tune runs calibration sweep across available backends and precision levels | Same root cause: sweep skips all OBB configs due to warmup failure inside `_measure_config` |
| TUNE-03 | Auto-tune results persist to device-specific profile file for instant startup on subsequent runs | Profile save is unreachable for OBB models because run_sweep returns empty results; fixing dispatch enables save |
</phase_requirements>

---

## Summary

Phase 7 closes a single, well-scoped functional gap: `_measure_config()` in `src/yowo/tune/_sweep.py` hardcodes `InferenceConfig` (detection) and `DetectionEngine` regardless of `spec.task`. When `yowo tune --model yolo11n-obb` is invoked, `run_sweep()` calls `_measure_config()` for every configuration combination. Each call instantiates a `DetectionEngine` with an OBB model spec, which fails at warmup validation because the OBB head emits shape `(B, 4+nc+1, A)` (20 channels for nc=15) while `DetectionEngine._validate_output_values()` expects detection-shaped output. All configurations are logged as skipped errors and `run_sweep()` returns an empty list. Because `run_sweep()` returned nothing, `tune_command` prints "No results — all configurations were skipped." and exits without writing a profile.

The exact fix is adding an `elif task == "obb"` branch inside `_measure_config` that mirrors the already-proven pattern in `src/yowo/benchmark/_runner.py` lines 140–147: instantiate `OBBConfig` + `OBBEngine`, call `engine.detect_obb()` instead of `engine.detect()` during warmup and measurement. The OBB dispatch pattern is well established across the codebase (benchmark, engine resolver, CLI); this is the one remaining site that was not updated when OBB was added.

The CLI (`tune_command`) already derives the task-aware profile key `"yolo11n-obb"` (INT-A1, Phase 6) and `OBBEngine._load_tune_profile` already uses that same key (INT-P1, Phase 5). End-to-end, once `_measure_config` dispatches correctly, the full flow — sweep → results → profile saved under `"yolo11n-obb"` → OBBEngine auto-loads on next construction — will work without any further changes.

**Primary recommendation:** Add one `elif task == "obb":` branch in `_measure_config` mirroring the benchmark runner OBB branch. Add two regression tests: (1) `_measure_config` dispatches `OBBEngine` for `task="obb"`, (2) `_measure_config` dispatches `DetectionEngine` for `task="detect"`.

---

## Standard Stack

### Core
| Component | Location | Purpose |
|-----------|----------|---------|
| `_measure_config()` | `src/yowo/tune/_sweep.py:163–219` | Creates engine, runs warmup + timed frames, returns FPS |
| `OBBConfig` | `src/yowo/config.py:261–330` | Configuration dataclass for OBBEngine |
| `OBBEngine` | `src/yowo/obb_engine.py` | Inference engine for OBB task; dispatched by benchmark runner |
| `DetectionEngine` | `src/yowo/engine.py` | Existing engine dispatched by `_measure_config` (needs OBB branch added) |
| `InferenceConfig` | `src/yowo/config.py:65+` | Detection config; used exclusively by `_measure_config` today |

### Reference Pattern (already implemented in benchmark)

`src/yowo/benchmark/_runner.py` lines 140–147 is the canonical OBB dispatch pattern:

```python
elif task == "obb":
    from yowo.obb_engine import OBBEngine  # lazy import — patchable in tests

    engine = OBBEngine(  # type: ignore[assignment]
        model_family=model_spec.family,
        model_size=model_spec.size,
        weights_path=model_spec.weights_path,
    )
```

Warmup and measurement inside `_runner.py` uses `engine.detect_obb([warmup_frame])` for `task == "obb"`.

### Supporting — OBBConfig fields needed in _measure_config
| Field | Type | Notes |
|-------|------|-------|
| `model_family` | `ModelFamily` | from `spec.family` |
| `model_size` | `ModelSize` | from `spec.size` |
| `num_classes` | `int \| None` | from `spec.num_classes` |
| `backend` | `BackendType \| None` | sweep parameter |
| `precision` | `Precision \| None` | sweep parameter |
| `batch_size` | `int` | sweep parameter |

---

## Architecture Patterns

### Recommended Change Structure

The change is minimal and surgical — one function in one file.

```
src/yowo/tune/_sweep.py   (MODIFY)
    _measure_config()     — add elif task == "obb" branch
tests/unit/test_sweep.py  (MODIFY)
    TestMeasureConfig     — new class with 2 regression tests
```

No new files. No public API surface changes. No config.json modifications needed.

### Pattern 1: Lazy Import for Patchability

**What:** Import `OBBEngine` inside the `elif task == "obb":` branch using a lazy local import, not at module level.

**When to use:** Every engine dispatch in the codebase that was added for OBB uses lazy imports (`benchmark/_runner.py` line 141). This makes the import patchable in tests via `patch("yowo.tune._sweep.OBBEngine")` rather than requiring a full module reimport.

**Example (from benchmark/_runner.py):**
```python
elif task == "obb":
    from yowo.obb_engine import OBBEngine  # lazy import — patchable in tests

    engine = OBBEngine(
        model_family=model_spec.family,
        model_size=model_spec.size,
        weights_path=model_spec.weights_path,
    )
```

**For _measure_config the equivalent:**
```python
# Source: src/yowo/tune/_sweep.py _measure_config body
task = spec.task
if task == "obb":
    from yowo.config import OBBConfig
    from yowo.obb_engine import OBBEngine  # lazy import — patchable in tests

    config = OBBConfig(
        model_family=spec.family,
        model_size=spec.size,
        num_classes=spec.num_classes,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
    )
    engine = OBBEngine(config)
else:
    from yowo.config import InferenceConfig
    from yowo.engine import DetectionEngine

    config = InferenceConfig(
        model_family=spec.family,
        model_size=spec.size,
        num_classes=spec.num_classes,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
    )
    engine = DetectionEngine(config)
```

**Note on existing imports:** `_measure_config` currently has `from yowo.config import InferenceConfig` and `from yowo.engine import DetectionEngine` as local imports inside the function body (lines 189–190). These can be moved into the `else` branch.

### Pattern 2: detect_obb() in warmup and measurement loops

The measurement loop inside `_measure_config` calls `engine.detect(frames)`. For OBB this must be `engine.detect_obb(frames)`. Since engine type is determined by the dispatch branch, the call site also needs branching — or a unified local variable approach:

**Option A (explicit branching — matches benchmark/_runner.py style):**
```python
# Warmup (not timed)
for _ in range(warmup_frames):
    if task == "obb":
        engine.detect_obb(frames)  # type: ignore[union-attr]
    else:
        engine.detect(frames)

# Timed measurement
t0 = time.monotonic()
for _ in range(measure_frames):
    if task == "obb":
        engine.detect_obb(frames)  # type: ignore[union-attr]
    else:
        engine.detect(frames)
elapsed = time.monotonic() - t0
```

**Option B (callable alias — cleaner):**
```python
infer_fn = engine.detect_obb if task == "obb" else engine.detect  # type: ignore[union-attr]

for _ in range(warmup_frames):
    infer_fn(frames)

t0 = time.monotonic()
for _ in range(measure_frames):
    infer_fn(frames)
elapsed = time.monotonic() - t0
```

Option B is cleaner and avoids repeating the branch inside tight loops. Either is correct; the planner can choose. The type: ignore comment is required because `engine` is typed as `DetectionEngine | OBBEngine` (a union) and mypy/pyright cannot narrow through assignment.

### Pattern 3: Test Regression Structure

Existing tests in `test_sweep.py` mock `_measure_config` via `patch("yowo.tune._sweep._measure_config", ...)` — they never test `_measure_config` internals directly. The new regression tests need to test `_measure_config` directly to verify dispatch behavior. Pattern from `test_obb_integration.py` and `test_obb_engine.py`:

```python
class TestMeasureConfigDispatch:
    def _make_spec(self, task: str) -> MagicMock:
        spec = MagicMock()
        spec.family = ModelFamily.YOLO11
        spec.size = ModelSize.NANO
        spec.task = task
        spec.num_classes = None
        spec.weights_path = None
        return spec

    def test_measure_config_dispatches_obb_engine_for_task_obb(self) -> None:
        """_measure_config must instantiate OBBEngine when spec.task == 'obb'."""
        from yowo.tune._sweep import _measure_config

        hw = MagicMock()
        spec = self._make_spec("obb")

        mock_engine = MagicMock()
        raw_obb = np.zeros((1, 20, 8400), dtype=np.float32)  # 4+15+1 channels
        mock_engine.detect_obb.return_value = [MagicMock()]

        with patch("yowo.tune._sweep.OBBEngine", return_value=mock_engine) as MockOBBEngine:
            # patch load() to succeed without real weights
            mock_engine.load.return_value = None
            _measure_config(spec, hw, BackendType.PYTORCH, Precision.FP32, 1, 1, 1)

        MockOBBEngine.assert_called_once()
        mock_engine.detect_obb.assert_called()

    def test_measure_config_dispatches_detection_engine_for_task_detect(self) -> None:
        """_measure_config must instantiate DetectionEngine when spec.task == 'detect'."""
        from yowo.tune._sweep import _measure_config

        hw = MagicMock()
        spec = self._make_spec("detect")

        mock_engine = MagicMock()

        with patch("yowo.tune._sweep.DetectionEngine", return_value=mock_engine) as MockDetEng:
            mock_engine.load.return_value = None
            _measure_config(spec, hw, BackendType.PYTORCH, Precision.FP32, 1, 1, 1)

        MockDetEng.assert_called_once()
        mock_engine.detect.assert_called()
```

**Important:** The patch targets `yowo.tune._sweep.OBBEngine` (not `yowo.obb_engine.OBBEngine`) because the lazy import brings it into the `_sweep` module namespace. This is consistent with how `benchmark/_runner.py` is tested in `test_obb_integration.py` (which patches `yowo.benchmark.OBBEngine`).

### Anti-Patterns to Avoid

- **Adding `OBBEngine` import at module level:** Breaks patchability in unit tests. All OBB engine additions in this codebase use lazy local imports inside the dispatch branch.
- **Using `engine.detect()` for OBB engine:** `OBBEngine` does not expose `.detect()`. It exposes `.detect_obb()`. Calling `.detect()` would raise `AttributeError`.
- **Modifying `run_sweep()` instead of `_measure_config()`:** The dispatch belongs in `_measure_config`. `run_sweep()` is task-agnostic; it just calls `_measure_config` with each configuration combination. The fix location is confirmed by the audit (INT-C1 explicitly names `_measure_config`).
- **Adding `classify` branch:** Phase 7 scope is OBB only. Classification tune dispatch is not part of this phase's gap closure.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OBB config construction | Custom parameter mapping | `OBBConfig(model_family=..., model_size=..., num_classes=..., backend=..., precision=..., batch_size=...)` | Already validated in `__post_init__`; same fields as InferenceConfig sweep parameters |
| OBB engine warmup | Custom warmup logic | `engine.detect_obb(frames)` — warmup is inherent in the first N calls | `OBBEngine.load()` performs warmup internally; the N extra calls in `_measure_config` are a calibration discard, not validation |
| Task detection | String parsing | `spec.task` field — already set to `"obb"` when model is `yolo11n-obb` via `_parse_model_spec()` | `ModelSpec.task` is authoritative; no string parsing needed |

---

## Common Pitfalls

### Pitfall 1: Patching the wrong module path for OBBEngine
**What goes wrong:** Test patches `yowo.obb_engine.OBBEngine` but the lazy import inside `_measure_config` creates a local reference `from yowo.obb_engine import OBBEngine`. The patch must target `yowo.tune._sweep.OBBEngine` (the name in the importing module's namespace, after the import executes).
**Why it happens:** Python `unittest.mock.patch` intercepts attribute lookup on the module where the name lives after import, not the source module.
**How to avoid:** Patch `yowo.tune._sweep.OBBEngine` for the sweep tests. Confirm by checking `test_obb_integration.py` which patches `yowo.benchmark.OBBEngine` for the benchmark runner — same pattern.
**Warning signs:** `MockOBBEngine.assert_called_once()` fails even though the code path runs.

### Pitfall 2: Forgetting detect_obb() in both warmup AND measure loops
**What goes wrong:** Developer changes the engine dispatch but only updates the timed measurement loop, leaving warmup calling `engine.detect()`. `OBBEngine` has no `.detect()` — `AttributeError` at warmup.
**Why it happens:** There are two call sites in `_measure_config` — the warmup loop and the measurement loop. Both must use the task-appropriate method.
**How to avoid:** Update both loops, or use a callable alias (`infer_fn`) assigned once before the loops.

### Pitfall 3: OBBConfig missing backend/precision/batch_size at default
**What goes wrong:** `OBBConfig` is constructed without `backend`, `precision`, `batch_size` when the sweep values are passed. `OBBEngine.__init__` then calls `_load_tune_profile` which may override sweep parameters from a stale profile, corrupting sweep results.
**Why it happens:** `OBBEngine` auto-applies tune profile in `__init__` when `backend_instance is None`. If the sweep config doesn't explicitly set backend/precision/batch_size, the profile overrides them.
**How to avoid:** Always pass `backend=backend`, `precision=precision`, `batch_size=batch_size` to `OBBConfig` in `_measure_config`. This matches how `InferenceConfig` is currently constructed (lines 192–199).

### Pitfall 4: OBBEngine type annotation clash in _measure_config return path
**What goes wrong:** Pyright reports type error: `engine` is typed as `DetectionEngine` at the `detect()` call site after being conditionally assigned as `OBBEngine`.
**Why it happens:** `_measure_config` currently has `engine: DetectionEngine` as the sole type. After the change, `engine` can be either.
**How to avoid:** The function can leave `engine` without an explicit annotation (Python's type inference handles the union), or annotate with `DetectionEngine | OBBEngine`. Use `# type: ignore[union-attr]` on the `.detect_obb()` calls if pyright can't narrow. The benchmark runner uses the same pattern (line 169: `# type: ignore[union-attr]`).

---

## Code Examples

### Full _measure_config after fix (verified pattern)
```python
# Source: src/yowo/tune/_sweep.py — _measure_config (with OBB branch added)
def _measure_config(
    spec: ModelSpec,
    hw: HardwareProfile,
    backend: BackendType,
    precision: Precision,
    batch_size: int,
    warmup_frames: int,
    measure_frames: int,
) -> float:
    task = spec.task
    if task == "obb":
        from yowo.config import OBBConfig
        from yowo.obb_engine import OBBEngine  # lazy import — patchable in tests

        config = OBBConfig(
            model_family=spec.family,
            model_size=spec.size,
            num_classes=spec.num_classes,
            backend=backend,
            precision=precision,
            batch_size=batch_size,
        )
        engine = OBBEngine(config)
    else:
        from yowo.config import InferenceConfig
        from yowo.engine import DetectionEngine

        config = InferenceConfig(
            model_family=spec.family,
            model_size=spec.size,
            num_classes=spec.num_classes,
            backend=backend,
            precision=precision,
            batch_size=batch_size,
        )
        engine = DetectionEngine(config)

    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    frames = [Frame(pixels=pixels)]

    try:
        engine.load()

        # Warmup (not timed)
        infer = engine.detect_obb if task == "obb" else engine.detect  # type: ignore[union-attr]
        for _ in range(warmup_frames):
            infer(frames)

        # Timed measurement
        t0 = time.monotonic()
        for _ in range(measure_frames):
            infer(frames)
        elapsed = time.monotonic() - t0

        return measure_frames / elapsed
    finally:
        engine.close()
```

### Key difference from current code
Current lines 189–200 in `_measure_config`:
```python
from yowo.config import InferenceConfig
from yowo.engine import DetectionEngine

config = InferenceConfig(
    model_family=spec.family,
    model_size=spec.size,
    num_classes=spec.num_classes,
    backend=backend,
    precision=precision,
    batch_size=batch_size,
)
engine = DetectionEngine(config)
```
These become the `else` branch. No other logic in the function changes.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (uv run pytest) |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest tests/unit/test_sweep.py -x -q` |
| Full suite command | `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TUNE-01 (OBB path) | `yowo tune --model yolo11n-obb` completes without warmup validation error | unit | `uv run pytest tests/unit/test_sweep.py::TestMeasureConfigDispatch -x -q` | Wave 0 |
| TUNE-02 (OBB path) | Sweep produces non-empty results for OBB spec | unit | `uv run pytest tests/unit/test_sweep.py::TestMeasureConfigDispatch -x -q` | Wave 0 |
| TUNE-03 (OBB path) | Profile is saved under OBB-keyed name (existing test_obb_integration.py covers profile key) | unit | `uv run pytest tests/unit/test_obb_integration.py -x -q` | Exists |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_sweep.py -x -q`
- **Per wave merge:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_sweep.py` — extend with `TestMeasureConfigDispatch` class (2 tests: OBB dispatch, detection dispatch)

*(Existing test infrastructure covers all other phase requirements — no new test files needed)*

---

## Sources

### Primary (HIGH confidence)
- `src/yowo/tune/_sweep.py` — full source read, `_measure_config` implementation at lines 163–219
- `src/yowo/benchmark/_runner.py` — OBB dispatch reference pattern lines 140–147, 168–169
- `src/yowo/obb_engine.py` — `OBBEngine.__init__`, `detect_obb()`, `_validate_output_values()` confirming incompatibility with DetectionEngine dispatch
- `src/yowo/config.py` — `OBBConfig` fields (lines 261–330) confirming constructor signature
- `.planning/v2.3-MILESTONE-AUDIT.md` — INT-C1 gap description with exact file/function/fix recommendation
- `tests/unit/test_sweep.py` — existing test coverage showing no `_measure_config` dispatch tests exist yet
- `tests/unit/test_obb_integration.py` — regression test pattern for OBB integration gaps

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` — decision log confirming lazy import for patchability pattern across OBB phases
- `.planning/ROADMAP.md` — Phase 7 success criteria confirming single-plan scope

---

## Metadata

**Confidence breakdown:**
- Bug location: HIGH — INT-C1 audit pinpoints `_measure_config()` in `src/yowo/tune/_sweep.py` exactly
- Fix approach: HIGH — reference pattern already exists and works in `benchmark/_runner.py`
- Test pattern: HIGH — existing tests in `test_sweep.py` and `test_obb_integration.py` show established patterns
- Scope: HIGH — single-function change, no API surface changes, no new files required

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable codebase, no external dependencies involved)

---

## Open Questions

None. The gap is fully characterized by INT-C1 in the audit, the fix pattern is established in the codebase, and all affected code has been read directly.
