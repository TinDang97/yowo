# Phase 8: OBB UX Fix and Tech Debt Cleanup - Research

**Researched:** 2026-03-08
**Domain:** CLI error messaging + internal function deduplication (Python, Click, pytest)
**Confidence:** HIGH

## Summary

Phase 8 is a two-change patch with zero ambiguity: both changes are fully specified by the audit trail and the existing code is directly readable. No external library research is required.

**Change 1 — OBB error message (cosmetic UX):** `benchmark_command` in `src/yowo/cli/_main.py` lines 79–96 dispatches on `is_cls = "-cls" in model` but has no OBB branch. When `--model yolo11n-obb` is passed with a nonexistent path, it falls into the `else` branch and prints COCO layout hints. The fix is to add `is_obb = "-obb" in model` and a third branch that prints DOTA layout hints, before the existing `else` branch.

**Change 2 — Duplicate function removal (tech debt):** `benchmark/_runner.py` defines a private `_check_backend_available(backend_type: BackendType) -> bool` at line 46 that reimplements the public `check_backend_available(backend: BackendType, hw: HardwareProfile) -> None` already exported from `yowo.backends`. The signatures differ: the private version does an importlib probe and returns bool; the public version takes a HardwareProfile and raises BackendError. The public API is richer (hardware-aware, raises on mismatch). The private version is used at one call site (`run_all_backends`, line 281). Replacement requires importing `check_backend_available` from `yowo.backends`, acquiring a `HardwareProfile`, and catching `BackendError` instead of checking a bool.

**Primary recommendation:** Make both changes in a single plan (08-01-PLAN.md). No new architecture, no new dependencies.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BENCH-01 (cosmetic gap closure) | `yowo benchmark --model yolo11n-obb --data /nonexistent` shows DOTA layout hint, not COCO layout hint | Change 1: add `is_obb` branch in `benchmark_command` before the existing `else` block. Verified from audit `BENCH-01-cosmetic` in `v2.3-MILESTONE-AUDIT.md`. |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| click | already in project | CLI testing via `CliRunner` | Existing test pattern for all CLI commands |
| pytest | already in project | Unit tests | Project standard; 1615 tests already in suite |
| `yowo.backends.check_backend_available` | internal | Backend availability check | Public API promoted in Phase 3; already used by `tune/_sweep.py` |
| `yowo.hardware.get_hardware_profile` | internal | Supplies `HardwareProfile` to public API | Already imported at module level in `cli/_main.py` |

### No New Dependencies
Both changes are internal refactors. No package installs required.

---

## Architecture Patterns

### Change 1: OBB Branch in benchmark_command

Current code (`cli/_main.py` lines 78–97):

```python
data_path = Path(data)
is_cls = "-cls" in model

if not data_path.is_dir():
    if is_cls:
        click.echo(
            f"Error: Dataset path does not exist: {data}\n\n"
            f"Expected: {data}/val/n01440764/*.JPEG (torchvision ImageFolder layout)\n"
            "Download from: https://image-net.org/download.php",
            err=True,
        )
    else:
        click.echo(
            f"Error: Dataset path does not exist: {data}\n\n"
            f"Expected: {data}/val2017/ and {data}/annotations/instances_val2017.json\n"
            "Download from: https://cocodataset.org/#download",
            err=True,
        )
    sys.exit(1)
```

Fix — add `is_obb` and insert an OBB branch before `else`:

```python
data_path = Path(data)
is_cls = "-cls" in model
is_obb = "-obb" in model

if not data_path.is_dir():
    if is_cls:
        click.echo(
            f"Error: Dataset path does not exist: {data}\n\n"
            f"Expected: {data}/val/n01440764/*.JPEG (torchvision ImageFolder layout)\n"
            "Download from: https://image-net.org/download.php",
            err=True,
        )
    elif is_obb:
        click.echo(
            f"Error: Dataset path does not exist: {data}\n\n"
            f"Expected: {data}/images/ and {data}/labelTxt/ (DOTA v1 layout)\n"
            "Download from: https://captain-whu.github.io/DOTA/dataset.html",
            err=True,
        )
    else:
        click.echo(
            f"Error: Dataset path does not exist: {data}\n\n"
            f"Expected: {data}/val2017/ and {data}/annotations/instances_val2017.json\n"
            "Download from: https://cocodataset.org/#download",
            err=True,
        )
    sys.exit(1)
```

**DOTA layout hint:** The benchmark `_runner.py` loads DOTA via `load_dota_dataset` which expects `{data}/images/` and `{data}/labelTxt/`. Confirmed by reading `benchmark/__init__.py` dispatch. The DOTA download URL is the official dataset page.

### Change 2: Remove _check_backend_available from _runner.py

Current: `_check_backend_available(backend_type: BackendType) -> bool` at `_runner.py:46–64`.
Called at: `run_all_backends`, line 281: `if not _check_backend_available(bt): ... continue`.

Public API signature: `check_backend_available(backend: BackendType, hw: HardwareProfile) -> None` — raises `BackendError` if unavailable.

Replacement pattern (matches how `tune/_sweep.py` uses the public API):

```python
# At top of _runner.py — add imports
from yowo.backends import check_backend_available
from yowo.errors import BackendError
from yowo.hardware import get_hardware_profile

# In run_all_backends — replace the bool-guard with try/except
hw = get_hardware_profile()  # call once before the loop

results: list[BenchmarkResult] = []
for bt in backends_to_test:
    try:
        check_backend_available(bt, hw)
    except BackendError:
        logger.info("Skipping %s: dependencies not available", bt.value)
        continue
    try:
        result = run_single_backend(...)
        results.append(result)
    except Exception:
        logger.warning("Backend %s failed during benchmark", bt.value, exc_info=True)
        continue
```

**Important:** `get_hardware_profile()` must be called once before the loop, not per-iteration. This matches the Phase 3 decision: `_hw_cache` parameter pattern avoids double get_hardware_profile() calls.

**Delete** the entire `_check_backend_available` function (lines 46–64) after replacing the call site.

### Anti-Patterns to Avoid
- **Calling `get_hardware_profile()` inside the loop:** Performance regression — call once before iterating backends.
- **Wrapping both BackendError and the run_single_backend Exception in one try block:** Loses the "skip vs warn" distinction.
- **Leaving the private function as dead code:** Delete it entirely to close the divergence risk.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend availability check | Another importlib probe | `check_backend_available` from `yowo.backends` | Hardware-aware, raises `BackendError`, already used by sweep |
| DOTA URL | Don't guess | `https://captain-whu.github.io/DOTA/dataset.html` | Official source |

---

## Common Pitfalls

### Pitfall 1: OBB model name detection order
**What goes wrong:** If `is_cls` check were `"-cls" in model` and OBB models also somehow contained "-cls", order would matter. In practice OBB models are `yolo11n-obb` — no overlap. Still, the `elif is_obb` must come before `else` but after `if is_cls`.
**How to avoid:** Use `elif is_obb` (not a second `if`).

### Pitfall 2: BackendError import path
**What goes wrong:** `BackendError` lives in `yowo.errors`, not `yowo.backends`. Importing from the wrong module causes ImportError.
**How to avoid:** `from yowo.errors import BackendError` — confirmed by existing usage in `tune/_sweep.py` line 25.

### Pitfall 3: Test patch target after import change
**What goes wrong:** Tests in `test_benchmark_runner.py` that mock `_check_backend_available` at the old location will break after deletion.
**How to avoid:** Check if any test patches `yowo.benchmark._runner._check_backend_available` — if so, update patch targets to `yowo.backends.check_backend_available`. Scan existing tests before implementing.

### Pitfall 4: Missing test for new OBB error branch
**What goes wrong:** The new `elif is_obb` branch has no test coverage. The existing `TestBenchmarkMissingData` class only tests detection and classification models.
**How to avoid:** Add `test_benchmark_missing_data_obb` to `TestBenchmarkMissingData` in `tests/unit/test_benchmark_cli.py`.

---

## Code Examples

### Existing test pattern for CLI missing data (from test_benchmark_cli.py)

```python
def test_benchmark_missing_data_detection(self, runner: CliRunner) -> None:
    result = runner.invoke(
        cli, ["benchmark", "--model", "yolo11n", "--data", "/nonexistent/path"]
    )
    assert result.exit_code != 0
    assert "val2017" in result.output or "cocodataset" in result.output.lower()
```

New test to add:

```python
def test_benchmark_missing_data_obb(self, runner: CliRunner) -> None:
    result = runner.invoke(
        cli, ["benchmark", "--model", "yolo11n-obb", "--data", "/nonexistent/obb/path"]
    )
    assert result.exit_code != 0
    assert "DOTA" in result.output or "labelTxt" in result.output
```

### How tune/_sweep.py uses the public check_backend_available (verified)

```python
# tune/_sweep.py lines 24, 94-100
from yowo.backends import check_backend_available
from yowo.errors import BackendError

for backend in _SWEEP_BACKENDS:
    try:
        check_backend_available(backend, hw)
        available.append(backend)
    except BackendError:
        _log.debug("Backend %s not available, skipping in sweep.", backend.value)
```

This is the exact pattern to replicate in `benchmark/_runner.py`.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, 1615 tests) |
| Config file | pyproject.toml (existing) |
| Quick run command | `uv run pytest tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_runner.py -x -q` |
| Full suite command | `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 (cosmetic) | `yowo benchmark --model yolo11n-obb --data /nonexistent` exits non-zero with DOTA hint | unit | `uv run pytest tests/unit/test_benchmark_cli.py::TestBenchmarkMissingData::test_benchmark_missing_data_obb -x` | ❌ Wave 0 |
| Tech debt | `run_all_backends` uses public `check_backend_available`, no local duplicate | unit | `uv run pytest tests/unit/test_benchmark_runner.py -x -q` | ✅ (existing, may need update) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_runner.py -x -q`
- **Per wave merge:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Add `test_benchmark_missing_data_obb` to `tests/unit/test_benchmark_cli.py::TestBenchmarkMissingData` — covers BENCH-01 cosmetic
- [ ] Verify no existing test patches `yowo.benchmark._runner._check_backend_available` (scan before implementing Change 2)

---

## Sources

### Primary (HIGH confidence)
- Direct code read: `src/yowo/cli/_main.py` lines 67–97 — benchmark_command, exact current logic
- Direct code read: `src/yowo/benchmark/_runner.py` lines 46–64, 280–283 — `_check_backend_available` definition and call site
- Direct code read: `src/yowo/backends/_selector.py` lines 361–388 — public `check_backend_available` signature
- Direct code read: `src/yowo/tune/_sweep.py` lines 24, 94–100 — precedent pattern for using public API
- Direct code read: `tests/unit/test_benchmark_cli.py` — existing test class structure
- `.planning/v2.3-MILESTONE-AUDIT.md` — BENCH-01-cosmetic gap definition, tech debt item for `_check_backend_available`

### Secondary (MEDIUM confidence)
- DOTA dataset URL: https://captain-whu.github.io/DOTA/dataset.html — standard dataset page, consistent with `images/` + `labelTxt/` layout used in `benchmark/__init__.py`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all changes are internal, no new dependencies
- Architecture: HIGH — both changes fully specified by existing code and audit trail
- Pitfalls: HIGH — identified from direct code inspection and existing test patterns

**Research date:** 2026-03-08
**Valid until:** Stable — no external dependencies involved; only valid while codebase stays at current state
