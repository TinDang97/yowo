---
phase: 03-adaptive-optimization-and-batch-processing
plan: 04
subsystem: inference
tags: [tune, calibration, cli, profile, click, tdd]

# Dependency graph
requires:
  - phase: 03-01
    provides: TuneProfile, load_profile, save_profile, compute_fingerprint
  - phase: 03-02
    provides: run_sweep, SweepResult, count_sweep_dimensions

provides:
  - _load_tune_profile() helper integrated into BaseEngine/DetectionEngine.__init__
  - yowo tune CLI subcommand with --model, --weights, --output, --force, --dry-run, --json
  - count_sweep_dimensions() public function in tune/_sweep.py
  - CliRunner-based tests for all tune CLI flags

affects:
  - phase 03-05 (batch CLI integration may reference tune profile loading pattern)
  - any future engine initialization path

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Profile auto-load transparency: no profile file present = zero behavior change"
    - "_hw_cache parameter on BaseEngine.__init__ to avoid double get_hardware_profile() call"
    - "Module-level imports for test patchability (get_hardware_profile, run_sweep, etc.)"
    - "TDD RED-GREEN: failing tests committed first then implementation"

key-files:
  created:
    - tests/unit/test_tune_cli.py
  modified:
    - src/yowo/engine.py
    - src/yowo/cli/_main.py
    - src/yowo/tune/_sweep.py

key-decisions:
  - "_hw_cache optional parameter on BaseEngine.__init__ avoids double get_hardware_profile() call when DetectionEngine pre-computes hw for profile lookup"
  - "Module-level imports in _main.py for get_hardware_profile, load_profile, save_profile, compute_fingerprint, run_sweep — test patchability requires name to live at yowo.cli._main.*"
  - "count_sweep_dimensions() added as public function to _sweep.py to avoid reportPrivateUsage pyright error from importing _enumerate_backends/_precisions_for_backend/_BATCH_SIZES"
  - "Profile auto-load skipped when backend_instance is provided — custom backend bypasses hardware selection entirely"

patterns-established:
  - "Config enrichment pattern: _load_tune_profile() applies defaults-only updates via dataclasses.replace()"
  - "Dry-run pattern: CLI computes count without side effects, prints human-readable message"
  - "Best-config selection: results[0] from run_sweep (already sorted by FPS descending)"

requirements-completed:
  - TUNE-01
  - TUNE-03

# Metrics
duration: 30min
completed: 2026-03-07
---

# Phase 3 Plan 4: Engine Profile Integration and Tune CLI Summary

**Tune profile auto-load wired into DetectionEngine.__init__ and `yowo tune` CLI subcommand with dry-run, JSON, and force modes**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-03-07T15:50:00Z
- **Completed:** 2026-03-07T16:20:00Z
- **Tasks:** 2 (Task 1: engine integration, Task 2: CLI + tests TDD)
- **Files modified:** 4

## Accomplishments
- Profile auto-load integrated into `DetectionEngine.__init__`: loads saved tune profile and applies backend/batch_size/precision only when config fields are at defaults; no behavior change when no profile exists
- `yowo tune` CLI subcommand with all required flags: `--model`, `--weights`, `--output`, `--force`, `--dry-run`, `--json`
- 10 CliRunner-based tests covering all flag combinations
- Added `count_sweep_dimensions()` as a public API in `tune/_sweep.py`

## Task Commits

1. **Task 1: BaseEngine profile auto-load integration** - `1ef2cfb` (feat)
2. **Task 2: tune CLI subcommand + TDD tests** - `a0e4707` (feat)

## Files Created/Modified
- `src/yowo/engine.py` - Added `_load_tune_profile()` helper, `_hw_cache` param on `BaseEngine.__init__`, profile call in `DetectionEngine.__init__`
- `src/yowo/cli/_main.py` - Added `tune_command()` Click subcommand, `_enumerate_sweep_dimensions()`, module-level imports for patchability
- `src/yowo/tune/_sweep.py` - Added `count_sweep_dimensions()` public function
- `tests/unit/test_tune_cli.py` - 10 CliRunner tests: help, dry-run, json, saves-profile, force flags

## Decisions Made
- `_hw_cache` optional parameter on `BaseEngine.__init__` avoids double `get_hardware_profile()` call. Without it, DetectionEngine would call it once for profile lookup and BaseEngine would call it again for backend selection — detected by existing test asserting `called_once`.
- Module-level imports in `_main.py` for `get_hardware_profile`, `load_profile`, `save_profile`, `compute_fingerprint`, `run_sweep` to support patching at `yowo.cli._main.*` in tests.
- `count_sweep_dimensions()` added as public function to avoid pyright `reportPrivateUsage` error from accessing `_enumerate_backends`, `_precisions_for_backend`, `_BATCH_SIZES` across module boundaries.
- Profile auto-load skipped when `backend_instance` is provided since custom backends bypass hardware selection entirely.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Double get_hardware_profile() call caused test assertion failure**
- **Found during:** Task 1 (engine.py integration)
- **Issue:** Adding `get_hardware_profile()` call in `DetectionEngine.__init__` for profile lookup caused existing test `test_none_uses_normal_auto_selection_path` to fail — it asserts `mock_hw.assert_called_once()` but hw was now called twice (once for profile, once in `BaseEngine.__init__`)
- **Fix:** Added `_hw_cache: HardwareProfile | None = None` optional parameter to `BaseEngine.__init__`; DetectionEngine passes pre-computed hw via `_hw_cache=_hw_cache`, BaseEngine uses it instead of calling `get_hardware_profile()` again
- **Files modified:** src/yowo/engine.py
- **Verification:** All 1739 previously-passing tests continued to pass
- **Committed in:** 1ef2cfb (Task 1 commit)

**2. [Rule 2 - Missing Critical] Pyright reportPrivateUsage for sweep internals**
- **Found during:** Task 2 (tune CLI implementation)
- **Issue:** `_enumerate_sweep_dimensions()` in `_main.py` imported `_BATCH_SIZES`, `_enumerate_backends`, `_precisions_for_backend` from `_sweep.py` — all private symbols, pyright reported 3 errors
- **Fix:** Added `count_sweep_dimensions()` as a public function to `tune/_sweep.py`; `_enumerate_sweep_dimensions()` in `_main.py` delegates to it
- **Files modified:** src/yowo/tune/_sweep.py, src/yowo/cli/_main.py
- **Verification:** `uv run pyright src/yowo/cli/_main.py` → 0 errors
- **Committed in:** a0e4707 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
- ruff auto-fixed UP017 (`datetime.timezone.utc` → `datetime.UTC`) and RUF002 (MULTIPLICATION SIGN `×` in docstring) on first commit attempt — corrected before final commit.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 03-05 (batch CLI) can proceed; tune profile auto-load is transparent to all existing code paths
- `yowo tune --model MODEL` is fully functional; requires real hardware to produce meaningful profiles
- All 10 tune CLI tests pass with mocked sweep for deterministic CI

---
*Phase: 03-adaptive-optimization-and-batch-processing*
*Completed: 2026-03-07*

## Self-Check: PASSED

- FOUND: src/yowo/engine.py
- FOUND: src/yowo/cli/_main.py
- FOUND: src/yowo/tune/_sweep.py
- FOUND: tests/unit/test_tune_cli.py
- FOUND: commit 1ef2cfb (feat: BaseEngine profile auto-load)
- FOUND: commit a0e4707 (feat: tune CLI subcommand)
