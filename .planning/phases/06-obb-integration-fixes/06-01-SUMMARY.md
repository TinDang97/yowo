---
phase: 06-obb-integration-fixes
plan: 01
subsystem: benchmark, engine, cli
tags: [obb, benchmark, dota, tune-profile, mAP, probiou]

# Dependency graph
requires:
  - phase: 04-obb-detection
    provides: OBBEngine, OBBDetection types, probiou_matrix, detect_obb() API
  - phase: 05-integration-bug-fixes
    provides: OBBConfig/InferenceConfig bridge, exports, engine task routing

provides:
  - task-aware tune profile key in engine._load_tune_profile (appends -obb/-cls suffix)
  - model_key derivation in tune_command replacing raw CLI string for profile storage
  - _MODEL_PATTERN accepting obb suffix in benchmark/__init__.py
  - DOTA v1 dataset loader (load_dota_dataset) in benchmark/_dota_evaluator.py
  - OBB mAP50-95 evaluator (evaluate_obb_map) using probiou rotated IoU
  - OBBEngine branch in run_single_backend for task=="obb"
  - 4 regression tests covering INT-A1 and INT-A2

affects: [benchmark, engine, cli-tune]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "task-aware key derivation: task_suffix = spec.task if spec.task not in ('detect',) else ''"
    - "module-level import for patchability: load_dota_dataset imported at yowo.benchmark level"
    - "lazy OBBEngine import inside run_single_backend for test patchability"

key-files:
  created:
    - src/yowo/benchmark/_dota_evaluator.py
    - tests/unit/test_obb_integration.py
  modified:
    - src/yowo/engine.py
    - src/yowo/cli/_main.py
    - src/yowo/benchmark/__init__.py
    - src/yowo/benchmark/_runner.py

key-decisions:
  - "module-level import of load_dota_dataset in benchmark/__init__.py (not lazy inline) — required for patch('yowo.benchmark.load_dota_dataset') to work in tests"
  - "lazy OBBEngine import inside elif task == 'obb' branch in _runner.py — allows patch('yowo.benchmark._runner.OBBEngine') in tests without importing at module load"
  - "model_key kept separate from raw model string in tune_command — model string retained for display (table title), model_key used for profile storage/lookup"
  - "np.trapezoid instead of np.trapz — trapz is deprecated in numpy; trapezoid is the current API"
  - "tasks committed as single atomic unit — pytest pre-commit hook requires all 4 tests to pass; INT-A2 tests are red without the benchmark fix staged"

patterns-established:
  - "task-aware key pattern: task_suffix derived from spec.task with detect exclusion, applied in both engine._load_tune_profile and cli tune_command"
  - "DOTA evaluator mirrors _evaluator.py load_coco_dataset structure: FileNotFoundError on missing dirs, sorted image_paths, per-image GT parsing"

requirements-completed: [TUNE-01, OBB-03, BENCH-01, OBB-01]

# Metrics
duration: 9min
completed: 2026-03-08
---

# Phase 06 Plan 01: OBB Integration Fixes Summary

**Closed INT-A1 (OBB tune profile silently loaded detection-calibrated key) and INT-A2 (benchmark ValueError on yolo11n-obb) with task-aware key derivation, DOTA v1 dataset loader, probiou OBB mAP evaluator, and 4 regression tests (1908 total)**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-08T08:09:47Z
- **Completed:** 2026-03-08T08:18:56Z
- **Tasks:** 3 (committed as 1 atomic unit due to pytest pre-commit hook)
- **Files modified:** 6 (2 created, 4 modified)

## Accomplishments

- Fixed INT-A1: `engine._load_tune_profile` and `tune_command` now derive the profile key with task suffix (`yolo11n-obb` vs `yolo11n`), preventing OBBEngine from silently loading a detection-calibrated profile
- Fixed INT-A2: `_MODEL_PATTERN` now accepts `obb` suffix; `run_benchmark` routes OBB models through `load_dota_dataset` instead of COCO path; `run_single_backend` has an `OBBEngine` branch with `detect_obb()` inference
- Added `benchmark/_dota_evaluator.py`: DOTA v1 val split loader + COCO-style mAP50-95 evaluator using `probiou_matrix` rotated IoU; 4 regression tests all green; full suite 1908 passed

## Task Commits

All tasks committed as one atomic unit (required by pytest pre-commit hook — INT-A2 tests fail if benchmark fix is not staged alongside test file):

1. **Tasks 1+2+3: TDD stubs + INT-A1 fix + INT-A2 fix** - `7b043b8` (feat)

## Files Created/Modified

- `tests/unit/test_obb_integration.py` - 4 regression tests for INT-A1 and INT-A2
- `src/yowo/engine.py` - `_load_tune_profile`: task_suffix appended to model_name key
- `src/yowo/cli/_main.py` - `tune_command`: model_key derived from spec, used for profile storage
- `src/yowo/benchmark/__init__.py` - `_MODEL_PATTERN` extended to `cls|obb`; OBB dispatch to `load_dota_dataset`; module-level import for patchability
- `src/yowo/benchmark/_dota_evaluator.py` - NEW: `load_dota_dataset`, `evaluate_obb_map`, `DOTA_CLASS_NAMES`
- `src/yowo/benchmark/_runner.py` - `run_single_backend`/`run_all_backends`: OBBEngine branch + gt_boxes/gt_classes params

## Decisions Made

- Module-level import of `load_dota_dataset` in `benchmark/__init__.py` rather than inline lazy import — required for `patch("yowo.benchmark.load_dota_dataset")` to intercept the call in tests
- Lazy `OBBEngine` import inside `elif task == "obb":` in `_runner.py` — preserves test patchability via `patch("yowo.benchmark._runner.OBBEngine")`
- Raw `model` string kept for display only in `tune_command`; `model_key` handles all profile storage and lookup
- Used `np.trapezoid` (not deprecated `np.trapz`) to satisfy pyright strict mode

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Module-level import required for test patchability**
- **Found during:** Task 3 (test_benchmark_obb_dispatches_dota_path)
- **Issue:** Plan specified lazy inline import `from yowo.benchmark._dota_evaluator import load_dota_dataset` inside `elif task == "obb":` block, but `patch("yowo.benchmark.load_dota_dataset")` fails with `AttributeError` when the name isn't in the module namespace at patch time
- **Fix:** Promoted `load_dota_dataset` to a module-level import in `benchmark/__init__.py`; removed inline import from the `elif` branch
- **Files modified:** `src/yowo/benchmark/__init__.py`
- **Verification:** `test_benchmark_obb_dispatches_dota_path` passes; `mock_load_dota.assert_called_once()` succeeds
- **Committed in:** `7b043b8`

**2. [Rule 1 - Bug] np.trapz deprecated; replaced with np.trapezoid**
- **Found during:** Task 3 (`_dota_evaluator.py` pyright check)
- **Issue:** `np.trapz` reported as deprecated by pyright (`reportDeprecated`); causes pyright error
- **Fix:** Changed to `np.trapezoid` (current numpy API)
- **Files modified:** `src/yowo/benchmark/_dota_evaluator.py`
- **Verification:** `uv run pyright src/yowo/benchmark/` → 0 errors
- **Committed in:** `7b043b8`

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs — import patchability and deprecated API)
**Impact on plan:** Both auto-fixes necessary for correctness and type safety. No scope creep.

## Issues Encountered

- Pre-commit pytest hook prevents partial task commits: INT-A2 tests fail if benchmark changes are stashed (hook stashes unstaged files). Resolved by staging all 6 files together in one commit rather than separate per-task commits.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- INT-A1 and INT-A2 gaps are fully closed with regression coverage
- `yowo tune --model yolo11n-obb` now saves profile under the correct key
- `yowo benchmark --model yolo11n-obb` no longer raises ValueError
- DOTA dataset path required for actual OBB benchmark evaluation (standard external dataset, not a code gap)

## Self-Check: PASSED

- test_obb_integration.py: FOUND
- _dota_evaluator.py: FOUND
- 06-01-SUMMARY.md: FOUND
- commit 7b043b8: FOUND

---
*Phase: 06-obb-integration-fixes*
*Completed: 2026-03-08*
