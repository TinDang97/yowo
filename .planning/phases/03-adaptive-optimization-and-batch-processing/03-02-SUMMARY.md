---
phase: 03-adaptive-optimization-and-batch-processing
plan: "02"
subsystem: tuning
tags: [sweep, fps, calibration, oom-guard, tensorrt, onnx, pytorch, openvino, coreml, tdd]

# Dependency graph
requires:
  - phase: 03-01
    provides: TuneProfile, compute_fingerprint, HardwareProfile, hardware detection
  - phase: src/yowo/backends/_selector.py
    provides: check_backend_available (promoted to public API in this plan)
provides:
  - SweepResult dataclass: backend/batch_size/precision/fps/skipped/skip_reason fields
  - run_sweep(): backend x precision x batch_size FPS sweep with OOM guard, dry_run flag
  - _enumerate_backends(hw): available backend discovery via check_backend_available
  - _precisions_for_backend(backend, hw): GPU-aware precision list per backend
  - _is_oom(exc): OOM exception classifier for MemoryError/OutOfMemoryError/RuntimeError
  - _measure_config(): per-config FPS measurement with DetectionEngine lifecycle
  - check_backend_available() promoted to public API in backends._selector + backends.__init__
affects:
  - 03-03-batch-runner
  - 03-04-tune-cli

# Tech tracking
tech-stack:
  added: [time.monotonic (stdlib), numpy.zeros synthetic frame generation]
  patterns:
    - OOM guard via _is_oom() helper — catches MemoryError/torch.cuda.OutOfMemoryError/RuntimeError
    - Sorted sweep results: key=(-fps, batch_size) — FPS desc, batch_size asc tiebreak
    - skip propagation: oom_hit flag skips remaining batch sizes for same backend/precision
    - dry_run early return before any engine instantiation
    - Module-level torch import with try/except ImportError fallback for CPU-only environments

key-files:
  created:
    - src/yowo/tune/_sweep.py
  modified:
    - src/yowo/backends/_selector.py
    - src/yowo/backends/__init__.py
    - tests/unit/test_sweep.py

key-decisions:
  - "check_backend_available promoted from private _check_backend_available to public API in _selector.py and backends/__init__.py — sweep is a legitimate caller that needs this function; private visibility was overly restrictive"
  - "Broad except + _is_oom() dispatch instead of bare except or tuple unpacking — avoids B030 ruff error (invalid except-tuple unpacking with *) and remains extensible"
  - "_measure_config passes spec fields (family/size/num_classes) directly to InferenceConfig — DetectionEngine(config) takes InferenceConfig, not ModelSpec; plan interface comment was incorrect"
  - "DetectionEngine(config) is the correct call signature — spec fields are passed via InferenceConfig fields, not as separate positional argument"

patterns-established:
  - "_is_oom() helper pattern: centralizes OOM detection, supports MemoryError + torch.cuda.OutOfMemoryError + RuntimeError 'out of memory' string for broad compatibility"
  - "Sweep skip propagation: oom_hit boolean per (backend, precision) combo — once set, all remaining batch_sizes appended as skipped SweepResults (not silently dropped)"

requirements-completed: [TUNE-02, TUNE-04]

# Metrics
duration: 11min
completed: "2026-03-07"
---

# Phase 3 Plan 02: Calibration Sweep Loop Summary

**FPS calibration sweep with OOM guard: run_sweep() iterates backend x precision x batch_size, measures with synthetic frames, returns results sorted by FPS descending**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-07T15:27:34Z
- **Completed:** 2026-03-07T15:38:32Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- `run_sweep()` iterates all available backends x precisions x [1,2,4,8,16,32] batch sizes with 50 warmup + 200 measure frames
- OOM guard: `_is_oom()` catches MemoryError/torch.cuda.OutOfMemoryError/RuntimeError-oom; calls `torch.cuda.empty_cache()`, propagates skipped SweepResult for all remaining batch sizes in that combo
- `check_backend_available` promoted from private to public API — exposes what the sweep legitimately needs without pyright violations
- 17 tests passing: sorted results, tiebreak by batch_size, OOM propagation, dry_run, cuda.empty_cache call verification

## Task Commits

1. **Task 1 + Task 2: SweepResult, helpers, run_sweep, full test coverage** - `32a454d` (feat)

## Files Created/Modified

- `src/yowo/tune/_sweep.py` — SweepResult dataclass, _enumerate_backends, _precisions_for_backend, _is_oom, _measure_config, run_sweep
- `src/yowo/backends/_selector.py` — check_backend_available renamed from _check_backend_available; added to __all__
- `src/yowo/backends/__init__.py` — exports check_backend_available in __all__
- `tests/unit/test_sweep.py` — 17 unit tests replacing 3 scaffold stubs; TestSweepResult, TestEnumerateBackends, TestPrecisionsForBackend, TestRunSweep classes

## Decisions Made

- `check_backend_available` promoted from private to public: the sweep module is a legitimate cross-module caller; keeping it private unnecessarily blocked pyright.
- `except Exception as exc` + `_is_oom()` dispatch: avoids B030 ruff violation from `*(tuple,) if cond else ()` unpacking in except clauses; also more readable.
- `_measure_config` uses `DetectionEngine(config)` passing spec fields via `InferenceConfig(model_family=spec.family, ...)` — the plan's documented interface `DetectionEngine(spec, config)` was incorrect; actual API has only `config` as first arg.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] DetectionEngine(spec, config) → DetectionEngine(config) API mismatch**
- **Found during:** Task 2 (GREEN phase, pyright error)
- **Issue:** Plan interface comment showed `DetectionEngine(spec, config)` but actual engine takes `config: InferenceConfig | None` as sole positional arg. `spec` fields are set through InferenceConfig fields.
- **Fix:** Changed `_measure_config` to build `InferenceConfig(model_family=spec.family, model_size=spec.size, num_classes=spec.num_classes, backend=..., precision=..., batch_size=...)` and call `DetectionEngine(config)`
- **Files modified:** src/yowo/tune/_sweep.py
- **Verification:** pyright passes, 17 tests pass
- **Committed in:** 32a454d

**2. [Rule 2 - Missing Critical] check_backend_available promoted to public API**
- **Found during:** Task 1 (pyright reportPrivateUsage error)
- **Issue:** `_check_backend_available` was private in `_selector.py`; importing it from another module raised pyright errors
- **Fix:** Renamed to `check_backend_available` in `_selector.py`, added to `__all__`, re-exported from `backends/__init__.py`
- **Files modified:** src/yowo/backends/_selector.py, src/yowo/backends/__init__.py
- **Verification:** pyright passes with 0 errors, existing backend tests still pass
- **Committed in:** 32a454d

**3. [Rule 1 - Bug] Exception handling pattern for OOM**
- **Found during:** Task 2 (ruff B030 error)
- **Issue:** `except (MemoryError, *(torch.cuda.OutOfMemoryError,) if torch else ())` raises B030 (invalid tuple unpacking in except clause)
- **Fix:** Used `except Exception as exc` + `_is_oom(exc)` dispatch helper; catches all exceptions but only treats OOM as skip-and-propagate; other errors get logged+skipped individually
- **Files modified:** src/yowo/tune/_sweep.py
- **Verification:** ruff passes, 17 tests pass
- **Committed in:** 32a454d

---

**Total deviations:** 3 auto-fixed (1 API mismatch bug, 1 missing public API, 1 exception pattern bug)
**Impact on plan:** All auto-fixes required for correctness and pyright compliance. No scope creep.

## Issues Encountered

- Pre-commit hook staged-vs-unstaged conflict: hook runs pytest on staged files only; staging _sweep.py without test_sweep.py caused test failures. Resolved by staging all modified files together.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `run_sweep()` importable from `yowo.tune._sweep`, fully tested with 17 passing tests
- Plan 03 (batch runner) can use run_sweep results to configure batch processing
- Plan 04 (tune CLI) can call run_sweep and pass results to save_profile
- check_backend_available is now public API for any future callers

## Self-Check: PASSED

All files verified below.
