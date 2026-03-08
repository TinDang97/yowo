---
phase: 05-integration-bug-fixes
plan: 01
subsystem: export, engine, api
tags: [obb, kv-cache, tune-profile, public-api, int-p0, int-p1, int-p2]

# Dependency graph
requires:
  - phase: 04-obb-detection
    provides: OBBEngine, OBBModel, OBBConfig, OBBHead, load_obb_weights
provides:
  - kv-cache export guard excludes obb task (tuple membership check)
  - OBBEngine calls _load_tune_profile at construction when backend_instance is None
  - 5 missing types re-exported at yowo top-level namespace (OBBBox, OBBDetection, WarmupValidationError, HealthReport, StreamConfig)
  - 3 regression/smoke test files closing INT-P0, INT-P1, INT-P2 gaps
affects: [any future export, obb, public-api work]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level imports of private functions in obb_engine.py for test patchability (yowo.obb_engine._load_tune_profile)"
    - "cast(Any, cfg) pattern to bridge OBBConfig/InferenceConfig type mismatch at _load_tune_profile call site"
    - "# type: ignore[reportPrivateUsage] on cross-module private function import"

key-files:
  created:
    - tests/unit/test_public_api.py
  modified:
    - src/yowo/export/_exporter.py
    - src/yowo/obb_engine.py
    - src/yowo/__init__.py
    - tests/unit/test_obb_export.py
    - tests/unit/test_obb_engine.py

key-decisions:
  - "INT-P0: tuple membership guard spec.task not in ('classify', 'obb') — cleaner than chained != conditions"
  - "INT-P1: cast(Any, cfg) for _load_tune_profile argument avoids Pyright error since cfg is OBBConfig not InferenceConfig"
  - "INT-P1: # type: ignore[reportPrivateUsage] on obb_engine import of _load_tune_profile — deliberate cross-module private API for test patchability"
  - "INT-P2: HealthReport imported from yowo.engine (defined there to avoid circular import with HealthStatus)"

patterns-established:
  - "OBB kv_cache detect test: FakeYOLOModel subclass with identity fuse/eval to satisfy isinstance check through the fuse/eval reassignment chain"
  - "Module-level imports for test patchability: consistent pattern across DetectionEngine, OBBEngine, CLI"

requirements-completed: [OBB-01, OBB-02, OBB-06, CORR-07, CORR-08, RELY-05, STRM-01, TUNE-01]

# Metrics
duration: 11min
completed: 2026-03-08
---

# Phase 05 Plan 01: Integration Bug Fixes Summary

**kv-cache guard extended to exclude obb task, OBBEngine now applies saved tune profile at construction, and 5 missing types (OBBBox, OBBDetection, WarmupValidationError, HealthReport, StreamConfig) promoted to top-level yowo namespace**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-03-08T04:24:18Z
- **Completed:** 2026-03-08T04:35:48Z
- **Tasks:** 3 (2 TDD + 1 quality gate)
- **Files modified:** 5 (3 source, 2 test) + 1 test created

## Accomplishments
- INT-P0 closed: `export_model(obb_spec, ExportFormat.ONNX, ..., kv_cache=True)` no longer raises AssertionError — guard now uses tuple membership
- INT-P1 closed: OBBEngine construction calls `_load_tune_profile` when `backend_instance is None`, mirroring DetectionEngine exactly
- INT-P2 closed: All 5 types importable from `yowo` top-level and present in `__all__`; 1904 tests pass (was 1615 before Phase 5)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix INT-P0 kv-cache guard + regression tests** - `49d1f4e` (fix)
2. **Task 2: Fix INT-P1 OBBEngine tune profile + INT-P2 public API exports** - `63dc77d` (fix)
3. **Task 3: Full quality gate** — covered by Task 2 commit (pre-commit hook ran full suite)

## Files Created/Modified
- `src/yowo/export/_exporter.py` - Line 139: `!= "classify"` → `not in ("classify", "obb")`
- `src/yowo/obb_engine.py` - Added `_load_tune_profile`, `HardwareProfile`, `get_hardware_profile` imports + tune profile block in `__init__`
- `src/yowo/__init__.py` - Added `HealthReport`, `OBBBox`, `OBBDetection`, `StreamConfig`, `WarmupValidationError` to imports and `__all__`
- `tests/unit/test_obb_export.py` - Added 3 kv_cache guard regression tests
- `tests/unit/test_obb_engine.py` - Added 2 INT-P1 tune profile tests
- `tests/unit/test_public_api.py` - Created: INT-P2 smoke test for all 5 types

## Decisions Made
- INT-P0: tuple membership guard `spec.task not in ("classify", "obb")` — cleaner than chained `!=` conditions and future-proofs for additional non-kv tasks
- INT-P1: `cast(Any, cfg)` bridges `OBBConfig`/`InferenceConfig` type mismatch at `_load_tune_profile` call site without changing shared function signature
- INT-P1: `# type: ignore[reportPrivateUsage]` on obb_engine import of `_load_tune_profile` — deliberate cross-module private API for test patchability, consistent with Phase 3 CLI patterns
- INT-P2: `HealthReport` imported from `yowo.engine` (defined there to avoid circular import with `HealthStatus`)
- kv_cache detect test: `_FakeYOLOModel` subclass with identity `fuse()`/`eval()` methods so the `model = model.fuse().eval()` reassignment still produces a `YOLOModel` instance that passes `isinstance`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test scaffolding fix — classify task stem mismatch**
- **Found during:** Task 1 (test_kv_cache_guard_classify_skips_kv_export)
- **Issue:** Test created `yolo11n-cls.onnx` but classify task stem is `yolo11n` (no suffix), causing FileNotFoundError in stat()
- **Fix:** Changed fake_onnx filename to `yolo11n.onnx` to match actual export path
- **Files modified:** tests/unit/test_obb_export.py
- **Verification:** Test passes
- **Committed in:** 49d1f4e (Task 1 commit)

**2. [Rule 1 - Bug] Test scaffolding fix — detect positive case mock chain**
- **Found during:** Task 1 (test_kv_cache_guard_detect_calls_kv_export)
- **Issue:** `build_model` returns `_FakeYOLOModel`, but `model = model.fuse().eval()` reassigns to MagicMock result which fails `isinstance(model, YOLOModel)`; also `YOLOKVWrapper` needed to be patched at source module
- **Fix:** Made `_FakeYOLOModel` subclass with identity `fuse()`/`eval()` chain; patched `yowo.export._kv_wrapper.YOLOKVWrapper` to prevent real constructor
- **Files modified:** tests/unit/test_obb_export.py
- **Verification:** Test passes with `mock_kv_export.assert_called_once()`
- **Committed in:** 49d1f4e (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - test scaffolding bugs)
**Impact on plan:** Both auto-fixes were in test code only, isolating test construction from inference behavior. No production code scope creep.

## Issues Encountered
- Pyright `reportPrivateUsage` on `_load_tune_profile` import from obb_engine — resolved with `# type: ignore[reportPrivateUsage]` consistent with the deliberate cross-module private API pattern established in Phase 3

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All three integration gaps (INT-P0, INT-P1, INT-P2) are closed
- 1904 unit tests pass with zero regressions
- Ready for remaining Phase 5 plans (if any) or v2.3.0 release

---
*Phase: 05-integration-bug-fixes*
*Completed: 2026-03-08*
