# Phase 5: Integration Bug Fixes - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning
**Source:** Claude's Discretion (all decisions delegated by user — see transparent rationale below)

<domain>
## Phase Boundary

Close three integration gaps (INT-P0, INT-P1, INT-P2) found by the v2.3 milestone audit. No new
inference capability is added. The three fixes are:

1. **INT-P0** (P0 runtime crash): Guard in `export/_exporter.py:139` passes OBB task through to
   `_wrap_model_for_kv_cache()`, causing `AssertionError` when `--kv-cache` is used with an OBB model.
2. **INT-P1** (P1 missing integration): `OBBEngine.__init__` does not call `_load_tune_profile()`,
   so saved tune profiles are silently ignored for OBB models even though `yowo tune` writes them.
3. **INT-P2** (P2 incomplete public API): Five types — `OBBBox`, `OBBDetection`,
   `WarmupValidationError`, `HealthReport`, `StreamConfig` — are absent from `yowo.__init__`,
   making them unreachable without submodule imports.

</domain>

<decisions>
## Implementation Decisions

### INT-P0: kv-cache guard fix (`export/_exporter.py:139`)
- Change `spec.task != "classify"` → `spec.task not in ("classify", "obb")`
- This is the minimal, correct fix — OBB models wrap `OBBModel` which is not a `YOLOModel` subclass
- No other changes in the export path needed

### INT-P0: Regression test design
- **Mock-based unit test** — no actual ONNX file produced, no filesystem side-effects
- Test via `unittest.mock.patch` on `_wrap_model_for_kv_cache` (or the assert call site) to verify
  it is NOT called when task is `"obb"`
- Test the three cases: `task="obb"` skips kv_cache wrap, `task="classify"` skips, `task="detect"` calls it
- One regression test per gap clause — fast, isolated, CI-safe

### INT-P1: OBBEngine tune profile behavior
- `_load_tune_profile()` called in `OBBEngine.__init__` at the same point DetectionEngine calls it
  (`engine.py:975–980`): after config is resolved, before `_backend` is selected
- **Identical behavior to DetectionEngine**: silent `DEBUG` log if no profile found for the device+model
  combo — no user-visible warning
- Rationale: consistency — `yowo tune --model yolo11n-obb` stores the profile; `OBBEngine` silently
  applies it on next load. Users should see identical behavior regardless of task type.
- The existing `_load_tune_profile()` function (module-level in `engine.py`) is reused as-is — no
  modification needed

### INT-P2: Public API exports (`src/yowo/__init__.py`)
- Add exactly the 5 types named in the audit: `OBBBox`, `OBBDetection`, `WarmupValidationError`,
  `HealthReport`, `StreamConfig`
- Sources:
  - `OBBBox`, `OBBDetection`, `StreamConfig` → from `yowo.types`
  - `WarmupValidationError` → from `yowo.errors`
  - `HealthReport` → from `yowo.engine`
- Scope is strictly the 5 named types — any additional gaps found during implementation become tech debt
  notes, not scope additions
- Add corresponding entries to `__all__` in `__init__.py`

### Test coverage for INT-P2
- Add a smoke test: `from yowo import OBBBox, OBBDetection, WarmupValidationError, HealthReport, StreamConfig`
  — verifies the import works without submodule qualification
- One test function, `test_public_api_exports`, covers all 5 types

### Claude's Discretion
- Exact test file placement (`tests/unit/test_public_api.py` or alongside export tests — planner decides)
- Whether to add `OBBConfig` to the same export batch pass (it's already exported; no-op)
- Order of exports in `__init__.py` (keep alphabetical within existing sections)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_load_tune_profile()` at `engine.py:136` — module-level function; already handles no-profile case
  gracefully (returns config unchanged); OBBEngine imports it from `yowo.engine`
- `OBBBox`, `OBBDetection`, `StreamConfig` defined in `src/yowo/types.py` (lines 326, 353, 141)
  and already in `types.__all__` (lines 592–597)
- `WarmupValidationError` defined in `src/yowo/errors.py:94`, in `errors.__all__:240`
- `HealthReport` defined in `src/yowo/engine.py:96`, in `engine.__all__:1064`
- `OBBEngine` already imported in `__init__.py:62` — same import block to extend for OBBBox/OBBDetection

### Established Patterns
- `unittest.mock.patch` used in existing export tests (Phase 1/4 pattern) — reuse for INT-P0 regression
- Module-level imports in `cli/_main.py` for test patchability (Phases 3–4); same pattern for
  `_load_tune_profile` if OBBEngine needs to import it differently
- `__init__.py` export sections are grouped by origin module — add new exports to the matching section

### Integration Points
- `src/yowo/export/_exporter.py:139` — 1-line guard change
- `src/yowo/obb_engine.py:47` — add `_load_tune_profile()` call in `__init__`
- `src/yowo/__init__.py` — add 5 import lines + `__all__` entries
- `tests/unit/` — add regression test file (or extend existing export test file)

</code_context>

<specifics>
## Specific Ideas

- The kv_cache guard fix is a 1-line change but must have a regression test to prevent re-introduction
- "Transparent step-by-step" — planner should document each fix as a discrete, independently-testable task
- All 3 fixes are independent and can be implemented in any order (no task dependencies between them)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-integration-bug-fixes*
*Context gathered: 2026-03-08*
