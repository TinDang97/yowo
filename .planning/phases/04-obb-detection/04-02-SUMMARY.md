---
phase: 04-obb-detection
plan: 02
subsystem: inference-engine
tags: [obb, oriented-bounding-box, pytorch, yolo11, dota, engine, config]

# Dependency graph
requires:
  - phase: 04-01
    provides: OBBHead, OBBModel, build_obb_model, postprocess_obb, get_obb registry

provides:
  - OBBEngine(BaseEngine) in obb_engine.py with detect_obb(), stream_obb(), adetect_obb()
  - OBBConfig dataclass in config.py with confidence_threshold, iou_threshold, nc defaults
  - load_obb_weights() in arch/_weights.py reusing _LAYER_MAP for head.cv4.* mapping
  - "obb" task branch in backends/_pytorch.py and _resolve_model_meta in engine.py
  - OBBEngine and OBBConfig exported from yowo.__init__

affects:
  - 04-03 (CLI + export surface will import OBBEngine directly)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - OBBEngine mirrors ClassificationEngine pattern exactly (config dataclass, ModelSpec task, _result_event_name, _process_batch)
    - load_obb_weights reuses existing _LAYER_MAP — no separate OBB layer map needed
    - "obb" task branch in PyTorch backend mirrors "classify" branch structure

key-files:
  created:
    - src/yowo/obb_engine.py
    - tests/unit/test_obb_engine.py
    - tests/unit/test_obb_weights.py
  modified:
    - src/yowo/config.py
    - src/yowo/arch/_weights.py
    - src/yowo/backends/_pytorch.py
    - src/yowo/engine.py
    - src/yowo/__init__.py

key-decisions:
  - "load_obb_weights reuses _LAYER_MAP (not a separate OBB map) — model.23.* -> head.* prefix covers cv4 angle branches automatically"
  - "OBBEngine does not use feature cache or kv_cache (same as ClassificationEngine — DOTA inference is offline, no streaming cache benefit)"
  - "_validate_output_values checks shape[1] == 4+nc+1 and class score range [0,1] to detect double-sigmoid or wrong nc"
  - "_process_batch reconstructs frozen OBBDetection dataclasses to attach elapsed_ms from engine timing"

patterns-established:
  - "Task branch pattern: if task=='classify' / elif task=='obb' / else (detection) in _pytorch.py and _resolve_model_meta"
  - "Engine mirrors ClassificationEngine: config dataclass -> ModelSpec(task=...) -> super().__init__() -> _result_event_name property"

requirements-completed: [OBB-03, OBB-04]

# Metrics
duration: 8min
completed: 2026-03-08
---

# Phase 4 Plan 02: OBBEngine Summary

**OBBEngine(BaseEngine) with detect_obb()/stream_obb(), OBBConfig dataclass, load_obb_weights() via shared _LAYER_MAP, and obb task wired into PyTorch backend and model meta resolution**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-08T01:57:01Z
- **Completed:** 2026-03-08T02:05:01Z
- **Tasks:** 1 (TDD: red -> green -> quality gate)
- **Files modified:** 8 (3 new, 5 modified)

## Accomplishments

- OBBEngine subclasses BaseEngine following ClassificationEngine pattern exactly — ModelSpec(task="obb"), _result_event_name="obb_detection", detect_obb()/stream_obb()/adetect_obb() public API
- OBBConfig dataclass in config.py with confidence_threshold=0.25, iou_threshold=0.45, nc=None (registry default 15), validates num_classes >= 1, no top_k field
- load_obb_weights() in arch/_weights.py reuses existing _LAYER_MAP — the "model.23." -> "head." prefix maps cv4 angle branches automatically, no separate OBB layer map required
- "obb" task branch wired into PyTorch backend (build_obb_model + load_obb_weights + fuse/eval) and _resolve_model_meta (uses _registry_get_obb)
- 25 new unit tests: TDD RED then GREEN; full suite 1885 passed (270 new vs baseline)

## Task Commits

1. **Task 1: OBBConfig + load_obb_weights + OBBEngine** - `a001dc0` (feat)

## Files Created/Modified

- `src/yowo/obb_engine.py` - OBBEngine(BaseEngine): detect_obb(), stream_obb(), adetect_obb(), _process_batch(), _validate_output_values()
- `src/yowo/config.py` - Added OBBConfig dataclass after ClassificationConfig
- `src/yowo/arch/_weights.py` - Added load_obb_weights() reusing _LAYER_MAP and _remap_key
- `src/yowo/backends/_pytorch.py` - Added elif task=="obb" branch calling build_obb_model + load_obb_weights
- `src/yowo/engine.py` - Fixed _resolve_model_meta to use _registry_get_obb for task="obb"
- `src/yowo/__init__.py` - Exported OBBEngine and OBBConfig
- `tests/unit/test_obb_engine.py` - 18 tests: OBBConfig validation, OBBEngine init/lifecycle/events/_process_batch
- `tests/unit/test_obb_weights.py` - 7 tests: _remap_key for cv4 keys, FileNotFoundError

## Decisions Made

- load_obb_weights reuses _LAYER_MAP (not a new OBB-specific map) because "model.23." -> "head." prefix already covers all OBBHead sub-modules including cv4 angle branches. No separate mapping table needed.
- OBBEngine skips feature cache and kv_cache (same as ClassificationEngine) — DOTA-domain OBB inference is primarily offline/batch, no streaming cache benefit.
- _validate_output_values checks `shape[1] == 4+nc+1` to detect nc mismatch at warmup time, and class score range [0,1] to detect double-sigmoid.
- _process_batch reconstructs each OBBDetection (frozen dataclass) to attach elapsed_ms from engine timing, since postprocess_obb returns OBBDetection(inference_time_ms=0.0).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Fixed _resolve_model_meta to handle task="obb"**
- **Found during:** Task 1 (OBBEngine implementation)
- **Issue:** The existing _resolve_model_meta in engine.py used a ternary: classify -> _registry_get_cls else -> _registry_get (detection). OBBEngine with task="obb" would fall through to the detection registry, causing ModelNotFoundError for YOLO11 OBB models.
- **Fix:** Refactored to if/elif/else: task=="classify" -> get_cls, task=="obb" -> get_obb, else -> get (detection).
- **Files modified:** src/yowo/engine.py
- **Verification:** OBBEngine instantiation test passes; OBBDetection returned correctly.
- **Committed in:** a001dc0 (Task 1 commit)

**2. [Rule 2 - Missing Critical] Added "obb" branch to PyTorch backend**
- **Found during:** Task 1 (inspection of _pytorch.py)
- **Issue:** PyTorch backend had classify/else (detection) branches but no "obb" branch. OBBEngine with task="obb" would have tried to call build_model (detection) + load_weights, producing wrong model architecture.
- **Fix:** Added elif self._spec.task == "obb" branch mirroring the classify branch: build_obb_model + load_obb_weights + fuse/eval/channels-last.
- **Files modified:** src/yowo/backends/_pytorch.py
- **Verification:** OBBEngine lifecycle test passes with mocked backend; type-checked by pyright.
- **Committed in:** a001dc0 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 2 — missing critical wiring for obb task)
**Impact on plan:** Both fixes essential for correctness. Without them OBBEngine would silently use wrong model architecture or fail registry lookup.

## Issues Encountered

- Initial test `test_cv4_key_maps_to_head` used wrong approach (patching `yowo.arch._weights.torch` which is a lazy import inside `_extract_state_dict`). Refactored to test _remap_key directly and verify OBBModel has cv4 params from state_dict() — simpler and more direct.

## Next Phase Readiness

- OBBEngine ready for CLI integration (04-03)
- detect_obb() / stream_obb() / adetect_obb() all available
- OBBConfig and OBBEngine in public yowo.__init__ exports

---
*Phase: 04-obb-detection*
*Completed: 2026-03-08*
