---
phase: 04-obb-detection
plan: "03"
subsystem: cli
tags: [obb, click, export, parse_model_name, detect-obb]

requires:
  - phase: 04-01
    provides: OBBHead arch, OBBModel, build_obb_model, OBB registry
  - phase: 04-02
    provides: OBBEngine, OBBConfig, load_obb_weights, postprocess_obb

provides:
  - parse_model_name extended with -obb suffix (YOLO11-only guard)
  - detect-obb Click command registered in CLI with full option parity to detect
  - OBBEngine + OBBConfig imported at module level in cli/_main.py
  - export_model OBB branch: build_obb_model + load_obb_weights + -obb stem suffix
  - 13 new unit tests covering CLI and export

affects:
  - future OBB benchmark plans
  - user-facing documentation

tech-stack:
  added: []
  patterns:
    - OBB CLI command mirrors detect command structure exactly
    - Module-level engine imports for test mockability (OBBEngine, OBBConfig at top of _main.py)
    - task-suffix model_stem pattern in exporter (prevents ONNX filename collision between detect and obb)
    - elif chain for task branching in export_model and parse_model_name

key-files:
  created:
    - tests/unit/test_obb_cli.py
    - tests/unit/test_obb_export.py
  modified:
    - src/yowo/_convenience.py
    - src/yowo/cli/_main.py
    - src/yowo/export/_exporter.py

key-decisions:
  - "OBB CLI command imports OBBEngine/OBBConfig at module level for test patchability (consistent with Phase 03-05 pattern)"
  - "model_stem uses task_suffix variable so detect (no suffix) and obb (-obb) produce distinct ONNX filenames"
  - "yolo26*-obb raises ConfigError with clear message at parse_model_name time, not at engine load"

patterns-established:
  - "Task-suffix pattern in exporter: task_suffix = '-obb' if spec.task == 'obb' else '' applied to model_stem"
  - "OBB-only guard: check name.startswith('yolo11') before accepting -obb suffix in parse_model_name"

requirements-completed:
  - OBB-05
  - OBB-06

duration: 7min
completed: 2026-03-08
---

# Phase 4 Plan 03: OBB CLI and Export Pipeline Summary

**OBB detect-obb CLI command + export_model OBB branch wired in; parse_model_name extended with -obb suffix enforcing YOLO11-only guard; 1898 tests pass**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-03-08T02:07:55Z
- **Completed:** 2026-03-08T02:14:49Z
- **Tasks:** 1 (+ 1 checkpoint awaiting human verification)
- **Files modified:** 5

## Accomplishments

- Extended `parse_model_name` with `-obb` suffix: `yolo11{n|s|m|l|x}-obb` returns `ModelSpec(task="obb")`; `yolo26*-obb` raises `ConfigError` at parse time
- Added `detect-obb` Click command to `cli/_main.py` with full option parity to `detect` (model, weights, num-classes, backend, device, precision, confidence, iou, batch, output, json, no-metrics); `OBBEngine` and `OBBConfig` imported at module level for test patchability
- Added `elif spec.task == "obb"` branch in `export_model`: calls `build_obb_model` + `load_obb_weights`; `model_stem` uses `-obb` suffix to prevent ONNX filename collision with detection exports
- 13 new unit tests (CLI + export) all pass; full suite 1898 passed, ruff clean, pyright 0 errors

## Task Commits

Each task was committed atomically:

1. **Task 1: parse_model_name -obb extension + detect-obb CLI + export OBB branch** - `ac77323` (feat)

**Plan metadata:** (pending — will be added in final commit)

## Files Created/Modified

- `src/yowo/_convenience.py` - Added elif -obb branch with YOLO11 guard; updated error message and docstring
- `src/yowo/cli/_main.py` - Added module-level OBBEngine/OBBConfig imports; added detect_obb_command Click command
- `src/yowo/export/_exporter.py` - Added elif spec.task == "obb" branch; task_suffix for model_stem
- `tests/unit/test_obb_cli.py` - 9 tests: parse_model_name -obb, detect-obb CLI via CliRunner (exit 0, JSON, help, module-level imports)
- `tests/unit/test_obb_export.py` - 4 tests: obb branch calls build_obb_model/load_obb_weights, not build_model/load_weights; stem contains "-obb"

## Decisions Made

- OBBEngine/OBBConfig imported at module level in `cli/_main.py` for test patchability (consistent with Phase 03-05 batch CLI pattern)
- `model_stem` uses `task_suffix` variable: `"-obb"` for OBB, `""` otherwise — prevents detect ONNX and OBB ONNX from sharing the same filename
- `yolo26*-obb` guard fires at `parse_model_name` time with a clear error, not silently at registry lookup

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- pre-commit hook reformatted two files with ruff-format; re-staged and re-committed on second attempt (standard workflow)

## Next Phase Readiness

- Complete OBB pipeline (arch + NMS + engine + CLI + export) is implemented and tested
- Human verification checkpoint required: `yowo --help` must show `detect-obb`, OBBEngine must construct, registry must return nc=15
- After checkpoint approval, Phase 4 is fully complete and ready for v2.4.0 release

---
*Phase: 04-obb-detection*
*Completed: 2026-03-08*
