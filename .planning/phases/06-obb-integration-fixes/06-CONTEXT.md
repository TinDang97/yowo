# Phase 6: OBB Integration Fixes - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Close two functional integration gaps found by the final v2.3 milestone audit:

1. **INT-A1** (tune profile key collision): `_load_tune_profile` at `engine.py:161` constructs key
   as `f"{family}{size}"` (e.g. `"yolo11n"`) ignoring task. `tune_command` stores profiles under
   the raw CLI string (`"yolo11n-obb"`). These namespaces never meet — OBBEngine silently loads a
   detection-calibrated profile, and OBB-tuned profiles are never applied.

2. **INT-A2** (OBB models rejected by benchmark): `_MODEL_PATTERN` in `benchmark/__init__.py:37`
   does not accept `-obb` suffix. `yowo benchmark --model yolo11n-obb` raises `ValueError`. OBB
   mAP/FPS cannot be measured via benchmark module or CLI.

No new inference capability is added. Closes FLOW-A1 as a consequence of fixing INT-A1.

</domain>

<decisions>
## Implementation Decisions

### Profile key format (INT-A1)

- **Key format**: Derive from `ModelSpec` — `f"{family}{size}"` for detect, `f"{family}{size}-{task}"` for obb/cls
  - Detection: `"yolo11n"` (no suffix — backward-compatible, existing saved profiles continue to work)
  - OBB: `"yolo11n-obb"`
  - Classification: `"yolo11n-cls"`
- **Key derivation point**: `_load_tune_profile` at `engine.py:161` derives key from spec, not raw CLI string
- **tune_command alignment**: Switch `tune_command` to also use spec-derived key (not raw CLI `model` string)
  — ensures tune and load use identical key format
- **Backward compat**: Existing detection profiles stored as `"yolo11n"` continue to work. Existing OBB profiles
  stored under `"yolo11n-obb"` (raw CLI) were non-functional anyway (unreachable by loader) — no migration needed

### OBB benchmark evaluation path (INT-A2)

- **Pattern fix**: Extend `_MODEL_PATTERN` to accept `(?:-(cls|obb))?` suffix
- **OBB measurement**: Full mAP50-95 (COCO-style range) using rotated IoU (probiou), DOTA v1 format
  — NOT FPS-only, NOT mAP50 only
- **Dataset layout**: DOTA v1 official layout:
  ```
  data_path/
    images/val/   — image files
    labelTxt/val/ — per-image .txt files (x1 y1 x2 y2 x3 y3 x4 y4 class difficulty)
  ```
  15 DOTA v1 categories; matches ultralytics DOTA convention
- **New evaluator file**: `src/yowo/benchmark/_dota_evaluator.py` — `load_dota_dataset()` +
  `evaluate_obb_map()`. Keeps `_evaluator.py` under 700-line limit; OBB eval path clearly isolated
- **`run_benchmark` dispatch**: Branch on `task == "obb"` to load DOTA dataset and compute OBB mAP50-95

### Test coverage strategy

- **Test file**: `tests/unit/test_obb_integration.py` — single new file covering both gaps
  (matches Phase 5's `test_public_api.py` pattern)
- **Test functions**:
  - `test_tune_profile_key_obb()` — verify key derivation includes `-obb` suffix for OBB task
  - `test_tune_profile_key_detect_unchanged()` — verify detect key remains `"yolo11n"` (backward compat)
  - `test_benchmark_pattern_accepts_obb()` — verify `_MODEL_PATTERN` matches `"yolo11n-obb"`
  - `test_benchmark_obb_dispatches_dota_path()` — mock `load_dota_dataset` + `OBBEngine`, verify OBB path reached
- **Mocking strategy**: Patch `load_dota_dataset()` and `OBBEngine` — test flow logic without real DOTA files
  CI-safe, fast

### Claude's Discretion

- Exact mAP50-95 computation approach for rotated boxes (implementation detail for planner)
- Whether `_dota_evaluator.py` uses probiou directly or a separate rotated-IoU wrapper
- Error messages when DOTA dataset directory structure is missing
- How ultralytics comparison works for OBB (if not available, skip gracefully)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

- `_load_tune_profile()` at `engine.py:136` — module-level function; accepts `ModelSpec`; line 161 is the key construction to fix
- `probiou_matrix()` in `src/yowo/arch/_heads.py` — rotated IoU implementation already exists (used by OBBHead NMS); reuse for mAP computation
- `load_coco_dataset()` / `load_imagenet_dataset()` in `_evaluator.py` — dataset loading pattern to follow for `load_dota_dataset()`
- `run_all_backends()` in `_runner.py` — accepts `task` param; already branches `detect` vs `classify`; extend for `"obb"`
- `BenchmarkResult` in `_runner.py` — existing result type; check if `map50` / `map50_95` fields exist or need extension

### Established Patterns

- `unittest.mock.patch` used in Phase 5 regression tests (INT-P0 pattern) — reuse for OBB benchmark mock
- Module-level imports in `cli/_main.py` for test patchability (Phases 3–5) — follow for any new CLI OBB benchmark wiring
- Atomic profile write (Phase 3): write to `.tmp` then `os.replace` — already in `save_profile()`; no change needed
- `task = "classify" if task_suffix == "cls" else "detect"` in `_parse_model_name()` — extend `else` branch for `"obb"`

### Integration Points

- `src/yowo/engine.py:161` — 1-line fix to key construction in `_load_tune_profile()`
- `src/yowo/cli/_main.py` (~line 1250) — `save_profile(profile, ...)` call in `tune_command`; switch `profile.model` to spec-derived key
- `src/yowo/benchmark/__init__.py:37` — pattern fix; dispatch branch for `task == "obb"`
- `src/yowo/benchmark/_dota_evaluator.py` — new file (load_dota_dataset, evaluate_obb_map)
- `src/yowo/benchmark/_runner.py` — extend `run_one_backend()` to handle `task == "obb"` with `OBBEngine`
- `tests/unit/test_obb_integration.py` — new file

</code_context>

<specifics>
## Specific Ideas

- The key fix at `engine.py:161` is one line: `model_name = f"{spec.family.value}{spec.size.value}{'-' + spec.task if spec.task not in ('detect',) else ''}"`
- `tune_command` must derive model name from spec (parse the model arg into a spec first) rather than storing raw `model` string
- DOTA mAP50-95: iterate IoU thresholds 0.50, 0.55, ..., 0.95 with probiou; average — matches COCO-style computation
- OBB benchmark test: `test_benchmark_obb_dispatches_dota_path` verifies `load_dota_dataset` is called (not `load_coco_dataset`) when task is obb

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-obb-integration-fixes*
*Context gathered: 2026-03-08*
