# Phase 6: OBB Integration Fixes - Research

**Researched:** 2026-03-08
**Domain:** OBB task suffix propagation and DOTA benchmark evaluation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Profile key format (INT-A1)**
- Key format: Derive from `ModelSpec` — `f"{family}{size}"` for detect, `f"{family}{size}-{task}"` for obb/cls
  - Detection: `"yolo11n"` (no suffix — backward-compatible, existing saved profiles continue to work)
  - OBB: `"yolo11n-obb"`
  - Classification: `"yolo11n-cls"`
- Key derivation point: `_load_tune_profile` at `engine.py:161` derives key from spec, not raw CLI string
- `tune_command` alignment: Switch `tune_command` to also use spec-derived key (not raw CLI `model` string)
- Backward compat: Existing detection profiles stored as `"yolo11n"` continue to work. Existing OBB profiles stored under `"yolo11n-obb"` (raw CLI) were non-functional anyway — no migration needed

**OBB benchmark evaluation path (INT-A2)**
- Pattern fix: Extend `_MODEL_PATTERN` to accept `(?:-(cls|obb))?` suffix
- OBB measurement: Full mAP50-95 (COCO-style range) using rotated IoU (probiou), DOTA v1 format
- Dataset layout: DOTA v1 official layout:
  ```
  data_path/
    images/val/   — image files
    labelTxt/val/ — per-image .txt files (x1 y1 x2 y2 x3 y3 x4 y4 class difficulty)
  ```
  15 DOTA v1 categories; matches ultralytics DOTA convention
- New evaluator file: `src/yowo/benchmark/_dota_evaluator.py` — `load_dota_dataset()` + `evaluate_obb_map()`. Keeps `_evaluator.py` under 700-line limit; OBB eval path clearly isolated
- `run_benchmark` dispatch: Branch on `task == "obb"` to load DOTA dataset and compute OBB mAP50-95

**Test coverage strategy**
- Test file: `tests/unit/test_obb_integration.py` — single new file covering both gaps
- Test functions:
  - `test_tune_profile_key_obb()` — verify key derivation includes `-obb` suffix for OBB task
  - `test_tune_profile_key_detect_unchanged()` — verify detect key remains `"yolo11n"` (backward compat)
  - `test_benchmark_pattern_accepts_obb()` — verify `_MODEL_PATTERN` matches `"yolo11n-obb"`
  - `test_benchmark_obb_dispatches_dota_path()` — mock `load_dota_dataset` + `OBBEngine`, verify OBB path reached
- Mocking strategy: Patch `load_dota_dataset()` and `OBBEngine` — test flow logic without real DOTA files; CI-safe, fast

### Claude's Discretion
- Exact mAP50-95 computation approach for rotated boxes (implementation detail for planner)
- Whether `_dota_evaluator.py` uses probiou directly or a separate rotated-IoU wrapper
- Error messages when DOTA dataset directory structure is missing
- How ultralytics comparison works for OBB (if not available, skip gracefully)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TUNE-01 | User can run `yowo tune --model MODEL` to auto-detect optimal backend, batch size, and precision for current hardware | Fix profile key construction in `_load_tune_profile` and `tune_command` so OBB profiles are stored/retrieved under spec-derived key |
| OBB-03 | OBB model variants match ultralytics OBB architecture for yolo11 family | Verified: `probiou_matrix` in `postprocess/_obb_nms.py` matches ultralytics exactly; architecture correct; fix is behavioral integration gap only |
| BENCH-01 | User can run `yowo benchmark --model MODEL` to get mAP + FPS + model size per export format | Fix `_MODEL_PATTERN` to accept `-obb` suffix; add DOTA evaluator; dispatch OBB path in `run_benchmark` |
| OBB-01 | OBB detection head produces oriented bounding boxes with rotation angle | Verified working; this phase closes the integration path — benchmark can now measure OBB output correctness |
</phase_requirements>

---

## Summary

Phase 6 closes two silent integration gaps discovered during the v2.3 milestone audit. Both gaps are narrow, precisely located, and require no new inference capability — they are behavioral fixes to existing plumbing.

**INT-A1 (tune profile key collision):** `_load_tune_profile` at `engine.py:161` builds the profile lookup key as `f"{spec.family.value}{spec.size.value}"` — ignoring `spec.task`. This means `OBBEngine` silently loads a detection-calibrated profile (e.g. `"yolo11n"`) instead of the OBB-tuned one (`"yolo11n-obb"`). Simultaneously, `tune_command` in `cli/_main.py` stores the profile using the raw CLI `model` string argument, which happens to be `"yolo11n-obb"` — so the two namespaces never meet. The fix is a 1-line change in `_load_tune_profile` plus aligning `tune_command` to derive the key from the parsed `ModelSpec` rather than the raw CLI string.

**INT-A2 (OBB models rejected by benchmark):** `_MODEL_PATTERN` in `benchmark/__init__.py` is `r"^(yolo(?:11|26))([nsmxl])(?:-(cls))?$"` — it only accepts `-cls` suffix. Passing `"yolo11n-obb"` raises `ValueError` before any evaluation occurs. The fix extends the pattern to `(?:-(cls|obb))?` and adds a new `_dota_evaluator.py` file providing `load_dota_dataset()` + `evaluate_obb_map()`. The `run_benchmark` function in `__init__.py` dispatches to the DOTA path when `task == "obb"`.

**Primary recommendation:** Fix `engine.py:161` first (1-line, isolated, testable immediately). Fix `benchmark/__init__.py:37` second (pattern + dispatch). Write the DOTA evaluator as a new isolated file. Align `tune_command`. Write four unit tests in `test_obb_integration.py`.

---

## Standard Stack

### Core (no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | existing | Tensor ops for `probiou_matrix` in mAP IoU sweep | Already in project |
| `numpy` | existing | Numeric arrays for DOTA label parsing | Already in project |
| `pathlib.Path` | stdlib | DOTA dataset directory traversal | Consistent with `load_coco_dataset` pattern |
| `re` | stdlib | Model pattern extension | Already used at `benchmark/__init__.py:37` |
| `unittest.mock` | stdlib | CI-safe patching of `OBBEngine` and `load_dota_dataset` | Phase 5 precedent |

### No New Dependencies

The entire phase requires zero new package dependencies. `probiou_matrix` already exists in `src/yowo/postprocess/_obb_nms.py` and is available for import. DOTA evaluation is pure Python + torch — no external toolkit needed.

---

## Architecture Patterns

### Recommended File Layout (changes only)

```
src/yowo/
├── engine.py                         # L161: 1-line key fix in _load_tune_profile
├── cli/_main.py                      # L1192: tune_command stores spec-derived key
├── benchmark/
│   ├── __init__.py                   # L37: pattern fix; dispatch branch for obb
│   ├── _dota_evaluator.py            # NEW: load_dota_dataset + evaluate_obb_map
│   └── _runner.py                    # run_single_backend: OBBEngine branch for task==obb
tests/unit/
└── test_obb_integration.py           # NEW: 4 tests covering INT-A1 + INT-A2
```

### Pattern 1: Spec-Derived Profile Key

**What:** Key construction pulled from `ModelSpec.task` field
**When to use:** Anywhere a profile key is built or consumed
**Example:**
```python
# src/yowo/engine.py L161 — before (broken):
model_name = f"{spec.family.value}{spec.size.value}"

# after (fixed):
task_suffix = spec.task if spec.task not in ("detect",) else ""
model_name = f"{spec.family.value}{spec.size.value}{'-' + task_suffix if task_suffix else ''}"
```

### Pattern 2: tune_command Key Alignment

**What:** `tune_command` must parse the CLI `model` arg into a `ModelSpec` first, then derive the key from the spec — not use the raw string
**When to use:** Wherever `TuneProfile(model=...)` is constructed
**Current code location:** `cli/_main.py` around L1247
```python
# current (broken — uses raw CLI string):
profile = TuneProfile(model=model, ...)

# fixed (derive from parsed spec):
spec = _parse_model_spec(model)  # already called at L1185
task_suffix = spec.task if spec.task not in ("detect",) else ""
model_key = f"{spec.family.value}{spec.size.value}{'-' + task_suffix if task_suffix else ''}"
profile = TuneProfile(model=model_key, ...)
```

Note: `_parse_model_spec` is already called at `tune_command` line 1185 — `spec` is already available. The `model_key` computation is the only addition.

### Pattern 3: _MODEL_PATTERN Extension

**What:** Regex now accepts `-(cls|obb)` optional suffix
**Current (broken):**
```python
_MODEL_PATTERN = re.compile(r"^(yolo(?:11|26))([nsmxl])(?:-(cls))?$")
```
**Fixed:**
```python
_MODEL_PATTERN = re.compile(r"^(yolo(?:11|26))([nsmxl])(?:-(cls|obb))?$")
```
**`_parse_model_name` extension:**
```python
# current:
task = "classify" if task_suffix == "cls" else "detect"
# fixed:
if task_suffix == "cls":
    task = "classify"
elif task_suffix == "obb":
    task = "obb"
else:
    task = "detect"
```

### Pattern 4: DOTA Evaluator Module (`_dota_evaluator.py`)

**What:** New file following the `load_coco_dataset` / `evaluate_coco_map` pattern from `_evaluator.py`
**Structure:**
```python
# src/yowo/benchmark/_dota_evaluator.py

DOTA_CLASS_NAMES: list[str] = [...]  # 15 names, matches DOTA_CLASSES in _obb_nms.py

def load_dota_dataset(
    data_path: str | Path,
    subset: int | None = None,
) -> tuple[list[Path], list[list[list[float]]], list[list[int]]]:
    """Load DOTA v1 val split.
    Returns (image_paths, gt_boxes_per_image, gt_classes_per_image).
    gt_boxes_per_image: each entry is list of [x1,y1,x2,y2,x3,y3,x4,y4] quad coords.
    """
    root = Path(data_path)
    images_dir = root / "images" / "val"
    labels_dir = root / "labelTxt" / "val"
    # FileNotFoundError if either missing

def evaluate_obb_map(
    detections: list[OBBDetection],
    gt_boxes: list[list[list[float]]],
    gt_classes: list[list[int]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute COCO-style mAP50-95 using probiou.
    Iterates thresholds 0.50, 0.55, ..., 0.95 (10 values), averages.
    Returns {"mAP_50_95": float, "mAP_50": float}.
    """
```

### Pattern 5: run_single_backend OBB Branch

**What:** `run_single_backend` in `_runner.py` currently has `if task == "classify": ... else: DetectionEngine`. Extend with OBB branch.
**Location:** `benchmark/_runner.py` around L127
```python
if task == "classify":
    engine = ClassificationEngine(...)
elif task == "obb":
    from yowo.obb_engine import OBBEngine
    engine = OBBEngine(...)
else:
    engine = DetectionEngine(...)
```

OBB inference loop calls `engine.detect_obb([frame])` (method name confirmed from `obb_engine.py`). Results are `list[OBBDetection]`.

### Pattern 6: run_benchmark OBB Dispatch

**What:** `run_benchmark` in `__init__.py` dispatches dataset loading by task
**Current:** `if task == "classify": ... else: load_coco_dataset(...)`
**Fixed:**
```python
if task == "classify":
    images, labels = load_imagenet_dataset(data, subset=subset)
    image_ids = labels
elif task == "obb":
    from yowo.benchmark._dota_evaluator import load_dota_dataset
    images, gt_boxes, gt_classes = load_dota_dataset(data, subset=subset)
    image_ids = None
    gt_ann_path = None
    # pass gt_boxes / gt_classes into run_all_backends via new param or store on side
else:
    images, image_ids, gt_ann_path = load_coco_dataset(data, subset=subset)
```

Note: `run_all_backends` currently only passes `image_ids` and `gt_ann_path`. For OBB, the ground-truth structure is different (list-of-quads, not COCO JSON). The planner must decide whether to extend `run_all_backends` signature or compute mAP separately after collecting OBBDetection results. The cleanest approach (matching CONTEXT.md guidance) is to branch `task == "obb"` inside `run_single_backend` and have the evaluator compute mAP at the end.

### Anti-Patterns to Avoid

- **Using raw CLI `model` string as profile key in `TuneProfile.model`:** This was the root cause of INT-A1. Always derive key from `ModelSpec`.
- **Hard-coding detection-only pattern in `_MODEL_PATTERN`:** Already caused INT-A2. Pattern must grow with supported tasks.
- **Importing `OBBEngine` at module level in `_runner.py`:** Breaks patchability. Use lazy import inside the branch (consistent with project's selective lazy-import pattern for optional engines).
- **Raising on missing DOTA data in mAP call, not dataset load:** `load_dota_dataset` should raise `FileNotFoundError` with helpful structure hint (matching `load_coco_dataset` pattern).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rotated IoU computation | Custom Bhattacharyya impl | `probiou_matrix` in `postprocess/_obb_nms.py` | Already validated against ultralytics; reuse avoids divergence |
| DOTA category mapping | Hard-code class index maps | `DOTA_CLASSES` list in `postprocess/_obb_nms.py` (15 names) | Shared source of truth; consistent with OBBEngine |
| mAP threshold iteration | Custom COCO-style sweep | Simple Python loop over `np.arange(0.5, 1.0, 0.05)` | Trivial, no library needed; 10 thresholds |
| Profile path resolution | Custom naming logic | `_default_profile_path(model_key, fingerprint)` from `_profile.py` | Already handles `~/.cache/yowo/profiles/{fp}/{model}.yaml` |

**Key insight:** Both fixes are primarily plumbing corrections — the computation infrastructure (probiou, OBBEngine, profile persistence) is fully built. This phase wires existing parts correctly.

---

## Common Pitfalls

### Pitfall 1: Partial Key Fix (engine.py only, not tune_command)
**What goes wrong:** Only fixing `_load_tune_profile` means the engine now looks up `"yolo11n-obb"` but `tune_command` still *saves* the profile under `"yolo11n-obb"` (raw CLI string) — which coincidentally matches. However, the `existing = load_profile(model, hw)` check at `tune_command:1192` still uses the raw `model` string, so force-skip detection and profile path resolution remain mismatched.
**Why it happens:** Two separate call sites both need the spec-derived key.
**How to avoid:** Fix both `_load_tune_profile` (line 161) AND `tune_command` (lines 1192 and 1247). The `_parse_model_spec` call already exists at line 1185 — just use `spec` to derive the key.
**Warning signs:** Test `test_tune_profile_key_detect_unchanged` passes but `test_tune_profile_key_obb` fails with wrong key.

### Pitfall 2: OBBEngine Import at Module Level in _runner.py
**What goes wrong:** Adding `from yowo.obb_engine import OBBEngine` at the top of `_runner.py` makes it impossible to patch `OBBEngine` in tests via `patch("yowo.benchmark._runner.OBBEngine")` if the import happens during module load.
**Why it happens:** Module-level imports bind the name at load time; patch may not intercept.
**How to avoid:** Use lazy import inside the `elif task == "obb":` branch — consistent with how detection/classification tests already patch `DetectionEngine` / `ClassificationEngine`.
**Warning signs:** `test_benchmark_obb_dispatches_dota_path` fails because mock is never called despite patching.

### Pitfall 3: Forgetting run_all_backends gt_boxes Threading
**What goes wrong:** `run_all_backends` currently passes `gt_ann_path` to `run_single_backend`. OBB uses different GT structure (`gt_boxes`, `gt_classes`). If the signature is extended naively, existing detect/classify callers break.
**Why it happens:** Function signature shared across all tasks.
**How to avoid:** Either (a) add optional `gt_boxes`/`gt_classes` params with `None` defaults, or (b) compute OBB mAP *after* `run_all_backends` returns raw detection results, or (c) compute mAP inside `run_single_backend` for the OBB branch without touching `run_all_backends` signature. Option (c) is cleanest — mirrors how detect mAP is computed at the end of `run_single_backend` already.
**Warning signs:** Existing detect benchmark tests fail after OBB branch is added.

### Pitfall 4: DOTA Label Format Misparse
**What goes wrong:** DOTA `labelTxt` files have format `x1 y1 x2 y2 x3 y3 x4 y4 class difficulty`. Parsing as 8 floats + class name + int is straightforward, but lines starting with `#` (file headers) must be skipped.
**Why it happens:** DOTA v1 labelTxt files include comment lines.
**How to avoid:** Skip lines starting with `#` when parsing. Split each valid line and take first 9 tokens.
**Warning signs:** `float()` conversion error on header lines.

### Pitfall 5: mAP IoU Threshold Accumulation Without Per-Class Handling
**What goes wrong:** COCO mAP50-95 averages over 10 IoU thresholds AND over all classes. A naive implementation that only averages over thresholds (not classes) gives incorrect mAP.
**Why it happens:** Confusion between image-level mAP and class-averaged mAP.
**How to avoid:** For each threshold, compute AP per class (using precision-recall with probiou), then average over classes, then average over thresholds. This matches the COCO-style computation described in CONTEXT.md.
**Warning signs:** OBB mAP numbers are systematically higher than expected (class averaging inflates if done wrong).

---

## Code Examples

Verified patterns from the codebase:

### Key Derivation (spec.task aware)
```python
# Replaces engine.py:161
task_suffix = spec.task if spec.task not in ("detect",) else ""
model_name = f"{spec.family.value}{spec.size.value}{'-' + task_suffix if task_suffix else ''}"
```
Source: CONTEXT.md `<specifics>` — exact proposed line

### probiou_matrix Signature (for reuse in mAP)
```python
# From src/yowo/postprocess/_obb_nms.py:51
def probiou_matrix(obb1: Tensor, obb2: Tensor, eps: float = 1e-7) -> Tensor:
    """obb1: (N, 5) xywhr, obb2: (M, 5) xywhr -> (N, M) IoU matrix [0,1]"""
```
Source: confirmed via direct file read

### load_coco_dataset Pattern (to mirror for DOTA)
```python
# From src/yowo/benchmark/_evaluator.py:231
def load_coco_dataset(
    data_path: str | Path,
    subset: int | None = None,
) -> tuple[list[Path], list[int], str]:
    root = Path(data_path)
    val_dir = root / "val2017"
    ann_file = root / "annotations" / "instances_val2017.json"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"... expected layout ...")
```
Source: direct file read — mirror this pattern for DOTA `images/val/` + `labelTxt/val/`

### OBBDetection Access Pattern
```python
# OBBDetection has .boxes: tuple[OBBBox, ...]
# OBBBox has .cx, .cy, .w, .h, .angle, .confidence, .class_id
# probiou_matrix expects (N, 5) xywhr tensors
# Convert OBBBox to tensor row: [cx, cy, w, h, angle]
```
Source: confirmed from `types.py` and `postprocess/_obb_nms.py`

### Test Mocking Pattern (Phase 5 reference)
```python
# Consistent with existing test_obb_engine.py pattern
@patch("yowo.benchmark._runner.OBBEngine")
def test_benchmark_obb_dispatches_dota_path(mock_obb_cls):
    mock_engine = MagicMock()
    mock_obb_cls.return_value = mock_engine
    mock_engine.__enter__ = MagicMock(return_value=mock_engine)
    mock_engine.__exit__ = MagicMock(return_value=False)
    mock_engine.detect_obb.return_value = []
    # verify load_dota_dataset is called, not load_coco_dataset
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `model_name = f"{spec.family.value}{spec.size.value}"` | `model_name = f"{spec.family.value}{spec.size.value}[-{task}]"` | Phase 6 | OBB tune profiles are now correctly stored and loaded |
| `_MODEL_PATTERN` without `-obb` | Pattern accepts `-(cls\|obb)` | Phase 6 | `yowo benchmark --model yolo11n-obb` no longer raises `ValueError` |
| COCO-only benchmark evaluation path | COCO + ImageNet + DOTA dispatch | Phase 6 | OBB mAP50-95 measurable via CLI |

**Deprecated/outdated:**
- Profile key from raw CLI string: not deprecated (still works for detect), but tunnel-narrowed to spec-derived key for all tasks.

---

## Open Questions

1. **run_single_backend OBB return of raw OBBDetection vs COCO format**
   - What we know: `BenchmarkResult.map_50_95` is `float | None`; OBB mAP must be computed inside `run_single_backend` using probiou
   - What's unclear: Whether to collect all OBBDetection results then compute mAP at end of `run_single_backend` (clean, mirrors detect path) or stream per-image
   - Recommendation: Collect `all_obb_detections: list[OBBDetection]` alongside `latencies`, then compute `evaluate_obb_map` once after the loop — mirrors the detect path exactly

2. **ultralytics comparison for OBB task**
   - What we know: `run_ultralytics_benchmark` currently handles `"detect"` and `"classify"` only; returns `None` on unknown task
   - What's unclear: Whether to add OBB support to `_comparison.py`
   - Recommendation: CONTEXT.md says "if not available, skip gracefully" — extend with `task == "obb"` branch that calls `model.val(task="obb")` and catches gracefully; if ultralytics OBB val is unavailable, returns `None` (already handled by outer `except Exception`)

3. **DOTA quad-to-xywhr conversion for mAP IoU**
   - What we know: Ground truth in DOTA format is 4-corner quad (x1,y1,...,x4,y4); `probiou_matrix` expects `(N, 5) xywhr` tensors
   - What's unclear: Exact quad-to-xywhr conversion formula
   - Recommendation: Use the minimum bounding box approach — cx = mean(x corners), cy = mean(y corners), w/h from corner distances, angle from longest edge. This matches ultralytics DOTA preprocessing convention.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_obb_integration.py -x -q` |
| Full suite command | `uv run pytest tests/unit/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TUNE-01 | Profile key includes `-obb` for OBB task | unit | `uv run pytest tests/unit/test_obb_integration.py::test_tune_profile_key_obb -x` | Wave 0 |
| TUNE-01 | Profile key unchanged for detect task | unit | `uv run pytest tests/unit/test_obb_integration.py::test_tune_profile_key_detect_unchanged -x` | Wave 0 |
| BENCH-01 | `_MODEL_PATTERN` accepts `"yolo11n-obb"` | unit | `uv run pytest tests/unit/test_obb_integration.py::test_benchmark_pattern_accepts_obb -x` | Wave 0 |
| OBB-01 | OBB benchmark dispatches DOTA path, not COCO | unit | `uv run pytest tests/unit/test_obb_integration.py::test_benchmark_obb_dispatches_dota_path -x` | Wave 0 |
| OBB-03 | Full regression suite (arch + integration) | unit | `uv run pytest tests/unit/ -x -q` | Existing |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_obb_integration.py -x -q`
- **Per wave merge:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_obb_integration.py` — covers TUNE-01, BENCH-01, OBB-01, OBB-03
- [ ] `src/yowo/benchmark/_dota_evaluator.py` — covers OBB mAP evaluation path

---

## Sources

### Primary (HIGH confidence)
- Direct read of `src/yowo/engine.py:136-184` — confirmed `_load_tune_profile` key construction at line 161
- Direct read of `src/yowo/benchmark/__init__.py:37` — confirmed `_MODEL_PATTERN` missing `-obb`
- Direct read of `src/yowo/benchmark/_runner.py` — confirmed `run_single_backend` detect/classify branches only
- Direct read of `src/yowo/benchmark/_evaluator.py` — confirmed `load_coco_dataset` + `evaluate_coco_map` patterns to mirror
- Direct read of `src/yowo/postprocess/_obb_nms.py:51` — confirmed `probiou_matrix(obb1: Tensor, obb2: Tensor) -> Tensor` signature
- Direct read of `src/yowo/cli/_main.py:1173-1257` — confirmed `tune_command` stores `TuneProfile(model=model, ...)` using raw CLI string
- Direct read of `src/yowo/tune/_profile.py` — confirmed `save_profile` / `load_profile` / `_default_profile_path` patterns
- Direct read of `src/yowo/obb_engine.py:1-177` — confirmed `OBBEngine.__init__` calls `_load_tune_profile(spec, ...)` with spec having `task="obb"`
- Direct read of `.planning/phases/06-obb-integration-fixes/06-CONTEXT.md` — locked decisions and implementation specifics

### Secondary (MEDIUM confidence)
- DOTA v1 dataset layout: verified against ultralytics DOTA convention documentation referenced in CONTEXT.md decisions

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already present; no new dependencies
- Architecture: HIGH — exact file locations, line numbers, and call sites confirmed via direct read
- Pitfalls: HIGH — derived from direct code inspection of both bug sites and surrounding call patterns
- DOTA evaluator design: MEDIUM — follows established pattern, DOTA quad-to-xywhr conversion is Claude's discretion

**Research date:** 2026-03-08
**Valid until:** Stable (no fast-moving dependencies; pure internal plumbing fix)
