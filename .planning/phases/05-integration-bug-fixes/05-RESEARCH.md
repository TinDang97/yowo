# Phase 5: Integration Bug Fixes - Research

**Researched:** 2026-03-08
**Domain:** Python bug fix / public API surface / type-safe refactoring
**Confidence:** HIGH

## Summary

Phase 5 closes three discrete integration gaps (INT-P0, INT-P1, INT-P2) identified by the v2.3
milestone audit. Every change is surgical: one 1-line guard fix, one constructor addition, and five
import lines. No new inference capability ships. All three gaps are independent — tasks can be
implemented and tested in any order.

The critical risk is the type-signature mismatch in INT-P1: `_load_tune_profile` is annotated
`config: InferenceConfig` but OBBEngine holds an `OBBConfig`. At runtime this is safe (both
dataclasses expose `backend`, `batch_size`, `precision` and `dataclasses.replace()` is duck-typed),
but Pyright will emit a type error unless the implementation uses `cast()` or the function signature
is updated to accept a union. The planner must choose one of the two approaches.

**Primary recommendation:** Fix INT-P0 first (P0 crash), then INT-P1 (silent misbehavior), then
INT-P2 (ergonomics). Each fix ships as a standalone, independently testable task.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**INT-P0: kv-cache guard fix (`export/_exporter.py:139`)**
- Change `spec.task != "classify"` to `spec.task not in ("classify", "obb")`
- Minimal, correct fix — OBB models wrap `OBBModel` which is not a `YOLOModel` subclass
- No other changes in the export path needed

**INT-P0: Regression test design**
- Mock-based unit test — no actual ONNX file produced, no filesystem side-effects
- Test via `unittest.mock.patch` on `_wrap_model_for_kv_cache` (or the assert call site) to verify
  it is NOT called when task is `"obb"`
- Test the three cases: `task="obb"` skips kv_cache wrap, `task="classify"` skips, `task="detect"`
  calls it
- One regression test per gap clause — fast, isolated, CI-safe

**INT-P1: OBBEngine tune profile behavior**
- `_load_tune_profile()` called in `OBBEngine.__init__` at the same point DetectionEngine calls it
  (`engine.py:975–980`): after config is resolved, before `_backend` is selected
- Identical behavior to DetectionEngine: silent `DEBUG` log if no profile found for the
  device+model combo — no user-visible warning
- The existing `_load_tune_profile()` function (module-level in `engine.py`) is reused as-is — no
  modification needed

**INT-P2: Public API exports (`src/yowo/__init__.py`)**
- Add exactly the 5 types named in the audit: `OBBBox`, `OBBDetection`, `WarmupValidationError`,
  `HealthReport`, `StreamConfig`
- Sources:
  - `OBBBox`, `OBBDetection`, `StreamConfig` from `yowo.types`
  - `WarmupValidationError` from `yowo.errors`
  - `HealthReport` from `yowo.engine`
- Scope is strictly the 5 named types — any additional gaps found during implementation become tech
  debt notes, not scope additions
- Add corresponding entries to `__all__` in `__init__.py`

**INT-P2: Test coverage**
- Add a smoke test: `from yowo import OBBBox, OBBDetection, WarmupValidationError, HealthReport, StreamConfig`
- One test function, `test_public_api_exports`, covers all 5 types

### Claude's Discretion
- Exact test file placement (`tests/unit/test_public_api.py` or alongside export tests — planner decides)
- Whether to add `OBBConfig` to the same export batch pass (it's already exported; no-op)
- Order of exports in `__init__.py` (keep alphabetical within existing sections)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| OBB-01 | OBB detection head produces oriented bounding boxes with rotation angle | INT-P2 closes gap: `OBBBox`/`OBBDetection` now reachable via top-level import |
| OBB-02 | OBB postprocessing includes rotation-aware NMS | INT-P2 closes gap: `OBBDetection` exported to public API |
| OBB-06 | OBB models export to ONNX and TensorRT correctly | INT-P0 closes gap: OBB+kv_cache export no longer crashes |
| CORR-07 | Export accuracy delta vs PyTorch baseline < 1% mAP per format | INT-P0 ensures OBB export path executes correctly (no AssertionError) |
| CORR-08 | Model warmup validates output shape/value range before accepting requests | INT-P2 closes gap: `WarmupValidationError` now reachable via top-level import |
| RELY-05 | Metrics export in Prometheus-compatible format or JSON | INT-P2 closes gap: `HealthReport` now reachable via top-level import |
| STRM-01 | Multi-stream pipeline handles 100+ concurrent streams | INT-P2 closes gap: `StreamConfig` now reachable via top-level import |
| TUNE-01 | User can run `yowo tune` to auto-detect optimal backend/batch/precision | INT-P1 closes gap: OBBEngine now auto-loads saved tune profile on construction |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `unittest.mock` | stdlib | Patching `_wrap_model_for_kv_cache` in INT-P0 test | Already used in test_obb_export.py and test_exporter.py |
| `dataclasses` | stdlib | `dataclasses.replace()` for config mutation in `_load_tune_profile` | Already used by the function itself |
| `pytest` | project | Test runner | Project standard (uv run pytest) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `typing.cast` | stdlib | Satisfying Pyright for OBBConfig→InferenceConfig arg | Required if planner keeps `_load_tune_profile` signature unchanged |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cast(InferenceConfig, cfg)` | Widen `_load_tune_profile` signature to `InferenceConfig \| OBBConfig \| ClassificationConfig` | Union is more accurate but changes a shared function signature; `cast` is zero-runtime-cost and keeps the diff minimal |

**Installation:**
No new dependencies. All stdlib.

## Architecture Patterns

### Recommended Project Structure

No new files required for the fixes themselves. One new test file is needed:

```
tests/unit/
├── test_exporter.py            # existing — extend with kv_cache guard tests (INT-P0)
│                               # OR create test_kv_cache_guard.py (planner decides)
└── test_public_api.py          # new — INT-P2 smoke test (5 types)
```

### Pattern 1: kv-cache guard expansion (INT-P0)

**What:** Extend the existing task-exclusion set from a single string equality to a tuple membership test.

**When to use:** Any time a feature is task-specific and exclusions expand.

**Current code (line 139 of `src/yowo/export/_exporter.py`):**
```python
# KV-cache export is detection-only — not supported for classify task
if kv_cache and spec.task != "classify":
```

**Fixed code:**
```python
# KV-cache export is detection-only — not supported for classify or obb tasks
if kv_cache and spec.task not in ("classify", "obb"):
```

**Why the bug:** `OBBModel` is not a `YOLOModel` subclass, so the `assert isinstance(model, YOLOModel)` on the next line raises `AssertionError` when task is `"obb"`.

### Pattern 2: INT-P0 mock-based regression test

**What:** Patch `_wrap_model_for_kv_cache` at the `yowo.export._exporter` namespace and assert it is
never called for obb/classify tasks, always called for detect task.

**Established pattern** from `test_obb_export.py`:
```python
from unittest.mock import MagicMock, patch

with (
    patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
    patch("yowo.arch.build_obb_model", return_value=mock_model),
    # ... other necessary patches ...
    patch("yowo.export._exporter._wrap_model_for_kv_cache") as mock_wrap,
):
    export_model(spec, ExportFormat.ONNX, tmp_path, kv_cache=True)
    mock_wrap.assert_not_called()
```

**Three test cases required (per locked decision):**
1. `task="obb"`, `kv_cache=True` → `_wrap_model_for_kv_cache` NOT called
2. `task="classify"`, `kv_cache=True` → NOT called (existing behavior, regression guard)
3. `task="detect"`, `kv_cache=True` → called (positive case ensures guard doesn't over-exclude)

### Pattern 3: OBBEngine tune profile integration (INT-P1)

**What:** Mirror the exact DetectionEngine pattern at `engine.py:975–980`.

**DetectionEngine reference (engine.py:977–980):**
```python
_hw_cache: HardwareProfile | None = None
if backend_instance is None:
    _hw_cache = get_hardware_profile()
    cfg = _load_tune_profile(spec, cfg, _hw_cache)
super().__init__(
    ...
    backend_override=cfg.backend.value if cfg.backend else None,
    ...
)
```

**OBBEngine insertion point:** After `spec = ModelSpec(...)` is constructed (line ~148), before
`super().__init__(...)` (line ~150). Insert:

```python
# Apply tune profile (only when user has not explicitly overridden backend/batch/precision
# and no custom backend instance was provided)
_hw_cache: HardwareProfile | None = None
if backend_instance is None:
    from yowo.hardware import get_hardware_profile
    from yowo.engine import _load_tune_profile
    _hw_cache = get_hardware_profile()
    cfg = cast(OBBConfig, _load_tune_profile(spec, cast(Any, cfg), _hw_cache))
```

**Type annotation note:** `_load_tune_profile` is annotated `config: InferenceConfig`. Since
`OBBConfig` has the exact same `backend`, `batch_size`, and `precision` fields and
`dataclasses.replace()` is duck-typed at runtime, the call is safe. Use `cast(Any, cfg)` to satisfy
Pyright without changing the shared function signature.

### Pattern 4: Public API export expansion (INT-P2)

**What:** Add 5 import lines + 5 `__all__` entries to `src/yowo/__init__.py`.

**Established pattern** — existing grouped imports in `__init__.py`:

OBBEngine is already imported at line 62:
```python
from yowo.obb_engine import OBBEngine
```

Extend the `yowo.types` import block (lines 81–99) to add OBBBox, OBBDetection, StreamConfig.
Add WarmupValidationError to the `yowo.errors` block (lines 39–56).
Add HealthReport to the `yowo.engine` import (line 38).

All 5 types are already in their respective module's `__all__` — this is purely a re-export.

### Anti-Patterns to Avoid

- **Modifying `_load_tune_profile` signature:** The function is a module-level utility shared by
  DetectionEngine. Changing its signature to accept a union type would require updating callers and
  type stubs. Use `cast` in OBBEngine instead.
- **Expanding INT-P2 scope:** Only the 5 named types. Discovering other missing exports during
  implementation → tech debt note, not scope addition.
- **Integration tests for INT-P0:** The locked decision requires mock-based unit tests. No actual
  ONNX file production, no torch dependency in the test.
- **Skipping the positive test case for INT-P0:** Testing only that obb/classify skip the wrap is
  insufficient — also assert that detect DOES call it (prevents over-broad guard regression).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Profile lookup in OBBEngine | Custom profile loading logic | `_load_tune_profile()` at `engine.py:136` | Already handles no-profile gracefully, atomic reads, correct fingerprinting |
| Hardware detection in OBBEngine | Direct nvml/subprocess calls | `get_hardware_profile()` from `yowo.hardware` | Cached singleton, fail-safe, already imported by engine.py |
| Export branch testing | Actual ONNX export in tests | `unittest.mock.patch` | Avoids torch/onnx dependency in unit tests; established pattern in test_obb_export.py |

**Key insight:** All three fixes reuse existing infrastructure exactly. Zero new abstractions needed.

## Common Pitfalls

### Pitfall 1: Pyright type error on _load_tune_profile call from OBBEngine
**What goes wrong:** Pyright flags `_load_tune_profile(spec, cfg, _hw_cache)` because `cfg` is
`OBBConfig`, not `InferenceConfig`.
**Why it happens:** `_load_tune_profile` signature is `config: InferenceConfig`.
**How to avoid:** Use `cast(Any, cfg)` for the argument, or import `typing.cast`. Quality gate
`uv run pyright src/yowo/` must pass before commit.
**Warning signs:** Pyright error `Argument of type "OBBConfig" cannot be assigned to parameter
"config" of type "InferenceConfig"`.

### Pitfall 2: _load_tune_profile returns InferenceConfig, not OBBConfig
**What goes wrong:** After the call, `cfg` becomes `InferenceConfig` (the return type), breaking
subsequent OBBEngine-specific attribute access.
**Why it happens:** `dataclasses.replace()` inside `_load_tune_profile` produces an `InferenceConfig`
if the type annotation drives the call, but at runtime it returns the same dataclass type as the
input. However Pyright will infer the return as `InferenceConfig`.
**How to avoid:** Cast the result back: `cfg = cast(OBBConfig, _load_tune_profile(spec, cast(Any, cfg), _hw_cache))`. Alternatively keep cfg as `Any` after the call — OBBEngine only uses `cfg.backend`, `cfg.batch_size`, `cfg.precision` in super().__init__.
**Warning signs:** AttributeError accessing OBBConfig-specific fields after profile load.

### Pitfall 3: INT-P0 test patches wrong namespace
**What goes wrong:** Patching `yowo.export._kv_wrapper._wrap_model_for_kv_cache` instead of
`yowo.export._exporter._wrap_model_for_kv_cache` — the patch doesn't intercept the call.
**Why it happens:** Mock patching must target the namespace where the name is USED, not where it's
defined.
**How to avoid:** Check `_exporter.py` imports for `_wrap_model_for_kv_cache` — if imported at the
top, patch `yowo.export._exporter._wrap_model_for_kv_cache`; if imported inline within the `if`
block, patch the definition site `yowo.export._kv_wrapper.YOLOKVWrapper.__init__` or assert via
the `assert isinstance` call site.
**Warning signs:** Test passes even when the guard is reverted (mock not intercepting).

### Pitfall 4: INT-P2 types missing from __all__
**What goes wrong:** Import works (`from yowo import OBBBox`) but `"OBBBox" not in yowo.__all__` —
breaking `import *` and tooling introspection.
**Why it happens:** Forgetting to update `__all__` after adding the import line.
**How to avoid:** Update both the import line AND the `__all__` list. The smoke test should also
assert `"OBBBox" in dir(yowo)` or `"OBBBox" in yowo.__all__`.

### Pitfall 5: OBBEngine tune profile applied before backend_instance check
**What goes wrong:** `_load_tune_profile` called even when `backend_instance` is provided, overriding
the injected backend's implicit config.
**Why it happens:** Missing the `if backend_instance is None:` guard (present in DetectionEngine).
**How to avoid:** Mirror the DetectionEngine pattern exactly — wrap the profile call in `if backend_instance is None:`.

## Code Examples

### INT-P0: The exact guard (verified from source)
```python
# src/yowo/export/_exporter.py, line ~139
# BEFORE (buggy):
if kv_cache and spec.task != "classify":
    from yowo.arch._yolo import YOLOModel
    from yowo.export._kv_wrapper import YOLOKVWrapper
    assert isinstance(model, YOLOModel), "kv_cache is only supported for detection models"
    wrapper = YOLOKVWrapper(model)
    _export_onnx_kv(wrapper, dummy, onnx_path, dynamic_batch=dynamic_batch)

# AFTER (fixed):
if kv_cache and spec.task not in ("classify", "obb"):
    from yowo.arch._yolo import YOLOModel
    from yowo.export._kv_wrapper import YOLOKVWrapper
    assert isinstance(model, YOLOModel), "kv_cache is only supported for detection models"
    wrapper = YOLOKVWrapper(model)
    _export_onnx_kv(wrapper, dummy, onnx_path, dynamic_batch=dynamic_batch)
```

### INT-P1: OBBEngine constructor addition (verified from source context)
```python
# src/yowo/obb_engine.py — inside __init__, after spec = ModelSpec(...), before super().__init__
from typing import Any, cast
from yowo.hardware import get_hardware_profile
from yowo.engine import _load_tune_profile

_hw_cache: HardwareProfile | None = None
if backend_instance is None:
    _hw_cache = get_hardware_profile()
    cfg = cast(OBBConfig, _load_tune_profile(spec, cast(Any, cfg), _hw_cache))
```

### INT-P2: Public API additions (verified from source — all 5 types confirmed in their modules)
```python
# src/yowo/__init__.py additions:

# In yowo.engine import block (line 38):
from yowo.engine import DetectionEngine, HealthReport, InferenceEngine

# In yowo.errors import block (lines 39–56), add:
from yowo.errors import (
    ...
    WarmupValidationError,
    ...
)

# In yowo.types import block (lines 81–99), add to existing block:
from yowo.types import (
    ...
    OBBBox,
    OBBDetection,
    StreamConfig,
    ...
)

# In __all__ (alphabetical within existing list):
"HealthReport",   # after "FrameDropPolicy"
"OBBBox",         # after "OBBConfig"
"OBBDetection",   # after "OBBBox"
"StreamConfig",   # after "ShutdownError"
"WarmupValidationError",  # after "TrackedDetection" / before "YowoError"
```

### INT-P2: Smoke test (planner decides file placement)
```python
def test_public_api_exports() -> None:
    """All 5 INT-P2 types are importable from top-level yowo without submodule qualification."""
    from yowo import (
        HealthReport,
        OBBBox,
        OBBDetection,
        StreamConfig,
        WarmupValidationError,
    )
    import yowo

    for name in ("HealthReport", "OBBBox", "OBBDetection", "StreamConfig", "WarmupValidationError"):
        assert name in yowo.__all__, f"{name!r} missing from yowo.__all__"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `spec.task != "classify"` guard | `spec.task not in ("classify", "obb")` | Phase 5 (INT-P0) | Prevents AssertionError for OBB+kv_cache export |
| OBBEngine ignores tune profiles | OBBEngine mirrors DetectionEngine profile load | Phase 5 (INT-P1) | `yowo tune --model yolo11n-obb` results actually apply |
| `OBBBox`/etc only accessible via submodules | All 5 types in top-level `yowo` namespace | Phase 5 (INT-P2) | User-facing import ergonomics |

**Deprecated/outdated:**
- Single-string task exclusion for kv_cache: replaced by tuple membership test

## Open Questions

1. **Import location for `_load_tune_profile` in OBBEngine**
   - What we know: `_load_tune_profile` is a module-level private function in `engine.py`; importing private names across modules is technically allowed but slightly unusual
   - What's unclear: Whether to `from yowo.engine import _load_tune_profile` at module level (for test patchability, consistent with Phase 3–4 CLI pattern) or inline within `__init__`
   - Recommendation: Module-level import for test patchability — consistent with established project pattern (cli/_main.py imports all tune functions at module level)

2. **Test file placement for INT-P0 regression**
   - What we know: Three options — extend `test_exporter.py`, extend `test_obb_export.py`, or create `test_kv_cache_guard.py`
   - What's unclear: Whether the kv_cache guard test belongs with general exporter tests or OBB-specific export tests
   - Recommendation: Extend `test_obb_export.py` — the guard specifically exists to protect OBB exports; co-location aids discoverability

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `uv run pytest tests/unit/test_obb_export.py tests/unit/test_public_api.py -x -q` |
| Full suite command | `uv run pytest tests/unit/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OBB-06 / CORR-07 / INT-P0 | OBB+kv_cache export does not crash (guard skips wrap) | unit | `uv run pytest tests/unit/test_obb_export.py -x -q -k kv_cache` | ❌ Wave 0 (add to test_obb_export.py) |
| TUNE-01 / INT-P1 | OBBEngine loads tune profile at construction (no-profile case is silent DEBUG) | unit | `uv run pytest tests/unit/test_obb_engine.py -x -q -k tune` | ❌ Wave 0 (add to test_obb_engine.py) |
| OBB-01 / OBB-02 / CORR-08 / RELY-05 / STRM-01 / INT-P2 | All 5 types importable from top-level `yowo` namespace | unit/smoke | `uv run pytest tests/unit/test_public_api.py -x -q` | ❌ Wave 0 (new file) |

### Sampling Rate
- **Per task commit:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Per wave merge:** `uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_obb_export.py` — extend with kv_cache guard tests (3 cases: obb skips, classify skips, detect calls)
- [ ] `tests/unit/test_obb_engine.py` — extend with tune profile integration test (mock `_load_tune_profile`, assert called when no backend_instance)
- [ ] `tests/unit/test_public_api.py` — new file, `test_public_api_exports` function covering all 5 INT-P2 types

## Sources

### Primary (HIGH confidence)
- Direct source read: `src/yowo/export/_exporter.py:139` — confirmed exact guard text and surrounding context
- Direct source read: `src/yowo/obb_engine.py:92–170` — confirmed `__init__` structure, absence of `_load_tune_profile` call
- Direct source read: `src/yowo/engine.py:136–184, 975–980` — confirmed `_load_tune_profile` signature, DetectionEngine pattern
- Direct source read: `src/yowo/__init__.py:1–188` — confirmed all 5 missing types, existing import groupings, `__all__` structure
- Direct source read: `src/yowo/types.py:326–368` — confirmed `OBBBox`, `OBBDetection`, `StreamConfig` definitions and `__all__` membership
- Direct source read: `src/yowo/errors.py` — confirmed `WarmupValidationError` at line 94, in `errors.__all__`
- Direct source read: `src/yowo/engine.py:96` — confirmed `HealthReport` dataclass location
- Direct source read: `tests/unit/test_obb_export.py` — confirmed mock pattern for OBB export tests
- Direct source read: `tests/unit/test_exporter.py` — confirmed general exporter test pattern

### Secondary (MEDIUM confidence)
- `.planning/phases/05-integration-bug-fixes/05-CONTEXT.md` — all implementation decisions from discuss-phase session
- `.planning/STATE.md` — accumulated project decisions including Phase 04 OBBEngine/export decisions

### Tertiary (LOW confidence)
None — all findings verified directly from source.

## Metadata

**Confidence breakdown:**
- INT-P0 fix: HIGH — guard change verified directly from source; exact line confirmed
- INT-P1 fix: HIGH — insertion point confirmed; type-casting pitfall identified and documented
- INT-P2 fix: HIGH — all 5 types confirmed in their source modules; `__all__` membership verified
- Test patterns: HIGH — existing test_obb_export.py patterns directly applicable

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable codebase, 30-day window appropriate)
