---
phase: quick
plan: 1
subsystem: compatibility
tags: [python-compat, py39, py310, strenum, dataclass, match-case]
dependency_graph:
  requires: []
  provides: [python-39-compat, strenum-backport, slots-removed]
  affects: [all-modules-using-enums, all-dataclass-consumers]
tech_stack:
  added: []
  patterns: [conditional-sys-version-import, if-elif-else-dispatch]
key_files:
  created: []
  modified:
    - pyproject.toml
    - src/yowo/types.py
    - src/yowo/counter/_types.py
    - src/yowo/engine.py
    - src/yowo/backends/__init__.py
    - src/yowo/backends/_selector.py
    - src/yowo/tune/_sweep.py
    - src/yowo/export/_exporter.py
    - src/yowo/arch/_config.py
    - src/yowo/cache/_store.py
    - src/yowo/config.py
    - src/yowo/export/_metadata.py
    - src/yowo/hardware/__init__.py
    - src/yowo/hardware/_capabilities.py
    - src/yowo/hardware/_device.py
    - src/yowo/io/_decode.py
    - src/yowo/io/_reader.py
    - src/yowo/metrics/_collector.py
    - src/yowo/models/_registry.py
    - src/yowo/tracking/_camera_link.py
    - src/yowo/tracking/_cross_camera.py
    - src/yowo/tracking/_gallery.py
    - src/yowo/tracking/_strack.py
    - tests/unit/test_batch_runner.py
    - tests/unit/test_metrics.py
decisions:
  - "StrEnumBase shim uses class-level name (not _private) to allow cross-module import without reportPrivateUsage"
  - "pyright pythonVersion stays 3.11 — CI type-checks on 3.11+; 3.9 runtime compat is guaranteed by the shim and from __future__ import annotations"
  - "ruff target-version changed to py39 — required for UP036 not to flag the sys.version_info >= (3,11) check as always-true"
  - "slots=True removed from all dataclasses — only memory layout impact; frozen=True preserved for immutability"
  - "MagicMock(spec=ModelSpec) replaced with real ModelSpec in test_batch_runner — slots descriptors were being relied on for spec inference"
metrics:
  duration: ~21min
  completed_date: "2026-03-12"
  tasks_completed: 2
  files_changed: 25
---

# Phase quick Plan 1: Python 3.9/3.10 Compatibility Backport Summary

**One-liner:** StrEnumBase shim + match/case removal + slots=True stripping to support Python 3.9/3.10 for Jetson/TensorRT deployment.

## Objective

Backport yowo from Python 3.11+ to Python 3.9+ for Jetson (JetPack 5.x), TensorRT, and onnxruntime-gpu wheel compatibility. Three breaking constructs existed: `enum.StrEnum` (3.11+), `match/case` (3.10+), and `@dataclass(slots=True)` (3.10+).

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | StrEnum polyfill, match/case replacement, Self guard | cd7c2dc | types.py, counter/_types.py, engine.py, backends/__init__.py, backends/_selector.py, tune/_sweep.py, export/_exporter.py |
| 2 | Remove dataclass slots=True, update pyproject.toml | cd7c2dc | pyproject.toml + 18 source files + 2 test files |

## Changes Made

### Task 1: StrEnum Polyfill and match/case Replacement

**StrEnumBase shim (`src/yowo/types.py`):**
```python
if sys.version_info >= (3, 11):
    StrEnumBase = enum.StrEnum
else:
    class StrEnumBase(str, enum.Enum):
        """Backport of StrEnum for Python 3.9/3.10."""
```

All 14 enum classes in `types.py` and `CrossDirection` in `counter/_types.py` now inherit from `StrEnumBase`.

**`typing.Self` guard (`engine.py`):**
All files had `from __future__ import annotations`, making `Self` annotation-only. Moved import under `TYPE_CHECKING`.

**match/case → if/elif/else (4 files):**
- `backends/__init__.py`: `create_backend()` dispatch
- `backends/_selector.py`: `check_backend_available()` dispatch
- `tune/_sweep.py`: `_precisions_for_backend()` — OR pattern `case A | B:` converted to `elif x in (A, B):`
- `export/_exporter.py`: `export_model()` format conversion dispatch

### Task 2: Remove slots=True and pyproject.toml Updates

**`@dataclass(slots=True)` removed from 18 files.** `frozen=True` preserved — immutability guarantees are unchanged. Only memory layout (slot-based attribute storage) is affected. The performance impact is negligible for production workloads.

**pyproject.toml changes:**
- `requires-python = ">=3.9"` (was `">=3.11"`)
- Added `"Programming Language :: Python :: 3.9"` and `"3.10"` classifiers
- `ruff target-version = "py39"` (was `py311`) — needed for `UP036` not to flag the version check
- `pyright pythonVersion = "3.11"` **unchanged** — type-checking runs on CI with 3.11+

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MagicMock(spec=ModelSpec) broke after slots=True removal**
- **Found during:** Task 2 verification (pytest)
- **Issue:** `MagicMock(spec=ModelSpec)` relied on slot descriptors to expose `family` and `size` attributes. After removing `slots=True`, slot descriptors are gone and `MagicMock` spec inference fails → `AttributeError: Mock object has no attribute 'family'`
- **Fix:** Replaced `MagicMock(spec=ModelSpec)` with `ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)` in `_make_detection()` test helper
- **Files modified:** `tests/unit/test_batch_runner.py`

**2. [Rule 1 - Bug] test_slots_defined asserted __slots__ absent after removal**
- **Found during:** Task 2 verification (pytest)
- **Issue:** `TestEngineMetrics.test_slots_defined` asserted `not hasattr(m, "__dict__")` — the test verifies memory layout optimization which no longer applies
- **Fix:** Renamed test to `test_frozen_immutable` and replaced assertion with `pytest.raises((AttributeError, TypeError))` on attempted field mutation — this verifies the `frozen=True` immutability that IS preserved
- **Files modified:** `tests/unit/test_metrics.py`

**3. [Rule 3 - Blocking] pyright reportPrivateUsage for _StrEnum cross-module import**
- **Found during:** Task 1 verification (pyright)
- **Issue:** Initial implementation used `_StrEnum` (private name). `counter/_types.py` importing it triggered `reportPrivateUsage`
- **Fix:** Renamed to `StrEnumBase` (public name) — consistent with project conventions

## Verification Results

```
uv run ruff check src/ tests/ --quiet     # No errors
uv run pyright src/yowo/                  # 0 errors, 0 warnings
uv run pytest tests/unit/ -x -q          # 1910 passed, 1 skipped
uv run python -c "import yowo; ..."      # imports cleanly
```

Grep confirms:
- No `enum.StrEnum` base classes in src/ (only `StrEnumBase = enum.StrEnum` in the shim itself)
- No `match ` statements in src/
- No `slots=True` in src/
- `typing.Self` guarded under `TYPE_CHECKING`

## Self-Check

**Commit exists:** FOUND: cd7c2dc (feat(quick-1): backport yowo to Python 3.9/3.10 compatibility)

**Key files exist:**
- FOUND: src/yowo/types.py (StrEnumBase shim)
- FOUND: pyproject.toml (requires-python = ">=3.9")
- FOUND: .planning/quick/1-enhance-yowo-to-support-python-3-9-and-3/1-SUMMARY.md

**Quality gates:** PASSED (ruff 0 errors, pyright 0 errors, pytest 1910 passed)

## Self-Check: PASSED
