---
phase: quick
plan: 260317-hyd
subsystem: compat
tags: [python38, jetson, compat, guard-tests]
dependency_graph:
  requires: []
  provides: [py38-compat]
  affects: [pyproject.toml, README.md, src/yowo/types.py, tests/unit/test_py38_compat.py, uv.lock]
tech_stack:
  added: []
  patterns: [future-annotations, StrEnum-backport, ast-guard-tests]
key_files:
  created:
    - tests/unit/test_py38_compat.py
  modified:
    - pyproject.toml
    - README.md
    - src/yowo/types.py
    - uv.lock
decisions:
  - "Keep [tool.uv] environments at >=3.9 (dev deps like pre-commit/scipy don't support 3.8; Jetson Nano is a deployment target not a dev target)"
  - "scipy tracking extra gated to python_version >= '3.9' (scipy>=1.11 doesn't support 3.8)"
  - "Widen future-annotations header search from first-15-lines to whole-file presence check (config.py has 28-line docstring)"
metrics:
  duration: "~10 minutes"
  completed: "2026-03-17"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 5
---

# Quick Task 260317-hyd: Python 3.8 Compatibility Summary

**One-liner:** Lowered `requires-python` to `>=3.8` with py38 ruff target, scipy/tracking version gates, and AST-based guard tests preventing future 3.9+ regressions.

## Tasks Completed

| Task | Name | Commit | Files |
| --- | --- | --- | --- |
| 1 | Add Python 3.8 compatibility guard tests | 15bf13e | tests/unit/test_py38_compat.py |
| 2 | Update configuration files for Python 3.8 minimum | ea5d202 | pyproject.toml, README.md, src/yowo/types.py, uv.lock |

## Changes Made

### pyproject.toml
- `requires-python = ">=3.8"` (was `>=3.9`)
- Added `"Programming Language :: Python :: 3.8"` classifier
- `target-version = "py38"` in `[tool.ruff]` (was `py39`)
- `tracking` optional dep: `scipy>=1.11; python_version >= '3.9'` (scipy has no 3.8-compatible wheel at >=1.11)
- Dev onnxruntime deps consolidated: `<1.20; python_version < '3.11'` covers 3.8, 3.9, and 3.10
- `[tool.uv] environments` kept at `>= '3.9'` (dev toolchain doesn't run on 3.8)

### README.md
- `Python >=3.8` (was `Python >=3.9`)

### src/yowo/types.py
- `StrEnumBase` docstring: `"""Backport of StrEnum for Python < 3.11."""`

### tests/unit/test_py38_compat.py (new)
- 4 test groups, 84 passing + 10 skipped (1994 total unit tests pass)
- Test 1: All non-`__init__.py` files declare `from __future__ import annotations`
- Test 2: No `__init__.py` uses bare `X | Y` union annotations at runtime
- Test 3: `StrEnumBase` backport usable — `str(member)` returns value
- Test 4: All public `yowo` package names importable without errors

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] future-annotations header search window was too narrow**
- **Found during:** Task 1 (test execution)
- **Issue:** Test checked only first 15 lines; `config.py` has a 28-line docstring pushing the import to line 30, causing false failure
- **Fix:** Changed to whole-file presence check (with module-level ordering guard) instead of first-N-lines heuristic
- **Files modified:** tests/unit/test_py38_compat.py

**2. [Rule 2 - Missing] scipy tracking optional dep needed python_version gate**
- **Found during:** Task 2 (pre-commit hook execution)
- **Issue:** `scipy>=1.11` has no Python 3.8 wheels — uv could not resolve dev lockfile for python 3.8 split
- **Fix:** Added `; python_version >= '3.9'` marker to tracking optional dep
- **Files modified:** pyproject.toml

**3. [Rule 2 - Missing] uv environments must stay at >=3.9**
- **Found during:** Task 2 (after scipy fix, pre-commit still failing on pre-commit>=3.8 dev dep)
- **Issue:** dev deps (pre-commit, onnxscript, ultralytics) don't support Python 3.8; lowering the uv environments filter to 3.8 caused unsatisfiable resolution
- **Fix:** Reverted `[tool.uv] environments` back to `>= '3.9'`; production `requires-python = ">=3.8"` still correctly declares the package's support range
- **Files modified:** pyproject.toml

## Self-Check: PASSED

All created files exist on disk. Both task commits verified in git log.
- FOUND: tests/unit/test_py38_compat.py
- FOUND: pyproject.toml (modified)
- FOUND: README.md (modified)
- FOUND: src/yowo/types.py (modified)
- FOUND commit: 15bf13e (Task 1)
- FOUND commit: ea5d202 (Task 2)
