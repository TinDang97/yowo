---
phase: quick
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - src/yowo/types.py
  - src/yowo/counter/_types.py
  - src/yowo/engine.py
  - src/yowo/config.py
  - src/yowo/models/_registry.py
  - src/yowo/metrics/_collector.py
  - src/yowo/io/_reader.py
  - src/yowo/io/_decode.py
  - src/yowo/arch/_config.py
  - src/yowo/tracking/_strack.py
  - src/yowo/tracking/_cross_camera.py
  - src/yowo/tracking/_gallery.py
  - src/yowo/tracking/_camera_link.py
  - src/yowo/hardware/_capabilities.py
  - src/yowo/hardware/__init__.py
  - src/yowo/hardware/_device.py
  - src/yowo/cache/_store.py
  - src/yowo/export/_metadata.py
  - src/yowo/export/_exporter.py
  - src/yowo/tune/_sweep.py
  - src/yowo/backends/__init__.py
  - src/yowo/backends/_selector.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "yowo installs and imports cleanly on Python 3.9"
    - "yowo installs and imports cleanly on Python 3.10"
    - "All 1615+ existing tests pass on Python 3.11+ unchanged"
    - "ONNX GPU and TensorRT backends remain functional"
  artifacts:
    - path: "pyproject.toml"
      provides: "requires-python >= 3.9, updated classifiers"
      contains: "requires-python"
    - path: "src/yowo/types.py"
      provides: "StrEnum polyfill, no slots=True on 3.9"
  key_links:
    - from: "src/yowo/types.py"
      to: "all modules importing enums"
      via: "StrEnum compat shim"
      pattern: "StrEnum"
---

<objective>
Backport yowo to support Python 3.9 and 3.10 for ONNX GPU / TensorRT / Jetson compatibility.

Purpose: Jetson devices (JetPack 5.x) ship Python 3.8-3.10. TensorRT and onnxruntime-gpu wheels for Jetson require Python 3.9 or 3.10. Current codebase requires 3.11+ due to StrEnum, match/case, dataclass slots=True, and typing.Self usage.

Output: A fully backward-compatible codebase that runs on Python 3.9+ while preserving all existing functionality on 3.11+.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@pyproject.toml
@src/yowo/types.py
@src/yowo/engine.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add StrEnum polyfill and replace match/case with if/elif</name>
  <files>
    src/yowo/types.py
    src/yowo/counter/_types.py
    src/yowo/engine.py
    src/yowo/export/_exporter.py
    src/yowo/tune/_sweep.py
    src/yowo/backends/__init__.py
    src/yowo/backends/_selector.py
  </files>
  <action>
1. In `src/yowo/types.py`, add a StrEnum compatibility shim at the top (after imports):

```python
import sys
if sys.version_info >= (3, 11):
    _StrEnum = enum.StrEnum
else:
    class _StrEnum(str, enum.Enum):
        """Backport of StrEnum for Python 3.9/3.10."""
        pass
```

Replace all `enum.StrEnum` base classes with `_StrEnum` (14 enum classes in types.py, 1 in counter/_types.py). For `counter/_types.py`, import `_StrEnum` from `yowo.types` or duplicate the shim locally.

2. In `src/yowo/engine.py`, replace `from typing import TYPE_CHECKING, Any, Self` with:
```python
from typing import TYPE_CHECKING, Any
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
```
Add `typing-extensions>=4.0; python_version < "3.11"` to dependencies in pyproject.toml OR use `typing_extensions` unconditionally (simpler). Since typing_extensions is lightweight and widely used, prefer unconditional: `from typing_extensions import Self`.

Actually, check if Self is only used in type annotations with `from __future__ import annotations`. If so, it is string-only and never evaluated at runtime on 3.9. In that case, guard the import under `TYPE_CHECKING`:
```python
if TYPE_CHECKING:
    from typing import Self
```
This avoids any runtime dependency. Verify engine.py already has `from __future__ import annotations` (it does per grep results).

3. Replace all `match/case` statements with `if/elif/else` chains in 4 files:
   - `src/yowo/export/_exporter.py` (1 match block)
   - `src/yowo/tune/_sweep.py` (1 match block)
   - `src/yowo/backends/__init__.py` (1 match block)
   - `src/yowo/backends/_selector.py` (1 match block)

   Pattern: `match x: case Foo.BAR: ...` becomes `if x == Foo.BAR: ... elif x == Foo.BAZ: ...`
   For the `case _:` wildcard, use `else:`.
   For the OR pattern `case BackendType.PYTORCH | BackendType.ONNX:` in _sweep.py, use `if backend in (BackendType.PYTORCH, BackendType.ONNX):`.

Do NOT change any logic, only syntax. Preserve all comments and error messages.
  </action>
  <verify>
    <automated>cd /Users/tindang/workspaces/tind-repo/yowo && uv run python -c "from yowo.types import BackendType, DeviceType, ModelFamily; print('enum import OK')" && uv run python -c "from yowo.engine import DetectionEngine; print('engine import OK')"</automated>
  </verify>
  <done>All StrEnum classes use compat shim. All match/case replaced with if/elif. Self import guarded under TYPE_CHECKING. Imports succeed.</done>
</task>

<task type="auto">
  <name>Task 2: Remove dataclass slots=True for 3.9 compat and update pyproject.toml</name>
  <files>
    pyproject.toml
    src/yowo/types.py
    src/yowo/engine.py
    src/yowo/config.py
    src/yowo/models/_registry.py
    src/yowo/metrics/_collector.py
    src/yowo/io/_reader.py
    src/yowo/io/_decode.py
    src/yowo/arch/_config.py
    src/yowo/tracking/_strack.py
    src/yowo/tracking/_cross_camera.py
    src/yowo/tracking/_gallery.py
    src/yowo/tracking/_camera_link.py
    src/yowo/hardware/_capabilities.py
    src/yowo/hardware/__init__.py
    src/yowo/hardware/_device.py
    src/yowo/cache/_store.py
    src/yowo/export/_metadata.py
    src/yowo/counter/_types.py
  </files>
  <action>
1. In all 18 files containing `slots=True`, remove the `slots=True` parameter from `@dataclass(...)` decorators:
   - `@dataclass(frozen=True, slots=True)` becomes `@dataclass(frozen=True)`
   - `@dataclass(slots=True)` becomes `@dataclass()`  (or just `@dataclass`)

   `slots=True` was added in Python 3.10. Removing it is safe -- it only affects memory layout (marginal savings). Frozen remains, preserving immutability.

   Files (18 total): types.py, engine.py, config.py, models/_registry.py, metrics/_collector.py, io/_reader.py, io/_decode.py, arch/_config.py, tracking/_strack.py, tracking/_cross_camera.py, tracking/_gallery.py, tracking/_camera_link.py, counter/_types.py, hardware/_capabilities.py, hardware/__init__.py, hardware/_device.py, cache/_store.py, export/_metadata.py.

2. Update `pyproject.toml`:
   - Change `requires-python = ">=3.11"` to `requires-python = ">=3.9"`
   - Change `target-version = "py311"` (ruff) to `target-version = "py39"`
   - Change `pythonVersion = "3.11"` (pyright) to `pythonVersion = "3.9"`
   - Add classifiers: `"Programming Language :: Python :: 3.9"`, `"Programming Language :: Python :: 3.10"`
   - If `typing-extensions` is needed as runtime dep (check Task 1 outcome -- if Self is TYPE_CHECKING only, skip this), add `"typing-extensions>=4.0; python_version < '3.11'"` to dependencies.

3. Verify ruff doesn't flag any remaining 3.10+ syntax by running `uv run ruff check src/ --select UP`.
  </action>
  <verify>
    <automated>cd /Users/tindang/workspaces/tind-repo/yowo && uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q --timeout=120</automated>
  </verify>
  <done>pyproject.toml targets Python 3.9+. All dataclass slots=True removed. Ruff, pyright, and all unit tests pass. No 3.10+ syntax remains in source.</done>
</task>

</tasks>

<verification>
1. `uv run ruff check src/ tests/ --quiet` -- no lint errors
2. `uv run pyright src/yowo/` -- no type errors
3. `uv run pytest tests/unit/ -x -q` -- all 1615+ tests pass
4. `uv run python -c "import yowo; print(yowo.__version__)"` -- imports cleanly
5. Grep confirms no remaining `StrEnum`, `match `, `slots=True`, or unguarded `Self` in src/
</verification>

<success_criteria>
- pyproject.toml requires-python is ">=3.9"
- Zero instances of enum.StrEnum, match/case, dataclass(slots=True) in src/
- typing.Self is guarded under TYPE_CHECKING (no runtime import)
- All existing tests pass without modification
- Ruff + pyright clean
</success_criteria>

<output>
After completion, create `.planning/quick/1-enhance-yowo-to-support-python-3-9-and-3/1-SUMMARY.md`
</output>
