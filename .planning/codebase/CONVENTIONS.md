# Coding Conventions

**Analysis Date:** 2026-03-07

## Naming Patterns

**Files:**
- Private implementation modules use leading underscore: `_nms.py`, `_streaming.py`, `_selector.py`, `_decode.py`
- Public `__init__.py` files re-export from private modules — callers import from the package, never from `_*.py` directly
- One exception: top-level engine files are not underscored: `engine.py`, `classify_engine.py`, `config.py`, `types.py`, `errors.py`

**Functions:**
- Use `snake_case` for all functions and methods: `postprocess()`, `preprocess_into()`, `select_backend()`
- Private helpers use leading underscore: `_class_aware_nms()`, `_resolve_model_meta()`, `_apply_env_overrides()`
- Factory functions: `create_backend()`, `open_source()`, `make_center_line()`, `make_half_zones()`
- Boolean predicates: `is_loaded`, `is_free_threaded()`, `has_detections`

**Variables:**
- `snake_case` for all variables and parameters
- Constants: `UPPER_SNAKE_CASE` — e.g. `COCO_CLASSES`, `IMAGE_EXTS`, `RTSP_SCHEMES`, `EVENT_DETECTION`
- Private module-level constants: `_SENTINEL`, `_SIZE_MAP`, `_FAMILY_MAP`, `_LOCAL_WEIGHTS`
- Type aliases: `_ListenerEntry` (private), `NDArray` (from numpy.typing)

**Types/Classes:**
- `PascalCase` for all classes: `InferenceEngine`, `DetectionEngine`, `ByteTracker`, `FeatureCache`
- Enums: `PascalCase` names, `UPPER_CASE` members — e.g. `BackendType.PYTORCH`, `ModelSize.NANO`
- All enums inherit from `enum.StrEnum` (string-valued) except `TrackState` which uses `enum.IntEnum`
- Protocols: `PascalCase` with descriptive names — `InferenceBackend`, `FrameSource`, `ReIDExtractor`, `GalleryProtocol`, `ModelBuilder`
- Dataclasses: `PascalCase` — `Frame`, `Detection`, `BoundingBox`, `ModelSpec`, `ClassificationResult`
- Test helper classes: leading underscore `_FakeBuilder`, `_MockTrackedBox`

## Code Style

**Formatting:**
- Tool: Ruff (v0.9.10) — configured in `pyproject.toml`
- Line length: 100 characters
- Target version: Python 3.11

**Linting:**
- Tool: Ruff with rule sets: `E`, `W`, `F`, `I`, `UP`, `B`, `SIM`, `RUF`
- Ignored: `B008` (function calls in default arguments — needed for Click)
- Source roots: `src/` and `tests/`

**Type Checking:**
- Tool: Pyright in `strict` mode
- Configured in `pyproject.toml` under `[tool.pyright]`
- Includes `src/` only; `tests/` are excluded
- Suppresses unknown type warnings for untyped external packages (pynvml, onnxruntime, openvino, tqdm)

**Quality Gate Command:**
```bash
uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q
```

## Import Organization

**Order:**
1. `from __future__ import annotations` — ALWAYS first line in every source file (100% adoption across 68+ source files)
2. Standard library imports (`os`, `logging`, `threading`, `pathlib`, `dataclasses`, etc.)
3. Third-party imports (`numpy`, `cv2`, `click`, `yaml`, `torch`, etc.)
4. Local imports from `yowo.*` — always use absolute package paths

**Path Aliases:**
- No path aliases configured (no `paths` in pyproject.toml)
- All imports are absolute: `from yowo.types import Frame`, `from yowo.backends import InferenceBackend`

**Lazy Imports:**
- Heavy optional dependencies are imported inside functions/methods, not at module level
- Pattern: `if TYPE_CHECKING:` block for type-only imports (used in 13 files)
- Example from `src/yowo/cache/__init__.py`:
  ```python
  if TYPE_CHECKING:
      import torch
  ```
- Backend factory `create_backend()` uses `match` + lazy import per branch to avoid loading unused SDKs

## Error Handling

**Exception Hierarchy:**
- All library exceptions inherit from `YowoError` (defined in `src/yowo/errors.py`)
- Each module raises only its own error subtype — never a sibling's type
- Hierarchy is documented in `src/yowo/errors.py` docstring
- Key subtypes: `BackendError` → `BackendLoadError` / `InferenceError` / `ShutdownError`; `ModelError` → `ModelNotFoundError` / `ModelLoadError`

**Patterns:**
- Raise domain-specific errors: `raise ConfigError(...)`, `raise BackendLoadError(...)`
- `DependencyError` provides structured install guidance: `DependencyError("onnxruntime", install_cmd="pip install yowo[onnx]")`
- Validation in `__post_init__` for dataclass configs — raises `ConfigError` on invalid values
- Backend `load()` raises `BackendLoadError`; `infer()` raises `InferenceError`
- Engine tracks cumulative errors and transitions to `DEGRADED` health when `error_threshold` is exceeded

**Error Propagation:**
- No bare `except:` blocks
- External operation failures are caught and wrapped in library error types
- `ShutdownError` is a subclass of `InferenceError` — rejected operations during shutdown

## Logging

**Framework:** Python standard `logging` module

**Patterns:**
- Module-level logger: `logger = logging.getLogger(__name__)` — used in 20+ modules
- Debug-level for cache hits, backend decisions, streaming events
- Info-level for lifecycle events (model loaded, stream started)
- Warning-level for degraded states, reconnections
- No custom formatters or handlers — consumers configure logging externally

## Comments

**When to Comment:**
- Section separators using `# ---------------------------------------------------------------------------` with a title line — used extensively to divide logical sections within files
- Brief inline comments for non-obvious logic (e.g., "Reuse fingerprint from check_and_load() if available")

**Docstrings:**
- Every public class, function, and module has a docstring
- Module docstrings include usage examples with `::` code blocks
- Class docstrings list all `Attributes:` with types and descriptions
- Method docstrings use Google-style `Args:`, `Returns:`, `Raises:` sections
- Exception classes document when they are raised

## Function Design

**Size:** Functions are focused and short. Files are kept under 700 lines (enforced by project rules).

**Parameters:**
- Keyword-only arguments for optional params: `def detect(source, *, model="yolo26n", confidence=0.25)`
- `None` as sentinel for "auto" behavior: `backend: BackendType | None = None`, `precision: Precision | None = None`
- Type annotations on all parameters and return values in production code

**Return Values:**
- Frozen dataclasses for result types: `Detection`, `ClassificationResult`, `ExportResult`, `BackendSelection`
- Tuple of immutable types for collection returns: `tuple[BoundingBox, ...]`, `tuple[float, ...]`
- Iterators/generators for streaming: `Iterator[Detection]`, `AsyncIterator[Detection]`

## Module Design

**Exports:**
- Every package `__init__.py` has an explicit `__all__` list
- Top-level `src/yowo/__init__.py` re-exports the full public API (85+ symbols)
- Sub-packages re-export from private `_*.py` modules

**Barrel Files:**
- All sub-packages use barrel-style `__init__.py` files
- Example from `src/yowo/io/__init__.py`: re-exports `FrameSource`, `open_source`, `preprocess`, etc. from `_source.py`, `_decode.py`, `_sink.py`
- Pattern: docstring explaining exports, then imports, then `__all__`

**Protocol-Based Design:**
- Structural typing via `Protocol` classes (6 protocols across codebase)
- `@runtime_checkable` on public protocols: `InferenceBackend`, `ModelBuilder`, `FrameSource`
- Dependency inversion: engines accept protocol instances, not concrete classes

**Dataclass Conventions:**
- Pure data types use `frozen=True, slots=True`: `ModelSpec`, `BoundingBox`, `Detection`
- Types containing numpy arrays use `slots=True` only (not frozen — numpy arrays are not hashable)
- These are documented as "logically immutable"
- Configuration dataclasses use plain `@dataclass` (mutable for env override layering)

## Enum Conventions

- All enums use `enum.StrEnum` for JSON-safe serialization (except `TrackState` which is `IntEnum`)
- Enum values are lowercase strings matching common CLI/config conventions: `"pytorch"`, `"cuda"`, `"fp16"`
- Centralized in `src/yowo/types.py` (14 enums) with one additional in `src/yowo/counter/_types.py`

## Context Manager Pattern

- Engines implement both sync (`__enter__`/`__exit__`) and async (`__aenter__`/`__aexit__`) context managers
- Return type is `Self` for correct subclass typing
- `close()` is idempotent — safe to call multiple times

## Thread Safety

- `threading.Lock()` protects shared mutable state (caches, metrics, galleries)
- `EventBus` uses a background daemon thread with `SimpleQueue`
- `ThreadedFrameReader` uses a bounded `deque` with a daemon reader thread

---

*Convention analysis: 2026-03-07*
