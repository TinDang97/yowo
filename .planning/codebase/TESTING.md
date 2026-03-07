# Testing Patterns

**Analysis Date:** 2026-03-07

## Test Framework

**Runner:**
- pytest >= 8.0
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest built-in `assert` statements
- `numpy.testing.assert_array_equal` for array comparisons
- `pytest.approx()` for float comparisons

**Additional Plugins:**
- `pytest-asyncio` >= 0.24 — async test support (mode: `auto`)
- `pytest-cov` >= 5.0 — coverage reporting
- `pytest-mock` >= 3.14 — `mocker` fixture (though `unittest.mock` is used directly in most tests)
- `pytest-timeout` >= 2.3 — test timeouts

**Run Commands:**
```bash
uv run pytest tests/unit/ -x -q          # Run all unit tests (fast, no GPU)
uv run pytest tests/unit/ -x -q --tb=short  # Shorter tracebacks
uv run pytest tests/unit/ --cov=src/yowo --cov-report=term-missing  # Coverage
uv run pytest tests/ -m "not slow"       # Skip GPU-dependent tests
uv run pytest tests/integration/         # Integration tests (require weights + hardware)
```

## Test File Organization

**Location:**
- All tests are in `tests/` directory, separate from source code
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Shared test helpers: `tests/unit/_tracking_helpers.py`

**Naming:**
- Test files: `test_<module>.py` — mirrors the source module being tested
- Examples: `test_engine.py` tests `src/yowo/engine.py`, `test_config.py` tests `src/yowo/config.py`
- Some files test cross-cutting concerns: `test_async_api.py`, `test_base_engine.py`, `test_health.py`
- Helper modules: `_tracking_helpers.py` (underscore prefix, not a test file)

**Structure:**
```
tests/
├── conftest.py                    # Effectively empty (1 line)
├── unit/
│   ├── __init__.py
│   ├── _tracking_helpers.py       # Shared factories for tracking tests
│   ├── test_engine.py             # 1327 lines — largest test file
│   ├── test_config.py             # 752 lines
│   ├── test_types.py              # 706 lines
│   ├── test_nms.py                # 699 lines
│   ├── test_tracking.py           # 689 lines
│   ├── ... (64 test files total)
└── integration/
    ├── __init__.py
    ├── conftest.py                # Session fixtures: runner, weights, sample images
    ├── test_cli_e2e.py
    ├── test_engine_integration.py
    └── test_chroma_gallery_persistence.py
```

**Total:** 73 test files, 1615+ tests

## Test Structure

**Suite Organization:**
```python
"""Unit tests for yowo.postprocess._nms."""

from __future__ import annotations

import numpy as np
import pytest

from yowo.postprocess._nms import COCO_CLASSES, _class_aware_nms, postprocess
from yowo.types import BackendType, Detection, Frame, ModelSpec, PreprocessedTensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(height: int = 480, width: int = 640) -> Frame:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0)


def _make_tensor_meta(batch: int, orig_h: int = 480, orig_w: int = 640) -> PreprocessedTensor:
    data = np.zeros((batch, 3, 640, 640), dtype=np.float32)
    return PreprocessedTensor(
        data=data,
        original_shapes=tuple((orig_h, orig_w) for _ in range(batch)),
        input_shape=(640, 640),
        scale_factors=tuple((1.0, 1.0) for _ in range(batch)),
        pad_offsets=tuple((0, 0) for _ in range(batch)),
    )


# ---------------------------------------------------------------------------
# Tests for _class_aware_nms()
# ---------------------------------------------------------------------------


class TestClassAwareNms:
    """Tests for _class_aware_nms using cv2.dnn.NMSBoxes + offset trick."""

    def test_empty_input_returns_empty(self) -> None:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        class_ids = np.empty(0, dtype=np.intp)
        result = _class_aware_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(result) == 0
```

**Patterns:**
- Module docstring describing what is tested
- `from __future__ import annotations` always first
- Helper functions at top, prefixed with underscore: `_make_frame()`, `_make_mock_backend()`, `_make_spec()`
- Tests grouped into classes by feature/concern: `class TestClassAwareNms:`, `class TestLoadConfigFromYaml:`
- Class docstrings explain the group scope
- Test method names: `test_<what>_<expected_behavior>` — e.g. `test_empty_input_returns_empty`, `test_selects_pytorch_on_cpu_only_profile`
- Return type annotation `-> None` on all test methods
- Section separators `# ---------------------------------------------------------------------------` between test groups

## Mocking

**Framework:** `unittest.mock` (MagicMock, patch)

**Patterns:**
```python
from unittest.mock import MagicMock, patch

def _make_mock_backend(
    *,
    backend_type: BackendType = BackendType.PYTORCH,
    load_raises: Exception | None = None,
    infer_output: np.ndarray | None = None,
) -> MagicMock:
    """Return a MagicMock satisfying the InferenceBackend protocol."""
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = backend_type
    mock.is_loaded = False
    mock.input_shape = (640, 640)

    if load_raises is not None:
        mock.load.side_effect = load_raises
    else:
        mock.load.return_value = None

    # Default: YOLO26 NMS-free output (B=1, 0 detections, 6 cols)
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.unload.return_value = None
    return mock
```

**Patching pattern for weight resolution:**
```python
_RESOLVE_PATCH = "yowo.engine.resolve_weights"

def _loaded_engine(mock_backend: MagicMock) -> InferenceEngine:
    engine = InferenceEngine(backend_instance=mock_backend)
    with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
        engine.load()
    return engine
```

**What to Mock:**
- Inference backends: `MagicMock(spec=InferenceBackend)` with controlled `infer()` output
- Weight resolution: `patch("yowo.engine.resolve_weights")`
- Hardware profile: `MagicMock()` for `HardwareProfile` when testing backend selection
- External libraries (torch, onnxruntime, etc.) — never imported in unit tests
- Network calls — session-scoped download in integration conftest only

**What NOT to Mock:**
- Pure computation: NMS, Kalman filter, geometry math, preprocessing
- Data types: `Frame`, `Detection`, `BoundingBox`, `ModelSpec` — instantiated directly
- Configuration: `InferenceConfig`, `ClassificationConfig` — validated directly
- Error hierarchy: real exception classes used in `pytest.raises()`

## Fixtures and Factories

**Test Data:**
```python
# Per-file helper factories (most common pattern — NOT conftest fixtures)
def _make_frame(index: int = 0) -> Frame:
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=index)

def _make_spec(
    family: ModelFamily = ModelFamily.YOLO26,
    size: ModelSize = ModelSize.NANO,
) -> ModelSpec:
    return ModelSpec(family, size)

# Module-level constants for test data
_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
```

**Shared Helper Module:**
- `tests/unit/_tracking_helpers.py` — shared factories used across 6+ tracking test files
- Exports: `make_box()`, `make_detection()`, `make_frame()`, `make_strack()`
- Imported as: `from unit._tracking_helpers import make_box, make_detection`

**Location:**
- Most factories are file-local `_make_*()` functions (not fixtures)
- `tests/conftest.py` is essentially empty (unit tests need no shared fixtures)
- `tests/integration/conftest.py` has session-scoped fixtures: `runner`, `yolo26_weights`, `sample_image_path`
- `pytest.tmp_path` and `pytest.tmp_path_factory` used for filesystem tests

**Pattern: Backend instance injection for unit tests:**
```python
# Engines accept backend_instance= to skip hardware detection in tests
engine = InferenceEngine(backend_instance=mock_backend)
engine = ClassificationEngine(backend_instance=mock_backend, model_family=..., model_size=...)
```

## Coverage

**Requirements:** Not enforced as a gate, but tracked. 1615+ tests as of v2.3.0-dev.

**View Coverage:**
```bash
uv run pytest tests/unit/ --cov=src/yowo --cov-report=term-missing
uv run pytest tests/unit/ --cov=src/yowo --cov-report=html
```

## Test Types

**Unit Tests (`tests/unit/`):**
- 64 test files, ~21,000 lines
- No real hardware, no model downloads, no network calls
- All backends and hardware are mocked
- Run fast: entire suite completes in seconds
- Strict markers enforced: `--strict-markers` in pytest config

**Integration Tests (`tests/integration/`):**
- 3 test files: CLI e2e, engine integration, ChromaDB persistence
- Require real model weights (local `.pt` files)
- May download test images from network
- Use `pytest.skip()` if weights or network unavailable
- Session-scoped fixtures minimize expensive setup

**Custom Markers:**
```python
# Defined in pyproject.toml
markers = [
    "slow: marks tests as slow (require GPU, large models)",
    "integration: marks tests that need full hardware setup",
]
```

## Common Patterns

**Async Testing:**
```python
# pytest-asyncio mode is "auto" — no @pytest.mark.asyncio needed
async def test_adetect_returns_detections(self) -> None:
    """adetect() yields Detection objects."""
    backend = _make_mock_backend()
    engine = _loaded_engine(backend)
    frame = _dummy_frame()
    results = [det async for det in engine.adetect([frame])]
    assert len(results) == 1
    assert isinstance(results[0], Detection)
    engine.close()
```

**Error Testing:**
```python
class TestLoadFailure:
    def test_backend_load_error_propagates(self) -> None:
        mock = _make_mock_backend(load_raises=BackendLoadError("corrupt"))
        engine = InferenceEngine(backend_instance=mock)
        with pytest.raises(BackendLoadError, match="corrupt"):
            with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
                engine.load()
```

**Parametrized Tests:**
```python
class TestLoadConfigFromYaml:
    def test_loads_model_family_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model_family: yolo11\n")
        cfg = load_config(path=cfg_file)
        assert cfg.model_family == ModelFamily.YOLO11
```

**Thread/Concurrency Testing:**
```python
def _wait_for(condition: object, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll condition callable until truthy or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False

# Used in EventBus tests to wait for async delivery
assert _wait_for(lambda: len(received) == 1)
```

**numpy Array Assertions:**
```python
np.testing.assert_array_equal(mean[4:], np.zeros(4))
assert np.trace(cov_pred) > np.trace(cov)
assert mean_pred[0] > mean[0]
```

## Test Naming by Domain

| Domain | Test Files | Key Patterns |
|--------|-----------|--------------|
| Engine lifecycle | `test_engine.py`, `test_base_engine.py`, `test_detection_engine.py` | Mock backend + patch resolve_weights |
| Classification | `test_classification_engine.py`, `test_classify_*.py` (5 files) | Mock backend with (1,1000) output |
| Config/types | `test_config.py`, `test_types.py`, `test_preset.py` | Direct instantiation, YAML tmp files |
| Backends | `test_onnx_backend.py`, `test_coreml_backend.py`, `test_tensorrt_backend.py` | Mock hardware profile |
| I/O | `test_source.py`, `test_reader.py`, `test_sink.py`, `test_decode.py` | tmp_path fixtures |
| Tracking | `test_tracking*.py` (7 files) | Shared `_tracking_helpers.py` |
| Pipeline | `test_pipeline.py`, `test_collector.py`, `test_router.py`, `test_scheduler.py` | Mock frames and detections |
| Counter | `test_counter.py` | Geometry math + mock tracked boxes |
| Export | `test_exporter.py`, `test_int8_export.py`, `test_kv_export.py` | Mock torch and onnx |
| Events | `test_events.py` | Thread polling with `_wait_for()` |
| Async | `test_async_api.py` | `async for` + engine lifecycle |

---

*Testing analysis: 2026-03-07*
