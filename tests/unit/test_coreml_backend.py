"""Tests for CoreML backend.

All tests use mocks — coremltools is NOT required to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends._coreml import CoreMLBackend
from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.types import BackendType, PreprocessedTensor

# ---------------------------------------------------------------------------
# Test factories
# ---------------------------------------------------------------------------


def _make_hw_profile(*, coremltools_version: str | None = "7.0") -> MagicMock:
    """Create a mock HardwareProfile for CoreML tests."""
    hw = MagicMock()
    hw.libraries.coremltools_version = coremltools_version
    return hw


def _make_tensor(batch: int = 1, h: int = 640, w: int = 640) -> PreprocessedTensor:
    """Create a dummy PreprocessedTensor for testing."""
    return PreprocessedTensor(
        data=np.zeros((batch, 3, h, w), dtype=np.float32),
        original_shapes=tuple((h, w) for _ in range(batch)),
        input_shape=(h, w),
        scale_factors=tuple((1.0, 1.0) for _ in range(batch)),
        pad_offsets=tuple((0, 0) for _ in range(batch)),
    )


def _make_mock_coremltools() -> MagicMock:
    """Create a mock coremltools module for sys.modules injection."""
    mock_ct = MagicMock()
    mock_ct.__version__ = "7.0"
    mock_ct.ComputeUnit.ALL = "all"
    return mock_ct


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestCoreMLBackendInit:
    def test_raises_dependency_error_without_coremltools(self) -> None:
        hw = _make_hw_profile(coremltools_version=None)
        with pytest.raises(DependencyError):
            CoreMLBackend(hw)

    def test_init_with_coremltools(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        assert backend.backend_type == BackendType.COREML
        assert not backend.is_loaded
        assert backend.input_shape == (640, 640)


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------


class TestCoreMLBackendLoad:
    def test_load_nonexistent_path_raises(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        mock_ct = _make_mock_coremltools()
        with (
            patch.dict(sys.modules, {"coremltools": mock_ct}),
            pytest.raises(BackendLoadError, match="not found"),
        ):
            backend.load("/nonexistent/model.mlpackage")

    def test_load_success(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.mlpackage"
        model_path.mkdir()  # .mlpackage is a directory

        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        # Mock the MLModel and spec
        mock_ct = _make_mock_coremltools()
        mock_model = MagicMock()
        mock_spec = MagicMock()
        mock_spec.description.input = []
        mock_model.get_spec.return_value = mock_spec
        mock_ct.models.MLModel.return_value = mock_model

        with patch.dict(sys.modules, {"coremltools": mock_ct}):
            backend.load(str(model_path))
            assert backend.is_loaded

    def test_load_model_failure_raises_backend_load_error(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.mlpackage"
        model_path.mkdir()

        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        mock_ct = _make_mock_coremltools()
        mock_ct.models.MLModel.side_effect = RuntimeError("bad model")

        with (
            patch.dict(sys.modules, {"coremltools": mock_ct}),
            pytest.raises(BackendLoadError, match="Failed to load"),
        ):
            backend.load(str(model_path))
        assert not backend.is_loaded

    def test_load_import_error_raises_dependency(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        # Ensure coremltools cannot be imported (not in sys.modules)
        with (
            patch.dict(sys.modules, {"coremltools": None}),
            pytest.raises(DependencyError),
        ):
            backend.load("/some/model.mlpackage")


# ---------------------------------------------------------------------------
# Infer tests
# ---------------------------------------------------------------------------


class TestCoreMLBackendInfer:
    def test_infer_not_loaded_raises(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        with pytest.raises(InferenceError, match="not loaded"):
            backend.infer(_make_tensor())

    def test_infer_success(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        # Simulate a loaded model
        mock_model = MagicMock()
        expected_output = np.ones((1, 84, 8400), dtype=np.float32)
        mock_model.predict.return_value = {"output0": expected_output}
        backend._model = mock_model

        result = backend.infer(_make_tensor())
        assert result.shape == (1, 84, 8400)
        assert result.dtype == np.float32

    def test_infer_fallback_output_key(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        mock_model = MagicMock()
        expected_output = np.ones((1, 84, 8400), dtype=np.float32)
        # Use a different output key — should fallback to first
        mock_model.predict.return_value = {"detection_out": expected_output}
        backend._model = mock_model

        result = backend.infer(_make_tensor())
        assert result.shape == (1, 84, 8400)

    def test_infer_runtime_error_wraps(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("ANE crash")
        backend._model = mock_model

        with pytest.raises(InferenceError, match="CoreML inference failed"):
            backend.infer(_make_tensor())


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestCoreMLBackendLifecycle:
    def test_unload_sets_none(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        backend._model = MagicMock()
        backend.unload()
        assert not backend.is_loaded

    def test_unload_idempotent(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        backend.unload()
        backend.unload()  # should not raise

    def test_clear_kv_cache_noop(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        backend.clear_kv_cache()  # should not raise

    def test_set_source_id_noop(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        backend.set_source_id("test")  # should not raise

    def test_warmup_noop_when_not_loaded(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)
        backend.warmup()  # should not raise

    def test_warmup_runs_inference(self) -> None:
        hw = _make_hw_profile()
        backend = CoreMLBackend(hw)

        mock_model = MagicMock()
        expected_output = np.zeros((1, 84, 8400), dtype=np.float32)
        mock_model.predict.return_value = {"output0": expected_output}
        backend._model = mock_model

        backend.warmup(batch_size=1)
        mock_model.predict.assert_called_once()
