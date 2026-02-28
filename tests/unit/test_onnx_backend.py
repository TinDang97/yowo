"""Unit tests for yowo.backends._onnx.OnnxBackend.

Covers init validation, load error mapping, infer error mapping, and
lifecycle properties. OrtValue path is tested in test_ortvalue.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.types import BackendType, PreprocessedTensor


def _make_hw(*, onnxruntime_version: str = "1.17.0") -> MagicMock:
    hw = MagicMock()
    hw.libraries.onnxruntime_version = onnxruntime_version
    hw.libraries.onnxruntime_has_cuda = False
    hw.libraries.onnxruntime_has_coreml = False
    hw.has_nvidia_gpu = False
    return hw


def _make_tensor() -> PreprocessedTensor:
    return PreprocessedTensor(
        data=np.zeros((1, 3, 640, 640), dtype=np.float32),
        original_shapes=((480, 640),),
        input_shape=(640, 640),
        scale_factors=((1.0, 1.0),),
        pad_offsets=((0, 0),),
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestOnnxBackendInit:
    def test_raises_dependency_error_without_onnxruntime(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw(onnxruntime_version="")
        with pytest.raises(DependencyError, match="onnxruntime"):
            OnnxBackend(hw)

    def test_backend_type_is_onnx(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        assert backend.backend_type == BackendType.ONNX

    def test_is_loaded_false_after_construction(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        assert backend.is_loaded is False

    def test_input_shape_default(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        assert backend.input_shape == (640, 640)


# ---------------------------------------------------------------------------
# load — error paths
# ---------------------------------------------------------------------------


class TestOnnxBackendLoad:
    def test_backend_load_error_on_session_failure(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("bad model")
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0
        mock_ort.SessionOptions.return_value = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}), pytest.raises(BackendLoadError):
            backend.load("/fake/model.onnx")


# ---------------------------------------------------------------------------
# infer — error paths
# ---------------------------------------------------------------------------


class TestOnnxBackendInfer:
    def test_inference_error_when_not_loaded(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        with pytest.raises(InferenceError, match="no session loaded"):
            backend.infer(_make_tensor())

    def test_inference_error_wraps_runtime_exception(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        backend._session = MagicMock()
        backend._session.run.side_effect = RuntimeError("ort crash")
        backend._has_kv_io = False
        backend._use_ortvalue = False
        backend._input_name = "images"
        with pytest.raises(InferenceError, match="inference failed"):
            backend.infer(_make_tensor())

    def test_inference_error_re_raises_inference_error(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        backend._session = MagicMock()
        backend._has_kv_io = False
        backend._use_ortvalue = False
        backend._input_name = "images"
        backend._session.run.side_effect = InferenceError("original")
        with pytest.raises(InferenceError, match="original"):
            backend.infer(_make_tensor())


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


class TestOnnxBackendLifecycle:
    def test_unload_sets_is_loaded_false(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        backend._session = MagicMock()
        assert backend.is_loaded is True
        backend.unload()
        assert backend.is_loaded is False

    def test_warmup_noop_when_not_loaded(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        backend.warmup()  # should not raise

    def test_clear_kv_cache_empties_state(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        backend._kv_state = {"past_k": np.zeros(1)}
        backend.clear_kv_cache()
        assert backend._kv_state == {}

    def test_set_source_id_noop(self) -> None:
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw()
        backend = OnnxBackend(hw)
        backend.set_source_id("some-source")  # should not raise
