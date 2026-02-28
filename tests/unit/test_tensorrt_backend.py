"""Unit tests for yowo.backends._tensorrt.TensorRTBackend.

Covers init validation, load error mapping, infer error mapping, lifecycle,
_parse_device_index, and _build_session_options.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.errors import BackendError, BackendLoadError, DependencyError, InferenceError
from yowo.types import BackendType, PreprocessedTensor


def _make_hw(*, tensorrt_version: str = "10.0", has_nvidia_gpu: bool = True) -> MagicMock:
    hw = MagicMock()
    hw.libraries.tensorrt_version = tensorrt_version
    hw.has_nvidia_gpu = has_nvidia_gpu
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


class TestTensorRTBackendInit:
    def test_raises_dependency_error_without_tensorrt(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw(tensorrt_version="")
        with pytest.raises(DependencyError, match="tensorrt"):
            TensorRTBackend(hw)

    def test_raises_backend_error_without_nvidia_gpu(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw(has_nvidia_gpu=False)
        with pytest.raises(BackendError, match="NVIDIA GPU"):
            TensorRTBackend(hw)

    def test_backend_type_is_tensorrt(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        assert backend.backend_type == BackendType.TENSORRT

    def test_is_loaded_false_after_construction(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        assert backend.is_loaded is False

    def test_input_shape_default(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        assert backend.input_shape == (640, 640)


# ---------------------------------------------------------------------------
# load — error paths
# ---------------------------------------------------------------------------


class TestTensorRTBackendLoad:
    def test_backend_load_error_on_session_failure(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("bad engine")
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0
        mock_ort.SessionOptions.return_value = MagicMock()
        with (
            patch.dict("sys.modules", {"onnxruntime": mock_ort}),
            pytest.raises(BackendLoadError, match="engine load failed"),
        ):
            backend.load("/fake/model.engine")


# ---------------------------------------------------------------------------
# infer — error paths
# ---------------------------------------------------------------------------


class TestTensorRTBackendInfer:
    def test_inference_error_when_not_loaded(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        with pytest.raises(InferenceError, match="no engine loaded"):
            backend.infer(_make_tensor())

    def test_inference_error_wraps_runtime_exception(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        backend._session = MagicMock()
        backend._session.run.side_effect = RuntimeError("ort crash")
        backend._has_kv_io = False
        backend._use_ortvalue = False
        backend._input_name = "images"
        with pytest.raises(InferenceError, match="inference failed"):
            backend.infer(_make_tensor())

    def test_inference_error_re_raises_inference_error(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
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


class TestTensorRTBackendLifecycle:
    def test_unload_sets_is_loaded_false(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        backend._session = MagicMock()
        assert backend.is_loaded is True
        backend.unload()
        assert backend.is_loaded is False

    def test_warmup_noop_when_not_loaded(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        backend.warmup()  # should not raise

    def test_clear_kv_cache_empties_state(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        backend._kv_state = {"past_k": np.zeros(1)}
        backend.clear_kv_cache()
        assert backend._kv_state == {}

    def test_set_source_id_noop(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)
        backend.set_source_id("some-source")  # should not raise


# ---------------------------------------------------------------------------
# _parse_device_index
# ---------------------------------------------------------------------------


class TestParseDeviceIndex:
    def test_auto_returns_zero(self) -> None:
        from yowo.backends._tensorrt import _parse_device_index

        assert _parse_device_index("auto") == 0

    def test_cuda_colon_index(self) -> None:
        from yowo.backends._tensorrt import _parse_device_index

        assert _parse_device_index("cuda:2") == 2

    def test_cuda_no_colon_returns_zero(self) -> None:
        from yowo.backends._tensorrt import _parse_device_index

        assert _parse_device_index("cuda") == 0

    def test_invalid_index_returns_zero(self) -> None:
        from yowo.backends._tensorrt import _parse_device_index

        assert _parse_device_index("cuda:abc") == 0

    def test_gpu_colon_one(self) -> None:
        from yowo.backends._tensorrt import _parse_device_index

        assert _parse_device_index("gpu:1") == 1


# ---------------------------------------------------------------------------
# _build_session_options
# ---------------------------------------------------------------------------


class TestBuildSessionOptions:
    def test_returns_session_options_with_optimization(self) -> None:
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw()
        backend = TensorRTBackend(hw)

        mock_ort = MagicMock()
        mock_opts = MagicMock()
        mock_ort.SessionOptions.return_value = mock_opts
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            opts = backend._build_session_options(mock_ort)

        assert opts.graph_optimization_level == 99
        assert opts.enable_mem_pattern is True
        assert opts.enable_mem_reuse is True
