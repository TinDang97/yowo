"""Tests for OrtValue standard path and ORT session optimization."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from yowo.types import PreprocessedTensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor(batch: int = 1) -> PreprocessedTensor:
    """Create a minimal PreprocessedTensor for testing."""
    return PreprocessedTensor(
        data=np.random.rand(batch, 3, 640, 640).astype(np.float32),
        original_shapes=tuple((480, 640) for _ in range(batch)),
        input_shape=(640, 640),
        scale_factors=tuple((0.5, 0.5) for _ in range(batch)),
        pad_offsets=tuple((80, 0) for _ in range(batch)),
    )


def _make_hw_onnx(*, has_coreml: bool = False) -> MagicMock:
    """Build a mock HardwareProfile suitable for OnnxBackend."""
    hw = MagicMock()
    hw.libraries.onnxruntime_version = "1.17.0"
    hw.libraries.onnxruntime_has_cuda = False
    hw.libraries.onnxruntime_has_coreml = has_coreml
    hw.has_nvidia_gpu = False
    return hw


def _make_hw_tensorrt() -> MagicMock:
    """Build a mock HardwareProfile suitable for TensorRTBackend."""
    hw = MagicMock()
    hw.libraries.tensorrt_version = "10.0.1"
    hw.libraries.onnxruntime_version = "1.17.0"
    hw.has_nvidia_gpu = True
    return hw


def _make_ort_session(
    *,
    has_run_with_ort_values: bool,
    providers: list[str] | None = None,
) -> MagicMock:
    """Return a mock ORT InferenceSession.

    Configures get_inputs(), get_outputs(), run(), get_providers(), and
    optionally run_with_ort_values() depending on ``has_run_with_ort_values``.

    ``providers`` controls what ``get_providers()`` returns.  Defaults to
    ``["CPUExecutionProvider"]``.
    """
    session = MagicMock()

    # Minimal single-input / single-output non-KV model
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    session.get_inputs.return_value = [mock_input]

    mock_output = MagicMock()
    mock_output.name = "output0"
    session.get_outputs.return_value = [mock_output]

    # Active execution providers (OrtValue gated to GPU EPs only)
    session.get_providers.return_value = providers or ["CPUExecutionProvider"]

    # Standard run() returns a list with one float32 array
    raw_out = np.zeros((1, 84, 8400), dtype=np.float32)
    session.run.return_value = [raw_out]

    if has_run_with_ort_values:
        ort_out = MagicMock()
        ort_out.numpy.return_value = raw_out
        session.run_with_ort_values.return_value = [ort_out]
    else:
        # Ensure attribute is absent so hasattr() returns False
        del session.run_with_ort_values

    return session


def _make_mock_ort_module(session: MagicMock) -> MagicMock:
    """Build a minimal mock onnxruntime module."""
    ort_mod = MagicMock()
    ort_mod.InferenceSession.return_value = session
    ort_mod.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
    ort_mod.ExecutionMode.ORT_SEQUENTIAL = 0

    mock_opts = MagicMock()
    ort_mod.SessionOptions.return_value = mock_opts

    ortvalue = MagicMock()
    ort_mod.OrtValue.ortvalue_from_numpy.return_value = ortvalue

    return ort_mod


# ---------------------------------------------------------------------------
# Part 1: OrtValue standard path — OnnxBackend
# ---------------------------------------------------------------------------


class TestOnnxStandardOrtValue:
    def test_onnx_standard_ortvalue_dispatched(self) -> None:
        """When run_with_ort_values + CUDA EP, infer() must use OrtValue path."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")
            assert backend._use_ortvalue is True

            tensor = _make_tensor()
            backend.infer(tensor)

        session.run_with_ort_values.assert_called_once()
        session.run.assert_not_called()

    def test_onnx_ortvalue_disabled_on_cpu_ep(self) -> None:
        """OrtValue must NOT activate on CPU-only EPs (no zero-copy benefit)."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CPUExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")
            assert backend._use_ortvalue is False

            tensor = _make_tensor()
            backend.infer(tensor)

        session.run.assert_called_once()

    def test_onnx_ortvalue_disabled_on_coreml_ep(self) -> None:
        """OrtValue must NOT activate on CoreML EP (CPU memory, no copy gain)."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx(has_coreml=True)
        backend = OnnxBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")
            assert backend._use_ortvalue is False

            tensor = _make_tensor()
            backend.infer(tensor)

        session.run.assert_called_once()

    def test_onnx_standard_fallback_without_ortvalue(self) -> None:
        """When run_with_ort_values is absent, infer() must call run()."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(has_run_with_ort_values=False)
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")
            assert backend._use_ortvalue is False

            tensor = _make_tensor()
            backend.infer(tensor)

        session.run.assert_called_once()
        assert not hasattr(session, "run_with_ort_values")

    def test_use_ortvalue_reset_on_reload(self) -> None:
        """_use_ortvalue must reset to False at the start of load()."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        # First load with OrtValue available + CUDA EP
        session_with = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session_with)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")
        assert backend._use_ortvalue is True

        # Second load without OrtValue — must see False
        session_without = _make_ort_session(has_run_with_ort_values=False)
        ort_mod2 = _make_mock_ort_module(session_without)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod2}):
            backend.load("fake.onnx", device="cpu")
        assert backend._use_ortvalue is False

    def test_ortvalue_fn_cached_on_load(self) -> None:
        """_ortvalue_fn must be set to the shared function when OrtValue is active."""
        from yowo.backends._onnx import OnnxBackend
        from yowo.backends._ortvalue import infer_standard_ortvalue

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")

        assert backend._ortvalue_fn is infer_standard_ortvalue

    def test_ortvalue_fn_none_when_disabled(self) -> None:
        """_ortvalue_fn must be None when OrtValue path is not active."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CPUExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")

        assert backend._ortvalue_fn is None

    def test_ortvalue_fn_reset_on_reload(self) -> None:
        """_ortvalue_fn must reset when reloading with different EP."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        # First load with CUDA EP — fn cached
        session_gpu = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ort_mod1 = _make_mock_ort_module(session_gpu)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod1}):
            backend.load("fake.onnx", device="cpu")
        assert backend._ortvalue_fn is not None

        # Reload with CPU-only EP — fn cleared
        session_cpu = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CPUExecutionProvider"],
        )
        ort_mod2 = _make_mock_ort_module(session_cpu)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod2}):
            backend.load("fake.onnx", device="cpu")
        assert backend._ortvalue_fn is None


# ---------------------------------------------------------------------------
# Part 1: OrtValue standard path — TensorRTBackend
# ---------------------------------------------------------------------------


class TestTensorRTStandardOrtValue:
    def test_tensorrt_standard_ortvalue_dispatched(self) -> None:
        """TensorRTBackend: infer() must use run_with_ort_values when available."""
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw_tensorrt()
        backend = TensorRTBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.engine", device="auto")
            assert backend._use_ortvalue is True

            tensor = _make_tensor()
            backend.infer(tensor)

        session.run_with_ort_values.assert_called_once()
        session.run.assert_not_called()

    def test_tensorrt_standard_fallback_without_ortvalue(self) -> None:
        """TensorRTBackend: infer() must call run() when OrtValue unavailable."""
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw_tensorrt()
        backend = TensorRTBackend(hw)

        session = _make_ort_session(
            has_run_with_ort_values=False,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.engine", device="auto")
            assert backend._use_ortvalue is False

            tensor = _make_tensor()
            backend.infer(tensor)

        session.run.assert_called_once()

    def test_tensorrt_use_ortvalue_reset_on_reload(self) -> None:
        """TensorRTBackend: _use_ortvalue resets at load() start."""
        from yowo.backends._tensorrt import TensorRTBackend

        hw = _make_hw_tensorrt()
        backend = TensorRTBackend(hw)

        session_with = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        )
        ort_mod = _make_mock_ort_module(session_with)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.engine", device="auto")
        assert backend._use_ortvalue is True

        session_without = _make_ort_session(
            has_run_with_ort_values=False,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        )
        ort_mod2 = _make_mock_ort_module(session_without)
        with patch.dict(sys.modules, {"onnxruntime": ort_mod2}):
            backend.load("fake.engine", device="auto")
        assert backend._use_ortvalue is False


# ---------------------------------------------------------------------------
# Part 1: float16 output cast
# ---------------------------------------------------------------------------


class TestOrtValueFloat16Cast:
    def test_ortvalue_float16_output_cast(self) -> None:
        """_infer_standard_ortvalue must cast float16 output to float32."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        raw_fp16 = np.zeros((1, 84, 8400), dtype=np.float16)
        session = _make_ort_session(
            has_run_with_ort_values=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ort_out = MagicMock()
        ort_out.numpy.return_value = raw_fp16
        session.run_with_ort_values.return_value = [ort_out]

        ort_mod = _make_mock_ort_module(session)

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")
            tensor = _make_tensor()
            result = backend.infer(tensor)

        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Part 2: Session options
# ---------------------------------------------------------------------------


class TestSessionOptions:
    def _get_opts(self) -> MagicMock:
        """Load OnnxBackend and capture the SessionOptions mock."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(has_run_with_ort_values=False)
        ort_mod = _make_mock_ort_module(session)
        opts_mock = ort_mod.SessionOptions.return_value

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")

        return opts_mock

    def test_session_opts_mem_pattern_enabled(self) -> None:
        """enable_mem_pattern must be set to True on the session options."""
        opts = self._get_opts()
        # Verify the attribute was set (MagicMock records attribute assignments)
        assert opts.enable_mem_pattern is True

    def test_session_opts_mem_reuse_enabled(self) -> None:
        """enable_mem_reuse must be set to True on the session options."""
        opts = self._get_opts()
        assert opts.enable_mem_reuse is True

    def test_session_opts_sequential_mode(self) -> None:
        """execution_mode must be ORT_SEQUENTIAL."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        session = _make_ort_session(has_run_with_ort_values=False)
        ort_mod = _make_mock_ort_module(session)
        sentinel = object()
        ort_mod.ExecutionMode.ORT_SEQUENTIAL = sentinel
        opts_mock = ort_mod.SessionOptions.return_value

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            backend.load("fake.onnx", device="cpu")

        assert opts_mock.execution_mode is sentinel

    def test_build_session_options_directly(self) -> None:
        """_build_session_options sets all three optimization flags."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx()
        backend = OnnxBackend(hw)

        ort_mod = MagicMock()
        ort_mod.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        sentinel_seq = object()
        ort_mod.ExecutionMode.ORT_SEQUENTIAL = sentinel_seq
        opts_mock = MagicMock()
        ort_mod.SessionOptions.return_value = opts_mock

        with patch.dict(sys.modules, {"onnxruntime": ort_mod}):
            result = backend._build_session_options(ort_mod)

        assert result.enable_mem_pattern is True
        assert result.enable_mem_reuse is True
        assert result.execution_mode is sentinel_seq


# ---------------------------------------------------------------------------
# Part 2: CoreML provider options
# ---------------------------------------------------------------------------


class TestCoreMLProviderOptions:
    def test_coreml_provider_options(self) -> None:
        """_select_providers must return CoreML EP as (name, opts) tuple."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx(has_coreml=True)
        backend = OnnxBackend(hw)

        providers = backend._select_providers("auto")

        assert ("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}) in providers

    def test_coreml_provider_includes_cpu_fallback(self) -> None:
        """CoreML provider list must retain CPU EP as fallback."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx(has_coreml=True)
        backend = OnnxBackend(hw)

        providers = backend._select_providers("auto")

        assert "CPUExecutionProvider" in providers

    def test_no_coreml_returns_string_list(self) -> None:
        """Without CoreML, providers must be plain strings (no tuples)."""
        from yowo.backends._onnx import OnnxBackend

        hw = _make_hw_onnx(has_coreml=False)
        backend = OnnxBackend(hw)

        providers = backend._select_providers("auto")

        assert providers == ["CPUExecutionProvider"]
        assert all(isinstance(p, str) for p in providers)


# ---------------------------------------------------------------------------
# Part 3: Shared infer_standard_ortvalue function
# ---------------------------------------------------------------------------


class TestSharedOrtValueFunction:
    def test_infer_standard_ortvalue_calls_run_with_ort_values(self) -> None:
        """Shared function must call run_with_ort_values on the session."""
        from yowo.backends._ortvalue import infer_standard_ortvalue

        tensor = _make_tensor()
        raw_out = np.zeros((1, 84, 8400), dtype=np.float32)

        ort_out = MagicMock()
        ort_out.numpy.return_value = raw_out

        session = MagicMock()
        session.run_with_ort_values.return_value = [ort_out]

        ort_mod = MagicMock()
        ortvalue = MagicMock()
        ort_mod.OrtValue.ortvalue_from_numpy.return_value = ortvalue

        result = infer_standard_ortvalue(
            ort_mod,
            session,
            "images",
            ["output0"],
            tensor,
        )

        ort_mod.OrtValue.ortvalue_from_numpy.assert_called_once_with(tensor.data)
        session.run_with_ort_values.assert_called_once_with(
            ["output0"],
            {"images": ortvalue},
        )
        assert result.dtype == np.float32

    def test_infer_standard_ortvalue_casts_fp16(self) -> None:
        """Shared function must cast float16 output to float32."""
        from yowo.backends._ortvalue import infer_standard_ortvalue

        tensor = _make_tensor()
        raw_fp16 = np.zeros((1, 84, 8400), dtype=np.float16)

        ort_out = MagicMock()
        ort_out.numpy.return_value = raw_fp16

        session = MagicMock()
        session.run_with_ort_values.return_value = [ort_out]

        ort_mod = MagicMock()
        ort_mod.OrtValue.ortvalue_from_numpy.return_value = MagicMock()

        result = infer_standard_ortvalue(
            ort_mod,
            session,
            "images",
            ["output0"],
            tensor,
        )

        assert result.dtype == np.float32
