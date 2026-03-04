"""Unit tests for yowo.backends._selector.

All tests use synthetic HardwareProfile objects — no real hardware required.
Tests cover backend auto-selection, precision selection, and the fallback chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yowo.backends._selector import get_fallback_backends, select_backend, select_precision
from yowo.errors import BackendError, DependencyError
from yowo.hardware import HardwareProfile
from yowo.hardware._capabilities import InstalledLibraries
from yowo.hardware._device import Device
from yowo.types import BackendType, CPUArch, DeviceType, GPUArch, Precision

# ---------------------------------------------------------------------------
# Test profile factory
# ---------------------------------------------------------------------------


def _make_gpu_device(
    *,
    memory_total_mb: int = 8192,
    memory_available_mb: int | None = None,
    is_jetson: bool = False,
    arch: GPUArch = GPUArch.AMPERE,
) -> Device:
    return Device(
        type=DeviceType.CUDA,
        index=0,
        name="NVIDIA Test GPU",
        arch=arch,
        cpu_arch=CPUArch.X86_64,
        memory_total_mb=memory_total_mb,
        memory_available_mb=(
            memory_available_mb if memory_available_mb is not None else memory_total_mb
        ),
        is_jetson=is_jetson,
    )


def _make_cpu_device(
    *,
    memory_total_mb: int = 16384,
    memory_available_mb: int = 16384,
    is_jetson: bool = False,
    cpu_arch: CPUArch = CPUArch.X86_64,
) -> Device:
    return Device(
        type=DeviceType.CPU,
        index=0,
        name="Test CPU",
        arch=None,
        cpu_arch=cpu_arch,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        is_jetson=is_jetson,
    )


def make_profile(
    has_gpu: bool = False,
    has_tensorrt: bool = False,
    has_onnx: bool = False,
    has_onnx_cuda: bool = False,
    has_onnx_coreml: bool = False,
    has_openvino: bool = False,
    has_torch: bool = False,
    has_coremltools: bool = False,
    gpu_memory_mb: int = 8192,
    cpu_memory_mb: int = 16384,
    is_jetson: bool = False,
    cpu_arch: CPUArch = CPUArch.X86_64,
) -> HardwareProfile:
    """Build a synthetic HardwareProfile for testing.

    All parameters default to the minimal configuration (no GPU, nothing installed).
    """
    gpus: tuple[Device, ...] = ()
    if has_gpu:
        gpu = _make_gpu_device(
            memory_total_mb=gpu_memory_mb,
            memory_available_mb=gpu_memory_mb,
            is_jetson=is_jetson,
        )
        gpus = (gpu,)

    cpu = _make_cpu_device(
        memory_total_mb=cpu_memory_mb,
        is_jetson=is_jetson,
        cpu_arch=cpu_arch,
    )

    libs = InstalledLibraries(
        torch_version="2.3.0" if has_torch else None,
        torch_cuda_available=has_gpu and has_torch,
        tensorrt_version="10.0.1" if has_tensorrt else None,
        onnxruntime_version="1.17.0" if has_onnx else None,
        onnxruntime_has_cuda=has_onnx_cuda,
        onnxruntime_has_coreml=has_onnx_coreml,
        openvino_version="2024.0" if has_openvino else None,
        coremltools_version="7.0" if has_coremltools else None,
    )

    return HardwareProfile(
        gpus=gpus,
        cpu=cpu,
        cpu_features=frozenset(),
        libraries=libs,
    )


# ---------------------------------------------------------------------------
# select_backend — auto-selection priority chain
# ---------------------------------------------------------------------------


class TestSelectBackendAutoSelection:
    def test_nvidia_gpu_and_tensorrt_selects_tensorrt(self) -> None:
        profile = make_profile(has_gpu=True, has_tensorrt=True, has_onnx=True, has_torch=True)
        result = select_backend(profile, "n")
        assert result.backend == BackendType.TENSORRT
        assert result.device_type == DeviceType.CUDA

    def test_nvidia_gpu_no_trt_onnx_gpu_selects_onnx(self) -> None:
        profile = make_profile(
            has_gpu=True,
            has_tensorrt=False,
            has_onnx=True,
            has_onnx_cuda=True,
            has_torch=True,
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.ONNX
        assert result.device_type == DeviceType.CUDA

    def test_onnx_coreml_selects_onnx_with_coreml_reason(self) -> None:
        profile = make_profile(has_gpu=False, has_onnx=True, has_onnx_coreml=True)
        result = select_backend(profile, "n")
        assert result.backend == BackendType.ONNX
        assert result.device_type == DeviceType.CPU
        assert "CoreML" in result.reason

    def test_coreml_beats_openvino_in_priority(self) -> None:
        profile = make_profile(
            has_gpu=False, has_onnx=True, has_onnx_coreml=True, has_openvino=True
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.ONNX
        assert "CoreML" in result.reason

    def test_no_gpu_openvino_installed_selects_openvino(self) -> None:
        profile = make_profile(has_gpu=False, has_openvino=True)
        result = select_backend(profile, "n")
        assert result.backend == BackendType.OPENVINO
        assert result.device_type == DeviceType.CPU

    def test_no_gpu_onnx_cpu_selects_onnx(self) -> None:
        profile = make_profile(has_gpu=False, has_onnx=True)
        result = select_backend(profile, "n")
        assert result.backend == BackendType.ONNX
        assert result.device_type == DeviceType.CPU

    def test_only_torch_selects_pytorch(self) -> None:
        profile = make_profile(has_gpu=False, has_torch=True)
        result = select_backend(profile, "n")
        assert result.backend == BackendType.PYTORCH

    def test_nothing_installed_raises_dependency_error(self) -> None:
        profile = make_profile()  # all False
        with pytest.raises(DependencyError) as exc_info:
            select_backend(profile, "n")
        # Message should contain install hints
        assert "uv add" in str(exc_info.value)

    def test_reason_field_is_non_empty(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n")
        assert result.reason != ""

    def test_device_index_is_zero_for_single_gpu(self) -> None:
        profile = make_profile(has_gpu=True, has_tensorrt=True)
        result = select_backend(profile, "n")
        assert result.device_index == 0


# ---------------------------------------------------------------------------
# select_backend — backend_override
# ---------------------------------------------------------------------------


class TestSelectBackendOverride:
    def test_force_onnx_when_only_torch_raises_backend_error(self) -> None:
        profile = make_profile(has_torch=True)
        with pytest.raises(BackendError):
            select_backend(profile, "n", backend_override="onnx")

    def test_force_pytorch_when_torch_installed_succeeds(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", backend_override="pytorch")
        assert result.backend == BackendType.PYTORCH

    def test_invalid_override_name_raises_backend_error(self) -> None:
        profile = make_profile(has_torch=True)
        with pytest.raises(BackendError):
            select_backend(profile, "n", backend_override="banana")

    def test_override_sets_reason_string(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", backend_override="pytorch")
        assert "pytorch" in result.reason.lower()

    def test_device_override_propagates_to_result(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", backend_override="pytorch", device_override="cpu")
        assert result.device_index == 0

    def test_precision_override_propagates_to_result(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", backend_override="pytorch", precision_override="int8")
        assert result.precision == Precision.INT8

    def test_invalid_precision_override_raises_backend_error(self) -> None:
        profile = make_profile(has_torch=True)
        with pytest.raises(BackendError):
            select_backend(profile, "n", backend_override="pytorch", precision_override="fp0")

    def test_tensorrt_override_without_gpu_raises_backend_error(self) -> None:
        profile = make_profile(has_gpu=False, has_tensorrt=True)
        with pytest.raises(BackendError):
            select_backend(profile, "n", backend_override="tensorrt")


# ---------------------------------------------------------------------------
# select_precision
# ---------------------------------------------------------------------------


class TestSelectPrecision:
    def test_8gb_vram_nano_model_selects_fp16(self) -> None:
        """8 GB available, nano needs 300 MB for FP16 -> FP16 fits."""
        profile = make_profile(has_gpu=True, gpu_memory_mb=8192)
        result = select_precision(profile, "n", BackendType.ONNX)
        # Default GPU precision prefers FP16; 8192 >= 300
        assert result == Precision.FP16

    def test_300mb_vram_large_model_degrades_to_int8(self) -> None:
        """300 MB available, large model needs 2400 MB FP16 / 1600 MB INT8.
        Neither FP32 (4096 MB) nor FP16 (2400 MB) fit, falls to INT8 (1600 MB).
        But 300 < 1600 too — so we get INT8 as last resort."""
        profile = make_profile(has_gpu=True, gpu_memory_mb=300)
        result = select_precision(profile, "l", BackendType.ONNX)
        assert result == Precision.INT8

    def test_preferred_fp16_with_sufficient_memory_honored(self) -> None:
        """8 GB available, small model: FP16 threshold is 600 MB -> honored."""
        profile = make_profile(has_gpu=True, gpu_memory_mb=8192)
        result = select_precision(profile, "s", BackendType.ONNX, Precision.FP16)
        assert result == Precision.FP16

    def test_preferred_fp32_honored_when_memory_allows(self) -> None:
        """Requesting FP32 explicitly with plenty of VRAM."""
        profile = make_profile(has_gpu=True, gpu_memory_mb=8192)
        result = select_precision(profile, "n", BackendType.PYTORCH, Precision.FP32)
        assert result == Precision.FP32

    def test_cpu_backend_defaults_to_fp32(self) -> None:
        profile = make_profile(has_gpu=False, has_openvino=True)
        result = select_precision(profile, "n", BackendType.OPENVINO)
        assert result == Precision.FP32

    def test_cpu_backend_explicit_int8_honored(self) -> None:
        profile = make_profile(has_gpu=False, has_openvino=True)
        result = select_precision(profile, "n", BackendType.OPENVINO, Precision.INT8)
        assert result == Precision.INT8

    def test_cpu_backend_with_no_gpu_no_cuda_fp32(self) -> None:
        profile = make_profile(has_gpu=False, has_torch=True)
        result = select_precision(profile, "n", BackendType.PYTORCH)
        assert result == Precision.FP32

    def test_unknown_model_size_uses_default(self) -> None:
        """Unknown size key should not raise; falls back to nano thresholds."""
        profile = make_profile(has_gpu=True, gpu_memory_mb=8192)
        result = select_precision(profile, "unknown_size", BackendType.ONNX)
        assert result in (Precision.FP32, Precision.FP16, Precision.INT8)

    def test_preferred_fp16_with_insufficient_memory_degrades(self) -> None:
        """Prefer FP16 but only 100 MB available -> degrades to INT8."""
        profile = make_profile(has_gpu=True, gpu_memory_mb=100)
        result = select_precision(profile, "s", BackendType.ONNX, Precision.FP16)
        # FP16 needs 600 MB, INT8 needs 400 MB — both exceed 100 MB
        # Falls to INT8 as last resort
        assert result == Precision.INT8


# ---------------------------------------------------------------------------
# get_fallback_backends
# ---------------------------------------------------------------------------


class TestGetFallbackBackends:
    def test_tensorrt_fallback_is_onnx_then_pytorch(self) -> None:
        result = get_fallback_backends(BackendType.TENSORRT)
        assert result == [BackendType.ONNX, BackendType.PYTORCH]

    def test_onnx_fallback_is_pytorch(self) -> None:
        result = get_fallback_backends(BackendType.ONNX)
        assert result == [BackendType.PYTORCH]

    def test_openvino_fallback_is_onnx_then_pytorch(self) -> None:
        result = get_fallback_backends(BackendType.OPENVINO)
        assert result == [BackendType.ONNX, BackendType.PYTORCH]

    def test_pytorch_has_no_fallback(self) -> None:
        result = get_fallback_backends(BackendType.PYTORCH)
        assert result == []

    def test_returns_new_list_each_call(self) -> None:
        """Callers can mutate the result without affecting the internal chain."""
        r1 = get_fallback_backends(BackendType.TENSORRT)
        r2 = get_fallback_backends(BackendType.TENSORRT)
        assert r1 == r2
        r1.clear()
        assert r2 == [BackendType.ONNX, BackendType.PYTORCH]


# ---------------------------------------------------------------------------
# Jetson detection: is_jetson=True with GPU -> TensorRT
# ---------------------------------------------------------------------------


class TestJetsonSelection:
    def test_jetson_gpu_with_tensorrt_selects_tensorrt(self) -> None:
        """Jetson devices have a GPU; TensorRT is the primary backend."""
        profile = make_profile(
            has_gpu=True,
            has_tensorrt=True,
            is_jetson=True,
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.TENSORRT
        assert result.device_type == DeviceType.CUDA

    def test_jetson_gpu_no_tensorrt_selects_onnx_cuda(self) -> None:
        profile = make_profile(
            has_gpu=True,
            has_tensorrt=False,
            has_onnx=True,
            has_onnx_cuda=True,
            is_jetson=True,
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.ONNX
        assert result.device_type == DeviceType.CUDA


# ---------------------------------------------------------------------------
# BackendSelection dataclass sanity
# ---------------------------------------------------------------------------


class TestBackendSelectionDataclass:
    def test_result_is_frozen(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n")
        with pytest.raises((AttributeError, TypeError)):
            result.backend = BackendType.ONNX  # type: ignore[misc]

    def test_result_fields_are_typed_correctly(self) -> None:
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n")
        assert isinstance(result.backend, BackendType)
        assert isinstance(result.device_type, DeviceType)
        assert isinstance(result.precision, Precision)
        assert isinstance(result.device_index, int)
        assert isinstance(result.reason, str)


# ---------------------------------------------------------------------------
# CoreML native backend selection
# ---------------------------------------------------------------------------


class TestCoreMLSelection:
    """Test native CoreML backend auto-selection on Apple Silicon."""

    @patch("yowo.backends._selector._is_macos", return_value=True)
    def test_coreml_native_preferred_on_apple_silicon(self, _mock_macos: MagicMock) -> None:
        profile = make_profile(
            has_coremltools=True,
            has_onnx=True,
            has_onnx_coreml=True,
            cpu_arch=CPUArch.AARCH64,
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.COREML
        assert "CoreML native" in result.reason

    @patch("yowo.backends._selector._is_macos", return_value=True)
    def test_coreml_native_beats_onnx_coreml_ep(self, _mock_macos: MagicMock) -> None:
        """Native CoreML (priority 3) takes precedence over ORT CoreML EP (priority 4)."""
        profile = make_profile(
            has_coremltools=True,
            has_onnx=True,
            has_onnx_coreml=True,
            cpu_arch=CPUArch.AARCH64,
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.COREML

    @patch("yowo.backends._selector._is_macos", return_value=False)
    def test_coreml_not_selected_on_linux(self, _mock_macos: MagicMock) -> None:
        """CoreML should not be selected on non-macOS even with AARCH64."""
        profile = make_profile(
            has_coremltools=True,
            has_onnx=True,
            has_onnx_coreml=True,
            has_torch=True,
            cpu_arch=CPUArch.AARCH64,
        )
        result = select_backend(profile, "n")
        assert result.backend != BackendType.COREML

    @patch("yowo.backends._selector._is_macos", return_value=True)
    def test_coreml_not_selected_on_x86(self, _mock_macos: MagicMock) -> None:
        """CoreML should not be selected on Intel Mac (x86_64)."""
        profile = make_profile(
            has_coremltools=True,
            has_onnx=True,
            has_onnx_coreml=True,
            cpu_arch=CPUArch.X86_64,
        )
        result = select_backend(profile, "n")
        # Should fall through to ORT CoreML EP or ONNX, not native CoreML
        assert result.backend != BackendType.COREML

    @patch("yowo.backends._selector._is_macos", return_value=True)
    def test_nvidia_gpu_still_beats_coreml(self, _mock_macos: MagicMock) -> None:
        """TensorRT/CUDA backends remain priority 1/2 even on macOS."""
        profile = make_profile(
            has_gpu=True,
            has_tensorrt=True,
            has_torch=True,
            has_coremltools=True,
            cpu_arch=CPUArch.AARCH64,
        )
        result = select_backend(profile, "n")
        assert result.backend == BackendType.TENSORRT

    def test_coreml_fallback_chain(self) -> None:
        chain = get_fallback_backends(BackendType.COREML)
        assert chain == [BackendType.ONNX, BackendType.PYTORCH]

    def test_coreml_override_without_coremltools_raises(self) -> None:
        profile = make_profile(has_torch=True)
        with pytest.raises(BackendError, match="coremltools"):
            select_backend(profile, "n", backend_override="coreml")

    @patch("yowo.backends._selector._is_macos", return_value=True)
    def test_coreml_override_with_coremltools_succeeds(self, _mock_macos: MagicMock) -> None:
        profile = make_profile(has_coremltools=True, has_torch=True)
        result = select_backend(profile, "n", backend_override="coreml")
        assert result.backend == BackendType.COREML
        assert "User override" in result.reason


# ---------------------------------------------------------------------------
# MPS device type handling
# ---------------------------------------------------------------------------


class TestMPSDeviceType:
    """Tests for DeviceType.MPS resolution in both override and auto paths."""

    def test_mps_device_type_with_backend_override(self) -> None:
        """Explicit backend override + device='mps' → DeviceType.MPS."""
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", backend_override="pytorch", device_override="mps")
        assert result.device_type == DeviceType.MPS

    def test_mps_device_type_in_auto_select(self) -> None:
        """Auto-selected backend + device='mps' → DeviceType.MPS."""
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", device_override="mps")
        assert result.device_type == DeviceType.MPS

    def test_non_mps_device_stays_cpu(self) -> None:
        """Auto-selected backend + device='cpu' → DeviceType.CPU (not MPS)."""
        profile = make_profile(has_torch=True)
        result = select_backend(profile, "n", device_override="cpu")
        assert result.device_type == DeviceType.CPU
