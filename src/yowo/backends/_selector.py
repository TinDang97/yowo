"""Backend auto-selection.

Determines the optimal backend and precision for the current hardware.
All selection logic is pure functions — no side effects, fully testable.

Priority chain (auto, evaluated top-to-bottom):
  1. NVIDIA GPU + TensorRT installed  -> TensorRT (FP16)
  2. NVIDIA GPU + ONNX-CUDA EP        -> ONNX with CUDAExecutionProvider (FP16)
  3. Apple Silicon + coremltools       -> CoreML native (Neural Engine)
  4. ONNX + CoreML EP                 -> ONNX with CoreMLExecutionProvider
  5. OpenVINO installed               -> OpenVINO (FP32)
  6. ONNX Runtime installed           -> ONNX with CPUExecutionProvider (FP32)
  7. PyTorch installed                -> PyTorch (FP32)
  8. Nothing available                -> raises DependencyError
"""

from __future__ import annotations

import platform

from yowo.errors import BackendError, DependencyError
from yowo.hardware import HardwareProfile
from yowo.types import BackendSelection, BackendType, CPUArch, DeviceType, Precision

__all__ = [
    "get_fallback_backends",
    "select_backend",
    "select_precision",
]

# ---------------------------------------------------------------------------
# VRAM thresholds (MB) required per model size and precision.
# Values are conservative estimates for YOLO detection models.
# ---------------------------------------------------------------------------

_VRAM_THRESHOLDS: dict[str, dict[Precision, int]] = {
    "n": {Precision.FP32: 512, Precision.FP16: 300, Precision.INT8: 200},
    "s": {Precision.FP32: 1024, Precision.FP16: 600, Precision.INT8: 400},
    "m": {Precision.FP32: 2048, Precision.FP16: 1200, Precision.INT8: 800},
    "l": {Precision.FP32: 4096, Precision.FP16: 2400, Precision.INT8: 1600},
    "x": {Precision.FP32: 8192, Precision.FP16: 4800, Precision.INT8: 3200},
}

# Default model size when key not present in thresholds table.
_DEFAULT_MODEL_SIZE = "n"

# ---------------------------------------------------------------------------
# Fallback chain: if primary backend fails at load time, try alternatives.
# ---------------------------------------------------------------------------

_FALLBACK_CHAIN: dict[BackendType, list[BackendType]] = {
    BackendType.TENSORRT: [BackendType.ONNX, BackendType.PYTORCH],
    BackendType.ONNX: [BackendType.PYTORCH],
    BackendType.OPENVINO: [BackendType.ONNX, BackendType.PYTORCH],
    BackendType.COREML: [BackendType.ONNX, BackendType.PYTORCH],
    BackendType.PYTORCH: [],
}

# ---------------------------------------------------------------------------
# Install hint messages for DependencyError
# ---------------------------------------------------------------------------

_INSTALL_HINTS = (
    "Install at least one inference backend:\n"
    "  PyTorch (fallback):  uv add yowo[pytorch]\n"
    "  ONNX (CPU):          uv add yowo[onnx]\n"
    "  ONNX (CUDA):         uv add yowo[onnx-gpu]\n"
    "  TensorRT:            uv add tensorrt\n"
    "  OpenVINO:            uv add yowo[openvino]\n"
    "  CoreML (macOS):      uv add yowo[coreml]"
)


def _is_macos() -> bool:
    """Return True when running on macOS."""
    return platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_backend(
    hw: HardwareProfile,
    model_size: str = "n",
    *,
    backend_override: str | None = None,
    device_override: str | None = None,
    precision_override: str | None = None,
) -> BackendSelection:
    """Select the optimal backend for the given hardware profile.

    When ``backend_override`` is provided the priority chain is bypassed but
    the selected backend is still validated for availability on this hardware.

    Args:
        hw: Immutable hardware snapshot to base decisions on.
        model_size: YOLO size variant key ("n","s","m","l","x").
        backend_override: Force a specific backend by string name.
        device_override: Override the resolved device string (e.g. "cuda:1").
        precision_override: Override the precision ("fp32", "fp16", "int8").

    Returns:
        ``BackendSelection`` with the resolved backend, device, and precision.

    Raises:
        DependencyError: No usable backend is installed.
        BackendError: ``backend_override`` specified but that backend is not
            available on this hardware.
    """
    safe_size = model_size if model_size in _VRAM_THRESHOLDS else _DEFAULT_MODEL_SIZE

    # --- Override path ---
    if backend_override is not None:
        return _resolve_override(
            hw, safe_size, backend_override, device_override, precision_override
        )

    # --- Auto-selection priority chain ---
    selected_backend, device_type, device_str, reason = _run_priority_chain(hw)

    preferred_precision = _parse_precision(precision_override)
    precision = select_precision(hw, safe_size, selected_backend, preferred_precision)

    if device_override is not None:
        device_str = device_override
        if device_override.startswith("mps"):
            device_type = DeviceType.MPS

    device_index = _parse_device_index(device_str)

    return BackendSelection(
        backend=selected_backend,
        device_type=device_type,
        precision=precision,
        device_index=device_index,
        reason=reason,
    )


def select_precision(
    hw: HardwareProfile,
    model_size: str,
    backend: BackendType,
    preferred: Precision | None = None,
) -> Precision:
    """Choose the best precision given hardware capabilities and memory.

    For GPU backends: check available VRAM against per-size thresholds.
    Degrades FP32 -> FP16 -> INT8 until one fits, or raises if none fits.

    For CPU backends: FP32 default. INT8 only if explicitly requested.

    Args:
        hw: Hardware profile for VRAM and device-capability checks.
        model_size: Size key for threshold lookup ("n","s","m","l","x").
        backend: Backend type (GPU vs CPU affects default precision).
        preferred: Caller-requested precision; honored if memory allows.

    Returns:
        Selected ``Precision`` value.
    """
    safe_size = model_size if model_size in _VRAM_THRESHOLDS else _DEFAULT_MODEL_SIZE
    thresholds = _VRAM_THRESHOLDS[safe_size]

    gpu = hw.primary_gpu
    # OpenVINO can target an Intel iGPU; treat it as a GPU backend only when
    # a non-NVIDIA GPU is present (hw.primary_gpu is set but NOT has_nvidia_gpu).
    is_gpu_backend = backend in (BackendType.PYTORCH, BackendType.ONNX, BackendType.TENSORRT) or (
        backend == BackendType.OPENVINO and gpu is not None and not hw.has_nvidia_gpu
    )

    if (
        gpu is None
        or not is_gpu_backend
        or (backend != BackendType.OPENVINO and not hw.has_nvidia_gpu)
    ):
        # CPU path: FP32 default unless INT8 explicitly requested
        if preferred == Precision.INT8:
            return Precision.INT8
        return Precision.FP32

    available_vram = gpu.memory_available_mb

    # Build candidate list starting from preferred, then degrades
    if preferred is not None:
        candidates = _degrade_from(preferred)
    else:
        # Default GPU precision: try FP16 first (best perf/accuracy tradeoff)
        candidates = [Precision.FP16, Precision.FP32, Precision.INT8]

    for precision in candidates:
        required = thresholds[precision]
        if available_vram >= required:
            return precision

    # Nothing fits — use INT8 as last resort
    return Precision.INT8


def get_fallback_backends(primary: BackendType) -> list[BackendType]:
    """Return ordered fallback backends to try if ``primary`` fails to load.

    Args:
        primary: The primary backend that failed.

    Returns:
        Ordered list of alternative backends (may be empty for PyTorch).
    """
    return list(_FALLBACK_CHAIN.get(primary, []))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_priority_chain(
    hw: HardwareProfile,
) -> tuple[BackendType, DeviceType, str, str]:
    """Evaluate the priority chain and return (backend, device_type, device_str, reason).

    Raises:
        DependencyError: No usable backend is installed.
    """
    libs = hw.libraries

    # Priority 1: NVIDIA GPU + TensorRT
    if hw.has_nvidia_gpu and libs.tensorrt_version:
        return (
            BackendType.TENSORRT,
            DeviceType.CUDA,
            "cuda:0",
            f"NVIDIA GPU detected with TensorRT {libs.tensorrt_version}",
        )

    # Priority 2: NVIDIA GPU + ONNX with CUDA EP
    if hw.has_nvidia_gpu and libs.onnxruntime_version and libs.onnxruntime_has_cuda:
        return (
            BackendType.ONNX,
            DeviceType.CUDA,
            "cuda:0",
            f"NVIDIA GPU detected with ONNX Runtime {libs.onnxruntime_version} (CUDA EP)",
        )

    # Priority 3: Native CoreML (Apple Silicon with coremltools)
    if hw.cpu.cpu_arch == CPUArch.AARCH64 and _is_macos() and libs.coremltools_version:
        return (
            BackendType.COREML,
            DeviceType.CPU,
            "cpu",
            f"CoreML native (coremltools {libs.coremltools_version})",
        )

    # Priority 4: ONNX with CoreML EP (Apple Silicon Neural Engine)
    if libs.onnxruntime_version and libs.onnxruntime_has_coreml:
        return (
            BackendType.ONNX,
            DeviceType.CPU,
            "cpu",
            f"ONNX Runtime {libs.onnxruntime_version} (CoreML EP — Apple Neural Engine)",
        )

    # Priority 5: OpenVINO (CPU or Intel iGPU)
    if libs.openvino_version:
        return (
            BackendType.OPENVINO,
            DeviceType.CPU,
            "cpu",
            f"OpenVINO {libs.openvino_version} installed",
        )

    # Priority 6: ONNX CPU EP
    if libs.onnxruntime_version:
        return (
            BackendType.ONNX,
            DeviceType.CPU,
            "cpu",
            f"ONNX Runtime {libs.onnxruntime_version} installed (CPU EP)",
        )

    # Priority 7: PyTorch fallback
    if libs.torch_version:
        device_type = DeviceType.CUDA if libs.torch_cuda_available else DeviceType.CPU
        device_str = "cuda:0" if libs.torch_cuda_available else "cpu"
        return (
            BackendType.PYTORCH,
            device_type,
            device_str,
            f"PyTorch {libs.torch_version} installed (fallback)",
        )

    # Priority 8: nothing available
    raise DependencyError(
        package="inference backend",
        install_cmd=_INSTALL_HINTS,
        message="No supported inference backend was found.",
    )


def _resolve_override(
    hw: HardwareProfile,
    model_size: str,
    backend_override: str,
    device_override: str | None,
    precision_override: str | None,
) -> BackendSelection:
    """Validate and resolve a user-specified backend override."""
    try:
        backend = BackendType(backend_override.lower())
    except ValueError:
        valid = [b.value for b in BackendType]
        raise BackendError(
            f"Unknown backend override '{backend_override}'. Valid: {valid}"
        ) from None

    libs = hw.libraries
    _check_backend_available(backend, hw)

    # Determine device type
    gpu_backend = backend in (BackendType.TENSORRT, BackendType.ONNX) and hw.has_nvidia_gpu
    torch_cuda_backend = backend == BackendType.PYTORCH and libs.torch_cuda_available
    mps_requested = device_override is not None and device_override.startswith("mps")
    if gpu_backend or torch_cuda_backend:
        device_type = DeviceType.CUDA
        device_str = device_override or "cuda:0"
    elif mps_requested and device_override is not None:
        device_type = DeviceType.MPS
        device_str = device_override
    else:
        device_type = DeviceType.CPU
        device_str = device_override or "cpu"

    preferred_precision = _parse_precision(precision_override)
    precision = select_precision(hw, model_size, backend, preferred_precision)
    device_index = _parse_device_index(device_str)

    return BackendSelection(
        backend=backend,
        device_type=device_type,
        precision=precision,
        device_index=device_index,
        reason=f"User override: {backend.value}",
    )


def _check_backend_available(backend: BackendType, hw: HardwareProfile) -> None:
    """Raise BackendError if the specified backend is not usable on this hardware."""
    libs = hw.libraries

    match backend:
        case BackendType.PYTORCH:
            if not libs.torch_version:
                raise BackendError(
                    "PyTorch (torch) is not installed. Install with: uv add yowo[pytorch]"
                )
        case BackendType.ONNX:
            if not libs.onnxruntime_version:
                raise BackendError(
                    "ONNX Runtime is not installed. Install with: uv add onnxruntime"
                )
        case BackendType.TENSORRT:
            if not libs.tensorrt_version:
                raise BackendError("TensorRT is not installed. Install with: uv add tensorrt")
            if not hw.has_nvidia_gpu:
                raise BackendError("TensorRT requires an NVIDIA GPU.")
        case BackendType.OPENVINO:
            if not libs.openvino_version:
                raise BackendError("OpenVINO is not installed. Install with: uv add openvino")
        case BackendType.COREML:
            if not libs.coremltools_version:
                raise BackendError(
                    "coremltools is not installed. Install with: uv add coremltools>=7.0"
                )


def _parse_precision(value: str | None) -> Precision | None:
    """Parse a precision string into a ``Precision`` enum, or None."""
    if value is None:
        return None
    try:
        return Precision(value.lower())
    except ValueError:
        valid = [p.value for p in Precision]
        raise BackendError(f"Unknown precision override '{value}'. Valid: {valid}") from None


def _degrade_from(precision: Precision) -> list[Precision]:
    """Return an ordered degradation list starting at ``precision``."""
    order = [Precision.FP32, Precision.FP16, Precision.INT8]
    idx = order.index(precision)
    # From chosen precision, try down then up (prefer staying close)
    result = [precision]
    for p in order[idx + 1 :]:
        result.append(p)
    return result


def _parse_device_index(device_str: str) -> int:
    """Extract device index from a string like 'cuda:0' or 'cpu'."""
    if ":" in device_str:
        try:
            return int(device_str.split(":", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0
