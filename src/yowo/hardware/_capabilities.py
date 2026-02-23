"""Detect installed inference library versions.

Each probe function returns optional version string (None = not installed).
Never raises on ImportError.

Uses lazy imports inside each probe function — no optional package is
imported at module level.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

__all__ = [
    "InstalledLibraries",
    "detect_libraries",
    "probe_onnxruntime",
    "probe_openvino",
    "probe_tensorrt",
    "probe_torch",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InstalledLibraries:
    """Snapshot of optional inference SDK availability.

    All version fields are ``None`` when the package is not installed.
    Boolean flags reflect whether the corresponding acceleration provider
    is available at runtime (not just installed).

    Attributes:
        torch_version: PyTorch version string, or None.
        torch_cuda_available: True when torch.cuda.is_available() returns True.
        cuda_version: CUDA runtime version reported by torch, or None.
        tensorrt_version: TensorRT version string, or None.
        onnxruntime_version: onnxruntime version string, or None.
        onnxruntime_has_cuda: True when CUDAExecutionProvider is available.
        openvino_version: OpenVINO runtime version string, or None.
    """

    torch_version: str | None = None
    torch_cuda_available: bool = False
    cuda_version: str | None = None
    tensorrt_version: str | None = None
    onnxruntime_version: str | None = None
    onnxruntime_has_cuda: bool = False
    openvino_version: str | None = None


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def probe_torch() -> tuple[str | None, bool]:
    """Probe PyTorch installation.

    Returns:
        Tuple of (version_string_or_None, cuda_available).
    """
    if importlib.util.find_spec("torch") is None:
        return (None, False)
    try:
        import torch  # type: ignore[import-untyped]

        version: str = torch.__version__
        cuda_available: bool = bool(torch.cuda.is_available())
        return (version, cuda_available)
    except Exception:
        _log.debug("probe_torch: import succeeded but introspection failed", exc_info=True)
        return (None, False)


def probe_tensorrt() -> str | None:
    """Probe TensorRT installation.

    Returns:
        Version string (e.g. "10.0.1"), or None if not installed.
    """
    if importlib.util.find_spec("tensorrt") is None:
        return None
    try:
        import tensorrt  # type: ignore[import-untyped]

        return str(tensorrt.__version__)
    except Exception:
        _log.debug("probe_tensorrt: import failed", exc_info=True)
        return None


def probe_onnxruntime() -> tuple[str | None, bool]:
    """Probe onnxruntime installation.

    Checks both ``onnxruntime`` and ``onnxruntime-gpu`` packages (same
    import name). CUDAExecutionProvider availability is queried from
    ``onnxruntime.get_available_providers()``.

    Returns:
        Tuple of (version_string_or_None, has_cuda_execution_provider).
    """
    if importlib.util.find_spec("onnxruntime") is None:
        return (None, False)
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        version: str = ort.__version__
        has_cuda: bool = "CUDAExecutionProvider" in ort.get_available_providers()
        return (version, has_cuda)
    except Exception:
        _log.debug("probe_onnxruntime: import failed", exc_info=True)
        return (None, False)


def probe_openvino() -> str | None:
    """Probe OpenVINO runtime installation.

    Returns:
        Version string, or None if not installed.
    """
    if importlib.util.find_spec("openvino") is None:
        return None
    try:
        import openvino as ov  # type: ignore[import-untyped]

        return str(ov.__version__)
    except Exception:
        _log.debug("probe_openvino: import failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def detect_libraries() -> InstalledLibraries:
    """Run all probes and aggregate results into an InstalledLibraries snapshot.

    Never raises — each probe handles its own exceptions internally.
    """
    torch_version, torch_cuda = probe_torch()
    ort_version, ort_has_cuda = probe_onnxruntime()

    # Resolve CUDA version via torch when available
    cuda_version: str | None = None
    if torch_cuda:
        try:
            import torch  # type: ignore[import-untyped]

            cuda_version = torch.version.cuda
        except Exception:
            pass

    return InstalledLibraries(
        torch_version=torch_version,
        torch_cuda_available=torch_cuda,
        cuda_version=cuda_version,
        tensorrt_version=probe_tensorrt(),
        onnxruntime_version=ort_version,
        onnxruntime_has_cuda=ort_has_cuda,
        openvino_version=probe_openvino(),
    )
