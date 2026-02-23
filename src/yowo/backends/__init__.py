"""Inference backend protocol and factory.

Defines the contract that every backend must satisfy.
Backends are black boxes: callers use Protocol methods only.
create_backend() uses lazy imports so unused SDKs are never loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from yowo.hardware import HardwareProfile
from yowo.types import BackendType, ModelSpec, PreprocessedTensor

__all__ = [
    "InferenceBackend",
    "create_backend",
]


@runtime_checkable
class InferenceBackend(Protocol):
    """Contract for every inference backend implementation.

    Backends are structural — no base class required, only this Protocol.
    All methods are safe to call after construction; ``load()`` must be
    called before ``infer()`` or ``warmup()``.
    """

    @property
    def backend_type(self) -> BackendType:
        """Identifies the concrete backend implementation."""
        ...

    @property
    def is_loaded(self) -> bool:
        """True after a successful ``load()`` call."""
        ...

    @property
    def input_shape(self) -> tuple[int, int]:
        """Expected (H, W) input size."""
        ...

    def load(self, model_path: str | Path, *, device: str = "auto") -> None:
        """Load model from disk.

        Raises:
            BackendLoadError: Artifact exists but cannot be loaded into the backend.
            ModelLoadError: File is corrupt or version-incompatible.
            DependencyError: Required SDK not installed.
        """
        ...

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference on a preprocessed tensor batch.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            Raw model output — shape varies by model family.

        Raises:
            InferenceError: Runtime failure during the forward pass.
        """
        ...

    def unload(self) -> None:
        """Release all device memory and file handles.

        Safe to call multiple times and before ``load()`` is called.
        """
        ...

    def warmup(self, batch_size: int = 1) -> None:
        """Run a dummy inference pass to prime JIT caches and CUDA contexts.

        No-op if backend does not benefit from warmup or is not loaded.
        """
        ...


def create_backend(
    backend_type: BackendType,
    hw_profile: HardwareProfile,
    *,
    model_spec: ModelSpec | None = None,
) -> InferenceBackend:
    """Instantiate a backend (does not load a model).

    Uses lazy imports — the SDK for the requested backend is only imported
    when this function is called, so importing ``yowo.backends`` on a machine
    without CUDA does not fail.

    Args:
        backend_type: Which backend implementation to create.
        hw_profile: Hardware snapshot used for device validation.
        model_spec: Model specification (required for PyTorch backend to build
            the native architecture).

    Returns:
        An unloaded ``InferenceBackend`` instance.

    Raises:
        DependencyError: Required SDK is not installed.
        BackendError: Backend cannot be used on this hardware.
    """
    match backend_type:
        case BackendType.PYTORCH:
            from yowo.backends._pytorch import PyTorchBackend

            return PyTorchBackend(hw_profile, model_spec=model_spec)
        case BackendType.ONNX:
            from yowo.backends._onnx import OnnxBackend

            return OnnxBackend(hw_profile)
        case BackendType.TENSORRT:
            from yowo.backends._tensorrt import TensorRTBackend

            return TensorRTBackend(hw_profile)
        case BackendType.OPENVINO:
            from yowo.backends._openvino import OpenVinoBackend

            return OpenVinoBackend(hw_profile)
