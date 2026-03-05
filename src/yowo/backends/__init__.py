"""Inference backend protocol and factory.

Defines the contract that every backend must satisfy.
Backends are black boxes: callers use Protocol methods only.
create_backend() uses lazy imports so unused SDKs are never loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from yowo.backends._selector import get_fallback_backends, select_backend
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, ModelSpec, PreprocessedTensor

__all__ = [
    "InferenceBackend",
    "ModelBuilder",
    "create_backend",
    "get_fallback_backends",
    "select_backend",
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

    def clear_kv_cache(self) -> None:
        """Reset KV cache state (e.g. on source change during streaming).

        No-op if backend does not use KV caching.
        """
        ...

    def set_source_id(self, source_id: str) -> None:
        """Set the active source identifier for feature cache keying.

        No-op for backends that do not support feature caching.
        """
        ...


@runtime_checkable
class ModelBuilder(Protocol):
    """Protocol for custom model architectures.

    Implement this to use custom (non-YOLO) architectures with
    ``PyTorchBackend``'s device management and inference pipeline.

    The ``build()`` method is fully responsible for architecture
    construction, weight loading, ``fuse()``/``eval()``/device placement.
    The returned module must be ready for ``forward(x)`` inference.
    """

    def build(self, num_classes: int, device: str) -> Any:
        """Build and return a ``torch.nn.Module`` ready for inference.

        Args:
            num_classes: Number of output classes.
            device: Target device string (``"cpu"``, ``"cuda:0"``, etc.).

        Returns:
            A ``torch.nn.Module`` on *device*, in eval mode, with weights
            loaded.
        """
        ...

    @property
    def input_shape(self) -> tuple[int, int]:
        """Expected ``(H, W)`` input spatial dimensions."""
        ...


def create_backend(
    backend_type: BackendType,
    hw_profile: HardwareProfile,
    *,
    model_spec: ModelSpec | None = None,
    model_builder: ModelBuilder | None = None,
    feature_cache: Any | None = None,
    kv_cache: bool = False,
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
        feature_cache: Optional FeatureCache for mmap-backed neck feature
            caching (PyTorch backend only, ignored by others).
        kv_cache: Enable attention KV cache and block output cache for
            streaming inference (PyTorch backend only).

    Returns:
        An unloaded ``InferenceBackend`` instance.

    Raises:
        DependencyError: Required SDK is not installed.
        BackendError: Backend cannot be used on this hardware.
    """
    match backend_type:
        case BackendType.PYTORCH:
            from yowo.backends._pytorch import PyTorchBackend

            return PyTorchBackend(
                hw_profile,
                model_spec=model_spec,
                model_builder=model_builder,
                feature_cache=feature_cache,
                kv_cache=kv_cache,
            )
        case BackendType.ONNX:
            from yowo.backends._onnx import OnnxBackend

            return OnnxBackend(hw_profile)
        case BackendType.TENSORRT:
            from yowo.backends._tensorrt import TensorRTBackend

            return TensorRTBackend(hw_profile)
        case BackendType.OPENVINO:
            from yowo.backends._openvino import OpenVinoBackend

            return OpenVinoBackend(hw_profile)
        case BackendType.COREML:
            from yowo.backends._coreml import CoreMLBackend

            return CoreMLBackend(hw_profile)
