"""PyTorch inference backend.

Uses ultralytics.YOLO for model loading.
Serves as the reference implementation and universal fallback.
All SDK imports are deferred to load() so importing this module
never fails on machines without torch or ultralytics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

__all__ = ["PyTorchBackend"]


class PyTorchBackend:
    """PyTorch backend using the ultralytics YOLO interface.

    This backend is the universal fallback. It requires ``torch`` and
    ``ultralytics`` to be installed, but neither is imported at module
    level — only inside ``load()``.
    """

    def __init__(self, hw_profile: HardwareProfile) -> None:
        if not hw_profile.libraries.torch_version:
            raise DependencyError(
                "torch",
                "uv add torch ultralytics",
            )
        self._hw = hw_profile
        self._model: Any = None  # ultralytics YOLO instance at runtime
        self._device_str: str = "cpu"
        self._input_shape: tuple[int, int] = (640, 640)

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PYTORCH

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def load(self, model_path: str | Path, *, device: str = "auto") -> None:
        """Load ``.pt`` weights via ultralytics YOLO.

        Args:
            model_path: Path to a ``.pt`` weights file.
            device: ``"auto"`` selects CUDA when available, otherwise CPU.

        Raises:
            DependencyError: ``ultralytics`` package not installed.
            BackendLoadError: Model file could not be loaded.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError("ultralytics", "uv add ultralytics") from exc

        resolved = self._resolve_device(device)
        try:
            model = YOLO(str(model_path))
            model.to(resolved)
            self._model = model
            self._device_str = resolved
        except Exception as exc:
            self._model = None
            raise BackendLoadError(f"PyTorchBackend: failed to load model: {exc}") from exc

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference using the ultralytics model's underlying nn.Module.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            Raw model output as float32 numpy array.

        Raises:
            InferenceError: Model not loaded or inference failed.
        """
        if self._model is None:
            raise InferenceError("PyTorchBackend: no model loaded — call load() first")

        try:
            import torch  # type: ignore[import-untyped]

            t = torch.from_numpy(tensor.data).to(self._device_str)
            # Access the underlying nn.Module for raw tensor output
            inner_model = getattr(self._model, "model", self._model)
            with torch.no_grad():
                output = inner_model(t)

            # Ultralytics may return a list/tuple; take the first element
            raw = output[0] if isinstance(output, list | tuple) else output

            if hasattr(raw, "cpu"):
                return raw.cpu().numpy().astype(np.float32)
            return np.asarray(raw, dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"PyTorchBackend: inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release model reference and allow GC to free device memory."""
        self._model = None

    def warmup(self, batch_size: int = 1) -> None:
        """Run a single dummy forward pass to prime the CUDA context.

        No-op when the model is not loaded.
        """
        if self._model is None:
            return

        try:
            import torch  # type: ignore[import-untyped]

            h, w = self._input_shape
            dummy = torch.zeros((batch_size, 3, h, w), device=self._device_str)
            inner_model = getattr(self._model, "model", self._model)
            with torch.no_grad():
                inner_model(dummy)
        except Exception:
            # Warmup failures are non-fatal
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        """Translate ``"auto"`` into a concrete device string."""
        if device != "auto":
            return device
        libs = self._hw.libraries
        if self._hw.has_nvidia_gpu and libs.torch_cuda_available:
            return "cuda:0"
        return "cpu"
