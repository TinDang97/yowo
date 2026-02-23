"""PyTorch inference backend.

Uses the native ``yowo.arch`` module for model loading and inference.
Serves as the reference implementation and universal fallback.
All SDK imports are deferred to load() so importing this module
never fails on machines without torch.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendLoadError, DependencyError, DeviceError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, ModelSpec, PreprocessedTensor

logger = logging.getLogger(__name__)

__all__ = ["PyTorchBackend"]


class PyTorchBackend:
    """PyTorch backend using native YOLO architecture.

    This backend is the universal fallback. It requires ``torch`` to be
    installed, but it is not imported at module level — only inside ``load()``.
    """

    def __init__(
        self,
        hw_profile: HardwareProfile,
        *,
        model_spec: ModelSpec | None = None,
    ) -> None:
        if not hw_profile.libraries.torch_version:
            raise DependencyError(
                "torch",
                "uv add yowo[pytorch]",
            )
        self._hw = hw_profile
        self._spec = model_spec
        self._model: Any = None  # YOLOModel at runtime
        self._torch: Any = None  # cached torch module from load()
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
        """Load ``.pt`` weights via native architecture.

        Builds the YOLO model from ``model_spec``, loads checkpoint
        weights, fuses Conv+BN, and applies inference optimisations.

        Args:
            model_path: Path to a ``.pt`` weights file.
            device: ``"auto"`` selects CUDA when available, otherwise CPU.

        Raises:
            BackendLoadError: Model file could not be loaded.
            DeviceError: Requested device is unavailable or out of memory.
        """
        if self._spec is None:
            raise BackendLoadError(
                "PyTorchBackend: model_spec is required to build the native model. "
                "Pass model_spec when creating the backend."
            )

        try:
            import torch  # type: ignore[import-untyped]

            from yowo.arch import build_model
            from yowo.arch._weights import load_weights
        except ImportError as exc:
            raise DependencyError("torch", "uv add yowo[pytorch]") from exc

        self._torch = torch  # cache for hot path

        resolved = self._resolve_device(device)
        try:
            # Build native model from spec
            model = build_model(self._spec.family, self._spec.size)
            load_weights(model, model_path)

            # Inference optimisations
            model = model.fuse()
            model.eval()
            model.to(resolved)

            # Channels-last for GPU Tensor Core optimisation
            if resolved.startswith("cuda"):
                model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

            self._model = model
            self._device_str = resolved
        except RuntimeError as exc:
            self._model = None
            msg = str(exc).lower()
            if "cuda" in msg or "device" in msg or "out of memory" in msg:
                raise DeviceError(f"PyTorchBackend: device error: {exc}") from exc
            raise BackendLoadError(f"PyTorchBackend: failed to load model: {exc}") from exc
        except Exception as exc:
            self._model = None
            raise BackendLoadError(f"PyTorchBackend: failed to load model: {exc}") from exc

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference using the native YOLO model.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            Raw model output as float32 numpy array.

        Raises:
            InferenceError: Model not loaded or inference failed.
        """
        if self._model is None:
            raise InferenceError("PyTorchBackend: no model loaded - call load() first")

        try:
            torch = self._torch

            t = torch.from_numpy(tensor.data).to(self._device_str, non_blocking=True)

            # Channels-last input on GPU
            if self._device_str.startswith("cuda"):
                t = t.to(memory_format=torch.channels_last)

            with torch.inference_mode():
                output = self._model(t)

            return output.cpu().numpy()
        except Exception as exc:
            raise InferenceError(f"PyTorchBackend: inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release model reference and free device memory."""
        if self._model is not None:
            device = self._device_str
            del self._model
            self._model = None
            if device.startswith("cuda"):
                with contextlib.suppress(Exception):
                    self._torch.cuda.empty_cache()

    def warmup(self, batch_size: int = 1) -> None:
        """Run a single dummy forward pass to prime the CUDA context.

        No-op when the model is not loaded.
        """
        if self._model is None:
            return

        try:
            torch = self._torch

            h, w = self._input_shape
            dummy = torch.zeros((batch_size, 3, h, w), device=self._device_str)
            if self._device_str.startswith("cuda"):
                dummy = dummy.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                self._model(dummy)
        except Exception as exc:
            logger.debug("warmup failed (non-fatal): %s", exc)

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
