"""TensorRT inference backend.

Loads pre-compiled ``.engine`` files using ONNX Runtime's
TensorrtExecutionProvider. This approach avoids direct CUDA buffer
management (cuda-python) while still leveraging TensorRT acceleration.

All SDK imports are deferred inside load() so this module can be imported
on machines without TensorRT installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendError, BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

__all__ = ["TensorRTBackend"]


class TensorRTBackend:
    """TensorRT backend via ONNX Runtime TensorrtExecutionProvider.

    Accepts ``.engine`` (serialised TensorRT engine) or ``.onnx`` files.
    When given an ``.onnx`` file the TRT EP will compile and cache the engine
    on first run. When given a ``.engine`` file the TRT EP deserialises it
    directly.

    Requires:
    - An NVIDIA GPU.
    - ``tensorrt`` installed (``uv add tensorrt``).
    - ``onnxruntime-gpu`` installed for the TensorrtExecutionProvider.
    """

    def __init__(self, hw_profile: HardwareProfile) -> None:
        if not hw_profile.libraries.tensorrt_version:
            raise DependencyError("tensorrt", "uv add tensorrt")
        if not hw_profile.has_nvidia_gpu:
            raise BackendError("TensorRTBackend: an NVIDIA GPU is required")

        self._hw = hw_profile
        self._session: Any = None  # ort.InferenceSession at runtime
        self._input_name: str = ""
        self._input_shape: tuple[int, int] = (640, 640)
        self._device_index: int = 0

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.TENSORRT

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def load(self, model_path: str | Path, *, device: str = "auto") -> None:
        """Deserialise a TensorRT ``.engine`` file via ONNX Runtime TRT EP.

        When ``model_path`` ends with ``.engine`` the engine is used directly.
        When it ends with ``.onnx`` the TRT EP compiles and caches the engine.

        Args:
            model_path: Path to ``.engine`` or ``.onnx`` artifact.
            device: ``"auto"`` or ``"cuda:<index>"`` (index defaults to 0).

        Raises:
            DependencyError: ``onnxruntime-gpu`` not installed.
            BackendLoadError: Engine deserialisation or session creation failed.
        """
        try:
            import onnxruntime as ort_mod  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "onnxruntime-gpu",
                "uv add onnxruntime-gpu",
            ) from exc

        self._device_index = _parse_device_index(device)

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": self._device_index,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(Path(model_path).parent),
                },
            ),
            (
                "CUDAExecutionProvider",
                {"device_id": self._device_index},
            ),
        ]

        try:
            self._session = ort_mod.InferenceSession(
                str(model_path),
                providers=providers,
            )
            inputs = self._session.get_inputs()
            self._input_name = inputs[0].name if inputs else ""
            input_shape = inputs[0].shape if inputs else None
            if (
                input_shape is not None
                and len(input_shape) == 4
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                self._input_shape = (int(input_shape[2]), int(input_shape[3]))
        except Exception as exc:
            self._session = None
            raise BackendLoadError(
                f"TensorRTBackend: engine load failed: {exc}"
            ) from exc

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run TensorRT inference.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            First output tensor as float32.

        Raises:
            InferenceError: Session not loaded or runtime failure.
        """
        if self._session is None:
            raise InferenceError("TensorRTBackend: no engine loaded — call load() first")

        try:
            outputs = self._session.run(None, {self._input_name: tensor.data})
            return np.asarray(outputs[0], dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"TensorRTBackend: inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release the ONNX Runtime session and free TRT resources."""
        self._session = None

    def warmup(self, batch_size: int = 1) -> None:
        """Run multiple dummy inferences to prime TRT caches.

        TensorRT benefits from 3 warmup passes to fill L2 cache and
        initialise the CUDA context. No-op when not loaded.
        """
        if self._session is None:
            return

        try:
            h, w = self._input_shape
            dummy = np.zeros((batch_size, 3, h, w), dtype=np.float32)
            for _ in range(3):
                self._session.run(None, {self._input_name: dummy})
        except Exception:
            # Warmup failures are non-fatal
            pass


# ---------------------------------------------------------------------------
# Module-level helper (not a method to avoid coupling)
# ---------------------------------------------------------------------------


def _parse_device_index(device: str) -> int:
    """Extract numeric device index from ``"cuda:N"`` or ``"auto"``."""
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0
