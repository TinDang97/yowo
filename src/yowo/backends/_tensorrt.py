"""TensorRT inference backend.

Loads pre-compiled ``.engine`` files using ONNX Runtime's
TensorrtExecutionProvider. This approach avoids direct CUDA buffer
management (cuda-python) while still leveraging TensorRT acceleration.

All SDK imports are deferred inside load() so this module can be imported
on machines without TensorRT installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendError, BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

logger = logging.getLogger(__name__)

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
        self._output_names: list[str] = []
        # KV cache I/O state (populated in load() when model has KV I/O)
        self._has_kv_io: bool = False
        self._kv_state: dict[str, Any] = {}  # numpy or OrtValue
        self._kv_input_names: list[str] = []
        self._kv_output_names: list[str] = []
        self._kv_shapes: dict[str, tuple[int, ...]] = {}
        self._kv_use_ortvalue: bool = False
        self._kv_output_indices: dict[str, int] = {}
        # Pre-allocated use_cache scalars (populated in load() for KV models)
        self._use_cache_cold: NDArray[np.float32] = np.array(0.0, dtype=np.float32)
        self._use_cache_warm: NDArray[np.float32] = np.array(1.0, dtype=np.float32)
        self._use_cache_cold_ort: Any = None
        self._use_cache_warm_ort: Any = None
        # Cached module ref and pre-allocated zero tensors (set in load())
        self._ort: Any = None
        self._kv_zeros: dict[str, NDArray[np.float32]] = {}
        self._kv_zeros_ort: dict[str, Any] = {}
        # Standard (non-KV) OrtValue path (set in load())
        self._use_ortvalue: bool = False
        self._ortvalue_fn: Any = None
        # Batch dimension info (set in load())
        self._dynamic_batch: bool = True
        self._static_batch: int | None = None

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
        # Reset KV state from any previous model to prevent stale routing
        self._has_kv_io = False
        self._kv_state = {}
        self._kv_input_names = []
        self._kv_output_names = []
        self._kv_shapes = {}
        self._kv_use_ortvalue = False
        self._kv_output_indices = {}
        self._use_cache_cold_ort = None
        self._use_cache_warm_ort = None
        self._ort = None
        self._kv_zeros = {}
        self._kv_zeros_ort = {}
        self._use_ortvalue = False
        self._ortvalue_fn = None

        try:
            import onnxruntime as ort_mod  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "onnxruntime-gpu",
                "uv add onnxruntime-gpu",
            ) from exc

        self._ort = ort_mod
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
            # Detect dynamic vs static batch dimension
            batch_dim = input_shape[0] if input_shape and len(input_shape) >= 1 else None
            self._dynamic_batch = not isinstance(batch_dim, int)
            self._static_batch = int(batch_dim) if isinstance(batch_dim, int) else None
            # Cache output names for run_with_ort_values()
            self._output_names = [o.name for o in self._session.get_outputs()]
            # OrtValue zero-copy path: TRT backend always has CUDA EP,
            # so the zero-copy benefit (avoids host→device copy) applies.
            _active = self._session.get_providers()
            _has_gpu_ep = any("CUDA" in p or "TensorRT" in p for p in _active)
            self._use_ortvalue = _has_gpu_ep and hasattr(self._session, "run_with_ort_values")
            if self._use_ortvalue:
                from yowo.backends._ortvalue import infer_standard_ortvalue

                self._ortvalue_fn = infer_standard_ortvalue
            # Detect KV cache I/O by inspecting input names
            kv_inputs = [i for i in inputs if i.name.startswith("past_")]
            if kv_inputs:
                self._has_kv_io = True
                self._kv_input_names = [i.name for i in kv_inputs]
                self._kv_shapes = {i.name: tuple(i.shape) for i in kv_inputs}
                self._kv_output_names = [
                    o.name for o in self._session.get_outputs() if o.name.startswith("present_")
                ]
                self._kv_use_ortvalue = hasattr(self._session, "run_with_ort_values")
                # Map present output names → indices for named lookup
                for idx, oname in enumerate(self._output_names):
                    if oname in set(self._kv_output_names):
                        self._kv_output_indices[oname] = idx
                # Pre-allocate zero tensors for cold-start KV feed
                self._kv_zeros = {
                    name: np.zeros(shape, dtype=np.float32)
                    for name, shape in self._kv_shapes.items()
                }
                # Pre-allocate OrtValue scalars and zero OrtValues for hot-path reuse
                if self._kv_use_ortvalue:
                    self._use_cache_cold_ort = ort_mod.OrtValue.ortvalue_from_numpy(
                        self._use_cache_cold
                    )
                    self._use_cache_warm_ort = ort_mod.OrtValue.ortvalue_from_numpy(
                        self._use_cache_warm
                    )
                    self._kv_zeros_ort = {
                        name: ort_mod.OrtValue.ortvalue_from_numpy(arr)
                        for name, arr in self._kv_zeros.items()
                    }
        except Exception as exc:
            self._session = None
            raise BackendLoadError(f"TensorRTBackend: engine load failed: {exc}") from exc

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
            if self._has_kv_io and self._kv_use_ortvalue:
                return self._infer_kv_ortvalue(tensor)
            if self._has_kv_io:
                return self._infer_kv_numpy(tensor)
            if self._use_ortvalue:
                return self._infer_standard_ortvalue(tensor)
            outputs = self._session.run(None, {self._input_name: tensor.data})
            return np.asarray(outputs[0], dtype=np.float32)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"TensorRTBackend: inference failed: {exc}") from exc

    def _infer_standard_ortvalue(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Standard inference using OrtValue for zero-copy input wrapping."""
        return self._ortvalue_fn(
            self._ort,
            self._session,
            self._input_name,
            self._output_names,
            tensor,
        )

    def _infer_kv_numpy(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """KV-cache inference using numpy arrays (fallback path)."""
        feed: dict[str, NDArray[np.float32]] = {
            self._input_name: tensor.data,
            "use_cache": self._use_cache_warm if self._kv_state else self._use_cache_cold,
        }
        for name in self._kv_input_names:
            feed[name] = self._kv_state.get(name, self._kv_zeros[name])
        outputs = self._session.run(None, feed)
        for present_name in self._kv_output_names:
            past_name = present_name.replace("present_", "past_")
            self._kv_state[past_name] = outputs[self._kv_output_indices[present_name]]
        return np.asarray(outputs[0], dtype=np.float32)

    def _infer_kv_ortvalue(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """KV-cache inference using OrtValue for zero-copy tensor passing."""
        ort = self._ort
        feed: dict[str, Any] = {
            self._input_name: ort.OrtValue.ortvalue_from_numpy(tensor.data),
            "use_cache": self._use_cache_warm_ort if self._kv_state else self._use_cache_cold_ort,
        }
        for name in self._kv_input_names:
            feed[name] = (
                self._kv_state[name] if name in self._kv_state else self._kv_zeros_ort[name]
            )
        ort_outputs = self._session.run_with_ort_values(self._output_names, feed)
        # Store present K,V as OrtValues (zero-copy for next frame)
        for present_name in self._kv_output_names:
            past_name = present_name.replace("present_", "past_")
            self._kv_state[past_name] = ort_outputs[self._kv_output_indices[present_name]]
        out = ort_outputs[0].numpy()
        return out if out.dtype == np.float32 else out.astype(np.float32)

    def clear_kv_cache(self) -> None:
        """Clear KV state for streaming reset (e.g. new video source)."""
        self._kv_state = {}

    def set_source_id(self, source_id: str) -> None:
        """No-op — feature caching not supported."""

    def unload(self) -> None:
        """Release the ONNX Runtime session and free TRT resources."""
        self._kv_state = {}
        self._session = None

    def warmup(self, batch_size: int = 1) -> None:
        """Run multiple dummy inferences to prime TRT caches.

        TensorRT benefits from 3 warmup passes to fill L2 cache and
        initialise the CUDA context. No-op when not loaded.
        """
        if self._session is None:
            return

        if not self._dynamic_batch and batch_size > (self._static_batch or 1):
            logger.warning(
                "TensorRTBackend: model has static batch=%d but requested batch=%d; "
                "inference may fail for batch > %d",
                self._static_batch or 1,
                batch_size,
                self._static_batch or 1,
            )

        try:
            h, w = self._input_shape
            dummy = np.zeros((batch_size, 3, h, w), dtype=np.float32)
            if self._has_kv_io:
                feed: dict[str, NDArray[np.float32]] = {
                    self._input_name: dummy,
                    "use_cache": np.array(0.0, dtype=np.float32),
                }
                for name in self._kv_input_names:
                    feed[name] = np.zeros(self._kv_shapes[name], dtype=np.float32)
                for _ in range(3):
                    self._session.run(None, feed)
            else:
                for _ in range(3):
                    self._session.run(None, {self._input_name: dummy})
        except Exception as exc:
            logger.debug("TensorRTBackend: warmup failed (non-fatal): %s", exc)


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
