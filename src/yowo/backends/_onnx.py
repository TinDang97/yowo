"""ONNX Runtime inference backend.

Supports CUDA and CPU execution providers.
Automatically selects CUDAExecutionProvider when the hardware profile
indicates an NVIDIA GPU and onnxruntime-gpu is installed.

All SDK imports are deferred inside load() so this module can be imported
on machines without onnxruntime installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

if TYPE_CHECKING:
    import onnxruntime as ort  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

__all__ = ["OnnxBackend"]


class OnnxBackend:
    """ONNX Runtime backend with automatic CUDA / CPU EP selection.

    Loads ``.onnx`` model files via ``onnxruntime.InferenceSession``.
    CUDA EP is used when the hardware profile reports GPU + CUDA EP support.
    """

    def __init__(self, hw_profile: HardwareProfile) -> None:
        if not hw_profile.libraries.onnxruntime_version:
            raise DependencyError(
                "onnxruntime",
                "uv add onnxruntime  # CPU\nuv add onnxruntime-gpu  # CUDA",
            )
        self._hw = hw_profile
        self._session: Any = None  # ort.InferenceSession at runtime
        self._input_name: str = ""
        self._input_shape: tuple[int, int] = (640, 640)
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

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.ONNX

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
        """Load an ``.onnx`` model file.

        Args:
            model_path: Path to a ``.onnx`` model.
            device: ``"auto"`` selects CUDA EP when available, otherwise CPU.

        Raises:
            DependencyError: ``onnxruntime`` not installed.
            BackendLoadError: Session creation failed.
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

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "onnxruntime",
                "uv add onnxruntime  # CPU\nuv add onnxruntime-gpu  # CUDA",
            ) from exc

        self._ort = ort
        providers = self._select_providers(device)
        opts = self._build_session_options(ort)

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=providers,
            )
            # Cache the input node name for infer()
            inputs = self._session.get_inputs()
            self._input_name = inputs[0].name if inputs else ""
            # Extract spatial dims from the model's input shape [B, C, H, W]
            input_shape = inputs[0].shape if inputs else None
            if (
                input_shape is not None
                and len(input_shape) == 4
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                self._input_shape = (int(input_shape[2]), int(input_shape[3]))
            # Cache output names for run_with_ort_values()
            self._output_names = [o.name for o in self._session.get_outputs()]
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
                    self._use_cache_cold_ort = ort.OrtValue.ortvalue_from_numpy(
                        self._use_cache_cold
                    )
                    self._use_cache_warm_ort = ort.OrtValue.ortvalue_from_numpy(
                        self._use_cache_warm
                    )
                    self._kv_zeros_ort = {
                        name: ort.OrtValue.ortvalue_from_numpy(arr)
                        for name, arr in self._kv_zeros.items()
                    }
        except Exception as exc:
            self._session = None
            raise BackendLoadError(f"OnnxBackend: session creation failed: {exc}") from exc

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference on a preprocessed tensor batch.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            First output tensor as float32.

        Raises:
            InferenceError: Session not loaded or runtime failure.
        """
        if self._session is None:
            raise InferenceError("OnnxBackend: no session loaded — call load() first")

        try:
            if self._has_kv_io and self._kv_use_ortvalue:
                return self._infer_kv_ortvalue(tensor)
            if self._has_kv_io:
                return self._infer_kv_numpy(tensor)
            outputs = self._session.run(None, {self._input_name: tensor.data})
            return np.asarray(outputs[0], dtype=np.float32)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"OnnxBackend: inference failed: {exc}") from exc

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
        """Release the ONNX Runtime session."""
        self._kv_state = {}
        self._session = None

    def warmup(self, batch_size: int = 1) -> None:
        """Run a dummy inference to prime the execution provider.

        No-op when the session is not loaded.
        """
        if self._session is None:
            return

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
                self._session.run(None, feed)
            else:
                self._session.run(None, {self._input_name: dummy})
        except Exception as exc:
            logger.debug("OnnxBackend: warmup failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session_options(self, ort: object) -> ort.SessionOptions:  # type: ignore[name-defined]
        """Build session options with all optimisations and thread parallelism.

        Thread counts are capped at half the logical CPUs to avoid
        scheduling onto slow efficiency cores (Apple Silicon) and
        memory-bandwidth saturation on many-core machines.
        """
        import onnxruntime as ort_mod  # type: ignore[import-untyped]

        opts = ort_mod.SessionOptions()
        opts.graph_optimization_level = ort_mod.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = os.cpu_count() or 1
        opts.intra_op_num_threads = max(1, cpu_count // 2)
        opts.inter_op_num_threads = max(1, cpu_count // 4)
        return opts

    def _select_providers(self, device: str) -> list[str]:
        """Return the ordered execution provider list.

        Selection priority:
        1. CUDA EP — when device is ``"cuda"`` or ``"auto"`` + GPU detected.
        2. CoreML EP — when device is ``"auto"`` or ``"cpu"`` and CoreML is
           available (macOS with Apple Silicon).
        3. CPU EP — universal fallback, always appended.

        Args:
            device: Requested device string.

        Returns:
            List of ORT execution provider name strings.
        """
        use_cuda = False
        if device.startswith("cuda"):
            use_cuda = True
        elif device == "auto":
            libs = self._hw.libraries
            use_cuda = self._hw.has_nvidia_gpu and libs.onnxruntime_has_cuda

        if use_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # CoreML EP: 4-5x faster than CPU on Apple Silicon (Neural Engine)
        libs = self._hw.libraries
        if libs.onnxruntime_has_coreml:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]
