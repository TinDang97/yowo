"""ONNX Runtime inference backend.

Supports CUDA and CPU execution providers.
Automatically selects CUDAExecutionProvider when the hardware profile
indicates an NVIDIA GPU and onnxruntime-gpu is installed.

All SDK imports are deferred inside load() so this module can be imported
on machines without onnxruntime installed.
"""

from __future__ import annotations

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
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "onnxruntime",
                "uv add onnxruntime  # CPU\nuv add onnxruntime-gpu  # CUDA",
            ) from exc

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
            outputs = self._session.run(None, {self._input_name: tensor.data})
            return np.asarray(outputs[0], dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"OnnxBackend: inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release the ONNX Runtime session."""
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
            self._session.run(None, {self._input_name: dummy})
        except Exception:
            # Warmup failures are non-fatal
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session_options(self, ort: object) -> ort.SessionOptions:  # type: ignore[name-defined]
        """Build session options with all optimisations and thread parallelism."""
        import onnxruntime as ort_mod  # type: ignore[import-untyped]

        opts = ort_mod.SessionOptions()
        opts.graph_optimization_level = ort_mod.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = os.cpu_count() or 1
        opts.intra_op_num_threads = cpu_count
        opts.inter_op_num_threads = max(1, cpu_count // 2)
        return opts

    def _select_providers(self, device: str) -> list[str]:
        """Return the ordered execution provider list.

        CUDA EP is selected when:
        - device is "auto" and hardware profile shows GPU + CUDA EP, or
        - device starts with "cuda"

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
        return ["CPUExecutionProvider"]
