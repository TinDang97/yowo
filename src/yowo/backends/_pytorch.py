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
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
        fp16: bool = False,
        feature_cache: Any | None = None,
        kv_cache: bool = False,
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
        self._compile: bool = compile
        self._compile_mode: str = compile_mode
        self._fp16: bool = fp16
        self._feature_cache: Any | None = feature_cache
        self._kv_cache: bool = kv_cache
        self._neck_hook_handle: Any = None
        self._last_neck_output: Any = None
        self._current_source_id: str = ""

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

        # Cap CPU thread pool to avoid memory-bandwidth saturation on
        # machines with many cores (e.g. Apple Silicon M-series).
        # Use half the logical CPUs for compute threads; keep interop low.
        if not resolved.startswith("cuda"):
            import os

            cpu_count = os.cpu_count() or 4
            # set_num_interop_threads() errors if called after parallel work
            # has started (e.g. a second load() call in the same process).
            # Silently skip — threads are already configured.
            try:
                torch.set_num_threads(max(1, cpu_count // 2))
                torch.set_num_interop_threads(max(1, min(2, cpu_count // 4)))
            except RuntimeError:
                pass

        try:
            # Build native model from spec
            model = build_model(self._spec.family, self._spec.size)
            load_weights(model, model_path)

            # Inference optimisations
            model = model.fuse()
            model.eval()
            model.to(resolved)

            # KV cache — opt-in attention/block caching for streaming
            if self._kv_cache:
                model.enable_kv_cache()
                logger.info("KV cache enabled for streaming inference")

            # Channels-last for GPU Tensor Core optimisation
            if resolved.startswith("cuda"):
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

            # torch.compile — opt-in kernel fusion (requires PyTorch >= 2.0)
            if self._compile:
                try:
                    model.compile_for_inference(mode=self._compile_mode)
                    logger.info("torch.compile enabled (mode=%s)", self._compile_mode)
                except Exception as exc:
                    logger.debug("torch.compile failed (falling back to eager): %s", exc)

            # Register neck forward hook to capture features for caching
            if self._feature_cache is not None:

                def _capture_neck(
                    _module: Any,
                    _input: Any,
                    output: Any,
                ) -> None:
                    self._last_neck_output = tuple(t.detach().cpu().numpy() for t in output)

                self._neck_hook_handle = model.neck.register_forward_hook(
                    _capture_neck,
                )

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

    def set_source_id(self, source_id: str) -> None:
        """Set the source identifier for feature cache keying.

        Called by the engine before ``infer()`` to enable per-source caching.
        Only effective when ``feature_cache`` was provided at construction.
        Clears KV cache when source changes to prevent stale cross-attention.
        """
        if source_id != self._current_source_id and self._kv_cache and self._model is not None:
            self._model.clear_kv_cache()
        self._current_source_id = source_id

    def clear_kv_cache(self) -> None:
        """Reset KV cache state (e.g. on source change during streaming)."""
        if self._kv_cache and self._model is not None:
            self._model.clear_kv_cache()

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference using the native YOLO model.

        When a feature cache is active and ``set_source_id()`` was called,
        this method checks for a cache hit first. On hit, only the detection
        head runs (skipping backbone + neck). On miss, full inference runs
        and the neck output is stored for future reuse.

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
            cache = self._feature_cache
            sid = self._current_source_id

            # Cache hit: head-only inference (skip backbone + neck)
            if cache is not None and sid:
                cached = cache.check_and_load(sid, tensor.data, self._device_str)
                if cached is not None:
                    with torch.inference_mode():
                        output = self._model.forward_head(cached)
                    return output.cpu().float().numpy()

            # Full inference path
            t = torch.from_numpy(tensor.data).to(self._device_str, non_blocking=True)

            # Channels-last input on GPU
            if self._device_str.startswith("cuda"):
                t = t.to(memory_format=torch.channels_last)

            with torch.inference_mode():
                if self._fp16 and self._device_str.startswith("cuda"):
                    with torch.amp.autocast("cuda", dtype=torch.float16):  # type: ignore[attr-defined]
                        output = self._model(t)
                else:
                    output = self._model(t)

            # Cache store: save neck features after full inference
            if cache is not None and sid and self._last_neck_output is not None:
                cache.update(sid, tensor.data, self._last_neck_output)
                self._last_neck_output = None

            return output.cpu().float().numpy()
        except Exception as exc:
            raise InferenceError(f"PyTorchBackend: inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release model reference and free device memory."""
        if self._neck_hook_handle is not None:
            self._neck_hook_handle.remove()
            self._neck_hook_handle = None
        self._last_neck_output = None
        self._current_source_id = ""
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
