"""CoreML backend for Apple Silicon Neural Engine inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

logger = logging.getLogger(__name__)

__all__ = ["CoreMLBackend"]


class CoreMLBackend:
    """CoreML inference backend for Apple Silicon.

    Uses coremltools to load ``.mlpackage`` models and run inference
    on the Apple Neural Engine, GPU, and CPU.

    Follows the ``InferenceBackend`` Protocol structurally —
    no base class required.
    """

    def __init__(self, hw_profile: HardwareProfile) -> None:
        if not hw_profile.libraries.coremltools_version:
            raise DependencyError("coremltools", "uv add coremltools>=7.0")

        self._hw = hw_profile
        self._model: Any = None
        self._input_shape: tuple[int, int] = (640, 640)
        self._input_name: str = "images"
        self._output_name: str = "output0"

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.COREML

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
        """Load a CoreML ``.mlpackage`` model.

        Args:
            model_path: Path to a ``.mlpackage`` directory.
            device: Ignored — CoreML always uses ``ALL`` compute units
                (Neural Engine + GPU + CPU).

        Raises:
            BackendLoadError: Model file invalid or cannot be loaded.
            DependencyError: coremltools not installed.
        """
        try:
            import coremltools as ct  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError("coremltools", "uv add coremltools>=7.0") from exc

        path = Path(model_path)
        if not path.exists():
            raise BackendLoadError(f"CoreML model not found: {path}")

        try:
            # Use ALL compute units (Neural Engine + GPU + CPU)
            self._model = ct.models.MLModel(
                str(path),
                compute_units=ct.ComputeUnit.ALL,
            )

            # Derive input shape from model spec
            spec = self._model.get_spec()
            for inp in spec.description.input:
                if inp.name == self._input_name or inp.type.HasField("imageType"):
                    # Try multiarray first
                    if inp.type.HasField("multiArrayType"):
                        shape = list(inp.type.multiArrayType.shape)
                        if len(shape) == 4:  # BCHW
                            self._input_shape = (int(shape[2]), int(shape[3]))
                    elif inp.type.HasField("imageType"):
                        self._input_shape = (
                            int(inp.type.imageType.height),
                            int(inp.type.imageType.width),
                        )
                    break

            logger.info(
                "CoreML model loaded: %s (input: %dx%d)",
                path.name,
                self._input_shape[0],
                self._input_shape[1],
            )
        except Exception as exc:
            self._model = None
            raise BackendLoadError(f"Failed to load CoreML model: {exc}") from exc

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference on preprocessed tensor batch.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            Raw model output as float32 ndarray.

        Raises:
            InferenceError: Inference failed.
        """
        if self._model is None:
            raise InferenceError("CoreML backend not loaded. Call load() first.")

        try:
            # CoreML prediction — input as numpy array
            prediction = self._model.predict({self._input_name: tensor.data})

            # Extract primary output
            output_key = self._output_name
            if output_key not in prediction:
                # Fallback: use first output key
                output_key = next(iter(prediction))

            result = prediction[output_key]
            return np.asarray(result, dtype=np.float32)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"CoreML inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release model resources."""
        self._model = None

    def warmup(self, batch_size: int = 1) -> None:
        """Run dummy inference to prime the Neural Engine."""
        if self._model is None:
            return
        try:
            h, w = self._input_shape
            dummy = np.zeros((batch_size, 3, h, w), dtype=np.float32)
            dummy_tensor = PreprocessedTensor(
                data=dummy,
                original_shapes=tuple((h, w) for _ in range(batch_size)),
                input_shape=(h, w),
                scale_factors=tuple((1.0, 1.0) for _ in range(batch_size)),
                pad_offsets=tuple((0, 0) for _ in range(batch_size)),
            )
            self.infer(dummy_tensor)
        except Exception as exc:
            logger.debug("CoreML warmup failed (non-fatal): %s", exc)

    def clear_kv_cache(self) -> None:
        """No-op — CoreML backend does not support KV caching."""

    def set_source_id(self, source_id: str) -> None:
        """No-op — CoreML backend does not support feature caching."""
