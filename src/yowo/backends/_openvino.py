"""OpenVINO inference backend.

Loads OpenVINO IR format (a ``.xml`` + ``.bin`` pair, or a directory
containing them). Compiles the model for the target device via
``openvino.Core.compile_model()``.

All SDK imports are deferred inside load() so this module can be imported
on machines without OpenVINO installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

__all__ = ["OpenVinoBackend"]


class OpenVinoBackend:
    """OpenVINO backend using ``openvino.Core``.

    Supports ``.xml`` files directly or a directory that contains a single
    ``.xml`` file (e.g. the ``_openvino_model/`` export directory).

    Compiled for ``"GPU"`` when an Intel GPU appears in
    ``Core.available_devices``, otherwise ``"CPU"``.
    """

    def __init__(self, hw_profile: HardwareProfile) -> None:
        if not hw_profile.libraries.openvino_version:
            raise DependencyError("openvino", "uv add openvino")

        self._hw = hw_profile
        self._compiled_model: object | None = None
        self._infer_request: object | None = None
        self._input_shape: tuple[int, int] = (640, 640)
        self._input_name: str = "images"
        # KV cache I/O state (populated in load() when model has KV I/O)
        self._has_kv_io: bool = False
        self._kv_state: dict[str, NDArray[np.float32]] = {}
        self._kv_input_names: list[str] = []
        self._kv_output_names: list[str] = []
        self._kv_shapes: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENVINO

    @property
    def is_loaded(self) -> bool:
        return self._compiled_model is not None

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def load(self, model_path: str | Path, *, device: str = "auto") -> None:
        """Load and compile an OpenVINO IR model.

        Args:
            model_path: Path to a ``.xml`` file or a directory containing one.
            device: ``"auto"`` picks GPU when available, otherwise CPU.
                    Pass ``"cpu"`` to force CPU inference.

        Raises:
            DependencyError: ``openvino`` not installed.
            BackendLoadError: Model load or compilation failed.
        """
        try:
            from openvino.runtime import Core  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError("openvino", "uv add openvino") from exc

        xml_path = self._resolve_xml(Path(model_path))

        try:
            core = Core()
            model = core.read_model(str(xml_path))

            ov_device = self._select_ov_device(core, device)
            self._compiled_model = core.compile_model(model, ov_device)
            self._infer_request = self._compiled_model.create_infer_request()  # type: ignore[attr-defined]

            # Derive input shape and detect KV cache I/O
            all_inputs = list(self._compiled_model.inputs)  # type: ignore[attr-defined]
            if all_inputs:
                input_node = all_inputs[0]
                self._input_name = input_node.get_any_name()  # type: ignore[attr-defined]
                shape = input_node.shape
                if len(shape) == 4:
                    self._input_shape = (int(shape[2]), int(shape[3]))
            kv_inputs = [
                n
                for n in all_inputs
                if n.get_any_name().startswith("past_")  # type: ignore[attr-defined]
            ]
            if kv_inputs:
                self._has_kv_io = True
                self._kv_input_names = [n.get_any_name() for n in kv_inputs]  # type: ignore[attr-defined]
                self._kv_shapes = {
                    n.get_any_name(): tuple(int(d) for d in n.shape)  # type: ignore[attr-defined]
                    for n in kv_inputs
                }
                self._kv_output_names = [
                    n.get_any_name()  # type: ignore[attr-defined]
                    for n in self._compiled_model.outputs  # type: ignore[attr-defined]
                    if n.get_any_name().startswith("present_")  # type: ignore[attr-defined]
                ]
        except Exception as exc:
            self._compiled_model = None
            self._infer_request = None
            raise BackendLoadError(f"OpenVinoBackend: load failed: {exc}") from exc

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Run inference via an OpenVINO InferRequest.

        Args:
            tensor: BCHW float32 array in [0, 1].

        Returns:
            First output tensor as float32.

        Raises:
            InferenceError: Model not loaded or runtime failure.
        """
        if self._infer_request is None:
            raise InferenceError("OpenVinoBackend: no model loaded — call load() first")

        try:
            if self._has_kv_io:
                use_cache = np.float32(1.0 if self._kv_state else 0.0)
                feed: dict[str, NDArray[np.float32]] = {
                    self._input_name: tensor.data,
                    "use_cache": np.array(use_cache, dtype=np.float32),
                }
                for name in self._kv_input_names:
                    feed[name] = self._kv_state.get(
                        name,
                        np.zeros(self._kv_shapes[name], dtype=np.float32),
                    )
                results = self._infer_request.infer(feed)  # type: ignore[attr-defined]
                # Collect present K,V into kv_state for the next frame
                for present_name in self._kv_output_names:
                    past_name = present_name.replace("present_", "past_")
                    self._kv_state[past_name] = np.asarray(results[present_name], dtype=np.float32)
            else:
                results = self._infer_request.infer({0: tensor.data})  # type: ignore[attr-defined]
            # First output is always the detection tensor
            output = next(iter(results.values()))
            return np.asarray(output, dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"OpenVinoBackend: inference failed: {exc}") from exc

    def clear_kv_cache(self) -> None:
        """Clear KV state for streaming reset (e.g. new video source)."""
        self._kv_state = {}

    def unload(self) -> None:
        """Release compiled model and infer request."""
        self._kv_state = {}
        self._compiled_model = None
        self._infer_request = None

    def warmup(self, batch_size: int = 1) -> None:
        """Run a dummy inference to prime OpenVINO caches.

        No-op when the model is not loaded.
        """
        if self._infer_request is None:
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
                self._infer_request.infer(feed)  # type: ignore[attr-defined]
            else:
                self._infer_request.infer({0: dummy})  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("OpenVinoBackend: warmup failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_xml(self, path: Path) -> Path:
        """Return the ``.xml`` path.

        If ``path`` is a directory, search for exactly one ``.xml`` file.
        If ``path`` is already a ``.xml`` file, return it directly.

        Raises:
            BackendLoadError: No ``.xml`` file found in the directory.
        """
        if path.is_dir():
            candidates = list(path.glob("*.xml"))
            if not candidates:
                raise BackendLoadError(f"OpenVinoBackend: no .xml file found in directory: {path}")
            return candidates[0]
        return path

    def _select_ov_device(self, core: object, device: str) -> str:
        """Select the OpenVINO device string.

        Args:
            core: ``openvino.runtime.Core`` instance.
            device: Requested device string (``"auto"``, ``"cpu"``, ``"gpu"``).

        Returns:
            ``"GPU"`` or ``"CPU"`` as accepted by ``Core.compile_model()``.
        """
        if device.lower() == "cpu":
            return "CPU"

        available: list[str] = getattr(core, "available_devices", [])
        if "GPU" in available:
            return "GPU"
        return "CPU"
