"""OpenVINO inference backend.

Loads OpenVINO IR format (a ``.xml`` + ``.bin`` pair, or a directory
containing them). Compiles the model for the target device via
``openvino.Core.compile_model()``.

All SDK imports are deferred inside load() so this module can be imported
on machines without OpenVINO installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from yowo.errors import BackendLoadError, DependencyError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType, PreprocessedTensor

if TYPE_CHECKING:
    pass

__all__ = ["OpenVinoBackend"]


class OpenVinoBackend:
    """OpenVINO backend using ``openvino.Core``.

    Supports ``.xml`` files directly or a directory that contains a single
    ``.xml`` file (e.g. the ``_openvino_model/`` export directory produced
    by ultralytics).

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

            # Derive input shape from compiled model
            input_node = next(iter(self._compiled_model.inputs), None)  # type: ignore[attr-defined]
            if input_node is not None:
                shape = input_node.shape
                if len(shape) == 4:
                    self._input_shape = (int(shape[2]), int(shape[3]))
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
            results = self._infer_request.infer({0: tensor.data})  # type: ignore[attr-defined]
            # ``results`` is a dict of {output_name: ndarray}; take the first
            output = next(iter(results.values()))
            return np.asarray(output, dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"OpenVinoBackend: inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release compiled model and infer request."""
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
            self._infer_request.infer({0: dummy})  # type: ignore[attr-defined]
        except Exception:
            # Warmup failures are non-fatal
            pass

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
                raise BackendLoadError(
                    f"OpenVinoBackend: no .xml file found in directory: {path}"
                )
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
