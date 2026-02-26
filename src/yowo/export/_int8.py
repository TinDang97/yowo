"""INT8 quantization support for TensorRT and ONNX Runtime.

Provides:
- ``create_tensorrt_calibrator``: Factory for TensorRT INT8 entropy calibrator.
- ``quantize_onnx_static``: Static INT8 quantization for ONNX models via
  ``onnxruntime.quantization``.

Both paths use ``calibration_batches()`` from ``_calibration`` as the shared
image preprocessing pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from yowo.errors import DependencyError, ExportError
from yowo.export._calibration import calibration_batches, resolve_calibration_images

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TensorRT INT8 calibrator
# ---------------------------------------------------------------------------


def create_tensorrt_calibrator(
    image_paths: list[Path],
    batch_size: int = 8,
    input_size: int = 640,
    cache_file: Path | None = None,
) -> object:
    """Create a TensorRT INT8 entropy calibrator.

    Returns an object implementing ``trt.IInt8EntropyCalibrator2``.
    The ``tensorrt`` import is deferred to call time so the module can be
    imported on machines without TensorRT installed.

    Args:
        image_paths: Calibration image file paths (from
            ``resolve_calibration_images``).
        batch_size: Images per calibration batch.
        input_size: Model input spatial size (square).
        cache_file: Optional path to read/write a calibration cache for
            faster repeated builds.

    Returns:
        A calibrator instance compatible with
        ``trt.BuilderConfig.int8_calibrator``.

    Raises:
        DependencyError: ``tensorrt`` or ``pycuda`` is not installed.
    """
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "tensorrt",
            "pip install tensorrt>=10.0 --extra-index-url https://pypi.nvidia.com",
        ) from exc

    try:
        import pycuda.autoinit as _autoinit  # type: ignore[import-untyped]  # noqa: F401
        import pycuda.driver as cuda  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "pycuda",
            "pip install pycuda",
            message="pycuda is required for TensorRT INT8 calibration.",
        ) from exc

    class _Calibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
        """Entropy calibrator feeding preprocessed image batches to TensorRT."""

        def __init__(self) -> None:
            super().__init__()
            self._batch_iter: Iterator[Any] | None = None
            self._device_buffer: Any = None

        def get_batch_size(self) -> int:
            return batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self._batch_iter is None:
                self._batch_iter = calibration_batches(
                    image_paths, batch_size=batch_size, input_size=input_size
                )
            try:
                batch = next(self._batch_iter)
            except StopIteration:
                return None

            if self._device_buffer is None:
                self._device_buffer = cuda.mem_alloc(batch.nbytes)  # type: ignore[attr-defined]
            cuda.memcpy_htod(self._device_buffer, batch)  # type: ignore[attr-defined]
            return [int(self._device_buffer)]

        def read_calibration_cache(self) -> bytes | None:
            if cache_file is not None and cache_file.exists():
                logger.info("Reading INT8 calibration cache: %s", cache_file)
                return cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache: memoryview) -> None:
            if cache_file is not None:
                cache_file.write_bytes(bytes(cache))
                logger.info("Wrote INT8 calibration cache: %s", cache_file)

    return _Calibrator()


# ---------------------------------------------------------------------------
# ONNX static INT8 quantization
# ---------------------------------------------------------------------------


def quantize_onnx_static(
    onnx_path: Path,
    output_path: Path,
    calibration_data: str,
    *,
    input_size: int = 640,
    batch_size: int = 8,
) -> Path:
    """Quantize an ONNX model to INT8 using static calibration.

    Uses ``onnxruntime.quantization.quantize_static`` with entropy calibration.
    Calibration images are preprocessed via ``calibration_batches()``.

    Args:
        onnx_path: Path to the source FP32/FP16 ONNX model.
        output_path: Destination path for the quantized INT8 ONNX model.
        calibration_data: Path to calibration image directory.
        input_size: Model input spatial size (square).
        batch_size: Images per calibration batch.

    Returns:
        The ``output_path`` on success.

    Raises:
        DependencyError: ``onnxruntime`` is not installed.
        ExportError: Quantization failed at runtime.
    """
    try:
        from onnxruntime import quantization as ort_quantization  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "onnxruntime",
            "pip install onnxruntime>=1.17",
            message="onnxruntime.quantization is required for ONNX INT8 export.",
        ) from exc

    images = resolve_calibration_images(calibration_data)

    class _CalibrationReader(ort_quantization.CalibrationDataReader):  # type: ignore[misc]
        """Feeds preprocessed batches to ORT quantization."""

        def __init__(self) -> None:
            self._batch_iter = calibration_batches(
                images, batch_size=batch_size, input_size=input_size
            )

        def get_next(self) -> dict[str, Any] | None:  # type: ignore[override]
            try:
                batch = next(self._batch_iter)
            except StopIteration:
                return None
            return {"images": batch}

    try:
        ort_quantization.quantize_static(
            str(onnx_path),
            str(output_path),
            _CalibrationReader(),
            quant_format=ort_quantization.QuantFormat.QDQ,
            activation_type=ort_quantization.QuantType.QInt8,
            weight_type=ort_quantization.QuantType.QInt8,
            calibrate_method=ort_quantization.CalibrationMethod.Entropy,
        )
    except Exception as exc:
        raise ExportError(f"ONNX INT8 quantization failed: {exc}") from exc

    logger.info(
        "ONNX INT8 quantization complete: %s -> %s",
        onnx_path.name,
        output_path.name,
    )
    return output_path


__all__ = ["create_tensorrt_calibrator", "quantize_onnx_static"]
