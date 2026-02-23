"""Core export orchestration (wraps ultralytics)."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from yowo.errors import ConfigError, DependencyError, ExportError
from yowo.export._calibration import resolve_calibration_source
from yowo.export._metadata import ExportMetadata
from yowo.hardware import get_hardware_profile
from yowo.models import resolve_weights
from yowo.types import ExportFormat, ModelSpec, Precision

logger = logging.getLogger(__name__)

_FORMAT_TO_ULTRALYTICS: dict[ExportFormat, str] = {
    ExportFormat.ONNX: "onnx",
    ExportFormat.TENSORRT: "engine",
    ExportFormat.OPENVINO: "openvino",
}


def export_model(
    spec: ModelSpec,
    target_format: ExportFormat,
    output_dir: Path,
    *,
    precision: Precision = Precision.FP16,
    dynamic_batch: bool = False,
    imgsz: int = 640,
    calibration_data: str | None = None,
) -> ExportMetadata:
    """Export a YOLO model to an optimized format.

    Wraps ultralytics.YOLO.export() with:
    - Dependency checking with install hints
    - Calibration data resolution for INT8
    - Metadata sidecar file (.yowo.json)

    Args:
        spec: Model to export.
        target_format: ONNX, TensorRT, or OpenVINO.
        output_dir: Where to write the exported model.
        precision: FP32, FP16, or INT8.
        dynamic_batch: Enable dynamic batch dimension (ONNX only).
        imgsz: Input image size.
        calibration_data: Required for INT8; path to YAML or image directory.

    Returns:
        ExportMetadata record with file path and sidecar written to disk.

    Raises:
        ConfigError: INT8 requested without calibration_data.
        DependencyError: ultralytics not installed.
        ExportError: Export operation failed or produced no output file.
    """
    if precision == Precision.INT8 and calibration_data is None:
        raise ConfigError("INT8 export requires --calibration-data")

    try:
        import ultralytics
    except ImportError as exc:
        raise DependencyError("ultralytics", "uv add ultralytics") from exc

    weights_path = resolve_weights(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    ultra_format = _FORMAT_TO_ULTRALYTICS[target_format]
    kwargs: dict[str, object] = {
        "format": ultra_format,
        "imgsz": imgsz,
        "dynamic": dynamic_batch,
    }

    if target_format == ExportFormat.ONNX:
        kwargs["simplify"] = True
        kwargs["opset"] = 17

    if precision == Precision.FP16:
        kwargs["half"] = True
    elif precision == Precision.INT8:
        kwargs["int8"] = True
        assert calibration_data is not None  # already checked above
        kwargs["data"] = resolve_calibration_source(calibration_data)

    logger.info(
        "Exporting %s%s -> %s (%s)",
        spec.family.value,
        spec.size.value,
        target_format.value,
        precision.value,
    )

    t0 = time.monotonic()
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        exported = model.export(**kwargs)
    except Exception as exc:
        raise ExportError(f"Export failed: {exc}") from exc
    elapsed = time.monotonic() - t0

    exported_path = Path(str(exported))
    if not exported_path.exists():
        raise ExportError(f"Export produced no file at {exported_path}")

    hw = get_hardware_profile()
    size_bytes = (
        _dir_size(exported_path) if exported_path.is_dir() else exported_path.stat().st_size
    )

    meta = ExportMetadata(
        model_name=f"{spec.family.value}{spec.size.value}",
        format=target_format.value,
        precision=precision.value,
        imgsz=imgsz,
        batch_size=1,
        dynamic=dynamic_batch,
        input_shape=[1, 3, imgsz, imgsz],
        file_path=str(exported_path.resolve()),
        file_size_bytes=size_bytes,
        created_at=datetime.now(UTC).isoformat(),
        export_duration_sec=round(elapsed, 2),
        source_weights=str(weights_path),
        ultralytics_version=ultralytics.__version__,
        gpu_name=hw.primary_gpu.name if hw.primary_gpu else None,
        calibration_data=calibration_data,
    )
    meta.save()

    logger.info(
        "Export complete: %s (%.1f MB, %.1fs)",
        exported_path.name,
        size_bytes / 1_048_576,
        elapsed,
    )
    return meta


def _dir_size(path: Path) -> int:
    """Return total byte size of all files in a directory tree."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


__all__ = ["export_model"]
