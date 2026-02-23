"""Model export pipeline.

Wraps ultralytics.YOLO.export() with validation, calibration handling,
and metadata sidecar files.

Usage::

    from yowo.export import export_model
    from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

    meta = export_model(
        ModelSpec(ModelFamily.YOLO26, ModelSize.NANO),
        ExportFormat.ONNX,
        output_dir=Path("./exports"),
        precision=Precision.FP16,
    )
    print(meta.file_path)
"""

from yowo.export._exporter import export_model
from yowo.export._metadata import ExportMetadata

__all__ = ["ExportMetadata", "export_model"]
