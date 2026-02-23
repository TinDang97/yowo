"""Model export pipeline.

Exports YOLO models to ONNX, TensorRT, and OpenVINO formats using
native ``torch.onnx.export`` with calibration handling and metadata sidecars.

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
