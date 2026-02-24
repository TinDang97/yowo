"""CLI entry point for yowo."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from yowo.types import BackendType, ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision


@click.group()
@click.version_option(package_name="yowo")
def cli() -> None:
    """yowo - Production YOLO inference and export."""


@cli.command("detect")
@click.argument("source")
@click.option("--model", "-m", default="yolo26n", help="Model name, e.g. yolo26n")
@click.option(
    "--weights",
    "-w",
    default=None,
    type=click.Path(exists=True),
    help="Path to local .pt weights file (skips download)",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "pytorch", "onnx", "tensorrt", "openvino"]),
)
@click.option("--device", default="auto")
@click.option(
    "--precision",
    default="auto",
    type=click.Choice(["auto", "fp32", "fp16", "int8"]),
)
@click.option("--confidence", default=0.25, type=float)
@click.option("--iou", default=0.45, type=float)
@click.option("--batch", default=1, type=int)
@click.option("--output", "-o", default=None, type=click.Path())
@click.option("--save-frames", default=None, type=click.Path())
def detect_command(
    source: str,
    model: str,
    weights: str | None,
    backend: str,
    device: str,
    precision: str,
    confidence: float,
    iou: float,
    batch: int,
    output: str | None,
    save_frames: str | None,
) -> None:
    """Run object detection on SOURCE (image/video/RTSP/directory)."""
    from yowo.engine import InferenceEngine
    from yowo.io._source import open_source

    spec = _parse_model_spec(model)
    if weights:
        spec = ModelSpec(spec.family, spec.size, spec.task, Path(weights))

    engine_kwargs: dict[str, object] = dict(
        batch_size=batch,
        confidence=confidence,
        iou_threshold=iou,
    )
    if backend != "auto":
        engine_kwargs["backend"] = BackendType(backend)
    if device != "auto":
        engine_kwargs["device"] = device
    if precision != "auto":
        engine_kwargs["precision"] = Precision(precision)

    detections = []
    try:
        with InferenceEngine(spec, **engine_kwargs) as engine:  # type: ignore[arg-type]
            src = open_source(source)
            for det in engine.stream(src):
                detections.append(det)
                click.echo(
                    f"Frame {det.frame.frame_index}: {det.num_boxes} detections "
                    f"({det.inference_time_ms:.1f}ms)"
                )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if output:
        out_path = Path(output)
        if out_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            from yowo.io._sink import write_annotated_frame

            for det in detections:
                write_annotated_frame(det, out_path)
        else:
            _write_json(detections, out_path)
        click.echo(f"Saved detections to {output}")
    if save_frames:
        from yowo.io._sink import write_annotated_frames

        out_dir = Path(save_frames)
        write_annotated_frames(detections, out_dir)
        click.echo(f"Saved {len(detections)} annotated frame(s) to {out_dir}/")


@cli.command("export")
@click.argument("model")
@click.option(
    "--weights",
    "-w",
    default=None,
    type=click.Path(exists=True),
    help="Path to local .pt weights file (skips download)",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    required=True,
    type=click.Choice(["onnx", "tensorrt", "openvino"]),
)
@click.option(
    "--precision",
    "-p",
    default="fp16",
    type=click.Choice(["fp32", "fp16", "int8"]),
)
@click.option("--calibration-data", default=None, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, type=click.Path())
@click.option("--dynamic-batch/--no-dynamic-batch", default=False)
@click.option("--imgsz", default=640, type=int)
def export_command(
    model: str,
    weights: str | None,
    fmt: str,
    precision: str,
    calibration_data: str | None,
    output_dir: str | None,
    dynamic_batch: bool,
    imgsz: int,
) -> None:
    """Export MODEL to an optimized inference format."""
    from yowo.export import export_model

    spec = _parse_model_spec(model)
    if weights:
        spec = ModelSpec(spec.family, spec.size, spec.task, Path(weights))
    out_dir = (
        Path(output_dir)
        if output_dir
        else Path.home() / ".yowo" / "models" / model / f"{fmt}_{precision}"
    )

    try:
        meta = export_model(
            spec,
            ExportFormat(fmt),
            out_dir,
            precision=Precision(precision),
            dynamic_batch=dynamic_batch,
            imgsz=imgsz,
            calibration_data=calibration_data,
        )
        click.echo(f"Exported: {meta.file_path}")
        click.echo(f"Size: {meta.file_size_bytes / 1_048_576:.1f} MB")
        click.echo(f"Duration: {meta.export_duration_sec:.1f}s")
    except Exception as exc:
        click.echo(f"Export failed: {exc}", err=True)
        sys.exit(1)


@cli.command("info")
def info_command() -> None:
    """Print hardware, backends, and installed library versions."""
    from yowo.hardware import get_hardware_profile

    hw = get_hardware_profile()
    click.echo("=== Hardware ===")
    click.echo(f"CPU: {hw.cpu}")
    for i, gpu in enumerate(hw.gpus):
        click.echo(f"GPU {i}: {gpu}")
    click.echo(f"CPU features: {', '.join(sorted(hw.cpu_features)) or 'none'}")
    click.echo("")
    click.echo("=== Libraries ===")
    libs = hw.libraries
    click.echo(f"torch:        {libs.torch_version or 'not installed'}")
    click.echo(f"cuda:         {libs.cuda_version or 'not available'}")
    click.echo(f"tensorrt:     {libs.tensorrt_version or 'not installed'}")
    ort_line = f"onnxruntime:  {libs.onnxruntime_version or 'not installed'}"
    if libs.onnxruntime_version:
        if libs.onnxruntime_has_cuda:
            ort_line += " (CUDA)"
        elif libs.onnxruntime_has_coreml:
            ort_line += " (CoreML)"
        else:
            ort_line += " (CPU)"
    click.echo(ort_line)
    click.echo(f"openvino:     {libs.openvino_version or 'not installed'}")


@cli.command("models")
@click.option("--family", default=None)
def models_command(family: str | None) -> None:
    """List registered model variants."""
    from yowo.models import list_available

    models = list_available()
    if family:
        models = [m for m in models if m.family.value == family]

    click.echo(f"{'Model':<12} {'Input':<10} {'Classes':<10} {'URL'}")
    click.echo("-" * 80)
    for m in models:
        name = f"{m.family.value}{m.size.value}"
        click.echo(
            f"{name:<12} {m.input_height}x{m.input_width:<5} "
            f"{m.num_classes:<10} {m.default_weights_url}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_model_spec(model_name: str) -> ModelSpec:
    """Parse 'yolo26n' -> ModelSpec(YOLO26, NANO).

    Raises:
        click.BadParameter: On unknown model name format.
    """
    size_map: dict[str, ModelSize] = {
        "n": ModelSize.NANO,
        "s": ModelSize.SMALL,
        "m": ModelSize.MEDIUM,
        "l": ModelSize.LARGE,
        "x": ModelSize.XLARGE,
    }
    family_map: dict[str, ModelFamily] = {
        "yolo11": ModelFamily.YOLO11,
        "yolo26": ModelFamily.YOLO26,
    }

    for prefix, family in sorted(family_map.items(), key=lambda x: -len(x[0])):
        if model_name.startswith(prefix):
            suffix = model_name[len(prefix) :]
            if suffix in size_map:
                return ModelSpec(family, size_map[suffix])

    raise click.BadParameter(
        f"Unknown model: {model_name!r}. Expected format: yolo{{11|26}}{{n|s|m|l|x}}, e.g. yolo26n"
    )


def _write_json(detections: list[object], path: Path) -> None:
    """Serialise detections to a JSON file."""
    import json

    from yowo.types import Detection

    out = []
    for det in detections:
        if not isinstance(det, Detection):
            continue
        out.append(
            {
                "frame_index": det.frame.frame_index,
                "source_id": det.frame.source_id,
                "inference_time_ms": det.inference_time_ms,
                "backend": det.backend.value,
                "model": f"{det.model_spec.family.value}{det.model_spec.size.value}",
                "boxes": [
                    {
                        "x1": b.x1,
                        "y1": b.y1,
                        "x2": b.x2,
                        "y2": b.y2,
                        "confidence": b.confidence,
                        "class_id": b.class_id,
                        "class_name": b.class_name,
                    }
                    for b in det.boxes
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


__all__ = ["cli"]
