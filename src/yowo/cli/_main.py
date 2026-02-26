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
@click.option(
    "--output-format",
    default="auto",
    type=click.Choice(["auto", "json", "image"]),
    help="Output format. 'auto' infers from -o file extension.",
)
@click.option("--save-frames", default=None, type=click.Path())
@click.option(
    "--preset",
    is_flag=True,
    default=False,
    help="Auto-tune config for detected device and source type",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Stream detections as JSONL to stdout (one JSON object per detection)",
)
@click.option(
    "--no-metrics",
    is_flag=True,
    default=False,
    help="Disable metrics collection (saves ~2µs/frame on critical paths)",
)
@click.pass_context
def detect_command(
    ctx: click.Context,
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
    output_format: str,
    save_frames: str | None,
    preset: bool,
    json_output: bool,
    no_metrics: bool,
) -> None:
    """Run object detection on SOURCE (image/video/RTSP/directory)."""
    from yowo.config import InferenceConfig
    from yowo.engine import InferenceEngine
    from yowo.io import open_source

    spec = _parse_model_spec(model)
    weights_path = Path(weights) if weights else spec.weights_path

    if preset:
        from click.core import ParameterSource

        from yowo.config import classify_source, preset_config
        from yowo.hardware import get_hardware_profile

        hw = get_hardware_profile()
        source_cat = classify_source(source)

        def _is_explicit(name: str) -> bool:
            return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE

        # Collect explicit CLI overrides (user-provided values only)
        cli_overrides: dict[str, object] = {
            "model_family": spec.family,
            "model_size": spec.size,
            "weights_path": weights_path,
        }
        if _is_explicit("backend"):
            cli_overrides["backend"] = BackendType(backend) if backend != "auto" else None
        if _is_explicit("device"):
            cli_overrides["device"] = device
        if _is_explicit("precision"):
            cli_overrides["precision"] = Precision(precision) if precision != "auto" else None
        if _is_explicit("batch"):
            cli_overrides["batch_size"] = batch
        if _is_explicit("confidence"):
            cli_overrides["confidence_threshold"] = confidence
        if _is_explicit("iou"):
            cli_overrides["iou_threshold"] = iou

        config = preset_config(hw, source_cat, **cli_overrides)
        # Propagate metrics flag into preset-derived config
        if no_metrics:
            from dataclasses import replace

            config = replace(config, metrics_enabled=False)
    else:
        config = InferenceConfig(
            model_family=spec.family,
            model_size=spec.size,
            weights_path=weights_path,
            batch_size=batch,
            confidence_threshold=confidence,
            iou_threshold=iou,
            backend=BackendType(backend) if backend != "auto" else None,
            device=device,
            precision=Precision(precision) if precision != "auto" else None,
            metrics_enabled=not no_metrics,
        )

    detections = []
    try:
        with InferenceEngine(config) as engine:
            src = open_source(source)
            for det in engine.stream(src):
                detections.append(det)
                if json_output:
                    click.echo(det.to_json())
                else:
                    click.echo(
                        f"Frame {det.frame.frame_index}: {det.num_boxes} detections "
                        f"({det.inference_time_ms:.1f}ms)"
                    )
            if not no_metrics and not json_output:
                m = engine.metrics
                click.echo(
                    f"\nMetrics: {m.frames_total} frames  {m.fps:.1f} FPS  "
                    f"p50={m.inference_p50_ms:.1f}ms  p95={m.inference_p95_ms:.1f}ms  "
                    f"p99={m.inference_p99_ms:.1f}ms  errors={m.errors_total}"
                )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if output:
        out_path = Path(output)
        fmt = output_format
        if fmt == "auto":
            fmt = (
                "image"
                if out_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                else "json"
            )
        if fmt == "image":
            from yowo.io import write_annotated_frame

            for det in detections:
                write_annotated_frame(det, out_path)
        else:
            _write_json(detections, out_path)
        click.echo(f"Saved detections to {output}")
    if save_frames:
        from yowo.io import write_annotated_frames

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
    type=click.Choice(["onnx", "tensorrt", "openvino", "coreml"]),
)
@click.option(
    "--precision",
    "-p",
    default="fp16",
    type=click.Choice(["fp32", "fp16", "int8"]),
)
@click.option("--calibration-data", default=None, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, type=click.Path())
@click.option("--dynamic-batch/--no-dynamic-batch", default=True)
@click.option("--imgsz", default=640, type=int)
@click.option(
    "--batch-sizes",
    default=None,
    help="Comma-separated batch sizes for CoreML EnumeratedShapes (e.g. '1,4,8').",
)
def export_command(
    model: str,
    weights: str | None,
    fmt: str,
    precision: str,
    calibration_data: str | None,
    output_dir: str | None,
    dynamic_batch: bool,
    imgsz: int,
    batch_sizes: str | None,
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

    parsed_batch_sizes: list[int] | None = None
    if batch_sizes:
        try:
            parsed_batch_sizes = [int(x.strip()) for x in batch_sizes.split(",")]
        except ValueError:
            click.echo("--batch-sizes must be comma-separated integers (e.g. '1,4,8')", err=True)
            sys.exit(1)

    try:
        meta = export_model(
            spec,
            ExportFormat(fmt),
            out_dir,
            precision=Precision(precision),
            dynamic_batch=dynamic_batch,
            imgsz=imgsz,
            calibration_data=calibration_data,
            batch_sizes=parsed_batch_sizes,
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
    """Serialise detections to a JSON file via Detection.to_dict()."""
    import json

    from yowo.types import Detection

    out = [det.to_dict() for det in detections if isinstance(det, Detection)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


__all__ = ["cli"]
