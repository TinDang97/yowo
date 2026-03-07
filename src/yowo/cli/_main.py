"""CLI entry point for yowo."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from yowo.types import BackendType, ExportFormat, ModelSpec, Precision


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
    help="Path to local model file (.pt for PyTorch, .onnx for ONNX backend)",
)
@click.option(
    "--num-classes",
    "num_classes",
    default=None,
    type=int,
    help="Override output class count.",
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
@click.option(
    "--auto-letterbox/--no-auto-letterbox",
    "auto_letterbox",
    default=False,
    help="Stride-aligned non-square tensors (reduces pixel count on 16:9 input)",
)
@click.pass_context
def detect_command(
    ctx: click.Context,
    source: str,
    model: str,
    weights: str | None,
    num_classes: int | None,
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
    auto_letterbox: bool,
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
            "num_classes": num_classes,
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

        if auto_letterbox:
            cli_overrides["auto_letterbox"] = True
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
            num_classes=num_classes,
            batch_size=batch,
            confidence_threshold=confidence,
            iou_threshold=iou,
            backend=BackendType(backend) if backend != "auto" else None,
            device=device,
            precision=Precision(precision) if precision != "auto" else None,
            metrics_enabled=not no_metrics,
            auto_letterbox=auto_letterbox,
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
        click.echo(f"Saved detections to {output}", err=json_output)
    if save_frames:
        from yowo.io import write_annotated_frames

        out_dir = Path(save_frames)
        write_annotated_frames(detections, out_dir)
        click.echo(f"Saved {len(detections)} annotated frame(s) to {out_dir}/", err=json_output)


@cli.command("export")
@click.argument("model")
@click.option(
    "--weights",
    "-w",
    default=None,
    type=click.Path(exists=True),
    help="Path to local model file (.pt for PyTorch, .onnx for ONNX backend)",
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
@click.option("--dynamic-batch/--no-dynamic-batch", default=False)
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


@cli.command("track")
@click.argument("source")
@click.option("--model", "-m", default="yolo26n")
@click.option("--weights", "-w", default=None, type=click.Path(exists=True))
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "pytorch", "onnx", "tensorrt", "openvino"]),
)
@click.option("--device", default="auto")
@click.option("--precision", default="auto", type=click.Choice(["auto", "fp32", "fp16", "int8"]))
@click.option("--confidence", default=0.25, type=float)
@click.option("--iou", default=0.45, type=float)
@click.option("--high-thresh", default=0.6, type=float, help="ByteTrack stage-1 confidence gate.")
@click.option("--low-thresh", default=0.1, type=float, help="ByteTrack stage-2 confidence gate.")
@click.option("--match-thresh", default=0.8, type=float, help="IoU distance threshold.")
@click.option("--max-age", default=30, type=int, help="Frames a lost track survives.")
@click.option("--min-hits", default=3, type=int, help="Hits before a track is confirmed.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Stream JSONL to stdout.")
@click.option(
    "--auto-letterbox/--no-auto-letterbox",
    "auto_letterbox",
    default=False,
    help="Stride-aligned non-square tensors (reduces pixel count on 16:9 input)",
)
def track_command(
    source: str,
    model: str,
    weights: str | None,
    backend: str,
    device: str,
    precision: str,
    confidence: float,
    iou: float,
    high_thresh: float,
    low_thresh: float,
    match_thresh: float,
    max_age: int,
    min_hits: int,
    json_output: bool,
    auto_letterbox: bool,
) -> None:
    """Run ByteTrack object tracking on SOURCE (image/video/RTSP/directory)."""
    from yowo.config import InferenceConfig
    from yowo.engine import InferenceEngine
    from yowo.io import open_source
    from yowo.tracking import ByteTracker, track_stream

    spec = _parse_model_spec(model)
    weights_path = Path(weights) if weights else spec.weights_path
    config = InferenceConfig(
        model_family=spec.family,
        model_size=spec.size,
        weights_path=weights_path,
        confidence_threshold=confidence,
        iou_threshold=iou,
        backend=BackendType(backend) if backend != "auto" else None,
        device=device,
        precision=Precision(precision) if precision != "auto" else None,
        auto_letterbox=auto_letterbox,
    )
    tracker = ByteTracker(
        track_high_thresh=high_thresh,
        track_low_thresh=low_thresh,
        match_thresh=match_thresh,
        max_age=max_age,
        min_hits=min_hits,
    )
    try:
        with InferenceEngine(config) as engine:
            src = open_source(source)
            for tracked in track_stream(engine, src, tracker=tracker):
                if json_output:
                    click.echo(tracked.to_json())
                else:
                    box_strs = [
                        f"#{b.track_id} {b.class_name}({b.confidence:.2f})"
                        + ("*" if b.is_confirmed else "")
                        for b in tracked.boxes
                    ]
                    click.echo(
                        f"Frame {tracked.frame.frame_index}: "
                        f"{tracked.num_boxes} tracks [{', '.join(box_strs)}] "
                        f"({tracked.tracking_time_ms:.2f}ms track)"
                    )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("count")
@click.argument("source")
@click.option("--model", "-m", default="yolo26n")
@click.option("--weights", "-w", default=None, type=click.Path(exists=True))
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "pytorch", "onnx", "tensorrt", "openvino"]),
)
@click.option("--device", default="auto")
@click.option("--precision", default="auto", type=click.Choice(["auto", "fp32", "fp16", "int8"]))
@click.option("--confidence", default=0.25, type=float)
@click.option("--iou", default=0.45, type=float)
@click.option(
    "--zone",
    "zone_file",
    default=None,
    type=click.Path(exists=True),
    help=(
        'JSON file defining polygon zones. Format: [{"zone_id": "name", "vertices": [[x,y], ...]}]'
    ),
)
@click.option(
    "--line",
    "line_file",
    default=None,
    type=click.Path(exists=True),
    help=(
        "JSON file defining counting lines. Requires --track. "
        'Format: [{"line_id": "name", "p1": [x,y], "p2": [x,y]}]'
    ),
)
@click.option(
    "--track",
    "use_tracking",
    is_flag=True,
    default=False,
    help="Enable ByteTrack (required for --line).",
)
@click.option("--json", "json_output", is_flag=True, default=False, help="Stream JSONL to stdout.")
@click.option(
    "--auto-letterbox/--no-auto-letterbox",
    "auto_letterbox",
    default=False,
    help="Stride-aligned non-square tensors (reduces pixel count on 16:9 input)",
)
def count_command(
    source: str,
    model: str,
    weights: str | None,
    backend: str,
    device: str,
    precision: str,
    confidence: float,
    iou: float,
    zone_file: str | None,
    line_file: str | None,
    use_tracking: bool,
    json_output: bool,
    auto_letterbox: bool,
) -> None:
    """Count detections by class, zone, or line crossing on SOURCE."""
    import json as json_mod

    from yowo.config import InferenceConfig
    from yowo.counter import ObjectCounter
    from yowo.engine import InferenceEngine
    from yowo.io import open_source

    if line_file and not use_tracking:
        click.echo("Error: --line requires --track (line crossing needs track IDs).", err=True)
        sys.exit(1)

    zones = _load_zones(zone_file)
    lines = _load_lines(line_file)

    spec = _parse_model_spec(model)
    weights_path = Path(weights) if weights else spec.weights_path
    config = InferenceConfig(
        model_family=spec.family,
        model_size=spec.size,
        weights_path=weights_path,
        confidence_threshold=confidence,
        iou_threshold=iou,
        backend=BackendType(backend) if backend != "auto" else None,
        device=device,
        precision=Precision(precision) if precision != "auto" else None,
        auto_letterbox=auto_letterbox,
    )
    counter = ObjectCounter(zones=zones, lines=lines)

    try:
        with InferenceEngine(config) as engine:
            src = open_source(source)
            if use_tracking:
                from yowo.tracking import track_stream

                stream = track_stream(engine, src)
            else:
                stream = engine.stream(src)

            for det in stream:
                result = counter.update(det)
                if json_output:
                    click.echo(
                        json_mod.dumps(
                            {
                                "frame_index": result.frame_index,
                                "timestamp_ms": result.timestamp_ms,
                                "live_counts": result.live_counts,
                                "cumulative_counts": result.cumulative_counts,
                                "zone_counts": result.zone_counts,
                                "line_events": [
                                    {
                                        "line_id": e.line_id,
                                        "track_id": e.track_id,
                                        "direction": e.direction,
                                        "class_name": e.class_name,
                                    }
                                    for e in result.line_events
                                ],
                            }
                        )
                    )
                else:
                    counts_str = ", ".join(
                        f"{k}={v}" for k, v in sorted(result.live_counts.items())
                    )
                    cum_str = ", ".join(
                        f"{k}={v}" for k, v in sorted(result.cumulative_counts.items())
                    )
                    click.echo(
                        f"Frame {result.frame_index}: live=[{counts_str}] cumulative=[{cum_str}]"
                    )
                    for evt in result.line_events:
                        click.echo(
                            f"  Line '{evt.line_id}': "
                            f"{evt.class_name} #{evt.track_id} → {evt.direction}"
                        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("classify")
@click.argument("source")
@click.option("--model", "-m", default="yolo11n-cls", help="Model name, e.g. yolo11n-cls")
@click.option("--weights", "-w", default=None, type=click.Path(exists=True))
@click.option(
    "--num-classes",
    "num_classes",
    default=None,
    type=int,
    help="Override output class count.",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "pytorch", "onnx", "tensorrt", "openvino"]),
)
@click.option("--device", default="auto")
@click.option("--top-k", default=5, type=int, help="Number of top predictions to display.")
@click.option("--batch-size", default=1, type=int)
@click.option(
    "--auto-letterbox/--no-auto-letterbox",
    "auto_letterbox",
    default=False,
    help="Stride-aligned non-square tensors (reduces pixel count on 16:9 input)",
)
def classify_command(
    source: str,
    model: str,
    weights: str | None,
    num_classes: int | None,
    backend: str,
    device: str,
    top_k: int,
    batch_size: int,
    auto_letterbox: bool,
) -> None:
    """Run image classification on SOURCE (image/video/RTSP/directory)."""
    from yowo.classify_engine import ClassificationEngine
    from yowo.io import open_source

    spec = _parse_cls_model_spec(model)
    weights_path = Path(weights) if weights else spec.weights_path

    try:
        with ClassificationEngine(
            model_family=spec.family,
            model_size=spec.size,
            weights_path=weights_path,
            num_classes=num_classes,
            backend=BackendType(backend) if backend != "auto" else None,
            device=device,
            batch_size=batch_size,
            top_k=top_k,
            auto_letterbox=auto_letterbox,
        ) as engine:
            src = open_source(source)
            for result in engine.stream(src):
                parts = [
                    f"Top-{rank + 1}: cls_{cid:04d} ({score:.3f})"
                    for rank, (cid, score) in enumerate(
                        zip(result.topk_class_ids, result.topk_scores, strict=True)
                    )
                ]
                click.echo(f"[{result.frame_index}] {' | '.join(parts)}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("info")
@click.option(
    "--compat",
    is_flag=True,
    default=False,
    help="Show export compatibility matrix for current system.",
)
def info_command(compat: bool) -> None:
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

    if compat:
        _print_compat_matrix(hw)


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


def _load_zones(zone_file: str | None) -> list | None:
    """Load CountZone list from a JSON file, or return None if no file given."""
    if zone_file is None:
        return None
    import json as _json

    from yowo.counter import CountZone

    data = _json.loads(Path(zone_file).read_text(encoding="utf-8"))
    return [
        CountZone(
            zone_id=z["zone_id"],
            vertices=tuple(tuple(pt) for pt in z["vertices"]),
            class_filter=frozenset(z.get("class_filter", [])),
        )
        for z in data
    ]


def _load_lines(line_file: str | None) -> list | None:
    """Load CountLine list from a JSON file, or return None if no file given."""
    if line_file is None:
        return None
    import json as _json

    from yowo.counter import CountLine

    data = _json.loads(Path(line_file).read_text(encoding="utf-8"))
    return [
        CountLine(
            line_id=ln["line_id"],
            p1=tuple(ln["p1"]),
            p2=tuple(ln["p2"]),
            class_filter=frozenset(ln.get("class_filter", [])),
        )
        for ln in data
    ]


def _parse_cls_model_spec(model_name: str) -> ModelSpec:
    """Parse a classification model name like ``"yolo11n-cls"`` into a :class:`ModelSpec`.

    The ``-cls`` suffix is required; bare detection names (``"yolo11n"``) are
    rejected with a clear error rather than silently forced to classify task.

    Raises:
        click.BadParameter: On unknown model name or non-classification task.
    """
    from yowo._convenience import parse_model_name
    from yowo.errors import ConfigError

    try:
        spec = parse_model_name(model_name)
    except ConfigError as exc:
        raise click.BadParameter(str(exc)) from exc
    if spec.task != "classify":
        raise click.BadParameter(
            f"Expected a classification model (e.g. yolo11n-cls), got: {model_name!r}"
        )
    return spec


def _parse_model_spec(model_name: str) -> ModelSpec:
    """Parse 'yolo26n' -> ModelSpec(YOLO26, NANO).

    Raises:
        click.BadParameter: On unknown model name format.
    """
    from yowo._convenience import parse_model_name
    from yowo.errors import ConfigError

    try:
        return parse_model_name(model_name)
    except ConfigError as exc:
        raise click.BadParameter(str(exc)) from exc


def _print_compat_matrix(hw: object) -> None:
    """Print an export compatibility matrix for the current system."""
    import platform as _platform

    from yowo.hardware import HardwareProfile

    assert isinstance(hw, HardwareProfile)
    libs = hw.libraries

    rows: list[tuple[str, str, str]] = []

    # Python
    rows.append(("Python", _platform.python_version(), "OK"))

    # PyTorch
    if libs.torch_version:
        rows.append(("PyTorch", libs.torch_version, "OK"))
    else:
        rows.append(("PyTorch", "not installed", "Missing"))

    # CUDA
    cuda_ver = getattr(libs, "cuda_version", None)
    if cuda_ver:
        rows.append(("CUDA", cuda_ver, "OK"))
    elif libs.torch_cuda_available:
        rows.append(("CUDA", "available (version unknown)", "OK"))
    else:
        rows.append(("CUDA", "not available", "N/A"))

    # cuDNN
    cudnn_ver = getattr(libs, "cudnn_version", None)
    if cudnn_ver:
        rows.append(("cuDNN", cudnn_ver, "OK"))
    else:
        rows.append(("cuDNN", "not detected", "N/A"))

    # TensorRT
    if libs.tensorrt_version:
        rows.append(("TensorRT", libs.tensorrt_version, "OK"))
    else:
        rows.append(("TensorRT", "not installed", "Missing"))

    # ONNX Runtime
    if libs.onnxruntime_version:
        ep = "CUDA" if libs.onnxruntime_has_cuda else "CPU"
        if getattr(libs, "onnxruntime_has_coreml", False):
            ep = "CoreML"
        rows.append(("ONNX Runtime", f"{libs.onnxruntime_version} ({ep})", "OK"))
    else:
        rows.append(("ONNX Runtime", "not installed", "Missing"))

    # OpenVINO
    if libs.openvino_version:
        rows.append(("OpenVINO", libs.openvino_version, "OK"))
    else:
        rows.append(("OpenVINO", "not installed", "Missing"))

    # CoreML tools
    coreml_ver = getattr(libs, "coremltools_version", None)
    if coreml_ver:
        rows.append(("CoreML Tools", coreml_ver, "OK"))
    else:
        status = "Missing" if _platform.system() == "Darwin" else "N/A"
        rows.append(("CoreML Tools", "not installed", status))

    # Print table
    click.echo("")
    click.echo("=== Export Compatibility ===")
    col1 = max(len(r[0]) for r in rows) + 2
    col2 = max(len(r[1]) for r in rows) + 2
    header = f"{'Component':<{col1}} {'Version':<{col2}} Status"
    click.echo(header)
    click.echo("-" * len(header))
    for name, version, status in rows:
        click.echo(f"{name:<{col1}} {version:<{col2}} {status}")


def _write_json(detections: list[object], path: Path) -> None:
    """Serialise detections to a JSON file via Detection.to_dict()."""
    import json

    from yowo.types import Detection

    out = [det.to_dict() for det in detections if isinstance(det, Detection)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


__all__ = ["cli"]
