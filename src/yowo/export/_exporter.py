"""Core export orchestration (native torch.onnx.export)."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yowo.errors import ConfigError, DependencyError, ExportError
from yowo.export._calibration import resolve_calibration_images
from yowo.export._metadata import ExportMetadata
from yowo.hardware import get_hardware_profile
from yowo.models import resolve_weights
from yowo.types import ExportFormat, ModelSpec, Precision

logger = logging.getLogger(__name__)


def export_model(
    spec: ModelSpec,
    target_format: ExportFormat,
    output_dir: Path,
    *,
    precision: Precision = Precision.FP16,
    dynamic_batch: bool = False,
    imgsz: int = 640,
    calibration_data: str | None = None,
    kv_cache: bool = False,
) -> ExportMetadata:
    """Export a YOLO model to an optimized inference format.

    Uses ``torch.onnx.export`` with the native ``yowo.arch`` module.
    TensorRT and OpenVINO exports first produce ONNX, then convert.
    CoreML export converts directly from PyTorch (no ONNX intermediate).

    Args:
        spec: Model to export.
        target_format: ONNX, TensorRT, OpenVINO, or CoreML.
        output_dir: Where to write the exported model.
        precision: FP32, FP16, or INT8.
        dynamic_batch: Enable dynamic batch dimension (ONNX only).
        imgsz: Input image size.
        calibration_data: Required for INT8; path to image directory.
        kv_cache: Export with K,V as explicit ONNX I/O for stateful
            streaming inference across all runtimes.

    Returns:
        ExportMetadata record with file path and sidecar written to disk.

    Raises:
        ConfigError: INT8 requested without calibration_data.
        DependencyError: Required packages not installed.
        ExportError: Export operation failed.
    """
    if precision == Precision.INT8 and calibration_data is None:
        raise ConfigError("INT8 export requires --calibration-data")

    try:
        import torch  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError("torch", "uv add yowo[pytorch]") from exc

    from yowo.arch import build_model
    from yowo.arch._weights import load_weights

    weights_path = resolve_weights(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and prepare model
    model = build_model(spec.family, spec.size)
    load_weights(model, weights_path)
    model = model.fuse().eval()

    # CoreML handles FP16 via compute_precision — keep model/dummy FP32
    if precision == Precision.FP16 and target_format != ExportFormat.COREML:
        model = model.half()

    dummy = torch.zeros(1, 3, imgsz, imgsz)
    if precision == Precision.FP16 and target_format != ExportFormat.COREML:
        dummy = dummy.half()

    logger.info(
        "Exporting %s%s -> %s (%s)%s",
        spec.family.value,
        spec.size.value,
        target_format.value,
        precision.value,
        " [kv_cache]" if kv_cache else "",
    )

    t0 = time.monotonic()

    model_stem = f"{spec.family.value}{spec.size.value}"

    # CoreML exports directly from PyTorch — skip ONNX intermediate
    if target_format == ExportFormat.COREML:
        exported_path = _convert_coreml(
            model, dummy, output_dir / f"{model_stem}.mlpackage", precision
        )
    else:
        # Step 1: Produce ONNX first
        onnx_path = output_dir / f"{model_stem}.onnx"

        if kv_cache:
            from yowo.export._kv_wrapper import YOLOKVWrapper

            wrapper = YOLOKVWrapper(model)
            _export_onnx_kv(wrapper, dummy, onnx_path, dynamic_batch=dynamic_batch)
        else:
            _export_onnx(model, dummy, onnx_path, dynamic_batch=dynamic_batch)

        # Step 1b: INT8 quantization for ONNX target
        if target_format == ExportFormat.ONNX and precision == Precision.INT8:
            from yowo.export._int8 import quantize_onnx_static

            quantized_path = output_dir / f"{model_stem}_int8.onnx"
            quantize_onnx_static(
                onnx_path,
                quantized_path,
                calibration_data,  # type: ignore[arg-type]  # validated non-None above
                input_size=imgsz,
                batch_size=1 if not dynamic_batch else 8,
            )
            onnx_path = quantized_path

        # Step 2: Convert if needed
        match target_format:
            case ExportFormat.ONNX:
                exported_path = onnx_path
            case ExportFormat.TENSORRT:
                exported_path = _convert_tensorrt(
                    onnx_path,
                    output_dir / f"{model_stem}.engine",
                    precision,
                    calibration_data,
                    imgsz=imgsz,
                )
            case ExportFormat.OPENVINO:
                exported_path = _convert_openvino(onnx_path, output_dir / f"{model_stem}_openvino")

    elapsed = time.monotonic() - t0

    hw = get_hardware_profile()
    size_bytes = (
        _dir_size(exported_path) if exported_path.is_dir() else exported_path.stat().st_size
    )

    import yowo

    meta = ExportMetadata(
        model_name=model_stem,
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
        yowo_version=getattr(yowo, "__version__", "0.1.0"),
        gpu_name=hw.primary_gpu.name if hw.primary_gpu else None,
        calibration_data=calibration_data,
        extra={"kv_cache": True} if kv_cache else {},
    )
    meta.save()

    logger.info(
        "Export complete: %s (%.1f MB, %.1fs)",
        exported_path.name,
        size_bytes / 1_048_576,
        elapsed,
    )
    return meta


# ---------------------------------------------------------------------------
# Format-specific export helpers
# ---------------------------------------------------------------------------


def _export_onnx(
    model: object,
    dummy: object,
    onnx_path: Path,
    *,
    dynamic_batch: bool,
) -> None:
    """Export model to ONNX format with optional simplification."""
    import torch  # type: ignore[import-untyped]

    dynamic_axes = {"images": {0: "batch"}, "output0": {0: "batch"}} if dynamic_batch else None

    try:
        torch.onnx.export(
            model,  # type: ignore[arg-type]
            dummy,  # type: ignore[arg-type]
            str(onnx_path),
            opset_version=17,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=dynamic_axes,
        )
    except Exception as exc:
        raise ExportError(f"ONNX export failed: {exc}") from exc

    # Simplify with onnxslim if available
    try:
        import onnxslim  # type: ignore[import-untyped]

        slim_model = onnxslim.slim(str(onnx_path))
        import onnx  # type: ignore[import-untyped]

        onnx.save(slim_model, str(onnx_path))  # type: ignore[arg-type]
        logger.debug("ONNX model simplified with onnxslim")
    except ImportError:
        logger.debug("onnxslim not installed, skipping ONNX simplification")
    except Exception as exc:
        logger.warning("ONNX simplification failed (non-fatal): %s", exc)


def _export_onnx_kv(
    wrapper: object,
    dummy_images: object,
    onnx_path: Path,
    *,
    dynamic_batch: bool,
) -> None:
    """Export KV-wrapper to ONNX with K,V as explicit model I/O.

    Applies onnxslim simplification with I/O validation guard.
    Internalizes external tensor data so CoreML EP can load the model.
    """
    import torch  # type: ignore[import-untyped]
    from torch import Tensor

    from yowo.export._kv_wrapper import YOLOKVWrapper

    assert isinstance(wrapper, YOLOKVWrapper)
    assert isinstance(dummy_images, Tensor)

    dummy_kvs = wrapper.build_dummy_kv_inputs(dummy_images.shape[-1], dtype=dummy_images.dtype)
    use_cache = torch.tensor(0.0, dtype=dummy_images.dtype)
    dummy_inputs = (dummy_images, use_cache, *dummy_kvs)

    input_names = ["images", *wrapper.kv_input_names]
    output_names = ["output0", *wrapper.kv_output_names]

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}, "output0": {0: "batch"}}
        for name in wrapper.kv_input_names:
            if name != "use_cache":
                dynamic_axes[name] = {0: "batch"}
        for name in wrapper.kv_output_names:
            dynamic_axes[name] = {0: "batch"}

    try:
        torch.onnx.export(
            wrapper,  # type: ignore[arg-type]
            dummy_inputs,  # type: ignore[arg-type]
            str(onnx_path),
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except Exception as exc:
        raise ExportError(f"ONNX KV export failed: {exc}") from exc

    # Internalize external tensor data so all runtimes (CoreML EP) can load
    # the model from a single file without needing the .onnx.data sidecar.
    try:
        import onnx  # type: ignore[import-untyped]

        model = onnx.load(str(onnx_path), load_external_data=True)
        tmp_path = onnx_path.with_suffix(".onnx.tmp")
        onnx.save(model, str(tmp_path))  # type: ignore[arg-type]
        tmp_path.replace(onnx_path)
        data_path = onnx_path.with_suffix(".onnx.data")
        data_path.unlink(missing_ok=True)
    except ImportError:
        logger.debug("onnx package not installed, skipping data internalization")
    except Exception as exc:
        tmp_cleanup = onnx_path.with_suffix(".onnx.tmp")
        tmp_cleanup.unlink(missing_ok=True)
        logger.warning("Failed to internalize ONNX data (non-fatal): %s", exc)

    # Simplify with onnxslim — KV I/O nodes have real data dependencies
    # (consumed by Where/blending ops, produced by QKV split) and survive.
    try:
        import onnx  # type: ignore[import-untyped]
        import onnxslim  # type: ignore[import-untyped]

        slim_model: Any = onnxslim.slim(str(onnx_path))
        # Validate KV I/O survived simplification
        slim_ins: set[str] = {i.name for i in slim_model.graph.input}
        slim_outs: set[str] = {o.name for o in slim_model.graph.output}
        io_ok = all(n in slim_ins for n in input_names) and all(
            n in slim_outs for n in output_names
        )
        if io_ok:
            onnx.save(slim_model, str(onnx_path))  # type: ignore[arg-type]
            logger.debug("KV ONNX model simplified with onnxslim")
        else:
            logger.warning("onnxslim stripped KV I/O — keeping unsimplified model")
    except ImportError:
        logger.debug("onnxslim not installed, skipping KV ONNX simplification")
    except Exception as exc:
        logger.warning("KV ONNX simplification failed (non-fatal): %s", exc)

    logger.debug("KV-cache ONNX export complete: %s", onnx_path.name)


def _convert_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    precision: Precision,
    calibration_data: str | None,
    *,
    imgsz: int = 640,
) -> Path:
    """Convert ONNX to TensorRT engine."""
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "tensorrt",
            "pip install tensorrt>=10.0 --extra-index-url https://pypi.nvidia.com",
        ) from exc

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ExportError(f"TensorRT ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if precision in (Precision.FP16, Precision.INT8):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == Precision.INT8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_data:
            from yowo.export._int8 import create_tensorrt_calibrator

            images = resolve_calibration_images(calibration_data)
            cache_file = engine_path.with_suffix(".calib")
            calibrator = create_tensorrt_calibrator(
                images, batch_size=8, input_size=imgsz, cache_file=cache_file
            )
            config.int8_calibrator = calibrator
            logger.info("INT8 calibration with %d images (cache: %s)", len(images), cache_file)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise ExportError("TensorRT engine build returned None")

    engine_path.write_bytes(serialized)
    return engine_path


def _convert_openvino(onnx_path: Path, output_dir: Path) -> Path:
    """Convert ONNX to OpenVINO IR format."""
    try:
        import openvino as ov  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError("openvino", "uv add openvino") from exc

    try:
        model = ov.convert_model(str(onnx_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(model, str(output_dir / "model.xml"))
        return output_dir
    except Exception as exc:
        raise ExportError(f"OpenVINO conversion failed: {exc}") from exc


def _convert_coreml(
    model: Any,
    dummy: Any,
    output_path: Path,
    precision: Precision,
) -> Path:
    """Convert PyTorch model directly to CoreML .mlpackage.

    Unlike other export formats, CoreML export goes directly from PyTorch
    (not through an ONNX intermediate) using ``coremltools.convert()``.

    Args:
        model: Traced or eval-mode PyTorch model.
        dummy: Dummy input tensor for tracing.
        output_path: Target ``.mlpackage`` path.
        precision: FP16 or FP32 compute precision.

    Returns:
        The output path (a directory for ``.mlpackage``).

    Raises:
        DependencyError: coremltools not installed.
        ExportError: Conversion failed.
    """
    try:
        import coremltools as ct  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError("coremltools", "uv add coremltools>=7.0") from exc

    try:
        # Warmup forward pass to initialize Detect head caches (strides,
        # anchors) so torch.jit.trace sees identical graphs on both its
        # internal sanity-check invocations.
        with torch.no_grad():
            model(dummy)

        # Trace for coremltools (it works with traced models)
        traced = torch.jit.trace(model, dummy)

        # Convert with ML Program format (modern CoreML)
        ct_precision = ct.precision.FLOAT16 if precision == Precision.FP16 else ct.precision.FLOAT32

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="images", shape=dummy.shape)],
            convert_to="mlprogram",
            compute_precision=ct_precision,
            compute_units=ct.ComputeUnit.ALL,
        )

        # Save as .mlpackage
        mlmodel.save(str(output_path))  # type: ignore[union-attr]
        return output_path

    except Exception as exc:
        raise ExportError(f"CoreML conversion failed: {exc}") from exc


def _dir_size(path: Path) -> int:
    """Return total byte size of all files in a directory tree."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


__all__ = ["export_model"]
