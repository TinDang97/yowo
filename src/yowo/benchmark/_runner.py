"""Benchmark runner: per-backend FPS measurement and accuracy evaluation.

Executes inference across available backends, collecting latency percentiles,
FPS, and optional mAP/accuracy metrics. Warmup passes are excluded from
timing measurements.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yowo.benchmark._evaluator import (
    detections_to_coco_results,
    evaluate_coco_map,
    evaluate_imagenet_accuracy,
)
from yowo.classify_engine import ClassificationEngine
from yowo.engine import DetectionEngine
from yowo.models import resolve_weights
from yowo.types import BackendType, Detection, Frame, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Metrics from benchmarking a single backend.

    ``format`` is the backend that EXECUTED, not the one requested. Those
    differ whenever a backend falls back, and the fallback is common: asking
    for onnx without an exported artifact silently runs pytorch. Reported as
    the request, the onnx row and the pytorch row were both pytorch — a format
    comparison in which both sides are the same format.

    ``requested_format`` keeps what was asked for, so a fallback is legible
    rather than merely absent.

    ``model_size_mb`` is ``None`` when the size of the artifact that actually
    loaded cannot be determined. It is not 0.0: a sentinel renders as
    ``0.0 MB`` and reads as a measurement.
    """

    format: str
    requested_format: str
    map_50_95: float | None
    map_50: float | None
    fps_avg: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    model_size_mb: float | None
    device: str
    num_images: int


def _check_backend_available(backend_type: BackendType) -> bool:
    """Check if a backend's dependencies are importable."""
    import importlib

    _backend_deps: dict[BackendType, str] = {
        BackendType.PYTORCH: "torch",
        BackendType.ONNX: "onnxruntime",
        BackendType.TENSORRT: "tensorrt",
        BackendType.OPENVINO: "openvino",
        BackendType.COREML: "coremltools",
    }
    dep = _backend_deps.get(backend_type)
    if dep is None:
        return False
    try:
        importlib.import_module(dep)
        return True
    except ImportError:
        return False


def _get_model_size_mb(spec: ModelSpec) -> float | None:
    """Size of the source checkpoint in MB, or ``None`` if it cannot be read.

    ``None`` rather than 0.0 on failure: 0.0 renders as ``0.0 MB`` and reads as
    a measured value. This reports the SOURCE checkpoint, so callers must only
    use it for a row that actually loaded that checkpoint — see
    :func:`_artifact_size_mb`.
    """
    path = spec.weights_path
    if path is None:
        try:
            path = resolve_weights(spec)
        except Exception:
            return None
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return None


def _artifact_size_mb(spec: ModelSpec, executed: str) -> float | None:
    """Size of the artifact that the executing backend actually loaded.

    Only the pytorch backend loads the source checkpoint, so only there is the
    checkpoint's size the executed artifact's size. For onnx, openvino and
    tensorrt the loaded artifact is a different file whose path no backend
    exposes (``selection`` carries backend, device, precision and reason;
    ``PyTorchBackend`` holds a spec, not a path). Reporting the checkpoint for
    those rows is how every format came to read 5.354 MB.

    Returning ``None`` is the honest answer until a backend can name the file
    it opened — owned by `backend-extension-contract`, whose scope includes the
    backend Protocol this would extend.
    """
    if executed == BackendType.PYTORCH.value:
        return _get_model_size_mb(spec)
    return None


def _load_image_as_frame(image_path: Path, index: int) -> Frame:
    """Load an image file as a Frame, falling back to a dummy frame on error."""
    try:
        import cv2

        pixels = cv2.imread(str(image_path))
        if pixels is not None:
            return Frame(
                pixels=np.asarray(pixels, dtype=np.uint8),
                source_id=str(image_path),
                frame_index=index,
            )
        import warnings

        warnings.warn(
            f"cv2.imread returned None for {image_path}",
            stacklevel=2,
        )
    except ImportError:
        import warnings

        warnings.warn(
            "cv2 not available — benchmark will use dummy black frames. "
            "Install opencv-python for real image benchmarks.",
            stacklevel=2,
        )
    # Fallback: create a dummy 640x480 frame
    return Frame(
        pixels=np.zeros((480, 640, 3), dtype=np.uint8),
        source_id=str(image_path),
        frame_index=index,
    )


def run_single_backend(
    model_spec: ModelSpec,
    backend_type: BackendType,
    images: list[Path],
    image_ids: list[int] | None,
    gt_ann_path: str | None,
    task: str,
    warmup_passes: int = 10,
    gt_boxes: list[list[list[float]]] | None = None,
    gt_classes: list[list[int]] | None = None,
) -> BenchmarkResult:
    """Run benchmark on a single backend.

    Args:
        model_spec: Model specification.
        backend_type: Backend to benchmark.
        images: List of image file paths.
        image_ids: COCO image IDs (for detection mAP). ``None`` to skip mAP.
        gt_ann_path: COCO annotation file path. ``None`` to skip mAP.
        task: ``"detect"``, ``"classify"``, or ``"obb"``.
        warmup_passes: Number of warmup inferences to discard.
        gt_boxes: DOTA ground truth quad boxes per image (for OBB mAP).
        gt_classes: DOTA ground truth class indices per image (for OBB mAP).

    Returns:
        BenchmarkResult with timing and optional accuracy metrics.
    """
    if task == "classify":
        engine = ClassificationEngine(
            model_family=model_spec.family,
            model_size=model_spec.size,
            weights_path=model_spec.weights_path,
            num_classes=model_spec.num_classes,
            backend=backend_type,
        )
    elif task == "obb":
        from yowo.obb_engine import OBBEngine  # lazy import — patchable in tests

        engine = OBBEngine(  # type: ignore[assignment]
            model_family=model_spec.family,
            model_size=model_spec.size,
            weights_path=model_spec.weights_path,
            # Both were omitted here while `classify` and `detect` passed them,
            # so the row was labelled with a backend it never executed.
            num_classes=model_spec.num_classes,
            backend=backend_type,
        )
    else:
        engine = DetectionEngine(
            model_family=model_spec.family,
            model_size=model_spec.size,
            weights_path=model_spec.weights_path,
            num_classes=model_spec.num_classes,
            backend=backend_type,
        )

    with engine:
        device = engine.selection.device_type.value
        # The backend that was SELECTED, which is not always the one requested:
        # a fallback is silent, and reporting the request as the result turns
        # two pytorch runs into an onnx-versus-pytorch comparison.
        executed = engine.selection.backend.value

        # Warmup: run N dummy inferences, discard results and timing
        warmup_frame = Frame(
            pixels=np.zeros((480, 640, 3), dtype=np.uint8),
            source_id="warmup",
        )
        for _ in range(warmup_passes):
            if task == "classify":
                engine.classify([warmup_frame])  # type: ignore[union-attr]
            elif task == "obb":
                engine.detect_obb([warmup_frame])  # type: ignore[union-attr]
            else:
                engine.detect([warmup_frame])  # type: ignore[union-attr]

        # Actual benchmark: time each image individually
        latencies: list[float] = []
        all_detections: list[Detection] = []
        all_obb_detections: list[Any] = []  # OBBDetection, populated for task=="obb"
        cls_predictions: list[tuple[int, int]] = []

        for i, img_path in enumerate(images):
            frame = _load_image_as_frame(img_path, i)
            t0 = time.perf_counter()
            if task == "classify":
                results = engine.classify([frame])  # type: ignore[union-attr]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                # For accuracy: need ground truth from image_ids
                if image_ids is not None and results:
                    cls_predictions.append((results[0].top1_class_id, image_ids[i]))
            elif task == "obb":
                results = engine.detect_obb([frame])  # type: ignore[union-attr]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                all_obb_detections.extend(results)
            else:
                results = engine.detect([frame])  # type: ignore[union-attr]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                all_detections.extend(results)

    # Compute FPS and latency percentiles
    if not latencies:
        return BenchmarkResult(
            format=executed,
            requested_format=backend_type.value,
            map_50_95=None,
            map_50=None,
            fps_avg=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            model_size_mb=_artifact_size_mb(model_spec, executed),
            # `device` is bound above; "unknown" was a literal standing in for
            # a value the code already had.
            device=device,
            num_images=len(images),
        )

    total_time_s = sum(latencies) / 1000.0
    fps_avg = len(images) / total_time_s if total_time_s > 0 else 0.0

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))

    # Compute accuracy metrics
    map_50_95: float | None = None
    map_50: float | None = None

    if task == "detect" and gt_ann_path is not None and image_ids is not None:
        coco_results = detections_to_coco_results(all_detections, image_ids)
        if coco_results:
            # Scope the evaluation to the images THIS RUN evaluated. Dropping
            # image_ids here scored a 500-image subset against all 5000
            # ground-truth images: measured 2026-09-15, 0.0419 against a true
            # 0.4158, with the 4500 unseen images counted as total misses.
            metrics = evaluate_coco_map(gt_ann_path, coco_results, image_ids=image_ids)
            map_50_95 = metrics["mAP_50_95"]
            map_50 = metrics["mAP_50"]

    if task == "classify" and cls_predictions:
        cls_metrics = evaluate_imagenet_accuracy(cls_predictions)
        map_50_95 = cls_metrics["top1_accuracy"]

    if task == "obb" and gt_boxes is not None and gt_classes is not None:
        from yowo.benchmark._dota_evaluator import evaluate_obb_map

        obb_metrics = evaluate_obb_map(all_obb_detections, gt_boxes, gt_classes)
        map_50_95 = obb_metrics["mAP_50_95"]
        map_50 = obb_metrics["mAP_50"]

    model_size = _artifact_size_mb(model_spec, executed)

    return BenchmarkResult(
        format=executed,
        requested_format=backend_type.value,
        map_50_95=map_50_95,
        map_50=map_50,
        fps_avg=fps_avg,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        model_size_mb=model_size,
        device=device,
        num_images=len(images),
    )


def run_all_backends(
    model_spec: ModelSpec,
    images: list[Path],
    image_ids: list[int] | None = None,
    gt_ann_path: str | None = None,
    task: str = "detect",
    formats: list[str] | None = None,
    warmup_passes: int = 10,
    gt_boxes: list[list[list[float]]] | None = None,
    gt_classes: list[list[int]] | None = None,
) -> list[BenchmarkResult]:
    """Run benchmarks across all available backends.

    Args:
        model_spec: Model specification.
        images: Image file paths.
        image_ids: Optional COCO image IDs for mAP evaluation.
        gt_ann_path: Optional COCO annotation file path.
        task: ``"detect"``, ``"classify"``, or ``"obb"``.
        formats: Backend names to test. ``None`` tests all available.
        warmup_passes: Number of warmup passes per backend.
        gt_boxes: DOTA ground truth quad boxes per image (for OBB mAP).
        gt_classes: DOTA ground truth class indices per image (for OBB mAP).

    Returns:
        List of BenchmarkResult (one per successfully tested backend).
    """
    if formats is not None:
        backends_to_test = [BackendType(f) for f in formats]
    else:
        backends_to_test = list(BackendType)

    results: list[BenchmarkResult] = []
    for bt in backends_to_test:
        if not _check_backend_available(bt):
            logger.info("Skipping %s: dependencies not available", bt.value)
            continue
        try:
            result = run_single_backend(
                model_spec=model_spec,
                backend_type=bt,
                images=images,
                image_ids=image_ids,
                gt_ann_path=gt_ann_path,
                task=task,
                warmup_passes=warmup_passes,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
            )
            results.append(result)
        except Exception:
            logger.warning("Backend %s failed during benchmark", bt.value, exc_info=True)
            continue

    return results


__all__ = [
    "BenchmarkResult",
    "run_all_backends",
    "run_single_backend",
]
