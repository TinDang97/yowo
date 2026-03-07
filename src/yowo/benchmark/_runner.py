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

import numpy as np

from yowo.benchmark._evaluator import (
    detections_to_coco_results,
    evaluate_coco_map,
    evaluate_imagenet_accuracy,
)
from yowo.classify_engine import ClassificationEngine
from yowo.engine import DetectionEngine
from yowo.types import BackendType, Detection, Frame, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Metrics from benchmarking a single backend."""

    format: str
    map_50_95: float | None
    map_50: float | None
    fps_avg: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    model_size_mb: float
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


def _get_model_size_mb(spec: ModelSpec) -> float:
    """Get model file size in MB from weights path."""
    if spec.weights_path is not None:
        try:
            size_bytes = Path(spec.weights_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except OSError:
            pass
    return 0.0


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
    except ImportError:
        pass
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
) -> BenchmarkResult:
    """Run benchmark on a single backend.

    Args:
        model_spec: Model specification.
        backend_type: Backend to benchmark.
        images: List of image file paths.
        image_ids: COCO image IDs (for detection mAP). ``None`` to skip mAP.
        gt_ann_path: COCO annotation file path. ``None`` to skip mAP.
        task: ``"detect"`` or ``"classify"``.
        warmup_passes: Number of warmup inferences to discard.

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

        # Warmup: run N dummy inferences, discard results and timing
        warmup_frame = Frame(
            pixels=np.zeros((480, 640, 3), dtype=np.uint8),
            source_id="warmup",
        )
        for _ in range(warmup_passes):
            if task == "classify":
                engine.classify([warmup_frame])  # type: ignore[union-attr]
            else:
                engine.detect([warmup_frame])  # type: ignore[union-attr]

        # Actual benchmark: time each image individually
        latencies: list[float] = []
        all_detections: list[Detection] = []
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
            else:
                results = engine.detect([frame])  # type: ignore[union-attr]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                all_detections.extend(results)

    # Compute FPS and latency percentiles
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
            metrics = evaluate_coco_map(gt_ann_path, coco_results)
            map_50_95 = metrics["mAP_50_95"]
            map_50 = metrics["mAP_50"]

    if task == "classify" and cls_predictions:
        cls_metrics = evaluate_imagenet_accuracy(cls_predictions)
        map_50_95 = cls_metrics["top1_accuracy"]

    model_size = _get_model_size_mb(model_spec)

    return BenchmarkResult(
        format=backend_type.value,
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
) -> list[BenchmarkResult]:
    """Run benchmarks across all available backends.

    Args:
        model_spec: Model specification.
        images: Image file paths.
        image_ids: Optional COCO image IDs for mAP evaluation.
        gt_ann_path: Optional COCO annotation file path.
        task: ``"detect"`` or ``"classify"``.
        formats: Backend names to test. ``None`` tests all available.
        warmup_passes: Number of warmup passes per backend.

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
