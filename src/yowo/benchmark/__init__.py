"""Benchmark evaluation module for YOWO.

Provides COCO mAP and ImageNet accuracy evaluation, per-backend FPS
measurement, rich table rendering, and optional ultralytics comparison.

Public API::

    from yowo.benchmark import run_benchmark

    results = run_benchmark(
        model="yolo11n",
        data="/path/to/coco",
        formats="pytorch,onnx",
        subset=100,
    )
"""

from __future__ import annotations

from yowo.benchmark._evaluator import (
    YOLO_TO_COCO,
    detections_to_coco_results,
    evaluate_coco_map,
    evaluate_imagenet_accuracy,
    load_coco_dataset,
    load_imagenet_dataset,
)


def run_benchmark(
    model: str,
    data: str,
    formats: str | None = None,
    subset: int | None = None,
    json_output: bool = False,
) -> dict[str, object]:
    """Run benchmark evaluation across backends.

    Stub -- full implementation added in Task 2 after runner and report
    modules are created.

    Args:
        model: Model name (e.g. ``"yolo11n"`` or ``"yolo11n-cls"``).
        data: Path to dataset root (COCO or ImageNet layout).
        formats: Comma-separated backend names to test, or ``None`` for all.
        subset: Evaluate on first *subset* images only.
        json_output: If True, return JSON-serializable dict instead of
            printing a rich table.

    Returns:
        Dict with benchmark results.
    """
    # Implementation completed in Task 2 with runner + report modules
    raise NotImplementedError("run_benchmark requires _runner and _report modules (Task 2)")


__all__ = [
    "YOLO_TO_COCO",
    "detections_to_coco_results",
    "evaluate_coco_map",
    "evaluate_imagenet_accuracy",
    "load_coco_dataset",
    "load_imagenet_dataset",
    "run_benchmark",
]
