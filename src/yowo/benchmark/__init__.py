"""Benchmark evaluation module for YOWO.

Provides COCO mAP and ImageNet accuracy evaluation, per-backend FPS
measurement, rich table rendering, and optional ultralytics comparison.

Public API::

    from yowo.benchmark import run_benchmark, BenchmarkResult

    results = run_benchmark(
        model="yolo11n",
        data="/path/to/coco",
        formats="pytorch,onnx",
        subset=100,
    )
"""

from __future__ import annotations

import re
from typing import Any

from yowo.benchmark._comparison import run_ultralytics_benchmark
from yowo.benchmark._dota_evaluator import load_dota_dataset
from yowo.benchmark._evaluator import (
    YOLO_TO_COCO,
    detections_to_coco_results,
    evaluate_coco_map,
    evaluate_imagenet_accuracy,
    load_coco_dataset,
    load_imagenet_dataset,
)
from yowo.benchmark._report import render_table, results_to_json
from yowo.benchmark._runner import BenchmarkResult, run_all_backends
from yowo.types import ModelFamily, ModelSize, ModelSpec

# Model name pattern: family + size (+ optional task suffix)
_MODEL_PATTERN = re.compile(r"^(yolo(?:11|26))([nsmxl])(?:-(cls|obb))?$")


def _parse_model_name(model: str) -> tuple[ModelSpec, str]:
    """Parse a model name string into a ModelSpec and task.

    Args:
        model: e.g. ``"yolo11n"``, ``"yolo26x-cls"``

    Returns:
        Tuple of (ModelSpec, task_string).
    """
    m = _MODEL_PATTERN.match(model)
    if not m:
        msg = f"Invalid model name '{model}'. Expected format: yolo{{11,26}}{{n,s,m,l,x}}[-cls]"
        raise ValueError(msg)
    family_str, size_str, task_suffix = m.group(1), m.group(2), m.group(3)
    family = ModelFamily(family_str)
    size = ModelSize(size_str)
    if task_suffix == "cls":
        task = "classify"
    elif task_suffix == "obb":
        task = "obb"
    else:
        task = "detect"
    return ModelSpec(family=family, size=size, task=task), task


def run_benchmark(
    model: str,
    data: str,
    formats: str | None = None,
    subset: int | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    """Run benchmark evaluation across backends.

    Args:
        model: Model name (e.g. ``"yolo11n"`` or ``"yolo11n-cls"``).
        data: Path to dataset root (COCO or ImageNet layout).
        formats: Comma-separated backend names to test, or ``None`` for all.
        subset: Evaluate on first *subset* images only.
        json_output: If True, return JSON-serializable dict instead of
            printing a rich table.

    Returns:
        Dict with benchmark results and optional ultralytics comparison.
    """
    spec, task = _parse_model_name(model)

    # Load dataset
    image_ids: list[int] | None = None
    gt_ann_path: str | None = None
    gt_boxes: list[list[list[float]]] | None = None
    gt_classes: list[list[int]] | None = None

    if task == "classify":
        images, labels = load_imagenet_dataset(data, subset=subset)
        image_ids = labels
    elif task == "obb":
        images, gt_boxes, gt_classes = load_dota_dataset(data, subset=subset)
    else:
        images, image_ids, gt_ann_path = load_coco_dataset(data, subset=subset)

    # Parse format filter
    format_list = [f.strip() for f in formats.split(",")] if formats else None

    # Run benchmarks
    results = run_all_backends(
        model_spec=spec,
        images=images,
        image_ids=image_ids,
        gt_ann_path=gt_ann_path,
        task=task,
        formats=format_list,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
    )

    # Optional ultralytics comparison
    ultra_results = run_ultralytics_benchmark(
        model_name=f"{model}.pt" if not model.endswith(".pt") else model,
        data_path=data,
        task=task,
    )

    if json_output:
        return results_to_json(
            results,
            ultralytics_results=ultra_results,
            model_name=model,
        )

    render_table(results, ultralytics_results=ultra_results, model_name=model)
    return results_to_json(results, ultralytics_results=ultra_results, model_name=model)


__all__ = [
    "YOLO_TO_COCO",
    "BenchmarkResult",
    "detections_to_coco_results",
    "evaluate_coco_map",
    "evaluate_imagenet_accuracy",
    "load_coco_dataset",
    "load_imagenet_dataset",
    "run_benchmark",
]
