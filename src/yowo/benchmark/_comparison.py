"""Ultralytics comparison runner for A/B benchmark validation.

Attempts to load ultralytics and run official val() to obtain reference
mAP/accuracy numbers. Returns ``None`` when ultralytics is not installed
or when evaluation fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_ultralytics_benchmark(
    model_name: str,
    data_path: str | Path,
    task: str,
) -> dict[str, float] | None:
    """Run ultralytics model validation and return metrics.

    Args:
        model_name: Ultralytics model name (e.g. ``"yolo11n.pt"``).
        data_path: Path to dataset root (COCO or ImageNet layout).
        task: Either ``"detect"`` or ``"classify"``.

    Returns:
        Dict with metric keys (``mAP_50_95``, ``mAP_50`` for detection;
        ``top1_accuracy`` for classification), or ``None`` if ultralytics
        is unavailable or evaluation fails.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        logger.info("ultralytics not installed -- skipping comparison benchmark")
        return None

    try:
        model = YOLO(model_name)

        if task == "detect":
            results = model.val(data=str(data_path), task="detect")
            box = results.box
            return {
                "mAP_50_95": float(box.map),
                "mAP_50": float(box.map50),
                "mAP_75": float(box.map75),
            }

        if task == "classify":
            results = model.val(data=str(data_path), task="classify")
            return {
                "top1_accuracy": float(results.top1),
            }

        logger.warning("Unknown task '%s' for ultralytics benchmark", task)
        return None

    except Exception:
        logger.warning(
            "Ultralytics benchmark failed for %s",
            model_name,
            exc_info=True,
        )
        return None


__all__ = ["run_ultralytics_benchmark"]
