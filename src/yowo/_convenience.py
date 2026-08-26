"""High-level convenience functions for one-call inference.

``detect()`` is the simplest way to run YOLO inference::

    from yowo import detect
    detections = detect("image.jpg")

``classify()`` is the equivalent for classification models::

    from yowo import classify
    results = classify("image.jpg", model="yolo11n-cls")

``parse_model_name()`` converts a short model string like ``"yolo26n"``
or ``"yolo11n-cls"`` into the corresponding :class:`~yowo.types.ModelSpec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yowo.errors import ConfigError
from yowo.types import (
    ClassificationResult,
    Detection,
    ModelFamily,
    ModelSize,
    ModelSpec,
    OBBDetection,
)

_SIZE_MAP: dict[str, ModelSize] = {
    "n": ModelSize.NANO,
    "s": ModelSize.SMALL,
    "m": ModelSize.MEDIUM,
    "l": ModelSize.LARGE,
    "x": ModelSize.XLARGE,
}

_FAMILY_MAP: dict[str, ModelFamily] = {
    "yolo11": ModelFamily.YOLO11,
    "yolo26": ModelFamily.YOLO26,
}


def parse_model_name(name: str) -> ModelSpec:
    """Parse a short model name into a :class:`ModelSpec`.

    Supports detection models (``"yolo26n"``), classification models
    (``"yolo11n-cls"``), and OBB models (``"yolo11n-obb"``).
    The ``-cls`` suffix sets ``task="classify"`` and the ``-obb`` suffix sets
    ``task="obb"``; both are stripped before family/size parsing.
    OBB models are YOLO11-only — ``"yolo26n-obb"`` raises :class:`ConfigError`.

    Args:
        name: Model string, e.g. ``"yolo11s"``, ``"yolo26x"``,
            ``"yolo11n-cls"``, ``"yolo26s-cls"``, ``"yolo11n-obb"``.

    Returns:
        Corresponding ModelSpec with ``task`` set appropriately.

    Raises:
        ConfigError: If *name* doesn't match a known model pattern.
    """
    task = "detect"
    if name.endswith("-cls"):
        task = "classify"
        name = name[:-4]  # strip "-cls" suffix
    elif name.endswith("-obb"):
        if not name.startswith("yolo11"):
            raise ConfigError(
                f"OBB models are only available for YOLO11 family, got: {name!r}. "
                "Only yolo11{n|s|m|l|x}-obb variants exist."
            )
        task = "obb"
        name = name[:-4]  # strip "-obb" suffix

    for prefix, family in sorted(_FAMILY_MAP.items(), key=lambda x: -len(x[0])):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix in _SIZE_MAP:
                return ModelSpec(family, _SIZE_MAP[suffix], task=task)

    raise ConfigError(
        f"Unknown model: {name!r}. Expected format: yolo{{11|26}}{{n|s|m|l|x}}[-cls|-obb], "
        f"e.g. yolo26n, yolo11n-cls, or yolo11n-obb"
    )


def detect(
    source: str | Path | int,
    *,
    model: str = "yolo26n",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    num_classes: int | None = None,
    **engine_kwargs: Any,
) -> list[Detection]:
    """Run YOLO inference and return all detections.

    This is the simplest entry point — one function call, no context
    manager boilerplate::

        from yowo import detect
        results = detect("photo.jpg", model="yolo11n", confidence=0.4)

    Args:
        source: Image/video path, RTSP URL, webcam index, or directory.
        model: Short model name (default ``"yolo26n"``).
        confidence: Confidence threshold (default 0.25).
        iou: IoU threshold for NMS (default 0.45).
        device: Device string (default ``"auto"``).
        num_classes: Override output class count (default ``None`` = registry).
        **engine_kwargs: Extra kwargs forwarded to :class:`InferenceEngine` -
            notably ``class_names=[...]`` to label a custom-trained model with
            its own class names instead of the COCO defaults.

    Returns:
        List of :class:`Detection` objects (one per frame).  For video
        sources all frames are materialised in memory; prefer
        :meth:`InferenceEngine.stream` for large videos.

    Raises:
        ConfigError: If *model* is not a valid model name.
        SourceError: If *source* cannot be opened.
    """
    # Late imports to avoid circular dependencies and heavy import cost
    # when the convenience module is merely imported.
    from yowo.engine import InferenceEngine
    from yowo.io import open_source

    spec = parse_model_name(model)
    with InferenceEngine(
        model_family=spec.family,
        model_size=spec.size,
        num_classes=num_classes,
        confidence_threshold=confidence,
        iou_threshold=iou,
        device=device,
        **engine_kwargs,
    ) as engine:
        return list(engine.stream(open_source(source)))


def classify(
    source: str | Path,
    *,
    model: str = "yolo11n-cls",
    top_k: int = 5,
    device: str = "auto",
    num_classes: int | None = None,
    **engine_kwargs: Any,
) -> list[ClassificationResult]:
    """Run YOLO classification inference and return all results.

    One-shot classification — loads the engine, runs inference, closes::

        from yowo import classify
        results = classify("photo.jpg", model="yolo11s-cls", top_k=3)
        print(results[0].top1_class_id, results[0].top1_score)

    Args:
        source: Image/video path, RTSP URL, webcam index, or directory.
        model: Short classification model name (default ``"yolo11n-cls"``).
        top_k: Number of top predictions to return per frame (default 5).
        device: Device string (default ``"auto"``).
        num_classes: Override output class count (default ``None`` = registry).
        **engine_kwargs: Extra kwargs forwarded to
            :class:`~yowo.classify_engine.ClassificationEngine`.

    Returns:
        List of :class:`~yowo.types.ClassificationResult` objects (one per
        frame). For video sources all frames are materialised in memory;
        prefer :meth:`ClassificationEngine.stream` for large videos.

    Raises:
        ConfigError: If *model* is not a valid classification model name.
        SourceError: If *source* cannot be opened.
    """
    from yowo.classify_engine import ClassificationEngine
    from yowo.io import open_source

    spec = parse_model_name(model)
    results: list[ClassificationResult] = []
    with ClassificationEngine(
        model_family=spec.family,
        model_size=spec.size,
        weights_path=spec.weights_path,
        num_classes=num_classes,
        top_k=top_k,
        device=device,
        **engine_kwargs,
    ) as engine:
        for result in engine.stream(open_source(source)):
            results.append(result)
    return results


def detect_obb(
    source: str | Path,
    *,
    model: str = "yolo11n-obb",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    num_classes: int | None = None,
    **engine_kwargs: Any,
) -> list[OBBDetection]:
    """Run YOLO OBB inference and return all oriented detections.

    One-shot OBB detection -- loads the engine, runs inference, closes::

        from yowo import detect_obb
        results = detect_obb("aerial.jpg", model="yolo11n-obb")
        for det in results:
            for box in det.boxes:
                print(f"{box.class_name}: angle={box.angle:.3f}")

    Args:
        source: Image/video path or directory.
        model: Short OBB model name (default ``"yolo11n-obb"``).
        confidence: Confidence threshold (default 0.25).
        iou: IoU threshold for NMS (default 0.45).
        device: Device string (default ``"auto"``).
        num_classes: Override output class count (default ``None`` = registry).
        **engine_kwargs: Extra kwargs forwarded to :class:`OBBEngine`.

    Returns:
        List of :class:`~yowo.types.OBBDetection` objects (one per frame).

    Raises:
        ConfigError: If *model* is not a valid OBB model name.
        SourceError: If *source* cannot be opened.
    """
    from yowo.io import open_source
    from yowo.obb_engine import OBBEngine

    spec = parse_model_name(model)
    results: list[OBBDetection] = []
    with OBBEngine(
        model_family=spec.family,
        model_size=spec.size,
        num_classes=num_classes,
        confidence_threshold=confidence,
        iou_threshold=iou,
        device=device,
        **engine_kwargs,
    ) as engine:
        for result in engine.stream_obb(open_source(source)):
            results.append(result)
    return results


__all__ = ["classify", "detect", "detect_obb", "parse_model_name"]
