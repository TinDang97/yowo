"""High-level convenience functions for one-call inference.

``detect()`` is the simplest way to run YOLO inference::

    from yowo import detect
    detections = detect("image.jpg")

``parse_model_name()`` converts a short model string like ``"yolo26n"``
into the corresponding :class:`~yowo.types.ModelSpec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yowo.errors import ConfigError
from yowo.types import Detection, ModelFamily, ModelSize, ModelSpec

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
    """Parse a short model name like ``"yolo26n"`` into a :class:`ModelSpec`.

    Args:
        name: Model string, e.g. ``"yolo11s"``, ``"yolo26x"``.

    Returns:
        Corresponding ModelSpec.

    Raises:
        ConfigError: If *name* doesn't match ``yolo{11|26}{n|s|m|l|x}``.
    """
    for prefix, family in sorted(_FAMILY_MAP.items(), key=lambda x: -len(x[0])):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix in _SIZE_MAP:
                return ModelSpec(family, _SIZE_MAP[suffix])

    raise ConfigError(
        f"Unknown model: {name!r}. Expected format: yolo{{11|26}}{{n|s|m|l|x}}, e.g. yolo26n"
    )


def detect(
    source: str | Path | int,
    *,
    model: str = "yolo26n",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
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
        **engine_kwargs: Extra kwargs forwarded to :class:`InferenceEngine`.

    Returns:
        List of :class:`Detection` objects (one per frame).

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
        confidence_threshold=confidence,
        iou_threshold=iou,
        device=device,
        **engine_kwargs,
    ) as engine:
        return list(engine.stream(open_source(source)))


__all__ = ["detect", "parse_model_name"]
