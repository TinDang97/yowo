"""Native YOLO architecture module.

Public API for building YOLO11 and YOLO26 models from configuration,
without any dependency on ultralytics.

Usage::

    from yowo.arch import build_model, build_classify_model
    from yowo.types import ModelFamily, ModelSize

    model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
    model = model.fuse().eval()

    cls_model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO)
    cls_model = cls_model.fuse().eval()
"""

from __future__ import annotations

from yowo.arch._config import (
    ClassifyConfig,
    ModelConfig,
    get_classify_config,
    get_config,
    scale_channels,
    scale_repeats,
)
from yowo.arch._heads import Classify
from yowo.arch._weights import load_weights
from yowo.arch._yolo import ClassifyModel, YOLOModel
from yowo.types import ModelFamily, ModelSize


def build_model(
    family: ModelFamily,
    size: ModelSize,
    *,
    num_classes: int = 80,
) -> YOLOModel:
    """Build a YOLO model from family and size.

    Args:
        family: Model family (YOLO11 or YOLO26).
        size: Size variant (nano … xlarge).
        num_classes: Number of detection classes (default 80 for COCO).

    Returns:
        An initialized ``YOLOModel`` ready for weight loading.

    Raises:
        ValueError: If the family is not supported.
    """
    config = get_config(family, size)
    if num_classes != config.num_classes:
        # Rebuild config with custom class count
        config = ModelConfig(
            family=config.family,
            size=config.size,
            depth_mult=config.depth_mult,
            width_mult=config.width_mult,
            max_channels=config.max_channels,
            num_classes=num_classes,
            reg_max=config.reg_max,
            end2end=config.end2end,
            sppf_shortcut=config.sppf_shortcut,
            neck_c3k=config.neck_c3k,
            max_det=config.max_det,
            input_size=config.input_size,
        )
    return YOLOModel(config)


def build_classify_model(
    family: ModelFamily,
    size: ModelSize,
    *,
    num_classes: int = 1000,
) -> ClassifyModel:
    """Build a YOLO classification model for the given family and size.

    Args:
        family: Model family (YOLO11 or YOLO26).
        size: Size variant (nano … xlarge).
        num_classes: Number of output classes (default 1000 for ImageNet).

    Returns:
        An initialized ``ClassifyModel`` ready for weight loading.

    Raises:
        ValueError: If the family is not supported.
    """
    config = get_classify_config(family, size, num_classes=num_classes)
    return ClassifyModel(config)


__all__ = [
    "Classify",
    "ClassifyConfig",
    "ClassifyModel",
    "ModelConfig",
    "YOLOModel",
    "build_classify_model",
    "build_model",
    "get_classify_config",
    "get_config",
    "load_weights",
    "scale_channels",
    "scale_repeats",
]
