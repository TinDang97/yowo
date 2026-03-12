"""Model family registry.

Maps (ModelFamily, ModelSize) -> ModelMeta with default weight URLs.
Register new model variants with register() without changing other code.
"""

from __future__ import annotations

from dataclasses import dataclass

from yowo.errors import ModelNotFoundError
from yowo.types import ModelFamily, ModelSize

_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


@dataclass(frozen=True)
class ModelMeta:
    """Metadata for a registered model variant.

    Attributes:
        family: YOLO model family.
        size: Size variant (nano … xlarge).
        input_height: Default spatial input height (pixels).
        input_width: Default spatial input width (pixels).
        num_classes: Number of output classes (80 for COCO).
        weight_stem: Name stem for weight files (e.g. ``"yolo11n"``).
        default_weights_url: HTTPS URL to the canonical ``.pt`` weights file.
    """

    family: ModelFamily
    size: ModelSize
    input_height: int
    input_width: int
    num_classes: int
    weight_stem: str
    default_weights_url: str


_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}
_CLS_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}
_OBB_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}


def register(meta: ModelMeta) -> None:
    """Add *meta* to the registry.

    Overwrites any existing entry for the same ``(family, size)`` key.
    """
    _REGISTRY[(meta.family, meta.size)] = meta


def get(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the ModelMeta for the given family/size combination.

    Raises:
        ModelNotFoundError: When the combination is not registered.
    """
    key = (family, size)
    if key not in _REGISTRY:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(_REGISTRY))
        raise ModelNotFoundError(
            f"Model {family.value}/{size.value} not found in registry. Available: {available}"
        )
    return _REGISTRY[key]


def get_cls(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the classification ModelMeta for the given family/size combination.

    Raises:
        ModelNotFoundError: When the combination is not registered.
    """
    key = (family, size)
    if key not in _CLS_REGISTRY:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(_CLS_REGISTRY))
        raise ModelNotFoundError(
            f"Classification model {family.value}/{size.value} not found in registry. "
            f"Available: {available}"
        )
    return _CLS_REGISTRY[key]


def get_obb(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the OBB ModelMeta for the given family/size combination.

    Raises:
        ModelNotFoundError: When the combination is not registered.
    """
    key = (family, size)
    if key not in _OBB_REGISTRY:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(_OBB_REGISTRY))
        raise ModelNotFoundError(
            f"OBB model {family.value}/{size.value} not found in registry. Available: {available}"
        )
    return _OBB_REGISTRY[key]


def list_available() -> list[ModelMeta]:
    """Return all registered ModelMeta entries sorted by (family, size)."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


# ---------------------------------------------------------------------------
# Built-in registrations -- 10 variants: YOLO11/26 x n/s/m/l/x
# ---------------------------------------------------------------------------

_ASSETS_V83 = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
_ASSETS_V84 = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


def _make_meta(family: ModelFamily, size: ModelSize) -> ModelMeta:
    name = f"{family.value}{size.value}"
    return ModelMeta(
        family=family,
        size=size,
        input_height=640,
        input_width=640,
        num_classes=80,
        weight_stem=name,
        default_weights_url=f"{_ASSETS_BASE}{name}.pt",
    )


def _make_obb_meta(size: ModelSize) -> ModelMeta:
    """Build ModelMeta for a YOLO11 OBB variant.

    All OBB variants use ultralytics assets v8.3.0 and DOTA v1 (15 classes).
    Input is 640x640 (same as detection).
    """
    name = f"yolo11{size.value}-obb"
    return ModelMeta(
        family=ModelFamily.YOLO11,
        size=size,
        input_height=640,
        input_width=640,
        num_classes=15,  # DOTA v1 — CRITICAL: not 80
        weight_stem=name,
        default_weights_url=f"{_ASSETS_V83}{name}.pt",
    )


def _make_cls_meta(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Build ModelMeta for a classification variant.

    YOLO11-cls weights use ultralytics assets v8.3.0.
    YOLO26-cls weights use ultralytics assets v8.4.0.
    Input is 224x224 (ImageNet convention); 1000 classes.
    """
    name = f"{family.value}{size.value}-cls"
    base = _ASSETS_V83 if family == ModelFamily.YOLO11 else _ASSETS_V84
    url = f"{base}{name}.pt"
    return ModelMeta(
        family=family,
        size=size,
        input_height=224,
        input_width=224,
        num_classes=1000,
        weight_stem=name,
        default_weights_url=url,
    )


def _register_builtins() -> None:
    for family in (ModelFamily.YOLO11, ModelFamily.YOLO26):
        for size in (
            ModelSize.NANO,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.XLARGE,
        ):
            register(_make_meta(family, size))
            _CLS_REGISTRY[(family, size)] = _make_cls_meta(family, size)
    # OBB variants: YOLO11 only (5 sizes), DOTA v1, nc=15
    for size in (
        ModelSize.NANO,
        ModelSize.SMALL,
        ModelSize.MEDIUM,
        ModelSize.LARGE,
        ModelSize.XLARGE,
    ):
        _OBB_REGISTRY[(ModelFamily.YOLO11, size)] = _make_obb_meta(size)


_register_builtins()

__all__ = [
    "_CLS_REGISTRY",
    "_OBB_REGISTRY",
    "ModelMeta",
    "get",
    "get_cls",
    "get_obb",
    "list_available",
    "register",
]
