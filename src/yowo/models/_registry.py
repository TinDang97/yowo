"""Model family registry.

Maps (ModelFamily, ModelSize) -> ModelMeta with default weight URLs.
Register new model variants with register() without changing other code.
"""

from __future__ import annotations

from dataclasses import dataclass

from yowo.errors import ModelNotFoundError
from yowo.types import ModelFamily, ModelSize

_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Metadata for a registered model variant.

    Attributes:
        family: YOLO model family.
        size: Size variant (nano … xlarge).
        input_height: Default spatial input height (pixels).
        input_width: Default spatial input width (pixels).
        num_classes: Number of output classes (80 for COCO).
        ultralytics_name: Name stem passed to ``ultralytics.YOLO()``
            (e.g. ``"yolo12n"``).
        default_weights_url: HTTPS URL to the canonical ``.pt`` weights file.
    """

    family: ModelFamily
    size: ModelSize
    input_height: int
    input_width: int
    num_classes: int
    ultralytics_name: str
    default_weights_url: str


_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}


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


def list_available() -> list[ModelMeta]:
    """Return all registered ModelMeta entries sorted by (family, size)."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


# ---------------------------------------------------------------------------
# Built-in registrations -- 15 variants: YOLO11/12/26 x n/s/m/l/x
# ---------------------------------------------------------------------------


def _make_meta(family: ModelFamily, size: ModelSize) -> ModelMeta:
    name = f"{family.value}{size.value}"
    return ModelMeta(
        family=family,
        size=size,
        input_height=640,
        input_width=640,
        num_classes=80,
        ultralytics_name=name,
        default_weights_url=f"{_ASSETS_BASE}{name}.pt",
    )


def _register_builtins() -> None:
    for family in (ModelFamily.YOLO11, ModelFamily.YOLO12, ModelFamily.YOLO26):
        for size in (
            ModelSize.NANO,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.XLARGE,
        ):
            register(_make_meta(family, size))


_register_builtins()

__all__ = [
    "ModelMeta",
    "get",
    "list_available",
    "register",
]
