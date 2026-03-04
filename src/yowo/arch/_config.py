"""Model configuration table for YOLO11 and YOLO26 architectures.

Maps ``(ModelFamily, ModelSize)`` → ``ModelConfig`` with exact width/depth
multipliers and architecture flags derived from published YOLO specifications.
"""

from __future__ import annotations

from dataclasses import dataclass

from yowo.types import ModelFamily, ModelSize


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture configuration for a single YOLO variant.

    Attributes:
        family: YOLO model family.
        size: Size variant (nano … xlarge).
        depth_mult: Layer repeat multiplier (n > 1 → ``max(round(n * depth_mult), 1)``).
        width_mult: Channel width multiplier.
        max_channels: Hard cap on scaled channel width.
        num_classes: Number of detection classes.
        reg_max: DFL distribution bins per box side (16 for YOLO11, 1 for YOLO26).
        end2end: True for NMS-free inference (YOLO26).
        sppf_shortcut: True to add residual in SPPF (YOLO26).
        neck_c3k: True to use C3k in neck C3k2 layers (layers 13, 16, 19).
            True for m/l/x (both families) and all YOLO26 sizes.
        backbone_c3k: True to use C3k in backbone C3k2 layers (layers 2, 4).
            True for m/l/x (both families); False for n/s uses plain Bottleneck.
        max_det: Maximum detections for end2end top-k selection.
        input_size: Default input spatial size (height, width).
    """

    family: ModelFamily
    size: ModelSize
    depth_mult: float
    width_mult: float
    max_channels: int
    num_classes: int = 80
    reg_max: int = 16
    end2end: bool = False
    sppf_shortcut: bool = False
    neck_c3k: bool = False
    backbone_c3k: bool = False
    has_sppf: bool = True
    max_det: int = 300
    input_size: tuple[int, int] = (640, 640)


# ---------------------------------------------------------------------------
# Scaling tables — exact values from ultralytics YAML configs
# ---------------------------------------------------------------------------

# (depth_mult, width_mult, max_channels)
_SCALE: dict[ModelSize, tuple[float, float, int]] = {
    ModelSize.NANO: (0.50, 0.25, 1024),
    ModelSize.SMALL: (0.50, 0.50, 1024),
    ModelSize.MEDIUM: (0.50, 1.00, 512),
    ModelSize.LARGE: (1.00, 1.00, 512),
    ModelSize.XLARGE: (1.00, 1.50, 512),
}

# Family-specific overrides
_FAMILY_DEFAULTS: dict[ModelFamily, dict[str, object]] = {
    ModelFamily.YOLO11: {
        "reg_max": 16,
        "end2end": False,
        "sppf_shortcut": False,
    },
    ModelFamily.YOLO26: {
        "reg_max": 1,
        "end2end": True,
        "sppf_shortcut": True,
        "neck_c3k_force": True,  # YOLO26 uses C3k in neck even for n/s sizes
    },
}


def get_config(family: ModelFamily, size: ModelSize) -> ModelConfig:
    """Return the architecture config for a given family and size.

    Raises:
        ValueError: If the family is not supported.
    """
    if family not in _FAMILY_DEFAULTS:
        raise ValueError(
            f"Unsupported model family: {family.value}. "
            f"Supported: {', '.join(f.value for f in _FAMILY_DEFAULTS)}"
        )

    depth, width, max_ch = _SCALE[size]
    overrides = _FAMILY_DEFAULTS[family]

    # backbone and neck C3k2 layers use C3k (not plain Bottleneck) for m/l/x in both families.
    # YOLO26 additionally forces neck_c3k=True for n/s (all sizes use C3k in neck).
    large_size = size in (ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE)
    backbone_c3k = large_size
    neck_c3k = large_size or bool(overrides.get("neck_c3k_force", False))

    return ModelConfig(
        family=family,
        size=size,
        depth_mult=depth,
        width_mult=width,
        max_channels=max_ch,
        reg_max=int(overrides["reg_max"]),  # type: ignore[arg-type]
        end2end=bool(overrides["end2end"]),
        sppf_shortcut=bool(overrides["sppf_shortcut"]),
        neck_c3k=neck_c3k,
        backbone_c3k=backbone_c3k,
    )


@dataclass(frozen=True, slots=True)
class ClassifyConfig:
    """Architecture configuration for a YOLO classification variant.

    Shares the backbone scaling table with the detection config but omits
    detection-specific fields (reg_max, end2end, neck_c3k, max_det).

    Attributes:
        family: YOLO model family.
        size: Size variant (nano … xlarge).
        depth_mult: Layer repeat multiplier.
        width_mult: Channel width multiplier.
        max_channels: Hard cap on scaled channel width.
        backbone_c3k: True to use C3k in backbone C3k2 layers.
        sppf_shortcut: True to add residual in SPPF (YOLO26 only).
        has_sppf: False for classification models — ultralytics cls backbone
            omits SPPF entirely (C3k2_4 → C2PSA directly).
        num_classes: Number of output classes (1000 for ImageNet).
        input_size: Default input spatial size (height, width).
        dropout: Dropout probability in the classification head.
    """

    family: ModelFamily
    size: ModelSize
    depth_mult: float
    width_mult: float
    max_channels: int
    backbone_c3k: bool
    sppf_shortcut: bool
    has_sppf: bool = False
    num_classes: int = 1000
    input_size: tuple[int, int] = (224, 224)
    dropout: float = 0.0


def get_classify_config(
    family: ModelFamily,
    size: ModelSize,
    *,
    num_classes: int = 1000,
) -> ClassifyConfig:
    """Return the classification architecture config for a given family and size.

    Reuses the shared ``_SCALE`` table and ``_FAMILY_DEFAULTS`` so backbone
    dimensions exactly match the detection variants of the same family/size.

    Raises:
        ValueError: If the family is not supported.
    """
    if family not in _FAMILY_DEFAULTS:
        raise ValueError(
            f"Unsupported model family: {family.value}. "
            f"Supported: {', '.join(f.value for f in _FAMILY_DEFAULTS)}"
        )

    depth, width, max_ch = _SCALE[size]
    overrides = _FAMILY_DEFAULTS[family]
    large_size = size in (ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE)

    return ClassifyConfig(
        family=family,
        size=size,
        depth_mult=depth,
        width_mult=width,
        max_channels=max_ch,
        backbone_c3k=large_size,
        sppf_shortcut=bool(overrides["sppf_shortcut"]),
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Channel / repeat scaling helpers
# ---------------------------------------------------------------------------


def scale_channels(base: int, config: ModelConfig) -> int:
    """Scale a base channel count using width multiplier and max_channels cap."""
    from yowo.arch._blocks import make_divisible

    return make_divisible(min(base, config.max_channels) * config.width_mult, 8)


def scale_repeats(n: int, config: ModelConfig) -> int:
    """Scale layer repeat count using depth multiplier.

    Layers with ``n == 1`` are never scaled (they are structural, not repeatable).
    """
    if n <= 1:
        return n
    return max(round(n * config.depth_mult), 1)


__all__ = [
    "ClassifyConfig",
    "ModelConfig",
    "get_classify_config",
    "get_config",
    "scale_channels",
    "scale_repeats",
]
