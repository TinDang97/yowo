"""Tests for OBBModel and build_obb_model."""

from __future__ import annotations

import pytest
import torch

from yowo.arch import build_obb_model
from yowo.types import ModelFamily, ModelSize


@pytest.mark.parametrize(
    "size",
    [
        ModelSize.NANO,
        ModelSize.SMALL,
        ModelSize.MEDIUM,
        ModelSize.LARGE,
        ModelSize.XLARGE,
    ],
)
def test_build_obb_model_all_sizes(size: ModelSize) -> None:
    """All 5 YOLO11 OBB scales build without RuntimeError."""
    model = build_obb_model(ModelFamily.YOLO11, size)
    assert model is not None


def test_obb_model_nano_forward_shape() -> None:
    """OBBModel nano: input (1,3,640,640) → output (1, 20, 8400)."""
    model = build_obb_model(ModelFamily.YOLO11, ModelSize.NANO, num_classes=15)
    model.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    # 4+15+1=20 channels, 80*80+40*40+20*20=8400 anchors
    assert out.shape == (1, 20, 8400), f"Expected (1, 20, 8400), got {out.shape}"


def test_obb_model_has_backbone_neck_head() -> None:
    """OBBModel must have backbone, neck, and head attributes."""
    from yowo.arch._heads import OBBHead

    model = build_obb_model(ModelFamily.YOLO11, ModelSize.NANO)
    assert hasattr(model, "backbone")
    assert hasattr(model, "neck")
    assert hasattr(model, "head")
    assert isinstance(model.head, OBBHead)


def test_obb_model_default_nc_15() -> None:
    """Default build uses nc=15 for DOTA v1."""
    from yowo.arch._heads import OBBHead

    model = build_obb_model(ModelFamily.YOLO11, ModelSize.NANO)
    assert isinstance(model.head, OBBHead)
    assert model.head.nc == 15
