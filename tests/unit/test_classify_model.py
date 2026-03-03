"""Unit tests for ClassifyModel (yowo.arch._yolo.ClassifyModel)."""

from __future__ import annotations

import pytest
import torch

from yowo.arch import build_classify_model
from yowo.arch._config import ClassifyConfig
from yowo.types import ModelFamily, ModelSize

# All 10 variants
_ALL_VARIANTS = [
    (ModelFamily.YOLO11, ModelSize.NANO),
    (ModelFamily.YOLO11, ModelSize.SMALL),
    (ModelFamily.YOLO11, ModelSize.MEDIUM),
    (ModelFamily.YOLO11, ModelSize.LARGE),
    (ModelFamily.YOLO11, ModelSize.XLARGE),
    (ModelFamily.YOLO26, ModelSize.NANO),
    (ModelFamily.YOLO26, ModelSize.SMALL),
    (ModelFamily.YOLO26, ModelSize.MEDIUM),
    (ModelFamily.YOLO26, ModelSize.LARGE),
    (ModelFamily.YOLO26, ModelSize.XLARGE),
]


class TestClassifyModel:
    @pytest.mark.parametrize("family,size", _ALL_VARIANTS)
    def test_build_all_10_variants(self, family: ModelFamily, size: ModelSize) -> None:
        """All 10 (family, size) combos build and produce (1, 1000) output."""
        model = build_classify_model(family, size).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1000), f"Failed for {family}/{size}"

    def test_output_shape_default_nc(self) -> None:
        """yolo11n-cls: output shape is (1, 1000)."""
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1000)

    def test_output_shape_custom_nc(self) -> None:
        """Custom num_classes=10 produces (1, 10) output."""
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10)

    def test_fuse_preserves_output(self) -> None:
        """fuse() then forward gives same output as unfused (within tolerance)."""
        torch.manual_seed(42)
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_unfused = model(x)

        # Clone model state before fuse, then fuse
        model.fuse().eval()
        with torch.no_grad():
            out_fused = model(x)

        assert torch.allclose(out_unfused, out_fused, atol=1e-4), (
            f"Max diff: {(out_unfused - out_fused).abs().max().item()}"
        )

    def test_only_p5_used(self) -> None:
        """Only p5 is passed to the head; p3 and p4 are discarded."""
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO).eval()
        x = torch.randn(1, 3, 224, 224)

        captured_p5: list[torch.Tensor] = []

        original_head = model.head.forward

        def _head_spy(inp: torch.Tensor) -> torch.Tensor:
            captured_p5.append(inp)
            return original_head(inp)

        model.head.forward = _head_spy  # type: ignore[method-assign]

        with torch.no_grad():
            model(x)

        assert len(captured_p5) == 1
        # p5 is the last feature map from backbone — shape (1, C, H, W)
        assert captured_p5[0].dim() == 4

    def test_config_stored(self) -> None:
        """model.config is a ClassifyConfig."""
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO)
        assert isinstance(model.config, ClassifyConfig)

    def test_batch_inference(self) -> None:
        """Batch size 4 produces (4, 1000) output."""
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO).eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1000)
