"""Unit tests for yowo.arch — model building and forward pass."""

from __future__ import annotations

import torch

from yowo.arch import build_model
from yowo.arch._config import get_config
from yowo.arch._yolo import Backbone, YOLOModel
from yowo.types import ModelFamily, ModelSize


class TestBuildModel:
    def test_returns_yolo_model(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        assert isinstance(model, YOLOModel)

    def test_yolo26_returns_yolo_model(self) -> None:
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        assert isinstance(model, YOLOModel)

    def test_custom_num_classes(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        assert model.config.num_classes == 10


class TestBackboneForward:
    def test_output_shapes_yolo11_nano(self) -> None:
        config = get_config(ModelFamily.YOLO11, ModelSize.NANO)
        backbone = Backbone(config)
        backbone.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            p3, p4, p5 = backbone(x)

        # P3/8: 640/8 = 80
        assert p3.shape[2:] == (80, 80)
        # P4/16: 640/16 = 40
        assert p4.shape[2:] == (40, 40)
        # P5/32: 640/32 = 20
        assert p5.shape[2:] == (20, 20)


class TestYOLOModelForward:
    def test_yolo11_nano_output_shape(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        # YOLO11: (B, 4+nc, total_anchors) = (1, 84, 8400)
        # Where 80 + 4*reg_max is in the raw form, but after DFL decode it's 4+nc
        assert out.shape[0] == 1
        assert out.shape[2] == 80 * 80 + 40 * 40 + 20 * 20  # 8400

    def test_yolo26_nano_output_shape(self) -> None:
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        # YOLO26 end2end: (B, max_det, 6) = (1, 300, 6)
        assert out.shape == (1, 300, 6)


class TestFuse:
    def test_fuse_produces_equivalent_output(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()

        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out_before = model(x)

        model.fuse()
        with torch.no_grad():
            out_after = model(x)

        torch.testing.assert_close(out_before, out_after, atol=1e-4, rtol=1e-4)

    def test_fuse_removes_bn(self) -> None:
        from yowo.arch._blocks import Conv

        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.fuse()

        for m in model.modules():
            if isinstance(m, Conv):
                assert not hasattr(m, "bn"), "BN should be removed after fuse()"
