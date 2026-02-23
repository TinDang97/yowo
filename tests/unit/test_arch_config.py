"""Unit tests for yowo.arch._config."""

from __future__ import annotations

import pytest

from yowo.arch._config import ModelConfig, get_config, scale_channels, scale_repeats
from yowo.types import ModelFamily, ModelSize


class TestGetConfig:
    def test_yolo11_nano(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.NANO)
        assert cfg.depth_mult == 0.50
        assert cfg.width_mult == 0.25
        assert cfg.max_channels == 1024
        assert cfg.reg_max == 16
        assert cfg.end2end is False
        assert cfg.sppf_shortcut is False
        assert cfg.neck_c3k is False

    def test_yolo26_nano(self) -> None:
        cfg = get_config(ModelFamily.YOLO26, ModelSize.NANO)
        assert cfg.depth_mult == 0.50
        assert cfg.width_mult == 0.25
        assert cfg.max_channels == 1024
        assert cfg.reg_max == 1
        assert cfg.end2end is True
        assert cfg.sppf_shortcut is True
        assert cfg.neck_c3k is True

    def test_yolo11_medium_max_channels_512(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.MEDIUM)
        assert cfg.max_channels == 512

    def test_yolo11_xlarge(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.XLARGE)
        assert cfg.depth_mult == 1.00
        assert cfg.width_mult == 1.50

    def test_all_10_variants_exist(self) -> None:
        for family in (ModelFamily.YOLO11, ModelFamily.YOLO26):
            for size in ModelSize:
                cfg = get_config(family, size)
                assert isinstance(cfg, ModelConfig)

    def test_unsupported_family_raises(self) -> None:
        with pytest.raises(ValueError):
            get_config(ModelFamily("unsupported"), ModelSize.NANO)  # type: ignore[arg-type]


class TestScaleChannels:
    def test_nano_64_base(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.NANO)
        # 64 * 0.25 = 16, divisible by 8
        assert scale_channels(64, cfg) == 16

    def test_medium_512_capped(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.MEDIUM)
        # min(1024, 512) * 1.0 = 512
        assert scale_channels(1024, cfg) == 512

    def test_xlarge_1024(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.XLARGE)
        # min(1024, 512) * 1.5 = 768
        assert scale_channels(1024, cfg) == 768


class TestScaleRepeats:
    def test_n_1_never_scaled(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.NANO)
        assert scale_repeats(1, cfg) == 1

    def test_n_2_with_nano(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.NANO)
        # round(2 * 0.5) = 1
        assert scale_repeats(2, cfg) == 1

    def test_n_2_with_large(self) -> None:
        cfg = get_config(ModelFamily.YOLO11, ModelSize.LARGE)
        # round(2 * 1.0) = 2
        assert scale_repeats(2, cfg) == 2
