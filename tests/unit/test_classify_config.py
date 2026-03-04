"""Unit tests for ClassifyConfig and get_classify_config()."""

from __future__ import annotations

import dataclasses

import pytest

from yowo.arch._config import get_classify_config, get_config
from yowo.types import ModelFamily, ModelSize

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


class TestGetClassifyConfig:
    @pytest.mark.parametrize("family,size", _ALL_VARIANTS)
    def test_all_10_variants_build(self, family: ModelFamily, size: ModelSize) -> None:
        """No exception for all 10 (family, size) combinations."""
        cfg = get_classify_config(family, size)
        assert cfg.family == family
        assert cfg.size == size

    @pytest.mark.parametrize("family,size", _ALL_VARIANTS)
    def test_backbone_params_match_detection(self, family: ModelFamily, size: ModelSize) -> None:
        """depth_mult/width_mult/max_channels match detection config for same variant."""
        cls_cfg = get_classify_config(family, size)
        det_cfg = get_config(family, size)
        assert cls_cfg.depth_mult == det_cfg.depth_mult
        assert cls_cfg.width_mult == det_cfg.width_mult
        assert cls_cfg.max_channels == det_cfg.max_channels

    @pytest.mark.parametrize("size", [ModelSize.NANO, ModelSize.SMALL])
    def test_sppf_shortcut_yolo26(self, size: ModelSize) -> None:
        """YOLO26 variants have sppf_shortcut=True."""
        cfg = get_classify_config(ModelFamily.YOLO26, size)
        assert cfg.sppf_shortcut is True

    @pytest.mark.parametrize("size", [ModelSize.NANO, ModelSize.SMALL, ModelSize.MEDIUM])
    def test_sppf_shortcut_yolo11(self, size: ModelSize) -> None:
        """YOLO11 variants have sppf_shortcut=False."""
        cfg = get_classify_config(ModelFamily.YOLO11, size)
        assert cfg.sppf_shortcut is False

    @pytest.mark.parametrize("size", [ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE])
    def test_backbone_c3k_large_sizes(self, size: ModelSize) -> None:
        """m/l/x sizes have backbone_c3k=True for both families."""
        for family in (ModelFamily.YOLO11, ModelFamily.YOLO26):
            cfg = get_classify_config(family, size)
            assert cfg.backbone_c3k is True, f"Expected backbone_c3k=True for {family}/{size}"

    @pytest.mark.parametrize("size", [ModelSize.NANO, ModelSize.SMALL])
    def test_backbone_c3k_small_sizes(self, size: ModelSize) -> None:
        """n/s sizes have backbone_c3k=False for both families."""
        for family in (ModelFamily.YOLO11, ModelFamily.YOLO26):
            cfg = get_classify_config(family, size)
            assert cfg.backbone_c3k is False, f"Expected backbone_c3k=False for {family}/{size}"

    def test_default_num_classes(self) -> None:
        """Default num_classes is 1000 (ImageNet)."""
        cfg = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO)
        assert cfg.num_classes == 1000

    def test_default_input_size(self) -> None:
        """Default input_size is (224, 224)."""
        cfg = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO)
        assert cfg.input_size == (224, 224)

    def test_custom_num_classes(self) -> None:
        """Custom num_classes is respected."""
        cfg = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        assert cfg.num_classes == 10

    def test_invalid_family_raises(self) -> None:
        """Unsupported family raises ValueError."""
        # We test by passing an invalid string value coerced to a fake enum value
        with pytest.raises((ValueError, KeyError)):
            # ModelFamily only has YOLO11 and YOLO26; pass an object that
            # satisfies the type but is not in _FAMILY_DEFAULTS
            class _FakeFamily:
                value = "yolo99"

            get_classify_config(_FakeFamily(), ModelSize.NANO)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        """ClassifyConfig is immutable — attribute assignment raises FrozenInstanceError."""
        cfg = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.num_classes = 999  # type: ignore[misc]
