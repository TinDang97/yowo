"""Unit tests for parse_model_name() with classification model strings."""

from __future__ import annotations

import pytest

from yowo._convenience import parse_model_name
from yowo.errors import ConfigError
from yowo.types import ModelFamily, ModelSize

_CLS_VARIANTS = [
    ("yolo11n-cls", ModelFamily.YOLO11, ModelSize.NANO),
    ("yolo11s-cls", ModelFamily.YOLO11, ModelSize.SMALL),
    ("yolo11m-cls", ModelFamily.YOLO11, ModelSize.MEDIUM),
    ("yolo11l-cls", ModelFamily.YOLO11, ModelSize.LARGE),
    ("yolo11x-cls", ModelFamily.YOLO11, ModelSize.XLARGE),
    ("yolo26n-cls", ModelFamily.YOLO26, ModelSize.NANO),
    ("yolo26s-cls", ModelFamily.YOLO26, ModelSize.SMALL),
    ("yolo26m-cls", ModelFamily.YOLO26, ModelSize.MEDIUM),
    ("yolo26l-cls", ModelFamily.YOLO26, ModelSize.LARGE),
    ("yolo26x-cls", ModelFamily.YOLO26, ModelSize.XLARGE),
]


class TestParseModelNameCls:
    def test_parse_yolo11n_cls(self) -> None:
        """'yolo11n-cls' → family=YOLO11, size=NANO, task='classify'."""
        spec = parse_model_name("yolo11n-cls")
        assert spec.family == ModelFamily.YOLO11
        assert spec.size == ModelSize.NANO
        assert spec.task == "classify"

    def test_parse_yolo26x_cls(self) -> None:
        """'yolo26x-cls' → family=YOLO26, size=XLARGE, task='classify'."""
        spec = parse_model_name("yolo26x-cls")
        assert spec.family == ModelFamily.YOLO26
        assert spec.size == ModelSize.XLARGE
        assert spec.task == "classify"

    @pytest.mark.parametrize("name,family,size", _CLS_VARIANTS)
    def test_parse_all_10_cls_variants(
        self, name: str, family: ModelFamily, size: ModelSize
    ) -> None:
        """All 10 cls variants parse without error."""
        spec = parse_model_name(name)
        assert spec.family == family
        assert spec.size == size
        assert spec.task == "classify"

    def test_detect_task_unchanged(self) -> None:
        """Detection model names have task='detect'."""
        spec = parse_model_name("yolo11n")
        assert spec.task == "detect"

    def test_double_cls_suffix_behavior(self) -> None:
        """'yolo11n-cls-cls' raises ConfigError (double suffix is invalid)."""
        with pytest.raises(ConfigError):
            parse_model_name("yolo11n-cls-cls")

    def test_invalid_cls_model_name_raises(self) -> None:
        """Unknown family name raises ConfigError."""
        with pytest.raises(ConfigError):
            parse_model_name("resnet50-cls")

    def test_parse_returns_model_spec(self) -> None:
        """Return type is ModelSpec with correct fields."""
        from yowo.types import ModelSpec

        spec = parse_model_name("yolo11s-cls")
        assert isinstance(spec, ModelSpec)
        assert spec.task == "classify"

    def test_detect_model_has_no_cls_task(self) -> None:
        """All detection model names produce task != 'classify'."""
        for name in ["yolo26n", "yolo26x", "yolo11m", "yolo11l"]:
            spec = parse_model_name(name)
            assert spec.task != "classify", f"Expected detect task for {name}"
