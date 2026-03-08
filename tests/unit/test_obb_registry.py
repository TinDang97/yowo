"""Tests for OBB model registry."""

from __future__ import annotations

import pytest

from yowo.errors import ModelNotFoundError
from yowo.models._registry import get_obb
from yowo.types import ModelFamily, ModelSize


class TestOBBRegistry:
    def test_get_obb_nano_returns_correct_meta(self) -> None:
        meta = get_obb(ModelFamily.YOLO11, ModelSize.NANO)
        assert meta.num_classes == 15
        assert meta.weight_stem == "yolo11n-obb"
        assert meta.family == ModelFamily.YOLO11
        assert meta.size == ModelSize.NANO
        assert meta.input_height == 640
        assert meta.input_width == 640

    def test_get_obb_url_contains_v83_and_pt(self) -> None:
        meta = get_obb(ModelFamily.YOLO11, ModelSize.NANO)
        assert "v8.3.0" in meta.default_weights_url
        assert "yolo11n-obb.pt" in meta.default_weights_url

    def test_all_five_sizes_registered(self) -> None:
        for size in (
            ModelSize.NANO,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.XLARGE,
        ):
            meta = get_obb(ModelFamily.YOLO11, size)
            assert meta.num_classes == 15
            assert "v8.3.0" in meta.default_weights_url

    def test_url_contains_correct_stem_per_size(self) -> None:
        expected = {
            ModelSize.NANO: "yolo11n-obb.pt",
            ModelSize.SMALL: "yolo11s-obb.pt",
            ModelSize.MEDIUM: "yolo11m-obb.pt",
            ModelSize.LARGE: "yolo11l-obb.pt",
            ModelSize.XLARGE: "yolo11x-obb.pt",
        }
        for size, stem in expected.items():
            meta = get_obb(ModelFamily.YOLO11, size)
            url = meta.default_weights_url
            assert stem in url, f"Expected {stem} in {url}"

    def test_yolo26_raises_model_not_found(self) -> None:
        """YOLO26 has no OBB variant — must raise ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            get_obb(ModelFamily.YOLO26, ModelSize.NANO)

    def test_get_obb_exported_from_models_init(self) -> None:
        """get_obb must be importable from yowo.models public surface."""
        from yowo.models import get_obb as _get_obb

        meta = _get_obb(ModelFamily.YOLO11, ModelSize.SMALL)
        assert meta.num_classes == 15
