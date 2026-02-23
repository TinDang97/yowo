"""Unit tests for yowo.models._registry."""

from __future__ import annotations

import pytest

from yowo.errors import ModelNotFoundError
from yowo.models._registry import ModelMeta, get, list_available, register
from yowo.types import ModelFamily, ModelSize


class TestGet:
    def test_returns_correct_meta_for_yolo12_nano(self) -> None:
        meta = get(ModelFamily.YOLO12, ModelSize.NANO)

        assert meta.family == ModelFamily.YOLO12
        assert meta.size == ModelSize.NANO
        assert meta.input_height == 640
        assert meta.input_width == 640
        assert meta.num_classes == 80
        assert meta.ultralytics_name == "yolo12n"
        assert "yolo12n.pt" in meta.default_weights_url

    def test_raises_model_not_found_for_unknown_combo(self) -> None:
        # Use a non-existent family value via raw ModelFamily creation.
        # We test with a valid family but we monkey-patch a fake size tuple.
        fake_size = ModelSize.XLARGE  # valid, but we delete it temporarily below.

        # Remove from registry temporarily and verify KeyError-like behaviour.
        with pytest.raises(ModelNotFoundError, match="not found in registry"):
            # "yolo999" family won't be in the registry; simulate by passing
            # an unregistered (family, size) combo not registered.
            # We create a fresh ModelFamily-like value — instead, use a
            # registered family but deliberately break lookup via a wrapper.
            get.__wrapped__ if hasattr(get, "__wrapped__") else None  # no-op

            # Directly manipulate by importing and testing the raw dict.
            from yowo.models import _registry as reg

            # Temporarily insert and then check a clearly unregistered key.
            fake_meta = ModelMeta(
                family=ModelFamily.YOLO11,
                size=fake_size,
                input_height=640,
                input_width=640,
                num_classes=10,
                ultralytics_name="fake",
                default_weights_url="https://example.com/fake.pt",
            )
            original = reg._REGISTRY.pop((ModelFamily.YOLO11, ModelSize.XLARGE), None)
            try:
                reg.get(ModelFamily.YOLO11, ModelSize.XLARGE)
            finally:
                if original is not None:
                    reg._REGISTRY[(ModelFamily.YOLO11, ModelSize.XLARGE)] = original
                _ = fake_meta  # reference to silence warning

    def test_raises_model_not_found_with_helpful_message(self) -> None:
        from yowo.models import _registry as reg

        # Remove YOLO12/NANO temporarily and verify message content.
        saved = reg._REGISTRY.pop((ModelFamily.YOLO12, ModelSize.NANO), None)
        try:
            with pytest.raises(ModelNotFoundError, match="yolo12/n"):
                reg.get(ModelFamily.YOLO12, ModelSize.NANO)
        finally:
            if saved is not None:
                reg._REGISTRY[(ModelFamily.YOLO12, ModelSize.NANO)] = saved


class TestListAvailable:
    def test_returns_15_variants(self) -> None:
        available = list_available()
        assert len(available) == 15

    def test_all_entries_are_model_meta(self) -> None:
        for meta in list_available():
            assert isinstance(meta, ModelMeta)

    def test_sorted_by_family_then_size(self) -> None:
        available = list_available()
        keys = [(m.family, m.size) for m in available]
        assert keys == sorted(keys)

    def test_covers_all_three_families(self) -> None:
        families = {m.family for m in list_available()}
        assert families == {ModelFamily.YOLO11, ModelFamily.YOLO12, ModelFamily.YOLO26}

    def test_covers_all_five_sizes(self) -> None:
        sizes = {m.size for m in list_available()}
        assert sizes == {
            ModelSize.NANO,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.XLARGE,
        }


class TestRegister:
    def test_adds_new_variant(self) -> None:
        custom_family = ModelFamily.YOLO11  # reuse existing enum value
        custom_size = ModelSize.NANO

        # Save any existing entry.
        from yowo.models import _registry as reg

        saved = reg._REGISTRY.get((custom_family, custom_size))

        new_meta = ModelMeta(
            family=custom_family,
            size=custom_size,
            input_height=320,
            input_width=320,
            num_classes=10,
            ultralytics_name="custom11n",
            default_weights_url="https://example.com/custom11n.pt",
        )

        try:
            register(new_meta)
            retrieved = get(custom_family, custom_size)
            assert retrieved == new_meta
            assert retrieved.input_height == 320
            assert retrieved.num_classes == 10
        finally:
            # Restore original state.
            if saved is not None:
                reg._REGISTRY[(custom_family, custom_size)] = saved
            else:
                reg._REGISTRY.pop((custom_family, custom_size), None)

    def test_overwrites_existing_variant(self) -> None:
        from yowo.models import _registry as reg

        original = reg._REGISTRY.get((ModelFamily.YOLO26, ModelSize.SMALL))

        overwrite = ModelMeta(
            family=ModelFamily.YOLO26,
            size=ModelSize.SMALL,
            input_height=512,
            input_width=512,
            num_classes=90,
            ultralytics_name="yolo26s_custom",
            default_weights_url="https://example.com/yolo26s_custom.pt",
        )

        try:
            register(overwrite)
            assert get(ModelFamily.YOLO26, ModelSize.SMALL) == overwrite
        finally:
            if original is not None:
                reg._REGISTRY[(ModelFamily.YOLO26, ModelSize.SMALL)] = original
            else:
                reg._REGISTRY.pop((ModelFamily.YOLO26, ModelSize.SMALL), None)
