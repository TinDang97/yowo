"""Tests for custom num_classes support.

Verifies that num_classes flows correctly from user API down to
build_model() in the PyTorch backend.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from unittest.mock import MagicMock, patch

import pytest

from yowo.config import ClassificationConfig, InferenceConfig
from yowo.errors import ConfigError
from yowo.types import ModelFamily, ModelSize, ModelSpec

# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


class TestModelSpecNumClasses:
    """Verify num_classes field on ModelSpec."""

    def test_default_is_none(self) -> None:
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
        assert spec.num_classes is None

    def test_set_value(self) -> None:
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        assert spec.num_classes == 10

    def test_frozen(self) -> None:
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        with pytest.raises(AttributeError):
            spec.num_classes = 20  # type: ignore[misc]

    def test_replace(self) -> None:
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
        spec2 = dc_replace(spec, num_classes=5)
        assert spec2.num_classes == 5
        assert spec.num_classes is None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigNumClassesValidation:
    """Both InferenceConfig and ClassificationConfig validate num_classes."""

    def test_inference_config_none_ok(self) -> None:
        cfg = InferenceConfig(num_classes=None)
        assert cfg.num_classes is None

    def test_inference_config_positive_ok(self) -> None:
        cfg = InferenceConfig(num_classes=10)
        assert cfg.num_classes == 10

    def test_inference_config_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="num_classes must be >= 1"):
            InferenceConfig(num_classes=0)

    def test_inference_config_negative_raises(self) -> None:
        with pytest.raises(ConfigError, match="num_classes must be >= 1"):
            InferenceConfig(num_classes=-1)

    def test_classification_config_none_ok(self) -> None:
        cfg = ClassificationConfig(num_classes=None)
        assert cfg.num_classes is None

    def test_classification_config_positive_ok(self) -> None:
        cfg = ClassificationConfig(num_classes=5)
        assert cfg.num_classes == 5

    def test_classification_config_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="num_classes must be >= 1"):
            ClassificationConfig(num_classes=0)

    def test_classification_config_negative_raises(self) -> None:
        with pytest.raises(ConfigError, match="num_classes must be >= 1"):
            ClassificationConfig(num_classes=-1)


# ---------------------------------------------------------------------------
# DetectionEngine threads num_classes
# ---------------------------------------------------------------------------


class TestDetectionEngineNumClasses:
    """Verify DetectionEngine passes num_classes to backend."""

    def test_threads_to_spec(self) -> None:
        """num_classes kwarg flows to ModelSpec in the engine."""
        from yowo.engine import DetectionEngine

        mock_backend = MagicMock()
        mock_backend.backend_type = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.input_shape = (640, 640)

        engine = DetectionEngine(
            backend_instance=mock_backend,
            num_classes=10,
        )
        assert engine._spec.num_classes == 10
        engine.close()

    def test_none_default(self) -> None:
        """num_classes defaults to None when not specified."""
        from yowo.engine import DetectionEngine

        mock_backend = MagicMock()
        mock_backend.backend_type = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.input_shape = (640, 640)

        engine = DetectionEngine(backend_instance=mock_backend)
        assert engine._spec.num_classes is None
        engine.close()


# ---------------------------------------------------------------------------
# ClassificationEngine threads num_classes
# ---------------------------------------------------------------------------


class TestClassificationEngineNumClasses:
    """Verify ClassificationEngine passes num_classes to backend."""

    def test_threads_to_spec(self) -> None:
        from yowo.classify_engine import ClassificationEngine

        mock_backend = MagicMock()
        mock_backend.backend_type = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.input_shape = (224, 224)

        engine = ClassificationEngine(
            backend_instance=mock_backend,
            num_classes=100,
        )
        assert engine._spec.num_classes == 100
        engine.close()

    def test_none_default(self) -> None:
        from yowo.classify_engine import ClassificationEngine

        mock_backend = MagicMock()
        mock_backend.backend_type = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.input_shape = (224, 224)

        engine = ClassificationEngine(backend_instance=mock_backend)
        assert engine._spec.num_classes is None
        engine.close()


# ---------------------------------------------------------------------------
# PyTorchBackend uses spec.num_classes
# ---------------------------------------------------------------------------


class TestPyTorchBackendNumClasses:
    """PyTorchBackend should pass spec.num_classes to build_model/build_classify_model."""

    @pytest.fixture()
    def _hw(self) -> MagicMock:
        hw = MagicMock()
        hw.libraries.torch_version = "2.0.0"
        hw.has_nvidia_gpu = False
        hw.libraries.torch_cuda_available = False
        return hw

    def test_detection_custom_nc(self, _hw: MagicMock) -> None:
        """Detection branch uses spec.num_classes when set."""
        from yowo.backends._pytorch import PyTorchBackend

        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        backend = PyTorchBackend(_hw, model_spec=spec)

        mock_model = MagicMock()
        mock_model.fuse.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
            patch("yowo.arch.build_model", return_value=mock_model) as build_fn,
            patch("yowo.arch._weights.load_weights"),
        ):
            backend.load("fake.pt", device="cpu")

        build_fn.assert_called_once_with(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)

    def test_detection_default_80(self, _hw: MagicMock) -> None:
        """Detection branch defaults to 80 when spec.num_classes is None."""
        from yowo.backends._pytorch import PyTorchBackend

        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
        backend = PyTorchBackend(_hw, model_spec=spec)

        mock_model = MagicMock()
        mock_model.fuse.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        with (
            patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
            patch("yowo.arch.build_model", return_value=mock_model) as build_fn,
            patch("yowo.arch._weights.load_weights"),
        ):
            backend.load("fake.pt", device="cpu")

        build_fn.assert_called_once_with(ModelFamily.YOLO11, ModelSize.NANO, num_classes=80)

    def test_classification_custom_nc(self, _hw: MagicMock) -> None:
        """Classification branch uses spec.num_classes when set."""
        from yowo.backends._pytorch import PyTorchBackend

        spec = ModelSpec(
            ModelFamily.YOLO11,
            ModelSize.NANO,
            task="classify",
            num_classes=100,
        )
        backend = PyTorchBackend(_hw, model_spec=spec)

        mock_model = MagicMock()
        mock_model.fuse.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_meta = MagicMock()
        mock_meta.num_classes = 1000
        mock_meta.input_height = 224
        mock_meta.input_width = 224

        with (
            patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
            patch("yowo.models._registry.get_cls", return_value=mock_meta),
            patch("yowo.arch.build_classify_model", return_value=mock_model) as build_fn,
            patch("yowo.arch._weights.load_classify_weights"),
        ):
            backend.load("fake.pt", device="cpu")

        build_fn.assert_called_once_with(
            ModelFamily.YOLO11,
            ModelSize.NANO,
            num_classes=100,
        )

    def test_classification_default_registry(self, _hw: MagicMock) -> None:
        """Classification branch uses registry default when spec.num_classes is None."""
        from yowo.backends._pytorch import PyTorchBackend

        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task="classify")
        backend = PyTorchBackend(_hw, model_spec=spec)

        mock_model = MagicMock()
        mock_model.fuse.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_meta = MagicMock()
        mock_meta.num_classes = 1000
        mock_meta.input_height = 224
        mock_meta.input_width = 224

        with (
            patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
            patch("yowo.models._registry.get_cls", return_value=mock_meta),
            patch("yowo.arch.build_classify_model", return_value=mock_model) as build_fn,
            patch("yowo.arch._weights.load_classify_weights"),
        ):
            backend.load("fake.pt", device="cpu")

        # Should use meta.num_classes (1000) not 80
        build_fn.assert_called_once_with(
            ModelFamily.YOLO11,
            ModelSize.NANO,
            num_classes=1000,
        )


# ---------------------------------------------------------------------------
# Convenience API threads num_classes
# ---------------------------------------------------------------------------


class TestConvenienceNumClasses:
    """detect() and classify() pass num_classes through."""

    def test_detect_num_classes(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = []

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source"),
        ):
            from yowo._convenience import detect

            detect("img.jpg", num_classes=10)

        call_kwargs = eng_cls.call_args.kwargs
        assert call_kwargs["num_classes"] == 10

    def test_classify_num_classes(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = iter([])

        with (
            patch("yowo.classify_engine.ClassificationEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source"),
        ):
            from yowo._convenience import classify

            classify("img.jpg", num_classes=100)

        call_kwargs = eng_cls.call_args.kwargs
        assert call_kwargs["num_classes"] == 100

    def test_detect_default_none(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = []

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source"),
        ):
            from yowo._convenience import detect

            detect("img.jpg")

        call_kwargs = eng_cls.call_args.kwargs
        assert call_kwargs["num_classes"] is None
