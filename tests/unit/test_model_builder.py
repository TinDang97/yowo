"""Tests for ModelBuilder protocol support.

Verifies that custom model architectures can be plugged into
PyTorchBackend's inference pipeline via the ModelBuilder protocol.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from yowo.backends import ModelBuilder, create_backend
from yowo.types import BackendType, ModelFamily, ModelSize, ModelSpec

# ---------------------------------------------------------------------------
# ModelBuilder Protocol
# ---------------------------------------------------------------------------


class _FakeBuilder:
    """Concrete ModelBuilder for tests."""

    def __init__(self, input_shape: tuple[int, int] = (416, 416)) -> None:
        self._input_shape = input_shape
        self.build_calls: list[dict[str, Any]] = []

    def build(self, num_classes: int, device: str) -> Any:
        self.build_calls.append({"num_classes": num_classes, "device": device})
        model = MagicMock()
        model.forward = MagicMock(return_value=MagicMock())
        return model

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape


class TestModelBuilderProtocol:
    """Verify ModelBuilder is a runtime-checkable structural protocol."""

    def test_structural_check(self) -> None:
        builder = _FakeBuilder()
        assert isinstance(builder, ModelBuilder)

    def test_missing_build_fails(self) -> None:
        class _Incomplete:
            @property
            def input_shape(self) -> tuple[int, int]:
                return (640, 640)

        assert not isinstance(_Incomplete(), ModelBuilder)

    def test_missing_input_shape_fails(self) -> None:
        class _Incomplete:
            def build(self, num_classes: int, device: str) -> Any:
                return MagicMock()

        assert not isinstance(_Incomplete(), ModelBuilder)


# ---------------------------------------------------------------------------
# create_backend threads ModelBuilder
# ---------------------------------------------------------------------------


class TestCreateBackend:
    """Factory passes model_builder to PyTorchBackend only."""

    def test_pytorch_receives_builder(self) -> None:
        """create_backend passes model_builder to PyTorchBackend.__init__."""
        builder = _FakeBuilder()
        hw = MagicMock()
        hw.libraries.torch_version = "2.0.0"
        hw.has_nvidia_gpu = False

        with patch("yowo.backends._pytorch.PyTorchBackend.__init__", return_value=None) as init_fn:
            create_backend(
                BackendType.PYTORCH,
                hw,
                model_spec=ModelSpec(ModelFamily.YOLO11, ModelSize.NANO),
                model_builder=builder,
            )

        _, kwargs = init_fn.call_args
        assert kwargs["model_builder"] is builder

    def test_onnx_ignores_builder(self) -> None:
        """Non-PyTorch backends ignore model_builder silently."""
        builder = _FakeBuilder()
        hw = MagicMock()
        hw.libraries.onnxruntime_version = "1.16.0"
        hw.has_nvidia_gpu = False

        with patch("yowo.backends._onnx.OnnxBackend.__init__", return_value=None):
            # Should not raise — builder is simply ignored
            backend = create_backend(
                BackendType.ONNX,
                hw,
                model_builder=builder,
            )
            assert backend is not None


# ---------------------------------------------------------------------------
# PyTorchBackend with ModelBuilder
# ---------------------------------------------------------------------------


class TestPyTorchBackendBuilder:
    """PyTorchBackend delegates to ModelBuilder when provided."""

    @pytest.fixture()
    def _hw(self) -> MagicMock:
        hw = MagicMock()
        hw.libraries.torch_version = "2.0.0"
        hw.has_nvidia_gpu = False
        hw.libraries.torch_cuda_available = False
        return hw

    def test_builder_called_on_load(self, _hw: MagicMock) -> None:
        """load() calls builder.build() with correct num_classes and device."""
        from yowo.backends._pytorch import PyTorchBackend

        builder = _FakeBuilder(input_shape=(416, 416))
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        backend = PyTorchBackend(_hw, model_spec=spec, model_builder=builder)

        with patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"):
            backend.load("dummy.pt", device="cpu")

        assert len(builder.build_calls) == 1
        assert builder.build_calls[0] == {"num_classes": 10, "device": "cpu"}

    def test_builder_sets_input_shape(self, _hw: MagicMock) -> None:
        """Backend.input_shape comes from builder.input_shape."""
        from yowo.backends._pytorch import PyTorchBackend

        builder = _FakeBuilder(input_shape=(320, 320))
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        backend = PyTorchBackend(_hw, model_spec=spec, model_builder=builder)

        with patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"):
            backend.load("dummy.pt", device="cpu")

        assert backend.input_shape == (320, 320)

    def test_builder_marks_loaded(self, _hw: MagicMock) -> None:
        """After builder.build(), backend.is_loaded is True."""
        from yowo.backends._pytorch import PyTorchBackend

        builder = _FakeBuilder()
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        backend = PyTorchBackend(_hw, model_spec=spec, model_builder=builder)

        with patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"):
            backend.load("dummy.pt", device="cpu")

        assert backend.is_loaded

    def test_builder_default_nc_80(self, _hw: MagicMock) -> None:
        """When spec.num_classes is None, builder gets 80 as default."""
        from yowo.backends._pytorch import PyTorchBackend

        builder = _FakeBuilder()
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
        backend = PyTorchBackend(_hw, model_spec=spec, model_builder=builder)

        with patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"):
            backend.load("dummy.pt", device="cpu")

        assert builder.build_calls[0]["num_classes"] == 80

    def test_builder_failure_raises_backend_load_error(self, _hw: MagicMock) -> None:
        """If builder.build() raises, BackendLoadError is raised."""
        from yowo.backends._pytorch import PyTorchBackend
        from yowo.errors import BackendLoadError

        class _FailBuilder:
            @property
            def input_shape(self) -> tuple[int, int]:
                return (416, 416)

            def build(self, num_classes: int, device: str) -> Any:
                raise RuntimeError("custom build failed")

        builder = _FailBuilder()
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        backend = PyTorchBackend(_hw, model_spec=spec, model_builder=builder)

        with (
            patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
            pytest.raises(BackendLoadError, match="model_builder.build\\(\\) failed"),
        ):
            backend.load("dummy.pt", device="cpu")


# ---------------------------------------------------------------------------
# Engine with ModelBuilder
# ---------------------------------------------------------------------------


class TestDetectionEngineBuilder:
    """DetectionEngine accepts model_builder kwarg."""

    def test_threads_to_backend(self) -> None:
        """model_builder is passed to create_backend."""
        from yowo.engine import DetectionEngine

        builder = _FakeBuilder()
        mock_backend = MagicMock()
        mock_backend.backend_type = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.input_shape = (640, 640)

        engine = DetectionEngine(
            backend_instance=mock_backend,
            model_builder=builder,
        )
        assert engine._model_builder is builder
        engine.close()

    def test_backend_instance_wins_over_builder(self) -> None:
        """When backend_instance is provided, model_builder doesn't affect backend selection."""
        from yowo.engine import DetectionEngine

        builder = _FakeBuilder()
        mock_backend = MagicMock()
        mock_backend.backend_type = BackendType.ONNX
        mock_backend.is_loaded = True
        mock_backend.input_shape = (640, 640)

        engine = DetectionEngine(
            backend_instance=mock_backend,
            model_builder=builder,
        )
        # backend_instance is used directly — builder is stored but not used for backend creation
        assert engine._backend is mock_backend
        engine.close()


class TestClassificationEngineBuilder:
    """ClassificationEngine accepts model_builder kwarg."""

    def test_threads_to_backend(self) -> None:
        from yowo.classify_engine import ClassificationEngine

        builder = _FakeBuilder(input_shape=(224, 224))
        mock_backend = MagicMock()
        mock_backend.backend_type = MagicMock()
        mock_backend.is_loaded = True
        mock_backend.input_shape = (224, 224)

        engine = ClassificationEngine(
            backend_instance=mock_backend,
            model_builder=builder,
        )
        assert engine._model_builder is builder
        engine.close()


# ---------------------------------------------------------------------------
# _resolve_model_meta with ModelBuilder
# ---------------------------------------------------------------------------


class TestResolveModelMeta:
    """_resolve_model_meta handles custom builder fallback."""

    def test_registry_hit_no_builder(self) -> None:
        """Registry model returns normal meta when no builder."""
        from yowo.engine import _resolve_model_meta

        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
        meta = _resolve_model_meta(spec, model_builder=None)
        assert meta.family == ModelFamily.YOLO11
        assert meta.num_classes == 80

    def test_registry_miss_with_builder(self) -> None:
        """Non-registry model with builder creates synthetic meta."""
        from yowo.engine import _resolve_model_meta
        from yowo.errors import ModelNotFoundError

        builder = _FakeBuilder(input_shape=(320, 320))
        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)

        with patch(
            "yowo.engine._registry_get",
            side_effect=ModelNotFoundError("not found"),
        ):
            meta = _resolve_model_meta(spec, model_builder=builder)

        assert meta.input_height == 320
        assert meta.input_width == 320
        assert meta.num_classes == 10
        assert meta.weight_stem == "custom"

    def test_registry_miss_no_builder_raises(self) -> None:
        """Non-registry model without builder re-raises ModelNotFoundError."""
        from yowo.engine import _resolve_model_meta
        from yowo.errors import ModelNotFoundError

        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)

        with (
            patch(
                "yowo.engine._registry_get",
                side_effect=ModelNotFoundError("not found"),
            ),
            pytest.raises(ModelNotFoundError),
        ):
            _resolve_model_meta(spec, model_builder=None)

    def test_num_classes_override_replaces_registry(self) -> None:
        """spec.num_classes overrides registry default."""
        from yowo.engine import _resolve_model_meta

        spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, num_classes=10)
        meta = _resolve_model_meta(spec, model_builder=None)
        assert meta.num_classes == 10
