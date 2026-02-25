"""Unit tests for yowo.types and yowo.errors.

Covers:
- All enum string-construction round-trips
- BoundingBox computed properties (area, as_xyxy)
- Detection computed properties (num_boxes, has_detections)
- ModelSpec construction with and without weights_path
- Frame and PreprocessedTensor construction and properties
- BackendSelection and ExportResult construction
- DependencyError package/install_cmd attributes
- Full error hierarchy membership
- InferenceConfig threshold validation
- ExportConfig INT8 / calibration_data validation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yowo.config import ExportConfig, InferenceConfig
from yowo.errors import (
    BackendError,
    BackendLoadError,
    ConfigError,
    DependencyError,
    DeviceError,
    ExportError,
    ExportUnsupportedError,
    InferenceError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    SourceError,
    SourceTimeoutError,
    YowoError,
)
from yowo.types import (
    BackendSelection,
    BackendType,
    BoundingBox,
    CPUArch,
    Detection,
    DeviceType,
    ExportFormat,
    ExportResult,
    Frame,
    FrameDropPolicy,
    GPUArch,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
    PreprocessedTensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 4, w: int = 6) -> Frame:
    """Return a minimal Frame with deterministic pixel data."""
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0)


def _make_box(
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 110.0,
    y2: float = 120.0,
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> BoundingBox:
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


# ---------------------------------------------------------------------------
# Enum round-trip tests
# ---------------------------------------------------------------------------


class TestEnumRoundTrips:
    """Every enum must be constructible from its string value."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("pytorch", BackendType.PYTORCH),
            ("onnx", BackendType.ONNX),
            ("tensorrt", BackendType.TENSORRT),
            ("openvino", BackendType.OPENVINO),
        ],
    )
    def test_backend_type(self, value: str, expected: BackendType) -> None:
        assert BackendType(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("cuda", DeviceType.CUDA),
            ("cpu", DeviceType.CPU),
        ],
    )
    def test_device_type(self, value: str, expected: DeviceType) -> None:
        assert DeviceType(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("x86_64", CPUArch.X86_64),
            ("aarch64", CPUArch.AARCH64),
        ],
    )
    def test_cpu_arch(self, value: str, expected: CPUArch) -> None:
        assert CPUArch(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("sm_75", GPUArch.TURING),
            ("sm_80", GPUArch.AMPERE),
            ("sm_86", GPUArch.AMPERE_GA10X),
            ("sm_87", GPUArch.ORIN),
            ("sm_89", GPUArch.ADA),
            ("sm_90", GPUArch.HOPPER),
            ("unknown", GPUArch.UNKNOWN),
        ],
    )
    def test_gpu_arch(self, value: str, expected: GPUArch) -> None:
        assert GPUArch(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("yolo11", ModelFamily.YOLO11),
            ("yolo26", ModelFamily.YOLO26),
        ],
    )
    def test_model_family(self, value: str, expected: ModelFamily) -> None:
        assert ModelFamily(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("n", ModelSize.NANO),
            ("s", ModelSize.SMALL),
            ("m", ModelSize.MEDIUM),
            ("l", ModelSize.LARGE),
            ("x", ModelSize.XLARGE),
        ],
    )
    def test_model_size(self, value: str, expected: ModelSize) -> None:
        assert ModelSize(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("onnx", ExportFormat.ONNX),
            ("tensorrt", ExportFormat.TENSORRT),
            ("openvino", ExportFormat.OPENVINO),
        ],
    )
    def test_export_format(self, value: str, expected: ExportFormat) -> None:
        assert ExportFormat(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("fp32", Precision.FP32),
            ("fp16", Precision.FP16),
            ("int8", Precision.INT8),
        ],
    )
    def test_precision(self, value: str, expected: Precision) -> None:
        assert Precision(value) is expected

    def test_invalid_backend_type_raises(self) -> None:
        with pytest.raises(ValueError):
            BackendType("invalid")

    def test_invalid_model_family_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelFamily("yolo99")


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_area_standard(self) -> None:
        box = _make_box(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        assert box.area == pytest.approx(10000.0)

    def test_area_zero_for_degenerate_box(self) -> None:
        box = _make_box(x1=50.0, y1=50.0, x2=50.0, y2=50.0)
        assert box.area == pytest.approx(0.0)

    def test_area_zero_when_inverted(self) -> None:
        # x2 < x1 → width clamped to 0
        box = _make_box(x1=100.0, y1=20.0, x2=50.0, y2=120.0)
        assert box.area == pytest.approx(0.0)

    def test_as_xyxy_returns_tuple(self) -> None:
        box = _make_box(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert box.as_xyxy == (1.0, 2.0, 3.0, 4.0)

    def test_as_xyxy_type(self) -> None:
        box = _make_box()
        result = box.as_xyxy
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_default_class_name_is_empty_string(self) -> None:
        box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0, confidence=0.5, class_id=1)
        assert box.class_name == ""

    def test_bounding_box_is_frozen(self) -> None:
        box = _make_box()
        with pytest.raises((AttributeError, TypeError)):
            box.confidence = 0.1  # type: ignore[misc]

    def test_fractional_area(self) -> None:
        box = _make_box(x1=0.0, y1=0.0, x2=1.5, y2=2.0)
        assert box.area == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ModelSpec tests
# ---------------------------------------------------------------------------


class TestModelSpec:
    def test_defaults(self) -> None:
        spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
        assert spec.task == "detect"
        assert spec.weights_path is None

    def test_with_weights_path(self) -> None:
        p = Path("/tmp/weights.pt")
        spec = ModelSpec(
            family=ModelFamily.YOLO11,
            size=ModelSize.SMALL,
            weights_path=p,
        )
        assert spec.weights_path == p

    def test_custom_task(self) -> None:
        spec = ModelSpec(
            family=ModelFamily.YOLO11,
            size=ModelSize.MEDIUM,
            task="segment",
        )
        assert spec.task == "segment"

    def test_is_frozen(self) -> None:
        spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
        with pytest.raises((AttributeError, TypeError)):
            spec.task = "segment"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
        b = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
        assert a == b

    def test_inequality_on_size(self) -> None:
        a = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
        b = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.LARGE)
        assert a != b


# ---------------------------------------------------------------------------
# Frame tests
# ---------------------------------------------------------------------------


class TestFrame:
    def test_shape_accessors(self) -> None:
        frame = _make_frame(h=480, w=640)
        assert frame.height == 480
        assert frame.width == 640
        assert frame.shape_hw == (480, 640)

    def test_default_fields(self) -> None:
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        frame = Frame(pixels=pixels)
        assert frame.source_id == ""
        assert frame.frame_index == 0
        assert frame.timestamp_ms == 0.0

    def test_pixels_dtype_is_uint8(self) -> None:
        frame = _make_frame()
        assert frame.pixels.dtype == np.uint8

    def test_frame_stores_correct_pixel_values(self) -> None:
        pixels = np.full((4, 6, 3), 128, dtype=np.uint8)
        frame = Frame(pixels=pixels)
        assert frame.pixels[0, 0, 0] == 128


# ---------------------------------------------------------------------------
# PreprocessedTensor tests
# ---------------------------------------------------------------------------


class TestPreprocessedTensor:
    def _make_tensor(self, batch: int = 1) -> PreprocessedTensor:
        data = np.zeros((batch, 3, 640, 640), dtype=np.float32)
        return PreprocessedTensor(
            data=data,
            original_shapes=tuple((480, 640) for _ in range(batch)),
            input_shape=(640, 640),
            scale_factors=tuple((1.0, 1.0) for _ in range(batch)),
            pad_offsets=tuple((0, 0) for _ in range(batch)),
        )

    def test_batch_size(self) -> None:
        t = self._make_tensor(batch=4)
        assert t.batch_size == 4

    def test_single_batch(self) -> None:
        t = self._make_tensor(batch=1)
        assert t.batch_size == 1

    def test_data_dtype(self) -> None:
        t = self._make_tensor()
        assert t.data.dtype == np.float32

    def test_original_shapes_length_matches_batch(self) -> None:
        t = self._make_tensor(batch=3)
        assert len(t.original_shapes) == 3


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetection:
    def _make_detection(self, boxes: tuple[BoundingBox, ...]) -> Detection:
        return Detection(
            frame=_make_frame(),
            boxes=boxes,
            inference_time_ms=12.5,
            backend=BackendType.PYTORCH,
            model_spec=ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO),
        )

    def test_num_boxes_zero(self) -> None:
        d = self._make_detection(())
        assert d.num_boxes == 0

    def test_num_boxes_multiple(self) -> None:
        boxes = (_make_box(), _make_box(class_id=1))
        d = self._make_detection(boxes)
        assert d.num_boxes == 2

    def test_has_detections_false_when_empty(self) -> None:
        d = self._make_detection(())
        assert d.has_detections is False

    def test_has_detections_true_when_not_empty(self) -> None:
        d = self._make_detection((_make_box(),))
        assert d.has_detections is True

    def test_is_frozen(self) -> None:
        d = self._make_detection(())
        with pytest.raises((AttributeError, TypeError)):
            d.inference_time_ms = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BackendSelection tests
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_defaults(self) -> None:
        sel = BackendSelection(
            backend=BackendType.TENSORRT,
            device_type=DeviceType.CUDA,
            precision=Precision.FP16,
        )
        assert sel.device_index == 0
        assert sel.reason == ""

    def test_is_frozen(self) -> None:
        sel = BackendSelection(
            backend=BackendType.ONNX,
            device_type=DeviceType.CPU,
            precision=Precision.FP32,
        )
        with pytest.raises((AttributeError, TypeError)):
            sel.reason = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExportResult tests
# ---------------------------------------------------------------------------


class TestExportResult:
    def test_construction(self) -> None:
        result = ExportResult(
            model_name="yolo26n",
            format=ExportFormat.ONNX,
            precision=Precision.FP16,
            output_path=Path("/tmp/model.onnx"),
            file_size_bytes=4096,
            export_time_s=3.14,
            created_at="2026-02-23T00:00:00Z",
        )
        assert result.model_name == "yolo26n"
        assert result.file_size_bytes == 4096

    def test_is_frozen(self) -> None:
        result = ExportResult(
            model_name="x",
            format=ExportFormat.TENSORRT,
            precision=Precision.INT8,
            output_path=Path("/tmp/m.engine"),
            file_size_bytes=1,
            export_time_s=1.0,
            created_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.model_name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DependencyError tests
# ---------------------------------------------------------------------------


class TestDependencyError:
    def test_stores_package(self) -> None:
        err = DependencyError("onnxruntime")
        assert err.package == "onnxruntime"

    def test_default_install_cmd(self) -> None:
        err = DependencyError("onnxruntime")
        assert err.install_cmd == "pip install onnxruntime"

    def test_custom_install_cmd(self) -> None:
        err = DependencyError("onnxruntime", install_cmd="pip install yowo[onnx]")
        assert err.install_cmd == "pip install yowo[onnx]"

    def test_message_includes_package(self) -> None:
        err = DependencyError("tensorrt")
        assert "tensorrt" in str(err)

    def test_message_includes_install_cmd(self) -> None:
        err = DependencyError("openvino", install_cmd="pip install yowo[openvino]")
        assert "pip install yowo[openvino]" in str(err)

    def test_optional_message_appended(self) -> None:
        err = DependencyError("torch", message="Required for PyTorch backend.")
        assert "Required for PyTorch backend." in str(err)

    def test_is_yowo_error(self) -> None:
        assert isinstance(DependencyError("x"), YowoError)


# ---------------------------------------------------------------------------
# Error hierarchy tests
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_backend_load_error_is_backend_error(self) -> None:
        assert issubclass(BackendLoadError, BackendError)

    def test_inference_error_is_backend_error(self) -> None:
        assert issubclass(InferenceError, BackendError)

    def test_backend_error_is_yowo_error(self) -> None:
        assert issubclass(BackendError, YowoError)

    def test_device_error_is_yowo_error(self) -> None:
        assert issubclass(DeviceError, YowoError)

    def test_model_not_found_is_model_error(self) -> None:
        assert issubclass(ModelNotFoundError, ModelError)

    def test_model_load_error_is_model_error(self) -> None:
        assert issubclass(ModelLoadError, ModelError)

    def test_model_error_is_yowo_error(self) -> None:
        assert issubclass(ModelError, YowoError)

    def test_export_unsupported_is_export_error(self) -> None:
        assert issubclass(ExportUnsupportedError, ExportError)

    def test_export_error_is_yowo_error(self) -> None:
        assert issubclass(ExportError, YowoError)

    def test_source_timeout_is_source_error(self) -> None:
        assert issubclass(SourceTimeoutError, SourceError)

    def test_source_error_is_yowo_error(self) -> None:
        assert issubclass(SourceError, YowoError)

    def test_config_error_is_yowo_error(self) -> None:
        assert issubclass(ConfigError, YowoError)

    def test_all_errors_catchable_as_yowo_error(self) -> None:
        errors = [
            DependencyError("pkg"),
            BackendLoadError("load"),
            InferenceError("inf"),
            DeviceError("dev"),
            ModelNotFoundError("not found"),
            ModelLoadError("load"),
            ExportUnsupportedError("unsup"),
            SourceTimeoutError("timeout"),
            ConfigError("cfg"),
        ]
        for err in errors:
            assert isinstance(err, YowoError), f"{type(err).__name__} is not a YowoError"


# ---------------------------------------------------------------------------
# InferenceConfig validation tests
# ---------------------------------------------------------------------------


class TestInferenceConfigValidation:
    def test_defaults_pass_validation(self) -> None:
        cfg = InferenceConfig()
        assert cfg.confidence_threshold == 0.25
        assert cfg.iou_threshold == 0.45
        assert cfg.batch_size == 1

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="confidence_threshold"):
            InferenceConfig(confidence_threshold=-0.01)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ConfigError, match="confidence_threshold"):
            InferenceConfig(confidence_threshold=1.001)

    def test_confidence_at_boundaries_accepted(self) -> None:
        InferenceConfig(confidence_threshold=0.0)
        InferenceConfig(confidence_threshold=1.0)

    def test_iou_below_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="iou_threshold"):
            InferenceConfig(iou_threshold=-0.1)

    def test_iou_above_one_raises(self) -> None:
        with pytest.raises(ConfigError, match="iou_threshold"):
            InferenceConfig(iou_threshold=1.5)

    def test_iou_at_boundaries_accepted(self) -> None:
        InferenceConfig(iou_threshold=0.0)
        InferenceConfig(iou_threshold=1.0)

    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="batch_size"):
            InferenceConfig(batch_size=0)

    def test_batch_size_negative_raises(self) -> None:
        with pytest.raises(ConfigError, match="batch_size"):
            InferenceConfig(batch_size=-1)

    def test_batch_size_one_accepted(self) -> None:
        cfg = InferenceConfig(batch_size=1)
        assert cfg.batch_size == 1

    def test_cache_default_false(self) -> None:
        cfg = InferenceConfig()
        assert cfg.cache is False

    def test_kv_cache_default_false(self) -> None:
        cfg = InferenceConfig()
        assert cfg.kv_cache is False

    def test_cache_dir_default_none(self) -> None:
        cfg = InferenceConfig()
        assert cfg.cache_dir is None

    def test_backend_none_is_default(self) -> None:
        cfg = InferenceConfig()
        assert cfg.backend is None

    def test_precision_none_is_default(self) -> None:
        cfg = InferenceConfig()
        assert cfg.precision is None


# ---------------------------------------------------------------------------
# ExportConfig validation tests
# ---------------------------------------------------------------------------


class TestExportConfigValidation:
    def test_defaults_pass_validation(self) -> None:
        cfg = ExportConfig()
        assert cfg.target_format == ExportFormat.ONNX
        assert cfg.precision == Precision.FP16

    def test_int8_without_calibration_raises(self) -> None:
        with pytest.raises(ConfigError, match="calibration_data"):
            ExportConfig(precision=Precision.INT8)

    def test_int8_with_calibration_accepted(self) -> None:
        cfg = ExportConfig(
            precision=Precision.INT8,
            calibration_data="/data/calib_images",
        )
        assert cfg.calibration_data == "/data/calib_images"

    def test_fp16_without_calibration_accepted(self) -> None:
        cfg = ExportConfig(precision=Precision.FP16)
        assert cfg.precision == Precision.FP16

    def test_fp32_without_calibration_accepted(self) -> None:
        cfg = ExportConfig(precision=Precision.FP32)
        assert cfg.precision == Precision.FP32

    def test_default_output_dir(self) -> None:
        cfg = ExportConfig()
        assert cfg.output_dir == Path.home() / ".yowo" / "models"

    def test_custom_output_dir(self) -> None:
        cfg = ExportConfig(output_dir=Path("/custom/dir"))
        assert cfg.output_dir == Path("/custom/dir")

    def test_imgsz_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="imgsz"):
            ExportConfig(imgsz=0)

    def test_imgsz_negative_raises(self) -> None:
        with pytest.raises(ConfigError, match="imgsz"):
            ExportConfig(imgsz=-640)

    def test_dynamic_batch_default_false(self) -> None:
        cfg = ExportConfig()
        assert cfg.dynamic_batch is False


# ---------------------------------------------------------------------------
# FrameDropPolicy tests
# ---------------------------------------------------------------------------


class TestFrameDropPolicy:
    def test_enum_values(self) -> None:
        assert FrameDropPolicy.NONE.value == "none"
        assert FrameDropPolicy.LATEST.value == "latest"
        assert FrameDropPolicy.SKIP_OLDEST.value == "skip_oldest"

    def test_from_string(self) -> None:
        assert FrameDropPolicy("none") == FrameDropPolicy.NONE
        assert FrameDropPolicy("latest") == FrameDropPolicy.LATEST
        assert FrameDropPolicy("skip_oldest") == FrameDropPolicy.SKIP_OLDEST

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            FrameDropPolicy("invalid")

    def test_all_members(self) -> None:
        members = {p.value for p in FrameDropPolicy}
        assert members == {"none", "latest", "skip_oldest"}


# ---------------------------------------------------------------------------
# is_free_threaded tests
# ---------------------------------------------------------------------------


def test_is_free_threaded_returns_bool() -> None:
    from yowo.types import is_free_threaded

    result = is_free_threaded()
    assert isinstance(result, bool)


def test_is_free_threaded_false_on_standard_python() -> None:
    """On standard (GIL) Python, should return False."""
    import sys

    try:
        gil_enabled = sys._is_gil_enabled()  # type: ignore[attr-defined]
        from yowo.types import is_free_threaded

        assert is_free_threaded() == (not gil_enabled)
    except AttributeError:
        # Python < 3.13 — no _is_gil_enabled, is_free_threaded returns False
        from yowo.types import is_free_threaded

        assert is_free_threaded() is False
