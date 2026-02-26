"""Unit tests for INT8 export: calibration batches, TRT calibrator, ONNX quantization.

Tests cover:
- calibration_batches() BCHW shape, normalization, tail batches, skip logic
- create_tensorrt_calibrator() factory: DependencyError, batch_size, cache I/O
- quantize_onnx_static() delegation to onnxruntime.quantization
- export_model() INT8 wiring for both TensorRT and ONNX targets
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from yowo.export._calibration import calibration_batches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_images(directory: Path, count: int, size: int = 100) -> list[Path]:
    """Write ``count`` solid-color JPEG images into ``directory``.

    Returns the sorted list of paths (matching resolve_calibration_images order).
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        img = np.full((size, size, 3), fill_value=i * 10 % 256, dtype=np.uint8)
        p = directory / f"img_{i:04d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return sorted(paths)


def _make_mock_trt_module() -> types.ModuleType:
    """Build a mock ``tensorrt`` module with IInt8EntropyCalibrator2 base."""
    mock_trt = types.ModuleType("tensorrt")

    class _MockCalibrator:
        """Fake base class for IInt8EntropyCalibrator2."""

        def __init__(self) -> None:
            pass

    mock_trt.IInt8EntropyCalibrator2 = _MockCalibrator  # type: ignore[attr-defined]
    return mock_trt


def _make_mock_pycuda_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Build mock ``pycuda.driver`` and ``pycuda.autoinit`` modules."""
    pycuda_driver = types.ModuleType("pycuda.driver")
    pycuda_driver.mem_alloc = MagicMock(return_value=12345)  # type: ignore[attr-defined]
    pycuda_driver.memcpy_htod = MagicMock()  # type: ignore[attr-defined]
    pycuda_driver.DeviceAllocation = int  # type: ignore[attr-defined]

    pycuda_autoinit = types.ModuleType("pycuda.autoinit")
    return pycuda_driver, pycuda_autoinit


def _patch_trt_and_pycuda() -> dict[str, types.ModuleType]:
    """Return a sys.modules patch dict with mock tensorrt + pycuda."""
    mock_trt = _make_mock_trt_module()
    pycuda_driver, pycuda_autoinit = _make_mock_pycuda_modules()
    return {
        "tensorrt": mock_trt,
        "pycuda": types.ModuleType("pycuda"),
        "pycuda.driver": pycuda_driver,
        "pycuda.autoinit": pycuda_autoinit,
    }


# ---------------------------------------------------------------------------
# TestCalibrationBatches
# ---------------------------------------------------------------------------


class TestCalibrationBatches:
    """Test calibration_batches() iterator."""

    def test_yields_correct_shape(self, tmp_path: Path) -> None:
        paths = _create_test_images(tmp_path, count=5)
        batches = list(calibration_batches(paths, batch_size=2, input_size=640))
        assert len(batches) == 3  # [2, 2, 1]
        assert batches[0].shape == (2, 3, 640, 640)
        assert batches[0].dtype == np.float32

    def test_tail_batch_smaller(self, tmp_path: Path) -> None:
        paths = _create_test_images(tmp_path, count=5)
        batches = list(calibration_batches(paths, batch_size=3, input_size=320))
        assert len(batches) == 2
        assert batches[0].shape[0] == 3
        assert batches[1].shape[0] == 2

    def test_single_image_batch(self, tmp_path: Path) -> None:
        paths = _create_test_images(tmp_path, count=1)
        batches = list(calibration_batches(paths, batch_size=4, input_size=64))
        assert len(batches) == 1
        assert batches[0].shape == (1, 3, 64, 64)

    def test_values_normalized_zero_to_one(self, tmp_path: Path) -> None:
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        p = tmp_path / "white.jpg"
        cv2.imwrite(str(p), img)
        batch = next(calibration_batches([p], batch_size=1, input_size=64))
        assert batch.max() <= 1.0 + 1e-5
        assert batch.min() >= 0.0 - 1e-5

    def test_black_image_near_zero(self, tmp_path: Path) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        p = tmp_path / "black.jpg"
        cv2.imwrite(str(p), img)
        batch = next(calibration_batches([p], batch_size=1, input_size=64))
        assert batch.max() <= 1e-5

    def test_skips_unreadable_images(self, tmp_path: Path) -> None:
        (tmp_path / "bad.jpg").write_text("not an image")
        good = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "good.jpg"), good)
        paths = [tmp_path / "bad.jpg", tmp_path / "good.jpg"]
        batches = list(calibration_batches(paths, batch_size=2, input_size=64))
        assert len(batches) == 1
        assert batches[0].shape[0] == 1  # only good image

    def test_all_unreadable_yields_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "bad1.jpg").write_text("garbage")
        (tmp_path / "bad2.jpg").write_text("garbage")
        paths = [tmp_path / "bad1.jpg", tmp_path / "bad2.jpg"]
        batches = list(calibration_batches(paths, batch_size=2, input_size=64))
        assert batches == []

    def test_empty_list_yields_nothing(self) -> None:
        batches = list(calibration_batches([], batch_size=4, input_size=640))
        assert batches == []

    def test_output_channels_are_rgb(self, tmp_path: Path) -> None:
        """Verify BGR->RGB swap: pure blue BGR (255,0,0) -> R=0, G=0, B=1."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # BGR blue channel
        p = tmp_path / "blue.jpg"
        cv2.imwrite(str(p), img)
        batch = next(calibration_batches([p], batch_size=1, input_size=32))
        # After swapRB: channel 0 = R (should be ~0), channel 2 = B (should be ~1)
        r_channel = batch[0, 0, :, :]
        b_channel = batch[0, 2, :, :]
        # JPEG compression introduces artifacts, so use a generous tolerance
        assert r_channel.mean() < 0.15
        assert b_channel.mean() > 0.85

    def test_custom_input_size(self, tmp_path: Path) -> None:
        paths = _create_test_images(tmp_path, count=2)
        batches = list(calibration_batches(paths, batch_size=2, input_size=224))
        assert batches[0].shape == (2, 3, 224, 224)

    def test_exact_batch_size_no_remainder(self, tmp_path: Path) -> None:
        paths = _create_test_images(tmp_path, count=6)
        batches = list(calibration_batches(paths, batch_size=3, input_size=64))
        assert len(batches) == 2
        assert all(b.shape[0] == 3 for b in batches)


# ---------------------------------------------------------------------------
# TestCreateTensorrtCalibrator
# ---------------------------------------------------------------------------


class TestCreateTensorrtCalibrator:
    """Test TensorRT calibrator factory (mocked TRT + pycuda).

    Since ``create_tensorrt_calibrator`` uses deferred imports inside the
    function body, we patch ``sys.modules`` to inject mock tensorrt/pycuda
    without needing ``importlib.reload``.
    """

    def test_raises_dependency_error_without_tensorrt(self) -> None:
        from yowo.errors import DependencyError
        from yowo.export._int8 import create_tensorrt_calibrator

        with (
            patch.dict(sys.modules, {"tensorrt": None}),
            pytest.raises(DependencyError, match="tensorrt"),
        ):
            create_tensorrt_calibrator([Path("/fake/img.jpg")])

    def test_raises_dependency_error_without_pycuda(self) -> None:
        from yowo.errors import DependencyError
        from yowo.export._int8 import create_tensorrt_calibrator

        mock_trt = _make_mock_trt_module()
        with (
            patch.dict(
                sys.modules,
                {
                    "tensorrt": mock_trt,
                    "pycuda": None,
                    "pycuda.driver": None,
                    "pycuda.autoinit": None,
                },
            ),
            pytest.raises(DependencyError, match="pycuda"),
        ):
            create_tensorrt_calibrator([Path("/fake/img.jpg")])

    def test_calibrator_get_batch_size(self, tmp_path: Path) -> None:
        from yowo.export._int8 import create_tensorrt_calibrator

        with patch.dict(sys.modules, _patch_trt_and_pycuda()):
            paths = _create_test_images(tmp_path, count=4)
            calibrator = create_tensorrt_calibrator(paths, batch_size=4, input_size=64)
            assert calibrator.get_batch_size() == 4  # type: ignore[attr-defined]

    def test_calibrator_get_batch_returns_device_ptr(self, tmp_path: Path) -> None:
        from yowo.export._int8 import create_tensorrt_calibrator

        with patch.dict(sys.modules, _patch_trt_and_pycuda()):
            paths = _create_test_images(tmp_path, count=2)
            calibrator = create_tensorrt_calibrator(paths, batch_size=2, input_size=64)
            result = calibrator.get_batch(["images"])  # type: ignore[attr-defined]
            assert result is not None
            assert isinstance(result, list)
            assert len(result) == 1

    def test_calibrator_get_batch_returns_none_when_exhausted(self, tmp_path: Path) -> None:
        from yowo.export._int8 import create_tensorrt_calibrator

        with patch.dict(sys.modules, _patch_trt_and_pycuda()):
            paths = _create_test_images(tmp_path, count=2)
            calibrator = create_tensorrt_calibrator(paths, batch_size=4, input_size=64)
            # First batch: 2 images
            result1 = calibrator.get_batch(["images"])  # type: ignore[attr-defined]
            assert result1 is not None
            # Second call: exhausted
            result2 = calibrator.get_batch(["images"])  # type: ignore[attr-defined]
            assert result2 is None

    def test_cache_file_roundtrip(self, tmp_path: Path) -> None:
        from yowo.export._int8 import create_tensorrt_calibrator

        with patch.dict(sys.modules, _patch_trt_and_pycuda()):
            cache_file = tmp_path / "test.calib"
            img_dir = tmp_path / "imgs"
            paths = _create_test_images(img_dir, count=2)
            calibrator = create_tensorrt_calibrator(
                paths, batch_size=2, input_size=64, cache_file=cache_file
            )

            # Write cache
            test_data = b"calibration_cache_data_v1"
            calibrator.write_calibration_cache(memoryview(test_data))  # type: ignore[attr-defined]
            assert cache_file.exists()
            assert cache_file.read_bytes() == test_data

            # Read cache
            result = calibrator.read_calibration_cache()  # type: ignore[attr-defined]
            assert result == test_data

    def test_read_calibration_cache_returns_none_when_missing(self, tmp_path: Path) -> None:
        from yowo.export._int8 import create_tensorrt_calibrator

        with patch.dict(sys.modules, _patch_trt_and_pycuda()):
            cache_file = tmp_path / "nonexistent.calib"
            img_dir = tmp_path / "imgs"
            paths = _create_test_images(img_dir, count=2)
            calibrator = create_tensorrt_calibrator(
                paths, batch_size=2, input_size=64, cache_file=cache_file
            )
            assert calibrator.read_calibration_cache() is None  # type: ignore[attr-defined]

    def test_no_cache_file_means_no_write(self, tmp_path: Path) -> None:
        from yowo.export._int8 import create_tensorrt_calibrator

        with patch.dict(sys.modules, _patch_trt_and_pycuda()):
            img_dir = tmp_path / "imgs"
            paths = _create_test_images(img_dir, count=2)
            calibrator = create_tensorrt_calibrator(
                paths, batch_size=2, input_size=64, cache_file=None
            )
            # Should not raise even without cache_file
            calibrator.write_calibration_cache(memoryview(b"data"))  # type: ignore[attr-defined]
            assert calibrator.read_calibration_cache() is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestQuantizeOnnxStatic
# ---------------------------------------------------------------------------


class TestQuantizeOnnxStatic:
    """Test ONNX static quantization path."""

    def test_raises_dependency_error_without_onnxruntime(self) -> None:
        from yowo.errors import DependencyError
        from yowo.export._int8 import quantize_onnx_static

        with (
            patch.dict(
                sys.modules,
                {"onnxruntime": None, "onnxruntime.quantization": None},
            ),
            pytest.raises(DependencyError, match="onnxruntime"),
        ):
            quantize_onnx_static(
                Path("/fake/model.onnx"),
                Path("/fake/out.onnx"),
                "/fake/cal",
            )

    def test_calls_quantize_static_with_correct_args(self, tmp_path: Path) -> None:
        from yowo.export._int8 import quantize_onnx_static

        cal_dir = tmp_path / "cal"
        _create_test_images(cal_dir, count=15)

        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()
        output_path = tmp_path / "model_int8.onnx"

        # Build mock onnxruntime.quantization
        mock_ort_quant = MagicMock()
        mock_ort_quant.QuantFormat.QDQ = "QDQ"
        mock_ort_quant.QuantType.QInt8 = "QInt8"
        mock_ort_quant.CalibrationMethod.Entropy = "Entropy"
        mock_ort_quant.CalibrationDataReader = type("CalibrationDataReader", (), {})

        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.quantization = mock_ort_quant  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "onnxruntime": mock_ort,
                "onnxruntime.quantization": mock_ort_quant,
            },
        ):
            result = quantize_onnx_static(
                onnx_path, output_path, str(cal_dir), input_size=320, batch_size=4
            )

            mock_ort_quant.quantize_static.assert_called_once()
            call_args = mock_ort_quant.quantize_static.call_args
            assert call_args[0][0] == str(onnx_path)
            assert call_args[0][1] == str(output_path)
            assert result == output_path

    def test_wraps_quantization_failure_in_export_error(self, tmp_path: Path) -> None:
        from yowo.errors import ExportError
        from yowo.export._int8 import quantize_onnx_static

        cal_dir = tmp_path / "cal"
        _create_test_images(cal_dir, count=15)

        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()
        output_path = tmp_path / "model_int8.onnx"

        mock_ort_quant = MagicMock()
        mock_ort_quant.QuantFormat.QDQ = "QDQ"
        mock_ort_quant.QuantType.QInt8 = "QInt8"
        mock_ort_quant.CalibrationMethod.Entropy = "Entropy"
        mock_ort_quant.CalibrationDataReader = type("CalibrationDataReader", (), {})
        mock_ort_quant.quantize_static.side_effect = RuntimeError("ORT crash")

        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.quantization = mock_ort_quant  # type: ignore[attr-defined]

        with (
            patch.dict(
                sys.modules,
                {
                    "onnxruntime": mock_ort,
                    "onnxruntime.quantization": mock_ort_quant,
                },
            ),
            pytest.raises(ExportError, match="ONNX INT8 quantization failed"),
        ):
            quantize_onnx_static(onnx_path, output_path, str(cal_dir))


# ---------------------------------------------------------------------------
# TestExporterInt8Wiring
# ---------------------------------------------------------------------------


class TestExporterInt8Wiring:
    """Test that export_model correctly validates INT8 preconditions."""

    def test_tensorrt_int8_without_calibration_raises(self) -> None:
        from yowo.errors import ConfigError
        from yowo.export._exporter import export_model
        from yowo.types import (
            ExportFormat,
            ModelFamily,
            ModelSize,
            ModelSpec,
            Precision,
        )

        spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
        with pytest.raises(ConfigError, match="calibration"):
            export_model(
                spec,
                ExportFormat.TENSORRT,
                Path("/tmp/out"),
                precision=Precision.INT8,
            )

    def test_onnx_int8_without_calibration_raises(self) -> None:
        from yowo.errors import ConfigError
        from yowo.export._exporter import export_model
        from yowo.types import (
            ExportFormat,
            ModelFamily,
            ModelSize,
            ModelSpec,
            Precision,
        )

        spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
        with pytest.raises(ConfigError, match="calibration"):
            export_model(
                spec,
                ExportFormat.ONNX,
                Path("/tmp/out"),
                precision=Precision.INT8,
            )

    def test_openvino_int8_without_calibration_raises(self) -> None:
        from yowo.errors import ConfigError
        from yowo.export._exporter import export_model
        from yowo.types import (
            ExportFormat,
            ModelFamily,
            ModelSize,
            ModelSpec,
            Precision,
        )

        spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
        with pytest.raises(ConfigError, match="calibration"):
            export_model(
                spec,
                ExportFormat.OPENVINO,
                Path("/tmp/out"),
                precision=Precision.INT8,
            )
