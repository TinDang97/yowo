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


def _stub_report(output_path: Path) -> object:
    """A passing ParityReport, so gate-independent checks isolate what they test."""
    from yowo.export._int8 import ParityReport

    return ParityReport(
        passed=True,
        enforced=True,
        output_path=str(output_path),
        total_recall=1.0,
        gated_recall=1.0,
        fp32_detections=4,
        int8_detections=4,
        gated_detections=4,
        gated_matched=4,
        int8_unmatched=0,
        missed=(),
        floor=1.0,
        margin=2.0,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        held_out=False,
        parity_set_source="calibration",
        provider="CPUExecutionProvider",
        images=("c0.jpg",),
    )


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
# TestCalibrationMatchesInference
# ---------------------------------------------------------------------------


class TestCalibrationMatchesInference:
    """Calibration must see the distribution inference produces, not a nearby one.

    ``calibration_batches`` stretch-resized to a square while the engine
    letterboxes (``yowo.io._decode.preprocess``). INT8 scales derived from a
    distribution that never occurs at inference are a silent accuracy loss with
    a docstring claiming the two match.

    Every check above this class feeds a SQUARE image, where stretch and
    letterbox are the same operation — so none of them could detect the
    difference (Q4). These use non-square inputs.
    """

    def test_calibration_batch_equals_the_inference_preprocess(self, tmp_path: Path) -> None:
        """Identical BY CONSTRUCTION, element for element — not by two implementations agreeing."""
        from yowo.io._decode import preprocess
        from yowo.types import Frame

        img = np.zeros((90, 160, 3), dtype=np.uint8)
        img[20:70, 40:120] = (30, 200, 90)
        p = tmp_path / "wide.png"
        cv2.imwrite(str(p), img)

        batch = next(calibration_batches([p], batch_size=1, input_size=64))
        expected = preprocess(
            [Frame(pixels=cv2.imread(str(p)), source_id=str(p), frame_index=0)], (64, 64)
        ).data

        assert batch.shape == expected.shape
        assert np.array_equal(batch, expected)

    def test_calibration_batch_letterboxes_rather_than_stretches(self, tmp_path: Path) -> None:
        """A 2:1 image gains gray bars; it does not get squashed into a square."""
        img = np.full((80, 160, 3), 255, dtype=np.uint8)
        p = tmp_path / "wide.png"
        cv2.imwrite(str(p), img)

        batch = next(calibration_batches([p], batch_size=1, input_size=64))
        # 160x80 into 64x64 -> 64x32 content, 16 rows of padding top and bottom.
        fill = pytest.approx(114 / 255, abs=1e-3)
        assert batch[0, 0, 0, 0] == fill, "letterbox fill, not content"
        assert batch[0, 0, 32, 32] == pytest.approx(1.0, abs=1e-2), "content in the middle"
        assert batch[0, 0, 63, 63] == fill

    def test_calibration_batches_still_skip_unreadable_images(self, tmp_path: Path) -> None:
        """``preprocess`` raises on an empty frame list; the skip-and-warn guard stays."""
        (tmp_path / "bad.jpg").write_text("not an image")
        cv2.imwrite(str(tmp_path / "good.png"), np.full((50, 90, 3), 12, dtype=np.uint8))
        batches = list(
            calibration_batches(
                [tmp_path / "bad.jpg", tmp_path / "good.png"], batch_size=2, input_size=64
            )
        )
        assert len(batches) == 1
        assert batches[0].shape == (1, 3, 64, 64)


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


def _detection_onnx(path: Path) -> Path:
    """A single-input graph shaped like a detection export: Conv -> decode tail."""
    import onnx
    from onnx import TensorProto, helper

    g = helper.make_graph(
        [
            helper.make_node("Conv", ["images"], ["b"], name="backbone"),
            helper.make_node("Conv", ["b"], ["h"], name="head"),
            helper.make_node("Softmax", ["h"], ["s"], name="dfl_softmax"),
            helper.make_node("Sigmoid", ["s"], ["output0"], name="cls_sigmoid"),
        ],
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, None)],
    )
    onnx.save(helper.make_model(g), str(path))
    return path


def _mock_ort_quant(crash: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.QuantFormat.QDQ = "QDQ"
    mock.QuantType.QInt8 = "QInt8"
    mock.CalibrationMethod.Entropy = "Entropy"
    mock.CalibrationDataReader = type("CalibrationDataReader", (), {})

    def _run(src: str, dst: str, *a: object, **kw: object) -> None:
        if crash:
            raise RuntimeError("ORT crash")
        Path(dst).write_bytes(b"quantized")

    mock.quantize_static.side_effect = _run
    return mock


def _patch_ort(mock: MagicMock) -> object:
    mod = types.ModuleType("onnxruntime")
    mod.quantization = mock  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"onnxruntime": mod, "onnxruntime.quantization": mock})


class TestQuantizeOnnxStatic:
    """Test ONNX static quantization path."""

    def test_raises_dependency_error_without_onnxruntime(self) -> None:
        from yowo.errors import DependencyError
        from yowo.export._int8 import quantize_onnx_static
        from yowo.types import ModelFamily, ModelSize, ModelSpec

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
                model_spec=ModelSpec(ModelFamily.YOLO11, ModelSize.NANO),
            )

    def test_quantize_excludes_the_computed_tail(self, tmp_path: Path) -> None:
        """``nodes_to_exclude`` is exactly ``sorted(decode_tail(model))``.

        Not a literal list and not an op-type allowlist: the graph here names
        its decode ops ``dfl_softmax`` and ``cls_sigmoid``, which no
        implementation could have known, and both Convs must stay in.
        """
        import onnx

        from yowo.export import _int8
        from yowo.types import ModelFamily, ModelSize, ModelSpec

        cal_dir = tmp_path / "cal"
        _create_test_images(cal_dir, count=15)
        onnx_path = _detection_onnx(tmp_path / "model.onnx")
        output_path = tmp_path / "model_int8.onnx"

        mock = _mock_ort_quant()
        with (
            _patch_ort(mock),
            patch.object(_int8, "measure_int8_parity", return_value=_stub_report(output_path)),
        ):
            _int8.quantize_onnx_static(
                onnx_path,
                output_path,
                str(cal_dir),
                model_spec=ModelSpec(ModelFamily.YOLO11, ModelSize.NANO),
                input_size=320,
                batch_size=4,
            )

        excluded = mock.quantize_static.call_args.kwargs["nodes_to_exclude"]
        assert excluded == sorted(_int8.decode_tail(onnx.load(str(onnx_path))))
        assert excluded == ["cls_sigmoid", "dfl_softmax"]
        assert "head" not in excluded and "backbone" not in excluded

    def test_calls_quantize_static_with_correct_args(self, tmp_path: Path) -> None:
        from yowo.export import _int8
        from yowo.types import ModelFamily, ModelSize, ModelSpec

        cal_dir = tmp_path / "cal"
        _create_test_images(cal_dir, count=15)
        onnx_path = _detection_onnx(tmp_path / "model.onnx")
        output_path = tmp_path / "model_int8.onnx"

        mock = _mock_ort_quant()
        with (
            _patch_ort(mock),
            patch.object(_int8, "measure_int8_parity", return_value=_stub_report(output_path)),
        ):
            report = _int8.quantize_onnx_static(
                onnx_path,
                output_path,
                str(cal_dir),
                model_spec=ModelSpec(ModelFamily.YOLO11, ModelSize.NANO),
                input_size=320,
                batch_size=4,
            )

        mock.quantize_static.assert_called_once()
        call_args = mock.quantize_static.call_args
        assert call_args[0][0] == str(onnx_path)
        assert call_args[0][1] != str(output_path), (
            "quantize into a partial, promote after the gate"
        )
        assert report.output_path == str(output_path)

    def test_the_gate_runs_one_image_per_session_call(self, tmp_path: Path) -> None:
        """Asserted on BEHAVIOUR: batch of 1, pinned provider, sessions in sequence.

        ``dynamic_batch=False`` fixes the graph batch dim at 1, and a gate that
        inherited the calibration batch size (8) would raise
        ``INVALID_ARGUMENT`` on exactly the configuration it most needs to run
        on. The provider is pinned because a number whose execution provider is
        unstated does not reproduce, and the FP32 session is released before the
        INT8 one opens so two full graphs are never resident at once.

        A signature check and a docstring grep would both pass an
        implementation that batched internally. This does not.
        """
        import numpy as np

        from yowo.export._int8 import measure_int8_parity
        from yowo.types import ModelFamily, ModelSize, ModelSpec

        img_dir = tmp_path / "imgs"
        paths = _create_test_images(img_dir, count=3)

        live: list[str] = []
        order: list[str] = []
        providers_seen: list[list[str]] = []
        batch_sizes: list[int] = []

        class _FakeSession:
            def __init__(self, path: str, providers: list[str]) -> None:
                self.path = path
                providers_seen.append(providers)
                order.append(f"open:{Path(path).name}")
                live.append(path)
                assert len(live) == 1, "two ORT sessions were resident at once"

            def run(self, _out: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
                batch_sizes.append(feeds["images"].shape[0])
                return [np.zeros((feeds["images"].shape[0], 84, 8400), dtype=np.float32)]

            def __del__(self) -> None:
                if self.path in live:
                    live.remove(self.path)
                    order.append(f"close:{Path(self.path).name}")

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = _FakeSession  # type: ignore[attr-defined]

        from yowo.errors import ExportError

        # All-zero output means no detections, so the measurement ends in the
        # R:VACUOUS refusal — after both sessions have already run.
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_ort}),
            pytest.raises(ExportError, match="no gated"),
        ):
            measure_int8_parity(
                tmp_path / "fp32.onnx",
                tmp_path / "int8.onnx",
                paths,
                model_spec=ModelSpec(ModelFamily.YOLO11, ModelSize.NANO),
                input_size=64,
            )

        assert batch_sizes == [1, 1, 1, 1, 1, 1], "3 images x 2 artifacts, one at a time"
        assert providers_seen == [["CPUExecutionProvider"]] * 2
        assert order.index("close:fp32.onnx") < order.index("open:int8.onnx")

    def test_wraps_quantization_failure_in_export_error(self, tmp_path: Path) -> None:
        from yowo.errors import ExportError
        from yowo.export._int8 import quantize_onnx_static
        from yowo.types import ModelFamily, ModelSize, ModelSpec

        cal_dir = tmp_path / "cal"
        _create_test_images(cal_dir, count=15)
        onnx_path = _detection_onnx(tmp_path / "model.onnx")
        output_path = tmp_path / "model_int8.onnx"

        with (
            _patch_ort(_mock_ort_quant(crash=True)),
            pytest.raises(ExportError, match="ONNX INT8 quantization failed"),
        ):
            quantize_onnx_static(
                onnx_path,
                output_path,
                str(cal_dir),
                model_spec=ModelSpec(ModelFamily.YOLO11, ModelSize.NANO),
            )
        assert not output_path.exists()


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
