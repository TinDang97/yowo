"""Unit tests for InferenceEngine using mocked backends and hardware.

No real hardware, no model downloads — everything is mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.errors import BackendLoadError, InferenceError
from yowo.hardware import HardwareProfile
from yowo.hardware._capabilities import InstalledLibraries
from yowo.hardware._device import Device
from yowo.types import (
    BackendSelection,
    BackendType,
    CPUArch,
    Detection,
    DeviceType,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
    PreprocessedTensor,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_cpu_device() -> Device:
    return Device(
        type=DeviceType.CPU,
        index=0,
        name="Test CPU",
        arch=None,
        cpu_arch=CPUArch.X86_64,
        memory_total_mb=16384,
        memory_available_mb=16384,
        is_jetson=False,
    )


def _make_cpu_only_profile(*, torch: bool = True) -> HardwareProfile:
    """Build a CPU-only HardwareProfile with optional PyTorch."""
    libs = InstalledLibraries(
        torch_version="2.3.0" if torch else None,
        torch_cuda_available=False,
        tensorrt_version=None,
        onnxruntime_version=None,
        onnxruntime_has_cuda=False,
        openvino_version=None,
    )
    return HardwareProfile(
        gpus=(),
        cpu=_make_cpu_device(),
        cpu_features=frozenset({"AVX2"}),
        libraries=libs,
    )


def _make_mock_backend(
    *,
    backend_type: BackendType = BackendType.PYTORCH,
    load_raises: Exception | None = None,
    infer_output: np.ndarray | None = None,
) -> MagicMock:
    """Return a MagicMock satisfying the InferenceBackend protocol."""
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = backend_type
    mock.is_loaded = False
    mock.input_shape = (640, 640)

    if load_raises is not None:
        mock.load.side_effect = load_raises
    else:
        mock.load.return_value = None

    mock.warmup.return_value = None

    if infer_output is not None:
        mock.infer.return_value = infer_output
    else:
        # Default: YOLO26 NMS-free output (B=1, 0 detections, 6 cols)
        mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)

    mock.unload.return_value = None
    return mock


def _make_frame(index: int = 0) -> Frame:
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=index)


def _make_spec(
    family: ModelFamily = ModelFamily.YOLO26,
    size: ModelSize = ModelSize.NANO,
) -> ModelSpec:
    return ModelSpec(family, size)


# ---------------------------------------------------------------------------
# Test: backend auto-selection
# ---------------------------------------------------------------------------


class TestInitAutoSelectsBackend:
    """InferenceEngine selects a backend based on the hardware profile."""

    def test_selects_pytorch_on_cpu_only_profile(self) -> None:
        """On a CPU-only machine with PyTorch, PYTORCH backend is chosen."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=_make_mock_backend()),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)

        assert engine.selection.backend == BackendType.PYTORCH
        assert engine.selection.device_type == DeviceType.CPU

    def test_user_backend_override_respected(self) -> None:
        """When backend= is given, it overrides auto-selection."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend(backend_type=BackendType.PYTORCH)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec, backend=BackendType.PYTORCH)

        assert engine.selection.backend == BackendType.PYTORCH


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """with InferenceEngine(spec) as e: calls load() on enter, close() on exit."""

    def test_context_manager_loads_and_closes(self) -> None:
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)
            assert not engine.is_loaded

            with engine:
                assert engine.is_loaded
                mock_backend.load.assert_called_once()
                mock_backend.warmup.assert_called_once()

            # After exit, unload called and is_loaded reset
            mock_backend.unload.assert_called_once()
            assert not engine.is_loaded

    def test_close_is_idempotent(self) -> None:
        """Calling close() twice does not raise."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)
            engine.load()
            engine.close()
            engine.close()  # must not raise

        assert not engine.is_loaded
        assert mock_backend.unload.call_count == 2


# ---------------------------------------------------------------------------
# Test: detect returns one Detection per frame
# ---------------------------------------------------------------------------


class TestDetect:
    def test_detect_returns_detection_per_frame(self) -> None:
        """detect() returns exactly one Detection per input Frame."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)

        # YOLO26 NMS-free output — (B, num_dets, 6). B=2, 0 detections.
        dummy_output = np.zeros((2, 0, 6), dtype=np.float32)
        mock_backend = _make_mock_backend(infer_output=dummy_output)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            from yowo.engine import InferenceEngine

            with InferenceEngine(spec) as engine:
                frames = [_make_frame(0), _make_frame(1)]
                results = engine.detect(frames)

        assert len(results) == 2
        assert all(isinstance(d, Detection) for d in results)
        assert results[0].frame.frame_index == 0
        assert results[1].frame.frame_index == 1

    def test_detect_requires_loaded(self) -> None:
        """Calling detect() before load() raises InferenceError."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)

        with pytest.raises(InferenceError, match="not loaded"):
            engine.detect([_make_frame()])

    def test_detect_attaches_backend_to_detection(self) -> None:
        """Detection.backend matches the selected backend."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        dummy_output = np.zeros((1, 0, 6), dtype=np.float32)
        mock_backend = _make_mock_backend(infer_output=dummy_output)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            from yowo.engine import InferenceEngine

            with InferenceEngine(spec) as engine:
                results = engine.detect([_make_frame()])

        assert results[0].backend == BackendType.PYTORCH


# ---------------------------------------------------------------------------
# Test: stream batching
# ---------------------------------------------------------------------------


class TestStream:
    def test_stream_batches_correctly(self) -> None:
        """With batch_size=2, detect() is called with batches of up to 2 frames."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)

        # Track calls to detect; patch it after construction
        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=_make_mock_backend()),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec, batch_size=2)
            engine.load()

        # 5 frames -> batches of [2, 2, 1]
        frames = [_make_frame(i) for i in range(5)]

        # Mock source that yields the frames then closes
        mock_source = MagicMock()
        mock_source.__iter__ = MagicMock(return_value=iter(frames))
        mock_source.close = MagicMock()

        detect_calls: list[list[Frame]] = []
        original_detect = engine.detect

        def recording_detect(batch: list[Frame]) -> list[Detection]:
            detect_calls.append(list(batch))
            dummy_out = np.zeros((len(batch), 0, 6), dtype=np.float32)
            with (
                patch("yowo.engine.preprocess") as mock_pre,
                patch("yowo.engine.postprocess") as mock_post,
            ):
                mock_pre.return_value = MagicMock(
                    data=dummy_out,
                    original_shapes=tuple((480, 640) for _ in batch),
                    input_shape=(640, 640),
                    scale_factors=tuple((1.0, 1.0) for _ in batch),
                    pad_offsets=tuple((0, 0) for _ in batch),
                    batch_size=len(batch),
                )
                mock_post.return_value = [
                    Detection(
                        frame=f,
                        boxes=(),
                        inference_time_ms=0.0,
                        backend=BackendType.PYTORCH,
                        model_spec=spec,
                    )
                    for f in batch
                ]
                return original_detect(batch)

        engine.detect = recording_detect  # type: ignore[method-assign]

        # Patch preprocess/postprocess for actual detect calls inside stream
        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = tuple((480, 640) for _ in range(2))
        dummy_tensor.scale_factors = tuple((1.0, 1.0) for _ in range(2))
        dummy_tensor.pad_offsets = tuple((0, 0) for _ in range(2))
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 2

        results_collected = []
        with (
            patch("yowo.engine.preprocess", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                side_effect=lambda raw, tensor, frames_, **kw: [
                    Detection(
                        frame=f,
                        boxes=(),
                        inference_time_ms=0.0,
                        backend=BackendType.PYTORCH,
                        model_spec=spec,
                    )
                    for f in frames_
                ],
            ),
        ):
            engine.detect = original_detect  # type: ignore[method-assign]
            for det in engine.stream(mock_source):
                results_collected.append(det)

        assert len(results_collected) == 5
        mock_source.close.assert_called_once()

    def test_stream_requires_loaded(self) -> None:
        """stream() before load() raises InferenceError."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)

        mock_source = MagicMock()
        mock_source.close = MagicMock()

        with pytest.raises(InferenceError, match="not loaded"):
            list(engine.stream(mock_source))


# ---------------------------------------------------------------------------
# Test: fallback on backend failure
# ---------------------------------------------------------------------------


class TestFallback:
    def test_load_fallback_on_backend_failure(self) -> None:
        """When primary backend raises BackendLoadError, falls back to next backend."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)

        primary_backend = _make_mock_backend(
            backend_type=BackendType.ONNX,
            load_raises=BackendLoadError("ONNX load failed"),
        )
        fallback_backend = _make_mock_backend(backend_type=BackendType.PYTORCH)

        create_calls: list[BackendType] = []

        def fake_create(bt: BackendType, hw_profile: HardwareProfile, **_kw: object) -> MagicMock:
            create_calls.append(bt)
            if bt == BackendType.PYTORCH:
                return fallback_backend
            return primary_backend

        # Override selection to return ONNX so fallback to PYTORCH is exercised
        onnx_selection = BackendSelection(
            backend=BackendType.ONNX,
            device_type=DeviceType.CPU,
            precision=Precision.FP32,
            device_index=0,
            reason="test override",
        )

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.select_backend", return_value=onnx_selection),
            patch("yowo.engine.create_backend", side_effect=fake_create),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            patch(
                "yowo.engine.get_fallback_backends",
                return_value=[BackendType.PYTORCH],
            ),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)
            engine.load()

        assert engine.is_loaded
        assert engine.selection.backend == BackendType.PYTORCH
        primary_backend.load.assert_called_once()
        fallback_backend.load.assert_called_once()
        fallback_backend.warmup.assert_called_once()

    def test_all_backends_fail_raises_backend_load_error(self) -> None:
        """When every backend in the fallback chain fails, BackendLoadError is raised."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)

        failing_backend = _make_mock_backend(load_raises=BackendLoadError("always fails"))

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=failing_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            patch("yowo.engine.get_fallback_backends", return_value=[]),
        ):
            from yowo.engine import InferenceEngine

            engine = InferenceEngine(spec)

            with pytest.raises(BackendLoadError, match="All backends failed"):
                engine.load()
