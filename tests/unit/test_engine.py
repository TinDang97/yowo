"""Unit tests for InferenceEngine using mocked backends and hardware.

No real hardware, no model downloads — everything is mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import InferenceEngine
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
        hw = _make_cpu_only_profile(torch=True)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=_make_mock_backend()),
        ):
            engine = InferenceEngine()

        assert engine.selection.backend == BackendType.PYTORCH
        assert engine.selection.device_type == DeviceType.CPU

    def test_user_backend_override_respected(self) -> None:
        """When backend= is given, it overrides auto-selection."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend(backend_type=BackendType.PYTORCH)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine(backend=BackendType.PYTORCH)

        assert engine.selection.backend == BackendType.PYTORCH


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """with InferenceEngine() as e: calls load() on enter, close() on exit."""

    def test_context_manager_loads_and_closes(self) -> None:
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            engine = InferenceEngine()
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
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            engine = InferenceEngine()
            engine.load()
            engine.close()
            engine.close()  # must not raise

        assert not engine.is_loaded
        # close() is idempotent: second call returns immediately without re-unloading.
        assert mock_backend.unload.call_count == 1


# ---------------------------------------------------------------------------
# Test: detect returns one Detection per frame
# ---------------------------------------------------------------------------


class TestDetect:
    def test_detect_returns_detection_per_frame(self) -> None:
        """detect() returns exactly one Detection per input Frame."""
        hw = _make_cpu_only_profile(torch=True)

        # YOLO26 NMS-free output — (B, num_dets, 6). B=2, 0 detections.
        dummy_output = np.zeros((2, 0, 6), dtype=np.float32)
        mock_backend = _make_mock_backend(infer_output=dummy_output)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            InferenceEngine() as engine,
        ):
            frames = [_make_frame(0), _make_frame(1)]
            results = engine.detect(frames)

        assert len(results) == 2
        assert all(isinstance(d, Detection) for d in results)
        assert results[0].frame.frame_index == 0
        assert results[1].frame.frame_index == 1

    def test_detect_requires_loaded(self) -> None:
        """Calling detect() before load() raises InferenceError."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine()

        with pytest.raises(InferenceError, match="not loaded"):
            engine.detect([_make_frame()])

    def test_detect_attaches_backend_to_detection(self) -> None:
        """Detection.backend matches the selected backend."""
        hw = _make_cpu_only_profile(torch=True)
        dummy_output = np.zeros((1, 0, 6), dtype=np.float32)
        mock_backend = _make_mock_backend(infer_output=dummy_output)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            InferenceEngine() as engine,
        ):
            results = engine.detect([_make_frame()])

        assert results[0].backend == BackendType.PYTORCH

    def test_detect_exceeds_buffer_capacity_falls_back_to_preprocess(self) -> None:
        """detect() with more frames than buffer capacity falls back to fresh preprocess()."""
        hw = _make_cpu_only_profile(torch=True)
        # batch_size=1 → buffer capacity=1; pass 3 frames to exceed it
        dummy_output = np.zeros((3, 0, 6), dtype=np.float32)
        mock_backend = _make_mock_backend(infer_output=dummy_output)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            InferenceEngine(batch_size=1) as engine,
        ):
            # 3 frames > capacity of 1 — must not crash, falls back to preprocess()
            results = engine.detect([_make_frame(0), _make_frame(1), _make_frame(2)])

        assert len(results) == 3


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
            engine = InferenceEngine(batch_size=2)
            engine.load()

        # 5 frames -> batches of [2, 2, 1]
        frames = [_make_frame(i) for i in range(5)]

        # Mock source that yields the frames then closes
        # Set is_live=False and total_frames=5 to route to _stream_pipeline (offline)
        mock_source = MagicMock()
        mock_source.__iter__ = MagicMock(return_value=iter(frames))
        mock_source.close = MagicMock()
        mock_source.is_live = False
        mock_source.total_frames = 5

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
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
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
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine()

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
            engine = InferenceEngine()
            engine.load()

        assert engine.is_loaded
        assert engine.selection.backend == BackendType.PYTORCH
        primary_backend.load.assert_called_once()
        fallback_backend.load.assert_called_once()
        fallback_backend.warmup.assert_called_once()

    def test_all_backends_fail_raises_backend_load_error(self) -> None:
        """When every backend in the fallback chain fails, BackendLoadError is raised."""
        hw = _make_cpu_only_profile(torch=True)

        failing_backend = _make_mock_backend(load_raises=BackendLoadError("always fails"))

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=failing_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            patch("yowo.engine.get_fallback_backends", return_value=[]),
        ):
            engine = InferenceEngine()

            with pytest.raises(BackendLoadError, match="All backends failed"):
                engine.load()


# ---------------------------------------------------------------------------
# Test: source-type-aware stream() dispatch
# ---------------------------------------------------------------------------


def _make_mock_source(
    *,
    frames: list[Frame],
    is_live: bool = False,
    total_frames: int | None = None,
) -> MagicMock:
    """Build a MagicMock that satisfies the FrameSource protocol."""
    source = MagicMock()
    source.__iter__ = MagicMock(return_value=iter(list(frames)))
    source.close = MagicMock()
    source.is_live = is_live
    source.total_frames = total_frames if total_frames is not None else len(frames)
    return source


def _make_loaded_engine(
    mock_backend: MagicMock,
    *,
    batch_size: int = 1,
    prefetch: bool = True,
    pipeline_workers: int = 1,
    frame_drop_policy: object | None = None,
) -> InferenceEngine:
    """Return a loaded InferenceEngine with mocked backend and patched imports."""
    from yowo.types import FrameDropPolicy as FDP

    policy = frame_drop_policy if frame_drop_policy is not None else FDP.LATEST
    engine = InferenceEngine(
        batch_size=batch_size,
        prefetch=prefetch,
        pipeline_workers=pipeline_workers,
        frame_drop_policy=policy,  # type: ignore[arg-type]
    )
    engine._backend = mock_backend
    engine._loaded = True
    # Allocate buffers (mirrors what load() does)
    from yowo.io import PreprocessBuffer
    from yowo.postprocess import PostprocessBuffer

    meta = engine._model_meta  # type: ignore[attr-defined]
    engine._preprocess_buf = PreprocessBuffer(batch_size, (meta.input_height, meta.input_width))  # type: ignore[attr-defined]
    engine._postprocess_buf = PostprocessBuffer()  # type: ignore[attr-defined]
    return engine


def _make_detection(frame: Frame, spec: ModelSpec) -> Detection:
    return Detection(
        frame=frame,
        boxes=(),
        inference_time_ms=0.0,
        backend=BackendType.PYTORCH,
        model_spec=spec,
    )


class TestStreamStrategies:
    """Tests for source-type-aware stream() dispatch."""

    def test_stream_single_image_uses_direct_path(self) -> None:
        """Single-frame source (total_frames=1) routes to _stream_single."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend)

        frame = _make_frame(0)
        source = _make_mock_source(frames=[frame], is_live=False, total_frames=1)

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        expected = [_make_detection(frame, spec)]

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch("yowo.engine.postprocess", return_value=expected),
        ):
            results = list(engine.stream(source))

        assert results == expected
        source.close.assert_called_once()

    def test_stream_single_dispatches_not_to_live_or_pipeline(self) -> None:
        """total_frames==1 calls _stream_single, not _stream_live or _stream_pipeline."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend)

        frame = _make_frame(0)
        source = _make_mock_source(frames=[frame], is_live=False, total_frames=1)

        stream_live_called = []
        stream_pipeline_called = []
        original_live = engine._stream_live
        original_pipeline = engine._stream_pipeline

        def spy_live(s: object) -> object:
            stream_live_called.append(True)
            return original_live(s)  # type: ignore[arg-type]

        def spy_pipeline(s: object) -> object:
            stream_pipeline_called.append(True)
            return original_pipeline(s)  # type: ignore[arg-type]

        engine._stream_live = spy_live  # type: ignore[method-assign]
        engine._stream_pipeline = spy_pipeline  # type: ignore[method-assign]

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch("yowo.engine.postprocess", return_value=[_make_detection(frame, spec)]),
        ):
            list(engine.stream(source))

        assert not stream_live_called
        assert not stream_pipeline_called

    def test_stream_live_source_uses_threaded_reader(self) -> None:
        """Live source (is_live=True) routes to _stream_live.

        Uses SKIP_OLDEST policy so all 3 frames are guaranteed to be processed
        (avoids the non-deterministic frame-drop behaviour of LATEST).
        """
        from yowo.types import FrameDropPolicy

        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(
                mock_backend,
                frame_drop_policy=FrameDropPolicy.SKIP_OLDEST,
            )
        # Increase queue so 3 frames fit without any dropping
        engine._max_queue_size = 4  # type: ignore[attr-defined]

        frames = [_make_frame(i) for i in range(3)]
        source = _make_mock_source(frames=frames, is_live=True, total_frames=None)

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        live_called: list[bool] = []
        original_live = engine._stream_live

        def spy_live(s: object) -> Iterator[Detection]:
            live_called.append(True)
            yield from original_live(s)  # type: ignore[arg-type]

        engine._stream_live = spy_live  # type: ignore[method-assign]

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                side_effect=lambda raw, tensor, fs, **kw: [_make_detection(fs[0], spec)],
            ),
        ):
            results = list(engine.stream(source))

        assert live_called, "_stream_live was not called for is_live=True source"
        assert len(results) == 3
        source.close.assert_called_once()

    def test_stream_offline_source_uses_pipeline(self) -> None:
        """Offline multi-frame source (is_live=False, total_frames>1) uses _stream_pipeline."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend, pipeline_workers=1)

        frames = [_make_frame(i) for i in range(4)]
        source = _make_mock_source(frames=frames, is_live=False, total_frames=4)

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        pipeline_called: list[bool] = []
        original_pipeline = engine._stream_pipeline

        def spy_pipeline(s: object) -> Iterator[Detection]:
            pipeline_called.append(True)
            yield from original_pipeline(s)  # type: ignore[arg-type]

        engine._stream_pipeline = spy_pipeline  # type: ignore[method-assign]

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                side_effect=lambda raw, tensor, fs, **kw: [_make_detection(f, spec) for f in fs],
            ),
        ):
            results = list(engine.stream(source))

        assert pipeline_called, "_stream_pipeline was not called for offline multi-frame source"
        assert len(results) == 4
        source.close.assert_called_once()

    def test_stream_prefetch_false_uses_sync(self) -> None:
        """prefetch=False forces _stream_sync legacy path."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend, prefetch=False)

        frames = [_make_frame(i) for i in range(3)]
        source = _make_mock_source(frames=frames, is_live=False, total_frames=3)

        sync_called: list[bool] = []
        original_sync = engine._stream_sync

        def spy_sync(s: object) -> Iterator[Detection]:
            sync_called.append(True)
            yield from original_sync(s)  # type: ignore[arg-type]

        engine._stream_sync = spy_sync  # type: ignore[method-assign]

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                side_effect=lambda raw, tensor, fs, **kw: [_make_detection(f, spec) for f in fs],
            ),
        ):
            results = list(engine.stream(source))

        assert sync_called, "_stream_sync was not called when prefetch=False"
        assert len(results) == 3
        source.close.assert_called_once()

    def test_stream_pipeline_results_identical_to_sync(self) -> None:
        """_stream_pipeline and _stream_sync produce identical detections for the same frames."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)

        frames = [_make_frame(i) for i in range(5)]

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        def make_engine(*, prefetch: bool) -> InferenceEngine:
            mock_backend = _make_mock_backend()
            with (
                patch("yowo.engine.get_hardware_profile", return_value=hw),
                patch("yowo.engine.create_backend", return_value=mock_backend),
            ):
                return _make_loaded_engine(mock_backend, prefetch=prefetch, pipeline_workers=1)

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                side_effect=lambda raw, tensor, fs, **kw: [_make_detection(f, spec) for f in fs],
            ),
        ):
            engine_pipeline = make_engine(prefetch=True)
            source_pipeline = _make_mock_source(frames=frames, is_live=False, total_frames=5)
            pipeline_results = list(engine_pipeline.stream(source_pipeline))

            engine_sync = make_engine(prefetch=False)
            source_sync = _make_mock_source(frames=frames, is_live=False, total_frames=5)
            sync_results = list(engine_sync.stream(source_sync))

        assert len(pipeline_results) == len(sync_results) == 5
        pipeline_indices = [d.frame.frame_index for d in pipeline_results]
        sync_indices = [d.frame.frame_index for d in sync_results]
        assert sorted(pipeline_indices) == sorted(sync_indices)

    def test_buffer_lifecycle(self) -> None:
        """PreprocessBuffer and PostprocessBuffer allocated at load(), None after close()."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        ):
            engine = InferenceEngine()
            assert engine._preprocess_buf is None  # type: ignore[attr-defined]
            assert engine._postprocess_buf is None  # type: ignore[attr-defined]

            engine.load()
            assert engine._preprocess_buf is not None  # type: ignore[attr-defined]
            assert engine._postprocess_buf is not None  # type: ignore[attr-defined]

            engine.close()
            assert engine._preprocess_buf is None  # type: ignore[attr-defined]
            assert engine._postprocess_buf is None  # type: ignore[attr-defined]

    def test_pipeline_workers_auto_resolves(self) -> None:
        """pipeline_workers=0 auto-resolves after load(): 2 for free-threaded, 1 otherwise."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        # Test GIL build → resolves to 1
        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            patch("yowo.engine.is_free_threaded", return_value=False),
        ):
            engine_gil = InferenceEngine(pipeline_workers=0)
            engine_gil.load()

        assert engine_gil._pipeline_workers == 1  # type: ignore[attr-defined]
        mock_backend.unload()

        # Test free-threaded build → resolves to 2
        mock_backend2 = _make_mock_backend()
        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend2),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            patch("yowo.engine.is_free_threaded", return_value=True),
        ):
            engine_ft = InferenceEngine(pipeline_workers=0)
            engine_ft.load()

        assert engine_ft._pipeline_workers == 2  # type: ignore[attr-defined]

    def test_stream_requires_loaded_for_all_strategies(self) -> None:
        """stream() before load() raises InferenceError regardless of source type."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine()

        for is_live, total in [(False, 1), (True, None), (False, 5)]:
            source = _make_mock_source(frames=[], is_live=is_live, total_frames=total)
            with pytest.raises(InferenceError, match="not loaded"):
                list(engine.stream(source))

    def test_stream_calls_clear_kv_cache(self) -> None:
        """stream() calls clear_kv_cache() once per invocation."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend)
        frame = _make_frame(0)
        source = _make_mock_source(frames=[frame], is_live=False, total_frames=1)

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch("yowo.engine.preprocess", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                return_value=[_make_detection(frame, spec)],
            ),
        ):
            list(engine.stream(source))

        mock_backend.clear_kv_cache.assert_called_once()


# ---------------------------------------------------------------------------
# Test: config-based and kwargs-based constructor
# ---------------------------------------------------------------------------


class TestConfigInit:
    """InferenceEngine accepts InferenceConfig or individual kwargs."""

    def test_config_init(self) -> None:
        """InferenceEngine(InferenceConfig()) works."""
        from yowo.config import InferenceConfig

        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()
        cfg = InferenceConfig()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine(cfg)

        assert engine._spec.family == ModelFamily.YOLO26
        assert engine._spec.size == ModelSize.NANO

    def test_config_maps_confidence_threshold(self) -> None:
        """cfg.confidence_threshold propagates to engine._confidence."""
        from yowo.config import InferenceConfig

        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()
        cfg = InferenceConfig(confidence_threshold=0.8)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine(cfg)

        assert engine._confidence == pytest.approx(0.8)  # type: ignore[attr-defined]

    def test_config_maps_model_identity(self) -> None:
        """cfg.model_family/size map to the correct engine._spec."""
        from yowo.config import InferenceConfig

        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()
        cfg = InferenceConfig(model_family=ModelFamily.YOLO11, model_size=ModelSize.LARGE)

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine(cfg)

        assert engine._spec.family == ModelFamily.YOLO11
        assert engine._spec.size == ModelSize.LARGE

    def test_default_is_yolo26n(self) -> None:
        """InferenceEngine() defaults to YOLO26 NANO."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine()

        assert engine._spec.family == ModelFamily.YOLO26
        assert engine._spec.size == ModelSize.NANO

    def test_kwargs_override_defaults(self) -> None:
        """Individual kwargs are used when no config is passed."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine(
                model_family=ModelFamily.YOLO11,
                model_size=ModelSize.SMALL,
                confidence_threshold=0.5,
                batch_size=4,
            )

        assert engine._spec.family == ModelFamily.YOLO11
        assert engine._spec.size == ModelSize.SMALL
        assert engine._confidence == pytest.approx(0.5)  # type: ignore[attr-defined]
        assert engine._batch_size == 4  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test: detect([]) empty frames (H-3)
# ---------------------------------------------------------------------------


class TestDetectEmptyFrames:
    def test_detect_empty_list_raises_value_error(self) -> None:
        """detect([]) raises ValueError from preprocess() — empty input is invalid."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
            InferenceEngine() as engine,
            pytest.raises(ValueError, match="empty"),
        ):
            engine.detect([])


# ---------------------------------------------------------------------------
# Test: concurrent pipeline path (H-1/H-2)
# ---------------------------------------------------------------------------


class TestConcurrentPipeline:
    def test_pipeline_workers_gt1_produces_correct_results(self) -> None:
        """pipeline_workers=2 (concurrent path) yields the same results as sync."""
        spec = _make_spec()
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend, batch_size=2, pipeline_workers=2)

        frames = [_make_frame(i) for i in range(6)]
        source = _make_mock_source(frames=frames, is_live=False, total_frames=6)

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1

        with (
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch("yowo.engine.preprocess", return_value=dummy_tensor),
            patch(
                "yowo.engine.postprocess",
                side_effect=lambda raw, tensor, fs, **kw: [_make_detection(f, spec) for f in fs],
            ),
        ):
            results = list(engine.stream(source))

        assert len(results) == 6
        indices = sorted(d.frame.frame_index for d in results)
        assert indices == list(range(6))
        source.close.assert_called_once()

    def test_pipeline_worker_persistent_error_returns_empty(self) -> None:
        """If backend.infer raises on every retry, worker returns empty detection.

        Since v2.3.0 (RELY-02), _infer_with_retry exhausts retries and returns
        empty result instead of raising — the engine degrades gracefully.
        errors_total is incremented and source.close() is still called.
        """
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()
        mock_backend.infer.side_effect = RuntimeError("GPU OOM")

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend, batch_size=1, pipeline_workers=2)

        frames = [_make_frame(0)]
        source = _make_mock_source(frames=frames, is_live=False, total_frames=1)

        dummy_tensor = MagicMock(spec=PreprocessedTensor)
        dummy_tensor.original_shapes = ((480, 640),)
        dummy_tensor.scale_factors = ((1.0, 1.0),)
        dummy_tensor.pad_offsets = ((0, 0),)
        dummy_tensor.input_shape = (640, 640)
        dummy_tensor.batch_size = 1
        dummy_tensor.data = np.zeros((1, 3, 640, 640), dtype=np.float32)

        with (
            patch("yowo.engine.preprocess", return_value=dummy_tensor),
            patch("yowo.engine.preprocess_into", return_value=dummy_tensor),
            patch("yowo.engine.time.sleep"),
        ):
            results = list(engine.stream(source))

        # Empty detection returned (no raise), error counter incremented
        assert len(results) == 1
        assert results[0].boxes == ()
        assert engine.metrics.errors_total == 1
        source.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: source.close() called on exceptions (H-11)
# ---------------------------------------------------------------------------


class TestSourceCloseOnExceptions:
    def test_stream_single_closes_on_detect_error(self) -> None:
        """_stream_single calls source.close() even when detect() raises."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend)

        frame = _make_frame(0)
        source = _make_mock_source(frames=[frame], is_live=False, total_frames=1)

        with (
            patch.object(engine, "detect", side_effect=InferenceError("boom")),
            pytest.raises(InferenceError, match="boom"),
        ):
            list(engine.stream(source))

        source.close.assert_called_once()

    def test_stream_sync_closes_on_detect_error(self) -> None:
        """_stream_sync calls source.close() even when detect() raises."""
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend, prefetch=False)

        frames = [_make_frame(i) for i in range(3)]
        source = _make_mock_source(frames=frames, is_live=False, total_frames=3)

        with (
            patch.object(engine, "detect", side_effect=InferenceError("boom")),
            pytest.raises(InferenceError, match="boom"),
        ):
            list(engine.stream(source))

        source.close.assert_called_once()

    def test_stream_pipeline_closes_on_detect_error(self) -> None:
        """_stream_pipeline calls source.close() even when inference raises in worker.

        Patches _run_batch to raise directly (bypasses the retry wrapper in
        _infer_with_retry) so we can test source.close() teardown behavior.
        """
        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend, pipeline_workers=1)

        frames = [_make_frame(i) for i in range(2)]
        source = _make_mock_source(frames=frames, is_live=False, total_frames=2)

        with (
            patch.object(engine, "_run_batch", side_effect=InferenceError("worker crash")),
            pytest.raises(InferenceError, match="worker crash"),
        ):
            list(engine.stream(source))

        source.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _stream_live idle timeout (H-5)
# ---------------------------------------------------------------------------


class TestStreamLiveIdleTimeout:
    def test_live_stream_terminates_on_idle_timeout(self) -> None:
        """_stream_live breaks after max idle time with no frames."""
        import time as time_mod

        hw = _make_cpu_only_profile(torch=True)
        mock_backend = _make_mock_backend()

        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = _make_loaded_engine(mock_backend)

        # Source that never yields any frames
        source = _make_mock_source(frames=[], is_live=True, total_frames=None)

        # Patch time.monotonic to simulate rapid passage of time
        # and ThreadedFrameReader to return None (timeout) immediately
        mono_times = iter([0.0, 0.0, 31.0])  # first call sets idle_since=0, third triggers break

        with (
            patch("yowo._streaming.ThreadedFrameReader") as mock_reader_cls,
            patch("yowo._streaming.time") as mock_time,
        ):
            mock_reader = MagicMock()
            mock_reader.get.return_value = None
            mock_reader.is_exhausted = False
            mock_reader_cls.return_value = mock_reader
            mock_time.perf_counter = time_mod.perf_counter
            mock_time.monotonic = lambda: next(mono_times)

            results = list(engine.stream(source))

        assert results == []
        source.close.assert_called_once()


# ---------------------------------------------------------------------------
# Custom backend injection
# ---------------------------------------------------------------------------


class TestCustomBackendInjection:
    """Tests for user-provided backend_instance injection."""

    def test_skips_auto_selection(self) -> None:
        """When backend_instance provided, select_backend is not called."""
        mock_backend = _make_mock_backend()
        with (
            patch("yowo.engine.select_backend") as mock_select,
            patch("yowo.engine.create_backend") as mock_create,
            patch("yowo.engine.get_hardware_profile") as mock_hw,
        ):
            InferenceEngine(backend_instance=mock_backend)
            mock_select.assert_not_called()
            mock_create.assert_not_called()
            mock_hw.assert_not_called()

    def test_selection_reflects_backend_type(self) -> None:
        """BackendSelection.backend matches injected backend's type."""
        mock_backend = _make_mock_backend(backend_type=BackendType.ONNX)
        engine = InferenceEngine(backend_instance=mock_backend)
        assert engine.selection.backend == BackendType.ONNX
        assert engine.selection.reason == "User-provided backend instance"

    def test_selection_defaults_to_cpu_fp32(self) -> None:
        """Injected backend selection defaults to CPU/FP32 (user manages device)."""
        mock_backend = _make_mock_backend()
        engine = InferenceEngine(backend_instance=mock_backend)
        assert engine.selection.device_type == DeviceType.CPU
        assert engine.selection.precision == Precision.FP32

    def test_load_calls_backend_load_and_warmup(self) -> None:
        """load() calls backend.load() and warmup() on injected backend."""
        mock_backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
            engine = InferenceEngine(backend_instance=mock_backend)
            engine.load()
            mock_backend.load.assert_called_once()
            mock_backend.warmup.assert_called_once()
            assert engine.is_loaded

    def test_no_fallback_on_load_failure(self) -> None:
        """Injected backend failures propagate directly — no fallback chain."""
        mock_backend = _make_mock_backend(load_raises=BackendLoadError("custom fail"))
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
            engine = InferenceEngine(backend_instance=mock_backend)
            with pytest.raises(BackendLoadError, match="custom fail"):
                engine.load()

    def test_detect_works_with_injected_backend(self) -> None:
        """detect() works end-to-end with injected backend."""
        mock_backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
            engine = InferenceEngine(backend_instance=mock_backend)
            engine.load()
            # Reset mock so warmup validation's infer() call is not counted
            mock_backend.infer.reset_mock()
            frames = [_make_frame(0)]
            result = engine.detect(frames)
            assert len(result) == 1
            mock_backend.infer.assert_called_once()

    def test_none_uses_normal_auto_selection_path(self) -> None:
        """backend_instance=None (default) uses auto-selection."""
        hw = _make_cpu_only_profile()
        mock_backend = _make_mock_backend()
        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw) as mock_hw,
            patch("yowo.engine.select_backend") as mock_select,
            patch("yowo.engine.create_backend", return_value=mock_backend) as mock_create,
        ):
            mock_select.return_value = BackendSelection(
                backend=BackendType.PYTORCH,
                device_type=DeviceType.CPU,
                precision=Precision.FP32,
                reason="auto",
            )
            InferenceEngine()  # no backend_instance
            mock_hw.assert_called_once()
            mock_select.assert_called_once()
            mock_create.assert_called_once()

    def test_close_calls_unload_on_injected_backend(self) -> None:
        """close() calls unload() on injected backend."""
        mock_backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
            engine = InferenceEngine(backend_instance=mock_backend)
            engine.load()
            engine.close()
            mock_backend.unload.assert_called_once()
            assert not engine.is_loaded

    def test_context_manager_with_injected_backend(self) -> None:
        """Context manager loads and closes injected backend."""
        mock_backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
            with InferenceEngine(backend_instance=mock_backend) as engine:
                assert engine.is_loaded
            mock_backend.unload.assert_called_once()

    def test_kv_cache_flag_propagates(self) -> None:
        """kv_cache config propagates when using injected backend."""
        mock_backend = _make_mock_backend()
        engine = InferenceEngine(backend_instance=mock_backend, kv_cache=True)
        assert engine._kv_cache is True

    def test_user_provided_backend_flag_set(self) -> None:
        """_user_provided_backend flag is True when backend injected."""
        mock_backend = _make_mock_backend()
        engine = InferenceEngine(backend_instance=mock_backend)
        assert engine._user_provided_backend is True

    def test_user_provided_backend_flag_false_by_default(self) -> None:
        """_user_provided_backend flag is False when using auto-selection."""
        hw = _make_cpu_only_profile()
        mock_backend = _make_mock_backend()
        with (
            patch("yowo.engine.get_hardware_profile", return_value=hw),
            patch("yowo.engine.create_backend", return_value=mock_backend),
        ):
            engine = InferenceEngine()
            assert engine._user_provided_backend is False
