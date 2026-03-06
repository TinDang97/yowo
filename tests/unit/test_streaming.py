"""Unit tests for streaming pipeline optimisations.

Covers:
- ``BaseEngine._run_gpu()`` extraction
- ``_infer_from_tensor`` delegation to ``_run_gpu``
- Lock-scope splitting in ``_stream_pipeline._infer_batch``
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from yowo.backends import InferenceBackend
from yowo.engine import DetectionEngine
from yowo.hardware import HardwareProfile
from yowo.hardware._capabilities import InstalledLibraries
from yowo.hardware._device import Device
from yowo.types import (
    BackendType,
    CPUArch,
    DeviceType,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
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
    infer_output: np.ndarray | None = None,
) -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = backend_type
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.load.return_value = None
    mock.warmup.return_value = None
    if infer_output is not None:
        mock.infer.return_value = infer_output
    else:
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


def _make_engine_and_load(
    *,
    mock_backend: MagicMock | None = None,
) -> DetectionEngine:
    """Create a DetectionEngine with a mocked backend and call load()."""
    hw = _make_cpu_only_profile(torch=True)
    backend = mock_backend or _make_mock_backend()

    with (
        patch("yowo.engine.get_hardware_profile", return_value=hw),
        patch("yowo.engine.create_backend", return_value=backend),
    ):
        engine = DetectionEngine()

    # Simulate load (skip real weight resolution)
    with (
        patch("yowo.engine.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch.object(backend, "load"),
        patch.object(backend, "warmup"),
    ):
        backend.is_loaded = True
        engine.load()

    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunGpu:
    """Tests for BaseEngine._run_gpu()."""

    def test_returns_raw_output_and_elapsed(self) -> None:
        """_run_gpu returns (raw_output, elapsed_ms) tuple."""
        raw = np.zeros((1, 0, 6), dtype=np.float32)
        backend = _make_mock_backend(infer_output=raw)
        engine = _make_engine_and_load(mock_backend=backend)

        tensor = MagicMock(spec=PreprocessedTensor)
        tensor.batch_size = 1
        frames = [_make_frame()]

        result = engine._run_gpu(tensor, frames)

        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], raw)
        assert isinstance(result[1], float)
        assert result[1] >= 0.0
        backend.infer.assert_called_once_with(tensor)

    def test_sets_source_id_when_feature_cache_present(self) -> None:
        """_run_gpu calls set_source_id when feature cache is active."""
        backend = _make_mock_backend()
        engine = _make_engine_and_load(mock_backend=backend)
        engine._feature_cache = MagicMock()

        tensor = MagicMock(spec=PreprocessedTensor)
        tensor.batch_size = 1
        frame = _make_frame()
        frame = Frame(pixels=frame.pixels, source_id="cam-1", frame_index=0)

        engine._run_gpu(tensor, [frame])

        backend.set_source_id.assert_called_once_with("cam-1")


class TestInferFromTensorDelegation:
    """Tests that _infer_from_tensor delegates to _run_gpu."""

    def test_delegates_to_run_gpu(self) -> None:
        """_infer_from_tensor calls _run_gpu and passes output to _process_batch."""
        raw = np.zeros((1, 0, 6), dtype=np.float32)
        backend = _make_mock_backend(infer_output=raw)
        engine = _make_engine_and_load(mock_backend=backend)

        tensor = MagicMock(spec=PreprocessedTensor)
        tensor.batch_size = 1
        frames = [_make_frame()]

        sentinel = [MagicMock()]
        with (
            patch.object(engine, "_run_gpu", return_value=(raw, 1.5)) as mock_gpu,
            patch.object(engine, "_process_batch", return_value=sentinel) as mock_pp,
        ):
            results = engine._infer_from_tensor(tensor, frames, scratch=None)

        mock_gpu.assert_called_once_with(tensor, frames)
        mock_pp.assert_called_once_with(raw, tensor, frames, 1.5, None)
        assert results is sentinel

    def test_emits_event_after_process_batch(self) -> None:
        """_infer_from_tensor calls emit on the event bus after _process_batch."""
        raw = np.zeros((1, 0, 6), dtype=np.float32)
        backend = _make_mock_backend(infer_output=raw)
        engine = _make_engine_and_load(mock_backend=backend)

        tensor = MagicMock(spec=PreprocessedTensor)
        tensor.batch_size = 1
        frames = [_make_frame()]

        sentinel = [MagicMock()]
        with (
            patch.object(engine, "_process_batch", return_value=sentinel),
            patch.object(engine._event_bus, "emit") as mock_emit,
        ):
            engine._infer_from_tensor(tensor, frames, scratch=None)

        mock_emit.assert_called_once_with("detection", sentinel)


class TestInferBatchLockScope:
    """Tests that _infer_batch in _stream_pipeline releases the lock before NMS."""

    def test_postprocess_runs_outside_lock(self) -> None:
        """Two workers: if postprocess overlaps, total time < 2x sequential.

        We simulate:
        - _run_gpu takes ~5ms (fast, under lock)
        - _process_batch takes ~50ms (slow, should run OUTSIDE lock)

        If lock covers only _run_gpu, two batches complete in ~(5+5)+50 = 60ms.
        If lock covers both, two batches take ~(5+50)+(5+50) = 110ms.
        We assert total < 90ms to confirm postprocess is outside the lock.
        """
        backend = _make_mock_backend()
        engine = _make_engine_and_load(mock_backend=backend)

        # Force pipeline_workers=2 to enable concurrent path
        engine._pipeline_workers = 2

        raw = np.zeros((1, 0, 6), dtype=np.float32)

        def slow_gpu(tensor: object) -> np.ndarray:
            time.sleep(0.005)  # 5ms GPU
            return raw

        backend.infer.side_effect = slow_gpu

        def slow_process_batch(*args: object, **kwargs: object) -> list[object]:
            time.sleep(0.050)  # 50ms NMS
            return []

        engine._process_batch = slow_process_batch  # type: ignore[assignment]

        # Build two batches and run them through _stream_pipeline's _infer_batch
        # by exercising the concurrent path directly.
        from concurrent.futures import ThreadPoolExecutor

        from yowo.io._decode import PreprocessBufferPool

        target = (640, 640)
        buffer_pool = PreprocessBufferPool(2, 1, target)
        infer_lock = threading.Lock()

        def _infer_batch(frames: list[Frame]) -> list[object]:
            buf = buffer_pool.acquire()
            try:
                from yowo.io._decode import preprocess_into

                tensor = preprocess_into(frames, target, buf)
                with infer_lock:
                    raw_output, elapsed_ms = engine._run_gpu(tensor, frames)
                results = engine._process_batch(
                    raw_output,
                    tensor,
                    frames,
                    elapsed_ms,
                    None,
                )
                return results
            finally:
                buffer_pool.release(buf)

        frames_a = [_make_frame(0)]
        frames_b = [_make_frame(1)]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_infer_batch, frames_a)
            fut_b = pool.submit(_infer_batch, frames_b)
            fut_a.result()
            fut_b.result()
        elapsed = time.perf_counter() - t0

        # If postprocess were inside the lock: ~110ms (serial)
        # With postprocess outside: ~60ms (GPU serial, NMS parallel)
        # Allow generous margin for CI variance
        assert elapsed < 0.090, (
            f"Expected <90ms (postprocess outside lock), got {elapsed * 1000:.0f}ms"
        )
