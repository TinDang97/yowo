"""Unit tests for streaming pipeline optimisations.

Covers:
- ``BaseEngine._run_gpu()`` extraction
- ``_infer_from_tensor`` delegation to ``_run_gpu``
- Lock-scope splitting in ``_stream_pipeline._infer_batch``
"""

from __future__ import annotations

import threading
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
        # Reset mock so warmup validation's infer() call is not counted
        backend.infer.reset_mock()

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


class _ThreadingShim:
    """`threading` as `_streaming` sees it, with `Lock()` handing back ours.

    Everything else delegates to the real module, so `Event` and anything added
    later keep working. Scoped to one module's namespace on purpose — see the
    comment at the patch site.
    """

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def Lock(self) -> threading.Lock:  # mirrors the stdlib name, deliberately
        return self._lock

    def __getattr__(self, name: str) -> object:
        return getattr(threading, name)


class TestInferBatchLockScope:
    """The real `_stream_pipeline` closure must not hold the inference lock
    across postprocessing.

    Replaces a stopwatch. The previous check built its OWN `_infer_batch` inside
    the test, chose the lock scope itself, ran two workers with 5ms and 50ms
    sleeps and asserted the wall clock came in under 90ms. Two problems, and the
    second is the serious one:

    * It flaked. A wall-clock budget on a loaded machine fails for reasons that
      have nothing to do with the property, and since 2026-09-10 this file sits
      inside a REQUIRED status check, so a flake blocks a merge.
    * It could not fail for the right reason either. The production
      `_infer_batch` is a closure defined inside `_stream_pipeline`
      (`_streaming.py:153`) and is unreachable from a test, so the old check
      timed a copy — and the copy called `_process_batch` where production calls
      `_postprocess_and_emit`. Moving the real postprocess inside the real lock
      would not have failed it.

    This drives the real `_stream_pipeline` and records, at the moment
    `_postprocess_and_emit` is entered, whether the inference lock is held. No
    sleeps, no timing, and it reads the production closure rather than a
    restatement of it.
    """

    class _Source:
        """The minimum `_stream_pipeline` asks of a source: it closes it."""

        def close(self) -> None: ...

    @staticmethod
    def _drive(engine: DetectionEngine, frame_count: int) -> list[bool]:
        """Run the real pipeline; return lock-held state at each postprocess."""
        held: list[bool] = []
        lock = threading.Lock()

        frames = [_make_frame(i) for i in range(frame_count)]

        class _Reader:
            """The queue `_stream_pipeline` drains: every frame, then None."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                self._items: list[Frame | None] = [*frames, None]

            def start(self) -> None: ...

            def stop(self) -> None: ...

            def get(self, timeout: float = 0.0) -> Frame | None:
                return self._items.pop(0) if self._items else None

            @property
            def is_exhausted(self) -> bool:
                return not self._items

        real_postprocess = engine._postprocess_and_emit

        def _recording(*args: object, **kwargs: object) -> list[object]:
            held.append(lock.locked())
            return real_postprocess(*args, **kwargs)

        with (
            # A shim MODULE, not `patch("...threading.Lock")`. `_streaming` does
            # `import threading`, so its `threading` attribute IS the stdlib
            # module — patching an attribute on it replaces `threading.Lock`
            # process-wide, every lock in the interpreter becomes the same
            # object, and the run deadlocks. Replacing the module reference in
            # this one namespace touches nothing else.
            patch("yowo._streaming.threading", _ThreadingShim(lock)),
            patch("yowo._streaming.ThreadedFrameReader", _Reader),
            patch.object(engine, "_postprocess_and_emit", _recording),
        ):
            list(engine._stream_pipeline(TestInferBatchLockScope._Source()))  # type: ignore[arg-type]
        return held

    def test_postprocess_runs_outside_lock(self) -> None:
        """Concurrent path: the lock is never held when postprocess is entered."""
        engine = _make_engine_and_load(mock_backend=_make_mock_backend())
        engine._pipeline_workers = 2
        engine._batch_size = 1

        held = self._drive(engine, frame_count=4)

        assert held, "postprocess was never reached — the fixture drove nothing"
        assert not any(held), (
            f"the inference lock was held on {sum(held)} of {len(held)} postprocess "
            "calls; _infer_batch must release it before _postprocess_and_emit"
        )

    def test_the_lock_is_actually_held_during_inference(self) -> None:
        """The companion the old check lacked.

        Without this, a build that never took the lock at all would satisfy the
        test above perfectly — an assertion that something does not happen is
        worth nothing unless something else proves the mechanism exists.
        """
        engine = _make_engine_and_load(mock_backend=_make_mock_backend())
        engine._pipeline_workers = 2
        engine._batch_size = 1

        seen: list[bool] = []
        lock = threading.Lock()
        real_run_gpu = engine._run_gpu

        def _recording(*args: object, **kwargs: object) -> object:
            seen.append(lock.locked())
            return real_run_gpu(*args, **kwargs)

        frames = [_make_frame(0)]

        class _Reader:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._items: list[Frame | None] = [*frames, None]

            def start(self) -> None: ...

            def stop(self) -> None: ...

            def get(self, timeout: float = 0.0) -> Frame | None:
                return self._items.pop(0) if self._items else None

            @property
            def is_exhausted(self) -> bool:
                return not self._items

        with (
            patch("yowo._streaming.threading", _ThreadingShim(lock)),
            patch("yowo._streaming.ThreadedFrameReader", _Reader),
            patch.object(engine, "_run_gpu", _recording),
        ):
            list(engine._stream_pipeline(self._Source()))  # type: ignore[arg-type]

        assert seen, "_run_gpu was never reached — the fixture drove nothing"
        assert all(seen), "the inference lock was NOT held during _run_gpu"
