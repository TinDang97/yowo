"""Unit tests for GPU error retry wrapper (_infer_with_retry).

Tests retry behavior, backoff delays, empty result on exhaustion, and
error recording. All tests mock backend.infer and time.sleep to avoid
actual delays and hardware dependencies.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

from yowo.backends import InferenceBackend
from yowo.engine import DetectionEngine
from yowo.types import BackendType, Frame, PreprocessedTensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.load.return_value = None
    mock.warmup.return_value = None
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.unload.return_value = None
    return mock


def _loaded_engine(mock_backend: MagicMock, **kwargs: object) -> DetectionEngine:
    engine = DetectionEngine(backend_instance=mock_backend, **kwargs)
    with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
        engine.load()
    # Reset call count after warmup validation (which calls infer once)
    mock_backend.infer.reset_mock()
    return engine


def _make_tensor() -> PreprocessedTensor:
    return PreprocessedTensor(
        data=np.zeros((1, 3, 640, 640), dtype=np.float32),
        original_shapes=((640, 640),),
        input_shape=(640, 640),
        scale_factors=((1.0, 1.0),),
        pad_offsets=((0, 0),),
    )


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestInferWithRetry:
    def test_success_on_first_attempt_no_retry(self) -> None:
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        result = engine._infer_with_retry(tensor)

        assert mock.infer.call_count == 1
        assert result.shape == (1, 0, 6)

    def test_success_on_second_attempt_no_error_recorded(self) -> None:
        """Success on 2nd attempt: no error recorded, no empty result."""
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        good_output = np.zeros((1, 5, 6), dtype=np.float32)
        mock.infer.side_effect = [RuntimeError("transient"), good_output]

        with patch("yowo.engine.time.sleep") as mock_sleep:
            result = engine._infer_with_retry(tensor)

        assert mock.infer.call_count == 2
        assert result is good_output
        # Slept once with first delay (0.1s)
        mock_sleep.assert_called_once_with(0.1)
        # No error recorded
        assert engine.metrics.errors_total == 0

    def test_retry_exhausted_returns_empty(self) -> None:
        """3 consecutive failures return zeros and record error."""
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        mock.infer.side_effect = [
            RuntimeError("gpu error"),
            RuntimeError("gpu error"),
            RuntimeError("gpu error"),
        ]

        with patch("yowo.engine.time.sleep"):
            result = engine._infer_with_retry(tensor)

        assert mock.infer.call_count == 3
        assert result.shape == (1, 0, 6)
        assert result.dtype == np.float32
        assert np.all(result == 0)

    def test_retry_exhausted_records_error(self) -> None:
        """Error counter is incremented exactly once on exhaustion."""
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        mock.infer.side_effect = RuntimeError("gpu error")

        with patch("yowo.engine.time.sleep"):
            engine._infer_with_retry(tensor)

        assert engine.metrics.errors_total == 1

    def test_retry_exhausted_emits_error_event(self) -> None:
        """error event is emitted on exhaustion."""
        import threading

        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        mock.infer.side_effect = RuntimeError("gpu error")
        received = threading.Event()
        events: list[object] = []

        def _cb(data: object) -> None:
            events.append(data)
            received.set()

        engine.on("error", _cb)

        with patch("yowo.engine.time.sleep"):
            engine._infer_with_retry(tensor)

        received.wait(timeout=1.0)
        assert len(events) == 1

    def test_backoff_delays_are_correct(self) -> None:
        """Backoff delays: 0.1s after attempt 0, 0.2s after attempt 1."""
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        mock.infer.side_effect = RuntimeError("gpu error")

        with patch("yowo.engine.time.sleep") as mock_sleep:
            engine._infer_with_retry(tensor)

        # 3 attempts, 2 sleeps (no sleep after last failed attempt)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list == [call(0.1), call(0.2)]

    def test_engine_does_not_crash_after_exhaustion(self) -> None:
        """Engine remains operational after retry exhaustion."""
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)
        tensor = _make_tensor()

        mock.infer.side_effect = [
            RuntimeError("gpu error"),
            RuntimeError("gpu error"),
            RuntimeError("gpu error"),
            np.zeros((1, 0, 6), dtype=np.float32),  # 4th call succeeds
        ]

        with patch("yowo.engine.time.sleep"):
            result1 = engine._infer_with_retry(tensor)

        # Reset for next call
        mock.infer.side_effect = None
        mock.infer.return_value = np.zeros((1, 3, 6), dtype=np.float32)

        with patch("yowo.engine.time.sleep"):
            result2 = engine._infer_with_retry(tensor)

        assert result1.shape == (1, 0, 6)
        assert result2.shape == (1, 3, 6)


# ---------------------------------------------------------------------------
# _run_gpu uses _infer_with_retry
# ---------------------------------------------------------------------------


class TestRunGpuUsesRetry:
    def test_run_gpu_delegates_to_infer_with_retry(self) -> None:
        """_run_gpu calls _infer_with_retry instead of direct backend.infer."""
        mock = _make_mock_backend()
        engine = _loaded_engine(mock)

        tensor = _make_tensor()
        frames: list[Frame] = []

        # Patch _infer_with_retry to verify it's called
        with patch.object(
            engine, "_infer_with_retry", return_value=np.zeros((1, 0, 6), dtype=np.float32)
        ) as mock_retry:
            engine._run_gpu(tensor, frames)

        mock_retry.assert_called_once_with(tensor)
