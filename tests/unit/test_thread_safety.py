"""Unit tests for engine thread safety.

Covers _infer_lock existence and concurrent detect() correctness.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from yowo.backends import InferenceBackend
from yowo.engine import DetectionEngine
from yowo.types import BackendType, Frame

_RESOLVE_PATCH = "yowo.engine.resolve_weights"


def _make_mock_backend() -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)

    def _load(*a: object, **kw: object) -> None:
        mock.is_loaded = True

    def _unload(*a: object, **kw: object) -> None:
        mock.is_loaded = False

    mock.load.side_effect = _load
    mock.unload.side_effect = _unload
    mock.warmup.return_value = None
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    return mock


def _loaded_engine(mock_be: MagicMock) -> DetectionEngine:
    engine = DetectionEngine(backend_instance=mock_be)
    with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


def _dummy_frame() -> Frame:
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0)


class TestInferLock:
    def test_lock_exists(self) -> None:
        """BaseEngine has _infer_lock attribute of type threading.Lock."""
        mock_be = _make_mock_backend()
        engine = DetectionEngine(backend_instance=mock_be)
        assert hasattr(engine, "_infer_lock")
        assert isinstance(engine._infer_lock, type(threading.Lock()))

    def test_concurrent_detect_no_crash(self) -> None:
        """4 threads calling detect() concurrently produce correct results."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)

        errors: list[Exception] = []
        results: list[int] = []
        barrier = threading.Barrier(4)

        def _worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                result = engine.detect([_dummy_frame()])
                results.append(len(result))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        engine.close()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 4
        assert all(r == 1 for r in results)
