"""Unit tests for ClassificationEngine using mocked backends."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.classify_engine import ClassificationEngine
from yowo.config import ClassificationConfig
from yowo.errors import InferenceError, ShutdownError
from yowo.types import (
    BackendType,
    ClassificationResult,
    Frame,
    ModelFamily,
    ModelSize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend(output_shape: tuple[int, int] = (1, 1000)) -> MagicMock:
    """MagicMock satisfying InferenceBackend protocol for classification."""
    backend = MagicMock(spec=InferenceBackend)
    backend.backend_type = BackendType.PYTORCH
    backend.is_loaded = False
    backend.input_shape = (224, 224)

    probs = np.zeros(output_shape, dtype=np.float32)
    probs[0, 42] = 1.0  # class 42 wins (already "softmaxed")
    backend.infer.return_value = probs

    def _load(*args: object, **kwargs: object) -> None:
        backend.is_loaded = True

    backend.load.side_effect = _load
    backend.warmup.return_value = None
    backend.unload.return_value = None
    backend.clear_kv_cache = MagicMock()
    return backend


def _make_frame(index: int = 0) -> Frame:
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassificationEngineInit:
    def test_init_with_kwargs(self) -> None:
        """Engine created via keyword args without error."""
        backend = _make_mock_backend()
        engine = ClassificationEngine(
            backend_instance=backend,
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.NANO,
            top_k=3,
        )
        assert engine is not None

    def test_init_with_config(self) -> None:
        """Engine created via ClassificationConfig without error."""
        backend = _make_mock_backend()
        config = ClassificationConfig(
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.NANO,
            top_k=5,
        )
        engine = ClassificationEngine(config, backend_instance=backend)
        assert engine is not None

    def test_spec_task_is_classify(self) -> None:
        """Engine._spec.task == 'classify'."""
        backend = _make_mock_backend()
        engine = ClassificationEngine(backend_instance=backend)
        assert engine._spec.task == "classify"


class TestClassificationEngineLifecycle:
    def test_load_and_classify(self) -> None:
        """load() + classify() returns a ClassificationResult."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            results = engine.classify([_make_frame()])
            engine.close()

        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)

    def test_top1_class_id(self) -> None:
        """top1_class_id == 42 because backend returns probs[0, 42]=1.0."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            results = engine.classify([_make_frame()])
            engine.close()

        assert results[0].top1_class_id == 42

    def test_top_k_controls_result_length(self) -> None:
        """top_k=3 returns exactly 3 entries in topk_class_ids."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend, top_k=3)
            engine.load()
            results = engine.classify([_make_frame()])
            engine.close()

        assert len(results[0].topk_class_ids) == 3

    def test_context_manager(self) -> None:
        """Context manager calls load() on enter, close() on exit."""
        backend = _make_mock_backend()
        with (
            patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")),
            ClassificationEngine(backend_instance=backend) as engine,
        ):
            assert engine.is_loaded
            results = engine.classify([_make_frame()])
        assert len(results) == 1
        assert not engine.is_loaded

    def test_classify_not_loaded_raises(self) -> None:
        """classify() before load() raises InferenceError."""
        backend = _make_mock_backend()
        engine = ClassificationEngine(backend_instance=backend)
        with pytest.raises(InferenceError, match="not loaded"):
            engine.classify([_make_frame()])

    def test_shutting_down_raises(self) -> None:
        """classify() after close() raises ShutdownError."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            engine.close()

        with pytest.raises(ShutdownError):
            engine.classify([_make_frame()])

    def test_events_fired_on_classify(self) -> None:
        """on('detection', cb) fires after classify() (engine emits 'detection')."""
        backend = _make_mock_backend()
        received: list[object] = []

        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.on("detection", lambda results: received.extend(results))
            engine.load()
            engine.classify([_make_frame()])
            engine.close()

        assert len(received) == 1
        assert isinstance(received[0], ClassificationResult)


# ---------------------------------------------------------------------------
# stream method tests
# ---------------------------------------------------------------------------


class TestClassificationEngineStream:
    def test_stream_not_loaded_raises_inference_error(self) -> None:
        """stream() before load() raises InferenceError."""
        from unittest.mock import MagicMock

        backend = _make_mock_backend()
        engine = ClassificationEngine(backend_instance=backend)
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False

        with pytest.raises(InferenceError, match="not loaded"):
            engine.stream(mock_source)

    def test_stream_after_close_raises_shutdown_error(self) -> None:
        """stream() after close() raises ShutdownError."""
        from unittest.mock import MagicMock

        backend = _make_mock_backend()
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False

        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            engine.close()

        with pytest.raises(ShutdownError):
            engine.stream(mock_source)


# ---------------------------------------------------------------------------
# aclassify method tests
# ---------------------------------------------------------------------------


class TestClassificationEngineAClassify:
    async def test_aclassify_returns_results(self) -> None:
        """await engine.aclassify([frame]) returns a list of ClassificationResult."""
        backend = _make_mock_backend()
        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            try:
                results = await engine.aclassify([_make_frame()])
            finally:
                engine.close()

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)

    async def test_aclassify_not_loaded_raises_inference_error(self) -> None:
        """await engine.aclassify([frame]) before load() raises InferenceError."""
        backend = _make_mock_backend()
        engine = ClassificationEngine(backend_instance=backend)

        with pytest.raises(InferenceError, match="not loaded"):
            await engine.aclassify([_make_frame()])


# ---------------------------------------------------------------------------
# astream method tests
# ---------------------------------------------------------------------------


class TestClassificationEngineAStream:
    async def test_astream_after_close_raises_shutdown_error(self) -> None:
        """astream() after close() raises ShutdownError."""
        from unittest.mock import MagicMock

        backend = _make_mock_backend()
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False

        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            engine.close()

        with pytest.raises(ShutdownError):
            async for _ in engine.astream(mock_source):
                pass

    async def test_astream_yields_classification_results(self) -> None:
        """astream() over a mock source yields ClassificationResult instances."""
        from unittest.mock import MagicMock

        backend = _make_mock_backend()
        frame = _make_frame()
        mock_source = MagicMock()
        mock_source.total_frames = 1
        mock_source.is_live = False
        mock_source.__iter__ = MagicMock(return_value=iter([frame]))
        mock_source.close = MagicMock()

        with patch("yowo.engine.resolve_weights", return_value=Path("/fake/cls.pt")):
            engine = ClassificationEngine(backend_instance=backend)
            engine.load()
            try:
                collected: list[ClassificationResult] = []
                async for result in engine.astream(mock_source):
                    collected.append(result)
            finally:
                engine.close()

        assert len(collected) == 1
        assert isinstance(collected[0], ClassificationResult)
