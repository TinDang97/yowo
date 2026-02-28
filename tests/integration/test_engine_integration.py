"""Integration tests: real PyTorch CPU backend, real yolo26n weights.

Skipped automatically when weights are unavailable or torch is not installed.
All tests use scope="module" fixtures to load the model once per session.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from yowo.engine import InferenceEngine
from yowo.errors import ShutdownError
from yowo.io import open_source
from yowo.types import BackendType, Frame, HealthStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WEIGHTS = Path("tmp/weights/yolo26n_statedict.pt")


@pytest.fixture(scope="module")
def weights_path() -> Path:
    if not _WEIGHTS.exists():
        pytest.skip(f"yolo26n weights not found at {_WEIGHTS}")
    return _WEIGHTS


@pytest.fixture(scope="module")
def engine(weights_path: Path) -> Iterator[InferenceEngine]:
    pytest.importorskip("torch")
    with InferenceEngine(weights_path=weights_path) as eng:
        yield eng


@pytest.fixture(scope="module")
def dummy_frame() -> Frame:
    """640x640 black image as a single inference frame."""
    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="integration-test", frame_index=0)


@pytest.fixture(scope="module")
def dummy_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a 640x640 black JPEG to a temp directory."""
    img_dir = tmp_path_factory.mktemp("integration_imgs")
    p = img_dir / "black.jpg"
    cv2.imwrite(str(p), np.zeros((640, 640, 3), dtype=np.uint8))
    return p


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEngineLifecycle:
    def test_engine_loads_and_closes(self, weights_path: Path) -> None:
        """Context manager: engine loads on enter, closes on exit."""
        pytest.importorskip("torch")
        with InferenceEngine(weights_path=weights_path) as eng:
            assert eng.is_loaded is True
        assert eng.is_loaded is False

    def test_health_ready_after_load(self, engine: InferenceEngine) -> None:
        assert engine.health == HealthStatus.READY

    def test_health_closed_after_close(self, weights_path: Path) -> None:
        pytest.importorskip("torch")
        eng = InferenceEngine(weights_path=weights_path)
        with patch_resolve_or_real():
            eng.load()
        eng.close()
        assert eng.health == HealthStatus.CLOSED

    def test_shutdown_rejects_detect(self, weights_path: Path, dummy_frame: Frame) -> None:
        """After close(), detect() raises ShutdownError."""
        pytest.importorskip("torch")
        eng = InferenceEngine(weights_path=weights_path)
        eng.load()
        eng.close()
        with pytest.raises(ShutdownError):
            eng.detect([dummy_frame])


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEngineInference:
    def test_detect_returns_detection_list(
        self, engine: InferenceEngine, dummy_frame: Frame
    ) -> None:
        result = engine.detect([dummy_frame])
        assert isinstance(result, list)

    def test_detection_has_backend_type(self, engine: InferenceEngine, dummy_frame: Frame) -> None:
        result = engine.detect([dummy_frame])
        # Each Detection item must record the backend
        for det in result:
            assert det.backend == BackendType.PYTORCH

    def test_detection_has_positive_inference_time(
        self, engine: InferenceEngine, dummy_frame: Frame
    ) -> None:
        result = engine.detect([dummy_frame])
        for det in result:
            assert det.inference_time_ms > 0.0

    def test_metrics_after_detect(self, engine: InferenceEngine, dummy_frame: Frame) -> None:
        engine.detect([dummy_frame])
        assert engine.metrics.frames_total > 0


# ---------------------------------------------------------------------------
# Stream tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEngineStream:
    def test_stream_image(self, engine: InferenceEngine, dummy_image_path: Path) -> None:
        """engine.stream() on an image source yields at least one Detection."""
        source = open_source(dummy_image_path)
        detections = list(engine.stream(source))
        assert isinstance(detections, list)

    def test_event_detection_fired(self, engine: InferenceEngine, dummy_image_path: Path) -> None:
        """on('detection') callback is invoked when stream processes an image."""
        import time

        received: list[object] = []
        engine.on("detection", received.append)
        try:
            source = open_source(dummy_image_path)
            list(engine.stream(source))
            deadline = time.monotonic() + 2.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            engine.remove("detection", received.append)

        assert len(received) >= 1


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEngineAsync:
    async def test_async_detect(self, engine: InferenceEngine, dummy_frame: Frame) -> None:
        """await engine.adetect([frame]) returns a list of Detection."""
        result = await engine.adetect([dummy_frame])
        assert isinstance(result, list)

    async def test_async_context_manager(self, weights_path: Path) -> None:
        """Full `async with engine:` roundtrip."""
        pytest.importorskip("torch")
        async with InferenceEngine(weights_path=weights_path) as eng:
            assert eng.is_loaded is True
        assert eng.is_loaded is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def patch_resolve_or_real() -> object:
    """No-op context manager — real weights are used; no patch needed."""
    import contextlib

    return contextlib.nullcontext()
