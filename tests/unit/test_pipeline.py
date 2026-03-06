"""Integration tests for run_pipeline() end-to-end wiring.

Verifies: sources -> collector -> scheduler -> engine.detect() -> router -> callbacks.
Uses real FrameCollector, BatchScheduler, DetectionRouter with mock FrameSource and mock engine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from yowo.pipeline import BatchScheduler, DetectionRouter, FrameCollector, run_pipeline
from yowo.types import (
    BackendType,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)


class _MockSource:
    """Minimal FrameSource implementation for testing."""

    def __init__(self, source_id: str, n: int, *, error_at: int | None = None) -> None:
        self._source_id = source_id
        self._n = n
        self._error_at = error_at
        self.closed = False

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        return self._n

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self._n):
            if self._error_at == i:
                raise RuntimeError(f"source {self._source_id} error at frame {i}")
            yield Frame(
                pixels=np.zeros((4, 4, 3), dtype=np.uint8),
                source_id=self._source_id,
                frame_index=i,
            )

    def close(self) -> None:
        self.closed = True


def _make_detect_fn() -> MagicMock:
    """Return a side_effect function that produces one Detection per frame."""

    def _detect(frames: list[Frame]) -> list[Detection]:
        return [
            Detection(
                frame=f,
                boxes=(),
                inference_time_ms=1.0,
                backend=BackendType.PYTORCH,
                model_spec=_SPEC,
            )
            for f in frames
        ]

    mock = MagicMock(side_effect=_detect)
    return mock


def _make_engine(detect_fn: MagicMock | None = None) -> MagicMock:
    """Build a mock InferenceEngine with a working detect() method."""
    engine = MagicMock()
    engine._feature_cache = None
    engine._loaded = True

    if detect_fn is None:
        detect_fn = _make_detect_fn()
    engine.detect = detect_fn
    return engine


# ---------------------------------------------------------------------------
# TestRunPipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """End-to-end integration tests for run_pipeline()."""

    def test_end_to_end_single_stream(self) -> None:
        """1 stream with 3 frames, batch_size=1 -> callback receives 3 calls."""
        source = _MockSource("cam-0", n=3)
        engine = _make_engine()
        received: list[tuple[str, list[Detection]]] = []

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: received.append((sid, dets)))

        run_pipeline(engine, collector, scheduler, router)

        assert len(received) == 3
        for stream_id, dets in received:
            assert stream_id == "cam-0"
            assert len(dets) == 1
            assert dets[0].backend == BackendType.PYTORCH
            assert dets[0].model_spec == _SPEC
        assert engine.detect.call_count == 3

    def test_end_to_end_multiple_streams(self) -> None:
        """2 streams, 2 frames each, batch_size=2 -> each callback gets its detections."""
        source_a = _MockSource("cam-A", n=2)
        source_b = _MockSource("cam-B", n=2)
        engine = _make_engine()
        received_a: list[tuple[str, list[Detection]]] = []
        received_b: list[tuple[str, list[Detection]]] = []

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-A", source_a)
        collector.add_stream("cam-B", source_b)

        scheduler = BatchScheduler(collector, max_batch_size=2, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-A", lambda sid, dets: received_a.append((sid, dets)))
        router.register("cam-B", lambda sid, dets: received_b.append((sid, dets)))

        run_pipeline(engine, collector, scheduler, router)

        # Total 4 frames across both streams.
        total_a = sum(len(dets) for _, dets in received_a)
        total_b = sum(len(dets) for _, dets in received_b)
        assert total_a == 2
        assert total_b == 2

        # All routed to the correct stream.
        for _, dets in received_a:
            for d in dets:
                assert d.frame.source_id == "cam-A"
        for _, dets in received_b:
            for d in dets:
                assert d.frame.source_id == "cam-B"

    def test_stop_event_terminates_pipeline(self) -> None:
        """Setting stop_event before start causes pipeline to return quickly."""
        source = _MockSource("cam-0", n=100)
        engine = _make_engine()

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        stop = threading.Event()
        stop.set()

        t0 = time.monotonic()
        run_pipeline(engine, collector, scheduler, router, stop_event=stop)
        elapsed = time.monotonic() - t0

        # With stop_event pre-set, at most 1 batch is processed before break.
        assert engine.detect.call_count <= 1
        # Pipeline should finish quickly (well under 5 seconds).
        assert elapsed < 5.0

    def test_feature_cache_auto_disabled(self) -> None:
        """Engine with _feature_cache set -> run_pipeline disables it."""
        source = _MockSource("cam-0", n=1)
        engine = _make_engine()

        # Simulate an active feature cache.
        mock_cache = MagicMock()
        engine._feature_cache = mock_cache

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        run_pipeline(engine, collector, scheduler, router)

        mock_cache.clear.assert_called_once()
        assert engine._feature_cache is None

    def test_collector_closed_on_normal_exit(self) -> None:
        """After normal pipeline completion, collector.close() has been called."""
        source = _MockSource("cam-0", n=2)
        engine = _make_engine()

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        run_pipeline(engine, collector, scheduler, router)

        # After run_pipeline, collector is closed (source gets closed too).
        assert source.closed

    def test_collector_closed_on_exception(self) -> None:
        """engine.detect() raises -> collector still closed via finally block."""
        source = _MockSource("cam-0", n=3)

        def _explode(frames: list[Any]) -> list[Detection]:
            raise RuntimeError("inference failure")

        engine = _make_engine(detect_fn=MagicMock(side_effect=_explode))

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        with pytest.raises(RuntimeError, match="inference failure"):
            run_pipeline(engine, collector, scheduler, router)

        # Collector must be closed even after exception.
        assert source.closed


# ---------------------------------------------------------------------------
# TestRunPipelineOverlap
# ---------------------------------------------------------------------------


class TestRunPipelineOverlap:
    """Tests for the overlap=True / overlap=False paths in run_pipeline()."""

    def test_run_pipeline_overlap_ordering(self) -> None:
        """overlap=True: 3 batches routed in correct sequential order."""
        source = _MockSource("cam-0", n=3)
        engine = _make_engine()
        received: list[tuple[str, list[Detection]]] = []

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: received.append((sid, dets)))

        run_pipeline(engine, collector, scheduler, router, overlap=True)

        # Must receive exactly 3 routed batches.
        assert len(received) == 3
        assert engine.detect.call_count == 3
        # Verify frame_index ordering (batches routed in the order yielded).
        for i, (stream_id, dets) in enumerate(received):
            assert stream_id == "cam-0"
            assert len(dets) == 1
            assert dets[0].frame.frame_index == i

    def test_run_pipeline_overlap_false_synchronous(self) -> None:
        """overlap=False uses synchronous path, identical results."""
        source = _MockSource("cam-0", n=3)
        engine = _make_engine()
        received: list[tuple[str, list[Detection]]] = []

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: received.append((sid, dets)))

        run_pipeline(engine, collector, scheduler, router, overlap=False)

        assert len(received) == 3
        assert engine.detect.call_count == 3
        for i, (stream_id, dets) in enumerate(received):
            assert stream_id == "cam-0"
            assert len(dets) == 1
            assert dets[0].frame.frame_index == i

    def test_run_pipeline_overlap_error_propagation(self) -> None:
        """overlap=True: error in detect() propagates, collector still closed."""
        source = _MockSource("cam-0", n=3)
        call_count = 0

        def _fail_on_second(frames: list[Any]) -> list[Detection]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("batch-2 failure")
            return [
                Detection(
                    frame=f,
                    boxes=(),
                    inference_time_ms=1.0,
                    backend=BackendType.PYTORCH,
                    model_spec=_SPEC,
                )
                for f in frames
            ]

        engine = _make_engine(detect_fn=MagicMock(side_effect=_fail_on_second))

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        with pytest.raises(RuntimeError, match="batch-2 failure"):
            run_pipeline(engine, collector, scheduler, router, overlap=True)

        # Collector must be closed even after error in overlapped path.
        assert source.closed

    def test_run_pipeline_overlap_stop_event(self) -> None:
        """overlap=True: pre-set stop_event causes clean early termination."""
        source = _MockSource("cam-0", n=100)
        engine = _make_engine()

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        stop = threading.Event()
        stop.set()

        t0 = time.monotonic()
        run_pipeline(engine, collector, scheduler, router, stop_event=stop, overlap=True)
        elapsed = time.monotonic() - t0

        # With stop_event pre-set, at most 1 batch is processed.
        assert engine.detect.call_count <= 1
        assert elapsed < 5.0
        assert source.closed


# ---------------------------------------------------------------------------
# TestPipelineStreamErrors (Phase 3b)
# ---------------------------------------------------------------------------


class TestPipelineStreamErrors:
    """Tests for _check_stream_errors surfacing stream failures."""

    def test_all_streams_error_raises(self) -> None:
        """All streams erroring raises RuntimeError from run_pipeline."""
        source_a = _MockSource("cam-A", n=3, error_at=0)
        source_b = _MockSource("cam-B", n=3, error_at=0)
        engine = _make_engine()

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-A", source_a)
        collector.add_stream("cam-B", source_b)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-A", lambda sid, dets: None)
        router.register("cam-B", lambda sid, dets: None)

        with pytest.raises(RuntimeError, match="All pipeline streams failed"):
            run_pipeline(engine, collector, scheduler, router, overlap=False)

    def test_partial_error_no_raise(self) -> None:
        """One stream ok + one erroring: run_pipeline completes without raising."""
        source_ok = _MockSource("cam-A", n=3)
        source_err = _MockSource("cam-B", n=3, error_at=0)
        engine = _make_engine()
        received: list[tuple[str, list[Detection]]] = []

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-A", source_ok)
        collector.add_stream("cam-B", source_err)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-A", lambda sid, dets: received.append((sid, dets)))
        router.register("cam-B", lambda sid, dets: received.append((sid, dets)))

        # Should not raise — partial failure is logged, not raised
        run_pipeline(engine, collector, scheduler, router, overlap=False)

        # cam-A should have delivered its 3 frames
        a_count = sum(1 for sid, _ in received if sid == "cam-A")
        assert a_count == 3

    def test_no_errors_no_raise(self) -> None:
        """Normal run with no errors completes cleanly."""
        source = _MockSource("cam-0", n=3)
        engine = _make_engine()

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-0", source)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-0", lambda sid, dets: None)

        # Must not raise
        run_pipeline(engine, collector, scheduler, router, overlap=False)

    def test_overlap_surfaces_all_stream_errors(self) -> None:
        """overlap=True: all streams erroring also raises RuntimeError."""
        source_a = _MockSource("cam-A", n=3, error_at=0)
        source_b = _MockSource("cam-B", n=3, error_at=0)
        engine = _make_engine()

        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-A", source_a)
        collector.add_stream("cam-B", source_b)

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-A", lambda sid, dets: None)
        router.register("cam-B", lambda sid, dets: None)

        with pytest.raises(RuntimeError, match="All pipeline streams failed"):
            run_pipeline(engine, collector, scheduler, router, overlap=True)
