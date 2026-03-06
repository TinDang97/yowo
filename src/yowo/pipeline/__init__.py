"""Multi-stream inference pipeline.

Provides four composable modules for running N concurrent video/RTSP
streams through a single :class:`~yowo.engine.InferenceEngine`:

- :class:`FrameCollector` — manages stream readers, yields TaggedFrame
- :class:`BatchScheduler` — accumulates frames into batches
- :class:`DetectionRouter` — routes results to per-stream callbacks
- :func:`run_pipeline` — thin wiring function connecting all modules

Usage::

    from yowo import InferenceEngine
    from yowo.pipeline import (
        BatchScheduler,
        DetectionRouter,
        FrameCollector,
        run_pipeline,
    )

    engine = InferenceEngine(...)
    engine.load()

    collector = FrameCollector(max_queue_size=4)
    collector.add_stream("cam-0", source_0)
    collector.add_stream("cam-1", source_1)

    scheduler = BatchScheduler(collector, max_batch_size=4, timeout_ms=50)
    router = DetectionRouter()
    router.register("cam-0", on_cam0)
    router.register("cam-1", on_cam1)

    run_pipeline(engine, collector, scheduler, router)
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from yowo.pipeline._collector import FrameCollector
from yowo.pipeline._router import DetectionRouter
from yowo.pipeline._scheduler import BatchScheduler
from yowo.types import StreamState

if TYPE_CHECKING:
    from yowo.engine import InferenceEngine
    from yowo.types import Detection, TaggedFrame

logger = logging.getLogger(__name__)

__all__ = [
    "BatchScheduler",
    "DetectionRouter",
    "FrameCollector",
    "run_pipeline",
]


def run_pipeline(
    engine: InferenceEngine,
    collector: FrameCollector,
    scheduler: BatchScheduler,
    router: DetectionRouter,
    *,
    stop_event: threading.Event | None = None,
    overlap: bool = True,
) -> None:
    """Wire collector -> scheduler -> engine.detect() -> router.

    This is a blocking call that runs until all streams are exhausted,
    ``stop_event`` is set, or an unhandled exception propagates.

    **Feature cache safety**: When multiple streams feed through a single
    engine, the per-source feature cache would be keyed incorrectly
    (``detect()`` uses ``frames[0].source_id``). This function
    automatically disables it and logs a warning.

    **Pipeline overlap** (``overlap=True``, the default): While
    ``engine.detect(batch_N)`` runs on a worker thread, the main thread
    assembles ``batch_N+1`` from the scheduler. This hides batch-assembly
    latency behind inference and improves throughput for multi-stream
    workloads.

    Args:
        engine: A loaded :class:`~yowo.engine.InferenceEngine`.
        collector: Manages the input streams.
        scheduler: Batches frames from the collector.
        router: Dispatches detection results to per-stream callbacks.
        stop_event: Optional external stop signal. When set, the scheduler
            flushes any partial batch and the pipeline returns. Also
            propagated into the scheduler for responsive cancellation.
        overlap: When ``True`` (default), overlap batch assembly with
            inference using a single-thread pool. When ``False``, process
            batches synchronously (original behaviour).
    """
    _disable_feature_cache(engine)

    # Propagate stop_event into the scheduler so it can break out of
    # upstream pulls promptly (fixes indefinite blocking when upstream
    # has no frames but is not yet exhausted).
    if stop_event is not None:
        scheduler.set_stop_event(stop_event)

    if overlap:
        _run_overlapped(engine, collector, scheduler, router)
    else:
        _run_synchronous(engine, collector, scheduler, router)

    _check_stream_errors(collector)


def _run_synchronous(
    engine: InferenceEngine,
    collector: FrameCollector,
    scheduler: BatchScheduler,
    router: DetectionRouter,
) -> None:
    """Process batches sequentially: assemble -> detect -> route."""
    try:
        for batch in scheduler:
            frames = [tagged.frame for tagged in batch]
            detections = engine.detect(frames)
            router.route(detections, batch)
    finally:
        collector.close()


def _run_overlapped(
    engine: InferenceEngine,
    collector: FrameCollector,
    scheduler: BatchScheduler,
    router: DetectionRouter,
) -> None:
    """Overlap batch assembly with inference via a single worker thread.

    While ``engine.detect(batch_N)`` executes on the worker, the main
    thread pulls frames and assembles ``batch_N+1``.  A
    ``deque[tuple[Future, batch]]`` with at most one pending future
    ensures back-pressure: we drain the oldest result before submitting
    the next job.
    """
    pending: deque[tuple[Future[list[Detection]], list[TaggedFrame]]] = deque()
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="yowo-pipe") as pool:
            for batch in scheduler:
                frames = [tagged.frame for tagged in batch]
                # Drain the oldest future before submitting new work.
                if pending:
                    fut, prev_batch = pending.popleft()
                    router.route(fut.result(), prev_batch)
                pending.append((pool.submit(engine.detect, frames), batch))
            # Drain remaining futures after scheduler is exhausted.
            while pending:
                fut, prev_batch = pending.popleft()
                router.route(fut.result(), prev_batch)
    finally:
        # Cancel futures that haven't started, then wait for any
        # in-flight future to finish so the worker thread is not left
        # accessing engine state after we return.
        for fut, _ in pending:
            fut.cancel()
        for fut, _ in pending:
            if not fut.cancelled():
                with contextlib.suppress(Exception):
                    fut.result(timeout=10.0)
        collector.close()


def _check_stream_errors(collector: FrameCollector) -> None:
    """Log or raise if stream errors occurred during the pipeline run.

    Raises ``RuntimeError`` when ALL streams failed (total pipeline failure).
    Logs a warning per failed stream for partial failures.
    """
    errors = collector.stream_errors
    if not errors:
        return

    states = collector.stream_states
    all_failed = states and all(s == StreamState.ERROR for s in states.values())
    if all_failed:
        msg = "; ".join(f"{sid}: {exc}" for sid, exc in errors.items())
        raise RuntimeError(f"All pipeline streams failed: {msg}")

    for sid, exc in errors.items():
        logger.warning("Stream %r failed during pipeline run: %s", sid, exc)


def _disable_feature_cache(engine: InferenceEngine) -> None:
    """Disable the feature cache on the engine if active.

    Mixed-source batches corrupt the per-source cache key
    (``engine.detect()`` uses ``frames[0].source_id``).
    """
    # Access the private attribute — acceptable since pipeline is part
    # of the same package and this is a documented constraint.
    # If the attribute is renamed in a future refactor, getattr returns
    # None and this becomes a safe no-op (log a warning at DEBUG).
    cache = getattr(engine, "_feature_cache", None)
    if cache is not None:
        logger.warning(
            "Multi-stream pipeline: disabling feature cache "
            "(incompatible with mixed-source batches)"
        )
        cache.clear()
        engine._feature_cache = None  # type: ignore[attr-defined]
