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

import logging
import threading
from typing import TYPE_CHECKING

from yowo.pipeline._collector import FrameCollector
from yowo.pipeline._router import DetectionRouter
from yowo.pipeline._scheduler import BatchScheduler

if TYPE_CHECKING:
    from yowo.engine import InferenceEngine

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
) -> None:
    """Wire collector → scheduler → engine.detect() → router.

    This is a blocking call that runs until all streams are exhausted,
    ``stop_event`` is set, or an unhandled exception propagates.

    **Feature cache safety**: When multiple streams feed through a single
    engine, the per-source feature cache would be keyed incorrectly
    (``detect()`` uses ``frames[0].source_id``). This function
    automatically disables it and logs a warning.

    Args:
        engine: A loaded :class:`~yowo.engine.InferenceEngine`.
        collector: Manages the input streams.
        scheduler: Batches frames from the collector.
        router: Dispatches detection results to per-stream callbacks.
        stop_event: Optional external stop signal. When set, the scheduler
            flushes any partial batch and the pipeline returns. Also
            propagated into the scheduler for responsive cancellation.
    """
    _disable_feature_cache(engine)

    # Propagate stop_event into the scheduler so it can break out of
    # upstream pulls promptly (fixes indefinite blocking when upstream
    # has no frames but is not yet exhausted).
    if stop_event is not None:
        scheduler.set_stop_event(stop_event)

    try:
        for batch in scheduler:
            # Extract raw Frame list (engine.detect expects list[Frame]).
            frames = [tagged.frame for tagged in batch]
            detections = engine.detect(frames)
            router.route(detections, batch)
    finally:
        collector.close()


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
