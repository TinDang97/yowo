"""ClassificationEngine: YOLO image classification inference engine.

Usage::

    from yowo.classify_engine import ClassificationEngine
    from yowo.types import ModelFamily, ModelSize

    engine = ClassificationEngine(
        model_family=ModelFamily.YOLO11,
        model_size=ModelSize.NANO,
    )
    with engine:
        results = engine.classify(frames)
        # results[0].top1_class_id, results[0].top1_score
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Generator, Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from yowo.backends import InferenceBackend
from yowo.config import ClassificationConfig
from yowo.engine import BaseEngine
from yowo.errors import InferenceError, ShutdownError
from yowo.io import FrameSource
from yowo.postprocess import PostprocessBuffer
from yowo.postprocess._classify import postprocess_classify
from yowo.types import (
    BackendType,
    ClassificationResult,
    Frame,
    FrameDropPolicy,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
    PreprocessedTensor,
)


class ClassificationEngine(BaseEngine):
    """YOLO image classification engine.

    Mirrors :class:`~yowo.engine.DetectionEngine` API but outputs
    :class:`~yowo.types.ClassificationResult` instead of
    :class:`~yowo.types.Detection`. No NMS, no bounding boxes — just
    top-k class probabilities.

    Usage::

        engine = ClassificationEngine(
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.NANO,
        )
        with engine:
            results = engine.classify(frames)
            print(results[0].top1_class_id, results[0].top1_score)

    Args:
        config: Optional :class:`~yowo.config.ClassificationConfig`. When
            provided all individual kwargs are ignored.
        backend_instance: Optional pre-built backend to inject. Bypasses
            auto-selection; caller owns device placement and error handling.
        model_family: YOLO model family (default YOLO11).
        model_size: Size variant (default NANO).
        weights_path: Path to a local ``-cls.pt`` file, or ``None`` to
            use the registry default.
        backend: Backend type. ``None`` triggers automatic selection.
        device: Device string (default ``"auto"``).
        precision: Numerical precision. ``None`` triggers auto-selection.
        batch_size: Number of frames per inference batch (default 1).
        top_k: Number of top predictions to return per frame (default 5).
        frame_drop_policy: Backlog policy for live streaming.
        max_queue_size: Bounded queue depth for ThreadedFrameReader.
        prefetch: Enable threaded frame prefetch in ``stream()``.
        pipeline_workers: Worker thread count. ``0`` = auto-detect.
        metrics_enabled: Enable latency/throughput metrics collection.
        error_threshold: Cumulative errors before health → DEGRADED.

    Raises:
        ValueError: If ``kv_cache=True`` is passed (not supported for
            classification models).
    """

    def __init__(
        self,
        config: ClassificationConfig | None = None,
        *,
        backend_instance: InferenceBackend | None = None,
        model_family: ModelFamily = ModelFamily.YOLO11,
        model_size: ModelSize = ModelSize.NANO,
        weights_path: Path | None = None,
        backend: BackendType | None = None,
        device: str = "auto",
        precision: Precision | None = None,
        batch_size: int = 1,
        top_k: int = 5,
        frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST,
        max_queue_size: int = 2,
        prefetch: bool = True,
        pipeline_workers: int = 0,
        metrics_enabled: bool = True,
        error_threshold: int = 10,
    ) -> None:
        if config is not None:
            family = config.model_family
            size = config.model_size
            wp = config.weights_path
            be = config.backend
            dev = config.device
            prec = config.precision
            bs = config.batch_size
            tk = config.top_k
            fdp = config.frame_drop_policy
            mqs = config.max_queue_size
            pf = config.prefetch
            pw = config.pipeline_workers
            me = config.metrics_enabled
            et = config.error_threshold
        else:
            family = model_family
            size = model_size
            wp = weights_path
            be = backend
            dev = device
            prec = precision
            bs = batch_size
            tk = top_k
            fdp = frame_drop_policy
            mqs = max_queue_size
            pf = prefetch
            pw = pipeline_workers
            me = metrics_enabled
            et = error_threshold

        self._top_k = tk
        spec = ModelSpec(family=family, size=size, task="classify", weights_path=wp)

        super().__init__(
            spec=spec,
            backend_instance=backend_instance,
            backend_override=be.value if be else None,
            device=dev,
            precision=prec,
            batch_size=bs,
            # Classification models don't support feature cache or kv_cache
            cache=False,
            cache_dir=None,
            kv_cache=False,
            frame_drop_policy=fdp,
            max_queue_size=mqs,
            prefetch=pf,
            pipeline_workers=pw,
            metrics_enabled=me,
            error_threshold=et,
        )

    def _process_batch(
        self,
        raw_output: NDArray[np.float32],
        tensor: PreprocessedTensor,
        frames: list[Frame],
        elapsed_ms: float,
        scratch: PostprocessBuffer | None,
    ) -> list[ClassificationResult]:
        return postprocess_classify(
            raw_output,
            frames,
            model_spec=self._spec,
            backend=self._selection.backend,
            top_k=self._top_k,
            inference_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Public classification API
    # ------------------------------------------------------------------

    def classify(self, frames: list[Frame]) -> list[ClassificationResult]:
        """Run classification on a list of frames.

        Returns one :class:`~yowo.types.ClassificationResult` per frame.

        Args:
            frames: Input frames to classify.

        Returns:
            One ClassificationResult per frame.

        Raises:
            ShutdownError: If the engine is shutting down.
            InferenceError: If the engine has not been loaded.
        """
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")

        try:
            return self._run_batch(frames)  # type: ignore[return-value]
        except Exception:
            raise

    async def aclassify(self, frames: list[Frame]) -> list[ClassificationResult]:
        """Async wrapper — offloads classify() to a thread pool.

        Args:
            frames: Input frames to classify.

        Returns:
            One ClassificationResult per frame.
        """
        return await asyncio.to_thread(self.classify, frames)

    def stream(self, source: FrameSource) -> Iterator[ClassificationResult]:
        """Yield classification results from a FrameSource, batching internally.

        Dispatches to source-type-aware strategy (same as DetectionEngine):
        - Single image → _stream_single (no threading overhead)
        - Live source (RTSP/webcam) → _stream_live (threaded reader + frame drop)
        - Offline multi-frame → _stream_pipeline (prefetch + infer overlap)
        - prefetch=False → _stream_sync (legacy sequential path)

        Args:
            source: Any :class:`~yowo.io.FrameSource` (image, video, RTSP, …).

        Yields:
            One :class:`~yowo.types.ClassificationResult` per frame.

        Raises:
            ShutdownError: If the engine is shutting down.
            InferenceError: If the engine has not been loaded.
        """
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")
        return self._stream_dispatch(source)  # type: ignore[return-value]

    async def astream(self, source: FrameSource) -> AsyncIterator[ClassificationResult]:
        """Async stream classification results.

        Runs the sync :meth:`stream` generator on a background thread and
        bridges results to the async caller via an asyncio.Queue.

        Args:
            source: Any :class:`~yowo.io.FrameSource`.

        Yields:
            One :class:`~yowo.types.ClassificationResult` per frame.

        Raises:
            ShutdownError: If the engine is shutting down.
        """
        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[ClassificationResult | None] = asyncio.Queue(maxsize=64)
        _internal_stop = threading.Event()

        def _background() -> None:
            # stream() returns a generator at runtime; type: ignore lets us call .close()
            gen: Generator[ClassificationResult, None, None] = self.stream(source)  # type: ignore[assignment]
            try:
                for result in gen:
                    if _internal_stop.is_set() or _stop.is_set():
                        break
                    future = asyncio.run_coroutine_threadsafe(q.put(result), loop)
                    try:
                        future.result(timeout=1.0)
                    except Exception:
                        break
            except Exception as exc:
                self._event_bus.emit("error", exc)
            finally:
                # Generator.close() triggers _stream_* finally blocks immediately
                # (reader.stop(), source.close()) rather than waiting for GC.
                with contextlib.suppress(Exception):
                    gen.close()
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(q.put(None), loop).result(timeout=2.0)

        thread = threading.Thread(target=_background, name="yowo-cls-astream", daemon=True)
        thread.start()
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            _internal_stop.set()
            raise
        finally:
            _internal_stop.set()
            thread.join(timeout=5.0)
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()

    def __enter__(self) -> ClassificationEngine:
        self.load()
        return self

    async def __aenter__(self) -> ClassificationEngine:
        await asyncio.to_thread(self.load)
        return self


__all__ = ["ClassificationEngine"]
