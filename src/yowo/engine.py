"""Inference engine orchestrator.

Wires all modules into a single coherent inference pipeline.
Handles backend lifecycle, batch accumulation, and graceful degradation.

Usage::

    from yowo import InferenceConfig, InferenceEngine

    config = InferenceConfig(confidence_threshold=0.35, batch_size=4)
    with InferenceEngine(config) as engine:
        for detection in engine.stream(open_source("video.mp4")):
            process(detection)

    # Keyword-only form (defaults to YOLO26 Nano)
    with InferenceEngine(confidence_threshold=0.35) as engine:
        ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.backends import (
    InferenceBackend,
    create_backend,
    get_fallback_backends,
    select_backend,
)
from yowo.config import InferenceConfig
from yowo.errors import BackendError, BackendLoadError, InferenceError, ShutdownError
from yowo.events import EventBus
from yowo.hardware import get_hardware_profile
from yowo.io import (
    FrameSource,
    PreparedItem,
    PreprocessBuffer,
    ThreadedFrameReader,
    preprocess,
    preprocess_into,
)
from yowo.metrics import EngineMetrics, MetricsCollector
from yowo.models import get as _registry_get
from yowo.models import resolve_weights
from yowo.models._registry import get_cls as _registry_get_cls
from yowo.postprocess import PostprocessBuffer, postprocess
from yowo.types import (
    BackendSelection,
    BackendType,
    Detection,
    DeviceType,
    Frame,
    FrameDropPolicy,
    HealthStatus,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
    PreprocessedTensor,
    is_free_threaded,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseEngine — shared lifecycle, streaming, metrics
# ---------------------------------------------------------------------------


class BaseEngine:
    """Shared lifecycle base for all YOWO inference engines.

    Subclasses implement :meth:`_process_batch` for task-specific output
    transformation (detection NMS, classification softmax, etc.).

    Do not instantiate directly — use :class:`DetectionEngine` or
    :class:`~yowo.classify_engine.ClassificationEngine`.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        backend_instance: InferenceBackend | None = None,
        backend_override: str | None = None,
        device: str = "auto",
        precision: Precision | None = None,
        batch_size: int = 1,
        cache: bool = False,
        cache_dir: Path | None = None,
        kv_cache: bool = False,
        frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST,
        max_queue_size: int = 2,
        prefetch: bool = True,
        pipeline_workers: int = 0,
        metrics_enabled: bool = True,
        error_threshold: int = 10,
    ) -> None:
        self._spec = spec
        self._batch_size = batch_size
        self._device = device
        self._feature_cache = None
        if cache or cache_dir is not None:
            from yowo.cache import FeatureCache

            self._feature_cache = FeatureCache(cache_dir=cache_dir)
        self._user_provided_backend = backend_instance is not None
        if backend_instance is not None:
            self._backend: InferenceBackend = backend_instance
            self._selection: BackendSelection = BackendSelection(
                backend=backend_instance.backend_type,
                device_type=DeviceType.CPU,
                precision=Precision.FP32,
                device_index=0,
                reason="User-provided backend instance",
            )
            self._hw = None
            self._kv_cache = kv_cache
        else:
            self._hw = get_hardware_profile()
            self._selection = select_backend(
                self._hw,
                model_size=spec.size.value,
                backend_override=backend_override,
                device_override=device if device != "auto" else None,
                precision_override=precision.value if precision else None,
            )
            self._kv_cache = kv_cache
            self._backend = create_backend(
                self._selection.backend,
                self._hw,
                model_spec=self._spec,
                feature_cache=self._feature_cache,
                kv_cache=kv_cache,
            )
        self._model_meta = (
            _registry_get_cls(spec.family, spec.size)
            if spec.task == "classify"
            else _registry_get(spec.family, spec.size)
        )
        self._loaded = False
        self._frame_drop_policy = frame_drop_policy
        self._max_queue_size = max_queue_size
        self._prefetch = prefetch
        self._pipeline_workers = pipeline_workers
        self._preprocess_buf: PreprocessBuffer | None = None
        self._postprocess_buf: PostprocessBuffer | None = None
        self._metrics = MetricsCollector(enabled=metrics_enabled)
        self._error_threshold = error_threshold
        self._health_state: HealthStatus = HealthStatus.STARTING
        self._event_bus = EventBus()
        self._shutting_down = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._active_streams: set[threading.Event] = set()
        self._streams_drained = threading.Event()
        self._streams_drained.set()

    @abstractmethod
    def _process_batch(
        self,
        raw_output: NDArray[np.float32],
        tensor: PreprocessedTensor,
        frames: list[Frame],
        elapsed_ms: float,
        scratch: PostprocessBuffer | None,
    ) -> list[Any]: ...

    @property
    def selection(self) -> BackendSelection:
        """The resolved backend selection (backend, device, precision)."""
        return self._selection

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def health(self) -> HealthStatus:
        """Derived engine health status (STARTING/READY/DEGRADED/CLOSED)."""
        if self._health_state == HealthStatus.CLOSED:
            return HealthStatus.CLOSED
        if self._health_state == HealthStatus.SHUTTING_DOWN:
            return HealthStatus.SHUTTING_DOWN
        if not self._loaded:
            return HealthStatus.STARTING
        if self._metrics.errors_total >= self._error_threshold:
            return HealthStatus.DEGRADED
        last = self._metrics.last_frame_time
        if self._metrics.frames_total > 0 and last > 0 and time.monotonic() - last > 30.0:
            return HealthStatus.DEGRADED
        return HealthStatus.READY

    @property
    def metrics(self) -> EngineMetrics:
        """Snapshot of current engine metrics."""
        return self._metrics.snapshot()

    def reset_metrics(self) -> None:
        """Zero all metric counters and the latency histogram."""
        self._metrics.reset()

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a synchronous event callback."""
        self._event_bus.on(event, callback)

    def on_async(
        self,
        event: str,
        callback: Callable[..., Any],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Register an async event callback."""
        self._event_bus.on_async(event, callback, loop=loop)

    def remove(self, event: str, callback: Callable[..., Any]) -> None:
        """Unregister an event callback."""
        self._event_bus.remove(event, callback)

    @property
    def events_dropped(self) -> int:
        """Number of events dropped due to queue overflow."""
        return self._event_bus.events_dropped

    def load(self) -> None:
        """Resolve weights, load into backend, warmup. Falls back on failure."""
        weights_path = resolve_weights(self._spec)
        if self._user_provided_backend:
            self._backend.load(weights_path, device=self._device)
            self._backend.warmup(batch_size=self._batch_size)
            self._finalize_load()
            return
        assert self._hw is not None
        backends_to_try = [self._selection.backend, *get_fallback_backends(self._selection.backend)]
        last_exc: Exception | None = None
        for bt in backends_to_try:
            try:
                if bt != self._selection.backend:
                    logger.warning(
                        "Primary backend %s failed, trying %s",
                        self._selection.backend.value,
                        bt.value,
                    )
                    self._backend = create_backend(
                        bt,
                        self._hw,
                        model_spec=self._spec,
                        feature_cache=self._feature_cache,
                        kv_cache=self._kv_cache,
                    )
                self._backend.load(weights_path, device=self._device)
                self._backend.warmup(batch_size=self._batch_size)
                if bt != self._selection.backend:
                    logger.warning("Using fallback backend: %s", bt.value)
                    _fs = select_backend(
                        self._hw, model_size=self._spec.size.value, backend_override=bt.value
                    )
                    self._selection = BackendSelection(
                        backend=bt,
                        device_type=_fs.device_type,
                        precision=_fs.precision,
                        device_index=_fs.device_index,
                        reason=f"Fallback from {self._selection.backend.value}",
                    )
                self._finalize_load()
                return
            except (BackendLoadError, BackendError) as exc:
                last_exc = exc
        raise BackendLoadError(
            f"All backends failed for {self._spec.family.value}{self._spec.size.value}. "
            f"Last error: {last_exc}"
        )

    def _finalize_load(self) -> None:
        """Allocate reusable buffers and resolve pipeline workers."""
        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        self._preprocess_buf = PreprocessBuffer(self._batch_size, target_size)
        self._allocate_postprocess_buffer()
        if self._pipeline_workers == 0:
            self._pipeline_workers = 2 if is_free_threaded() else 1
        self._loaded = True
        self._health_state = HealthStatus.READY

    def _allocate_postprocess_buffer(self) -> None:
        """Hook: allocate task-specific postprocess buffer. Default: no-op."""

    def close(self, timeout: float = 5.0) -> None:
        """Gracefully shut down: signal streams, drain event bus, release backend."""
        with self._shutdown_lock:
            if self._health_state == HealthStatus.CLOSED:
                return
            self._shutting_down.set()
            self._health_state = HealthStatus.SHUTTING_DOWN
            for stop_event in list(self._active_streams):
                stop_event.set()
        deadline = time.monotonic() + timeout
        self._streams_drained.wait(timeout=max(0.0, deadline - time.monotonic()))
        self._event_bus.emit("health_change", HealthStatus.SHUTTING_DOWN)
        self._event_bus.close(timeout=max(0.1, deadline - time.monotonic()))
        try:
            if self._loaded:
                self._backend.unload()
        except Exception:
            pass
        finally:
            self._loaded = False
            self._health_state = HealthStatus.CLOSED
            if self._feature_cache is not None:
                self._feature_cache.clear()
            self._preprocess_buf = None
            self._postprocess_buf = None

    def __enter__(self) -> BaseEngine:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> BaseEngine:
        await asyncio.to_thread(self.load)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await asyncio.to_thread(self.close)

    def _infer_from_tensor(
        self,
        tensor: PreprocessedTensor,
        frames: list[Frame],
        *,
        scratch: PostprocessBuffer | None,
    ) -> list[Any]:
        """Run backend inference + task-specific postprocess."""
        if self._feature_cache is not None and frames:
            sid = frames[0].source_id
            if sid:
                self._backend.set_source_id(sid)
        t0 = time.perf_counter()
        raw_output = self._backend.infer(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._metrics.record_inference(elapsed_ms, batch_size=tensor.batch_size, frame_time=t0)
        results = self._process_batch(raw_output, tensor, frames, elapsed_ms, scratch)
        self._event_bus.emit("detection", results)
        return results

    def _run_batch(self, frames: list[Frame]) -> list[Any]:
        """Preprocess frames and run _infer_from_tensor."""
        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        try:
            if self._preprocess_buf is not None and len(frames) <= self._preprocess_buf.capacity:
                tensor = preprocess_into(frames, target_size, self._preprocess_buf)
            else:
                tensor = preprocess(frames, target_size)
            return self._infer_from_tensor(tensor, frames, scratch=self._postprocess_buf)
        except Exception:
            self._metrics.record_error()
            raise

    def _dispatch_frames(self, frames: list[Frame]) -> list[Any]:
        """Route frames through the public inference method.

        Default delegates to ``_run_batch``. Subclasses that expose a public
        batch method override this so tests can patch it.
        """
        return self._run_batch(frames)

    def _stream_dispatch(self, source: FrameSource) -> Iterator[Any]:
        self._backend.clear_kv_cache()
        if not self._prefetch:
            yield from self._stream_sync(source)
        elif source.total_frames == 1:
            yield from self._stream_single(source)
        elif source.is_live:
            yield from self._stream_live(source)
        else:
            yield from self._stream_pipeline(source)

    def _stream_single(self, source: FrameSource) -> Iterator[Any]:
        """Fast path for single-image sources — no threading overhead."""
        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()
        try:
            for frame in source:
                if _stop.is_set():
                    break
                yield from self._dispatch_frames([frame])
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            source.close()

    def _stream_live(self, source: FrameSource) -> Iterator[Any]:
        """Live source path: threaded reader, batch=1, 30 s idle timeout."""
        _MAX_IDLE_S = 30.0
        _POLL_TIMEOUT = 1.0
        if self._batch_size > 1:
            logger.debug(
                "Live source: using batch=1 (configured batch_size=%d ignored)", self._batch_size
            )
        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        reader = ThreadedFrameReader(
            source,
            max_queue_size=self._max_queue_size,
            policy=self._frame_drop_policy,
            preprocess_fn=preprocess,
            target_size=target_size,
        )
        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()
        reader.start()
        try:
            idle_since: float | None = None
            while not _stop.is_set():
                item = reader.get(timeout=_POLL_TIMEOUT)
                if item is not None:
                    idle_since = None
                    assert isinstance(item, PreparedItem)
                    yield from self._infer_from_tensor(
                        item.tensor, [item.frame], scratch=self._postprocess_buf
                    )
                elif reader.is_exhausted:
                    break
                else:
                    now = time.monotonic()
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since >= _MAX_IDLE_S:
                        logger.warning(
                            "Live source idle for %.0fs, terminating stream", now - idle_since
                        )
                        break
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            reader.stop()
            source.close()

    def _stream_pipeline(self, source: FrameSource) -> Iterator[Any]:
        """Offline source: threaded prefetch + pipeline overlap."""
        from collections import deque as Deque
        from concurrent.futures import Future, ThreadPoolExecutor

        from yowo.io._decode import PreprocessBufferPool

        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()
        reader = ThreadedFrameReader(
            source, max_queue_size=self._batch_size * 2, policy=FrameDropPolicy.NONE
        )
        reader.start()
        concurrent = self._pipeline_workers > 1
        infer_lock = threading.Lock() if concurrent else None
        target = (self._model_meta.input_height, self._model_meta.input_width)
        buffer_pool: PreprocessBufferPool | None = None
        if concurrent:
            buffer_pool = PreprocessBufferPool(self._pipeline_workers, self._batch_size, target)

        def _infer_batch(frames: list[Frame]) -> list[Any]:
            if concurrent:
                assert buffer_pool is not None
                buf = buffer_pool.acquire()
                try:
                    tensor = preprocess_into(frames, target, buf)
                    with infer_lock:  # type: ignore[union-attr]
                        return self._infer_from_tensor(tensor, frames, scratch=None)
                except Exception:
                    self._metrics.record_error()
                    raise
                finally:
                    buffer_pool.release(buf)
            return self._run_batch(frames)

        pending: Deque[Future[list[Any]]] = Deque()
        try:
            batch: list[Frame] = []
            max_pending = self._pipeline_workers
            with ThreadPoolExecutor(max_workers=self._pipeline_workers) as pool:
                while not _stop.is_set():
                    raw = reader.get(timeout=5.0)
                    frame: Frame | None = raw  # type: ignore[assignment]
                    if frame is not None:
                        batch.append(frame)
                    if len(batch) >= self._batch_size or (frame is None and batch):
                        if len(pending) >= max_pending:
                            yield from pending.popleft().result()
                        pending.append(pool.submit(_infer_batch, batch))
                        batch = []
                    if frame is None and reader.is_exhausted:
                        break
                while pending:
                    yield from pending.popleft().result()
        finally:
            for fut in pending:
                fut.cancel()
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            reader.stop()
            source.close()

    def _stream_sync(self, source: FrameSource) -> Iterator[Any]:
        """Legacy sequential streaming path (prefetch=False)."""
        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()
        batch: list[Frame] = []
        try:
            for frame in source:
                if _stop.is_set():
                    break
                batch.append(frame)
                if len(batch) >= self._batch_size:
                    yield from self._dispatch_frames(batch)
                    batch.clear()
            if batch and not _stop.is_set():
                yield from self._dispatch_frames(batch)
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            source.close()


# ---------------------------------------------------------------------------
# DetectionEngine — object detection
# ---------------------------------------------------------------------------


class DetectionEngine(BaseEngine):
    """Stateful object detection engine.

    Lifecycle:
      1. __init__: detect hardware, select backend (no model loaded)
      2. load(): resolve weights, load backend, warmup, fallback if needed
      3. detect() / stream(): run inference
      4. close(): release resources

    Context manager support (load on enter, close on exit).

    Custom backends: pass ``backend_instance`` to inject a user-provided
    :class:`InferenceBackend` implementation.  Hardware detection, backend
    selection, and the fallback chain are all bypassed; the caller owns
    device placement and error handling.
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        *,
        backend_instance: InferenceBackend | None = None,
        model_family: ModelFamily = ModelFamily.YOLO26,
        model_size: ModelSize = ModelSize.NANO,
        weights_path: Path | None = None,
        backend: BackendType | None = None,
        device: str = "auto",
        precision: Precision | None = None,
        batch_size: int = 1,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        cache: bool = False,
        cache_dir: Path | None = None,
        kv_cache: bool = False,
        frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST,
        max_queue_size: int = 2,
        prefetch: bool = True,
        pipeline_workers: int = 0,
        metrics_enabled: bool = True,
        error_threshold: int = 10,
    ) -> None:
        if config is not None:
            cfg = config
        else:
            cfg = InferenceConfig(
                model_family=model_family,
                model_size=model_size,
                weights_path=weights_path,
                backend=backend,
                device=device,
                precision=precision,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                cache=cache,
                cache_dir=cache_dir,
                kv_cache=kv_cache,
                frame_drop_policy=frame_drop_policy,
                max_queue_size=max_queue_size,
                prefetch=prefetch,
                pipeline_workers=pipeline_workers,
                metrics_enabled=metrics_enabled,
                error_threshold=error_threshold,
            )
        self._confidence = cfg.confidence_threshold
        self._iou_threshold = cfg.iou_threshold
        spec = ModelSpec(cfg.model_family, cfg.model_size, weights_path=cfg.weights_path)
        super().__init__(
            spec=spec,
            backend_instance=backend_instance,
            backend_override=cfg.backend.value if cfg.backend else None,
            device=cfg.device,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            cache=cfg.cache,
            cache_dir=cfg.cache_dir,
            kv_cache=cfg.kv_cache,
            frame_drop_policy=cfg.frame_drop_policy,
            max_queue_size=cfg.max_queue_size,
            prefetch=cfg.prefetch,
            pipeline_workers=cfg.pipeline_workers,
            metrics_enabled=cfg.metrics_enabled,
            error_threshold=cfg.error_threshold,
        )

    def _allocate_postprocess_buffer(self) -> None:
        self._postprocess_buf = PostprocessBuffer()

    def _dispatch_frames(self, frames: list[Frame]) -> list[Any]:
        """Route through detect() so tests can patch it."""
        return self.detect(frames)  # type: ignore[return-value]

    def _process_batch(
        self,
        raw_output: NDArray[np.float32],
        tensor: PreprocessedTensor,
        frames: list[Frame],
        elapsed_ms: float,
        scratch: PostprocessBuffer | None,
    ) -> list[Detection]:
        return postprocess(
            raw_output,
            tensor,
            frames,
            model_spec=self._spec,
            backend=self._selection.backend,
            confidence_threshold=self._confidence,
            iou_threshold=self._iou_threshold,
            inference_time_ms=elapsed_ms,
            scratch=scratch,
        )

    def detect(self, frames: list[Frame]) -> list[Detection]:
        """Run detection on a list of frames. Returns one Detection per frame."""
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")
        try:
            return self._run_batch(frames)  # type: ignore[return-value]
        except Exception:
            raise

    def _detect_from_tensor(
        self,
        tensor: PreprocessedTensor,
        frames: list[Frame],
        *,
        scratch: PostprocessBuffer | None,
    ) -> list[Detection]:
        """Backward-compatible alias for _infer_from_tensor."""
        return self._infer_from_tensor(tensor, frames, scratch=scratch)  # type: ignore[return-value]

    async def adetect(self, frames: list[Frame]) -> list[Detection]:
        """Async wrapper — offloads detect() to a thread pool."""
        return await asyncio.to_thread(self.detect, frames)

    def stream(self, source: FrameSource) -> Iterator[Detection]:
        """Yield detections from a FrameSource, batching internally."""
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")
        return self._stream_dispatch(source)  # type: ignore[return-value]

    async def astream(self, source: FrameSource) -> AsyncIterator[Detection]:
        """Async stream; stop-event registered in _active_streams for close() signaling."""
        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()
        from yowo._async import astream as _astream

        try:
            async for det in _astream(self.stream, source, self._event_bus.emit, _stop):
                yield det
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()

    def __enter__(self) -> DetectionEngine:
        self.load()
        return self

    async def __aenter__(self) -> DetectionEngine:
        await asyncio.to_thread(self.load)
        return self


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Alias for :class:`DetectionEngine`.
InferenceEngine = DetectionEngine

__all__ = ["BaseEngine", "DetectionEngine", "InferenceEngine"]
