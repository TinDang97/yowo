"""Inference engine orchestrator.

Wires all modules into a single coherent inference pipeline.
Handles backend lifecycle, batch accumulation, and graceful degradation.

Usage::

    # Option A: Pass an InferenceConfig
    from yowo import InferenceConfig, InferenceEngine

    config = InferenceConfig(confidence_threshold=0.35, batch_size=4)
    with InferenceEngine(config) as engine:
        for detection in engine.stream(open_source("video.mp4")):
            process(detection)

    # Option B: Pass individual kwargs (defaults to YOLO26 Nano)
    with InferenceEngine(confidence_threshold=0.35) as engine:
        ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

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


class InferenceEngine:
    """Stateful inference engine.

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

        spec = ModelSpec(cfg.model_family, cfg.model_size, weights_path=cfg.weights_path)

        self._spec = spec
        self._batch_size = cfg.batch_size
        self._confidence = cfg.confidence_threshold
        self._iou_threshold = cfg.iou_threshold
        self._device = cfg.device

        self._feature_cache = None
        if cfg.cache or cfg.cache_dir is not None:
            from yowo.cache import FeatureCache

            self._feature_cache = FeatureCache(cache_dir=cfg.cache_dir)

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
            self._kv_cache = cfg.kv_cache
        else:
            self._hw = get_hardware_profile()
            self._selection = select_backend(
                self._hw,
                model_size=spec.size.value,
                backend_override=cfg.backend.value if cfg.backend else None,
                device_override=cfg.device if cfg.device != "auto" else None,
                precision_override=cfg.precision.value if cfg.precision else None,
            )
            self._kv_cache = cfg.kv_cache
            self._backend = create_backend(
                self._selection.backend,
                self._hw,
                model_spec=self._spec,
                feature_cache=self._feature_cache,
                kv_cache=cfg.kv_cache,
            )

        self._model_meta = _registry_get(spec.family, spec.size)
        self._loaded = False

        self._frame_drop_policy = cfg.frame_drop_policy
        self._max_queue_size = cfg.max_queue_size
        self._prefetch = cfg.prefetch
        self._pipeline_workers = cfg.pipeline_workers
        self._preprocess_buf: PreprocessBuffer | None = None
        self._postprocess_buf: PostprocessBuffer | None = None
        self._metrics = MetricsCollector(enabled=cfg.metrics_enabled)
        self._error_threshold = cfg.error_threshold
        self._health_state: HealthStatus = HealthStatus.STARTING
        self._event_bus = EventBus()
        self._shutting_down = threading.Event()  # set() means shutting down
        self._shutdown_lock = threading.Lock()
        self._active_streams: set[threading.Event] = set()
        self._streams_drained = threading.Event()
        self._streams_drained.set()  # starts as "drained" (no active streams)

    @property
    def selection(self) -> BackendSelection:
        """The resolved backend selection (backend, device, precision)."""
        return self._selection

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def health(self) -> HealthStatus:
        """Derived engine health status.

        Computed from internal state — no active probing:

        - ``CLOSED``: after ``close()`` completes.
        - ``STARTING``: before ``load()`` succeeds.
        - ``DEGRADED``: cumulative errors >= ``error_threshold``,
          or no frames received for >30 s during active streaming.
        - ``READY``: otherwise.
        """
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

    # ------------------------------------------------------------------
    # Event bus delegation
    # ------------------------------------------------------------------

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
            # Direct load — no fallback chain; user owns error handling
            self._backend.load(weights_path, device=self._device)
            self._backend.warmup(batch_size=self._batch_size)
            self._finalize_load()
            return

        # _hw is always set when _user_provided_backend is False
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
                    _fallback_sel = select_backend(
                        self._hw,
                        model_size=self._spec.size.value,
                        backend_override=bt.value,
                    )
                    self._selection = BackendSelection(
                        backend=bt,
                        device_type=_fallback_sel.device_type,
                        precision=_fallback_sel.precision,
                        device_index=_fallback_sel.device_index,
                        reason=f"Fallback from {self._selection.backend.value}",
                    )

                self._finalize_load()
                return
            except (BackendLoadError, BackendError) as exc:
                last_exc = exc
                continue

        raise BackendLoadError(
            f"All backends failed for {self._spec.family.value}{self._spec.size.value}. "
            f"Last error: {last_exc}"
        )

    def _finalize_load(self) -> None:
        """Allocate reusable buffers and resolve pipeline workers."""
        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        self._preprocess_buf = PreprocessBuffer(self._batch_size, target_size)
        self._postprocess_buf = PostprocessBuffer()
        if self._pipeline_workers == 0:
            self._pipeline_workers = 2 if is_free_threaded() else 1
        self._loaded = True
        self._health_state = HealthStatus.READY

    def detect(self, frames: list[Frame]) -> list[Detection]:
        """Run detection on a list of frames. Returns one Detection per frame."""
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")

        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        try:
            if self._preprocess_buf is not None and len(frames) <= self._preprocess_buf.capacity:
                tensor = preprocess_into(frames, target_size, self._preprocess_buf)
            else:
                tensor = preprocess(frames, target_size)
            return self._detect_from_tensor(tensor, frames, scratch=self._postprocess_buf)
        except Exception:
            self._metrics.record_error()
            raise

    def _detect_from_tensor(
        self,
        tensor: PreprocessedTensor,
        frames: list[Frame],
        *,
        scratch: PostprocessBuffer | None,
    ) -> list[Detection]:
        """Run inference + postprocess on an already-preprocessed tensor."""
        # Set source_id for feature caching (PyTorch backend only; no-op on others)
        if self._feature_cache is not None and frames:
            sid = frames[0].source_id
            if sid:
                self._backend.set_source_id(sid)

        t0 = time.perf_counter()
        raw_output = self._backend.infer(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._metrics.record_inference(elapsed_ms, batch_size=tensor.batch_size, frame_time=t0)

        results = postprocess(
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
        # Emit detection event (off hot-path cost: SimpleQueue.put ~1µs)
        self._event_bus.emit("detection", results)
        return results

    async def adetect(self, frames: list[Frame]) -> list[Detection]:
        """Async wrapper — offloads detect() to a thread pool."""
        return await asyncio.to_thread(self.detect, frames)

    def stream(self, source: FrameSource) -> Iterator[Detection]:
        """Yield detections from a FrameSource, batching internally.

        Dispatches to source-type-aware strategy:
        - Single image → _stream_single (no threading overhead)
        - Live source (RTSP/webcam) → _stream_live (threaded reader + frame drop)
        - Offline multi-frame → _stream_pipeline (prefetch + infer overlap)
        - prefetch=False → _stream_sync (legacy sequential path)
        """
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")
        return self._stream_dispatch(source)

    def _stream_dispatch(self, source: FrameSource) -> Iterator[Detection]:
        """Internal generator: clear KV cache then delegate to stream strategy."""
        self._backend.clear_kv_cache()
        if not self._prefetch:
            yield from self._stream_sync(source)
        elif source.total_frames == 1:
            yield from self._stream_single(source)
        elif source.is_live:
            yield from self._stream_live(source)
        else:
            yield from self._stream_pipeline(source)

    async def astream(self, source: FrameSource) -> AsyncIterator[Detection]:
        """Async stream; stop-event registered in _active_streams for close() signaling.

        Note: if the caller creates the generator but never iterates it, the
        stop-event leaks in _active_streams until GC finalizes the async generator.
        close() handles this via its timeout (default 5 s).
        """
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

    def _stream_single(self, source: FrameSource) -> Iterator[Detection]:
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
                yield from self.detect([frame])
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            source.close()

    def _stream_live(self, source: FrameSource) -> Iterator[Detection]:
        """Live source path: threaded reader with frame drop policy, batch=1.

        Always uses batch=1 regardless of configured batch_size to minimise
        latency on live sources.  Terminates after 30 s of consecutive
        timeouts (dead/hung source) to prevent infinite hangs.

        When the backend is loaded the reader thread also preprocesses each
        frame (resize + blobFromImages), overlapping CPU work with inference
        on the main thread.
        """
        _MAX_IDLE_S = 30.0
        _POLL_TIMEOUT = 1.0

        if self._batch_size > 1:
            logger.debug(
                "Live source: using batch=1 for latency (configured batch_size=%d ignored)",
                self._batch_size,
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
                    assert isinstance(item, PreparedItem)  # preprocess_fn always set
                    yield from self._detect_from_tensor(
                        item.tensor,
                        [item.frame],
                        scratch=self._postprocess_buf,
                    )
                elif reader.is_exhausted:
                    break
                else:
                    # Timeout — track consecutive idle time.
                    now = time.monotonic()
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since >= _MAX_IDLE_S:
                        logger.warning(
                            "Live source idle for %.0fs, terminating stream",
                            now - idle_since,
                        )
                        break
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            reader.stop()
            source.close()

    def _stream_pipeline(self, source: FrameSource) -> Iterator[Detection]:
        """Offline source path: threaded prefetch + pipeline overlap.

        Overlaps frame reading with inference via ThreadPoolExecutor.
        When ``pipeline_workers > 1``, a PreprocessBufferPool gives each
        worker its own buffer; a lock serialises backend.infer() calls.
        """
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
            source,
            max_queue_size=self._batch_size * 2,
            policy=FrameDropPolicy.NONE,  # offline: process every frame
        )
        reader.start()

        concurrent = self._pipeline_workers > 1
        infer_lock = threading.Lock() if concurrent else None
        target = (self._model_meta.input_height, self._model_meta.input_width)
        buffer_pool: PreprocessBufferPool | None = None
        if concurrent:
            buffer_pool = PreprocessBufferPool(self._pipeline_workers, self._batch_size, target)

        def _infer_batch(frames: list[Frame]) -> list[Detection]:
            if concurrent:
                assert buffer_pool is not None
                buf = buffer_pool.acquire()
                try:
                    tensor = preprocess_into(frames, target, buf)
                    with infer_lock:  # type: ignore[union-attr]
                        return self._detect_from_tensor(tensor, frames, scratch=None)
                except Exception:
                    self._metrics.record_error()
                    raise
                finally:
                    buffer_pool.release(buf)
            return self.detect(frames)

        pending: Deque[Future[list[Detection]]] = Deque()
        try:
            batch: list[Frame] = []
            max_pending = self._pipeline_workers

            with ThreadPoolExecutor(max_workers=self._pipeline_workers) as pool:
                # Pipeline path never uses preprocess_fn — reader returns Frame | None only.
                # (This is enforced by not passing preprocess_fn= to ThreadedFrameReader above.)
                while not _stop.is_set():
                    raw = reader.get(timeout=5.0)
                    frame: Frame | None = raw  # type: ignore[assignment]  # pipeline reader never uses preprocess_fn
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

    def _stream_sync(self, source: FrameSource) -> Iterator[Detection]:
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
                    yield from self.detect(batch)
                    batch.clear()
            if batch and not _stop.is_set():
                yield from self.detect(batch)
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()
            source.close()

    def close(self, timeout: float = 5.0) -> None:
        """Gracefully shut down: signal streams, drain event bus, release backend."""
        with self._shutdown_lock:
            if self._health_state == HealthStatus.CLOSED:
                return
            self._shutting_down.set()
            self._health_state = HealthStatus.SHUTTING_DOWN
            # Signal streams inside the lock to close the race: any stream that
            # tries to add its stop-event after this point sees _shutting_down=True
            # and raises ShutdownError before adding to _active_streams.
            for stop_event in list(self._active_streams):
                stop_event.set()

        deadline = time.monotonic() + timeout
        self._streams_drained.wait(timeout=max(0.0, deadline - time.monotonic()))
        self._event_bus.emit("health_change", HealthStatus.SHUTTING_DOWN)
        # Drain and close event bus
        self._event_bus.close(timeout=max(0.1, deadline - time.monotonic()))

        # Release backend resources — always update state even if unload() raises.
        try:
            if self._loaded:
                self._backend.unload()
        except Exception:
            pass  # unload errors must not prevent state cleanup
        finally:
            self._loaded = False
            self._health_state = HealthStatus.CLOSED
            if self._feature_cache is not None:
                self._feature_cache.clear()
            self._preprocess_buf = None
            self._postprocess_buf = None

    def __enter__(self) -> InferenceEngine:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> InferenceEngine:
        """Load asynchronously."""
        await asyncio.to_thread(self.load)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close asynchronously."""
        await asyncio.to_thread(self.close)


__all__ = ["InferenceEngine"]
