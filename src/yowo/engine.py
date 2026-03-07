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
import contextlib
import dataclasses
import logging
import threading
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from yowo.hardware import HardwareProfile

import numpy as np
from numpy.typing import NDArray

from yowo._streaming import StreamingMixin
from yowo.backends import (
    InferenceBackend,
    create_backend,
    get_fallback_backends,
    select_backend,
)
from yowo.config import InferenceConfig
from yowo.errors import (
    BackendError,
    BackendLoadError,
    ConfigError,
    InferenceError,
    ModelNotFoundError,
    ShutdownError,
    WarmupValidationError,
)
from yowo.events import EventBus
from yowo.hardware import get_hardware_profile
from yowo.io import (
    FrameSource,
    PreprocessBuffer,
    preprocess,
    preprocess_into,
)
from yowo.metrics import EngineMetrics, MetricsCollector
from yowo.models import get as _registry_get
from yowo.models import resolve_weights
from yowo.models._registry import ModelMeta
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
# HealthReport dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Immutable snapshot of engine health at a point in time.

    Attributes:
        status: Current :class:`~yowo.types.HealthStatus` value.
        uptime_s: Seconds since metrics collector was created or last reset.
        errors_total: Total inference errors since last reset.
        frames_total: Total frames processed since last reset.
        memory_pct: GPU memory reserved / total (0.0-1.0).  ``None`` on CPU.
        stream_count: Number of currently active streams.
        batch_size_current: Active batch size (may have been reduced by OOM recovery).
        precision_current: String representation of the current precision (e.g. ``"fp32"``).
    """

    status: HealthStatus
    uptime_s: float
    errors_total: int
    frames_total: int
    memory_pct: float | None
    stream_count: int
    batch_size_current: int
    precision_current: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict with ``status`` as its string value."""
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# OOM monitor thresholds
# ---------------------------------------------------------------------------

_OOM_TIER1 = 0.80  # halve batch size
_OOM_TIER2 = 0.90  # attempt precision fallback
_OOM_TIER3 = 0.95  # evict lowest-activity streams
_OOM_CLEAR = 0.75  # restore batch size when below this


def _load_tune_profile(
    spec: ModelSpec,
    config: InferenceConfig,
    hw: HardwareProfile,
) -> InferenceConfig:
    """Apply saved tune profile to config when caller has not explicitly set values.

    Profile values are applied ONLY when config fields are at their defaults:
    - ``config.backend is None`` → apply ``profile.backend``
    - ``config.batch_size == 1`` → apply ``profile.batch_size``
    - ``config.precision is None`` → apply ``profile.precision``

    Uses ``dataclasses.replace()`` — never mutates config in-place.

    Args:
        spec: Model spec (used to derive model name for profile lookup).
        config: Resolved inference config (may be partially populated).
        hw: Current hardware profile for fingerprint computation.

    Returns:
        Updated :class:`InferenceConfig` with profile values applied, or the
        original config unchanged when no profile exists.
    """
    from yowo.tune._profile import load_profile

    model_name = f"{spec.family.value}{spec.size.value}"
    profile = load_profile(model_name, hw)
    if profile is None:
        return config

    updates: dict[str, object] = {}
    if config.backend is None:
        updates["backend"] = BackendType(profile.backend)
    if config.batch_size == 1:
        updates["batch_size"] = profile.batch_size
    if config.precision is None:
        updates["precision"] = Precision(profile.precision)

    if not updates:
        return config

    logger.debug(
        "Loaded tune profile: backend=%s batch=%d precision=%s fps=%.1f",
        profile.backend,
        profile.batch_size,
        profile.precision,
        profile.fps_achieved,
    )
    return dataclasses.replace(config, **updates)


def _resolve_model_meta(spec: ModelSpec, model_builder: Any | None) -> ModelMeta:
    """Resolve model metadata from registry, with custom builder fallback."""
    try:
        meta = (
            _registry_get_cls(spec.family, spec.size)
            if spec.task == "classify"
            else _registry_get(spec.family, spec.size)
        )
    except ModelNotFoundError:
        if model_builder is None:
            raise
        h, w = model_builder.input_shape
        return ModelMeta(
            family=spec.family,
            size=spec.size,
            input_height=h,
            input_width=w,
            num_classes=spec.num_classes or 0,
            weight_stem="custom",
            default_weights_url="",
        )
    if spec.num_classes is not None:
        from dataclasses import replace as _dc_replace

        meta = _dc_replace(meta, num_classes=spec.num_classes)
    return meta


# ---------------------------------------------------------------------------
# BaseEngine — shared lifecycle, streaming, metrics
# ---------------------------------------------------------------------------


class BaseEngine(StreamingMixin):
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
        model_builder: Any | None = None,
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
        auto_letterbox: bool = False,
        pipeline_workers: int = 0,
        metrics_enabled: bool = True,
        error_threshold: int = 10,
        _hw_cache: HardwareProfile | None = None,
    ) -> None:
        self._spec = spec
        self._model_builder = model_builder
        self._batch_size = batch_size
        self._device = device
        self._auto_letterbox = auto_letterbox
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
            self._hw = _hw_cache if _hw_cache is not None else get_hardware_profile()
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
                model_builder=self._model_builder,
                feature_cache=self._feature_cache,
                kv_cache=kv_cache,
            )
        self._model_meta = _resolve_model_meta(spec, model_builder)
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
        self._infer_lock = threading.Lock()
        self._shutting_down = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._active_streams: set[threading.Event] = set()
        self._streams_drained = threading.Event()
        self._streams_drained.set()
        # OOM monitor state (initialised here; monitor only starts on CUDA load)
        self._oom_stop: threading.Event = threading.Event()
        self._oom_thread: threading.Thread | None = None
        self._oom_recovering: bool = False
        self._original_batch_size: int = batch_size

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
    def _result_event_name(self) -> str:
        """Event name emitted after each inference batch.

        Subclasses override to emit task-specific events (e.g.
        ``EVENT_CLASSIFICATION`` for classification engines).
        """
        return "detection"

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

    def health_report(self) -> HealthReport:
        """Return a frozen :class:`HealthReport` snapshot of current engine state.

        ``memory_pct`` is ``None`` unless the engine is running on a CUDA device.
        GPU memory reads are best-effort; any failure returns ``None``.
        """
        snap = self._metrics.snapshot()

        memory_pct: float | None = None
        if self._is_cuda:
            try:
                import torch

                device_index = self._selection.device_index
                reserved = torch.cuda.memory_reserved(device_index)
                total = torch.cuda.get_device_properties(device_index).total_memory
                if total > 0:
                    memory_pct = reserved / total
            except Exception:
                memory_pct = None

        return HealthReport(
            status=self.health,
            uptime_s=snap.uptime_s,
            errors_total=snap.errors_total,
            frames_total=snap.frames_total,
            memory_pct=memory_pct,
            stream_count=len(self._active_streams),
            batch_size_current=self._batch_size,
            precision_current=self._selection.precision.value,
        )

    def export_metrics(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of current engine metrics.

        Keys correspond to :class:`~yowo.metrics.EngineMetrics` field names.
        """
        return dataclasses.asdict(self._metrics.snapshot())

    def export_metrics_prometheus(self) -> str:
        """Return a Prometheus text exposition string for current metrics.

        Delegates to :meth:`~yowo.metrics.MetricsCollector.export_prometheus`.
        Always ends with a newline.
        """
        return self._metrics.export_prometheus()

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
        if self._model_builder is not None and self._spec.weights_path is None:
            weights_path = Path("")  # builder handles weights internally
        else:
            weights_path = resolve_weights(self._spec)
        if self._user_provided_backend:
            self._backend.load(weights_path, device=self._device)
            self._backend.warmup(batch_size=self._batch_size)
            self._validate_warmup_output()
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
                        model_builder=self._model_builder,
                        feature_cache=self._feature_cache,
                        kv_cache=self._kv_cache,
                    )
                self._backend.load(weights_path, device=self._device)
                self._backend.warmup(batch_size=self._batch_size)
                self._validate_warmup_output()
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
        if self._auto_letterbox and self._backend.backend_type != BackendType.PYTORCH:
            with contextlib.suppress(Exception):
                self._backend.unload()
            raise ConfigError(
                f"auto_letterbox requires a backend with dynamic spatial input shapes. "
                f"Backend '{self._backend.backend_type.value}' has a fixed compiled input "
                f"shape and will reject tensors with non-square dimensions at inference time. "
                f"Use BackendType.PYTORCH or disable auto_letterbox."
            )
        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        if not self._auto_letterbox:
            self._preprocess_buf = PreprocessBuffer(self._batch_size, target_size)
        self._allocate_postprocess_buffer()
        if self._pipeline_workers == 0:
            self._pipeline_workers = 2 if is_free_threaded() else 1
        self._loaded = True
        self._health_state = HealthStatus.READY
        self._start_oom_monitor()
        # Wire structured logging after load completes.
        # Use best-effort config read — config may not always be accessible.
        try:
            from yowo.logging import configure_logging as _configure_logging

            log_level: str = getattr(self, "_config_log_level", "WARNING")
            structured: bool = getattr(self, "_config_structured_logging", False)
            _configure_logging(level=log_level, structured=structured)
        except Exception:
            pass

    def _allocate_postprocess_buffer(self) -> None:
        """Hook: allocate task-specific postprocess buffer. Default: no-op."""

    def _validate_warmup_output(self) -> None:
        """Run a dummy inference and validate output shape and values.

        Called during :meth:`load` after ``backend.warmup()`` to catch
        corrupt or incompatible models before the engine enters READY state.

        Raises:
            WarmupValidationError: If output has unexpected shape or values.
        """
        h, w = self._model_meta.input_height, self._model_meta.input_width
        dummy = PreprocessedTensor(
            data=np.zeros((1, 3, h, w), dtype=np.float32),
            original_shapes=((h, w),),
            input_shape=(h, w),
            scale_factors=((1.0, 1.0),),
            pad_offsets=((0, 0),),
        )
        try:
            output = self._backend.infer(dummy)
        except Exception as exc:
            raise WarmupValidationError(
                f"backend.infer() raised {type(exc).__name__}: {exc}"
            ) from exc
        if output.ndim < 2:
            raise WarmupValidationError(f"expected output ndim >= 2, got shape {output.shape}")
        self._validate_output_values(output)

    def _validate_output_values(self, output: NDArray[np.float32]) -> None:
        """Validate output value ranges. Subclasses override for task-specific checks."""

    # ---------------------------------------------------------------------------
    # OOM monitor daemon
    # ---------------------------------------------------------------------------

    @property
    def _is_cuda(self) -> bool:
        """True when the active backend is running on a CUDA device."""
        return self._selection.device_type == DeviceType.CUDA

    def _start_oom_monitor(self) -> None:
        """Start the OOM monitor daemon thread (CUDA backends only).

        No-op for CPU, ONNX-CPU, CoreML, and other non-CUDA backends.
        """
        if not self._is_cuda:
            return
        self._oom_stop.clear()
        self._oom_thread = threading.Thread(
            target=self._oom_monitor_loop,
            name="yowo-oom-monitor",
            daemon=True,
        )
        self._oom_thread.start()

    def _oom_monitor_loop(self) -> None:
        """Poll GPU memory utilisation every 5 s and apply recovery as needed.

        Uses Event.wait() so the daemon exits quickly when _oom_stop is set.
        GPU memory reads are thread-safe (no _infer_lock needed per RESEARCH.md).
        """
        try:
            import torch
        except ImportError:
            logger.warning("OOM monitor: torch not available, exiting")
            return

        device_index = self._selection.device_index
        while not self._oom_stop.wait(timeout=5.0):
            try:
                reserved = torch.cuda.memory_reserved(device_index)
                total = torch.cuda.get_device_properties(device_index).total_memory
                if total > 0:
                    pct = reserved / total
                    self._apply_oom_recovery(pct)
            except Exception as exc:
                logger.warning("OOM monitor error: %s", exc)

    def _apply_oom_recovery(self, pct: float) -> None:
        """Apply the three-tier recovery ladder based on *pct* GPU utilisation.

        Tiers (applied top-down, only one action per call):
          - >= _OOM_TIER3 (0.95): evict lowest-activity streams
          - >= _OOM_TIER2 (0.90): attempt precision fallback
          - >= _OOM_TIER1 (0.80): halve batch size
          - <  _OOM_CLEAR (0.75): restore batch size if recovering
        """
        if pct < _OOM_CLEAR:
            if self._oom_recovering:
                # Recovery complete — restore original batch size
                self._batch_size = self._original_batch_size
                if self._preprocess_buf is not None:
                    self._preprocess_buf = PreprocessBuffer(
                        self._batch_size,
                        (self._model_meta.input_height, self._model_meta.input_width),
                    )
                self._oom_recovering = False
                self._health_state = HealthStatus.READY
                self._event_bus.emit("health_change", HealthStatus.READY)
                logger.info(
                    "OOM monitor: utilisation %.1f%% — batch size restored to %d",
                    pct * 100,
                    self._batch_size,
                )
            return

        if pct >= _OOM_TIER3:
            self._evict_lowest_activity_streams()
            return

        if pct >= _OOM_TIER2:
            self._try_precision_fallback()
            return

        if pct >= _OOM_TIER1:
            self._halve_batch_size()

    def _halve_batch_size(self) -> None:
        """Tier-1 recovery: halve batch size and reallocate PreprocessBuffer."""
        if not self._oom_recovering:
            # Save original on the first reduction only
            self._original_batch_size = self._batch_size
        self._batch_size = max(1, self._batch_size // 2)
        if self._preprocess_buf is not None:
            self._preprocess_buf = PreprocessBuffer(
                self._batch_size,
                (self._model_meta.input_height, self._model_meta.input_width),
            )
        self._oom_recovering = True
        self._health_state = HealthStatus.DEGRADED
        self._event_bus.emit("health_change", HealthStatus.DEGRADED)
        logger.warning(
            "OOM monitor: GPU %.1f%% — batch size halved to %d",
            0.0,
            self._batch_size,
        )

    def _try_precision_fallback(self) -> None:
        """Tier-2 recovery: attempt FP16 precision fallback if backend supports it."""
        if hasattr(self._backend, "set_precision"):
            try:
                self._backend.set_precision("fp16")  # type: ignore[attr-defined]
                logger.warning("OOM monitor: switched backend to FP16 precision")
            except Exception as exc:
                logger.warning("OOM monitor: precision fallback failed: %s", exc)
        else:
            logger.debug("OOM monitor: precision fallback not supported by this backend")
        self._health_state = HealthStatus.DEGRADED

    def _evict_lowest_activity_streams(self) -> None:
        """Tier-3 recovery: evict lowest-activity FrameCollector streams.

        BaseEngine stub — no FrameCollector reference held at this layer.
        Subclasses that manage a pipeline with a FrameCollector should override.
        """
        logger.warning("OOM monitor: tier-3 eviction triggered but no stream collector attached")

    def close(self, timeout: float = 5.0) -> None:
        """Gracefully shut down: signal streams, drain event bus, release backend."""
        with self._shutdown_lock:
            if self._health_state == HealthStatus.CLOSED:
                return
            self._shutting_down.set()
            self._health_state = HealthStatus.SHUTTING_DOWN
            for stop_event in list(self._active_streams):
                stop_event.set()
        # Signal OOM monitor daemon to exit (no join needed — daemon=True)
        if hasattr(self, "_oom_stop"):
            self._oom_stop.set()
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

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        await asyncio.to_thread(self.load)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await asyncio.to_thread(self.close)

    def stream(self, source: FrameSource) -> Iterator[Any]:
        """Yield results from a FrameSource. Implemented by subclasses."""
        raise NotImplementedError

    async def astream(self, source: FrameSource) -> AsyncIterator[Any]:
        """Async stream results from a FrameSource.

        Delegates to :func:`yowo._async.astream` for thread-safe async bridging.
        Stop-event is registered in _active_streams for close() signaling.

        Args:
            source: Any FrameSource (image, video, RTSP, …).

        Yields:
            One result per frame (Detection or ClassificationResult depending on engine).

        Raises:
            ShutdownError: If the engine is shutting down.
        """
        _stop = threading.Event()
        with self._shutdown_lock:
            if self._shutting_down.is_set():
                raise ShutdownError("Engine is shutting down")
            self._active_streams.add(_stop)
            self._streams_drained.clear()
        from yowo._async import astream as _astream

        try:
            async for result in _astream(
                self.stream,
                source,
                self._event_bus.emit,
                _stop,
                self._metrics.record_frame_dropped,
            ):
                yield result
        finally:
            self._active_streams.discard(_stop)
            if not self._active_streams:
                self._streams_drained.set()

    # Retry delays for _infer_with_retry: 100ms, 200ms, 400ms
    _RETRY_DELAYS = (0.1, 0.2, 0.4)

    def _infer_with_retry(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """Call backend.infer() with exponential backoff retry.

        Retries up to 3 times on any exception (100ms, 200ms, 400ms backoff).
        On exhaustion: records error, emits "error" event, returns an empty
        result array so the engine does not crash.

        Must be called with ``_infer_lock`` already held (called from _run_gpu).

        Returns:
            Raw model output or zeros on exhaustion.
        """
        delays = self._RETRY_DELAYS
        last_exc: Exception | None = None
        for attempt, delay in enumerate(delays):
            try:
                return self._backend.infer(tensor)
            except Exception as exc:
                last_exc = exc
                if attempt < len(delays) - 1:
                    logger.warning(
                        "backend.infer() attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1,
                        len(delays),
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    # All attempts exhausted
                    logger.warning(
                        "backend.infer() failed after %d attempts: %s — returning empty result",
                        len(delays),
                        exc,
                    )
                    self._metrics.record_error()
                    self._event_bus.emit("error", last_exc)
        return np.zeros((1, 0, 6), dtype=np.float32)

    def _run_gpu(
        self,
        tensor: PreprocessedTensor,
        frames: list[Frame],
    ) -> tuple[NDArray[np.float32], float]:
        """GPU-only: set_source_id + backend.infer + record metrics.

        Acquires ``_infer_lock`` internally to serialise GPU access.
        Postprocessing (``_process_batch``) and event emission happen
        *outside* the lock so that NMS work does not block the GPU for
        the next batch.
        """
        with self._infer_lock:
            if self._feature_cache is not None and frames:
                sid = frames[0].source_id
                if sid:
                    self._backend.set_source_id(sid)
            t0 = time.perf_counter()
            raw_output = self._infer_with_retry(tensor)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._metrics.record_inference(elapsed_ms, batch_size=tensor.batch_size, frame_time=t0)
        return raw_output, elapsed_ms

    def _postprocess_and_emit(
        self,
        raw_output: NDArray[np.float32],
        tensor: PreprocessedTensor,
        frames: list[Frame],
        elapsed_ms: float,
        scratch: PostprocessBuffer | None,
    ) -> list[Any]:
        """Run task-specific postprocess and emit result event."""
        results = self._process_batch(raw_output, tensor, frames, elapsed_ms, scratch)
        self._event_bus.emit(self._result_event_name, results)
        return results

    def _infer_from_tensor(
        self,
        tensor: PreprocessedTensor,
        frames: list[Frame],
        *,
        scratch: PostprocessBuffer | None,
    ) -> list[Any]:
        """Run backend inference + task-specific postprocess."""
        raw_output, elapsed_ms = self._run_gpu(tensor, frames)
        return self._postprocess_and_emit(raw_output, tensor, frames, elapsed_ms, scratch)

    def _run_batch(self, frames: list[Frame]) -> list[Any]:
        """Preprocess frames and run _infer_from_tensor."""
        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        try:
            if (
                not self._auto_letterbox
                and self._preprocess_buf is not None
                and len(frames) <= self._preprocess_buf.capacity
            ):
                tensor = preprocess_into(frames, target_size, self._preprocess_buf)
            else:
                tensor = preprocess(frames, target_size, auto_letterbox=self._auto_letterbox)
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

    # Streaming strategies (_stream_dispatch, _stream_single, _stream_live,
    # _stream_pipeline, _stream_sync) are inherited from StreamingMixin.


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
        model_builder: Any | None = None,
        backend_instance: InferenceBackend | None = None,
        model_family: ModelFamily = ModelFamily.YOLO26,
        model_size: ModelSize = ModelSize.NANO,
        weights_path: Path | None = None,
        num_classes: int | None = None,
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
        auto_letterbox: bool = False,
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
                num_classes=num_classes,
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
                auto_letterbox=auto_letterbox,
                pipeline_workers=pipeline_workers,
                metrics_enabled=metrics_enabled,
                error_threshold=error_threshold,
            )
        self._confidence = cfg.confidence_threshold
        self._iou_threshold = cfg.iou_threshold
        self._config_log_level: str = getattr(cfg, "log_level", "WARNING")
        self._config_structured_logging: bool = getattr(cfg, "structured_logging", False)
        spec = ModelSpec(
            cfg.model_family,
            cfg.model_size,
            weights_path=cfg.weights_path,
            num_classes=cfg.num_classes,
        )
        # Apply tune profile (only when user has not explicitly overridden backend/batch/precision
        # and no custom backend instance was provided)
        _hw_cache: HardwareProfile | None = None
        if backend_instance is None:
            _hw_cache = get_hardware_profile()
            cfg = _load_tune_profile(spec, cfg, _hw_cache)
        super().__init__(
            spec=spec,
            model_builder=model_builder,
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
            auto_letterbox=cfg.auto_letterbox,
            pipeline_workers=cfg.pipeline_workers,
            metrics_enabled=cfg.metrics_enabled,
            error_threshold=cfg.error_threshold,
            _hw_cache=_hw_cache,
        )

    def _allocate_postprocess_buffer(self) -> None:
        self._postprocess_buf = PostprocessBuffer()

    def _validate_output_values(self, output: NDArray[np.float32]) -> None:
        """Check detection output for NaN/Inf (raw logits — no [0,1] range check)."""
        if np.any(np.isnan(output)):
            raise WarmupValidationError("detection output contains NaN values")
        if np.any(np.isinf(output)):
            raise WarmupValidationError("detection output contains Inf values")

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
        return self._run_batch(frames)  # type: ignore[return-value]

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


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Alias for :class:`DetectionEngine`.
InferenceEngine = DetectionEngine

__all__ = ["BaseEngine", "DetectionEngine", "HealthReport", "InferenceEngine"]
