"""Inference engine orchestrator.

Wires all modules into a single coherent inference pipeline.
Handles backend lifecycle, batch accumulation, and graceful degradation.

Usage::

    spec = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
    with InferenceEngine(spec) as engine:
        for detection in engine.stream(open_source("video.mp4")):
            process(detection)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

from yowo.backends import InferenceBackend, create_backend
from yowo.backends._selector import get_fallback_backends, select_backend
from yowo.errors import BackendError, BackendLoadError, InferenceError
from yowo.hardware import get_hardware_profile
from yowo.io import PreprocessBuffer, ThreadedFrameReader, preprocess, preprocess_into
from yowo.io._source import FrameSource
from yowo.models import resolve_weights
from yowo.models._registry import get as _registry_get
from yowo.postprocess import PostprocessBuffer, postprocess
from yowo.types import (
    BackendSelection,
    BackendType,
    Detection,
    Frame,
    FrameDropPolicy,
    ModelSpec,
    Precision,
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
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        backend: BackendType | None = None,
        device: str = "auto",
        precision: Precision | None = None,
        batch_size: int = 1,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        cache: bool = False,
        cache_dir: Path | None = None,
        kv_cache: bool = False,
        # NEW — backward-compatible additions
        frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST,
        max_queue_size: int = 2,
        prefetch: bool = True,
        pipeline_workers: int = 0,
    ) -> None:
        self._spec = spec
        self._batch_size = batch_size
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._device = device

        # Feature cache (opt-in, PyTorch backend only)
        # cache=True → in-memory; cache_dir → mmap-backed
        self._feature_cache = None
        if cache or cache_dir is not None:
            from yowo.cache import FeatureCache

            self._feature_cache = FeatureCache(cache_dir=cache_dir)

        # Detect hardware once
        self._hw = get_hardware_profile()

        # Select backend (may be overridden by user)
        self._selection: BackendSelection = select_backend(
            self._hw,
            model_size=spec.size.value,
            backend_override=backend.value if backend else None,
            device_override=device if device != "auto" else None,
            precision_override=precision.value if precision else None,
        )

        self._kv_cache = kv_cache
        self._backend: InferenceBackend = create_backend(
            self._selection.backend,
            self._hw,
            model_spec=self._spec,
            feature_cache=self._feature_cache,
            kv_cache=kv_cache,
        )
        self._model_meta = _registry_get(spec.family, spec.size)
        self._loaded = False

        # Streaming pipeline parameters
        self._frame_drop_policy = frame_drop_policy
        self._max_queue_size = max_queue_size
        self._prefetch = prefetch
        self._pipeline_workers = pipeline_workers
        self._preprocess_buf: PreprocessBuffer | None = None
        self._postprocess_buf: PostprocessBuffer | None = None

    @property
    def selection(self) -> BackendSelection:
        """The resolved backend selection (backend, device, precision)."""
        return self._selection

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Resolve weights, load into backend, warmup. Falls back on failure."""
        weights_path = resolve_weights(self._spec)

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
                    # Re-derive device_type from the actual fallback backend
                    # and hardware — not from the (now-failed) original selection.
                    from yowo.backends._selector import select_backend as _select

                    _fallback_sel = _select(
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

                # Allocate reusable buffers for streaming
                target_size = (self._model_meta.input_height, self._model_meta.input_width)
                self._preprocess_buf = PreprocessBuffer(self._batch_size, target_size)
                self._postprocess_buf = PostprocessBuffer()

                # Resolve pipeline worker count
                if self._pipeline_workers == 0:
                    self._pipeline_workers = 2 if is_free_threaded() else 1

                self._loaded = True
                return
            except (BackendLoadError, BackendError) as exc:
                last_exc = exc
                continue

        raise BackendLoadError(
            f"All backends failed for {self._spec.family.value}{self._spec.size.value}. "
            f"Last error: {last_exc}"
        )

    def detect(self, frames: list[Frame]) -> list[Detection]:
        """Run detection on a list of frames. Returns one Detection per frame."""
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")

        target_size = (self._model_meta.input_height, self._model_meta.input_width)
        if self._preprocess_buf is not None and len(frames) <= self._preprocess_buf.capacity:
            tensor = preprocess_into(frames, target_size, self._preprocess_buf)
        else:
            tensor = preprocess(frames, target_size)

        # Set source_id for feature caching (PyTorch backend only)
        if self._feature_cache is not None and frames:
            sid = frames[0].source_id
            if sid and hasattr(self._backend, "set_source_id"):
                self._backend.set_source_id(sid)  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        raw_output = self._backend.infer(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return postprocess(
            raw_output,
            tensor,
            frames,
            model_spec=self._spec,
            backend=self._selection.backend,
            confidence_threshold=self._confidence,
            iou_threshold=self._iou_threshold,
            inference_time_ms=elapsed_ms,
            scratch=self._postprocess_buf,
        )

    def stream(self, source: FrameSource) -> Iterator[Detection]:
        """Yield detections from a FrameSource, batching internally.

        Dispatches to source-type-aware strategy:
        - Single image → _stream_single (no threading overhead)
        - Live source (RTSP/webcam) → _stream_live (threaded reader + frame drop)
        - Offline multi-frame → _stream_pipeline (prefetch + infer overlap)
        - prefetch=False → _stream_sync (legacy sequential path)
        """
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")

        # Reset KV state at the start of each new source
        if hasattr(self._backend, "clear_kv_cache"):
            self._backend.clear_kv_cache()  # type: ignore[attr-defined]

        if not self._prefetch:
            yield from self._stream_sync(source)
        elif source.total_frames == 1:
            yield from self._stream_single(source)
        elif source.is_live:
            yield from self._stream_live(source)
        else:
            yield from self._stream_pipeline(source)

    def _stream_single(self, source: FrameSource) -> Iterator[Detection]:
        """Fast path for single-image sources — no threading overhead."""
        try:
            for frame in source:
                yield from self.detect([frame])
        finally:
            source.close()

    def _stream_live(self, source: FrameSource) -> Iterator[Detection]:
        """Live source path: threaded reader with frame drop policy, batch=1.

        Always uses batch=1 regardless of configured batch_size to minimise
        latency on live sources.
        """
        if self._batch_size > 1:
            logger.debug(
                "Live source: using batch=1 for latency (configured batch_size=%d ignored)",
                self._batch_size,
            )
        reader = ThreadedFrameReader(
            source,
            max_queue_size=self._max_queue_size,
            policy=self._frame_drop_policy,
        )
        reader.start()
        try:
            while True:
                frame = reader.get(timeout=1.0)
                if frame is not None:
                    yield from self.detect([frame])
                elif reader.is_exhausted:
                    break
                # else: timeout — retry
        finally:
            reader.stop()
            source.close()

    def _stream_pipeline(self, source: FrameSource) -> Iterator[Detection]:
        """Offline source path: threaded prefetch + pipeline overlap.

        Uses concurrent.futures.ThreadPoolExecutor to overlap frame reading
        with inference. Maintains up to ``pipeline_workers`` futures in flight
        so that batch N+1 can preprocess while batch N infers.

        On free-threaded Python 3.14t, worker threads achieve true CPU
        parallelism. On GIL Python, inference C++ code releases the GIL
        enabling I/O overlap.

        When ``pipeline_workers > 1``, shared preprocess/postprocess buffers
        are bypassed to avoid data races between concurrent worker threads.
        """
        from collections import deque as Deque
        from concurrent.futures import Future, ThreadPoolExecutor

        reader = ThreadedFrameReader(
            source,
            max_queue_size=self._batch_size * 2,
            policy=FrameDropPolicy.NONE,  # offline: process every frame
        )
        reader.start()

        concurrent = self._pipeline_workers > 1

        def _infer_batch(frames: list[Frame]) -> list[Detection]:
            if concurrent:
                # Per-call allocation — no shared buffers across workers.
                target_size = (self._model_meta.input_height, self._model_meta.input_width)
                tensor = preprocess(frames, target_size)
                t0 = time.perf_counter()
                raw_output = self._backend.infer(tensor)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return postprocess(
                    raw_output,
                    tensor,
                    frames,
                    model_spec=self._spec,
                    backend=self._selection.backend,
                    confidence_threshold=self._confidence,
                    iou_threshold=self._iou_threshold,
                    inference_time_ms=elapsed_ms,
                )
            return self.detect(frames)

        try:
            pending: Deque[Future[list[Detection]]] = Deque()
            batch: list[Frame] = []
            max_pending = self._pipeline_workers

            with ThreadPoolExecutor(max_workers=self._pipeline_workers) as pool:
                while True:
                    frame = reader.get(timeout=5.0)
                    if frame is not None:
                        batch.append(frame)

                    if len(batch) >= self._batch_size or (frame is None and batch):
                        # Drain oldest future if pipeline is full.
                        if len(pending) >= max_pending:
                            yield from pending.popleft().result()
                        pending.append(pool.submit(_infer_batch, batch.copy()))
                        batch.clear()

                    if frame is None and reader.is_exhausted:
                        break

                # Drain remaining futures in submission order.
                while pending:
                    yield from pending.popleft().result()
        finally:
            reader.stop()
            source.close()

    def _stream_sync(self, source: FrameSource) -> Iterator[Detection]:
        """Legacy sequential streaming path (prefetch=False)."""
        batch: list[Frame] = []
        try:
            for frame in source:
                batch.append(frame)
                if len(batch) >= self._batch_size:
                    yield from self.detect(batch)
                    batch.clear()
            if batch:
                yield from self.detect(batch)
        finally:
            source.close()

    def close(self) -> None:
        """Release all backend resources."""
        self._backend.unload()
        if self._feature_cache is not None:
            self._feature_cache.clear()
        self._preprocess_buf = None
        self._postprocess_buf = None
        self._loaded = False

    def __enter__(self) -> InferenceEngine:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["InferenceEngine"]
