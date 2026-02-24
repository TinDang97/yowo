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
from yowo.io._decode import preprocess
from yowo.io._source import FrameSource
from yowo.models import resolve_weights
from yowo.models._registry import get as _registry_get
from yowo.postprocess._nms import postprocess
from yowo.types import BackendSelection, BackendType, Detection, Frame, ModelSpec, Precision

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
        )

    def stream(self, source: FrameSource) -> Iterator[Detection]:
        """Yield detections from a FrameSource, batching internally."""
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")

        # Reset KV state at the start of each new source
        if hasattr(self._backend, "clear_kv_cache"):
            self._backend.clear_kv_cache()

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
        self._loaded = False

    def __enter__(self) -> InferenceEngine:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["InferenceEngine"]
