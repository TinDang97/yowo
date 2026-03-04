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
from collections.abc import Iterator
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
        cfg = config or ClassificationConfig(
            model_family=model_family,
            model_size=model_size,
            weights_path=weights_path,
            backend=backend,
            device=device,
            precision=precision,
            batch_size=batch_size,
            top_k=top_k,
            frame_drop_policy=frame_drop_policy,
            max_queue_size=max_queue_size,
            prefetch=prefetch,
            pipeline_workers=pipeline_workers,
            metrics_enabled=metrics_enabled,
            error_threshold=error_threshold,
        )

        self._top_k = cfg.top_k
        spec = ModelSpec(
            family=cfg.model_family,
            size=cfg.model_size,
            task="classify",
            weights_path=cfg.weights_path,
        )

        super().__init__(
            spec=spec,
            backend_instance=backend_instance,
            backend_override=cfg.backend.value if cfg.backend else None,
            device=cfg.device,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            # Classification models don't support feature cache or kv_cache
            cache=False,
            cache_dir=None,
            kv_cache=False,
            frame_drop_policy=cfg.frame_drop_policy,
            max_queue_size=cfg.max_queue_size,
            prefetch=cfg.prefetch,
            pipeline_workers=cfg.pipeline_workers,
            metrics_enabled=cfg.metrics_enabled,
            error_threshold=cfg.error_threshold,
        )

    @property
    def _result_event_name(self) -> str:
        return "classification"

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

        return self._run_batch(frames)  # type: ignore[return-value]

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


__all__ = ["ClassificationEngine"]
