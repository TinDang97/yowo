"""OBBEngine: YOLO oriented bounding box detection inference engine.

Usage::

    from yowo.obb_engine import OBBEngine
    from yowo.types import ModelFamily, ModelSize

    engine = OBBEngine(
        model_family=ModelFamily.YOLO11,
        model_size=ModelSize.NANO,
    )
    with engine:
        results = engine.detect_obb(frames)
        # results[0].boxes — tuple of OBBBox with cx, cy, w, h, angle
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

# Imported at runtime, not only for typing: `_empty_raw_output` builds the
# failure sentinel with it, and that path runs when a backend dies mid-stream.
import numpy as np

from yowo.backends import InferenceBackend
from yowo.config import OBBConfig
from yowo.engine import BaseEngine, _load_tune_profile  # type: ignore[reportPrivateUsage]
from yowo.errors import InferenceError, ShutdownError, WarmupValidationError
from yowo.hardware import HardwareProfile, get_hardware_profile
from yowo.io import FrameSource
from yowo.postprocess import PostprocessBuffer
from yowo.types import (
    BackendType,
    Frame,
    FrameDropPolicy,
    ModelFamily,
    ModelSize,
    ModelSpec,
    OBBDetection,
    Precision,
    PreprocessedTensor,
)


class OBBEngine(BaseEngine):
    """YOLO OBB (oriented bounding box) detection engine.

    Mirrors :class:`~yowo.classify_engine.ClassificationEngine` but outputs
    :class:`~yowo.types.OBBDetection` with rotation angles instead of
    axis-aligned bounding boxes.

    Default nc=15 for DOTA v1 (yolo11-obb weights).

    Usage::

        engine = OBBEngine(
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.NANO,
        )
        with engine:
            results = engine.detect_obb(frames)
            for det in results:
                for box in det.boxes:
                    print(f"{box.class_name}: {box.confidence:.2f} angle={box.angle:.3f}")

    Args:
        config: Optional :class:`~yowo.config.OBBConfig`. When provided all
            individual kwargs are ignored.
        backend_instance: Optional pre-built backend to inject.
        model_family: YOLO model family (default YOLO11 — only family with OBB).
        model_size: Size variant (default NANO).
        weights_path: Path to a local ``-obb.pt`` file, or ``None`` to use
            the registry default.
        num_classes: Override number of output classes. ``None`` uses registry
            default (15 for DOTA v1).
        backend: Backend type. ``None`` triggers automatic selection.
        device: Device string (default ``"auto"``).
        precision: Numerical precision. ``None`` triggers auto-selection.
        confidence_threshold: Minimum confidence to keep a detection (default 0.25).
        iou_threshold: probiou NMS threshold (default 0.45).
        batch_size: Number of frames per inference batch (default 1).
        frame_drop_policy: Backlog policy for live streaming.
        max_queue_size: Bounded queue depth for ThreadedFrameReader.
        prefetch: Enable threaded frame prefetch in ``stream()``.
        pipeline_workers: Worker thread count. ``0`` = auto-detect.
        metrics_enabled: Enable latency/throughput metrics collection.
        error_threshold: Cumulative errors before health → DEGRADED.
    """

    def __init__(
        self,
        config: OBBConfig | None = None,
        *,
        model_builder: Any | None = None,
        backend_instance: InferenceBackend | None = None,
        model_family: ModelFamily = ModelFamily.YOLO11,
        model_size: ModelSize = ModelSize.NANO,
        weights_path: Path | None = None,
        num_classes: int | None = None,
        backend: BackendType | None = None,
        device: str = "auto",
        precision: Precision | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        batch_size: int = 1,
        frame_drop_policy: FrameDropPolicy = FrameDropPolicy.LATEST,
        max_queue_size: int = 2,
        prefetch: bool = True,
        auto_letterbox: bool = False,
        pipeline_workers: int = 0,
        metrics_enabled: bool = True,
        error_threshold: int = 10,
    ) -> None:
        cfg = config or OBBConfig(
            model_family=model_family,
            model_size=model_size,
            weights_path=weights_path,
            num_classes=num_classes,
            backend=backend,
            device=device,
            precision=precision,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
            frame_drop_policy=frame_drop_policy,
            max_queue_size=max_queue_size,
            prefetch=prefetch,
            auto_letterbox=auto_letterbox,
            pipeline_workers=pipeline_workers,
            metrics_enabled=metrics_enabled,
            error_threshold=error_threshold,
        )

        self._conf = cfg.confidence_threshold
        self._iou = cfg.iou_threshold
        # Store logging config so BaseEngine._finalize_load can wire it
        self._config_log_level = cfg.log_level
        self._config_structured_logging = cfg.structured_logging

        spec = ModelSpec(
            family=cfg.model_family,
            size=cfg.model_size,
            task="obb",
            weights_path=cfg.weights_path,
            num_classes=cfg.num_classes,
        )

        # Apply tune profile (only when user has not explicitly overridden backend/batch/precision
        # and no custom backend instance was provided)
        _hw_cache: HardwareProfile | None = None
        if backend_instance is None:
            _hw_cache = get_hardware_profile()
            cfg = cast(OBBConfig, _load_tune_profile(spec, cast(Any, cfg), _hw_cache))

        super().__init__(
            spec=spec,
            model_builder=model_builder,
            backend_instance=backend_instance,
            backend_override=cfg.backend.value if cfg.backend else None,
            device=cfg.device,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            # OBB models skip feature cache and KV cache (same as classification)
            cache=False,
            cache_dir=None,
            kv_cache=False,
            frame_drop_policy=cfg.frame_drop_policy,
            max_queue_size=cfg.max_queue_size,
            prefetch=cfg.prefetch,
            auto_letterbox=cfg.auto_letterbox,
            pipeline_workers=cfg.pipeline_workers,
            metrics_enabled=cfg.metrics_enabled,
            error_threshold=cfg.error_threshold,
        )

    @property
    def _result_event_name(self) -> str:
        return "obb_detection"

    def _validate_output_values(self, output: np.ndarray) -> None:
        """Validate OBB head output shape and class score range.

        OBB output has shape ``(B, 4+nc+1, A)`` where the last axis is anchors.
        Class scores (channels 4:4+nc) must be in [0, 1] (post-sigmoid).
        """
        nc = self._spec.num_classes or 15
        expected_channels = 4 + nc + 1
        if output.shape[1] != expected_channels:
            raise WarmupValidationError(
                f"OBB output channels {output.shape[1]} != expected {expected_channels} "
                f"(4+{nc}+1). Check num_classes matches checkpoint."
            )
        cls_scores = output[:, 4 : 4 + nc, :]
        if cls_scores.max() > 1.0 + 1e-3 or cls_scores.min() < -1e-3:
            raise WarmupValidationError(
                "OBB class scores outside [0, 1] — sigmoid may have been applied twice"
            )

    def _empty_raw_output(self, batch_size: int) -> np.ndarray[Any, Any]:
        """A failed OBB inference stands in with ZERO anchors.

        ``(B, 4+nc+1, 0)`` is the OBBHead output contract with no instances; the
        base class's ``(B, 0, 6)`` raised ``max(): Expected reduction dim 1 to
        have non-zero size``. Zero anchors rather than one zero-filled anchor:
        the latter yields no boxes only because it falls under the confidence
        threshold, so dropping the threshold would start emitting a phantom box.
        """
        nc = self._spec.num_classes or 15
        return np.zeros((batch_size, 4 + nc + 1, 0), dtype=np.float32)

    def _process_batch(
        self,
        raw_output: np.ndarray[Any, Any],
        tensor: PreprocessedTensor,
        frames: list[Frame],
        elapsed_ms: float,
        scratch: PostprocessBuffer | None,
    ) -> list[OBBDetection]:
        # Both imports are deferred to the one place OBB decoding actually
        # happens. postprocess/__init__.py already routes _obb_nms through a
        # lazy __getattr__ "to avoid top-level import torch"; reaching past it
        # into the private module put torch back on the import path of anyone
        # who merely typed `import yowo`.
        import torch

        from yowo.postprocess import postprocess_obb

        raw_t = torch.from_numpy(raw_output)
        results = postprocess_obb(
            raw_t,
            frames,
            self._spec,
            tensor,
            conf_threshold=self._conf,
            iou_threshold=self._iou,
        )
        # OBBDetection is frozen — reconstruct to attach inference_time_ms
        return [
            OBBDetection(
                frame_index=r.frame_index,
                source_id=r.source_id,
                boxes=r.boxes,
                frame=r.frame,
                inference_time_ms=elapsed_ms,
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Public OBB API
    # ------------------------------------------------------------------

    def detect_obb(self, frames: list[Frame]) -> list[OBBDetection]:
        """Run OBB detection on a list of frames.

        Returns one :class:`~yowo.types.OBBDetection` per frame.

        Args:
            frames: Input frames to run OBB detection on.

        Returns:
            One OBBDetection per frame (may have zero boxes if nothing detected).

        Raises:
            ShutdownError: If the engine is shutting down.
            InferenceError: If the engine has not been loaded.
        """
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")

        return self._run_batch(frames)  # type: ignore[return-value]

    async def adetect_obb(self, frames: list[Frame]) -> list[OBBDetection]:
        """Async wrapper — offloads detect_obb() to a thread pool.

        Args:
            frames: Input frames to run OBB detection on.

        Returns:
            One OBBDetection per frame.
        """
        return await asyncio.to_thread(self.detect_obb, frames)

    def stream(self, source: FrameSource) -> Iterator[OBBDetection]:
        """Yield OBB detections from a FrameSource, batching internally.

        Overrides :meth:`BaseEngine.stream` so that ``astream()`` works.

        Args:
            source: Any :class:`~yowo.io.FrameSource`.

        Yields:
            One :class:`~yowo.types.OBBDetection` per frame.

        Raises:
            ShutdownError: If the engine is shutting down.
            InferenceError: If the engine has not been loaded.
        """
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")
        return self._stream_dispatch(source)  # type: ignore[return-value]

    def stream_obb(self, source: FrameSource) -> Iterator[OBBDetection]:
        """Yield OBB detections from a FrameSource, batching internally.

        Uses the same source-aware dispatch strategy as DetectionEngine:
        - Single image → _stream_single
        - Live source (RTSP/webcam) → _stream_live
        - Offline multi-frame → _stream_pipeline
        - prefetch=False → _stream_sync

        Args:
            source: Any :class:`~yowo.io.FrameSource`.

        Yields:
            One :class:`~yowo.types.OBBDetection` per frame.

        Raises:
            ShutdownError: If the engine is shutting down.
            InferenceError: If the engine has not been loaded.
        """
        if self._shutting_down.is_set():
            raise ShutdownError("Engine is shutting down")
        if not self._loaded:
            raise InferenceError("Engine not loaded. Call load() or use as context manager.")
        return self._stream_dispatch(source)  # type: ignore[return-value]


__all__ = ["OBBEngine"]
