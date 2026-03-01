"""CLIP-ReID ViT-B/16 vehicle re-identification extractor.

Fine-tuned on VeRi-776 training set using CLIP prompt learning (Stage 2).
Produces 1280-dim L2-normalized embeddings optimized for cross-camera vehicle
re-identification. Uses ONNX Runtime for inference.

Reference: "CLIP-ReID: Exploiting Vision-Language Model for Image
Re-identification without Concrete Text Labels" (AAAI 2023)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["CLIPReIDExtractor"]

# CLIP-ReID VeRi preprocessing constants (NOT standard CLIP or ImageNet!)
_CLIPREID_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_CLIPREID_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_CLIPREID_INPUT_SIZE = 256


class CLIPReIDExtractor:
    """CLIP-ReID ViT-B/16 ONNX extractor for vehicle re-identification.

    Fine-tuned on VeRi-776 (576 vehicles, 20 cameras). Produces 1280-dim
    embeddings: concat(CLS_768, CLS_512) from ViT hidden + CLIP projection.

    Preprocessing: resize 256x256, RGB, normalize with mean=0.5, std=0.5.
    """

    __slots__ = (
        "_embedding_dim",
        "_input_name",
        "_input_size",
        "_min_crop_area",
        "_output_name",
        "_session",
    )

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        embedding_dim: int = 1280,
        input_size: int = _CLIPREID_INPUT_SIZE,
        device: str = "cpu",
        min_crop_area: int = 1024,
    ) -> None:
        """Initialize CLIP-ReID extractor with ONNX model.

        Args:
            model_path: Path to CLIP-ReID ONNX file (exported by
                ``tmp/export_clip_reid_onnx.py``).
            embedding_dim: Output dimension (1280 = 768 + 512 for ViT-B/16).
            input_size: Input resolution (256 for CLIP-ReID VeRi).
            device: "cpu", "cuda", or "coreml" for execution provider.
            min_crop_area: Minimum crop area in pixels to extract.

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If model_path does not exist.
        """
        resolved = Path(model_path)
        if not resolved.exists():
            msg = f"CLIP-ReID ONNX model not found: {resolved}"
            raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            msg = (
                "onnxruntime is required for CLIPReIDExtractor."
                " Install with: pip install onnxruntime"
            )
            raise ImportError(msg) from None

        self._embedding_dim = embedding_dim
        self._input_size = input_size
        self._min_crop_area = min_crop_area

        opts = ort.SessionOptions()
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers: list[str]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "coreml":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(resolved),
            sess_options=opts,
            providers=providers,
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings (1280 for ViT-B/16)."""
        return self._embedding_dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Extract L2-normalized CLIP-ReID embeddings for detection crops.

        Args:
            frame_pixels: Full frame, HWC BGR uint8, shape (H, W, 3).
            boxes_xyxy: N bounding boxes as (x1, y1, x2, y2) tuples.

        Returns:
            (N, 1280) float32 L2-normalized embeddings. Rows for filtered
            crops are zero vectors. Returns None if no valid crops.
        """
        if not boxes_xyxy:
            return None

        n = len(boxes_xyxy)
        batch, valid_indices = self._crop_and_preprocess(frame_pixels, boxes_xyxy)
        if batch is None:
            return None

        raw = np.asarray(
            self._session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )

        # L2-normalize
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        np.clip(norms, 1e-8, None, out=norms)
        normalized = raw / norms

        # Map back to full (N, D) — zeros for filtered crops
        result = np.zeros((n, self._embedding_dim), dtype=np.float32)
        for i, vi in enumerate(valid_indices):
            result[vi] = normalized[i]
        return result

    def _crop_and_preprocess(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> tuple[NDArray[np.float32] | None, list[int]]:
        """Crop and preprocess detections for CLIP-ReID inference.

        Pipeline: crop -> BGR->RGB -> resize(256, 256) -> /255 ->
        (x - 0.5) / 0.5 -> HWC->CHW.

        CRITICAL: Uses mean=0.5, std=0.5 (NOT ImageNet or standard CLIP).

        Args:
            frame_pixels: Full frame HWC BGR uint8.
            boxes_xyxy: Bounding boxes as (x1, y1, x2, y2).

        Returns:
            (batch, valid_indices) where batch is (N', 3, 256, 256) float32
            or (None, []) if no valid crops.
        """
        h, w = frame_pixels.shape[:2]
        crops: list[NDArray[np.float32]] = []
        valid_indices: list[int] = []

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(w, int(x2))
            iy2 = min(h, int(y2))

            crop_area = (ix2 - ix1) * (iy2 - iy1)
            if crop_area < self._min_crop_area:
                continue

            crop = frame_pixels[iy1:iy2, ix1:ix2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                crop_rgb,
                (self._input_size, self._input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - _CLIPREID_MEAN) / _CLIPREID_STD
            chw = normalized.transpose(2, 0, 1)
            crops.append(chw)
            valid_indices.append(i)

        if not crops:
            return None, []

        batch = np.stack(crops, axis=0)  # (N', 3, 256, 256)
        return batch, valid_indices
