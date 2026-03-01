"""ReID feature extraction for appearance-based track association."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CLIPExtractor",
    "CLIPReIDExtractor",
    "FastReIDExtractor",
    "ReIDExtractor",
    "VehicleReIDExtractor",
]

from yowo.tracking._clip_reid import CLIPReIDExtractor as CLIPReIDExtractor

# CLIP normalization constants (OpenAI CLIP / OpenCLIP convention)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ImageNet normalization constants (FastReID / torchvision convention)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@runtime_checkable
class ReIDExtractor(Protocol):
    """Protocol for pluggable ReID feature extractors.

    Any class implementing this protocol can be passed to ByteTracker
    as the reid_extractor parameter. Enables swapping CLIP for FastReID,
    a fine-tuned model, or a mock extractor in tests.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings (e.g. 512 for CLIP ViT-B/16)."""
        ...

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Extract L2-normalized appearance embeddings for detection crops.

        Args:
            frame_pixels: Full frame, HWC BGR uint8, shape (H, W, 3).
            boxes_xyxy: N bounding boxes as (x1, y1, x2, y2) tuples.

        Returns:
            (N, D) float32 L2-normalized embeddings. Rows for filtered crops
            (e.g. too small) are zero vectors. Returns None if no valid crops.
        """
        ...


class CLIPExtractor:
    """CLIP ViT-B/16 visual encoder for zero-shot ReID feature extraction.

    Loads a CLIP visual encoder ONNX model and provides batched embedding
    extraction from detection crops. Uses ONNX Runtime — same infrastructure
    as yowo's YOLO backends.
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
        embedding_dim: int = 512,
        input_size: int = 224,
        device: str = "cpu",
        min_crop_area: int = 1024,
    ) -> None:
        """Initialize CLIP extractor with ONNX model.

        Args:
            model_path: Path to CLIP visual encoder ONNX file.
            embedding_dim: Expected output dimension (512 for ViT-B/16).
            input_size: Input resolution (224 for ViT-B/16).
            device: "cpu", "cuda", or "coreml" for EP selection.
            min_crop_area: Minimum crop area in pixels to extract (skip tiny crops).

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If model_path does not exist.
        """
        resolved = Path(model_path)
        if not resolved.exists():
            msg = f"CLIP ONNX model not found: {resolved}"
            raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            msg = "onnxruntime is required for CLIPExtractor. Install with: pip install onnxruntime"
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
        """Dimensionality of output embeddings."""
        return self._embedding_dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Extract L2-normalized CLIP embeddings for detection crops.

        Args:
            frame_pixels: Full frame, HWC BGR uint8, shape (H, W, 3).
            boxes_xyxy: N bounding boxes as (x1, y1, x2, y2) tuples.

        Returns:
            (N, D) float32 embeddings with zero rows for filtered crops.
            None if no valid crops remain.
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
        """Crop and preprocess detections for CLIP inference.

        Args:
            frame_pixels: Full frame HWC BGR uint8.
            boxes_xyxy: Bounding boxes as (x1, y1, x2, y2).

        Returns:
            (batch, valid_indices) where batch is (N', 3, H, W) float32
            or (None, []) if no valid crops.
        """
        h, w = frame_pixels.shape[:2]
        crops: list[NDArray[np.float32]] = []
        valid_indices: list[int] = []

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            # Clip to frame bounds
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(w, int(x2))
            iy2 = min(h, int(y2))

            crop_area = (ix2 - ix1) * (iy2 - iy1)
            if crop_area < self._min_crop_area:
                continue

            crop = frame_pixels[iy1:iy2, ix1:ix2]
            # BGR -> RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Resize to input_size
            resized = cv2.resize(
                crop_rgb,
                (self._input_size, self._input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            # Normalize: /255, subtract mean, divide std
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - _CLIP_MEAN) / _CLIP_STD
            # HWC -> CHW
            chw = normalized.transpose(2, 0, 1)
            crops.append(chw)
            valid_indices.append(i)

        if not crops:
            return None, []

        batch = np.stack(crops, axis=0)  # (N', 3, H, W)
        return batch, valid_indices


class FastReIDExtractor:
    """FastReID SBS-S50 ONNX-based embedding extractor for person ReID.

    Input: (batch, 3, 256, 128) — portrait 2:1 aspect, ImageNet normalization.
    Output: (batch, 256) L2-normalized embeddings.

    Uses the same ORT session pattern as CLIPExtractor. Designed for
    person re-identification — less effective on non-person objects.
    """

    __slots__ = (
        "_embedding_dim",
        "_input_name",
        "_input_size_hw",
        "_min_crop_area",
        "_output_name",
        "_session",
    )

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        embedding_dim: int = 256,
        input_size: tuple[int, int] = (256, 128),
        device: str = "cpu",
        min_crop_area: int = 1024,
    ) -> None:
        """Initialize FastReID extractor with ONNX model.

        Args:
            model_path: Path to FastReID SBS-S50 ONNX file.
            embedding_dim: Expected output dimension (256 for SBS-S50).
            input_size: Input resolution as (height, width). Default (256, 128).
            device: "cpu", "cuda", or "coreml" for EP selection.
            min_crop_area: Minimum crop area in pixels to extract.

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If model_path does not exist.
        """
        resolved = Path(model_path)
        if not resolved.exists():
            msg = f"FastReID ONNX model not found: {resolved}"
            raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            msg = (
                "onnxruntime is required for FastReIDExtractor."
                " Install with: pip install onnxruntime"
            )
            raise ImportError(msg) from None

        self._embedding_dim = embedding_dim
        self._input_size_hw = input_size
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
        """Dimensionality of output embeddings."""
        return self._embedding_dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Extract L2-normalized FastReID embeddings for detection crops.

        Args:
            frame_pixels: Full frame, HWC BGR uint8, shape (H, W, 3).
            boxes_xyxy: N bounding boxes as (x1, y1, x2, y2) tuples.

        Returns:
            (N, D) float32 embeddings with zero rows for filtered crops.
            None if no valid crops remain.
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
        """Crop and preprocess detections for FastReID inference.

        Pipeline: crop -> BGR->RGB -> resize(128, 256) -> /255 -> ImageNet normalize -> HWC->CHW

        Args:
            frame_pixels: Full frame HWC BGR uint8.
            boxes_xyxy: Bounding boxes as (x1, y1, x2, y2).

        Returns:
            (batch, valid_indices) where batch is (N', 3, 256, 128) float32
            or (None, []) if no valid crops.
        """
        fh, fw = frame_pixels.shape[:2]
        target_h, target_w = self._input_size_hw
        crops: list[NDArray[np.float32]] = []
        valid_indices: list[int] = []

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(fw, int(x2))
            iy2 = min(fh, int(y2))

            crop_area = (ix2 - ix1) * (iy2 - iy1)
            if crop_area < self._min_crop_area:
                continue

            crop = frame_pixels[iy1:iy2, ix1:ix2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # cv2.resize takes (width, height) — portrait 256x128 needs (128, 256)
            resized = cv2.resize(
                crop_rgb,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - _IMAGENET_MEAN) / _IMAGENET_STD
            chw = normalized.transpose(2, 0, 1)
            crops.append(chw)
            valid_indices.append(i)

        if not crops:
            return None, []

        batch = np.stack(crops, axis=0)  # (N', 3, 256, 128)
        return batch, valid_indices


class VehicleReIDExtractor:
    """ONNX-based vehicle ReID extractor optimized for traffic CCTV.

    Input: (batch, 3, 224, 224) — square aspect for vehicles, ImageNet normalization.
    Output: (batch, embedding_dim) L2-normalized embeddings.

    Accepts any ONNX vehicle ReID model (e.g. VeRi-trained ResNet50,
    VehicleID-trained models). For zero-shot fallback, use CLIPExtractor.
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
        embedding_dim: int = 512,
        input_size: int = 224,
        device: str = "cpu",
        min_crop_area: int = 2048,
    ) -> None:
        """Initialize vehicle ReID extractor with ONNX model.

        Args:
            model_path: Path to vehicle ReID ONNX file.
            embedding_dim: Expected output dimension (512 for ResNet50-based).
            input_size: Input resolution (224 for square vehicle crops).
            device: "cpu", "cuda", or "coreml" for EP selection.
            min_crop_area: Minimum crop area in pixels (2048 — vehicles are larger).

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If model_path does not exist.
        """
        resolved = Path(model_path)
        if not resolved.exists():
            msg = f"Vehicle ReID ONNX model not found: {resolved}"
            raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            msg = (
                "onnxruntime is required for VehicleReIDExtractor."
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
        """Dimensionality of output embeddings."""
        return self._embedding_dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Extract L2-normalized vehicle ReID embeddings for detection crops.

        Args:
            frame_pixels: Full frame, HWC BGR uint8, shape (H, W, 3).
            boxes_xyxy: N bounding boxes as (x1, y1, x2, y2) tuples.

        Returns:
            (N, D) float32 embeddings with zero rows for filtered crops.
            None if no valid crops remain.
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
        """Crop and preprocess detections for vehicle ReID inference.

        Pipeline: crop -> BGR->RGB -> resize(224, 224) -> /255 -> ImageNet norm -> HWC->CHW

        Args:
            frame_pixels: Full frame HWC BGR uint8.
            boxes_xyxy: Bounding boxes as (x1, y1, x2, y2).

        Returns:
            (batch, valid_indices) where batch is (N', 3, 224, 224) float32
            or (None, []) if no valid crops.
        """
        fh, fw = frame_pixels.shape[:2]
        crops: list[NDArray[np.float32]] = []
        valid_indices: list[int] = []

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            ix1 = max(0, int(x1))
            iy1 = max(0, int(y1))
            ix2 = min(fw, int(x2))
            iy2 = min(fh, int(y2))

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
            normalized = (normalized - _IMAGENET_MEAN) / _IMAGENET_STD
            chw = normalized.transpose(2, 0, 1)
            crops.append(chw)
            valid_indices.append(i)

        if not crops:
            return None, []

        batch = np.stack(crops, axis=0)  # (N', 3, 224, 224)
        return batch, valid_indices
