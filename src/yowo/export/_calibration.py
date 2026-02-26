"""INT8 calibration data resolution and batch generation.

Validates and resolves calibration image directories for INT8 quantization.
Provides batch iteration over calibration images for TensorRT and ONNX
quantization workflows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from yowo.types import IMAGE_EXTS

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
_MIN_CALIBRATION_IMAGES = 10
_RECOMMENDED_CALIBRATION_IMAGES = 300


def resolve_calibration_images(calibration_data: str) -> list[Path]:
    """Resolve calibration source to a list of image file paths.

    Accepts:
    - Path to image directory → returns list of image paths

    Args:
        calibration_data: Path to an image directory.

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Directory contains fewer than minimum required images.
    """
    src = Path(calibration_data)

    if not src.exists():
        raise FileNotFoundError(f"Calibration source not found: {src}")

    if not src.is_dir():
        raise ValueError(f"Calibration source must be an image directory, got: {src}")

    image_files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    count = len(image_files)

    if count == 0:
        raise ValueError(
            f"No supported image files found in calibration directory: {src}. "
            f"Supported extensions: {sorted(IMAGE_EXTS)}"
        )

    if count < _MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Calibration directory contains only {count} images (minimum: "
            f"{_MIN_CALIBRATION_IMAGES}). Found: {src}"
        )

    if count < _RECOMMENDED_CALIBRATION_IMAGES:
        logger.warning(
            "Calibration directory contains %d images; %d recommended for accurate INT8. "
            "Engine will be built but may show higher accuracy degradation.",
            count,
            _RECOMMENDED_CALIBRATION_IMAGES,
        )

    return image_files


def calibration_batches(
    image_paths: list[Path],
    batch_size: int = 8,
    input_size: int = 640,
) -> Iterator[NDArray[np.float32]]:
    """Yield BCHW float32 batches from calibration images.

    Uses ``cv2.dnn.blobFromImages`` for efficient BGR->RGB + resize + normalize
    in a single C++ call, matching the inference preprocessing pipeline.

    Args:
        image_paths: List of image file paths.
        batch_size: Images per batch.
        input_size: Target spatial size (square).

    Yields:
        BCHW float32 arrays with shape ``(B, 3, input_size, input_size)``.
    """
    import cv2

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        images: list[cv2.typing.MatLike] = []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Skipping unreadable calibration image: %s", p)
                continue
            resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            images.append(resized)  # type: ignore[arg-type]
        if not images:
            continue
        batch: NDArray[np.float32] = cv2.dnn.blobFromImages(  # type: ignore[assignment]
            images, scalefactor=1.0 / 255.0, size=(input_size, input_size), swapRB=True
        )
        yield batch


__all__ = ["calibration_batches", "resolve_calibration_images"]
