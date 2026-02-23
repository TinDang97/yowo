"""INT8 calibration data resolution.

Validates and resolves calibration image directories for INT8 quantization.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
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

    image_files = sorted(f for f in src.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    count = len(image_files)

    if count == 0:
        raise ValueError(
            f"No supported image files found in calibration directory: {src}. "
            f"Supported extensions: {sorted(_IMAGE_EXTS)}"
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


__all__ = ["resolve_calibration_images"]
