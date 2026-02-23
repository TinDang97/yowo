"""INT8 calibration data resolution.

Converts user-provided paths into a YAML path that ultralytics can consume.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_MIN_CALIBRATION_IMAGES = 10
_RECOMMENDED_CALIBRATION_IMAGES = 300

# Global list to track temp files so callers can clean up if needed.
_temp_files: list[str] = []


def resolve_calibration_source(calibration_data: str) -> str:
    """Convert user-provided path to a YAML path ultralytics can consume.

    Accepts:
    - Path to existing .yaml/.yml -> returned as-is
    - Path to image directory -> generates temporary YAML

    Args:
        calibration_data: String path to YAML file or image directory.

    Returns:
        str path to YAML file suitable for passing to ultralytics.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Directory contains fewer than _MIN_CALIBRATION_IMAGES images.
    """
    src = Path(calibration_data)

    if not src.exists():
        raise FileNotFoundError(f"Calibration source not found: {src}")

    # Already a YAML file — pass through.
    if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
        return str(src.resolve())

    if src.is_dir():
        return _resolve_image_directory(src)

    raise ValueError(
        f"Calibration source must be a .yaml/.yml file or an image directory, got: {src}"
    )


def _resolve_image_directory(directory: Path) -> str:
    """Generate a temporary YAML pointing at an image directory."""
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
    count = len(image_files)

    if count == 0:
        raise ValueError(
            f"No supported image files found in calibration directory: {directory}. "
            f"Supported extensions: {sorted(_IMAGE_EXTS)}"
        )

    if count < _MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Calibration directory contains only {count} images (minimum: "
            f"{_MIN_CALIBRATION_IMAGES}). Found: {directory}"
        )

    if count < _RECOMMENDED_CALIBRATION_IMAGES:
        logger.warning(
            "Calibration directory contains %d images; %d recommended for accurate INT8. "
            "Engine will be built but may show higher accuracy degradation.",
            count,
            _RECOMMENDED_CALIBRATION_IMAGES,
        )

    # Generate a temporary YAML file pointing at the directory.
    # ultralytics accepts a YAML with a 'path' key for calibration.
    yaml_content = f"path: {directory.resolve()}\ntrain: .\nval: .\nnc: 80\nnames: []\n"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="yowo_calib_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(yaml_content)
        tmp_name = tmp.name

    _temp_files.append(tmp_name)
    logger.debug("Generated calibration YAML: %s (images=%d)", tmp_name, count)
    return tmp_name


__all__ = ["resolve_calibration_source"]
