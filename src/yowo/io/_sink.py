"""Output writers for detection results.

Provides JSON serialization and annotated-frame JPEG output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2

from yowo.types import Detection
from yowo.utils._draw import draw_bounding_boxes


def write_json(detections: list[Detection], path: Path) -> None:
    """Write detections to a JSON file atomically.

    Each detection serializes via :meth:`Detection.to_dict`, which includes
    ``source_id``, ``frame_index``, ``timestamp_ms``, ``inference_time_ms``,
    ``backend``, ``model``, and ``boxes``.

    Writes via a sibling ``.tmp`` file and ``os.replace()`` for atomicity.

    Args:
        detections: List of Detection objects to serialize.
        path: Destination JSON file path. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    records = [det.to_dict() for det in detections]
    try:
        tmp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_annotated_frames(detections: list[Detection], output_dir: Path) -> None:
    """Draw bounding boxes on frames and save as JPEG images.

    Filenames are zero-padded six-digit frame indices: ``000000.jpg``,
    ``000001.jpg``, etc.

    Box color is determined by ``class_id % 20`` mapped to a fixed palette.
    Label format: ``"{class_name} {confidence:.2f}"`` drawn above the box.

    Args:
        detections: Detection objects whose frames will be annotated.
        output_dir: Destination directory. Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for det in detections:
        canvas = det.frame.pixels.copy()
        draw_bounding_boxes(canvas, det.boxes)
        out_path = output_dir / f"{det.frame.frame_index:06d}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])


def write_annotated_frame(detection: Detection, output_path: Path) -> None:
    """Draw bounding boxes on a single frame and save to *output_path*.

    Unlike :func:`write_annotated_frames`, this writes to an explicit file
    path rather than a directory with auto-generated names.  Suitable for
    single-image inference where the caller controls the output filename.

    Args:
        detection: Detection result containing the source frame and boxes.
        output_path: Destination file path (e.g. ``/tmp/result.jpg``).
            The parent directory is created if it does not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = detection.frame.pixels.copy()
    draw_bounding_boxes(canvas, detection.boxes)

    ext = output_path.suffix.lower()
    params: list[int] = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else []
    cv2.imwrite(str(output_path), canvas, params)


__all__ = [
    "write_annotated_frames",
    "write_json",
]
