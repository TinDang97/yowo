"""Output writers for detection results.

Provides JSON serialization and annotated-frame JPEG output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2

from yowo.types import Detection

# Fixed 20-color palette (BGR) for class ID annotation.
_PALETTE: list[tuple[int, int, int]] = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def write_json(detections: list[Detection], path: Path) -> None:
    """Write detections to a JSON file atomically.

    Each detection serializes as::

        {
          "frame_index": int,
          "source_id":   str,
          "boxes": [
            {"x1": float, "y1": float, "x2": float, "y2": float,
             "class_id": int, "class_name": str, "confidence": float}
          ]
        }

    Writes via a sibling ``.tmp`` file and ``os.replace()`` for atomicity.

    Args:
        detections: List of Detection objects to serialize.
        path: Destination JSON file path. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")

    records = [
        {
            "frame_index": det.frame.frame_index,
            "source_id": det.frame.source_id,
            "boxes": [
                {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "class_id": box.class_id,
                    "class_name": box.class_name,
                    "confidence": box.confidence,
                }
                for box in det.boxes
            ],
        }
        for det in detections
    ]

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

        for box in det.boxes:
            color = _PALETTE[box.class_id % len(_PALETTE)]
            x1, y1, x2, y2 = (
                round(box.x1),
                round(box.y1),
                round(box.x2),
                round(box.y2),
            )

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

            label = f"{box.class_name} {box.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw filled rectangle behind text for readability.
            label_y = max(y1 - baseline, text_h)
            cv2.rectangle(
                canvas,
                (x1, label_y - text_h - baseline),
                (x1 + text_w, label_y + baseline),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                canvas,
                label,
                (x1, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

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

    for box in detection.boxes:
        color = _PALETTE[box.class_id % len(_PALETTE)]
        x1, y1, x2, y2 = round(box.x1), round(box.y1), round(box.x2), round(box.y2)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

        label = f"{box.class_name} {box.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - baseline, text_h)
        cv2.rectangle(
            canvas,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas, label, (x1, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    ext = output_path.suffix.lower()
    params: list[int] = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else []
    cv2.imwrite(str(output_path), canvas, params)


__all__ = [
    "write_annotated_frames",
    "write_json",
]
