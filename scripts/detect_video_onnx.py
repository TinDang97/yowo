#!/usr/bin/env python3
"""
Detect objects in a video using a yowo-exported ONNX model.

Usage:

    python scripts/detect_onnx_video.py \
        --video input.mp4 \
        --onnx tmp/exports/yolo26l.onnx \
        --model yolo26l \
        --conf 0.25

    # Persons only (COCO class 0)
    python scripts/detect_onnx_video.py \
        --video input.mp4 \
        --onnx tmp/exports/yolo26l.onnx \
        --model yolo26l \
        --classes 0 \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from yowo.backends import OnnxBackend
from yowo.hardware import get_hardware_profile
from yowo.io import preprocess
from yowo.postprocess import postprocess
from yowo.types import BackendType, Frame, ModelFamily, ModelSize, ModelSpec


# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, tuple[ModelFamily, ModelSize]] = {
    "yolo11n": (ModelFamily.YOLO11, ModelSize.NANO),
    "yolo11s": (ModelFamily.YOLO11, ModelSize.SMALL),
    "yolo11m": (ModelFamily.YOLO11, ModelSize.MEDIUM),
    "yolo11l": (ModelFamily.YOLO11, ModelSize.LARGE),
    "yolo11x": (ModelFamily.YOLO11, ModelSize.XLARGE),
    "yolo26n": (ModelFamily.YOLO26, ModelSize.NANO),
    "yolo26s": (ModelFamily.YOLO26, ModelSize.SMALL),
    "yolo26m": (ModelFamily.YOLO26, ModelSize.MEDIUM),
    "yolo26l": (ModelFamily.YOLO26, ModelSize.LARGE),
    "yolo26x": (ModelFamily.YOLO26, ModelSize.XLARGE),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect objects in a video using a yowo ONNX model."
    )
    p.add_argument("--video", required=True, type=Path, help="Input video path.")
    p.add_argument("--onnx", required=True, type=Path, help="Path to ONNX model.")
    p.add_argument(
        "--model",
        required=True,
        choices=list(_VARIANTS),
        help="Model variant (e.g. yolo26l).",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Filter by class IDs (e.g. --classes 0 2)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _class_colour(class_id: int) -> list[int]:
    hue = int(class_id * 180 / 80) % 180
    hsv = np.uint8([[[hue, 200, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return [int(bgr[0]), int(bgr[1]), int(bgr[2])]


def _draw_boxes(img: np.ndarray, boxes) -> np.ndarray:
    for box in boxes:
        colour = _class_colour(box.class_id)
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        label = f"{box.class_name} {box.confidence:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    return img


# ---------------------------------------------------------------------------
# Main Detection
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if not args.onnx.exists():
        print(f"ONNX not found: {args.onnx}", file=sys.stderr)
        sys.exit(1)

    family, size = _VARIANTS[args.model]
    spec = ModelSpec(family=family, size=size)

    hw = get_hardware_profile()
    backend = OnnxBackend(hw)

    print("Loading model...")
    backend.load(args.onnx, device=args.device)
    backend.warmup()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print("Cannot open video", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = args.video.parent / (args.video.stem + "_detected.mp4")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames...")

    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break

        frame = Frame(
            pixels=img_bgr,
            source_id=str(args.video),
            frame_index=frame_index,
        )

        tensor = preprocess([frame], (args.imgsz, args.imgsz))
        raw = backend.infer(tensor)

        detections = postprocess(
            raw,
            tensor,
            [frame],
            model_spec=spec,
            backend=BackendType.ONNX,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            inference_time_ms=0.0,
        )

        boxes = detections[0].boxes

        if args.classes is not None:
            class_set = set(args.classes)
            boxes = tuple(b for b in boxes if b.class_id in class_set)

        annotated = _draw_boxes(img_bgr.copy(), boxes)

        writer.write(annotated)

        if not args.no_show:
            cv2.imshow("detect_onnx_video", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_index += 1

        if frame_index % 30 == 0:
            print(f"Processed {frame_index}/{total_frames}")

    cap.release()
    writer.release()
    backend.unload()
    cv2.destroyAllWindows()

    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()