"""Example: Annotated video with detection, tracking, and counting.

Processes a video file through the full yowo pipeline:
  1. YOLO detection (any model variant / backend)
  2. ByteTrack multi-object tracking (persistent IDs)
  3. ObjectCounter with zone occupancy + line crossing
  4. Annotated output video with overlays

Usage:
    # Minimal — auto-selects backend for your hardware
    uv run python examples/annotated_video.py input.mp4

    # Specify model + backend
    uv run python examples/annotated_video.py input.mp4 \
        --weights weights/yolo26n.onnx \
        --backend onnx \
        --family yolo26 --size n

    # PyTorch MPS on Apple Silicon
    uv run python examples/annotated_video.py input.mp4 \
        --weights weights/yolo26s_statedict.pt \
        --backend pytorch --device mps

    # Custom output path + confidence
    uv run python examples/annotated_video.py input.mp4 \
        -o output_annotated.mp4 --conf 0.3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from yowo import InferenceEngine, open_source
from yowo.counter import CrossDirection, ObjectCounter
from yowo.tracking import ByteTracker, track_stream
from yowo.types import BackendType, ModelFamily, ModelSize
from yowo.utils import (
    draw_count_lines,
    draw_text_panel,
    draw_tracked_boxes,
    draw_zones,
    make_center_line,
    make_half_zones,
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    output_path: str,
    *,
    weights: str,
    backend: BackendType,
    device: str,
    family: ModelFamily,
    size: ModelSize,
    confidence: float,
) -> None:
    """Run full detect → track → count → annotate pipeline."""
    # Probe video metadata
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Input:    {video_path} ({w}x{h}, {fps_in:.0f}fps, {total} frames)")
    print(f"Output:   {output_path}")
    print(f"Model:    {family.value}{size.value} | Backend: {backend.value}")
    print(f"Device:   {device}")
    print()

    # Setup zones + line
    zones = list(make_half_zones(w, h))
    line = make_center_line(w, h)

    # Pipeline components
    tracker = ByteTracker(
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        match_thresh=0.8,
        max_age=30,
        min_hits=3,
    )
    counter = ObjectCounter(zones=zones, lines=[line])
    engine = InferenceEngine(
        model_family=family,
        model_size=size,
        weights_path=Path(weights),
        backend=backend,
        device=device,
        confidence_threshold=confidence,
        iou_threshold=0.45,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    source = open_source(video_path)
    seen_ids: dict[str, set[int]] = {}
    frame_count = 0
    t_start = time.perf_counter()
    fps_smooth = 0.0

    with engine:
        for tracked in track_stream(engine, source, tracker=tracker):
            frame_count += 1
            counter.update(tracked)

            # Accumulate unique objects per class
            for box in tracked.boxes:
                seen_ids.setdefault(box.class_name, set()).add(
                    box.track_id,
                )
            unique = {cls: len(ids) for cls, ids in seen_ids.items()}

            # Smoothed FPS
            elapsed = time.perf_counter() - t_start
            cur = frame_count / elapsed if elapsed > 0 else 0
            fps_smooth = 0.9 * fps_smooth + 0.1 * cur if fps_smooth > 0 else cur

            # Line crossing totals
            lc: dict[str, tuple[int, int]] = {}
            for lid, dirs in counter.line_totals.items():
                lc[lid] = (
                    dirs.get(CrossDirection.IN, 0),
                    dirs.get(CrossDirection.OUT, 0),
                )

            # Draw annotations on frame copy
            frame = tracked.frame.pixels.copy()
            draw_zones(frame, zones)
            draw_count_lines(frame, [line])
            draw_tracked_boxes(frame, tracked)

            stats = [
                f"Frame: {tracked.frame.frame_index}",
                f"FPS: {fps_smooth:.1f}",
                f"Inference: {tracked.inference_time_ms:.1f}ms",
                f"Tracking: {tracked.tracking_time_ms:.2f}ms",
                f"Active: {tracker.active_track_count} | Lost: {tracker.lost_track_count}",
                f"Unique: {dict(unique)}",
            ]
            for lid, (in_c, out_c) in lc.items():
                stats.append(f"  {lid}: IN={in_c} OUT={out_c}")
            draw_text_panel(frame, stats)
            writer.write(frame)

            # Progress
            if frame_count % 100 == 0:
                pct = frame_count / total * 100 if total > 0 else 0
                print(
                    f"  [{frame_count}/{total}] {pct:.0f}% | "
                    f"FPS: {fps_smooth:.1f} | "
                    f"Active: {tracker.active_track_count} | "
                    f"Unique: {unique}"
                )

    writer.release()
    total_elapsed = time.perf_counter() - t_start
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
    unique = {cls: len(ids) for cls, ids in seen_ids.items()}

    lt = counter.line_totals.get("gate", {})
    li = lt.get(CrossDirection.IN, 0)
    lo = lt.get(CrossDirection.OUT, 0)

    print(f"\nDone — {frame_count} frames in {total_elapsed:.1f}s ({avg_fps:.1f} FPS)")
    print(f"Unique objects: {unique}")
    print(f"Line crossings: IN={li}, OUT={lo}")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotated video: YOLO detection + ByteTrack + ObjectCounter",
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path (default: <input>_annotated.mp4)",
    )
    parser.add_argument(
        "--weights",
        default="tmp/exports/yolo26n.onnx",
        help="Model weights path (.onnx or .pt state dict)",
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="Inference backend (default: onnx)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, mps, cuda (default: auto)",
    )
    parser.add_argument(
        "--family",
        choices=["yolo11", "yolo26"],
        default="yolo26",
        help="Model family (default: yolo26)",
    )
    parser.add_argument(
        "--size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size (default: n)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        stem = Path(args.video).stem
        args.output = f"{stem}_annotated.mp4"

    backend_map = {"onnx": BackendType.ONNX, "pytorch": BackendType.PYTORCH}
    family_map = {"yolo11": ModelFamily.YOLO11, "yolo26": ModelFamily.YOLO26}
    size_map = {
        "n": ModelSize.NANO,
        "s": ModelSize.SMALL,
        "m": ModelSize.MEDIUM,
        "l": ModelSize.LARGE,
        "x": ModelSize.XLARGE,
    }

    process_video(
        args.video,
        args.output,
        weights=args.weights,
        backend=backend_map[args.backend],
        device=args.device,
        family=family_map[args.family],
        size=size_map[args.size],
        confidence=args.conf,
    )


if __name__ == "__main__":
    main()
