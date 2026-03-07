"""Batch inference runner: directory walk, per-file inference, JSONL output,
atomic checkpoint for resume-on-interruption, and rich progress display.

Usage::

    from yowo.batch import run_batch, BatchConfig
    from yowo.engine import DetectionEngine

    cfg = BatchConfig(source_dir=Path("images/"), output_dir=Path("out/"), model_name="yolo11n")
    with DetectionEngine(...) as engine:
        exit_code = run_batch(cfg, engine)
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np

from yowo.types import IMAGE_EXTS, VIDEO_EXTS, Frame

if TYPE_CHECKING:
    from yowo.engine import DetectionEngine

logger = logging.getLogger(__name__)

# Module-level cv2 reference — patched in tests
try:
    import cv2 as _cv2_module
except ImportError:  # pragma: no cover
    _cv2_module = None  # type: ignore[assignment]

# Re-export as 'cv2' so tests can patch 'yowo.batch._runner.cv2'
cv2 = _cv2_module

_CHECKPOINT_FILENAME = ".yowo_checkpoint.json"
_CHECKPOINT_INTERVAL_FILES = 100
_CHECKPOINT_INTERVAL_SECS = 30.0
_FPS_WINDOW_SECS = 5.0


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """Configuration for a batch inference run.

    Attributes:
        source_dir: Directory containing input files (images and/or videos).
        output_dir: Directory where results, checkpoints, and logs are written.
        model_name: Model identifier (e.g. ``"yolo11n"``).
        weights_path: Optional path to custom weights file.
        no_annotate: Skip writing annotated output frames when ``True``.
        output_format: Output format for results. Currently only ``"jsonl"``.
        recursive: Walk source_dir recursively when ``True``.
        no_resume: Ignore any existing checkpoint and reprocess all files.
        workers: Number of pipeline workers (0 = synchronous).
    """

    source_dir: Path
    output_dir: Path
    model_name: str
    weights_path: Path | None = None
    no_annotate: bool = False
    output_format: str = "jsonl"
    recursive: bool = False
    no_resume: bool = False
    workers: int = 0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(path: Path, completed: list[str], stats: dict[str, object]) -> None:
    """Write checkpoint atomically using write-to-tmp then os.replace.

    Args:
        path: Destination checkpoint file path.
        completed: List of absolute path strings for completed files.
        stats: Arbitrary stats dict (total_frames, etc.).
    """
    data: dict[str, object] = {
        "completed": completed,
        "stats": stats,
        "updated_at": time.time(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_checkpoint(path: Path) -> dict[str, object] | None:
    """Read checkpoint, returning parsed dict or None on missing/corrupt.

    Args:
        path: Checkpoint file path.

    Returns:
        Parsed dict with ``completed`` and ``stats`` keys, or ``None``.
    """
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            return None
        return result  # type: ignore[return-value]
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------


def _collect_sources(source_dir: Path, recursive: bool) -> list[Path]:
    """Collect all image and video files in source_dir.

    Args:
        source_dir: Root directory to search.
        recursive: If ``True``, descend into subdirectories.

    Returns:
        Sorted list of matching :class:`~pathlib.Path` objects.
    """
    all_exts = IMAGE_EXTS | VIDEO_EXTS
    candidates = source_dir.rglob("*") if recursive else source_dir.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in all_exts)


# ---------------------------------------------------------------------------
# Internal processing helpers
# ---------------------------------------------------------------------------


def _process_image(
    path: Path,
    engine: DetectionEngine,
    no_annotate: bool,
    output_dir: Path,
    jsonl_file: IO[str],
) -> int:
    """Process a single image file.

    Args:
        path: Path to image file.
        engine: Loaded detection engine.
        no_annotate: Skip saving annotated frame when ``True``.
        output_dir: Output directory for annotated frames.
        jsonl_file: Open text file for JSONL output.

    Returns:
        Number of frames processed (always 1).

    Raises:
        ValueError: When ``cv2.imread`` returns ``None`` (unreadable file).
    """
    assert cv2 is not None, "cv2 must be available"
    raw = cv2.imread(str(path))
    if raw is None:
        raise ValueError(f"cv2.imread returned None for {path}")

    frame_arr: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] = raw.astype(np.uint8)
    frame = Frame(pixels=frame_arr, source_id=str(path), frame_index=0)
    detections = engine.detect([frame])
    detection = detections[0]

    row = {"path": str(path), "detections": detection.to_dict()}
    jsonl_file.write(json.dumps(row) + "\n")

    if not no_annotate:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_path = frames_dir / f"{path.stem}_annotated{path.suffix}"
        cv2.imwrite(str(out_path), frame_arr)

    return 1


def _process_video(
    path: Path,
    engine: DetectionEngine,
    no_annotate: bool,
    output_dir: Path,
    jsonl_file: IO[str],
) -> int:
    """Process a video file frame by frame using VideoFileSource.

    Args:
        path: Path to video file.
        engine: Loaded detection engine.
        no_annotate: Skip saving annotated frames when ``True``.
        output_dir: Output directory for annotated frames.
        jsonl_file: Open text file for JSONL output.

    Returns:
        Number of frames processed.
    """
    from yowo.io._source import VideoFileSource

    frame_detections: list[object] = []
    frame_count = 0

    source = VideoFileSource(path)
    for frame in source:
        detections = engine.detect([frame])
        frame_detections.append(detections[0].to_dict())
        frame_count += 1

        if not no_annotate:
            assert cv2 is not None, "cv2 must be available"
            frames_dir = output_dir / "frames" / path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)
            out_path = frames_dir / f"{frame.frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), frame.pixels)

    source.close()

    row = {"path": str(path), "detections": frame_detections}
    jsonl_file.write(json.dumps(row) + "\n")
    return frame_count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_batch(config: BatchConfig, engine: DetectionEngine) -> int:
    """Run offline batch inference over all files in ``config.source_dir``.

    Args:
        config: Batch run configuration.
        engine: A loaded :class:`~yowo.engine.DetectionEngine` instance.

    Returns:
        Exit code: ``0`` success, ``1`` partial errors, ``2`` fatal error.
    """
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    console = Console()

    # ---- Setup ---------------------------------------------------------------
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output dir %s: %s", config.output_dir, exc)
        return 2

    source_files = _collect_sources(config.source_dir, config.recursive)
    total_files = len(source_files)

    checkpoint_path = config.output_dir / _CHECKPOINT_FILENAME

    # Resume logic
    completed: set[str] = set()
    jsonl_mode = "w"
    if not config.no_resume:
        checkpoint = _read_checkpoint(checkpoint_path)
        if checkpoint:
            raw_completed = checkpoint.get("completed", [])
            completed = set(raw_completed) if isinstance(raw_completed, list) else set()
            jsonl_mode = "a"
            console.print(f"[yellow]Resuming: {len(completed)}/{total_files} files done[/yellow]")

    jsonl_path = config.output_dir / "results.jsonl"
    errors_log_path = config.output_dir / "errors.log"

    # ---- State ---------------------------------------------------------------
    error_count = 0
    total_frames = 0
    completed_list: list[str] = list(completed)
    files_since_flush = 0
    start_epoch = time.monotonic()

    # Rolling FPS window: deque of (monotonic_time, cumulative_frame_count) pairs
    fps_window: collections.deque[tuple[float, int]] = collections.deque()
    fps_window.append((time.monotonic(), 0))

    # ---- Checkpoint timer ----------------------------------------------------
    timer_lock = threading.Lock()
    # Use a mutable container so the closure can rebind the timer reference
    _timer_holder: list[threading.Timer] = []

    def _flush_checkpoint() -> None:
        _write_checkpoint(
            checkpoint_path,
            completed_list,
            {"total_frames": total_frames, "error_count": error_count},
        )
        # Restart timer after flush
        t = threading.Timer(_CHECKPOINT_INTERVAL_SECS, _flush_checkpoint)
        t.daemon = True
        with timer_lock:
            _timer_holder.clear()
            _timer_holder.append(t)
        t.start()

    def _cancel_timer() -> None:
        with timer_lock:
            if _timer_holder:
                _timer_holder[0].cancel()
                _timer_holder.clear()

    def _restart_timer() -> None:
        _cancel_timer()
        t = threading.Timer(_CHECKPOINT_INTERVAL_SECS, _flush_checkpoint)
        t.daemon = True
        with timer_lock:
            _timer_holder.append(t)
        t.start()

    # Start initial timer
    _restart_timer()

    # ---- Main loop -----------------------------------------------------------
    try:
        with (
            open(jsonl_path, jsonl_mode, encoding="utf-8") as jsonl_file,
            open(errors_log_path, "a", encoding="utf-8") as err_file,
            Progress(
                SpinnerColumn(),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[files_done]}/{task.total} files"),
                TextColumn("[green]{task.fields[fps]:.1f} FPS"),
                TimeRemainingColumn(),
                console=console,
            ) as progress,
        ):
            task_id = progress.add_task(
                "Batch inference",
                total=total_files,
                files_done=len(completed),
                fps=0.0,
            )

            for file_path in source_files:
                file_key = str(file_path)
                if file_key in completed:
                    continue

                try:
                    if file_path.suffix.lower() in IMAGE_EXTS:
                        n_frames = _process_image(
                            file_path,
                            engine,
                            config.no_annotate,
                            config.output_dir,
                            jsonl_file,
                        )
                    else:
                        n_frames = _process_video(
                            file_path,
                            engine,
                            config.no_annotate,
                            config.output_dir,
                            jsonl_file,
                        )

                    total_frames += n_frames
                    completed.add(file_key)
                    completed_list.append(file_key)
                    files_since_flush += 1
                    jsonl_file.flush()

                except Exception as exc:
                    logger.warning("Failed to process %s: %s", file_path, exc)
                    err_file.write(f"{file_path}: {exc}\n")
                    err_file.flush()
                    error_count += 1

                # Update FPS rolling window
                now = time.monotonic()
                fps_window.append((now, total_frames))
                cutoff = now - _FPS_WINDOW_SECS
                while len(fps_window) > 1 and fps_window[0][0] < cutoff:
                    fps_window.popleft()
                if len(fps_window) >= 2:
                    window_secs = fps_window[-1][0] - fps_window[0][0]
                    window_frames = fps_window[-1][1] - fps_window[0][1]
                    fps = window_frames / window_secs if window_secs > 0 else 0.0
                else:
                    fps = 0.0

                done_count = len(completed) + error_count
                progress.update(
                    task_id,
                    advance=1,
                    files_done=done_count,
                    fps=fps,
                )

                # Periodic file-count-based checkpoint flush
                if files_since_flush >= _CHECKPOINT_INTERVAL_FILES:
                    _cancel_timer()
                    _write_checkpoint(
                        checkpoint_path,
                        completed_list,
                        {"total_frames": total_frames, "error_count": error_count},
                    )
                    files_since_flush = 0
                    _restart_timer()

    except Exception as exc:
        logger.exception("Fatal error in run_batch: %s", exc)
        return 2
    finally:
        # Cancel timer and do final flush
        _cancel_timer()
        with contextlib.suppress(Exception):
            _write_checkpoint(
                checkpoint_path,
                completed_list,
                {"total_frames": total_frames, "error_count": error_count},
            )

    # ---- Summary -------------------------------------------------------------
    elapsed = time.monotonic() - start_epoch
    avg_fps = total_frames / elapsed if elapsed > 0 else 0.0
    console.print(
        f"Completed: {total_files} files | {total_frames} frames | "
        f"{elapsed:.1f}s | {avg_fps:.1f} FPS avg | Output: {config.output_dir}"
    )

    if error_count == 0:
        return 0
    return 1
