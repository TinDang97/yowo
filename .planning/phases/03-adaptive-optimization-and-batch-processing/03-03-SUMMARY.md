---
phase: 03-adaptive-optimization-and-batch-processing
plan: "03"
subsystem: batch
tags: [batch, offline-inference, checkpoint, progress, jsonl]
dependency_graph:
  requires:
    - "03-01"  # tune profile package
  provides:
    - yowo.batch (run_batch, BatchConfig)
  affects:
    - src/yowo/batch/
    - tests/unit/test_batch_runner.py
tech_stack:
  added:
    - rich.progress (SpinnerColumn, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn)
    - collections.deque (rolling FPS window)
    - threading.Timer (periodic checkpoint flush)
    - os.replace (atomic write)
    - yowo.io._source.VideoFileSource (video frame iteration)
  patterns:
    - TDD: RED tests first, then GREEN implementation
    - Atomic write: .tmp -> os.replace pattern (consistent with tune profile write)
    - Module-level cv2 import patched in tests (not lazy local import)
key_files:
  created:
    - src/yowo/batch/__init__.py
    - src/yowo/batch/_runner.py
    - tests/unit/test_batch_runner.py
  modified: []
decisions:
  - "cv2 imported at module level (not lazily) to enable test patching via patch('yowo.batch._runner.cv2')"
  - "VideoFileSource used for video iteration (not ThreadedFrameReader) — simpler API, no start()/get()/stop() complexity for offline batch"
  - "Timer holder uses list[Timer] (not Optional[Timer]) to avoid pyright type-narrowing false positives with None comparison"
  - "assert cv2 is not None used before cv2 method calls — satisfies pyright without disabling checks"
  - "run_batch captures fatal exception and returns 2 — engine stays alive, caller decides"
metrics:
  duration: ~7min
  completed_date: "2026-03-07"
  tasks_completed: 2
  files_created: 3
  tests_added: 16
---

# Phase 03 Plan 03: Batch Processing Runner Summary

Offline high-throughput batch inference runner with atomic checkpoint resume, rolling FPS tracking, and Rich progress display.

## What Was Built

`yowo.batch` package providing `run_batch(config, engine) -> int` for offline processing of image/video directories.

Key behaviors:
- Directory walk with `IMAGE_EXTS | VIDEO_EXTS` filter (recursive optional)
- Per-image: `cv2.imread` + `engine.detect([Frame])` + JSONL row
- Per-video: `VideoFileSource` frame iteration + grouped JSONL row
- Atomic checkpoint: write `.tmp` then `os.replace` every 100 files or 30s
- Resume mode: loads checkpoint, opens `results.jsonl` in append mode, skips completed files
- `errors.log` for files that fail to open; exit code 1 on partial errors
- Rich Progress: SpinnerColumn, BarColumn, TaskProgressColumn, FPS TextColumn (5s rolling window), TimeRemainingColumn
- Final summary: `Completed: N files | M frames | Xs | Y.Y FPS avg | Output: path`

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1+2  | BatchConfig + checkpoint + run_batch | 2ba2670 | src/yowo/batch/__init__.py, src/yowo/batch/_runner.py, tests/unit/test_batch_runner.py |

## Test Results

16 tests, 0 skips, 0 failures:
- `test_batch_config_defaults` — default field values
- `test_checkpoint_atomic_write` — round-trip write/read with os.replace
- `test_checkpoint_read_returns_none_when_missing` — graceful missing file
- `test_checkpoint_read_returns_none_on_corrupt` — graceful corrupt JSON
- `test_collect_sources_images_only` — IMAGE_EXTS filter
- `test_collect_sources_videos` — VIDEO_EXTS filter
- `test_collect_sources_mixed` — combined filter
- `test_collect_sources_recursive` — recursive vs flat traversal
- `test_collect_sources_sorted` — sorted output
- `test_checkpoint_resume_skips_completed` — resume skips already-done files
- `test_no_resume_flag_reprocesses_all` — no_resume bypasses checkpoint
- `test_errors_written_to_log` — cv2.imread=None triggers errors.log
- `test_jsonl_output_written` — one row per image file
- `test_uses_profile_batch_size` — engine.detect called with list[Frame]
- `test_output_dir_created` — mkdir(parents=True) on missing output dir
- `test_progress_columns` — integration path produces exit code 0

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ThreadedFrameReader API mismatch**
- **Found during:** Task 2 implementation
- **Issue:** Plan specified `ThreadedFrameReader(source=str(path))` for video processing, but `ThreadedFrameReader` takes a `FrameSource` protocol (not str) and lacks `__iter__`/`close()` — it requires `start()`/`get()`/`stop()`
- **Fix:** Used `VideoFileSource(path)` directly — it implements `FrameSource` protocol with `__iter__` and `close()`, simpler API for offline batch context
- **Files modified:** src/yowo/batch/_runner.py
- **Commit:** 2ba2670

**2. [Rule 1 - Bug] Duplicate `total` kwarg in progress.add_task**
- **Found during:** Task 2 implementation
- **Issue:** `progress.add_task(total=total_files, ..., total=total_files)` — SyntaxError
- **Fix:** Removed duplicate kwarg; referenced `task.total` in TextColumn format string
- **Files modified:** src/yowo/batch/_runner.py
- **Commit:** 2ba2670

## Self-Check: PASSED
