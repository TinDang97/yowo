---
phase: 03-adaptive-optimization-and-batch-processing
plan: 05
subsystem: cli
tags: [click, cli, batch-processing, yowo-batch, yowo-tune]

# Dependency graph
requires:
  - phase: 03-03
    provides: BatchConfig dataclass and run_batch() function in batch/_runner.py
  - phase: 03-04
    provides: tune_command() in cli/_main.py and DetectionEngine profile auto-load

provides:
  - batch_command() Click subcommand registered on cli group with all flags
  - yowo batch SOURCE_DIR --model --output [--no-annotate] [--format jsonl|json] [--recursive] [--no-resume] [--workers N]
  - 8 CliRunner unit tests covering all batch CLI flags and exit code propagation
  - Full Phase 3 CLI surface: yowo tune + yowo batch both functional

affects:
  - Phase 04 CLI additions (follows same module-level import pattern for test patchability)
  - Any documentation/changelog work for v2.3.0 release

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level imports of BatchConfig/run_batch in cli/_main.py for test patchability at yowo.cli._main.*"
    - "sys.exit(result) to propagate integer exit codes 0/1/2 from run_batch through Click"
    - "DetectionEngine lifecycle in finally block: load() before run_batch, close() unconditionally"
    - "CliRunner tests mock run_batch + DetectionEngine to avoid real inference; assert BatchConfig fields"

key-files:
  created:
    - tests/unit/test_batch_cli.py
  modified:
    - src/yowo/cli/_main.py

key-decisions:
  - "Module-level imports in cli/_main.py for run_batch/BatchConfig/DetectionEngine for test patchability — consistent with tune/_main.py pattern established in 03-04"
  - "sys.exit(result) wraps run_batch exit code — Click absorbs SystemExit cleanly for 0; CliRunner captures non-zero exit codes correctly in tests"

patterns-established:
  - "CLI batch flag pattern: boolean is_flag options map 1:1 to BatchConfig dataclass fields"
  - "Click required option validation happens at Click layer; workers >= 0 validated with UsageError in function body"

requirements-completed: [BATC-01, TUNE-01]

# Metrics
duration: 15min
completed: 2026-03-07
---

# Phase 03 Plan 05: yowo batch CLI Subcommand Summary

**`yowo batch` Click subcommand with full flag coverage wired to BatchConfig/run_batch, completing Phase 3 user-facing CLI surface alongside `yowo tune`**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-07T16:05:00Z
- **Completed:** 2026-03-07T16:20:00Z
- **Tasks:** 1 auto + 1 human-verify checkpoint (approved)
- **Files modified:** 2

## Accomplishments

- Added `batch_command()` to `src/yowo/cli/_main.py` with all 7 options: `--model` (required), `--output` (required), `--no-annotate`, `--format jsonl|json`, `--recursive`, `--no-resume`, `--workers`
- Workers validation: raises `click.UsageError` if `--workers < 0`
- Engine lifecycle fully guarded: `engine.load()` in try, `engine.close()` in finally; `sys.exit(result)` propagates run_batch exit codes 0/1/2
- Replaced scaffold stubs in `test_batch_cli.py` with 8 CliRunner tests (help, missing output, workers validation, flag passthrough for no_resume/recursive/format/no_annotate, exit code propagation)
- All 1822 unit tests passing; `yowo batch --help` and `yowo tune --help` both verified by human checkpoint

## Task Commits

Each task was committed atomically:

1. **Task 1: yowo batch CLI subcommand and test coverage** - `496c1d4` (feat)

**Plan metadata:** (this docs commit)

## Files Created/Modified

- `src/yowo/cli/_main.py` - Added `batch_command()` with all flags, module-level imports of BatchConfig/run_batch/DetectionEngine
- `tests/unit/test_batch_cli.py` - 8 CliRunner tests replacing scaffold stubs

## Decisions Made

- Module-level imports for `run_batch`, `BatchConfig`, `DetectionEngine` in `cli/_main.py` so tests can patch at `yowo.cli._main.run_batch` — consistent with tune CLI pattern from 03-04
- `sys.exit(result)` used directly to propagate integer exit codes from `run_batch` (0 success, 1 partial errors, 2 fatal) — Click absorbs `SystemExit(0)` cleanly; `CliRunner.invoke()` captures non-zero values in `result.exit_code`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 is complete: all 8 requirements delivered (TUNE-01/02/03/04, BATC-01/02/03/04)
- CLI surface: `yowo detect`, `yowo classify`, `yowo track`, `yowo count`, `yowo benchmark`, `yowo export`, `yowo tune`, `yowo batch` all functional
- Ready for Phase 4 or v2.3.0 release work (changelog + README update)

---
*Phase: 03-adaptive-optimization-and-batch-processing*
*Completed: 2026-03-07*
