---
phase: 02-reliability-and-multi-stream-scaling
plan: "02"
subsystem: engine
tags: [oom-monitor, retry, graceful-degradation, cuda, config, threading]

requires:
  - phase: 01-correctness-benchmarking-proactive-fixes
    provides: BaseEngine lifecycle, _run_gpu, MetricsCollector, HealthStatus, EventBus

provides:
  - OOM monitor daemon thread with three-tier recovery ladder (halve batch / precision fallback / evict streams)
  - _infer_with_retry: 3-attempt exponential backoff (100ms/200ms/400ms), empty result on exhaustion
  - InferenceConfig.log_level and structured_logging fields with env var mappings
  - ClassificationConfig.log_level and structured_logging fields
  - StreamConfig dataclass in types.py for per-stream failure handling
  - FrameCollector._auto_removed_errors: persists errors from auto-removed streams for pipeline failure detection

affects:
  - 02-03: RELY-03/04 structured logging will use log_level/structured_logging from config
  - future: subclasses can override _evict_lowest_activity_streams to attach FrameCollector

tech-stack:
  added: []
  patterns:
    - "OOM monitor as daemon thread (Event.wait loop, not time.sleep) for fast shutdown"
    - "Three-tier recovery ladder: tier1=halve+reallocate, tier2=precision fallback, tier3=evict streams"
    - "_oom_recovering flag gates clear path (batch restore only fires when recovering=True)"
    - "Retry wrapper absorbs backend.infer() errors — engine never crashes from transient GPU errors"
    - "Auto-removed stream errors persisted in _auto_removed_errors for pipeline failure detection"
    - "frozenset class variable for valid enum-style string validation in dataclasses"

key-files:
  created:
    - tests/unit/test_oom_monitor.py
    - tests/unit/test_gpu_retry.py
  modified:
    - src/yowo/engine.py
    - src/yowo/config.py
    - src/yowo/types.py
    - src/yowo/pipeline/__init__.py
    - src/yowo/pipeline/_collector.py
    - tests/unit/test_async_api.py
    - tests/unit/test_engine.py
    - tests/unit/test_collector.py

key-decisions:
  - "OOM monitor loop uses Event.wait(timeout=5.0) NOT time.sleep(5.0) for fast clean shutdown on close()"
  - "_halve_batch_size reallocates PreprocessBuffer after halving to prevent oversized batch corruption"
  - "_evict_lowest_activity_streams is BaseEngine stub (no FrameCollector ref at this layer); subclasses override"
  - "_infer_with_retry returns np.zeros((1,0,6)) on exhaustion — detection postprocess handles empty gracefully"
  - "Auto-removed stream errors tracked in _auto_removed_errors so _check_stream_errors can detect total pipeline failure"
  - "log_level validated at __post_init__ time against _VALID_LOG_LEVELS frozenset class variable"

requirements-completed:
  - RELY-01
  - RELY-02

duration: 24min
completed: 2026-03-07
---

# Phase 02 Plan 02: OOM Monitor and GPU Retry Summary

**OOM monitor daemon with three-tier recovery (halve batch / FP16 fallback / evict streams) plus _infer_with_retry exponential backoff absorbing transient GPU errors**

## Performance

- **Duration:** 24 min
- **Started:** 2026-03-07T13:38:44Z
- **Completed:** 2026-03-07T14:02:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- OOM monitor daemon starts on CUDA load via `_start_oom_monitor()`, polls GPU memory every 5s with `Event.wait()` (fast exit on `close()`), applies three-tier recovery at 80%/90%/95% thresholds
- `_halve_batch_size` halves `_batch_size` and reallocates `PreprocessBuffer` to prevent stale buffer capacity issuing oversized batches
- `_infer_with_retry` retries `backend.infer()` up to 3 times (100ms/200ms/400ms), records error and emits "error" event on exhaustion, returns empty zeros — engine stays alive
- `InferenceConfig` and `ClassificationConfig` gain `log_level` (validated) and `structured_logging` (bool) with YOWO_LOG_LEVEL / YOWO_STRUCTURED_LOGGING env var mappings

## Task Commits

1. **Task 1: OOM monitor daemon and three-tier recovery ladder** - `e38f9c5` (feat)
2. **Task 2: GPU error retry wrapper and config fields** - `87580ab` (feat)

## Files Created/Modified

- `src/yowo/engine.py` - OOM monitor methods, _infer_with_retry, _RETRY_DELAYS, _OOM_TIER* constants
- `src/yowo/config.py` - log_level and structured_logging fields on InferenceConfig and ClassificationConfig
- `src/yowo/types.py` - StreamConfig dataclass (per-stream failure config, auto_reconnect, max_consecutive_errors)
- `src/yowo/pipeline/__init__.py` - _check_stream_errors handles empty stream_states (all streams auto-removed)
- `src/yowo/pipeline/_collector.py` - _auto_removed_errors dict, stream_errors includes auto-removed streams
- `tests/unit/test_oom_monitor.py` - 17 tests: lifecycle, thresholds, events, buffer reallocation
- `tests/unit/test_gpu_retry.py` - 8 tests: retry behavior, backoff delays, empty result, error recording
- `tests/unit/test_async_api.py` - Updated adetect error test to reflect retry semantics
- `tests/unit/test_engine.py` - Updated 2 tests: persistent error returns empty (not raise), pipeline source.close()
- `tests/unit/test_collector.py` - Updated 2 tests: auto-removed stream errors visible in stream_errors

## Decisions Made

- `Event.wait(timeout=5.0)` in OOM monitor loop instead of `time.sleep(5.0)` — daemon exits in ≤5s when `_oom_stop.set()` is called
- `_halve_batch_size` saves `_original_batch_size` only on first recovery, not subsequent halvings — preserves true original for restore
- `_evict_lowest_activity_streams` is a stub in BaseEngine with warning log; no FrameCollector reference at this layer by design
- Empty result `np.zeros((1, 0, 6))` on retry exhaustion — detection postprocess already handles empty gracefully (zero-box path)
- `_auto_removed_errors` added to FrameCollector to bridge the gap between auto-remove (stream gone from `_streams`) and `_check_stream_errors` needing to detect total pipeline failure

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FrameCollector auto-removed streams lost their errors from stream_errors**
- **Found during:** Task 1 (test_oom_monitor.py triggering full test suite)
- **Issue:** Pipeline code (pre-existing working-tree changes for STRM features) had auto-remove behavior where stream errors were removed from `_streams` — `_check_stream_errors` couldn't detect "all streams failed" because `stream_states` was empty `{}`
- **Fix:** Added `_auto_removed_errors` dict to FrameCollector; populated before `remove_stream()` in auto-remove path; `stream_errors` merges it; `_check_stream_errors` raises when `states` is empty but `errors` is non-empty
- **Files modified:** src/yowo/pipeline/_collector.py, src/yowo/pipeline/__init__.py, tests/unit/test_collector.py
- **Verification:** `test_all_streams_error_raises` and all pipeline tests pass
- **Committed in:** e38f9c5 (Task 1 commit, bundled with in-progress pipeline changes)

**2. [Rule 1 - Bug] Three existing tests expected backend.infer() raises to propagate but retry wrapper absorbs them**
- **Found during:** Task 2 (full unit suite run after _infer_with_retry implementation)
- **Issue:** `test_adetect_propagates_inference_error`, `test_pipeline_worker_exception_propagates`, `test_stream_pipeline_closes_on_detect_error` expected RuntimeError/InferenceError to bubble up — now retry wrapper absorbs and returns empty result
- **Fix:** Updated tests to reflect new behavior: check empty boxes + errors_total==1 instead of `pytest.raises`; patched `_run_batch` instead of `backend.infer` for pipeline teardown test
- **Files modified:** tests/unit/test_async_api.py, tests/unit/test_engine.py
- **Verification:** All 1726 unit tests pass
- **Committed in:** 87580ab (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2x Rule 1 - Bug)
**Impact on plan:** Both fixes necessary for test suite correctness. The FrameCollector fix ensures pipeline failure detection works end-to-end. The test updates reflect the correct behavior mandated by RELY-02.

## Issues Encountered

- Pre-commit hook stash/restore pattern caused intermediate file reads to see stale content — required careful re-reads before edits
- EventBus dispatches callbacks on background daemon thread (not synchronous) — OOM monitor tests needed `threading.Event` synchronization for event assertions

## Next Phase Readiness

- RELY-01 and RELY-02 complete: GPU memory pressure handled gracefully, transient errors absorbed
- `log_level` and `structured_logging` config fields ready for RELY-03/04 structured logging implementation
- All 1726 unit tests passing

---
*Phase: 02-reliability-and-multi-stream-scaling*
*Completed: 2026-03-07*
