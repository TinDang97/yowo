---
phase: 01-correctness-benchmarking-and-proactive-fixes
plan: 01
subsystem: engine, postprocess, io, cli
tags: [warmup-validation, thread-safety, nms, rtsp, reconnect, cli-compat]

# Dependency graph
requires: []
provides:
  - WarmupValidationError exception for catching corrupt models at load time
  - Thread-safe _infer_lock in BaseEngine._run_gpu()
  - Deterministic NMS ordering via np.lexsort (confidence desc, class_id asc, x1 asc)
  - RTSP periodic reconnect in ThreadedFrameReader (prevents OpenCV memory leaks)
  - Enhanced backend error messages with uv add install commands and status checklist
  - yowo info --compat export compatibility matrix CLI command
affects: [all-phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Warmup validation pattern: dummy inference during load() to catch bad models early"
    - "Thread-safe GPU access via threading.Lock wrapping _run_gpu()"
    - "Duck-typed reconnect: source.reconnect() checked via getattr/callable"

key-files:
  created:
    - tests/unit/test_warmup_validation.py
    - tests/unit/test_thread_safety.py
  modified:
    - src/yowo/errors.py
    - src/yowo/engine.py
    - src/yowo/classify_engine.py
    - src/yowo/postprocess/_nms.py
    - src/yowo/io/_reader.py
    - src/yowo/io/_source.py
    - src/yowo/backends/_selector.py
    - src/yowo/cli/_main.py
    - tests/unit/test_nms.py
    - tests/unit/test_async_api.py
    - tests/unit/test_engine.py
    - tests/unit/test_streaming.py
    - tests/unit/test_reader.py
    - tests/unit/test_selector.py
    - tests/unit/test_cli.py

key-decisions:
  - "threading.Lock (not RLock) for _infer_lock -- inference path is linear, no reentrant risk"
  - "Warmup validation uses dummy zeros tensor rather than real calibration data"
  - "Classification validation checks softmax sum within 0.01 tolerance"
  - "Detection validation uses 1e-3 tolerance for [0, 1] range check"
  - "RTSP reconnect uses duck-typing (hasattr reconnect) not isinstance check"

patterns-established:
  - "Warmup validation: subclass _validate_output_values() for task-specific checks"
  - "Lock-internal pattern: _run_gpu holds _infer_lock internally, callers don't manage it"

requirements-completed: [CORR-08, PFIX-01, PFIX-02, PFIX-03, PFIX-04, PFIX-05]

# Metrics
duration: 25min
completed: 2026-03-07
---

# Phase 01 Plan 01: Proactive Fixes Summary

**Warmup validation catching corrupt models, thread-safe GPU inference via _infer_lock, deterministic NMS via lexsort, RTSP periodic reconnect, and yowo info --compat CLI**

## Performance

- **Duration:** ~25 min
- **Tasks:** 2/2
- **Files modified:** 15 (7 source, 8 test)
- **Tests:** 1676 total (1615 baseline + 61 new)

## Accomplishments
- Engine load() validates warmup output shape (ndim >= 2) and value ranges before entering READY state
- Detection engines reject models with class scores outside [0, 1]; classification engines reject non-softmax outputs
- Concurrent detect()/classify() calls from multiple threads are serialized via _infer_lock
- NMS output ordering is deterministic across runs via lexsort (confidence desc, class_id asc, x1 asc)
- ThreadedFrameReader reconnects RTSP sources every 300s (configurable) to prevent OpenCV memory leaks
- Missing backend errors now show detailed status checklist with uv add install commands
- `yowo info --compat` prints system export compatibility matrix

## Task Commits

1. **Task 1: Warmup validation + thread safety + deterministic NMS** - `e9dc634` (feat)
2. **Task 2: RTSP reconnect + backend errors + info --compat** - `661f0cc` (feat)

## Files Created/Modified
- `src/yowo/errors.py` - WarmupValidationError exception class
- `src/yowo/engine.py` - _validate_warmup_output(), _infer_lock, DetectionEngine._validate_output_values()
- `src/yowo/classify_engine.py` - ClassificationEngine._validate_output_values()
- `src/yowo/postprocess/_nms.py` - np.lexsort replacing np.sort for deterministic ordering
- `src/yowo/io/_reader.py` - _maybe_reconnect() with reconnect_interval_sec parameter
- `src/yowo/io/_source.py` - RTSPStreamSource.reconnect() method
- `src/yowo/backends/_selector.py` - Enhanced no-backend error with status checklist
- `src/yowo/cli/_main.py` - info --compat flag and _print_compat_matrix()
- `tests/unit/test_warmup_validation.py` - 10 tests for warmup validation
- `tests/unit/test_thread_safety.py` - 2 tests for lock existence and concurrent detect
- `tests/unit/test_nms.py` - 2 new tests for deterministic NMS ordering
- `tests/unit/test_reader.py` - 3 new tests for RTSP reconnect
- `tests/unit/test_selector.py` - 4 new tests for error messages
- `tests/unit/test_cli.py` - 3 new tests for info --compat

## Decisions Made
- threading.Lock (not RLock) for _infer_lock -- inference path is linear, no reentrant risk
- Warmup validation runs a dummy zeros tensor through backend.infer() and validates output
- Classification validation uses softmax sum tolerance of 0.01; detection uses 1e-3 for [0, 1] range
- RTSP reconnect uses duck-typing (getattr/callable) rather than isinstance to stay protocol-compatible
- _maybe_reconnect() runs inside the reader thread after each frame read (same thread as reads, no race condition)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_adetect_propagates_inference_error ordering**
- **Found during:** Task 1 (test suite validation)
- **Issue:** Test set mock_be.infer.side_effect before _loaded_engine(), but warmup validation now calls infer() during load()
- **Fix:** Moved side_effect assignment after _loaded_engine() call
- **Files modified:** tests/unit/test_async_api.py
- **Committed in:** e9dc634

**2. [Rule 1 - Bug] Fixed test_detect_works_with_injected_backend mock count**
- **Found during:** Task 1 (test suite validation)
- **Issue:** assert_called_once() fails because warmup validation adds an extra infer() call during load()
- **Fix:** Added mock_backend.infer.reset_mock() after engine.load()
- **Files modified:** tests/unit/test_engine.py
- **Committed in:** e9dc634

**3. [Rule 1 - Bug] Fixed test_returns_raw_output_and_elapsed mock count**
- **Found during:** Task 1 (test suite validation)
- **Issue:** Same warmup validation extra infer() call issue
- **Fix:** Added backend.infer.reset_mock() after engine creation
- **Files modified:** tests/unit/test_streaming.py
- **Committed in:** e9dc634

**4. [Rule 1 - Bug] Updated test_result_sorted_ascending for new NMS ordering**
- **Found during:** Task 1 (NMS deterministic ordering)
- **Issue:** Old test expected ascending index order from np.sort, but lexsort produces confidence-descending order
- **Fix:** Renamed test, updated assertion to expect [1, 2, 0] for confidence order
- **Files modified:** tests/unit/test_nms.py
- **Committed in:** e9dc634

---

**Total deviations:** 4 auto-fixed (4 Rule 1 - Bug)
**Impact on plan:** All auto-fixes necessary to maintain existing test suite compatibility with new warmup validation feature. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Warmup validation foundation ready for all engine subclasses
- Thread safety foundation enables safe concurrent inference
- Deterministic NMS enables reproducible benchmarking in plan 01-02
- RTSP reconnect ready for production streaming workloads

---
*Phase: 01-correctness-benchmarking-and-proactive-fixes*
*Completed: 2026-03-07*
