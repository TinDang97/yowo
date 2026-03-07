---
phase: 02-reliability-and-multi-stream-scaling
plan: 03
subsystem: observability
tags: [logging, prometheus, health, metrics, json-formatter, cli, structured-logging]

# Dependency graph
requires:
  - phase: 02-02
    provides: OOM monitor, _is_cuda, _oom_recovering, _original_batch_size, log_level/structured_logging config fields
  - phase: 02-01
    provides: HealthStatus enum, BaseEngine lifecycle, MetricsCollector, EngineMetrics
provides:
  - JsonFormatter (stdlib-only, ISO8601 UTC JSON log lines) in src/yowo/logging.py
  - configure_logging() idempotent handler setup wired in _finalize_load
  - HealthReport frozen dataclass with as_dict() in engine.py
  - BaseEngine.health_report() method
  - BaseEngine.export_metrics() returning JSON-serialisable dict
  - BaseEngine.export_metrics_prometheus() delegating to MetricsCollector
  - MetricsCollector.export_prometheus() 7-metric Prometheus text exposition
  - yowo health CLI (exit 0/1/2 by status, --format json|text)
  - yowo metrics CLI (--format json|prometheus, exit 0 always)
affects:
  - future phases consuming structured logs or metrics scraping
  - operators using yowo health / yowo metrics in dashboards

# Tech tracking
tech-stack:
  added: []
  patterns:
    - JsonFormatter subclassing logging.Formatter for stdlib-only structured logging
    - configure_logging() clearing handlers before re-adding (idempotent)
    - HealthReport frozen dataclass with as_dict() converting enum to .value
    - export_prometheus() 7-metric Prometheus text exposition with HELP+TYPE headers
    - CLI health exits non-zero (2) without running engine; metrics CLI always exits 0

key-files:
  created:
    - src/yowo/logging.py
    - tests/unit/test_logging.py
    - tests/unit/test_metrics_export.py
    - tests/unit/test_health_report.py
  modified:
    - src/yowo/engine.py
    - src/yowo/metrics/_collector.py
    - src/yowo/cli/_main.py

key-decisions:
  - "HealthReport defined in engine.py (not types.py) to avoid circular imports with HealthStatus"
  - "configure_logging uses logger.handlers.clear() (not remove-one-by-one) for guaranteed idempotency"
  - "yowo health exits 2 (unhealthy/closed) when invoked standalone; programmatic use goes through engine.health_report()"
  - "yowo metrics exits 0 always (metrics endpoint should never fail); uses zero-value snapshot without engine"
  - "memory_pct uses best-effort try/except — returns None on CPU or any torch error rather than raising"

patterns-established:
  - "Observability surface: health_report() + export_metrics() + export_metrics_prometheus() on BaseEngine"
  - "Prometheus text format: # HELP + # TYPE + value lines per metric, ending with newline"

requirements-completed: [RELY-03, RELY-04, RELY-05]

# Metrics
duration: 8min
completed: 2026-03-07
---

# Phase 2 Plan 3: Observability - JsonFormatter, HealthReport, Prometheus Metrics Summary

**Structured JSON logging via stdlib JsonFormatter, HealthReport dataclass with health_report(), and Prometheus/JSON metrics export via MetricsCollector.export_prometheus() and two new CLI subcommands**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-07T14:26:11Z
- **Completed:** 2026-03-07T14:33:57Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- JsonFormatter (stdlib-only) emitting ISO8601 UTC JSON lines with required and optional context fields; configure_logging() clearing handlers before re-adding prevents duplicate handlers on repeated calls
- HealthReport frozen dataclass (slots=True) with as_dict() returning JSON-serialisable dict with status as string value; BaseEngine.health_report() reading live metrics, OOM state, stream count, and CUDA memory utilization
- MetricsCollector.export_prometheus() emitting 7-metric Prometheus text exposition (frames_total, errors_total, inference_mean_ms, inference_p95_ms, fps, uptime_seconds, memory_utilization) always ending with newline; engine methods export_metrics() and export_metrics_prometheus() delegating appropriately
- yowo health CLI (exit 2 for closed/no-engine, --format json|text) and yowo metrics CLI (--format json|prometheus, always exit 0) added to existing click group
- 146 new tests passing; full suite 1761 tests clean

## Task Commits

Each task was committed atomically (TDD: tests + implementation committed together after GREEN):

1. **Task 1+2: JsonFormatter, HealthReport, metrics export, and CLI commands** - `9eeafab` (feat)

_Note: Task 1 and Task 2 were committed together as all tests (including CLI tests from Task 2) were staged together after the full green pass._

## Files Created/Modified

- `src/yowo/logging.py` - JsonFormatter and configure_logging() (new file, stdlib only)
- `src/yowo/engine.py` - HealthReport dataclass, health_report(), export_metrics(), export_metrics_prometheus(), configure_logging wired in _finalize_load
- `src/yowo/metrics/_collector.py` - export_prometheus() method added to MetricsCollector
- `src/yowo/cli/_main.py` - health_command and metrics_command added to cli group
- `tests/unit/test_logging.py` - JsonFormatter + configure_logging tests (new file)
- `tests/unit/test_metrics_export.py` - Prometheus export + JSON export + CLI tests (new file)
- `tests/unit/test_health_report.py` - HealthReport dataclass + CLI exit code tests (new file)

## Decisions Made

- HealthReport defined in engine.py (not types.py) to avoid circular imports - HealthStatus is already in types.py, and putting HealthReport there too would create a cycle through engine.py imports
- configure_logging uses logger.handlers.clear() for guaranteed idempotency - cleaner than iterating handlers
- yowo health exits 2 (unhealthy/closed) when invoked without a running engine - matches the plan's exit code convention (0=ready, 1=degraded, 2=unhealthy/closed)
- yowo metrics always exits 0 using a zero-value MetricsCollector snapshot - metrics endpoint must not fail for scraping reliability
- memory_pct returns None on CPU via best-effort try/except around torch.cuda calls - avoids import errors on CPU-only installs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Pre-commit hook caught one ruff lint error (EN DASH in docstring `0.0-1.0`) and two E501 line-too-long in test files that the formatter auto-fixed. Corrected and re-committed cleanly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- RELY-03, RELY-04, RELY-05 all complete: operators can query engine health, emit structured logs for log aggregation, and scrape metrics in Prometheus or JSON format
- Phase 2 (02-reliability-and-multi-stream-scaling) is now complete with all 3 plans done
- Phase 3 planning can proceed
