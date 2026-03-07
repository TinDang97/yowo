# Phase 2: Reliability and Multi-Stream Scaling - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Make YOWO deployable in sustained production environments with 100+ concurrent RTSP/video streams, automatic failure recovery, and full operational visibility. This phase does NOT add new inference capabilities — it makes existing inference robust under load. Covers: multi-stream isolation, OOM detection and recovery, health check API, structured logging, and metrics export.

</domain>

<decisions>
## Implementation Decisions

### OOM Detection and Recovery
- Use `torch.cuda.memory_reserved() / torch.cuda.get_device_properties().total_memory` to track GPU memory utilization continuously
- Three-tier recovery ladder (executed in order, each reported via EventBus):
  1. **80% utilization** → halve current batch size (minimum 1)
  2. **90% utilization** → attempt precision fallback: FP32 → FP16 if backend supports it
  3. **95% utilization** → evict lowest-activity streams (fewest frames in last 60s) until utilization drops below 85%
- Memory check runs on a background daemon thread (check interval: 5s) — not on inference hot path
- CPU-only deployments: skip OOM check (not applicable); ONNX/TensorRT backends: use psutil for system RAM check instead
- Engine transitions to DEGRADED health state when any recovery action is taken; returns to READY when utilization drops below 75% and batch size is restored

### GPU Error Recovery (RELY-02)
- Wrap `backend.infer()` calls in try/except for `torch.cuda.OutOfMemoryError`, `RuntimeError` (GPU errors), and `Exception`
- On transient error: log structured warning, retry up to 3 times with exponential backoff (100ms, 200ms, 400ms)
- If all 3 retries fail: emit error event, increment `errors_total` counter, return empty result for that frame — do NOT crash engine
- If `errors_total >= error_threshold` (configurable, default 10): set engine DEGRADED
- Full engine restart is triggered by user via `engine.close()` + `engine.load()` — no automatic full restart

### Health Check API (RELY-03)
- `engine.health` property already exists returning `HealthStatus` enum — extend to `HealthReport` dataclass
- `HealthReport` fields: `status` (ready/degraded/unhealthy/starting/closed), `uptime_s`, `errors_total`, `frames_total`, `memory_pct` (0.0–1.0, None for CPU), `stream_count` (for pipeline mode), `batch_size_current`, `precision_current`
- New CLI subcommand: `yowo health --engine-metrics` — prints HealthReport as JSON; exit code 0=ready, 1=degraded, 2=unhealthy
- Programmatic: `engine.health_report()` method returns `HealthReport` dataclass
- No HTTP server dependency — health is pull-based via CLI or direct API call

### Structured Logging (RELY-04)
- Custom `JsonFormatter` (stdlib only, zero new deps) added to `src/yowo/logging.py`
- Structured fields per log record: `timestamp` (ISO8601), `level`, `event`, `engine_id`, `model`, `backend`, optionally `stream_id`, `frame_index`, `latency_ms`
- Log level configurable via `InferenceConfig(log_level="INFO")` and env var `YOWO_LOG_LEVEL`
- Default: standard Python logging format; opt-in JSON via `InferenceConfig(structured_logging=True)` or `YOWO_STRUCTURED_LOGGING=1`
- All engine lifecycle events (load, infer, error, health change, OOM recovery) emit structured log entries at appropriate levels

### Metrics Export (RELY-05)
- `engine.metrics` property already returns `EngineMetrics` snapshot
- Add `engine.export_metrics()` → `dict` (JSON-serializable, matches EngineMetrics fields)
- Add `engine.export_metrics_prometheus()` → `str` (Prometheus text exposition format, no prometheus_client dependency — generate format string directly)
- Prometheus metric names: `yowo_frames_total`, `yowo_errors_total`, `yowo_inference_mean_ms`, `yowo_inference_p95_ms`, `yowo_fps`, `yowo_uptime_seconds`, `yowo_memory_utilization`
- New CLI subcommand: `yowo metrics [--format json|prometheus]` — default JSON
- No HTTP server — user can expose `/metrics` endpoint via their own server using `export_metrics_prometheus()`

### Multi-Stream Isolation (STRM-01 to STRM-05)
- `FrameCollector` already has per-stream `_StreamEntry` with error tracking — extend with per-stream stats: `frames_processed`, `errors`, `bytes_queued`, `last_frame_time`
- Stream failure: on 3 consecutive read errors, call `remove_stream()` automatically, log structured warning with stream_id — other streams unaffected
- Stream reconnect: optional auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s); configurable via `StreamConfig(auto_reconnect=True)`
- Memory bound documentation: per-stream overhead measured and documented in README (queue size × frame bytes × buffer depth) — not enforced as hard cap (too complex without significant architecture change)
- Memory leak validation (STRM-03): `remove_stream()` must drain queue, join thread, and clear all references — verify with explicit gc + tracemalloc in test

### Claude's Discretion
- OOM check polling interval (starting point: 5s, may need tuning)
- Error threshold default (starting point: 10 errors)
- Exact Prometheus label scheme
- `JsonFormatter` field ordering
- Auto-reconnect backoff max interval

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BaseEngine.health` → `HealthStatus` enum (engine.py:235) — extend to `HealthReport` dataclass
- `MetricsCollector.snapshot()` → `EngineMetrics` (metrics/_collector.py:198) — base for Prometheus export
- `FrameCollector` + `_StreamEntry` (pipeline/_collector.py) — per-stream isolation already structured, needs auto-remove on failure
- `ThreadedFrameReader` (io/_reader.py) — RTSP reconnect pattern from Phase 1 reusable for stream auto-reconnect
- EventBus in `events/` — already used for detection/classification events; OOM recovery events fit naturally

### Established Patterns
- `InferenceConfig` / `ClassificationConfig` — add `log_level`, `structured_logging`, `error_threshold` fields
- Backend Protocol `infer()` — wrap call site in engine._run_gpu() with retry logic
- `_StreamEntry` pattern for per-stream state — extend with health fields, not replace

### Integration Points
- `engine.py` `BaseEngine._run_gpu()` — error retry wrapper goes here
- `engine.py` `BaseEngine.load()` — start OOM monitor daemon thread after successful load
- `metrics/_collector.py` — add `export_prometheus()` method to `MetricsCollector`
- `cli/_main.py` — add `yowo health` and `yowo metrics` subcommands
- New file: `src/yowo/logging.py` — `JsonFormatter`, `configure_logging()`

</code_context>

<specifics>
## Specific Ideas

- Health CLI exit codes should follow standard health check conventions (0=ok, 1=warn, 2=crit) for use in shell scripts and monitoring systems
- Prometheus format generated as raw string — user can serve it from any WSGI/ASGI endpoint (FastAPI, Flask, plain HTTP) without coupling to a specific framework
- OOM recovery should be silent/automatic in normal operation, only becoming visible in logs and health report — operators shouldn't need to react unless engine stays DEGRADED

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-reliability-and-multi-stream-scaling*
*Context gathered: 2026-03-07*
