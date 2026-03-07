# Phase 2: Reliability and Multi-Stream Scaling - Research

**Researched:** 2026-03-07
**Domain:** Python daemon threads, GPU memory monitoring, structured logging, Prometheus text exposition format, multi-stream lifecycle management
**Confidence:** HIGH

## Summary

Phase 2 adds operational hardening to the existing YOWO engine: OOM detection and recovery, transient GPU error retry, health report API, structured JSON logging, metrics export, and multi-stream stream isolation with auto-remove and reconnect. All implementation touches well-understood existing code — no new architectural surfaces are introduced.

The codebase already has the load-bearing infrastructure: `BaseEngine._run_gpu()` wraps `backend.infer()` in a single call site, `MetricsCollector` tracks `errors_total` and `frames_total`, `HealthStatus` enum is already used by `engine.health`, `FrameCollector._StreamEntry` holds per-stream state, and `EventBus` is wired for lifecycle events. Phase 2 is additive extension of existing patterns — not a redesign.

The two hard-to-get-wrong risks are (1) the OOM monitor daemon thread holding `_infer_lock` while reading GPU memory — it must NOT acquire the lock; GPU memory reads are thread-safe without it — and (2) `remove_stream()` must fully drain the queue and join the bridge thread before clearing references to avoid tracemalloc false positives in memory-leak tests.

**Primary recommendation:** Implement sequentially as five orthogonal units: [1] OOM monitor daemon, [2] GPU error retry in `_run_gpu`, [3] `HealthReport` dataclass + CLI, [4] `JsonFormatter` + `configure_logging()`, [5] `export_metrics_prometheus()` + CLI. Each unit has independent tests. Multi-stream isolation (STRM-01 to STRM-05) is the sixth unit extending `FrameCollector`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### OOM Detection and Recovery
- Use `torch.cuda.memory_reserved() / torch.cuda.get_device_properties().total_memory` to track GPU memory utilization continuously
- Three-tier recovery ladder (executed in order, each reported via EventBus):
  1. **80% utilization** → halve current batch size (minimum 1)
  2. **90% utilization** → attempt precision fallback: FP32 → FP16 if backend supports it
  3. **95% utilization** → evict lowest-activity streams (fewest frames in last 60s) until utilization drops below 85%
- Memory check runs on a background daemon thread (check interval: 5s) — not on inference hot path
- CPU-only deployments: skip OOM check (not applicable); ONNX/TensorRT backends: use psutil for system RAM check instead
- Engine transitions to DEGRADED health state when any recovery action is taken; returns to READY when utilization drops below 75% and batch size is restored

#### GPU Error Recovery (RELY-02)
- Wrap `backend.infer()` calls in try/except for `torch.cuda.OutOfMemoryError`, `RuntimeError` (GPU errors), and `Exception`
- On transient error: log structured warning, retry up to 3 times with exponential backoff (100ms, 200ms, 400ms)
- If all 3 retries fail: emit error event, increment `errors_total` counter, return empty result for that frame — do NOT crash engine
- If `errors_total >= error_threshold` (configurable, default 10): set engine DEGRADED
- Full engine restart is triggered by user via `engine.close()` + `engine.load()` — no automatic full restart

#### Health Check API (RELY-03)
- `engine.health` property already exists returning `HealthStatus` enum — extend to `HealthReport` dataclass
- `HealthReport` fields: `status` (ready/degraded/unhealthy/starting/closed), `uptime_s`, `errors_total`, `frames_total`, `memory_pct` (0.0–1.0, None for CPU), `stream_count` (for pipeline mode), `batch_size_current`, `precision_current`
- New CLI subcommand: `yowo health --engine-metrics` — prints HealthReport as JSON; exit code 0=ready, 1=degraded, 2=unhealthy
- Programmatic: `engine.health_report()` method returns `HealthReport` dataclass
- No HTTP server dependency — health is pull-based via CLI or direct API call

#### Structured Logging (RELY-04)
- Custom `JsonFormatter` (stdlib only, zero new deps) added to `src/yowo/logging.py`
- Structured fields per log record: `timestamp` (ISO8601), `level`, `event`, `engine_id`, `model`, `backend`, optionally `stream_id`, `frame_index`, `latency_ms`
- Log level configurable via `InferenceConfig(log_level="INFO")` and env var `YOWO_LOG_LEVEL`
- Default: standard Python logging format; opt-in JSON via `InferenceConfig(structured_logging=True)` or `YOWO_STRUCTURED_LOGGING=1`
- All engine lifecycle events (load, infer, error, health change, OOM recovery) emit structured log entries at appropriate levels

#### Metrics Export (RELY-05)
- `engine.metrics` property already returns `EngineMetrics` snapshot
- Add `engine.export_metrics()` → `dict` (JSON-serializable, matches EngineMetrics fields)
- Add `engine.export_metrics_prometheus()` → `str` (Prometheus text exposition format, no prometheus_client dependency — generate format string directly)
- Prometheus metric names: `yowo_frames_total`, `yowo_errors_total`, `yowo_inference_mean_ms`, `yowo_inference_p95_ms`, `yowo_fps`, `yowo_uptime_seconds`, `yowo_memory_utilization`
- New CLI subcommand: `yowo metrics [--format json|prometheus]` — default JSON
- No HTTP server — user can expose `/metrics` endpoint via their own server using `export_metrics_prometheus()`

#### Multi-Stream Isolation (STRM-01 to STRM-05)
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STRM-01 | Multi-stream pipeline handles 100+ concurrent RTSP/video streams without crashing on CUDA server | FrameCollector shared-queue design already scales; auto-remove on 3 consecutive errors prevents cascade failures |
| STRM-02 | Per-stream memory overhead is bounded and documented per device | Overhead = `max_queue_size × H × W × 3 bytes × buffer_depth`; document in README with measured values |
| STRM-03 | No memory leaks over 24-hour sustained operation with 50+ streams | `remove_stream()` must drain queue + join bridge thread; verify with tracemalloc in unit test |
| STRM-04 | Frame drop rate stays below 5% at target stream count per device | `FrameDropPolicy.LATEST` already in place; per-stream `frames_dropped` counter needed for measurement |
| STRM-05 | Pipeline gracefully handles individual stream disconnection without affecting other streams | Extend `_StreamEntry` with consecutive-error counter; auto-remove on threshold |
| RELY-01 | Engine detects approaching OOM and reduces batch size or drops precision before crash | Background daemon thread reading `torch.cuda.memory_reserved()`; three-tier recovery ladder |
| RELY-02 | Engine recovers from transient GPU errors without full restart | Retry wrapper in `_run_gpu()` with exponential backoff (100ms, 200ms, 400ms); return empty result after 3 failures |
| RELY-03 | Health check API reports engine status queryable via CLI or programmatic API | `HealthReport` dataclass extending existing `HealthStatus`; `yowo health` CLI subcommand |
| RELY-04 | Structured JSON logging with configurable log levels for all engine operations | `JsonFormatter` in `src/yowo/logging.py`; zero new deps (stdlib `logging.Formatter`) |
| RELY-05 | Metrics export in Prometheus-compatible format or JSON | `export_metrics_prometheus()` generates Prometheus text format as string; `yowo metrics` CLI |
</phase_requirements>

## Standard Stack

### Core (all stdlib — zero new production dependencies)
| Module | Purpose | Why Standard |
|--------|---------|--------------|
| `threading` | OOM monitor daemon thread, `_infer_lock` | Already used throughout codebase |
| `time` | Exponential backoff sleeps, uptime calculation | Already used in `MetricsCollector` |
| `logging` | `JsonFormatter(logging.Formatter)` subclass | Python stdlib, zero deps |
| `dataclasses` | `HealthReport` frozen dataclass | Already used for `EngineMetrics` |
| `json` | `export_metrics()` dict serialization | stdlib |

### Supporting (optional, already declared)
| Module | Purpose | When to Use |
|--------|---------|-------------|
| `torch.cuda` | `memory_reserved()`, `get_device_properties()`, `OutOfMemoryError` | CUDA backend only; guarded by `try/import` |
| `psutil` | System RAM check for CPU/ONNX/TensorRT backends | Already available in dev env; guard with try/import |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled Prometheus string | `prometheus_client` | Third-party dep; format is simple enough to generate directly as locked |
| `threading.Timer` for OOM poll | `threading.Thread` with `Event.wait(timeout)` | Timer is one-shot; daemon thread with Event.wait loops naturally |

**Installation:** No new production dependencies. All stdlib.

## Architecture Patterns

### Recommended Project Structure (new files only)
```
src/yowo/
├── logging.py              # NEW: JsonFormatter, configure_logging()
└── (all other files extend existing modules)
```

### Pattern 1: OOM Monitor Daemon Thread
**What:** Background thread started in `BaseEngine._finalize_load()`, stopped in `BaseEngine.close()`
**When to use:** After `backend.load()` succeeds; only when `DeviceType.CUDA` detected
**Example:**
```python
# src/yowo/engine.py — inside BaseEngine
def _start_oom_monitor(self) -> None:
    """Start background OOM monitor daemon. No-op on CPU backends."""
    if not self._is_cuda:
        return
    self._oom_stop = threading.Event()
    self._oom_thread = threading.Thread(
        target=self._oom_monitor_loop,
        name="yowo-oom-monitor",
        daemon=True,
    )
    self._oom_thread.start()

def _oom_monitor_loop(self) -> None:
    while not self._oom_stop.wait(timeout=5.0):  # 5s interval (Claude discretion)
        try:
            reserved = torch.cuda.memory_reserved(self._gpu_index)
            total = torch.cuda.get_device_properties(self._gpu_index).total_memory
            pct = reserved / total
            self._apply_oom_recovery(pct)
        except Exception:
            logger.exception("OOM monitor error")
```

### Pattern 2: GPU Error Retry Wrapper in `_run_gpu`
**What:** Wrap `self._backend.infer(tensor)` call in try/except with exponential backoff retry
**When to use:** Every inference call; replaces bare `self._backend.infer(tensor)` at engine.py:495
**Example:**
```python
def _run_gpu(self, tensor, frames):
    with self._infer_lock:
        if self._feature_cache is not None and frames:
            sid = frames[0].source_id
            if sid:
                self._backend.set_source_id(sid)
        t0 = time.perf_counter()
        raw_output = self._infer_with_retry(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._metrics.record_inference(elapsed_ms, batch_size=tensor.batch_size, frame_time=t0)
    return raw_output, elapsed_ms

def _infer_with_retry(self, tensor):
    delays = [0.1, 0.2, 0.4]
    for attempt, delay in enumerate(delays):
        try:
            return self._backend.infer(tensor)
        except Exception as exc:
            if attempt == len(delays) - 1:
                self._metrics.record_error()
                self._event_bus.emit("error", exc)
                logger.warning("All retries failed: %s", exc)
                return np.zeros((1, 0, 6), dtype=np.float32)  # empty result
            logger.warning("Transient GPU error (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay)
```

### Pattern 3: HealthReport Dataclass
**What:** Extend `engine.health` (returns `HealthStatus` enum) with `engine.health_report()` (returns `HealthReport` dataclass)
**When to use:** Programmatic health query; plumbed into `yowo health` CLI
**Example:**
```python
# src/yowo/types.py or engine.py
@dataclass(frozen=True, slots=True)
class HealthReport:
    status: HealthStatus
    uptime_s: float
    errors_total: int
    frames_total: int
    memory_pct: float | None  # None on CPU
    stream_count: int
    batch_size_current: int
    precision_current: str

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "uptime_s": round(self.uptime_s, 2),
            "errors_total": self.errors_total,
            "frames_total": self.frames_total,
            "memory_pct": self.memory_pct,
            "stream_count": self.stream_count,
            "batch_size_current": self.batch_size_current,
            "precision_current": self.precision_current,
        }
```

### Pattern 4: JsonFormatter (stdlib only)
**What:** `logging.Formatter` subclass that emits JSON lines; wired by `configure_logging()`
**When to use:** When `InferenceConfig(structured_logging=True)` or `YOWO_STRUCTURED_LOGGING=1`
**Example:**
```python
# src/yowo/logging.py
import json, logging, datetime

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "event": record.getMessage(),
        }
        # Optional extra fields injected via logger.warning("msg", extra={...})
        for key in ("engine_id", "model", "backend", "stream_id", "frame_index", "latency_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)

def configure_logging(level: str = "INFO", structured: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if structured else logging.Formatter())
    root = logging.getLogger("yowo")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
```

### Pattern 5: Prometheus Text Exposition Format
**What:** Hand-generated string following the Prometheus text format (no prometheus_client dep)
**When to use:** `engine.export_metrics_prometheus()`; passed through to user's `/metrics` endpoint
**Example:**
```python
def export_metrics_prometheus(self) -> str:
    m = self.metrics
    lines = [
        "# HELP yowo_frames_total Total frames processed",
        "# TYPE yowo_frames_total counter",
        f"yowo_frames_total {m.frames_total}",
        "# HELP yowo_errors_total Total inference errors",
        "# TYPE yowo_errors_total counter",
        f"yowo_errors_total {m.errors_total}",
        "# HELP yowo_inference_mean_ms Mean inference latency ms",
        "# TYPE yowo_inference_mean_ms gauge",
        f"yowo_inference_mean_ms {m.inference_mean_ms:.3f}",
        "# HELP yowo_inference_p95_ms P95 inference latency ms",
        "# TYPE yowo_inference_p95_ms gauge",
        f"yowo_inference_p95_ms {m.inference_p95_ms:.3f}",
        "# HELP yowo_fps Frames per second",
        "# TYPE yowo_fps gauge",
        f"yowo_fps {m.fps:.2f}",
        "# HELP yowo_uptime_seconds Engine uptime seconds",
        "# TYPE yowo_uptime_seconds counter",
        f"yowo_uptime_seconds {m.uptime_s:.1f}",
    ]
    return "\n".join(lines) + "\n"
```

### Pattern 6: `_StreamEntry` Consecutive-Error Tracking
**What:** Add `consecutive_errors: int` field to `_StreamEntry`; auto-remove on threshold 3
**When to use:** In `_run_bridge()` error handling path
**Example:**
```python
class _StreamEntry:
    __slots__ = (
        "bridge", "consecutive_errors", "error", "frames_processed",
        "last_frame_time", "reader", "source", "state", "stop_event"
    )

    def __init__(self, reader, source):
        ...
        self.consecutive_errors: int = 0
        self.frames_processed: int = 0
        self.last_frame_time: float = 0.0
```

### Anti-Patterns to Avoid
- **Acquiring `_infer_lock` from the OOM monitor thread:** GPU memory reads (`torch.cuda.memory_reserved()`) are safe to call from any thread — no lock needed. Acquiring `_infer_lock` in the daemon would deadlock.
- **Auto-restarting the engine on sustained OOM:** The decision specifies user-driven restart via `engine.close()` + `engine.load()`. Auto-restart would mask root cause and complicate lifecycle.
- **Calling `sys.exit()` from the OOM monitor thread:** Daemon threads must not call `sys.exit()`. Signal degraded health; let the main application decide.
- **Blocking `remove_stream()` indefinitely:** `bridge.join(timeout=5.0)` already in `_stop_entry()`. Do not increase timeout; log a warning and move on if thread doesn't exit.
- **Using `time.sleep()` in the OOM daemon instead of `Event.wait()`:** `Event.wait(timeout=5.0)` returns immediately when `_oom_stop` is set, enabling fast clean shutdown.
- **Adding prometheus_client as a production dep:** Format is simple; avoid the dep as locked.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe integer counters | Custom atomic class | Plain Python `int` with GIL / CPython guarantee | `MetricsCollector` comment explicitly covers this; GIL makes `+= 1` atomic in CPython |
| Rolling percentile history | Custom sorted structure | Existing `_RollingHistogram` (`deque(maxlen=1000)`) | Already tested, < 5µs per frame overhead |
| Health state machine | Custom FSM class | `HealthStatus` enum + `engine.health` property logic | Already implemented; just extend to `HealthReport` |
| JSON log serialization | `orjson` or `structlog` | `json.dumps()` in `JsonFormatter` | Zero new deps; stdlib is sufficient for log-path performance |
| Prometheus client | `prometheus_client` | Hand-rolled text format string | Single-method export; format is documented and stable |

**Key insight:** Every "problem" in Phase 2 already has 80% of its infrastructure in the existing codebase. The correct strategy is surgical extension, not new subsystems.

## Common Pitfalls

### Pitfall 1: OOM monitor reads stale `_batch_size` after recovery
**What goes wrong:** The OOM monitor halves `self._batch_size`, but `PreprocessBuffer` was pre-allocated with the original batch size. Inference still sends batch_size=original tensors because the preprocess path uses `self._preprocess_buf.capacity` as the upper bound.
**Why it happens:** `PreprocessBuffer` is allocated in `_finalize_load()` once; `_batch_size` change at runtime isn't reflected.
**How to avoid:** OOM recovery must also reallocate `_preprocess_buf = PreprocessBuffer(new_batch_size, target_size)` or clamp inference to `min(len(frames), self._batch_size)` in `_run_batch`.
**Warning signs:** Frame batches after OOM recovery are still the old batch size; memory doesn't decrease.

### Pitfall 2: `remove_stream()` called from bridge thread causes deadlock
**What goes wrong:** `_run_bridge()` detects 3 consecutive errors and calls `collector.remove_stream(stream_id)` while holding no lock — but `remove_stream()` acquires `self._lock`, which is fine. However, `_stop_entry()` calls `entry.bridge.join(timeout=5.0)`, which is the same bridge thread calling it — deadlock.
**Why it happens:** A thread cannot `join()` itself.
**How to avoid:** Auto-remove must be done from the bridge thread by setting a flag, then posting a sentinel to the shared queue. The `FrameCollector` or a monitoring thread handles actual `remove_stream()` asynchronously, OR the bridge signals via a separate auto-remove queue and the `__iter__` loop calls `remove_stream` outside the bridge context.
**Warning signs:** `bridge.join()` hangs for 5 seconds on every auto-remove.

### Pitfall 3: tracemalloc false positives in memory-leak tests
**What goes wrong:** `remove_stream()` returns, but the bridge thread is still running for up to 5 seconds (bridge join timeout). During this window, tracemalloc reports live allocations from the bridge thread as leaks.
**Why it happens:** `_stop_entry()` joins with timeout=5.0 and continues if thread doesn't exit. Thread may still hold refs to `entry.reader` and the shared queue.
**How to avoid:** In leak tests, call `remove_stream()` then explicitly wait for bridge to die: `entry.bridge.join(timeout=6.0)` before taking tracemalloc snapshot. Or ensure bridge exits promptly via `entry.stop_event.set()` then `entry.reader.stop()`.

### Pitfall 4: `HealthReport.stream_count` is meaningless for single-stream engines
**What goes wrong:** `BaseEngine` has no `FrameCollector`; `stream_count` is always 0 unless explicitly tracked.
**Why it happens:** `FrameCollector` is a separate class from `BaseEngine`; single-stream usage via `engine.stream(source)` doesn't go through `FrameCollector`.
**How to avoid:** `stream_count` returns 0 for non-pipeline mode (single source). Document this in `HealthReport` docstring. For pipeline usage, the user wires `FrameCollector` externally.

### Pitfall 5: Prometheus text format without trailing newline
**What goes wrong:** Prometheus scraper rejects exposition that doesn't end with `\n`.
**Why it happens:** `"\n".join(lines)` omits the final newline.
**How to avoid:** Always end with `"\n".join(lines) + "\n"`.

### Pitfall 6: `configure_logging()` adds duplicate handlers on repeated calls
**What goes wrong:** Calling `configure_logging()` twice (e.g., in tests) adds a second `StreamHandler`, duplicating all log output.
**Why it happens:** `logging.getLogger("yowo").addHandler()` accumulates handlers.
**How to avoid:** `configure_logging()` must call `root.handlers.clear()` before `addHandler()`.

## Code Examples

### OOM Recovery Three-Tier Ladder
```python
# In BaseEngine._apply_oom_recovery(pct: float) -> None
_OOM_TIER1 = 0.80  # halve batch size
_OOM_TIER2 = 0.90  # FP32 → FP16
_OOM_TIER3 = 0.95  # evict lowest-activity streams
_OOM_CLEAR  = 0.75  # return to READY

def _apply_oom_recovery(self, pct: float) -> None:
    if pct < _OOM_CLEAR and self._oom_recovering:
        self._oom_recovering = False
        self._batch_size = self._original_batch_size
        self._health_state = HealthStatus.READY
        logger.info("OOM cleared (%.1f%%) — batch size restored", pct * 100)
        return
    if pct >= _OOM_TIER3:
        self._evict_lowest_activity_streams()
    elif pct >= _OOM_TIER2:
        self._try_precision_fallback()
    elif pct >= _OOM_TIER1:
        self._halve_batch_size()
```

### CLI Health Subcommand Exit Codes
```python
# src/yowo/cli/_main.py
@cli.command("health")
def health_command() -> None:
    """Report engine health status as JSON."""
    # Without a running engine, health is reported from config + process state.
    # For a live check, users instantiate engine and call engine.health_report().
    # CLI reports "closed" when called without a running engine.
    report = {"status": "closed", "message": "No running engine — use engine.health_report() programmatically"}
    click.echo(json.dumps(report, indent=2))
    sys.exit(2)  # unhealthy = 2
```

Note: The health CLI's primary use case is when called from a monitoring script against a running engine via programmatic API. The CLI alone is for convenience. See CONTEXT.md specifics section.

### StreamConfig for Auto-Reconnect
```python
# New dataclass in types.py or pipeline/_collector.py
@dataclass(frozen=True, slots=True)
class StreamConfig:
    auto_reconnect: bool = False
    max_consecutive_errors: int = 3
    reconnect_backoff_base_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `time.sleep(N)` in daemon threads | `Event.wait(timeout=N)` | Python 3.x standard | Fast shutdown without sleep duration wait |
| `prometheus_client` mandatory | Prometheus text format generated as string | Community pattern (2022+) | Zero dep for basic `/metrics` endpoint |
| Monolithic engine health as bool | `HealthStatus` StrEnum + `HealthReport` dataclass | Phase 1 established `HealthStatus` | Structured, JSON-serializable, forward-compatible |

**Deprecated/outdated:**
- `threading.Timer` for repeating tasks: use `Event.wait(timeout)` loop in daemon thread

## Open Questions

1. **OOM monitor on ONNX/TensorRT backends with psutil**
   - What we know: CONTEXT.md says "use psutil for system RAM check instead" for non-CUDA
   - What's unclear: What psutil memory metric maps best to "approaching OOM"? `psutil.virtual_memory().percent`?
   - Recommendation: Use `psutil.virtual_memory().percent / 100.0` as the utilization ratio, same three-tier thresholds. Document in code.

2. **Precision fallback for non-PyTorch backends**
   - What we know: Tier 2 recovery attempts FP32 → FP16 "if backend supports it"
   - What's unclear: ONNX/TensorRT backends have fixed precision post-export; FP16 fallback is a no-op there
   - Recommendation: Check `hasattr(self._backend, 'set_precision')` or add a `supports_dynamic_precision` property to the Backend Protocol. No-op gracefully if not supported; log a debug message.

3. **`stream_count` in single-stream engine context**
   - What we know: `BaseEngine` has no `FrameCollector` reference
   - What's unclear: Should `HealthReport.stream_count` be 1 when `engine.stream(source)` is active?
   - Recommendation: Return 0 for non-pipeline mode (simpler); document that `stream_count > 0` implies `run_pipeline()` usage.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio 0.24 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STRM-01 | 100+ streams without crash (shared queue O(1) dispatch) | unit | `uv run pytest tests/unit/test_collector.py -x -q` | ✅ exists (extend) |
| STRM-02 | Per-stream memory overhead documented | manual | n/a | ❌ Wave 0: measurement script |
| STRM-03 | No memory leaks on stream remove (tracemalloc) | unit | `uv run pytest tests/unit/test_collector.py::test_remove_stream_no_leak -x -q` | ❌ Wave 0 |
| STRM-04 | Frame drop rate tracked per stream | unit | `uv run pytest tests/unit/test_collector.py::test_frame_drop_stats -x -q` | ❌ Wave 0 |
| STRM-05 | Stream disconnect doesn't affect others | unit | `uv run pytest tests/unit/test_collector.py::test_stream_isolation -x -q` | ❌ Wave 0 |
| RELY-01 | OOM detection triggers batch size reduction | unit | `uv run pytest tests/unit/test_oom_monitor.py -x -q` | ❌ Wave 0 |
| RELY-02 | Transient GPU error → retry → empty result (no crash) | unit | `uv run pytest tests/unit/test_gpu_retry.py -x -q` | ❌ Wave 0 |
| RELY-03 | `engine.health_report()` returns `HealthReport`; `yowo health` exits 0 for ready | unit | `uv run pytest tests/unit/test_health_report.py -x -q` | ❌ Wave 0 |
| RELY-04 | `JsonFormatter` outputs valid JSON with required fields | unit | `uv run pytest tests/unit/test_logging.py -x -q` | ❌ Wave 0 |
| RELY-05 | `export_metrics_prometheus()` matches Prometheus text format | unit | `uv run pytest tests/unit/test_metrics_export.py -x -q` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_oom_monitor.py` — covers RELY-01 (OOM three-tier recovery with mocked `torch.cuda`)
- [ ] `tests/unit/test_gpu_retry.py` — covers RELY-02 (retry backoff, empty result on exhaustion, error counter increment)
- [ ] `tests/unit/test_health_report.py` — covers RELY-03 (`HealthReport` fields, CLI exit codes)
- [ ] `tests/unit/test_logging.py` — covers RELY-04 (`JsonFormatter`, `configure_logging`, no duplicate handlers)
- [ ] `tests/unit/test_metrics_export.py` — covers RELY-05 (Prometheus format, JSON export, CLI)
- [ ] New test methods in `tests/unit/test_collector.py` — covers STRM-03, STRM-04, STRM-05 (auto-remove on 3 errors, tracemalloc leak check, per-stream stats)

## Sources

### Primary (HIGH confidence)
- Direct codebase read — `src/yowo/engine.py`, `src/yowo/pipeline/_collector.py`, `src/yowo/metrics/_collector.py`, `src/yowo/events/__init__.py`, `src/yowo/types.py`, `src/yowo/config.py` — full implementation reviewed
- `.planning/phases/02-reliability-and-multi-stream-scaling/02-CONTEXT.md` — locked decisions reviewed in full

### Secondary (MEDIUM confidence)
- Prometheus text exposition format specification (stable since 2016): `# HELP`, `# TYPE`, metric line, trailing `\n` — well-established
- Python `logging.Formatter` subclassing pattern: stdlib documentation behavior, stable

### Tertiary (LOW confidence)
- OOM polling interval of 5s: reasonable starting point, empirical tuning needed at 100+ streams
- psutil `virtual_memory().percent` as CPU-OOM proxy: community convention, not formally specified for this use case

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all stdlib; torch.cuda API well-known
- Architecture: HIGH — six discrete extension points identified in existing code; no new surfaces
- Pitfalls: HIGH — identified from direct code inspection of existing `_run_gpu`, `_stop_entry`, bridge thread, logging handler patterns
- Test gaps: HIGH — all new files identified; existing test files confirmed

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable stdlib patterns; torch.cuda API stable within major version)
