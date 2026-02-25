# pipeline — Multi-Stream Inference Pipeline

Orchestrates N concurrent video/RTSP streams through a single `InferenceEngine` with batched inference, per-stream result routing, and health monitoring.

---

## Design Principles

- **Single-responsibility decomposition**: 4 modules with clear boundaries — collect, batch, infer, route. No God objects.
- **Composition over modification**: Reuses `ThreadedFrameReader` from `yowo.io` and `InferenceEngine` from `yowo.engine` without modifying either.
- **Positional index routing**: `detections[i]` maps to `batch[i]` — deterministic, no object identity contracts.
- **Thread-safe dynamic membership**: Streams can be added/removed while the pipeline is running.

---

## Module Structure

```
pipeline/
├── __init__.py       — public surface: run_pipeline(), re-exports
├── _collector.py     — FrameCollector: N-stream management + round-robin dispatch
├── _scheduler.py     — BatchScheduler: capacity/timeout batch formation
└── _router.py        — DetectionRouter: per-stream callback dispatch
```

---

## Data Flow

```
N sources → FrameCollector (ThreadedFrameReader per stream) → TaggedFrame
  → BatchScheduler (capacity/timeout flush) → list[TaggedFrame]
  → engine.detect([tagged.frame for tagged in batch]) → list[Detection]
  → DetectionRouter (positional index) → per-stream callbacks
```

---

## FrameCollector — `_collector.py`

```python
class FrameCollector:
    def __init__(self, max_queue_size: int = 2) -> None: ...

    def add_stream(
        self, stream_id: str, source: FrameSource, *, policy: FrameDropPolicy | None = None,
    ) -> None:
        """Register and start a new stream.
        Auto-selects LATEST for live sources, NONE for offline."""

    def remove_stream(self, stream_id: str) -> None:
        """Stop and remove a stream. Idempotent."""

    def __iter__(self) -> Iterator[TaggedFrame]:
        """Round-robin across active streams. Terminates when all exhausted/errored."""

    def close(self) -> None:
        """Stop all readers, close all sources. Idempotent."""

    # State inspection
    stream_states: dict[str, StreamState]     # per-stream health
    stream_errors: dict[str, BaseException]   # ERROR streams only
    stream_count: int                         # all registered streams
    active_count: int                         # RUNNING + RECONNECTING
```

### How it works

1. `add_stream()` creates a `ThreadedFrameReader` per source (bounded deque, daemon thread).
2. `__iter__` takes a snapshot of active streams, polls each with 50ms timeout (round-robin).
3. A reader returning `None` + `is_exhausted=True` transitions the stream to `STOPPED`.
4. Exceptions during `get()` are caught, stored, and the stream transitions to `ERROR`.
5. Iteration terminates when all streams are in a terminal state (`STOPPED` or `ERROR`).

---

## BatchScheduler — `_scheduler.py`

```python
class BatchScheduler:
    def __init__(
        self,
        upstream: Iterable[TaggedFrame],
        *,
        max_batch_size: int = 1,
        timeout_ms: float = 100.0,
        stop_event: threading.Event | None = None,
    ) -> None: ...

    def __iter__(self) -> BatchScheduler: ...
    def __next__(self) -> list[TaggedFrame]: ...

    def set_stop_event(self, event: threading.Event) -> None:
        """Attach external cancellation signal."""

    # Metrics
    max_batch_size: int      # configured limit
    timeout_ms: float        # configured flush timeout
    pending: int             # current unflushed count
```

### Flush triggers

| Trigger | Condition | Effect |
|---------|-----------|--------|
| Capacity | `len(batch) >= max_batch_size` | Immediate flush |
| Timeout | `elapsed >= timeout_ms` since first frame in batch | Flush partial batch |
| Exhaustion | Upstream raises `StopIteration` | Flush remaining frames, then stop |
| Cancellation | `stop_event.is_set()` | Flush partial batch (or stop if empty) |

The scheduler accepts `Iterable[TaggedFrame]` (not `Iterator`), calling `iter()` in `__iter__` — enabling reuse across multiple pipeline runs.

---

## DetectionRouter — `_router.py`

```python
class DetectionRouter:
    def register(
        self, stream_id: str, callback: Callable[[str, list[Detection]], None],
    ) -> None:
        """Register callback for stream. Overwrites previous."""

    def unregister(self, stream_id: str) -> None:
        """Remove callback. No-op if not registered."""

    def route(self, detections: list[Detection], batch: list[TaggedFrame]) -> None:
        """Match detections to streams by positional index, invoke callbacks."""
```

### Routing semantics

- `detections[i]` maps to `batch[i]` — guaranteed by `engine.detect()` returning one Detection per input frame in order.
- Detections are grouped by `stream_id`, then each callback receives its full `list[Detection]` in one invocation.
- Callbacks are invoked on a snapshot taken under the lock, so `register`/`unregister` inside a callback does not affect the current dispatch round.
- Unregistered streams are silently dropped (logged at DEBUG).

---

## run_pipeline — `__init__.py`

```python
def run_pipeline(
    engine: InferenceEngine,
    collector: FrameCollector,
    scheduler: BatchScheduler,
    router: DetectionRouter,
    *,
    stop_event: threading.Event | None = None,
) -> None:
    """Wire collector -> scheduler -> engine.detect() -> router.

    Blocking call that runs until all streams exhausted, stop_event set,
    or unhandled exception. Auto-disables feature cache for mixed-source safety.
    """
```

### Feature cache safety

`engine.detect()` keys its feature cache on `frames[0].source_id`. Mixed-source batches corrupt this key. `run_pipeline()` automatically clears and disables the feature cache, logging a warning.

---

## Usage

```python
from yowo import InferenceEngine
from yowo.pipeline import (
    BatchScheduler,
    DetectionRouter,
    FrameCollector,
    run_pipeline,
)

engine = InferenceEngine(...)
engine.load()

collector = FrameCollector(max_queue_size=4)
collector.add_stream("cam-0", source_0)
collector.add_stream("cam-1", source_1)

scheduler = BatchScheduler(collector, max_batch_size=4, timeout_ms=50)
router = DetectionRouter()
router.register("cam-0", on_cam0)
router.register("cam-1", on_cam1)

run_pipeline(engine, collector, scheduler, router)
```

### With graceful shutdown

```python
import threading

stop = threading.Event()
# In another thread or signal handler: stop.set()

run_pipeline(engine, collector, scheduler, router, stop_event=stop)
```

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `types.py` | `Frame`, `Detection`, `TaggedFrame`, `StreamState`, `FrameDropPolicy` |
| Upstream | `io/_reader.py` | `ThreadedFrameReader` (composed by FrameCollector) |
| Upstream | `io/_source.py` | `FrameSource` protocol (passed to add_stream) |
| Upstream | `engine.py` | `InferenceEngine.detect()` (called by run_pipeline) |
| Downstream | `cli/` | Can wire pipeline from CLI arguments |
