# Architecture Research

**Domain:** Production CV inference platform — adaptive optimization, device benchmarking, 100+ stream scaling
**Researched:** 2026-03-07
**Confidence:** MEDIUM (patterns well-established in industry; integration with YOWO's specific layered architecture requires design work)

## System Overview

New components layer on top of YOWO's existing architecture. Three major subsystems are added: **Device Profiler**, **Adaptive Optimizer**, and **Scaled Stream Manager**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI / Convenience API                       │
│  yowo detect/classify/track/count + yowo bench + yowo autotune     │
├─────────────────────────────────────────────────────────────────────┤
│                         Engine Layer (existing)                     │
│  BaseEngine → DetectionEngine / ClassificationEngine                │
│  StreamingMixin → _stream_live / _stream_pipeline / _stream_sync   │
├──────────────┬──────────────────────────┬───────────────────────────┤
│   Adaptive   │    Scaled Stream         │    Device Profiler        │
│   Optimizer  │    Manager               │                           │
│              │                          │                           │
│ ┌──────────┐ │ ┌────────────────┐       │ ┌──────────────┐          │
│ │ Tuner    │ │ │ StreamPool     │       │ │ BenchRunner  │          │
│ └──────────┘ │ └────────────────┘       │ └──────────────┘          │
│ ┌──────────┐ │ ┌────────────────┐       │ ┌──────────────┐          │
│ │ Adaptor  │ │ │ LoadBalancer   │       │ │ ThermalMon   │          │
│ └──────────┘ │ └────────────────┘       │ └──────────────┘          │
│ ┌──────────┐ │ ┌────────────────┐       │ ┌──────────────┐          │
│ │ Profile  │ │ │ MemoryGuard    │       │ │ ProfileStore │          │
│ │ Store    │ │ └────────────────┘       │ └──────────────┘          │
│ └──────────┘ │ ┌────────────────┐       │                           │
│              │ │ AdaptiveBatcher│       │                           │
│              │ └────────────────┘       │                           │
├──────────────┴──────────────────────────┴───────────────────────────┤
│                    Existing Infrastructure                           │
│  backends/ | hardware/ | metrics/ | events/ | pipeline/ | config/   │
├─────────────────────────────────────────────────────────────────────┤
│                    Types / Errors                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| **BenchRunner** | Execute structured benchmarks: latency, throughput, memory, accuracy per backend/precision/batch-size combination on real hardware | hardware/, backends/, ProfileStore |
| **ThermalMonitor** | Poll GPU/CPU temperature and throttling state; emit events when thermal limits approach | hardware/, events/ |
| **ProfileStore** | Persist benchmark results and device profiles as JSON/SQLite; keyed by (device_fingerprint, model_spec, backend, precision) | BenchRunner, Tuner |
| **Tuner** | Analyze ProfileStore data to select optimal (backend, precision, batch_size) for a given device + model + workload pattern | ProfileStore, config/ |
| **Adaptor** | Runtime feedback loop: monitor MetricsCollector + ThermalMonitor, adjust batch_size / precision / frame_drop_policy in-flight | metrics/, ThermalMonitor, engine |
| **StreamPool** | Manage N concurrent stream workers with bounded concurrency; replace current FrameCollector for 100+ streams | io/, pipeline/ |
| **LoadBalancer** | Distribute streams across GPU instances or time-slice a single GPU; priority-based scheduling | StreamPool, backends/ |
| **MemoryGuard** | Track GPU/CPU memory watermarks; enforce limits; trigger stream shedding or precision downgrade before OOM | hardware/, Adaptor |
| **AdaptiveBatcher** | Extend BatchScheduler with dynamic batch sizing based on current throughput and queue depth | pipeline/_scheduler, metrics/ |

## Recommended Project Structure

New modules added to the existing `src/yowo/` tree:

```
src/yowo/
├── profiler/              # Device benchmarking and profiling (NEW)
│   ├── __init__.py        # Public API: run_benchmark(), DeviceProfile
│   ├── _bench.py          # BenchRunner — structured benchmark execution
│   ├── _thermal.py        # ThermalMonitor — temp/throttle polling
│   ├── _store.py          # ProfileStore — persist benchmark results
│   └── _fingerprint.py    # Device fingerprinting for cache keys
├── optimizer/             # Adaptive optimization (NEW)
│   ├── __init__.py        # Public API: AutoTuner, RuntimeAdaptor
│   ├── _tuner.py          # Tuner — offline optimal config selection
│   ├── _adaptor.py        # Adaptor — runtime feedback loop
│   └── _strategy.py       # Optimization strategies (latency-first, throughput-first)
├── scaler/                # 100+ stream scaling (NEW)
│   ├── __init__.py        # Public API: StreamPool, run_scaled_pipeline()
│   ├── _pool.py           # StreamPool — bounded concurrent stream management
│   ├── _balancer.py       # LoadBalancer — stream-to-GPU distribution
│   ├── _memory.py         # MemoryGuard — OOM prevention
│   └── _batcher.py        # AdaptiveBatcher — dynamic batch sizing
├── pipeline/              # (existing, extended)
│   ├── _collector.py      # FrameCollector (existing)
│   ├── _scheduler.py      # BatchScheduler (existing, AdaptiveBatcher wraps this)
│   ├── _router.py         # DetectionRouter (existing)
│   └── __init__.py        # run_pipeline() (existing)
├── metrics/               # (existing, extended)
│   ├── _collector.py      # MetricsCollector (existing — add memory/thermal fields)
│   └── __init__.py
├── hardware/              # (existing, extended)
│   ├── _thermal.py        # NEW: thermal sensor reading (nvidia-smi, sysfs, jtop)
│   └── ...existing...
└── ...existing modules...
```

### Structure Rationale

- **profiler/:** Isolated from engine hot path. Benchmarks are run offline or at startup, producing data that Tuner consumes. No import dependency on engine.
- **optimizer/:** Depends on profiler/ (reads ProfileStore) and metrics/ (reads runtime telemetry). The Adaptor hooks into engine via EventBus, not direct coupling.
- **scaler/:** Extends pipeline/ concepts to 100+ streams. StreamPool replaces FrameCollector at scale; AdaptiveBatcher wraps BatchScheduler. Backward compatible — existing pipeline/ API unchanged for small stream counts.

## Architectural Patterns

### Pattern 1: Offline Profile + Runtime Adapt (Two-Phase Optimization)

**What:** Separate optimization into an offline profiling phase (slow, thorough) and a runtime adaptation phase (fast, reactive). The offline phase runs benchmarks across backend/precision/batch combinations and stores results. The runtime phase uses those results as a starting point and makes incremental adjustments based on live metrics.

**When to use:** Always for adaptive optimization. This is how NVIDIA Triton, TensorRT, and TVM all work — calibration/profiling happens once, runtime adaptation is lightweight.

**Trade-offs:** Requires a profiling step before first deployment (10-30 minutes per device). But avoids expensive search during production serving.

**Example:**
```python
# Phase 1: Offline (run once per device)
profile = run_benchmark(
    model="yolo11n",
    device=get_hardware_profile(),
    sweep=["backend", "precision", "batch_size"],
)
store.save(profile)

# Phase 2: Runtime (continuous)
adaptor = RuntimeAdaptor(
    engine=engine,
    profile=store.load(device_fingerprint),
    strategy=Strategy.LATENCY_FIRST,
)
adaptor.start()  # subscribes to EventBus, adjusts config in-flight
```

### Pattern 2: Hierarchical Stream Management (Pool + Groups + Workers)

**What:** Organize 100+ streams into a three-level hierarchy: StreamPool (global resource manager) -> StreamGroup (logical groupings sharing a GPU or batch window) -> StreamWorker (per-source frame reader). The pool enforces global concurrency limits and memory budgets. Groups batch frames from related streams. Workers handle individual source I/O.

**When to use:** When stream count exceeds what a single BatchScheduler + FrameCollector can handle (roughly >16-32 streams on a single GPU based on DeepStream benchmarks).

**Trade-offs:** More complex than flat pipeline. But necessary because a single BatchScheduler with 100 upstream readers will either timeout-flush too often (wasting GPU cycles on small batches) or accumulate too many frames (memory pressure). Grouping allows tuning batch formation per GPU.

**Example:**
```python
# StreamPool manages lifecycle; groups handle batching
pool = StreamPool(max_streams=128, memory_budget_mb=4096)
for source_url in rtsp_sources:
    pool.add_stream(source_url)  # auto-assigns to StreamGroup

# Pool internally creates groups, each with its own AdaptiveBatcher
async for results in pool.run(engine):
    router.dispatch(results)
```

### Pattern 3: Circuit Breaker for Thermal/Memory Protection

**What:** Monitor GPU temperature and memory usage. When thresholds are crossed, progressively degrade: first reduce batch size, then lower precision, then shed lowest-priority streams. When conditions improve, gradually restore.

**When to use:** Edge devices (Jetson) where thermal throttling is a real risk, and any GPU server under sustained multi-stream load.

**Trade-offs:** Adds complexity to the control loop. But without it, Jetson devices will thermal-throttle unpredictably (clock drops 30-50%), and GPU servers will OOM-crash. This is a production necessity, not a nice-to-have.

**Example:**
```python
class MemoryGuard:
    WATERMARKS = {
        "normal": 0.70,    # below 70% — no action
        "warning": 0.85,   # 70-85% — reduce batch size
        "critical": 0.95,  # 85-95% — downgrade precision
        "emergency": 0.98, # >95% — shed streams
    }
```

### Pattern 4: EventBus-Driven Adaptation (Decoupled Control)

**What:** The Adaptor does not directly modify engine internals. Instead, it listens to metrics/thermal events via EventBus and emits "config_change" events that the engine consumes. This keeps the adaptation logic decoupled from the inference hot path.

**When to use:** Always. YOWO already has EventBus infrastructure. The Adaptor should be a subscriber/emitter, not a direct engine mutator.

**Trade-offs:** Slightly higher latency for config changes (event dispatch overhead ~microseconds). But dramatically simpler testing and zero risk of race conditions on the hot path.

## Data Flow

### Benchmark Results to Optimization Decisions

```
[BenchRunner]
    │ runs N configurations on real hardware
    ↓
[ProfileStore]  ←── JSON/SQLite keyed by (device_fingerprint, model, backend, precision, batch)
    │
    ↓
[Tuner]
    │ analyzes profiles, selects Pareto-optimal configs
    ↓
[OptimalConfig]  →  [Engine.load()]  (initial configuration)
    │
    ↓ (engine running)
[MetricsCollector] ──→ [Adaptor] ←── [ThermalMonitor]
                           │
                           ↓
                    [EventBus: "config_adjust"]
                           │
                           ↓
                    [Engine applies: batch_size, precision, frame_drop_policy]
```

### Scaled Stream Pipeline

```
[100+ RTSP Sources]
    │
    ↓
[StreamPool]
    │ assigns sources to StreamGroups (N groups, ~8-16 streams each)
    ↓
[StreamGroup_0]          [StreamGroup_1]          [StreamGroup_N]
    │                        │                        │
    ↓                        ↓                        ↓
[AdaptiveBatcher]        [AdaptiveBatcher]        [AdaptiveBatcher]
    │                        │                        │
    ↓                        ↓                        ↓
[LoadBalancer]  ←── distributes batches across GPU(s) or time-slices
    │
    ↓
[Engine.detect(batch)]  ←── GPU inference (serialized via infer_lock)
    │
    ↓
[DetectionRouter]  →  per-stream callbacks
```

### Runtime Adaptation Loop

```
                    ┌──────────────────────────┐
                    │     Feedback Loop         │
                    │                          │
  metrics.snapshot()│   ┌───────────────┐      │
  ─────────────────→│   │   Adaptor     │      │
                    │   │               │      │
  thermal.read()    │   │ if fps < target:     │
  ─────────────────→│   │   increase batch     │
                    │   │ if temp > 80C:       │
  memory.usage()    │   │   decrease batch     │
  ─────────────────→│   │ if mem > 85%:        │
                    │   │   downgrade prec     │
                    │   └───────┬───────┘      │
                    │           │ emit          │
                    │           ↓               │
                    │   EventBus("config_adj")  │
                    │           │               │
                    │           ↓               │
                    │   Engine applies change   │
                    └──────────────────────────┘
                    (runs every 1-5 seconds)
```

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1-8 streams | Existing pipeline/ (FrameCollector + BatchScheduler) works. No changes needed. |
| 8-32 streams | AdaptiveBatcher improves throughput 15-30% over fixed BatchScheduler. MemoryGuard recommended for edge. |
| 32-100 streams | StreamPool with StreamGroups required. Single FrameCollector cannot efficiently batch from this many sources. LoadBalancer distributes across time slices. |
| 100+ streams | Full StreamPool + LoadBalancer + MemoryGuard. Multi-GPU if available. Stream priority shedding under load. |

### Scaling Priorities

1. **First bottleneck: GPU saturation.** A single GPU can process roughly 16-30 concurrent 720p streams at YOLO11n (TensorRT FP16) based on DeepStream reference numbers. Beyond that, frames queue faster than inference drains them. Fix: AdaptiveBatcher increases effective batch size; LoadBalancer time-slices or distributes to multiple GPUs.
2. **Second bottleneck: CPU decode.** OpenCV VideoCapture is single-threaded per stream. 100 streams each decoding at 30fps = 3000 decode ops/second. Fix: StreamPool bounds concurrent decoders; frame dropping (LATEST policy) discards stale frames.
3. **Third bottleneck: Memory.** Each 720p frame = ~2.7MB (1280x720x3). 100 streams with 2 buffered frames each = ~540MB just for raw frames, plus GPU tensor memory. Fix: MemoryGuard enforces watermarks; StreamPool caps active decoders.

## Anti-Patterns

### Anti-Pattern 1: Benchmarking in the Inference Hot Path

**What people do:** Run micro-benchmarks during live inference to "adaptively" find optimal settings.
**Why it's wrong:** Benchmark warmup, outlier detection, and statistical significance require hundreds of iterations. Running this during serving causes latency spikes and throughput drops.
**Do this instead:** Profile offline (or at startup before accepting streams). Use lightweight runtime metrics (FPS, p95 latency) for incremental adaptation, not full benchmarks.

### Anti-Pattern 2: Global Batch Size for All Streams

**What people do:** Use one batch_size for the entire pipeline regardless of stream count or GPU load.
**Why it's wrong:** Optimal batch size depends on: number of active streams (determines queue fill rate), GPU memory available (larger batches need more VRAM), latency target (larger batches increase per-frame latency). A fixed batch of 8 is too small for 100 streams (GPU idle between batches) and too large for 2 streams (frames sit waiting to fill the batch).
**Do this instead:** AdaptiveBatcher that adjusts batch size based on queue depth and throughput measurements. Start with profiled optimal, adjust at runtime.

### Anti-Pattern 3: Precision Downgrade Without Accuracy Validation

**What people do:** Auto-switch from FP16 to INT8 under memory pressure without checking mAP impact.
**Why it's wrong:** INT8 can lose 1-5% mAP depending on the model. Some deployments cannot tolerate this (safety-critical applications). Switching precision silently violates accuracy SLAs.
**Do this instead:** BenchRunner validates accuracy at each precision level during profiling. Adaptor only downgrades to precisions that passed accuracy thresholds in the offline profile.

### Anti-Pattern 4: Unbounded Stream Acceptance

**What people do:** Accept all incoming streams without capacity checks.
**Why it's wrong:** Leads to cascading degradation — every stream gets worse as the GPU is overloaded, rather than some streams being healthy.
**Do this instead:** StreamPool enforces a hard cap based on profiled capacity. New streams above cap are rejected with a clear error or queued. MemoryGuard can trigger stream shedding for lowest-priority streams under pressure.

## Integration Points with Existing YOWO Architecture

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Profiler -> Hardware | Direct import | Extends HardwareProfile with thermal data; reads GPU memory via existing Device dataclass |
| Profiler -> Backends | Direct import | BenchRunner creates backend instances to benchmark; uses existing InferenceBackend Protocol |
| Optimizer -> Metrics | Direct import | Adaptor reads MetricsCollector.snapshot() on a timer |
| Optimizer -> Events | EventBus pub/sub | Adaptor emits "config_adjust" events; engine subscribes |
| Optimizer -> Profiler | Direct import | Tuner reads ProfileStore |
| Scaler -> Pipeline | Composition | StreamPool composes FrameCollector instances; AdaptiveBatcher wraps BatchScheduler |
| Scaler -> Engine | Direct import | StreamPool calls engine.detect() same as existing run_pipeline() |
| MemoryGuard -> Hardware | Direct import | Polls GPU memory via hardware/ module |
| ThermalMonitor -> Hardware | Direct import | New _thermal.py in hardware/ reads nvidia-smi/sysfs/jtop |

### Key Protocol Extensions

| Protocol | Extension | Rationale |
|----------|-----------|-----------|
| `InferenceBackend` | No change needed | BenchRunner uses existing load/infer/unload cycle |
| `MetricsCollector` | Add `memory_used_mb`, `gpu_temp_c` fields to `EngineMetrics` | Adaptor needs these for decisions; low overhead (read cached values) |
| `EventBus` | Add `EVENT_CONFIG_ADJUST` constant | New event type for optimizer -> engine communication |
| `InferenceConfig` | Add `autotune: bool`, `adaptation_strategy: str` fields | Enable/disable adaptive features; backward compatible with defaults |

## Build Order (Dependencies)

Build order is constrained by data flow dependencies:

```
Phase A: Device Profiler (no dependencies on new code)
    hardware/_thermal.py        → reads nvidia-smi/sysfs
    profiler/_fingerprint.py    → hashes HardwareProfile
    profiler/_bench.py          → uses existing backends + hardware
    profiler/_store.py          → JSON persistence
    profiler/_thermal.py        → wraps hardware/_thermal.py

Phase B: Adaptive Optimizer (depends on Phase A: ProfileStore)
    optimizer/_tuner.py         → reads ProfileStore, outputs OptimalConfig
    optimizer/_strategy.py      → latency-first vs throughput-first logic
    optimizer/_adaptor.py       → runtime loop using MetricsCollector + EventBus

Phase C: Scaled Stream Manager (depends on Phase B: Adaptor for full benefit)
    scaler/_memory.py           → GPU/CPU memory watermarks
    scaler/_batcher.py          → AdaptiveBatcher wrapping BatchScheduler
    scaler/_pool.py             → StreamPool managing N StreamGroups
    scaler/_balancer.py         → stream-to-GPU distribution

Phase D: Integration + CLI
    CLI commands: yowo bench, yowo autotune
    InferenceConfig extensions
    End-to-end: 100 streams + auto-tuned config + runtime adaptation
```

**Phase A has zero coupling to new code** — it only extends hardware/ and creates profiler/. This makes it safe to build and validate independently.

**Phase B depends on Phase A's ProfileStore** for offline tuning, and on existing metrics/ for runtime adaptation. The Tuner is useful standalone (offline config recommendation). The Adaptor adds runtime value.

**Phase C depends on Phase B** for full benefit (AdaptiveBatcher needs Adaptor feedback), but the StreamPool and MemoryGuard provide value even without adaptation (static config, just at higher scale).

**Phase D ties everything together** with CLI exposure and config integration.

## Sources

- [NVIDIA Triton Inference Server Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html) — dynamic batching, instance groups, model scheduling
- [Triton Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) — max_queue_delay, preferred batch sizes
- [NVIDIA DeepStream Multi-Camera Analytics](https://developer.nvidia.com/blog/multi-camera-large-scale-iva-deepstream-sdk/) — 100+ stream pipeline architecture, hardware-accelerated decode
- [DeepStream Performance](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html) — stream density benchmarks per GPU
- [Multi-Stream GPU Scheduling (ACM)](https://dl.acm.org/doi/10.1145/3677378) — DRL-based multi-stream scheduling on edge devices
- [GPU Scheduling for Large-Scale Inference](https://medium.com/@fahey_james/gpu-scheduling-for-large-scale-inference-beyond-more-gpus-dcac81f952a2) — adaptive batching, memory management patterns
- [AI Inference Optimization (RunPod)](https://www.runpod.io/articles/guides/ai-inference-optimization-achieving-maximum-throughput-with-minimal-latency) — precision selection, batch tuning, monitoring
- [Edge AI Memory Benchmarking](https://www.prompts.ai/blog/5-steps-to-benchmark-edge-ai-memory-utilization) — memory profiling methodology for edge devices
- [Reef: Microsecond-scale GPU Preemption](https://ipads.se.sjtu.edu.cn/_media/publications/hanosgi22.pdf) — concurrent DNN inference scheduling

---
*Architecture research for: YOWO adaptive optimization, device benchmarking, 100+ stream scaling*
*Researched: 2026-03-07*
