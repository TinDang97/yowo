# Project Research Summary

**Project:** YOWO Production Hardening
**Domain:** Production CV inference platform -- edge devices, multi-stream scaling, adaptive optimization
**Researched:** 2026-03-07
**Confidence:** MEDIUM-HIGH

## Executive Summary

YOWO v2.3.0 is a mature inference library (1615 tests, 5 backends, 10 model variants, detection + classification + tracking + ReID). The next milestone is production hardening: proving the system works correctly on real hardware, scaling to 100+ concurrent streams, and adding adaptive self-optimization. This is not a greenfield build -- it is hardening an existing, well-tested codebase for sustained production deployment on Jetson, Intel NUC, and CUDA GPU servers.

The recommended approach follows a strict dependency chain: first validate correctness (mAP parity with ultralytics, export validation on real devices), then build the measurement infrastructure (benchmarking, profiling, thermal monitoring), then scale (100+ streams with memory guards), and finally add adaptive optimization (auto-tuning, runtime load adaptation). This order is non-negotiable because each phase produces data or infrastructure that later phases consume. Auto-tuning without benchmarks is guesswork; scaling without memory guards is a crash waiting to happen; adaptive optimization without correctness validation risks silently degrading accuracy.

The primary risks are: OpenCV RTSP memory leaks killing long-running deployments (switch to subprocess FFmpeg), TensorRT engine portability failures (ship ONNX and build on-device), unbounded GPU memory growth at 100+ streams (global memory budget with stream shedding), and thermal throttling destroying sustained throughput on edge devices (temperature-aware feedback loop). All four have well-documented mitigations. The stack additions are lightweight -- psutil, pytest-benchmark, optuna, and uvloop are the main new dependencies. The architecture extends YOWO's existing module structure with three new packages (profiler/, optimizer/, scaler/) that layer on top of existing infrastructure without modifying the hot path.

## Key Findings

### Recommended Stack

The existing inference stack (PyTorch, ONNX, TensorRT, OpenVINO, CoreML) is complete. New tooling focuses on measurement, monitoring, and optimization. All recommendations are standard, well-maintained tools with no exotic dependencies.

**Core technologies:**
- **pytest-benchmark**: Inference latency/throughput regression testing integrated with existing pytest suite
- **psutil + pynvml + jetson-stats**: Cross-platform hardware/thermal monitoring (psutil for CPU/memory, pynvml for NVIDIA GPU, jtop for Jetson-specific telemetry)
- **optuna**: Bayesian search for optimal backend/precision/batch-size per device with SQLite persistence
- **uvloop**: Drop-in asyncio replacement for 2-4x faster I/O coordination at 100+ streams
- **fabric**: SSH-based remote test execution on real devices (Jetson, NUC, GPU servers)
- **py-spy + memray**: Production-safe CPU profiling and memory leak detection (zero-overhead, attach-to-process)
- **prometheus-client**: Metrics export for production monitoring at scale

**Critical version notes:** memray is Linux-only (use on target devices, not macOS dev); uvloop is Linux/macOS only; jetson-stats only installs on Jetson hardware.

### Expected Features

**Must have (table stakes -- P1):**
- Correctness validation vs ultralytics (mAP parity on COCO across all 10 variants)
- Multi-format export validation on real hardware (ONNX/TRT/OpenVINO on each target device)
- Per-format benchmark mode (mAP + FPS + model size comparison table)
- Multi-stream stress testing at 100+ concurrent streams
- Graceful OOM/error recovery (memory monitoring + degradation, not crashes)
- Thermal-aware throttling on edge devices
- Health check / readiness endpoint
- Warm-up validation on load

**Should have (differentiators -- P2):**
- Adaptive auto-tuning per device (calibration sweep replaces static presets)
- OBB (Oriented Bounding Box) detection (next logical task per PROJECT.md)
- Runtime load adaptation (dynamic quality/speed tradeoff under pressure)
- Structured logging + metrics export (Prometheus-compatible)
- Export-and-validate pipeline (export + accuracy validation in one step)

**Defer (P3 / future milestones):**
- Learning from deployment metrics (requires mature adaptation system)
- Zero-config edge deployment (integration feature, needs auto-tune + runtime adapt first)
- Batch processing mode, multi-model ensemble, cross-device benchmark comparison
- Segmentation / Pose estimation (separate milestones per PROJECT.md)

**Anti-features (explicitly reject):**
- Training/fine-tuning, web UI, K8s orchestrator, gRPC server, dynamic model hot-swap, plugin system

### Architecture Approach

Three new subsystems layer on YOWO's existing architecture: Device Profiler (benchmarking + thermal monitoring + profile persistence), Adaptive Optimizer (offline tuning + runtime feedback loop), and Scaled Stream Manager (stream pooling + load balancing + memory guards + adaptive batching). All three communicate through existing infrastructure (EventBus, MetricsCollector, hardware/) using composition and pub/sub, not direct engine mutation. The build order follows data flow: profiler produces data that optimizer consumes, optimizer produces configs that scaler uses.

**Major components:**
1. **profiler/** -- BenchRunner, ThermalMonitor, ProfileStore, device fingerprinting. Runs offline benchmarks, persists results keyed by device+model+backend+precision.
2. **optimizer/** -- Tuner (offline config selection from profiles), Adaptor (runtime feedback loop via EventBus), Strategy (latency-first vs throughput-first).
3. **scaler/** -- StreamPool (bounded concurrency for 100+ streams), LoadBalancer (stream-to-GPU distribution), MemoryGuard (OOM prevention with watermarks), AdaptiveBatcher (dynamic batch sizing wrapping existing BatchScheduler).

**Key patterns:** Two-phase optimization (offline profile + runtime adapt), hierarchical stream management (pool -> groups -> workers), circuit breaker for thermal/memory protection, EventBus-driven adaptation (decoupled from hot path).

### Critical Pitfalls

1. **OpenCV RTSP memory leaks** -- Native C++ buffers leak ~1MB/2-3hrs per stream. At 100+ streams, OOM within hours. Switch to subprocess FFmpeg or PyAV for RTSP; keep OpenCV for files/webcams only.
2. **TensorRT engine version/hardware lock-in** -- Serialized engines are bound to exact TRT version + GPU architecture. Never ship pre-built engines. Ship ONNX, build on-device, cache with version+arch key.
3. **Unbounded GPU memory at 100+ streams** -- Per-stream overhead (decode buffers, feature caches, tensors) grows non-linearly. Implement global memory budget, cap FeatureMapCache, enforce per-device max stream count.
4. **Thermal throttling on edge devices** -- Jetson GPU/CPU clocks drop 30-60% after 10-20 minutes at full load. Benchmark sustained (30+ min) performance, not cold-start. Implement temperature-driven batch size reduction.
5. **EventBus backpressure blocking inference** -- 100 streams x 30 FPS = 3000 events/sec. Slow callbacks block emit() which blocks inference thread. Make emit() non-blocking with drop-on-overflow.
6. **ONNX opset incompatibility** -- Pin opset 13-16 for broad compatibility. Validate exports with target onnxruntime versions, not just latest.
7. **Silent pipeline shutdown corruption** -- At 100 streams, shutdown takes 30+ seconds. Add timeout, track futures, log exceptions instead of swallowing.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Correctness and Export Validation
**Rationale:** Nothing else matters if inference results are wrong or exports fail on target hardware. This is the foundation all subsequent phases build on. Feature research explicitly states "without accuracy parity, nothing else matters."
**Delivers:** Verified mAP parity with ultralytics across all 10 detection + classification variants. Validated ONNX/TRT/OpenVINO exports on each target device. Per-format benchmark mode (mAP + FPS + size table).
**Addresses:** Correctness validation, multi-format export validation, per-format benchmark mode, warm-up validation
**Avoids:** ONNX opset incompatibility (pin opset, validate on target runtimes), silent accuracy degradation (automated mAP comparison), TensorRT version lock-in (on-device engine building)
**Stack:** pytest-benchmark for regression testing, trtexec/benchmark_app for backend profiling

### Phase 2: Hardware Monitoring and Real-Device Testing
**Rationale:** Before scaling or optimizing, need measurement infrastructure on real hardware. Thermal monitoring and device profiling produce the data that auto-tuning consumes. Architecture research shows the profiler/ module has zero coupling to new code -- safe to build independently.
**Delivers:** Device Profiler subsystem (BenchRunner, ThermalMonitor, ProfileStore), real-device CI (self-hosted runners on Jetson/NUC/GPU server), hardware-specific test markers, 24-hour RTSP stability test.
**Addresses:** Thermal-aware throttling (measurement side), health check / readiness endpoint
**Avoids:** Thermal throttling (sustained benchmarks, not cold-start), OpenCV RTSP memory leaks (24-hour stability test to detect), TensorRT engine portability (validate on-device builds)
**Stack:** psutil, pynvml, jetson-stats, fabric, GitHub Actions self-hosted runners, py-spy, memray

### Phase 3: Multi-Stream Scaling and Stress Testing
**Rationale:** Depends on Phase 2 (monitoring infrastructure must exist to measure scaling behavior). Architecture research identifies three sequential bottlenecks: GPU saturation (>16-30 streams), CPU decode (>50 streams), memory (>100 streams). Each needs targeted solutions.
**Delivers:** Scaled Stream Manager subsystem (StreamPool, MemoryGuard, AdaptiveBatcher, LoadBalancer), validated 100+ stream operation, graceful OOM recovery, shutdown-under-load stability.
**Addresses:** Multi-stream stability at 100+ streams, graceful OOM/error recovery
**Avoids:** Unbounded GPU memory growth (global memory budget + stream cap), EventBus backpressure (non-blocking emit), pipeline shutdown corruption (timeout + future tracking)
**Stack:** uvloop for async performance, prometheus-client for scale monitoring

### Phase 4: Adaptive Optimization
**Rationale:** Depends on Phase 2 (ProfileStore data) and Phase 3 (scaled pipeline to optimize). This is where YOWO's "self-optimizing" value proposition materializes. Feature research shows this is the primary differentiator vs ultralytics/DeepStream/Triton (none do runtime adaptation well).
**Delivers:** Adaptive Optimizer subsystem (Tuner for offline config selection, Adaptor for runtime feedback loop), auto-tuning CLI (`yowo autotune`), runtime load adaptation under thermal/memory pressure.
**Addresses:** Adaptive auto-tuning, runtime load adaptation, structured logging + metrics export
**Avoids:** Benchmarking in hot path (offline profile + lightweight runtime metrics), precision downgrade without accuracy validation (Tuner only allows profiled-safe precisions)
**Stack:** optuna for Bayesian search, SQLite for profile persistence

### Phase 5: OBB Detection and Advanced Features
**Rationale:** Independent of phases 1-4 (new detection task with its own head/postprocessing). Can start in parallel with Phase 3-4 if resources allow. Lower urgency than production hardening but high user value.
**Delivers:** OBB detection head, rotation-aware NMS, export-and-validate pipeline, batch processing mode.
**Addresses:** OBB detection, export-and-validate pipeline, batch processing mode

### Phase 6: Deployment Intelligence (Future)
**Rationale:** Requires mature adaptation system (Phase 4) to act on learned data. This is genuinely novel -- no competitor does this.
**Delivers:** Learning from deployment metrics (historical profile refinement), zero-config edge deployment (auto-tune + runtime adapt + hardware detect combined).
**Addresses:** Learning from deployment metrics, zero-config edge deployment, cross-device benchmark comparison

### Phase Ordering Rationale

- **Correctness before scale:** Scaling a system that produces wrong results is worse than useless. Phase 1 validates accuracy.
- **Measurement before optimization:** Architecture research shows profiler/ is a prerequisite for optimizer/. You cannot tune what you cannot measure.
- **Scale before adapt:** The Adaptor needs a scaled pipeline to optimize. Building adaptation for 1-stream is artificial.
- **OBB is parallel:** New detection task has no dependency on production hardening work. Schedule based on team capacity.
- **Learning is last:** Requires weeks of production runtime data to be useful. Build the infrastructure first.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Needs research on COCO val-set evaluation methodology and acceptable mAP delta thresholds per export format
- **Phase 3:** Needs research on StreamPool design -- DeepStream's GStreamer approach vs custom thread-pool; optimal stream group sizing per GPU
- **Phase 4:** Needs research on feedback controller design -- PID vs threshold-based; adaptation interval tuning; stability guarantees
- **Phase 5:** Needs research on OBB head architecture -- ultralytics OBB implementation details, rotation-aware NMS algorithms

Phases with standard patterns (skip research-phase):
- **Phase 2:** Well-documented -- psutil/pynvml APIs, GitHub self-hosted runners, fabric SSH automation are all standard tooling with extensive docs
- **Phase 6:** Standard data persistence + lookup patterns; novel aspect is the integration, not the individual components

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All recommended tools are well-established, verified via official docs and PyPI. No exotic or unproven dependencies. |
| Features | MEDIUM-HIGH | Feature landscape verified against ultralytics, DeepStream, Triton. Dependency graph is sound. Uncertainty: exact scope of "adaptive optimization" needs refinement during planning. |
| Architecture | MEDIUM | Patterns (two-phase optimization, hierarchical streams, circuit breaker) are proven in industry (Triton, DeepStream). Integration with YOWO's specific layered architecture needs design work -- the module boundaries and EventBus extensions are proposed, not validated. |
| Pitfalls | HIGH | Cross-referenced with NVIDIA forums, OpenCV GitHub issues, and YOWO's own CONCERNS.md. All pitfalls have documented occurrences in production systems. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **RTSP decode strategy:** Research identified OpenCV memory leaks but did not fully evaluate PyAV vs subprocess FFmpeg vs GStreamer as replacements. Needs prototyping during Phase 2.
- **Stream group sizing:** Architecture proposes 8-16 streams per group but this is extrapolated from DeepStream benchmarks on different hardware. Needs empirical validation during Phase 3 stress testing.
- **Adaptation stability:** The runtime feedback loop (Adaptor) could oscillate if thresholds are poorly tuned. No research on control theory for inference adaptation was found. Consider PID controller patterns during Phase 4 planning.
- **COCO val-set baseline:** Correctness validation requires ground-truth mAP numbers for each of the 10 variants. These need to be established (run ultralytics baseline) before Phase 1 can define pass/fail criteria.
- **INT8 accuracy delta:** Pitfalls research flags silent accuracy degradation but does not quantify acceptable mAP loss. Industry standard is <1% for general detection, but safety-critical applications may need <0.1%. Define per-deployment.

## Sources

### Primary (HIGH confidence)
- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/) -- benchmarking API and integration
- [psutil docs](https://psutil.readthedocs.io/) -- hardware monitoring API
- [NVIDIA TensorRT Version Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html) -- engine portability constraints
- [OpenCV VideoCapture Memory Leak (GitHub #8151)](https://github.com/opencv/opencv/issues/8151) -- RTSP memory leak confirmation
- [NVIDIA DeepStream Multi-Camera Analytics](https://developer.nvidia.com/blog/multi-camera-large-scale-iva-deepstream-sdk/) -- 100+ stream architecture patterns
- [Triton Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) -- adaptive batching patterns
- [Ultralytics Benchmark Mode](https://docs.ultralytics.com/modes/benchmark/) -- competitor feature baseline
- [ONNX Runtime Compatibility Matrix](https://onnxruntime.ai/docs/reference/compatibility.html) -- opset version constraints

### Secondary (MEDIUM confidence)
- [optuna docs](https://optuna.readthedocs.io/) -- Bayesian optimization for auto-tuning
- [fabric docs](https://www.fabfile.org/) -- remote device test orchestration
- [uvloop benchmarks](https://pypi.org/project/uvloop/) -- async event loop performance
- [Edge AI Inference Optimization Survey](https://www.mdpi.com/2079-9292/14/7/1345) -- general edge optimization patterns
- [GPU Scheduling for Large-Scale Inference](https://medium.com/@fahey_james/gpu-scheduling-for-large-scale-inference-beyond-more-gpus-dcac81f952a2) -- memory management patterns

### Tertiary (LOW confidence)
- Stream group sizing (8-16 per group) -- extrapolated from DeepStream numbers, not validated for YOWO's architecture
- Adaptation feedback loop stability -- no specific research found; PID controller analogy is author inference

---
*Research completed: 2026-03-07*
*Ready for roadmap: yes*
