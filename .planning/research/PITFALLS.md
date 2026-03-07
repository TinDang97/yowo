# Pitfalls Research

**Domain:** Production CV inference platform -- edge deployment, multi-stream scaling (100+ streams)
**Researched:** 2026-03-07
**Confidence:** HIGH (cross-referenced codebase analysis, NVIDIA forums, OpenCV issues, framework docs)

## Critical Pitfalls

### Pitfall 1: OpenCV VideoCapture Memory Leaks on Long-Running RTSP Streams

**What goes wrong:**
`cv2.VideoCapture` leaks memory when reading RTSP streams over hours/days. The `read()` path leaks ~1 MB every 2-3 hours per stream. Calling `release()` does not fully free RAM. Reconnecting cameras (creating new VideoCapture instances) compounds the leak -- 120 MB in 16 minutes with grab+retrieve on Ethernet cameras. At 100+ streams running 24/7, this leads to OOM within hours.

**Why it happens:**
OpenCV's FFmpeg/GStreamer backend allocates internal buffers for RTSP decoding that are not fully reclaimed on `release()`. The leak is in native C++ code, not Python. Reconnection logic that creates new VideoCapture instances without proper cleanup exacerbates the issue. YOWO's `RTSPStreamSource` (in `src/yowo/io/_source.py`) passes URLs directly to `cv2.VideoCapture()` and the reconnect path uses `time.sleep()` blocking, both identified as concerns in CONCERNS.md.

**How to avoid:**
- Use subprocess-based FFmpeg decoding (PyAV or raw ffmpeg pipes) instead of OpenCV for RTSP streams in production. OpenCV's VideoCapture is fine for files and webcams but leaks on network streams.
- If staying with OpenCV: implement a watchdog that monitors RSS memory per-stream and force-restarts the capture process (not just the object) when memory exceeds a threshold.
- Use GStreamer pipeline strings with explicit buffer limits: `rtspsrc location=... latency=100 ! queue max-size-buffers=3 leaky=downstream`.
- YOWO's `ThreadedFrameReader` bounded deque partially mitigates frame accumulation but does not address the native memory leak inside VideoCapture.

**Warning signs:**
- RSS memory growth in `tegrastats` or `/proc/self/status` VmRSS over hours
- System OOM-killer targeting the inference process
- Gradually increasing swap usage on Jetson devices

**Phase to address:**
Phase 1 (Correctness Validation) -- benchmark memory stability over 24-hour RTSP sessions before scaling to multi-stream.

---

### Pitfall 2: TensorRT Engine Version/Hardware Lock-In

**What goes wrong:**
TensorRT serialized engines (`.engine` / `.plan` files) are bound to the exact TensorRT version, CUDA version, and GPU architecture (compute capability) used to build them. An engine built on a development workstation with TensorRT 8.6 + RTX 4090 will fail to deserialize on a Jetson Orin with TensorRT 8.5 + SM87. The error is a hard crash at load time: "engine plan file is not compatible with this version of TensorRT."

**Why it happens:**
TensorRT aggressively optimizes for the specific GPU -- kernel selection, memory layout, and tactic choices are hardware-specific. This is by design for performance. Developers build engines on dev machines and ship the binary to edge devices, expecting portability that does not exist. YOWO's export pipeline likely generates engines on the build machine.

**How to avoid:**
- Never ship pre-built TensorRT engines. Ship ONNX models and build engines on-device at first-run. Cache the built engine keyed by `{model_hash}_{trt_version}_{cuda_version}_{gpu_arch}`.
- YOWO already has ONNX export. The TensorRT backend must implement on-device engine building with a cache directory.
- For Jetson devices: engines MUST be built on the Jetson itself (or cross-compiled with matching JetPack version).
- Use TensorRT's version compatibility mode (available since TRT 8.6) for forward-compatibility, but do not rely on it for cross-architecture.
- INT8 calibration tables are also hardware-specific -- recalibrate on target device.

**Warning signs:**
- "engine plan file is not compatible" errors in deployment logs
- Export tests pass on CI but fail on edge devices
- Different inference results between dev and production (engine rebuilt with different tactics)

**Phase to address:**
Phase 2 (Real-Device Testing) -- implement on-device engine building and validate on all three target platforms (Jetson Orin/Xavier, Intel NUC, CUDA server).

---

### Pitfall 3: Unbounded GPU Memory Growth in Multi-Stream Batching

**What goes wrong:**
When scaling from 1-5 streams to 100+ streams, GPU memory consumption grows non-linearly. Each stream requires decode buffers, preprocessing tensors, and feature map caches. YOWO's `FeatureMapCache` is per-source and memory-unbounded (noted in CONCERNS.md). At 100 streams on a Jetson Orin (8-16 GB unified memory), the system hits OOM before reaching target stream count.

**Why it happens:**
Developers optimize per-stream throughput first, then try to scale stream count. Per-stream memory overhead that is invisible at 5 streams (50 MB each = 250 MB) becomes catastrophic at 100 streams (50 MB each = 5 GB). The batch scheduler allocates tensors per batch without a global memory budget. Feature map caches grow linearly with source count. Frame decode buffers in ThreadedFrameReader hold numpy arrays per stream.

**How to avoid:**
- Implement a global memory budget that caps total allocation across all streams. Use a memory pool (pre-allocated tensor buffer) shared across streams.
- Cap `FeatureMapCache` entries with a maximum count or total byte budget. Evict LRU entries, not just on fingerprint mismatch.
- Use frame dropping aggressively: not every stream needs every frame processed. Adaptive frame skip based on available GPU memory.
- Pre-calculate maximum stream count per device: `(available_gpu_memory - model_memory - overhead) / per_stream_memory`.
- On Jetson (unified memory): account for CPU+GPU sharing the same pool. Leave 20% headroom for OS and other processes.

**Warning signs:**
- CUDA OOM errors when adding streams beyond a threshold
- `nvidia-smi` or `tegrastats` showing memory approaching device limit
- Inference latency increasing as stream count grows (memory pressure causing swapping)

**Phase to address:**
Phase 3 (Stress Testing) -- test with progressively increasing stream counts, measure per-stream memory overhead, establish per-device maximum stream counts.

---

### Pitfall 4: Thermal Throttling Destroys Sustained Throughput on Edge Devices

**What goes wrong:**
Edge devices (Jetson Nano/Xavier/Orin) throttle GPU and CPU clocks when temperature exceeds thresholds (~70C on Nano, ~85C on Orin). Under multi-stream inference load, sustained throughput drops 30-60% from peak within 10-20 minutes. Benchmarks taken in the first 5 minutes of operation are misleading -- they reflect burst performance, not sustained performance. "System throttled due to overcurrent" messages appear on Jetson when running at MAXN power mode.

**Why it happens:**
Edge devices have limited thermal dissipation. Developers benchmark cold-start performance and ship those numbers. Production runs 24/7 at ambient temperatures of 25-40C (server rooms, outdoor enclosures). The thermal envelope is reached quickly under full GPU utilization.

**How to avoid:**
- Always benchmark sustained performance: run inference for 30+ minutes before measuring. Report p95 latency, not average.
- Use Jetson's power mode presets (15W, 30W, MAXN) deliberately. MAXN triggers throttling; 15W mode sustains without throttling but at lower throughput.
- Implement adaptive quality/speed tradeoff: when temperature exceeds threshold, reduce batch size, skip frames, or switch to a lighter model variant (e.g., YOLO11n instead of YOLO11m).
- Monitor temperature via `tegrastats` parsing or `/sys/devices/virtual/thermal/` and expose it through YOWO's metrics/events system.
- Physical: ensure adequate heatsinking. The Jetson developer kit heatsink is insufficient for sustained load; production enclosures need active cooling or larger heatsinks.

**Warning signs:**
- FPS dropping after 10+ minutes of continuous inference
- `tegrastats` showing GPU temp >80C
- "System throttled" messages in `dmesg`
- Batch latency variance increasing over time (thermal oscillation)

**Phase to address:**
Phase 2 (Real-Device Testing) for measurement, Phase 4 (Adaptive Optimization) for runtime adaptation.

---

### Pitfall 5: ONNX Opset and Operator Incompatibility Across Runtimes

**What goes wrong:**
An ONNX model exported with opset 17 on PyTorch 2.x fails to load on an edge device running ONNX Runtime 1.14 (which only guarantees opset 16). Or worse: the model loads but a specific operator (e.g., `GridSample`, `ScatterND`) has different behavior between opset versions, producing silently wrong results. Dynamic shapes exported from PyTorch can also cause shape inference failures at runtime.

**Why it happens:**
PyTorch's ONNX exporter defaults to the latest opset. Developers export on a dev machine with the latest onnxruntime and PyTorch, but edge devices run older versions pinned by JetPack or system packages. YOLO architectures use operators (Resize with coordinate_transformation_mode, Concat with dynamic axes) that have changed semantics between opsets.

**How to avoid:**
- Pin ONNX opset version explicitly during export (opset 13-16 is the safe range for broad compatibility as of 2026).
- Validate the exported ONNX model with `onnx.checker.check_model()` AND with the target runtime version before shipping.
- Use static input shapes for edge deployment. Dynamic shapes add overhead and compatibility risk.
- Test export on a matrix of onnxruntime versions matching target devices.
- YOWO's export pipeline should record the opset version in model metadata and validate at load time.

**Warning signs:**
- "Unsupported opset version" errors during model loading
- Different inference results between PyTorch and ONNX backends (NMS score differences, bounding box offsets)
- Export warnings about unsupported operators being replaced with fallbacks

**Phase to address:**
Phase 1 (Correctness Validation) -- validate export artifacts on all target runtimes as part of correctness testing.

---

### Pitfall 6: Silent Data Corruption in Concurrent Pipeline Shutdown

**What goes wrong:**
When shutting down a multi-stream pipeline, in-flight frames in ThreadPoolExecutor futures are cancelled. Cancelled futures may have partially processed frames, and the exceptions are lost (noted in CONCERNS.md). In production, this manifests as: graceful shutdown takes too long, operator force-kills the process, GPU memory is not released, and the next restart finds stale lock files or corrupted cache entries.

**Why it happens:**
Python's ThreadPoolExecutor cancellation is cooperative -- `future.cancel()` only prevents execution of not-yet-started tasks. Running tasks cannot be interrupted. Developers test shutdown with 1-2 streams where everything completes quickly. At 100 streams with queued batches, shutdown can take 30+ seconds. The `_stream_pipeline` path in YOWO's StreamingMixin has this exact issue (line 198-199 per CONCERNS.md).

**How to avoid:**
- Implement a shutdown timeout: wait up to N seconds for in-flight work, then force-close backends.
- Track all in-flight futures and log (not swallow) exceptions from cancelled ones.
- Backend `unload()` must be robust to being called while inference is in progress -- use a lock or reference count.
- Add logging to `close()` exception handler (currently silent per CONCERNS.md).
- Test shutdown under load: start 50+ streams, then call `close()` while inference is active.

**Warning signs:**
- Shutdown taking >10 seconds
- GPU memory not released after engine.close() (visible in `nvidia-smi`)
- "CUDA error: device-side assert" on next engine creation
- Log silence during shutdown (no errors logged = errors swallowed)

**Phase to address:**
Phase 3 (Stress Testing) -- include shutdown-under-load as a stress test scenario.

---

### Pitfall 7: EventBus Backpressure Blocking the Inference Hot Path

**What goes wrong:**
YOWO's EventBus uses a bounded semaphore (256 slots) for the dispatch queue. At 100 streams x 30 FPS, each emitting detection events, the bus receives 3000 events/second. If any event callback is slow (e.g., writing to disk, network call, database insert), the queue fills and `emit()` blocks the inference thread. This creates a cascading failure: blocked inference causes frame drops, which causes the frame reader to fall behind, which causes RTSP buffer overflow.

**Why it happens:**
Event-driven architectures work well at low throughput. At high throughput, the producer (inference) and consumer (callbacks) must be decoupled with proper backpressure. The current design (blocking `emit()`) couples them. The silent exception swallowing in dispatch (CONCERNS.md) makes debugging impossible.

**How to avoid:**
- Make EventBus queue size configurable and scale with stream count.
- Switch `emit()` to non-blocking with drop-on-overflow semantics. Log dropped events at WARNING level.
- Move event emission off the inference hot path: batch events and emit asynchronously.
- Add event callback performance monitoring: log callbacks that take >1ms.
- Consider per-event-type queues to prevent slow detection events from blocking health events.

**Warning signs:**
- Inference FPS dropping when event subscribers are attached
- `emit()` showing up in profiler hot path
- Frame drop rate increasing with number of event subscribers

**Phase to address:**
Phase 3 (Stress Testing) for detection, Phase 4 (Adaptive Optimization) for implementation.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| 59 `type: ignore` in StreamingMixin | Mixin works without Protocol boilerplate | Any BaseEngine attribute rename silently breaks streaming at runtime | Never in production-hardened code; fix before scaling work |
| `torch.load(weights_only=False)` | Loads ultralytics checkpoint format | Remote code execution via crafted .pt files | Only with hash-verified official weights; add checksum validation |
| Swallowing exceptions in `close()` and EventBus | Process does not crash on cleanup | Resource leaks invisible, debugging impossible | Acceptable to catch, never acceptable to silently discard; add logging |
| Per-source unbounded FeatureMapCache | Simple implementation, good single-stream perf | OOM at high stream count | Only acceptable with <10 streams; add memory budget before scaling |
| Global pyright strict-mode suppressions | Silences noise from untyped C libs | False sense of type safety across entire codebase | Move to per-file suppression in backend files only |

## Integration Gotchas

Common mistakes when connecting to external services and hardware.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| TensorRT on Jetson | Shipping pre-built .engine files from dev machine | Ship ONNX, build engine on-device, cache with version+arch key |
| OpenVINO on Intel NUC | Using FP32 model on integrated GPU | Export with FP16; iGPU has limited memory and FP16 is 2x faster |
| RTSP cameras | Single VideoCapture per stream, no health monitoring | Subprocess-based decode with watchdog; restart on memory growth |
| INT8 calibration | Calibrating on dev GPU, deploying on Jetson | Calibration must run on target hardware with representative data |
| ONNX Runtime | Using latest opset, deploying on old runtime | Pin opset 13-16; validate with target onnxruntime version |
| GStreamer on Jetson | Using OpenCV's default FFmpeg backend | Use `cv2.CAP_GSTREAMER` with nvv4l2decoder for hardware decode |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-stream preprocessing on CPU | Low CPU util at 5 streams | Batch preprocessing on GPU; use CUDA resize/normalize | >20 streams on Jetson (CPU becomes bottleneck) |
| Synchronous NMS per frame | Acceptable latency at 1 stream | Batch NMS or async NMS on separate thread | >30 FPS * 10 streams = 300 NMS/sec |
| Unbounded frame buffers | No drops at low load | Bounded deque with LATEST drop policy (YOWO has this) | >50 streams; old frames accumulate faster than processed |
| Python GIL in inference loop | Invisible at low concurrency | Release GIL during backend.infer() (numpy/torch do this); use multiprocessing for CPU-bound work | >8 CPU threads competing for GIL |
| Creating new tensors per frame | Clean code, easy to read | Pre-allocate tensor buffers, reuse across frames | >60 FPS; GC pressure causes latency spikes |
| Logging in hot path | Useful for debugging | Use DEBUG level; disable in production; never format strings eagerly | >1000 inferences/sec; string formatting overhead adds up |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| `torch.load(weights_only=False)` with user-supplied .pt files | Remote code execution via pickle deserialization | Use `weights_only=True` + `add_safe_globals()` for known types; validate file hash for official weights |
| RTSP URL passed directly to OpenCV without validation | Potential SSRF or malformed URL exploitation via ffmpeg backend | Validate URL scheme (rtsp/rtsps only) and format before passing to VideoCapture |
| No rate limiting on weight downloads | Multiple concurrent instances can hammer GitHub CDN, get IP-banned | Download weights to shared cache dir with file-lock; pre-download in deployment scripts |
| TensorRT engine cache without integrity check | Tampered engine file could load malicious CUDA kernels | Hash-verify cached engine files before deserialization |

## UX Pitfalls

Common user experience mistakes in this domain (for library consumers / developers).

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Default config assumes CUDA availability | Crash on CPU-only or MPS machines | Auto-detect hardware and select best available backend (YOWO does this, but validate edge cases) |
| No feedback during model download/build | User thinks app is hung during 2+ minute TensorRT build | Progress callbacks via EventBus; CLI progress bar; log estimated time |
| Cryptic backend errors | "CUDA error: invalid device ordinal" with no context | Wrap backend errors with actionable messages: "GPU 1 not found. Available GPUs: [0]. Set YOWO_DEVICE=cuda:0" |
| Preset config that does not match actual hardware | Suboptimal performance with wrong preset | Validate preset against detected hardware; warn if mismatch |
| Silent model accuracy degradation after export | User trusts exported model without validation | Run validation on export: compare N sample outputs between PyTorch and exported model; warn if delta exceeds threshold |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Multi-stream pipeline:** Often missing global memory budget -- verify per-device max stream count is enforced
- [ ] **TensorRT export:** Often missing on-device build step -- verify engine is built on target hardware, not shipped pre-built
- [ ] **RTSP streaming:** Often missing long-running stability -- verify 24-hour memory stability test passes
- [ ] **INT8 quantization:** Often missing accuracy validation -- verify mAP on COCO matches FP32 within acceptable delta (<1%)
- [ ] **Graceful shutdown:** Often missing shutdown-under-load -- verify close() with 50+ active streams completes within timeout
- [ ] **Backend auto-selection:** Often missing fallback chain -- verify behavior when preferred backend is unavailable (e.g., no TensorRT on Intel NUC)
- [ ] **Adaptive optimization:** Often missing sustained thermal testing -- verify auto-tuning decisions hold after 30 minutes of thermal equilibrium
- [ ] **Health monitoring:** Often missing resource leak detection -- verify GPU memory, thread count, and file descriptor count are stable over hours
- [ ] **Event system:** Often missing backpressure handling -- verify inference FPS is unaffected by slow event callbacks
- [ ] **Model correctness:** Often missing edge-case inputs -- verify with blank frames, 1x1 images, 8K resolution, and corrupted frames

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| OpenCV RTSP memory leak | MEDIUM | Switch to subprocess FFmpeg decode; requires refactoring FrameSource implementations |
| TensorRT version mismatch | LOW | Ship ONNX + on-device build script; rebuild cache on version change |
| GPU OOM at high stream count | MEDIUM | Add memory budget + stream count cap; requires global resource manager |
| Thermal throttling | LOW | Add temperature monitoring + adaptive frame skip; expose via existing metrics |
| ONNX opset incompatibility | LOW | Pin opset at export time; add runtime version check at load |
| Pipeline shutdown corruption | MEDIUM | Add future tracking + timeout + logging; refactor StreamingMixin shutdown path |
| EventBus blocking inference | MEDIUM | Make emit() non-blocking with drop semantics; requires EventBus API change |
| Silent accuracy degradation | HIGH | Add automated accuracy validation in export pipeline; requires test dataset and baseline |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| OpenCV RTSP memory leak | Phase 1: Correctness Validation | 24-hour RTSP stream test with memory monitoring; RSS growth <100 MB |
| TensorRT version lock-in | Phase 2: Real-Device Testing | Engine built on each target device; inference results match ONNX baseline |
| GPU memory growth | Phase 3: Stress Testing | 100-stream test with memory stable after warmup period |
| Thermal throttling | Phase 2: Real-Device Testing + Phase 4: Adaptive Optimization | 30-minute sustained benchmark; FPS variance <10% after thermal equilibrium |
| ONNX opset incompatibility | Phase 1: Correctness Validation | Export validated on onnxruntime versions matching all target devices |
| Pipeline shutdown corruption | Phase 3: Stress Testing | Shutdown-under-load test: 50 streams, close() completes in <10s, no GPU memory leak |
| EventBus backpressure | Phase 3: Stress Testing | Inference FPS unaffected with slow callback (100ms sleep) attached |
| Silent accuracy degradation | Phase 1: Correctness Validation | Automated mAP comparison: PyTorch vs each exported format on COCO val subset |
| StreamingMixin type safety | Phase 1: Correctness Validation | Replace type:ignore with Protocol; pyright strict passes without global suppressions |
| torch.load RCE risk | Phase 2: Real-Device Testing | weights_only=True with safe_globals; hash verification for official weights |

## Sources

- [Ultralytics Model Deployment Best Practices](https://docs.ultralytics.com/guides/model-deployment-practices/)
- [NVIDIA DeepStream Multi-Stream Management](https://developer.nvidia.com/blog/managing-video-streams-in-runtime-with-the-deepstream-sdk/)
- [NVIDIA Power Optimization with Jetson](https://developer.nvidia.com/blog/power-optimization-with-nvidia-jetson/)
- [TensorRT Version Compatibility Issue (GitHub)](https://github.com/NVIDIA/TensorRT/issues/2624)
- [TensorRT Advanced Topics -- Version Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html)
- [OpenCV VideoCapture Memory Leak (GitHub #8151)](https://github.com/opencv/opencv/issues/8151)
- [OpenCV VideoCapture.read() Stuck After Reconnect (GitHub #22677)](https://github.com/opencv/opencv/issues/22677)
- [ONNX Runtime Compatibility Matrix](https://onnxruntime.ai/docs/reference/compatibility.html)
- [Detectron2 Memory Leak During Video Inference (GitHub #5408)](https://github.com/facebookresearch/detectron2/issues/5408)
- [Ultralytics GPU Memory Growth During Inference (GitHub #281)](https://github.com/ultralytics/ultralytics/issues/281)
- [Jetson Orin Overcurrent During TensorRT (NVIDIA Forums)](https://forums.developer.nvidia.com/t/system-throttled-due-to-overcurrent-while-running-tensorrt/286431)
- [Jetson Xavier Memory Leak with GStreamer RTSP (NVIDIA Forums)](https://forums.developer.nvidia.com/t/memory-leak-when-gstreamer-pull-rtsp-stream/243519)
- [PacketGame: Multi-Stream Packet Gating at Scale (SIGCOMM 2023)](https://yuanmu97.github.io/preprint/packetgame_sigcomm23.pdf)
- YOWO Codebase Analysis: `.planning/codebase/CONCERNS.md` (2026-03-07)

---
*Pitfalls research for: Production CV inference platform -- edge deployment, 100+ streams*
*Researched: 2026-03-07*
