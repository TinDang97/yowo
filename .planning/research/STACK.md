# Stack Research: Production Hardening CV Inference Platform

**Domain:** Production CV inference across edge devices (Jetson, Intel NUC, CUDA servers)
**Researched:** 2026-03-07
**Confidence:** MEDIUM (verified tool ecosystem via official sources; adaptive optimization patterns from multiple credible sources but no single canonical solution)

## Context

YOWO v2.3.0 already has a mature inference stack (PyTorch, ONNX, TensorRT, OpenVINO, CoreML), hardware auto-detection (pynvml, /proc probes), and 18 preset configs. This research focuses exclusively on **new tooling needed** for production hardening: benchmarking, profiling, thermal/resource monitoring, real-device testing automation, adaptive optimization, and 100+ stream scaling.

---

## Recommended Stack

### Benchmarking and Performance Measurement

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| pytest-benchmark | >=5.2.3 | Inference latency/throughput regression testing | Integrates directly with existing pytest suite. Tracks min/max/avg/stddev across runs, supports warm-up, calibrated rounds, and historical comparison via JSON export. Standard choice for Python perf testing. **Confidence: HIGH** |
| OpenVINO benchmark_app | (bundled with openvino>=2024.0) | Intel NUC backend-level throughput/latency measurement | Already a transitive dependency. Reports FPS, latency percentiles. Use for Intel-specific device profiling without custom code. **Confidence: HIGH** |
| trtexec | (bundled with TensorRT>=10.0) | TensorRT engine-level profiling on Jetson/CUDA | Ships with TensorRT. Measures layer-by-layer timing, memory footprint, and optimal batch size for .engine files. Standard NVIDIA profiling tool. **Confidence: HIGH** |

**Not recommended:** MLPerf Inference -- overkill for a library's internal benchmarking. MLPerf is designed for cross-vendor hardware comparison with strict submission rules. Use pytest-benchmark for regression tracking and trtexec/benchmark_app for backend-specific deep-dives.

### Profiling (CPU, Memory, GPU)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| py-spy | >=0.3.14 | Production-safe CPU profiling of inference hot paths | Zero-overhead sampling profiler (Rust-based, out-of-process). Can attach to running inference without restart or code changes. Generates flamegraphs. Ideal for profiling sustained multi-stream workloads. **Confidence: HIGH** |
| memray | >=1.14 | Memory leak detection under sustained load | Bloomberg's memory profiler. Tracks Python + native C/C++ allocations (critical for numpy/torch). Can attach to running processes. Flamegraph and temporal reporters. Used by Google, Meta, NVIDIA. **Confidence: HIGH** |
| pyinstrument | >=5.0 | Developer-facing call-stack profiling during development | Statistical profiler with minimal overhead (~1ms sampling). Better DX than cProfile for finding slow code paths. Use during development, not production monitoring. **Confidence: HIGH** |

**Not recommended:** cProfile -- deterministic profiler with high overhead. Distorts timing of hot inference paths. Use py-spy for production profiling instead.

### Hardware and Thermal Monitoring

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| psutil | >=6.1 | Cross-platform CPU/memory/thermal monitoring | De-facto standard. `sensors_temperatures()` for thermal, `virtual_memory()` for RAM pressure, `cpu_percent()` for load. Works on Linux (Jetson, NUC, servers) and macOS (dev). Lightweight, no native deps. **Confidence: HIGH** |
| nvidia-ml-py (pynvml) | >=12.560 | GPU utilization, VRAM, temperature, power draw | Official NVIDIA Python bindings for NVML. Already used in `_detect.py` for GPU detection. Extend usage to continuous monitoring: `nvmlDeviceGetTemperature`, `nvmlDeviceGetPowerUsage`, `nvmlDeviceGetUtilizationRates`. **Confidence: HIGH** |
| jetson-stats (jtop) | >=4.5 | Jetson-specific monitoring (GPU, CPU, thermal zones, power rails) | Purpose-built for Jetson devices. Python API (`from jtop import jtop`) exposes all Tegra-specific telemetry not available through standard NVML. Install only on Jetson devices. **Confidence: HIGH** |

**Not recommended:** Custom /sys/class/thermal parsing -- fragile across kernel versions. psutil abstracts this correctly. Also avoid gpustat -- thin wrapper around pynvml with no additional value for programmatic use.

### Real-Device Testing Automation

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| fabric | >=3.2 | SSH-based remote test execution on Jetson/NUC/GPU servers | Pythonic SSH automation built on paramiko. Define tasks that deploy code, run pytest suites, and collect results from remote devices. Simpler and more Pythonic than Ansible for single-device test orchestration. **Confidence: MEDIUM** |
| pytest markers + conftest | (built-in) | Device-specific test selection | Already have `slow` and `integration` markers. Add device-specific markers (`@pytest.mark.jetson`, `@pytest.mark.cuda_server`, `@pytest.mark.intel_nuc`) to run correct test subsets on each device. Zero additional deps. **Confidence: HIGH** |
| GitHub Actions self-hosted runners | (infrastructure) | CI on real hardware | Install self-hosted runner on each target device. Tests tagged with device markers run on correct hardware. Standard approach for hardware-in-the-loop CI. **Confidence: HIGH** |

**Not recommended:** labgrid -- designed for embedded Linux board management (flash firmware, power cycle via relay). Overkill for devices that already run a full OS with SSH. Use fabric for simplicity. Also avoid Ansible -- designed for fleet management, not test execution on 3 devices.

### Adaptive Optimization and Auto-Tuning

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| optuna | >=4.7 | Bayesian search for optimal batch size, precision, backend per device | Define an objective function (e.g., maximize FPS subject to latency < 30ms and VRAM < 80%). Optuna's TPE sampler efficiently explores the search space. Persist results in SQLite for cross-session learning. **Confidence: MEDIUM** |
| sqlite3 | (stdlib) | Persist tuning results and deployment metrics | Zero-dependency. Store per-device optimization results, historical metrics, and learned configurations. Optuna natively supports SQLite storage. Ships with Python. **Confidence: HIGH** |

**Build custom, not adopt a framework for:** Runtime load adaptation (dynamic quality/speed tradeoff). No off-the-shelf library fits the pattern of "adjust batch size and frame skip based on real-time thermal/memory pressure." This should be a custom feedback controller using psutil + pynvml readings to throttle or accelerate the pipeline. The preset config system already has the right structure -- extend it with runtime adjustment logic.

**Not recommended:** NVIDIA Triton Inference Server -- adds a gRPC/HTTP server layer. YOWO is a library, not a microservice. Triton's model analyzer and perf_analyzer are useful concepts but tightly coupled to Triton's serving architecture. Also avoid Ray Serve -- same microservice overhead problem.

### Multi-Stream Scaling (100+ streams)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| uvloop | >=0.21 | High-performance asyncio event loop for stream orchestration | Drop-in replacement for asyncio event loop. 2-4x faster than default loop for I/O-bound coordination of 100+ streams. Already using asyncio (astream). **Confidence: HIGH** |
| prometheus-client | >=0.21 | Metrics export for production monitoring at scale | Expose stream count, FPS per stream, latency percentiles, VRAM usage as Prometheus metrics. Standard for production observability. Lightweight, no server required (pull model). **Confidence: MEDIUM** |

**Not recommended:** multiprocessing for stream scaling -- GIL is not the bottleneck. Backend inference (TensorRT, ONNX) releases the GIL. Frame collection is I/O bound. Threading + asyncio is the correct model, which YOWO already uses.

---

## Supporting Libraries (Dev/Test Only)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-benchmark | >=5.2.3 | Perf regression tests | Every CI run on device hardware |
| pytest-xdist | >=3.5 | Parallel test execution | Speed up large test suite on multi-core devices |
| hypothesis | >=6.100 | Property-based testing for edge cases | Stress-test config combinations, frame sizes, batch sizes |
| pytest-repeat | >=0.9 | Repeat tests for flakiness/stability | Soak testing on real devices |

---

## Installation

```bash
# Benchmarking + profiling (dev dependency group)
uv add --group dev pytest-benchmark py-spy memray pyinstrument

# Hardware monitoring (new optional dependency)
uv add psutil
# psutil is cross-platform, add to core dependencies

# Jetson-only (install on device, not in pyproject.toml)
pip install jetson-stats  # requires Jetson hardware

# Remote device testing
uv add --group dev fabric

# Adaptive optimization
uv add --group dev optuna

# Multi-stream scaling (optional dependency)
uv add --group dev uvloop prometheus-client

# Already available (no install needed)
# - trtexec: ships with TensorRT
# - benchmark_app: ships with OpenVINO
# - pynvml: already used in hardware/_detect.py
# - sqlite3: Python stdlib
```

### Dependency Group Structure

```toml
[project.optional-dependencies]
# ... existing groups ...
monitoring = ["psutil>=6.1"]

[dependency-groups]
dev = [
    # ... existing ...
    "pytest-benchmark>=5.2.3",
    "pytest-xdist>=3.5",
    "fabric>=3.2",
    "optuna>=4.7",
]
profiling = [
    "memray>=1.14",
    "pyinstrument>=5.0",
]
```

Note: `py-spy` is a CLI tool installed via `pip install py-spy` or `cargo install py-spy`, not imported as a library. Keep it outside dependency groups.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| pytest-benchmark | airspeed velocity (asv) | If you need a dedicated benchmarking website with historical tracking. asv is heavier but better for public-facing perf dashboards. |
| fabric | ansible | If managing 10+ devices. For 3 test devices, fabric is simpler. |
| optuna | custom grid search | If the search space is tiny (< 20 configs). Grid search is simpler to debug. Optuna wins when search space is large. |
| py-spy | scalene | If you want CPU + memory + GPU profiling in one tool. Scalene is good for development but has higher overhead than py-spy for production. |
| psutil | custom /proc parsing | Never. psutil handles kernel version differences and cross-platform correctly. |
| prometheus-client | structlog + JSON | If you don't have Prometheus infrastructure. JSON logs with structured metrics are simpler to start with. |
| uvloop | default asyncio | On Windows or if uvloop causes compatibility issues. uvloop is Linux/macOS only. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| NVIDIA Triton Inference Server | Adds gRPC serving layer; YOWO is a library, not a microservice. Triton's model analyzer assumes Triton deployment. | Custom benchmarking with pytest-benchmark + trtexec |
| Ray Serve | Same microservice overhead. Adds cluster management complexity for single-device inference. | Threading + asyncio (already in YOWO) |
| MLPerf loadgen | Rigid submission framework designed for hardware vendor comparisons. Not suitable for library-internal regression testing. | pytest-benchmark for regression, trtexec/benchmark_app for backend profiling |
| TensorBoard for production | Training visualization tool. No support for inference-specific metrics (FPS, latency percentiles, thermal). | prometheus-client or structured logging |
| Docker for device testing | Adds complexity on Jetson (NVIDIA container runtime quirks). Direct SSH execution is simpler and more reliable. | fabric + self-hosted GitHub runners |
| multiprocessing.Pool | GIL is not the bottleneck for inference. Process overhead (memory duplication) is worse than threading for this workload. | ThreadPoolExecutor + asyncio (already in YOWO) |
| Weights & Biases / MLflow | Designed for training experiment tracking. YOWO is inference-only. Adds cloud dependency. | SQLite for local metrics + optuna storage |

---

## Stack Patterns by Device

**Jetson Orin/Xavier:**
- Monitor with: jetson-stats (jtop) + pynvml
- Profile with: trtexec (TensorRT engines), py-spy (Python overhead)
- Benchmark with: pytest-benchmark (regression), trtexec (raw engine perf)
- Thermal management: Read via jtop API, throttle batch size when > 80C
- Constraint: Shared memory (no discrete VRAM), watch unified memory pressure

**Intel NUC (OpenVINO):**
- Monitor with: psutil (CPU/thermal), no GPU-specific monitoring needed for iGPU
- Profile with: OpenVINO benchmark_app, py-spy
- Benchmark with: pytest-benchmark, benchmark_app
- Constraint: Limited compute, focus on INT8 precision and throughput hints

**CUDA GPU Server:**
- Monitor with: pynvml (VRAM, utilization, power), psutil (system)
- Profile with: py-spy + memray (Python side), trtexec (TensorRT), nsight (deep GPU)
- Benchmark with: pytest-benchmark (regression), trtexec (raw engine)
- Scaling: Primary target for 100+ stream testing with uvloop

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| psutil >=6.1 | Python 3.11+ | v6.1 dropped Python 2.7 support. Use >=6.1 for modern API. |
| pytest-benchmark >=5.2 | pytest >=8.0 | v5.2+ dropped pytest 7 support. Aligns with existing pytest>=8.0. |
| optuna >=4.7 | SQLite (stdlib) | Native SQLite storage. No additional DB required. |
| fabric >=3.2 | paramiko >=3.0 | Fabric 3.x is a full rewrite. Do NOT use fabric2 or fabric1 APIs. |
| memray >=1.14 | Linux only | memray does NOT support macOS well. Use on target devices only. |
| uvloop >=0.21 | Python 3.12 | Linux/macOS only. Does not work on Windows. |
| jetson-stats >=4.5 | JetPack 5.x/6.x | Only installs on actual Jetson hardware. |
| nvidia-ml-py >=12.560 | CUDA 12.x drivers | Breaking changes between major versions. Pin to match driver. |

---

## Sources

- [pytest-benchmark 5.2.3 docs](https://pytest-benchmark.readthedocs.io/) -- verified latest version via PyPI (Nov 2025)
- [psutil docs](https://psutil.readthedocs.io/) -- verified sensors_temperatures() API
- [nvidia-ml-py on PyPI](https://pypi.org/project/nvidia-ml-py/) -- confirmed official NVIDIA package, 2025 releases
- [jetson-stats GitHub](https://github.com/rbonghi/jetson_stats) -- verified Python API, Orin/Xavier support
- [memray docs](https://bloomberg.github.io/memray/) -- verified native allocation tracking, attach-to-process feature
- [py-spy GitHub](https://github.com/benfred/py-spy) -- verified out-of-process, zero-overhead profiling
- [fabric docs](https://www.fabfile.org/) -- verified v3.2 SSH automation API
- [optuna 4.7.0 docs](https://optuna.readthedocs.io/) -- verified TPE sampler, SQLite storage
- [OpenVINO 2025.4 benchmark_app](https://docs.openvino.ai/2025/about-openvino/performance-benchmarks/getting-performance-numbers.html) -- verified current version
- [NVIDIA trtexec docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html) -- TensorRT profiling tool
- [labgrid on PyPI](https://pypi.org/project/labgrid/) -- evaluated and rejected (overkill for full-OS devices)
- [uvloop on PyPI](https://pypi.org/project/uvloop/) -- verified Linux/macOS, asyncio drop-in

---
*Stack research for: Production hardening CV inference platform*
*Researched: 2026-03-07*
