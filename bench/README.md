# yowo Benchmark Suite

Self-contained benchmarks for measuring yowo inference performance across backends, hardware, and model variants.

## Quick Start

```bash
# 1. Install yowo with benchmark extras
uv sync --group dev

# 2. Download model weights (all 10 variants, ~120 MB each)
uv run python bench/setup.py

# 3. Run architecture latency benchmark
uv run python bench/bench_arch.py --all --device cpu

# 4. Run PyTorch vs ONNX benchmark
uv run python bench/bench_onnx.py --all

# 5. Run yowo vs ultralytics comparison
uv run python bench/bench_compare.py --all
```

---

## Prerequisites

| Requirement     | Version        | Notes                                        |
|-----------------|----------------|----------------------------------------------|
| Python          | ≥ 3.11         | 3.13 free-threaded supported                 |
| uv              | any            | Package manager — `pip install uv`           |
| torch           | ≥ 2.0          | Installed via `uv sync --group dev`          |
| onnxruntime     | ≥ 1.17         | Installed via `uv sync --group dev`          |
| onnxslim        | ≥ 0.1.85       | ONNX graph simplification                   |
| psutil          | any            | Optional — CPU/memory profiling in compare   |
| ultralytics     | any            | Optional — required only for bench_compare   |

### Apple Silicon (CoreML)

Install `onnxruntime` with CoreML support:
```bash
pip install onnxruntime-silicon  # or standard onnxruntime on macOS arm64
```
The ONNX backend auto-detects CoreML availability at runtime.

### NVIDIA GPU (CUDA / TensorRT)

```bash
# CUDA via ORT
pip install onnxruntime-gpu

# TensorRT (requires NVIDIA driver + CUDA toolkit)
pip install tensorrt>=10.0 --extra-index-url https://pypi.nvidia.com
```

---

## Setup

```bash
uv run python bench/setup.py [OPTIONS]
```

| Flag                  | Default   | Description                                  |
|-----------------------|-----------|----------------------------------------------|
| `--models MODELS`     | all 10    | Comma-separated list, e.g. `yolo26n,yolo11n` |
| `--weights-dir DIR`   | bench/weights/ | Target directory for .pt files          |
| `--no-weights`        | off       | Skip weight download (check packages only)   |
| `--install-extras`    | off       | Install onnxruntime, psutil, onnxslim via pip|
| `--force`             | off       | Re-download even if file already exists      |

### Examples

```bash
# Download only nano variants
uv run python bench/setup.py --models yolo26n,yolo11n

# Check packages only (no download)
uv run python bench/setup.py --no-weights

# Download + install optional packages
uv run python bench/setup.py --install-extras
```

---

## bench_arch.py — Architecture Latency

Measures raw PyTorch inference latency per model variant. No I/O overhead.

```bash
uv run python bench/bench_arch.py --all --device cpu
uv run python bench/bench_arch.py --all --device cuda --fp16
uv run python bench/bench_arch.py --model yolo26n --weights bench/weights/yolo26n.pt
```

| Flag              | Default | Description                                      |
|-------------------|---------|--------------------------------------------------|
| `--all`           | —       | All variants found in `bench/weights/`           |
| `--model NAME`    | —       | Single variant (mutually exclusive with `--all`) |
| `--weights PATH`  | —       | Required with `--model`                          |
| `--device DEVICE` | cpu     | torch device (cpu, cuda, mps)                    |
| `--compile`       | off     | torch.compile with reduce-overhead               |
| `--fp16`          | off     | FP16 autocast (CUDA only)                        |
| `--warmup N`      | 10      | Warmup iterations                                |
| `--iters N`       | 100     | Benchmark iterations                             |

### Expected Output (Apple M4 Pro, CPU, FP32)

```
model        device    compile   fp16    mean(ms)    p50(ms)     p95(ms)     p99(ms)     fps
-----------------------------------------------------------------------
yolo11n      cpu       no        no      27.34       27.10       28.95       30.12       36.6
yolo26n      cpu       no        no      23.45       23.20       24.80       26.10       42.6
...
```

> **Note**: `torch.compile` regresses ~30–43% on CPU. Only enable with `--device cuda`.

---

## bench_onnx.py — PyTorch vs ONNX Runtime

Exports each variant to ONNX (cached), then benchmarks PyTorch vs ONNX-CPU vs CoreML (optional) vs KV-cache variants.

```bash
uv run python bench/bench_onnx.py --all
uv run python bench/bench_onnx.py --all --coreml
uv run python bench/bench_onnx.py --all --kv-cache
uv run python bench/bench_onnx.py --all --coreml --kv-cache
uv run python bench/bench_onnx.py --model yolo26n --no-export  # use cached ONNX
```

| Flag          | Default | Description                                     |
|---------------|---------|-------------------------------------------------|
| `--all`       | —       | All variants in `bench/weights/`                |
| `--model`     | —       | Single variant                                  |
| `--no-export` | off     | Skip re-export; use cached ONNX from exports/   |
| `--coreml`    | off     | Include CoreML EP (macOS arm64 only)            |
| `--kv-cache`  | off     | Include KV-cache ONNX variants                  |
| `--runs N`    | 50      | Inference iterations                            |
| `--warmup N`  | 5       | Warmup iterations                               |

### Expected Output (Apple M4 Pro, CoreML EP)

```
model      backend        mean(ms)    p50(ms)    p95(ms)    p99(ms)      fps
------------------------------------------------------------------------
yolo26n    pytorch           23.45      23.20      24.80      26.10      42.6
yolo26n    onnx-cpu          18.10      17.90      19.20      20.50      55.2   (+1.30x vs pytorch)
yolo26n    coreml             5.32       5.28       5.85       6.20     187.9   (+4.41x vs pytorch)
```

> CoreML EP offloads 98%+ of compute to the Neural Engine on Apple Silicon, yielding 4–5× over PyTorch.
> KV-CPU is 2–8× slower than standard ONNX-CPU (extra Where ops); KV-CoreML matches standard CoreML.

---

## bench_compare.py — yowo vs ultralytics

Side-by-side latency and resource comparison using identical weights.

```bash
# Requires: uv add ultralytics --group dev
uv run python bench/bench_compare.py --all
uv run python bench/bench_compare.py --all --profile     # includes RSS + CPU%
uv run python bench/bench_compare.py --all --compile     # torch.compile on yowo
uv run python bench/bench_compare.py --model yolo26n --weights bench/weights/yolo26n.pt
```

| Flag          | Default | Description                                          |
|---------------|---------|------------------------------------------------------|
| `--all`       | —       | All variants in `bench/weights/`                     |
| `--model`     | —       | Single variant                                       |
| `--weights`   | —       | Required with `--model`                              |
| `--device`    | cpu     | torch device                                         |
| `--compile`   | off     | torch.compile on yowo side                          |
| `--profile`   | off     | Also report RSS (MB) and CPU% (requires psutil)      |
| `--warmup N`  | 10      | Warmup iterations                                    |
| `--iters N`   | 100     | Benchmark iterations                                 |

### Expected Output (Apple M4 Pro, CPU, FP32)

```
--- LATENCY ---
model       impl           mean(ms)   p50(ms)    p95(ms)   fps      speedup
---------------------------------------------------------------------------
yolo26n     ultralytics    28.50      28.20      30.10     35.1
___________ yowo           23.45      23.20      24.80     42.6     1.21x (faster)
```

### Benchmark Summary (all 10 variants, CPU, FP32, 100 iterations)

| Metric                | Value       |
|-----------------------|-------------|
| Average speedup       | 1.07×       |
| Variants faster       | 9/10        |
| Best speedup (yowo26) | 1.10–1.17×  |
| Box IoU vs ultralytics| 0.967–0.995 |

---

## Weights Directory Structure

```
bench/
└── weights/
    ├── yolo11n.pt   (6.5 MB)
    ├── yolo11s.pt   (18.7 MB)
    ├── yolo11m.pt   (38.8 MB)
    ├── yolo11l.pt   (49.0 MB)
    ├── yolo11x.pt   (109 MB)
    ├── yolo26n.pt   (4.9 MB)
    ├── yolo26s.pt   (9.5 MB)
    ├── yolo26m.pt   (38.8 MB)
    ├── yolo26l.pt   (49.7 MB)
    └── yolo26x.pt   (109 MB)
```

ONNX exports are cached in `bench/exports/` on first run.

---

## Reproducibility

To produce results comparable to `docs/experiments/`:

```bash
# Arch benchmark — same as Phase 2 results
uv run python bench/bench_arch.py --all --device cpu --warmup 10 --iters 100

# ONNX/CoreML — same as 2026-02-24 CoreML benchmark
uv run python bench/bench_onnx.py --all --coreml --warmup 5 --runs 50

# yowo vs ultralytics — same as arch-inference-optimization
uv run python bench/bench_compare.py --all --device cpu --iters 100
```
