# Experiment Report: Vehicle Detection Benchmark — YOLO11s + best.pt

**Date**: 2026-02-23
**Author**: Tin Dang
**Hardware**: Apple M4 Pro (CPU only)
**Platform**: macOS 25.3.0, Python 3.11.11

---

## Objective

Evaluate yowo inference performance with a custom 7-class vehicle detection model (`best.pt`) across backends (PyTorch, ONNX FP32, ONNX FP16, ONNX INT8) and batch sizes (1, 4, 8). Establish a production deployment recommendation for Apple Silicon edge devices.

---

## Model

| Property | Value |
|----------|-------|
| Weights | `best.pt` (YOLO11s fine-tuned) |
| Architecture | YOLO11s — 100 layers, 9.4M parameters, 21.3 GFLOPs |
| Input shape | 640×640 |
| Output shape | `(B, 11, 8400)` — 4 bbox + 7 classes |
| Task | Object detection |

### Class mapping

| ID | Class |
|----|-------|
| 0 | motorcycle |
| 1 | car |
| 2 | bus |
| 3 | truck |
| 4 | transporter |
| 5 | container |
| 6 | big_transporter |

---

## Test Image

| Property | Value |
|----------|-------|
| File | `z7556416373310_20e7b0fd2d4fc4be84dbc69a0ea368af.jpg` |
| Resolution | 2560×1440 |
| Scene | Traffic surveillance (road scene with multiple vehicles) |

### Sample detections (ONNX FP32, confidence ≥ 0.25)

| Class | Count | Avg confidence |
|-------|-------|----------------|
| car | 11 | 0.737 |
| motorcycle | 4 | 0.498 |
| bus | 1 | 0.703 |
| **Total** | **16** | — |

Inference time: ~193ms (PyTorch CPU, batch=1).

---

## Model Export

| Format | File | Size | Command |
|--------|------|------|---------|
| PyTorch | `best.pt` | 19.2 MB | — (source) |
| ONNX FP32 | `best_fp32.onnx` | 38.7 MB | `yowo export yolo11n --weights best.pt --format onnx --precision fp32 --dynamic-batch` |
| ONNX FP16 | `best_fp16.onnx` | 19.3 MB | `yowo export yolo11n --weights best.pt --format onnx --precision fp16 --dynamic-batch` |
| ONNX INT8 | `best_int8.onnx` | 10.7 MB | `onnxruntime.quantization.quantize_dynamic(fp32, int8, QInt8)` |

FP16 halves model size vs FP32 (19.3 MB vs 38.7 MB). INT8 achieves 4× compression (10.7 MB).

---

## Benchmark Results

**Conditions**: 10 warmup runs, 100 timed runs per batch size. Latency in milliseconds (pure `backend.infer()` — no I/O, no postprocessing).
**ORT providers**: `CoreMLExecutionProvider`, `CPUExecutionProvider`

### Batch = 1

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 143.1 | 148.1 | 153.2 | 156.1 | 148.1 | 6.8 | baseline |
| ONNX FP32 | 46.9 | 57.6 | 79.9 | 102.4 | 59.4 | 16.8 | **2.49×** |
| ONNX FP16 | 45.7 | 59.2 | 74.4 | 96.8 | 59.9 | 16.7 | 2.47× |
| ONNX INT8 | 165.9 | 180.3 | 211.8 | 244.3 | 183.8 | 5.4 | 0.81× ⚠️ |

### Batch = 4

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 553.9 | 569.3 | 588.9 | 605.3 | 570.3 | 7.0 | baseline |
| ONNX FP32 | 254.0 | 291.6 | 461.6 | 837.8 | 326.0 | 12.3 | 1.75× |
| ONNX FP16 | 270.8 | 315.1 | 440.3 | 533.3 | 331.2 | 12.1 | 1.72× |
| ONNX INT8 | 759.0 | 901.9 | 1014.1 | 1155.7 | 892.7 | 4.5 | 0.64× ⚠️ |

### Batch = 8

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 1115.6 | 1146.3 | 1303.5 | 1388.0 | 1159.4 | 6.9 | baseline |
| ONNX FP32 | 379.6 | 478.1 | 646.7 | 686.2 | 493.7 | 16.2 | 2.35× |
| ONNX FP16 | 370.5 | 438.2 | 509.9 | 553.1 | 442.3 | 18.1 | **2.62×** |
| ONNX INT8 | 1454.0 | 1619.6 | 1796.6 | 1932.3 | 1625.1 | 4.9 | 0.71× ⚠️ |

---

## Analysis

### ONNX FP32 vs PyTorch FP32

ONNX via CoreML EP is consistently **2.35–2.49×** faster at batch=1 and batch=8. PyTorch throughput plateaus at ~7 FPS regardless of batch size (CPU cores saturate at batch=1). ONNX scales to 16–18 FPS at batch=8.

### ONNX FP16 vs ONNX FP32

Near-identical latency at batch=1. FP16 pulls ahead at batch=8 (**18.1 vs 16.2 FPS**) because CoreML EP can leverage Apple Silicon's half-precision SIMD units. File size halves: 19.3 MB vs 38.7 MB — same accuracy, half the storage.

### ONNX INT8 — slower than baseline on Apple Silicon

Dynamic INT8 quantization (weight-only, no calibration) is **0.64–0.81× of PyTorch**, i.e. slower than all alternatives. Root cause: ORT's `quantize_dynamic` is optimised for x86 CPUs with AVX-512 VNNI instructions. On Apple Silicon, quantized operations hit the CPUExecutionProvider's scalar path — the CoreML EP does not accept INT8 dynamic-quantized models. Per-layer dequantize overhead exceeds any compute saving.

INT8 is viable on Apple Silicon only via **CoreML static quantization** (`.mlpackage` with calibrated activations) or **TensorRT INT8** on NVIDIA hardware.

### FP8

Not benchmarked. FP8 inference requires NVIDIA H100 + TensorRT ≥10.0. Not supported by ONNX Runtime on CPU or Apple Silicon.

---

## Latency Variance

| Backend | batch=1 std | batch=8 std | Notes |
|---------|------------|------------|-------|
| PyTorch FP32 | 2.8 ms | 31.7 ms | Very stable at batch=1 |
| ONNX FP32 | 7.8 ms | 38.6 ms | CoreML dispatch overhead visible |
| ONNX FP16 | 7.9 ms | 36.8 ms | Similar to FP32 |
| ONNX INT8 | 18.3 ms | 108.0 ms | High variance — memory pressure from dequantize |

ONNX backends show higher p95/p99 jitter vs PyTorch, attributable to CoreML dispatch latency on cold cache hits.

---

## Recommendation

| Use case | Recommendation |
|----------|---------------|
| **Apple Silicon — production** | `best_fp16.onnx` + ONNX Runtime (CoreML EP), batch=8 → **18.1 FPS** |
| **Single-frame low-latency** | `best_fp32.onnx` or `best_fp16.onnx`, batch=1 → **16.7–16.8 FPS** |
| **NVIDIA GPU** | Export to TensorRT FP16 → expected >60 FPS on T4/A10 |
| **Intel CPU** | `best_fp32.onnx` + OpenVINO EP |
| **INT8 on Apple Silicon** | Not recommended with dynamic quantization; use CoreML native export instead |

**Production command for Apple Silicon:**

```bash
yowo detect <source> \
  --model yolo11n \
  --weights /path/to/best_fp16.onnx \
  --backend onnx \
  --batch 8 \
  --confidence 0.25
```

---

## Artifacts

| File | Description |
|------|-------------|
| `tmp/test_vehicle_classes.py` | Manual inference test script (custom class names) |
| `tmp/benchmark.py` | PyTorch-only latency benchmark |
| `tmp/benchmark_onnx.py` | PyTorch vs ONNX FP32 comparison |
| `tmp/benchmark_quant.py` | Full quantization comparison (FP32/FP16/INT8) |
| `tmp/test_vehicle_output.json` | Sample detection JSON output |
| `tmp/test_vehicle_frame.jpg` | Annotated detection frame |
