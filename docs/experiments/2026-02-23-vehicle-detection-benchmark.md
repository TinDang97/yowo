# Experiment Report: Vehicle Detection Benchmark

**Date**: 2026-02-23
**Author**: Tin Dang
**Hardware**: Apple M4 Pro (CPU only)
**Platform**: macOS 25.3.0, Python 3.11.11

---

## Objective

Evaluate yowo inference performance across two models and multiple backends (PyTorch, ONNX FP32, ONNX FP16, ONNX INT8) on batch sizes 1, 4, 8. Establish a production deployment recommendation for Apple Silicon edge devices.

**Models under test:**
- `best.pt` — YOLO11s fine-tuned, 7-class vehicle detector
- `Ultralytics YOLO26.pt` — YOLO26m base model, 80-class COCO detector (NMS-free head)

---

## Models

### Model A — YOLO11s (best.pt)

| Property | Value |
|----------|-------|
| Weights | `best.pt` |
| Architecture | YOLO11s — 100 layers, 9.4M parameters, 21.3 GFLOPs |
| Input shape | 640×640 |
| Output shape | `(B, 11, 8400)` — 4 bbox + 7 classes |
| Postprocess | Standard YOLO (NMS applied) |
| Task | Custom vehicle detection |

**Class mapping:**

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | motorcycle | 4 | transporter |
| 1 | car | 5 | container |
| 2 | bus | 6 | big_transporter |
| 3 | truck | | |

### Model B — YOLO26m (Ultralytics YOLO26.pt)

| Property | Value |
|----------|-------|
| Weights | `Ultralytics YOLO26.pt` |
| Architecture | YOLO26m — 132 layers, 20.4M parameters, 68.2 GFLOPs |
| Input shape | 640×640 |
| Output shape | `(B, 300, 6)` — NMS-free: 300 proposals × [x1,y1,x2,y2,conf,class_id] |
| Postprocess | NMS-free (confidence filter only) |
| Task | COCO 80-class detection |

---

## Test Image

| Property | Value |
|----------|-------|
| File | `z7556416373310_20e7b0fd2d4fc4be84dbc69a0ea368af.jpg` |
| Resolution | 2560×1440 |
| Scene | Traffic surveillance — road scene with multiple vehicles |

### Detections on test image (PyTorch, confidence ≥ 0.25)

**YOLO11s — best.pt (7 vehicle classes):**

| Class | Count | Avg conf |
|-------|-------|---------|
| car | 11 | 0.737 |
| motorcycle | 4 | 0.498 |
| bus | 1 | 0.703 |
| **Total** | **16** | — |

**YOLO26m — Ultralytics YOLO26.pt (COCO 80 classes):**

| Class | Count | Avg conf |
|-------|-------|---------|
| car | 16 | 0.574 |
| motorcycle | 5 | 0.387 |
| bus | 1 | 0.690 |
| truck | 1 | 0.547 |
| person | 1 | 0.385 |
| stop sign | 1 | 0.280 |
| **Total** | **25** | — |

YOLO26m detects more objects (+9 vs YOLO11s) including persons and stop signs. YOLO11s has higher confidence per class due to domain-specific fine-tuning.

---

## Model Export

### YOLO11s (best.pt)

| Format | File | Size | Command |
|--------|------|------|---------|
| PyTorch | `best.pt` | 19.2 MB | — (source) |
| ONNX FP32 | `best_fp32.onnx` | 38.7 MB | `yowo export yolo11n --weights best.pt --format onnx --precision fp32 --dynamic-batch` |
| ONNX FP16 | `best_fp16.onnx` | 19.3 MB | `yowo export yolo11n --weights best.pt --format onnx --precision fp16 --dynamic-batch` |
| ONNX INT8 | `best_int8.onnx` | 10.7 MB | `onnxruntime.quantization.quantize_dynamic(fp32, int8, QInt8)` |

### YOLO26m (Ultralytics YOLO26.pt)

| Format | File | Size | Command |
|--------|------|------|---------|
| PyTorch | `Ultralytics YOLO26.pt` | 42.2 MB | — (source) |
| ONNX FP32 | `yolo26m_fp32.onnx` | 82.7 MB | `yowo export yolo26m --weights "Ultralytics YOLO26.pt" --format onnx --precision fp32 --dynamic-batch` |
| ONNX FP16 | `yolo26m_fp16.onnx` | 41.9 MB | `yowo export yolo26m --weights "Ultralytics YOLO26.pt" --format onnx --precision fp16 --dynamic-batch` |

---

## Benchmark Results

**Conditions**: 10 warmup runs, 100 timed runs per batch size. Latency in milliseconds — pure `backend.infer()` (no I/O, no postprocessing).
**ORT providers**: `CoreMLExecutionProvider`, `CPUExecutionProvider`

---

### YOLO11s (best.pt) — 9.4M params, 21.3 GFLOPs

#### Batch = 1

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 143.1 | 148.1 | 153.2 | 156.1 | 148.1 | 6.8 | baseline |
| ONNX FP32 | 46.9 | 57.6 | 79.9 | 102.4 | 59.4 | 16.8 | **2.49×** |
| ONNX FP16 | 45.7 | 59.2 | 74.4 | 96.8 | 59.9 | 16.7 | 2.47× |
| ONNX INT8 | 165.9 | 180.3 | 211.8 | 244.3 | 183.8 | 5.4 | 0.81× ⚠️ |

#### Batch = 4

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 553.9 | 569.3 | 588.9 | 605.3 | 570.3 | 7.0 | baseline |
| ONNX FP32 | 254.0 | 291.6 | 461.6 | 837.8 | 326.0 | 12.3 | 1.75× |
| ONNX FP16 | 270.8 | 315.1 | 440.3 | 533.3 | 331.2 | 12.1 | 1.72× |
| ONNX INT8 | 759.0 | 901.9 | 1014.1 | 1155.7 | 892.7 | 4.5 | 0.64× ⚠️ |

#### Batch = 8

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 1115.6 | 1146.3 | 1303.5 | 1388.0 | 1159.4 | 6.9 | baseline |
| ONNX FP32 | 379.6 | 478.1 | 646.7 | 686.2 | 493.7 | 16.2 | 2.35× |
| ONNX FP16 | 370.5 | 438.2 | 509.9 | 553.1 | 442.3 | **18.1** | **2.62×** |
| ONNX INT8 | 1454.0 | 1619.6 | 1796.6 | 1932.3 | 1625.1 | 4.9 | 0.71× ⚠️ |

---

### YOLO26m (Ultralytics YOLO26.pt) — 20.4M params, 68.2 GFLOPs

#### Batch = 1

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 295.3 | 326.1 | 358.4 | 640.0 | 329.7 | 3.0 | baseline |
| ONNX FP32 | 105.7 | 142.6 | 182.5 | 207.5 | 145.7 | 6.9 | **2.26×** |
| ONNX FP16 | 124.1 | 147.9 | 186.4 | 237.5 | 151.6 | 6.6 | 2.18× |

#### Batch = 4

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 1114.3 | 1310.0 | 1551.0 | 1777.4 | 1327.5 | 3.0 | baseline |
| ONNX FP32 | 759.7 | 805.2 | 898.0 | 1016.7 | 811.6 | 4.9 | 1.64× |
| ONNX FP16 | 758.9 | 805.1 | 939.7 | 966.6 | 815.4 | 4.9 | 1.63× |

#### Batch = 8

| Backend | min | p50 | p95 | max | mean | FPS | Speedup |
|---------|-----|-----|-----|-----|------|-----|---------|
| PyTorch FP32 | 1753.3 | 1898.7 | 2021.5 | 2083.6 | 1903.2 | 4.2 | baseline |
| ONNX FP32 | 1012.9 | 1144.8 | 1342.4 | 1756.7 | 1158.6 | **6.9** | **1.64×** |
| ONNX FP16 | 1018.3 | 1204.9 | 1488.6 | 1542.3 | 1219.0 | 6.6 | 1.56× |

---

## Cross-Model Comparison (ONNX FP16, best backend per model)

| Metric | YOLO11s | YOLO26m | Notes |
|--------|---------|---------|-------|
| Parameters | 9.4M | 20.4M | YOLO26m is 2.2× larger |
| GFLOPs | 21.3 | 68.2 | YOLO26m is 3.2× more compute |
| PT file size | 19.2 MB | 42.2 MB | |
| ONNX FP16 size | 19.3 MB | 41.9 MB | |
| Batch=1 FPS (ONNX FP16) | **16.7** | 6.6 | YOLO11s 2.5× faster |
| Batch=8 FPS (ONNX FP16) | **18.1** | 6.6 | YOLO11s 2.7× faster |
| Detections on test image | 16 | 25 | YOLO26m detects more classes |
| NMS required | Yes | **No** | YOLO26m NMS-free head |
| Postprocess overhead | +NMS | +confidence filter only | |

YOLO11s delivers **2.5–2.7× higher throughput** than YOLO26m on Apple Silicon due to significantly fewer parameters and GFLOPs. YOLO26m's NMS-free head eliminates postprocessing latency but the compute savings are not visible at this scale — the model itself is the bottleneck.

---

## Analysis

### ONNX (CoreML EP) vs PyTorch

ONNX via CoreML EP is **2.26–2.49×** faster than PyTorch at batch=1 for both models. The gap narrows at batch=4/8 for YOLO26m (1.56–1.64×) because YOLO26m is memory-bandwidth bound at larger batches.

### ONNX FP16 vs ONNX FP32

- **YOLO11s**: FP16 outperforms FP32 at batch=8 (18.1 vs 16.2 FPS). Negligible difference at batch=1.
- **YOLO26m**: FP16 and FP32 are effectively identical across all batch sizes. YOLO26m at 68 GFLOPs saturates CoreML's compute budget; precision savings are masked by memory bandwidth.

### ONNX INT8 — slower on Apple Silicon (YOLO11s only)

Dynamic INT8 quantization (weight-only, no calibration) is **0.64–0.81× of PyTorch baseline** — slower than all other options. ORT's `quantize_dynamic` targets x86 CPUs with AVX-512 VNNI. On Apple Silicon, the CoreML EP rejects dynamic-quantized models; they fall back to the scalar CPUExecutionProvider path where per-layer dequantize overhead exceeds compute savings.

INT8 is viable on Apple Silicon only via CoreML static quantization (`.mlpackage` with calibrated activations) or TensorRT INT8 on NVIDIA hardware.

### FP8

Not benchmarked. Requires NVIDIA H100 + TensorRT ≥10.0. Not supported by ONNX Runtime on CPU or Apple Silicon.

---

## Recommendation

| Use case | Model | Backend | Config | FPS |
|----------|-------|---------|--------|-----|
| **Apple Silicon — max throughput** | YOLO11s | ONNX FP16 (CoreML) | batch=8 | **18.1** |
| **Apple Silicon — low latency** | YOLO11s | ONNX FP32 (CoreML) | batch=1 | 16.8 |
| **General COCO detection** | YOLO26m | ONNX FP32 (CoreML) | batch=8 | 6.9 |
| **NVIDIA GPU** | Either | TensorRT FP16 | batch=8 | >60 (est.) |
| **Intel CPU** | YOLO11s | ONNX FP32 + OpenVINO EP | batch=4 | est. >20 |
| **INT8 on Apple Silicon** | — | Not recommended | — | slower than PyTorch |

**Recommended production command (Apple Silicon, YOLO11s vehicle detection):**

```bash
yowo detect <source> \
  --model yolo11n \
  --weights /path/to/best_fp16.onnx \
  --backend onnx \
  --batch 8 \
  --confidence 0.25
```

**Recommended production command (Apple Silicon, YOLO26m COCO):**

```bash
yowo detect <source> \
  --model yolo26m \
  --weights /path/to/yolo26m_fp32.onnx \
  --backend onnx \
  --batch 8 \
  --confidence 0.25
```

---

## Artifacts

| File | Description |
|------|-------------|
| `tmp/test_vehicle_classes.py` | YOLO11s detection test with custom class names |
| `tmp/test_yolo26.py` | YOLO26m detection test + full benchmark |
| `tmp/benchmark.py` | PyTorch-only latency benchmark |
| `tmp/benchmark_onnx.py` | PyTorch vs ONNX FP32 comparison |
| `tmp/benchmark_quant.py` | YOLO11s quantization comparison (FP32/FP16/INT8) |
| `tmp/test_vehicle_output.json` | YOLO11s detection JSON output |
| `tmp/test_yolo26_output.json` | YOLO26m detection JSON output |
| `tmp/test_vehicle_frame.jpg` | YOLO11s annotated frame |
| `tmp/test_yolo26_frame.jpg` | YOLO26m annotated frame |
| `tmp/benchmark_yolo26.json` | YOLO26m raw benchmark results |
