# yowo — Project Context

> Production YOLO Inference & Export Library for Edge and Cloud Deployment

---

## Project Identity

**Name**: yowo
**Purpose**: A Python library and CLI for deploying YOLO models in production — optimized for edge devices, cloud GPUs, and every platform in between.
**Tagline**: *Native YOLO inference, production-hardened.*

---

## Problem Statement

Standard YOLO libraries focus on training and basic inference but fall short in production:

| Problem | Detail |
|---------|--------|
| **No hardware auto-selection** | Users must manually choose the right backend and precision per device |
| **No graceful degradation** | If TensorRT fails, the program crashes rather than falling back to ONNX |
| **No edge memory management** | Heavy SDK imports even on ARM devices that don't have those SDKs |
| **No production stream resilience** | RTSP disconnects crash the inference loop |
| **No model metadata tracking** | No sidecar JSON describing what was exported, when, and for which hardware |

yowo solves all of these with native YOLO11 and YOLO26 architectures, Apache-2.0 licensed.

---

## Scope

**In scope:**
- Multi-backend inference (TensorRT, ONNX Runtime, OpenVINO, PyTorch)
- Model export pipeline with INT8/FP16 quantization (native torch.onnx.export)
- Hardware auto-detection and backend auto-selection
- Production-grade stream processing (images, video files, RTSP, webcam, batch)
- Python library API + CLI tool

**Out of scope (explicitly):**
- Model training
- Model validation / accuracy benchmarking
- Custom model architectures (YOLO11 and YOLO26 families only)
- Windows support (Linux-first; macOS for development only)
- Non-YOLO models

---

## Target Users

- ML engineers deploying YOLO models to production environments
- Robotics / embedded systems engineers running on Jetson or ARM SBCs
- DevOps / MLOps engineers building inference pipelines
- Researchers who want reproducible edge inference without platform-specific boilerplate

---

## Supported Models

| Family | Sizes | Notes |
|--------|-------|-------|
| YOLO11 | n, s, m, l, x | Most stable, best for production baseline |
| YOLO26 | n, s, m, l, x | NMS-free, best CPU speed and INT8 quantization |

All models implemented natively in `src/yowo/arch/`. Weights loaded from `.pt` checkpoints.
Adding a new YOLO family: register in `src/yowo/models/_registry.py` — no other files change.

---

## Supported Platforms

| Platform | Device Examples | Backend Priority |
|----------|----------------|-----------------|
| NVIDIA GPU (server) | A100, L4, T4, RTX 30xx/40xx | TensorRT > ONNX (CUDA) > PyTorch |
| NVIDIA Jetson | Orin, Xavier, Nano | TensorRT > ONNX (CUDA) > PyTorch |
| Intel CPU/iGPU | i7, Xeon, Arc | OpenVINO > ONNX > PyTorch |
| x86 CPU | Generic Linux x86_64 | ONNX > OpenVINO > PyTorch |
| ARM CPU | Raspberry Pi 5, AWS Graviton, Ampere Altra | ONNX > PyTorch |

**OS**: Linux required for production. macOS supported for development (no TensorRT).

---

## Supported Backends

| Backend | Format | Precision | Install |
|---------|--------|-----------|---------|
| TensorRT | `.engine` | FP32, FP16, INT8 | `pip install yowo[tensorrt]` |
| ONNX Runtime | `.onnx` | FP32, FP16 | `pip install yowo[onnx]` or `yowo[onnx-gpu]` |
| OpenVINO | `_openvino_model/` | FP32, FP16, INT8 | `pip install yowo[openvino]` |
| PyTorch | `.pt` | FP32 | `pip install yowo[pytorch]` |

Backends are **optional extras**. Only the ones you install are loaded. Unused backends never import their SDK — critical for edge devices.

---

## Architecture Principles

yowo follows **black box module design**:

1. **Black Box Interfaces** — Every module exposes a clean Protocol/ABC. Implementation is hidden.
2. **Single Responsibility** — One module = one clear job. One person can own it.
3. **Replaceable Components** — Any module can be rewritten using only its interface.
4. **Primitive-First** — Core data types (`Frame`, `Detection`, `PreprocessedTensor`) flow through the entire system unchanged.
5. **KISS** — Simplest solution that works. No premature abstraction.
6. **YAGNI** — Build only what is needed now. Design interfaces that survive future needs.

---

## Versioning & Compatibility

yowo follows **semantic versioning** (MAJOR.MINOR.PATCH).
- MAJOR: Breaking API changes
- MINOR: New features, backward-compatible
- PATCH: Bug fixes only

---

## Repository Structure - use mcp__serena

Each subdirectory under `src/yowo/` contains its own `README.md` documenting that module's contract.

---

## Getting Started for Contributors

1. Read this file (`CONTEXT.md`) first
2. Read `src/yowo/README.md` for library architecture
3. Read the `README.md` in the specific module you're working on
4. Never add code to a module that crosses its responsibility boundary
5. If a change requires modifying 3+ modules, stop and redesign the interface
