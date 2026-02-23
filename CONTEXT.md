# yowo — Project Context

> Production YOLO Inference & Export Library for Edge and Cloud Deployment

---

## Project Identity

**Name**: yowo
**Purpose**: A Python library and CLI for deploying YOLO models in production — optimized for edge devices, cloud GPUs, and every platform in between.
**Tagline**: *Ultralytics inference, production-hardened.*

---

## Problem Statement

Ultralytics provides excellent training, validation, and basic inference. But in production, it falls short:

| Problem | Detail |
|---------|--------|
| **No hardware auto-selection** | Users must manually choose the right backend and precision per device |
| **No graceful degradation** | If TensorRT fails, the program crashes rather than falling back to ONNX |
| **No edge memory management** | Heavy SDK imports even on ARM devices that don't have those SDKs |
| **No production stream resilience** | RTSP disconnects crash the inference loop |
| **No model metadata tracking** | No sidecar JSON describing what was exported, when, and for which hardware |

yowo solves all of these while staying thin — it wraps ultralytics rather than reimplementing it.

---

## Scope

**In scope:**
- Multi-backend inference (TensorRT, ONNX Runtime, OpenVINO, PyTorch)
- Model export pipeline with INT8/FP16 quantization (wraps ultralytics export)
- Hardware auto-detection and backend auto-selection
- Production-grade stream processing (images, video files, RTSP, webcam, batch)
- Python library API + CLI tool

**Out of scope (explicitly):**
- Model training (use ultralytics directly)
- Model validation / accuracy benchmarking (use ultralytics `model.val()`)
- Custom model architectures (YOLO families only)
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
| YOLO12 | n, s, m, l, x | Attention-based, better accuracy, slower CPU inference |
| YOLO26 | n, s, m, l, x | Newest, NMS-free, best CPU speed and INT8 quantization |

All models sourced as `.pt` weights from the ultralytics asset registry.
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
| PyTorch | `.pt` | FP32 | Installed with ultralytics (core dep) |

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

## Core Data Flow

```
[Input Source]
    │  (image file | video | RTSP | webcam | directory)
    ▼
[FrameSource]  →  Frame (BGR uint8 HWC numpy)
    │
    ▼
[Preprocessor]  →  PreprocessedTensor (float32 BCHW numpy, normalized)
    │
    ▼
[InferenceBackend]  →  NDArray[float32] (raw model output)
    │
    ▼
[Postprocessor / NMS]  →  Detection (Frame + BoundingBox list)
    │
    ▼
[Output / Caller]
```

`Frame` and `PreprocessedTensor` are the two system primitives. Every module boundary passes one of these.

---

## Module Dependency Graph

```
types.py  errors.py          ← leaf nodes, no dependencies
    │         │
    ▼         ▼
hardware/  models/  io/  postprocess/   ← depend only on types + errors
    │         │      │         │
    └────┬────┘      │         │
         ▼           │         │
    backends/   ─────┘─────────┘
         │
         ▼
    export/      ← depends on models/ + hardware/
         │
         ▼
    engine.py    ← wires all modules
         │
         ▼
    cli/         ← CLI entry point
```

No circular dependencies. To add a feature: find the lowest layer it belongs to, modify only that module.

---

## Dependency Philosophy

| Dependency | Type | Reason |
|------------|------|--------|
| `numpy` | Core | Primitive tensor type |
| `opencv-python-headless` | Core | Frame decoding, letterbox, video IO |
| `ultralytics>=8.3` | Core | Weight download, export backend, model configs |
| `click>=8.1` | Core | CLI framework |
| `pyyaml>=6.0` | Core | Config parsing |
| `onnxruntime` | Optional extra | CPU inference |
| `onnxruntime-gpu` | Optional extra | CUDA inference |
| `tensorrt>=10.0` | Optional extra | TensorRT inference |
| `openvino>=2024.0` | Optional extra | OpenVINO inference |

Install only what your deployment needs:
```bash
# Edge ARM device (CPU only)
pip install yowo[onnx]

# NVIDIA GPU server
pip install yowo[tensorrt]

# Intel server
pip install yowo[openvino]

# Everything
pip install yowo[all]
```

---

## Versioning & Compatibility

| Component | Requirement |
|-----------|-------------|
| Python | >=3.11 |
| ultralytics | >=8.3 |
| CUDA | >=12.0 (if using GPU) |
| TensorRT | >=10.0 (if using TensorRT) |
| OpenVINO | >=2024.0 (if using OpenVINO) |
| Linux kernel | >=5.15 recommended |

yowo follows **semantic versioning** (MAJOR.MINOR.PATCH).
- MAJOR: Breaking API changes
- MINOR: New features, backward-compatible
- PATCH: Bug fixes only

---

## Repository Structure

```
yowo/
├── CONTEXT.md              ← This file. Start here.
├── README.md               ← User-facing quick start
├── pyproject.toml          ← Package config, deps, tools
├── uv.lock                 ← Deterministic lockfile (commit to git)
├── src/
│   └── yowo/
│       ├── README.md       ← Library architecture overview
│       ├── types.py        ← Core primitives (frozen dataclasses + enums)
│       ├── errors.py       ← Exception hierarchy
│       ├── config.py       ← Configuration dataclasses
│       ├── engine.py       ← InferenceEngine orchestrator
│       ├── hardware/       ← Hardware detection + backend selection
│       ├── models/         ← Model registry + weight management
│       ├── backends/       ← Inference backend implementations
│       ├── io/             ← Input sources + preprocessing + output
│       ├── postprocess/    ← NMS + Detection output
│       ├── export/         ← Model export pipeline
│       └── cli/            ← CLI commands
└── tests/
    ├── unit/               ← Pure unit tests (no hardware required)
    └── integration/        ← End-to-end tests (may require GPU)
```

Each subdirectory under `src/yowo/` contains its own `README.md` documenting that module's contract.

---

## Getting Started for Contributors

1. Read this file (`CONTEXT.md`) first
2. Read `src/yowo/README.md` for library architecture
3. Read the `README.md` in the specific module you're working on
4. Never add code to a module that crosses its responsibility boundary
5. If a change requires modifying 3+ modules, stop and redesign the interface
