# YOWO — Production-Grade Computer Vision Inference Platform

## What This Is

A high-performance computer vision inference platform built on native YOLO11/YOLO26 architectures, designed for large-scale production deployments across edge devices (NVIDIA Jetson, Intel NUC/OpenVINO, CUDA GPU servers). YOWO provides parallel multi-stream inference, adaptive self-optimization per hardware target, and production-hardened reliability for real-world environments. It targets full feature parity with ultralytics while delivering superior deployment automation and scalability.

## Core Value

Inference that is production-ready out of the box — deploy to any supported device and it works correctly, fast, and reliably under sustained real-world load without manual tuning.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from existing codebase. -->

- Native YOLO11 + YOLO26 architecture (10 detection variants, classification variants) — existing
- Multi-backend inference: PyTorch, ONNX, TensorRT, OpenVINO, CoreML — existing
- Multi-stream pipeline with FrameCollector + BatchScheduler + DetectionRouter — existing
- ByteTrack object tracking + ObjectCounter + drawing utils — existing
- Cross-camera ReID with CLIP/FastReID embedding gallery — existing
- INT8 quantization export + TensorRT calibrator — existing
- Source-aware streaming dispatch (live, pipeline, sync, single) — existing
- Hardware auto-detection with cached singleton — existing
- CLI: detect, classify, track, count, info, models, export — existing
- Preset config system (6 devices x 3 sources = 18 presets) — existing
- Event-driven observability via EventBus — existing
- 1615 unit tests passing — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] Deep correctness validation vs ultralytics (inference speed, accuracy, export quality, API completeness)
- [ ] Real-device testing on Jetson Orin/Xavier, Intel NUC, CUDA GPU server
- [ ] Production stress testing: memory pressure, thermal throttling, multi-stream load
- [ ] Export reliability validation on actual hardware (ONNX, TensorRT, OpenVINO)
- [ ] 100+ concurrent stream handling without frame drops or crashes
- [ ] Adaptive auto-tuning per device (best backend, batch size, precision selection)
- [ ] Runtime load adaptation (quality/speed tradeoff based on real-time metrics)
- [ ] Learning from deployment metrics to improve optimization decisions over time
- [ ] OBB (Oriented Bounding Box) detection support
- [ ] High-throughput batch processing for millions of images/clips

### Out of Scope

<!-- Explicit boundaries. -->

- Segmentation (instance/semantic) — defer to future milestone
- Pose estimation — defer to future milestone
- Apple Silicon/CoreML testing — no Mac hardware available currently
- Web UI / dashboard — this is an inference library, not a web app
- Training / fine-tuning — inference-only platform
- Cloud orchestration / Kubernetes — focus on single-device and multi-stream first

## Context

YOWO (v2.3.0-dev) is a mature inference library with native YOLO architectures, 5 backend implementations, streaming pipelines, tracking, and cross-camera ReID. Architecture correctness has been validated against ultralytics (10/10 variants pass `tmp/compare_arch.py`).

However, the platform has only been tested in controlled/small-scale environments. The gap between "works in unit tests" and "production-ready on real devices under sustained load" is the primary focus of this milestone. Key concerns from codebase analysis:
- No real-device benchmarking against ultralytics inference speed
- Multi-stream pipeline untested beyond small concurrent counts
- Export artifacts not validated on actual target hardware
- No adaptive optimization — config is manual per deployment
- Memory and thermal behavior under sustained load is unknown

Target devices available for testing:
- NVIDIA Jetson Orin/Xavier (TensorRT)
- Intel NUC with integrated GPU (OpenVINO)
- CUDA GPU server (PyTorch/TensorRT baseline)

## Constraints

- **Tech stack**: Python 3.11+, uv package manager, existing backend Protocol architecture
- **Accuracy**: Must match ultralytics mAP on COCO for all supported model variants
- **Latency**: Must be competitive with or faster than ultralytics per-frame inference
- **Compatibility**: Existing public API (engine, CLI, convenience functions) must not break
- **Testing**: All features must have unit tests; quality gates (ruff + pyright + pytest) must pass
- **Edge devices**: Must work within device memory/thermal constraints without manual tuning

## Key Decisions

<!-- Decisions that constrain future work. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Correctness before scale | Can't optimize what isn't correct; validate vs ultralytics first | — Pending |
| Detection + Classification + OBB only | Focus on what exists + one new task; defer seg/pose | — Pending |
| Real-device testing required | Unit tests aren't enough; production means real hardware | — Pending |
| Auto-tune + runtime adapt + learning | Full adaptive self-optimization, not just static config | — Pending |
| No Mac testing this milestone | No Apple Silicon hardware available | — Pending |

---
*Last updated: 2026-03-07 after initialization*
