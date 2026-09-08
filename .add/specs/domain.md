---
type: Spec
title: Domain
lens: domain
project: yowo
description: what the product must be true about
tags: [inference, detection, edge, yolo]
sources: [README.md, CLAUDE.md, src/yowo/types.py, src/yowo/engine.py]
generated: { by: add/3.5.0, at: 2026-09-08 }
---
## Now

yowo is a **library** (plus a thin CLI) that runs YOLO detection, classification and oriented-box
models in production — on edge devices and on servers — and exports them to the runtime the target
hardware actually wants. It is not a training framework and not a service.

**The entities of its world**

- **Model** — a YOLO11 or YOLO26 variant (`n/s/m/l/x`) in one of three task families: detection,
  classification (`-cls`), oriented bounding box (`-obb`). Architectures are implemented natively in
  `src/yowo/arch/`; ultralytics is a dev-time comparison oracle, never a runtime dependency.
- **Weights** — a checkpoint, either resolved-and-downloaded by name from a published asset URL or
  supplied by the operator as a local file. A custom checkpoint may carry a non-default `num_classes`.
- **Backend** — the runtime that executes the model: PyTorch, ONNX Runtime, TensorRT, OpenVINO,
  CoreML, or an operator-supplied object satisfying the backend Protocol. Backends are
  interchangeable behind one contract; selection is automatic unless pinned.
- **Hardware** — what the machine actually is (CPU class, CUDA GPU + compute capability, Jetson,
  Apple Silicon, Intel iGPU). Detection drives backend selection and preset config.
- **Source** — where frames come from: an image, a directory, a video file, a webcam, an RTSP
  stream. Sources are *live* or *finite*, and that distinction changes the streaming strategy.
- **Frame → Detection** — one inference produces zero or more detections (box, score, class id,
  class name), or a top-k classification, or oriented boxes.
- **Track / Count** — detections across time become tracked identities (ByteTrack), optionally
  re-identified across cameras, optionally counted against lines and zones.
- **Engine** — the object that owns a loaded model + backend + config and turns a Source into
  results, synchronously, streaming, async, or batched across many streams.

**The rules of its world**

- The same model, weights and input must give the same answer on every backend, within the numeric
  tolerance the precision allows. A backend is an implementation detail, never a semantic one.
- Hardware detection may be wrong or absent; the fallback chain must still terminate on something
  that runs, and must say what it chose and why.
- A live source never blocks forever, and never grows a queue without bound. Frames may be dropped —
  but a dropped frame is counted, never silent.
- Accuracy is a property the operator must be able to *measure*, not a claim they must trust —
  especially after quantization.

## Decisions that bind

- **No ultralytics at runtime.** The architectures are ours; ultralytics is a dev-only oracle used
  by the equivalence harness. This is what makes the dependency footprint deployable on edge.
- **The engine hierarchy is the API.** `BaseEngine(StreamingMixin)` → `DetectionEngine` /
  `ClassificationEngine` / `OBBEngine`. `InferenceEngine` is a backward-compatible alias for
  `DetectionEngine`. Adding a task family means adding an engine, not widening one.
- **Backends are a Protocol, not a closed enum.** `InferenceEngine(backend_instance=...)` is a
  supported extension point, which makes the backend contract a public promise.
- **Weights are addressed by model name with a local-file escape hatch.** Anything that changes how
  a checkpoint is located, verified, or deserialized is a security-floor change.
- **Version 2.5.0 under semver** — the public surface is a promise already made.

## Deltas
- 2026-09-08 · bundle initialised; domain authored from the shipped code and README rather than from
  intent. Evidence: `src/yowo/engine.py`, `src/yowo/backends/`, `README.md`, `CHANGELOG.md`.
