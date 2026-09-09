---
type: Spec
title: System
lens: system
project: yowo
description: how it is built, and what that forecloses
tags: [python, multi-backend, packaging, ci]
sources: [pyproject.toml, .github/workflows/release.yml, src/yowo/, tests/]
generated: { by: add/3.5.0, at: 2026-09-08 }
delta_seq: 2
---
## Now

**Shape.** A single installable Python package, `src/`-layout, ~23.8k LOC under `src/yowo/`
against ~28.6k LOC of tests. Entry point `yowo = yowo.cli._main:cli`. Built and managed with `uv`.

**Layers, outermost first**
- `cli/` — click commands: `detect · detect-obb · classify · track · count · info · models · export`.
- `engine.py · classify_engine.py · obb_engine.py · _streaming.py · _async.py · _convenience.py` —
  the engines and the streaming strategies (`_stream_live`, `_stream_pipeline`, `_stream_sync`,
  `_stream_single`), plus the module-level convenience functions (`detect()`, `classify()`).
- `pipeline/ · batch/` — multi-stream collection, batch scheduling, positional-index detection routing.
- `io/` — sources (image, dir, video, webcam, RTSP), `ThreadedFrameReader`, preprocessing/letterbox.
- `postprocess/` — NMS, box decode, softmax, OBB decode.
- `backends/` — PyTorch, ONNX Runtime, TensorRT, OpenVINO, CoreML, behind a `Protocol`.
- `arch/` — native YOLO11/YOLO26 modules (Conv, C3k, C3k2, C2PSA, SPPF, Attention, Detect, …).
- `hardware/ · tune/ · cache/` — device detection, preset config, feature/KV caches.
- `export/` — ONNX / TensorRT / OpenVINO / CoreML export, INT8 calibration and static quantization.
- `tracking/ · counter/` — ByteTrack, cross-camera ReID + embedding gallery, line/zone counting.
- `metrics/ · events/ · logging/ · errors.py · types.py · config.py` — the cross-cutting surface.
- `benchmark/` — throughput/accuracy harness.

**Dependency strategy.** A deliberately thin core (`numpy`, `opencv-python-headless`, `click`,
`pyyaml`, `requests`, `tqdm`) with every heavy runtime behind an extra
(`pytorch · export · onnx · onnx-gpu · openvino · coreml · tracking · chromadb · benchmark · all`).
TensorRT and the Jetson onnxruntime wheel are documented manual installs — they are not on PyPI in a
form this package can depend on.

**Declared reach.** `requires-python >=3.8`, classifiers through 3.12, Linux for production /
macOS for development, plus Jetson aarch64 as a named target.

**Verification today.** `ruff` + `ruff format --check` + `pyright src/yowo/` + `pytest tests/unit/`,
run by one workflow — `.github/workflows/release.yml` — on push to `main`, on
`ubuntu-latest` / Python 3.11 only, followed by semantic-release.

## Decisions that bind

- **`src/` layout + extras-gated heavy deps.** Importing `yowo` must not import torch. Any new
  import at module scope in the core is a packaging decision, not a style one.
- **`uv` is the toolchain.** `uv.lock` is the reproducibility story; commands are `uv run …`.
- **One CI workflow, release-triggered.** Everything CI does not run is, today, unverified — the
  matrix gap is a system property, not an oversight to be assumed away.
- **Semantic-release owns the version number.** Version bumps come from commit types, so commit
  message discipline is load-bearing infrastructure.
- **pyright over `src/yowo/` is a gate.** The typed surface is checked; the tests are not.

## What this forecloses

- No per-backend CI verification without new runners (GPU, Jetson, macOS, Windows) — so backend
  parity has to be provable by something other than "the suite is green on ubuntu".
- Python 3.8 support constrains syntax and pins several dependency ceilings
  (`onnxruntime<1.20`, `onnxruntime-gpu<1.20`) across the whole matrix.
- The open backend factory means the backend Protocol cannot quietly change shape.

## Deltas
- 2026-09-08 · bundle initialised; system authored from `pyproject.toml`, the single workflow, and
- [SDD · S2 · open · 2026-09-09] A pickle_module shim has four read entry points, not one: Unpickler, load, loads, and torch's legacy non-zip reader, which calls pickle_module.load on the file header three times BEFORE any Unpickler exists. Restricting only Unpickler leaves that header read unrestricted, so a non-zip .pt whose first object is a REDUCE executes it and only then fails on 'Invalid magic number'. Reproduced against the shipped loader: os.mkdir ran. Restrict every entry point the module exposes, not the one the happy path uses. (evidence: /tasks/narrow-loader-allowlist.d/runs/4.md)
- [SDD · S1 · open · 2026-09-09] A milestone exit box can hold more clauses than the task pointed at it. Author RULES from the BOX, not from the task title, or the gate goes green on a contract that covers a third of what was asked. (evidence: /tasks/reproducible-sdist.md)
  the module tree as they stand at v2.5.0.
