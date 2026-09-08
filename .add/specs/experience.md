---
type: Spec
title: Experience
lens: experience
project: yowo
description: who uses it and what they feel
tags: [dx, cli, api]
sources: [README.md, docs/user-guide.md, src/yowo/cli/]
generated: { by: add/3.5.0, at: 2026-09-08 }
---
## Now

yowo has no end users. It has **operators and integrators**, and three of them:

- **The evaluator** — 10 minutes, a laptop, one image. `pip install yowo && yowo detect image.jpg`
  must work with no GPU, no weights on disk, and no config. What they feel: either "this just ran"
  or "this wanted three things I do not have". There is no third outcome and no second chance.
- **The integrator** — embeds `DetectionEngine` in their own service. They read types, not prose.
  They need to know what is public, what will still exist next minor version, what exception a
  failure raises, and what a backend swap does to their numbers. What they feel: confidence that
  pinning `yowo>=2.5,<3` is safe.
- **The edge operator** — puts a model on a Jetson or an Intel box and leaves it running. They care
  about latency, memory over days, what happens when the camera drops, and whether the log says
  enough at 3am. What they feel: either the box is boring, or it is a pager.

**The interface they meet it through**
- The CLI is the demo and the smoke test — it must be self-explanatory from `--help` and must fail
  with an actionable sentence, not a traceback.
- The Python API is the product — `detect()` for the one-liner, the engines for real integration.
- The logs and metrics are the operations interface, and are as much a deliverable as the API.

## Decisions that bind

- **Zero-config first run.** Hardware detection, backend selection and weight resolution are
  automatic. The library chooses, then *tells you* what it chose.
- **The one-liner and the engine are both first-class.** `from yowo import detect` must stay as
  short as the README shows it, and `DetectionEngine` must stay as controllable as a service needs.
- **A failure names its remedy.** "TensorRT not installed" is not an error message; "TensorRT
  backend requested but `tensorrt` is not installed — install it or pass `backend='onnx'`" is.

## Deltas
- 2026-09-08 · authored at bundle init from the README's own promises and the CLI surface.
