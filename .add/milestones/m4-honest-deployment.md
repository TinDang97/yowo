---
type: Milestone
title: Exported and quantized artifacts do what their metadata says
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: milestone-planner
---
## CARD
goal: An exported or quantized artifact does exactly what its metadata says, and its cost is measurable before deployment.
why: `precision` is accepted by config and reported by `health_report()` while being consumed by zero backends. ONNX export writes a graph whose own sidecar cannot describe it and which fails to load on a default install. INT8 is calibrated on stretch-resized images against letterboxed inference, under a docstring claiming they match — with no accuracy gate anywhere. Caches are keyed on filenames. An operator currently cannot answer what latency or accuracy they will get before shipping.
next: add new task <slug>

## SCOPE
In:  ONNX external-data and opset truth · precision plumbed to backends or removed from the surface · TensorRT engine and INT8 calibration cache keying · INT8 calibration/inference preprocessing parity and a measured accuracy delta · bounded postprocess (pre-NMS top-k, max_det) · benchmark backend attribution · feature-cache honesty · the `ExportResult`/`ExportMetadata` contract
Out: the public API stability policy around these types (→ m5-declare-ga) · new backends or new model families

## GROUND
touches: src/yowo/export/ · src/yowo/backends/ · src/yowo/postprocess/ · src/yowo/benchmark/ · src/yowo/tune/ · src/yowo/cache/ · src/yowo/config.py · src/yowo/engine.py
risks:
  - Plumbing `precision` for real will change output numerics on backends that silently ran FP32 while reporting FP16. Users' current numbers are the wrong ones, but they are the ones they have.
  - Fixing INT8 calibration to letterbox changes every existing quantized artifact's scales. Old `.calib` caches must be invalidated, not reused.
  - Adding `max_det` bounds p99 by discarding detections in dense scenes — a behaviour change that needs a default chosen deliberately and documented.

## EXIT
- [ ] Every exported artifact loads from a clean environment using only what its sidecar names, and the recorded opset is the produced opset   (← onnx-external-data)
- [ ] `precision` reaches the backend that executes, or no longer appears in config, health or metrics. Plumb-vs-remove is a human decision — the two differ in reversibility, not effort   (← precision-plumbing)
- [ ] A test perturbs each element of every cache key's stated invalidation set — GPU arch, TRT version, driver, precision, shape profile, calibration set, and the cgroup-aware cpu count `container-aware-sizing` changed — and asserts the key changes   (← artifact-cache-keys)
- [ ] INT8 calibration uses the inference preprocessing path, and every quantized artifact ships a measured accuracy delta against its source, computed on the m3 dataset   (← int8-calibration-parity)
- [ ] Postprocess cost is bounded independent of scene content, with the bound measured   (← bounded-postprocess)
- [ ] Benchmark and tune report the backend that actually executed, never the one requested   (← benchmark-truth)
- [ ] The feature cache's default is asserted by a test; its failure mode is documented where a user meets it (review item, not a command)   (← feature-cache-honesty)
- [ ] The documented export return type is the one the function returns   (← export-result-contract)
## CLOSE
evidence: <one row per task>
