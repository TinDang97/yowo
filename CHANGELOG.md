# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Releases after `v0.1.0` are generated automatically by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
from [Conventional Commits](https://www.conventionalcommits.org/).

---

## [0.1.0] — 2026-02-24

### Features

- **arch**: Remove ultralytics dependency — native YOLO11 and YOLO26 architectures
  (`Backbone + FPNPANNeck + Detect`) implemented from published specifications.
  `build_model()` + `load_weights()` public API. Supports all 10 variants:
  `yolo11{n,s,m,l,x}` and `yolo26{n,s,m,l,x}`.

### Fixes

- **arch**: Correct YOLO11/26 weight mapping for all 10 model variants.
  `C3k` now correctly inherits from `C3` (not `C2f`). Added `C3k2PSA`
  for YOLO26 neck layer 22. `backbone_c3k` and `neck_c3k` are now
  size-based (True for m/l/x) instead of family-based — matching
  actual ultralytics checkpoint structure.

- **arch, backend**: Eliminate non-leaf parameter warning in PyTorch backend.
  `fuse_conv_and_bn` wrapped in `torch.no_grad()` to prevent `CopyBackwards`
  grad_fn on fused Conv2d parameters. EMA checkpoint tensors detached before
  `load_state_dict`. CPU thread pool capped at `cpu_count // 2` to prevent
  memory-bandwidth saturation on many-core machines.

### Performance

- **postprocess, io**: Optimize inference hot path with C++ NMS and zero-alloc ops.
  `preprocess()` uses `cv2.dnn.blobFromImages` (single C++ call).
  NMS replaced with `cv2.dnn.NMSBoxes` + class-offset trick.
  Anchor caching in `Detect._decode()`, DFL `.contiguous()` before softmax,
  `non_blocking=True` H2D transfer.

---

## [0.0.1] — 2026-02-23

Initial beta release.

[0.1.0]: https://github.com/TinDang97/yowo/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/TinDang97/yowo/releases/tag/v0.0.1
