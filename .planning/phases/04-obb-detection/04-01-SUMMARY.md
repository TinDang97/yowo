---
phase: 04-obb-detection
plan: "01"
subsystem: arch/postprocess/models
tags: [obb, oriented-bounding-box, detection, dota, probiou, nms, registry]
dependency_graph:
  requires: []
  provides: [OBBHead, OBBModel, dist2rbox, probiou_matrix, postprocess_obb, OBB_REGISTRY, OBBBox, OBBDetection]
  affects: [arch, postprocess, models, types]
tech_stack:
  added: []
  patterns:
    - OBBHead subclasses Detect, adding cv4 angle branch
    - dist2rbox geometry: DFL distances + rotation → (cx,cy,w,h)
    - probiou via Bhattacharyya distance between Gaussians
    - Class-aware offset trick for multi-class NMS separation
key_files:
  created:
    - src/yowo/postprocess/_obb_nms.py
    - tests/unit/test_obb_head.py
    - tests/unit/test_obb_nms.py
    - tests/unit/test_obb_model.py
    - tests/unit/test_obb_registry.py
  modified:
    - src/yowo/types.py
    - src/yowo/arch/_heads.py
    - src/yowo/arch/__init__.py
    - src/yowo/arch/_yolo.py
    - src/yowo/postprocess/__init__.py
    - src/yowo/models/_registry.py
    - src/yowo/models/__init__.py
decisions:
  - OBBHead inherits Detect: reuses stride/anchor cache init, DFL, cv2/cv3 branches — only cv4 angle branch is new
  - dist2rbox placed in _heads.py alongside dist2bbox for co-location and import convenience
  - probiou_matrix matches ultralytics batch_probiou exactly (Bhattacharyya with Gaussian covariance)
  - OBB registry is YOLO11-only (nc=15 DOTA v1); YOLO26 has no OBB weights at v8.4.0
  - angle encoding (sigmoid-0.25)*pi applied exactly once in OBBHead.forward() — not in dist2rbox
metrics:
  duration_minutes: 13
  completed_date: "2026-03-08"
  tasks_completed: 2
  files_created: 5
  files_modified: 7
---

# Phase 4 Plan 01: OBB Architecture Foundation Summary

**One-liner:** OBB detection stack with OBBHead (cv4 angle branch), dist2rbox, probiou NMS, OBBModel assembler, and YOLO11 OBB registry for DOTA v1 (nc=15).

## What Was Built

### Task 1: Types + OBBHead + dist2rbox

- `OBBBox` frozen dataclass in `types.py`: cx, cy, w, h, angle (radians [-pi/4, 3pi/4]), confidence, class_id, class_name
- `OBBDetection` frozen dataclass in `types.py`: frame_index, source_id, boxes tuple, inference_time_ms
- `dist2rbox(pred_dist, pred_angle, anchor_points)` in `_heads.py`: decodes (B,4,A) DFL distances + (B,1,A) rotation angle into (B,4,A) (cx,cy,w,h) rotated boxes
- `OBBHead(Detect)` in `_heads.py`: extends Detect with `cv4` ModuleList (angle branch: Conv→Conv→Conv2d); forward produces `(B, 4+nc+ne, total_anchors)` with angle encoding `(sigmoid-0.25)*pi` applied once

### Task 2: probiou NMS + OBBModel + Registry

- `probiou_matrix(obb1, obb2)` in `postprocess/_obb_nms.py`: pairwise probabilistic IoU via Bhattacharyya distance between Gaussian distributions fitted to rotated boxes
- `postprocess_obb(raw, frames, spec, ...)` in `postprocess/_obb_nms.py`: full NMS pipeline with class-aware offset trick, returns `list[OBBDetection]`
- `OBBModel(nn.Module)` in `arch/_yolo.py`: backbone + FPNPANNeck + OBBHead assembler, with `fuse()` support
- `build_obb_model(family, size, *, num_classes=15)` in `arch/__init__.py`
- `_OBB_REGISTRY` in `models/_registry.py`: 5 YOLO11 OBB entries (n/s/m/l/x), nc=15, v8.3.0 weight URLs
- `get_obb(family, size)` raises `ModelNotFoundError` for YOLO26 or unknown keys

## Verification Results

- `uv run pytest tests/unit/test_obb_head.py tests/unit/test_obb_nms.py tests/unit/test_obb_model.py tests/unit/test_obb_registry.py` → 38 passed
- `uv run pytest tests/unit/` → 1860 passed, 1 skipped (chromadb), 5 warnings — no regressions
- `uv run ruff check src/ tests/ --quiet` → clean
- `uv run pyright src/yowo/` → 0 errors, 0 warnings

## Commits

| Hash    | Message                                                        |
|---------|----------------------------------------------------------------|
| aa42ca1 | feat(04-01): add OBBBox/OBBDetection types, OBBHead nn.Module, and dist2rbox helper |
| 05eb379 | feat(04-01): add probiou NMS, OBBModel assembler, and OBB model registry |

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

- [x] `src/yowo/postprocess/_obb_nms.py` — created
- [x] `src/yowo/types.py` — OBBBox, OBBDetection added
- [x] `src/yowo/arch/_heads.py` — dist2rbox, OBBHead added
- [x] `src/yowo/arch/_yolo.py` — OBBModel added
- [x] `src/yowo/models/_registry.py` — _OBB_REGISTRY, get_obb(), _make_obb_meta() added
- [x] 38 new tests across 4 test files — all pass
- [x] 1860 total unit tests pass (no regressions)
- [x] Commits aa42ca1 and 05eb379 exist in git log
