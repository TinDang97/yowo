---
phase: 04-obb-detection
verified: 2026-03-08T02:30:00Z
status: passed
score: 17/17 must-haves verified
re_verification: false
gaps: []
human_verification:
  - test: "Run `yowo --help` and confirm `detect-obb` appears in the command list"
    expected: "detect-obb appears as a registered CLI command with description"
    why_human: "CLI help text rendering requires a terminal invocation; CliRunner tests confirm registration but not visual help output"
  - test: "Run `yowo detect-obb <satellite_image.jpg> --model yolo11n-obb` with a real DOTA-domain image after downloading weights"
    expected: "OBBDetection results with rotation angles printed; inference completes without error"
    why_human: "Requires actual ultralytics weights download and a real aerial image; network access needed"
---

# Phase 4: OBB Detection Verification Report

**Phase Goal:** Implement full OBB (Oriented Bounding Box) detection support — OBBHead architecture, probiou NMS, OBBEngine inference, CLI detect-obb command, and export pipeline.
**Verified:** 2026-03-08T02:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | OBBHead.forward() produces output of shape (B, 4+nc+1, total_anchors) | VERIFIED | `_heads.py:450` returns `cat([dbox, cls_cat, ang_cat], dim=1)`; test_obb_head.py asserts `(1, 20, 8400)` |
| 2  | dist2rbox decodes DFL distances + angle into (cx, cy, w, h) tensors | VERIFIED | `_heads.py:120-153`; identity at zero-angle tested in test_obb_head.py |
| 3  | probiou_nms suppresses overlapping rotated boxes and preserves non-overlapping ones | VERIFIED | `postprocess/_obb_nms.py:51-125`; NMS suppression tested in test_obb_nms.py |
| 4  | OBBModel builds for all 5 scales (n/s/m/l/x) without error | VERIFIED | `arch/_yolo.py:304`; parameterized test across all 5 sizes in test_obb_model.py |
| 5  | OBB registry resolves yolo11n-obb to v8.3.0 weights URL with nc=15 | VERIFIED | `models/_registry.py:131-140`: `_ASSETS_V83 + "yolo11n-obb.pt"`, `num_classes=15` |
| 6  | Weight loading maps model.23.cv4.* to head.cv4.* with no missing keys | VERIFIED | `arch/_weights.py:326`; `_LAYER_MAP` "model.23."→"head." covers cv4 automatically; tested in test_obb_weights.py |
| 7  | OBBEngine can be constructed and loaded (via context manager) | VERIFIED | `obb_engine.py:47`; lifecycle tests in test_obb_engine.py |
| 8  | detect_obb(frames) returns a list of OBBDetection with one entry per frame | VERIFIED | `obb_engine.py:225-244`; _process_batch calls postprocess_obb and rebuilds OBBDetection with elapsed_ms |
| 9  | OBBEngine emits 'obb_detection' event (not 'detection') | VERIFIED | `obb_engine.py:173`: `return "obb_detection"` |
| 10 | load_obb_weights maps model.23.cv4.* to head.cv4.* with no missing keys | VERIFIED | `arch/_weights.py:326`; reuses `_LAYER_MAP`; 7 tests in test_obb_weights.py |
| 11 | OBBConfig dataclass accepts model_family, model_size, confidence, iou, nc=15 | VERIFIED | `config.py:261`; validates num_classes >= 1, confidence/iou in (0,1) |
| 12 | Running `yowo detect-obb SOURCE --model yolo11n-obb` exits 0 (with mock engine) | VERIFIED | `cli/_main.py:318,358,395-397`; CliRunner tests in test_obb_cli.py assert exit_code==0 |
| 13 | parse_model_name('yolo11n-obb') returns ModelSpec with task='obb' and family=YOLO11, size=NANO | VERIFIED | `_convenience.py:62-69`; tested in test_obb_cli.py |
| 14 | export_model with spec.task='obb' calls build_obb_model and load_obb_weights (not the detect path) | VERIFIED | `export/_exporter.py:83-91`; elif branch tested in test_obb_export.py |
| 15 | OBBEngine task branch wired in PyTorch backend | VERIFIED | `backends/_pytorch.py:168-177`: elif task=="obb" calls build_obb_model + load_obb_weights |
| 16 | OBBEngine task branch wired in engine._resolve_model_meta | VERIFIED | `engine.py:192-193`: elif task=="obb" uses _registry_get_obb |
| 17 | Full test suite passes with no regressions | VERIFIED | 1898 passed, 1 skipped (chromadb optional dep), 5 warnings — no failures |

**Score:** 17/17 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/yowo/types.py` | OBBBox and OBBDetection dataclasses | VERIFIED | Lines 326, 353; both in `__all__` at line 592-593 |
| `src/yowo/arch/_heads.py` | OBBHead nn.Module with cv4 angle branch; dist2rbox | VERIFIED | Lines 120-153 (dist2rbox), 372-450 (OBBHead); both exported from arch/__init__.py |
| `src/yowo/arch/_yolo.py` | OBBModel assembler reusing Backbone + FPNPANNeck | VERIFIED | Line 304; uses OBBHead as head |
| `src/yowo/postprocess/_obb_nms.py` | dist2rbox, probiou_matrix, postprocess_obb | VERIFIED | Lines 51, 130, 216; all in __all__ |
| `src/yowo/models/_registry.py` | _OBB_REGISTRY with 5 yolo11-obb variants, get_obb() | VERIFIED | Line 42 (_OBB_REGISTRY), 84 (get_obb), 183 (5 sizes registered) |
| `src/yowo/obb_engine.py` | OBBEngine(BaseEngine) with detect_obb() + stream_obb() | VERIFIED | Lines 47, 225, 258; also adetect_obb() at 247 |
| `src/yowo/config.py` | OBBConfig dataclass | VERIFIED | Line 261; confidence_threshold=0.25, iou_threshold=0.45, num_classes validated |
| `src/yowo/arch/_weights.py` | load_obb_weights() | VERIFIED | Line 326; in __all__ at 414 |
| `src/yowo/_convenience.py` | parse_model_name extended with -obb suffix | VERIFIED | Lines 62-69; YOLO11-only guard raises ConfigError for yolo26 |
| `src/yowo/cli/_main.py` | detect-obb Click command | VERIFIED | Line 318 (@cli.command("detect-obb")); OBBEngine/OBBConfig imported at module level (lines 12, 15) |
| `src/yowo/export/_exporter.py` | elif spec.task == 'obb' branch | VERIFIED | Lines 83-91; model_stem uses -obb suffix (line 122) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `arch/_heads.py OBBHead.forward` | `postprocess/_obb_nms.py dist2rbox` | calls `dist2rbox(dfl_out, ang_cat, anchors)` | WIRED | `_heads.py:444`: `dist2rbox(dfl_out, ang_cat, self._cached_anchors, anchors_t=...)` |
| `arch/_yolo.py OBBModel` | `arch/_heads.py OBBHead` | `self.head = OBBHead(nc=..., ch=neck_channels)` | WIRED | `_yolo.py` OBBModel constructor uses OBBHead |
| `models/_registry.py _OBB_REGISTRY` | `num_classes=15` | `_make_obb_meta sets num_classes=15` | WIRED | `_registry.py:137`: `num_classes=15  # DOTA v1` |
| `obb_engine.py OBBEngine._build_model_spec` | `models/_registry.py get_obb` | `ModelSpec(task='obb') -> engine.load() -> build_obb_model + load_obb_weights` | WIRED | `engine.py:192-193`, `backends/_pytorch.py:168-177` |
| `obb_engine.py OBBEngine._process_batch` | `postprocess/_obb_nms.py postprocess_obb` | `calls postprocess_obb(raw_t, frames, self._spec, ...)` | WIRED | `obb_engine.py:203-209` |
| `obb_engine.py OBBEngine._result_event_name` | `'obb_detection'` | `property returns literal string` | WIRED | `obb_engine.py:173` |
| `cli/_main.py detect_obb_command` | `_convenience.py parse_model_name` | `_parse_model_spec('yolo11n-obb') -> ModelSpec(task='obb')` | WIRED | `_main.py` calls `_parse_model_spec(model)` which delegates to `parse_model_name` |
| `cli/_main.py detect_obb_command` | `obb_engine.py OBBEngine` | `OBBEngine(config) as engine: engine.stream_obb(src)` | WIRED | `_main.py:395,397` |
| `export/_exporter.py` | `arch/__init__.py build_obb_model` | `elif spec.task == 'obb': build_obb_model(...)` | WIRED | `_exporter.py:84,90` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| OBB-01 | 04-01 | OBB detection head produces oriented bounding boxes with rotation angle | SATISFIED | OBBHead in `_heads.py:372`; angle encoding `(sigmoid-0.25)*pi` at line 425; 38 tests in test_obb_head.py |
| OBB-02 | 04-01 | OBB postprocessing includes rotation-aware NMS | SATISFIED | `probiou_matrix` + `_nms_rotated` in `postprocess/_obb_nms.py`; suppression tested in test_obb_nms.py |
| OBB-03 | 04-01, 04-02 | OBB model variants match ultralytics OBB architecture for yolo11 family | SATISFIED | `OBBModel` builds all 5 scales; `_OBB_REGISTRY` has 5 YOLO11 entries; ultralytics-compatible weight loading |
| OBB-04 | 04-01, 04-02 | OBB weights load correctly from ultralytics-trained checkpoints | SATISFIED | `load_obb_weights` reuses `_LAYER_MAP`; "model.23.cv4.*" maps to "head.cv4.*"; tested in test_obb_weights.py |
| OBB-05 | 04-03 | CLI supports `yowo detect-obb SOURCE --model yolo11n-obb` | SATISFIED | `cli/_main.py:318`; full option parity with detect command; CliRunner tests confirm exit 0 and JSON output |
| OBB-06 | 04-03 | OBB models export to ONNX and TensorRT correctly | SATISFIED | `export/_exporter.py:83-91`; elif obb branch calls build_obb_model + load_obb_weights + fuse/eval; model_stem uses "-obb" suffix |

No orphaned requirements — all 6 OBB requirements appear in plan frontmatter and are covered by implementation.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/yowo/cli/_main.py` | 1-1359 | File length 1359 lines (limit: 700) | Warning | Pre-existing violation (file was 1230 lines before phase 04); phase 04 added ~129 lines for detect-obb command. Not a phase-04-introduced blocker but should be refactored. |

No TODO/FIXME/placeholder comments found in any modified files.
No stub implementations found (no `return {}`, `return []`, `return null` in production paths).
No empty handlers found.

---

### Human Verification Required

#### 1. CLI Help Text

**Test:** Run `uv run yowo --help`
**Expected:** `detect-obb` appears in the command list with its description "Run oriented bounding box (OBB) detection on SOURCE"
**Why human:** CLI help text rendering requires a live terminal invocation; unit tests via CliRunner confirm command registration but not visual formatting

#### 2. End-to-End OBB Inference with Real Weights

**Test:** Download `yolo11n-obb.pt` from ultralytics v8.3.0 assets, run `yowo detect-obb <aerial_image.jpg> --model yolo11n-obb --weights yolo11n-obb.pt`
**Expected:** OBBDetection results printed with frame index, count, and inference time; no errors
**Why human:** Requires network access to download weights and a real DOTA-domain aerial image to verify non-trivial detection output

---

### Gaps Summary

No gaps found. All 17 observable truths are verified, all artifacts exist and are substantive, all key links are wired. The only anti-pattern is a pre-existing file length violation in `cli/_main.py` that predates phase 04.

---

## Test Results

| Test Suite | Tests | Result |
|------------|-------|--------|
| test_obb_head.py | — | PASSED (part of 76 OBB tests) |
| test_obb_nms.py | — | PASSED |
| test_obb_model.py | — | PASSED |
| test_obb_registry.py | — | PASSED |
| test_obb_engine.py | — | PASSED |
| test_obb_weights.py | — | PASSED |
| test_obb_cli.py | — | PASSED |
| test_obb_export.py | — | PASSED |
| **All OBB tests** | **76** | **76 passed** |
| **Full unit suite** | **1899** | **1898 passed, 1 skipped (chromadb)** |

**Quality gate:** ruff clean, pyright 0 errors 0 warnings

---

## Commit Verification

| Hash | Description | Valid |
|------|-------------|-------|
| `aa42ca1` | feat(04-01): OBBBox/OBBDetection types, OBBHead, dist2rbox | Confirmed in git log |
| `05eb379` | feat(04-01): probiou NMS, OBBModel assembler, OBB registry | Confirmed |
| `a001dc0` | feat(04-02): OBBEngine, OBBConfig, load_obb_weights, obb task branch | Confirmed |
| `ac77323` | feat(04-obb-detection): OBB CLI and export pipeline | Confirmed |
| `84bbcaf` | docs(04-03): checkpoint — human verify approved | Confirmed |

---

_Verified: 2026-03-08T02:30:00Z_
_Verifier: Claude (gsd-verifier)_
