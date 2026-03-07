---
phase: 4
slug: obb-detection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_obb*.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_obb*.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 1 | OBB-01 | unit | `uv run pytest tests/unit/arch/test_obb_detect.py -x -q` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 1 | OBB-01 | unit | `uv run pytest tests/unit/arch/test_obb_detect.py -x -q` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 1 | OBB-02 | unit | `uv run pytest tests/unit/arch/test_obb_detect.py -x -q` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 1 | OBB-02 | unit | `uv run pytest tests/unit/test_obb_nms.py -x -q` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 2 | OBB-03 | unit | `uv run pytest tests/unit/test_obb_engine.py -x -q` | ❌ W0 | ⬜ pending |
| 4-02-03 | 02 | 2 | OBB-04 | unit | `uv run pytest tests/unit/test_obb_engine.py -x -q` | ❌ W0 | ⬜ pending |
| 4-03-01 | 03 | 3 | OBB-05 | unit | `uv run pytest tests/unit/cli/test_obb_cli.py -x -q` | ❌ W0 | ⬜ pending |
| 4-03-02 | 03 | 3 | OBB-06 | unit | `uv run pytest tests/unit/test_obb_export.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/arch/test_obb_detect.py` — stubs for OBB-01, OBB-02
- [ ] `tests/unit/test_obb_nms.py` — stubs for OBB-02 (probiou NMS)
- [ ] `tests/unit/test_obb_engine.py` — stubs for OBB-03, OBB-04
- [ ] `tests/unit/cli/test_obb_cli.py` — stubs for OBB-05
- [ ] `tests/unit/test_obb_export.py` — stubs for OBB-06

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `yowo detect-obb SOURCE --model yolo11n-obb` E2E output | OBB-05 | Real weights needed | Download yolo11n-obb.pt, run CLI, verify OBB boxes rendered |
| TensorRT OBB export + inference | OBB-06 | TensorRT GPU required | Export model, run inference, verify angle preserved |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
