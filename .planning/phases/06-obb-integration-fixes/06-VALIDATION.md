---
phase: 6
slug: obb-integration-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_obb_integration.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_obb_integration.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 0 | TUNE-01 | unit stub | `uv run pytest tests/unit/test_obb_integration.py -x -q` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | TUNE-01 | unit | `uv run pytest tests/unit/test_obb_integration.py::test_tune_profile_key_obb -x -q` | ✅ | ⬜ pending |
| 06-01-03 | 01 | 1 | OBB-01 | unit | `uv run pytest tests/unit/test_obb_integration.py::test_tune_profile_key_detect_unchanged -x -q` | ✅ | ⬜ pending |
| 06-01-04 | 01 | 1 | BENCH-01 | unit | `uv run pytest tests/unit/test_obb_integration.py::test_benchmark_pattern_accepts_obb -x -q` | ✅ | ⬜ pending |
| 06-01-05 | 01 | 1 | OBB-03 | unit | `uv run pytest tests/unit/test_obb_integration.py::test_benchmark_obb_dispatches_dota_path -x -q` | ✅ | ⬜ pending |
| 06-01-06 | 01 | 2 | TUNE-01, BENCH-01 | integration | `uv run pytest tests/unit/ -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_obb_integration.py` — stubs for TUNE-01, OBB-03, BENCH-01, OBB-01

*Wave 0 creates the test file with stub tests that initially fail (red), then tasks make them green.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| DOTA dataset mAP output format | OBB-03 | Requires real DOTA v1 dataset files | Run `yowo benchmark --model yolo11n-obb --data-path /path/to/DOTA` and verify mAP50-95 + FPS reported |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
