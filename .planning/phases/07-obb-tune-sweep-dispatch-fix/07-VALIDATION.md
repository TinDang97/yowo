---
phase: 7
slug: obb-tune-sweep-dispatch-fix
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (uv run pytest) |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_sweep.py -x -q` |
| **Full suite command** | `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_sweep.py -x -q`
- **After every plan wave:** Run `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 7-01-01 | 01 | 1 | TUNE-01 (OBB path) | unit | `uv run pytest tests/unit/test_sweep.py::TestMeasureConfigDispatch::test_measure_config_dispatches_obb_engine -x -q` | ❌ W0 | ⬜ pending |
| 7-01-02 | 01 | 1 | TUNE-02 (OBB path) | unit | `uv run pytest tests/unit/test_sweep.py::TestMeasureConfigDispatch::test_measure_config_dispatches_detection_engine -x -q` | ❌ W0 | ⬜ pending |
| 7-01-03 | 01 | 1 | TUNE-03 (OBB path) | unit | `uv run pytest tests/unit/test_obb_integration.py -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_sweep.py` — extend with `TestMeasureConfigDispatch` class containing:
  - `test_measure_config_dispatches_obb_engine` — verifies OBBEngine is used for task=obb
  - `test_measure_config_dispatches_detection_engine` — verifies DetectionEngine is used for task=detect

*Existing infrastructure covers all other phase requirements — no new test files needed.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
