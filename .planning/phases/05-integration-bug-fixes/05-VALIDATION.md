---
phase: 5
slug: integration-bug-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/unit/test_obb_export.py tests/unit/test_public_api.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 0 | OBB-06, CORR-07, INT-P0 | unit | `uv run pytest tests/unit/test_obb_export.py -x -q -k kv_cache` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 0 | TUNE-01, INT-P1 | unit | `uv run pytest tests/unit/test_obb_engine.py -x -q -k tune` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 0 | OBB-01, OBB-02, CORR-08, RELY-05, STRM-01, INT-P2 | unit/smoke | `uv run pytest tests/unit/test_public_api.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_obb_export.py` — extend with kv_cache guard tests (3 cases: obb skips, classify skips, detect calls)
- [ ] `tests/unit/test_obb_engine.py` — extend with tune profile integration test (mock `_load_tune_profile`, assert called when no backend_instance)
- [ ] `tests/unit/test_public_api.py` — new file, `test_public_api_exports` covering all 5 INT-P2 types

*All three test files required as Wave 0 — stubs/extensions must exist before implementation tasks run.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | — | — |

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
