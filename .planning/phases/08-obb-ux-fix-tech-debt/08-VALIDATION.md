---
phase: 8
slug: obb-ux-fix-tech-debt
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, 1615 tests) |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_runner.py -x -q` |
| **Full suite command** | `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~15 seconds (quick), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_runner.py -x -q`
- **After every plan wave:** Run `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 0 | BENCH-01 (cosmetic) | unit | `uv run pytest tests/unit/test_benchmark_cli.py::TestBenchmarkMissingData::test_benchmark_missing_data_obb -x` | ❌ W0 | ⬜ pending |
| 8-01-02 | 01 | 1 | BENCH-01 (cosmetic) | unit | `uv run pytest tests/unit/test_benchmark_cli.py::TestBenchmarkMissingData -x -q` | ✅ | ⬜ pending |
| 8-01-03 | 01 | 1 | Tech debt | unit | `uv run pytest tests/unit/test_benchmark_runner.py -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_benchmark_cli.py` — add `test_benchmark_missing_data_obb` to `TestBenchmarkMissingData` class
- [ ] Scan `tests/unit/test_benchmark_runner.py` for patches of `yowo.benchmark._runner._check_backend_available` — update if found

*Wave 0 must complete before Wave 1 begins.*

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
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
