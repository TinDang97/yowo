---
phase: 3
slug: adaptive-optimization-and-batch-processing
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0 |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/unit/test_tune_cli.py tests/unit/test_sweep.py tests/unit/test_tune_profile.py tests/unit/test_batch_cli.py tests/unit/test_batch_runner.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_tune_cli.py tests/unit/test_sweep.py tests/unit/test_tune_profile.py tests/unit/test_batch_cli.py tests/unit/test_batch_runner.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-T1 | 01 | 0 | TUNE-01, TUNE-02, TUNE-03, TUNE-04, BATC-01, BATC-02, BATC-03, BATC-04 | unit stubs | `uv run pytest tests/unit/test_tune_cli.py tests/unit/test_sweep.py tests/unit/test_tune_profile.py tests/unit/test_batch_cli.py tests/unit/test_batch_runner.py -x -q` | ❌ W0 | ⬜ pending |
| 03-01-T2 | 01 | 1 | TUNE-01 | unit (CLI invoke) | `uv run pytest tests/unit/test_tune_cli.py -x -q` | ❌ W0 | ⬜ pending |
| 03-01-T3 | 01 | 1 | TUNE-02 | unit | `uv run pytest tests/unit/test_sweep.py -x -q` | ❌ W0 | ⬜ pending |
| 03-01-T4 | 01 | 1 | TUNE-03 | unit | `uv run pytest tests/unit/test_tune_profile.py -x -q` | ❌ W0 | ⬜ pending |
| 03-01-T5 | 01 | 1 | TUNE-04 | unit | `uv run pytest tests/unit/test_sweep.py::test_oom_skip -x -q` | ❌ W0 | ⬜ pending |
| 03-02-T1 | 02 | 2 | BATC-01 | unit (CLI invoke) | `uv run pytest tests/unit/test_batch_cli.py -x -q` | ❌ W0 | ⬜ pending |
| 03-02-T2 | 02 | 2 | BATC-02 | unit | `uv run pytest tests/unit/test_batch_runner.py::test_uses_profile_batch_size -x -q` | ❌ W0 | ⬜ pending |
| 03-02-T3 | 02 | 2 | BATC-03 | unit | `uv run pytest tests/unit/test_batch_runner.py::test_checkpoint_resume -x -q` | ❌ W0 | ⬜ pending |
| 03-02-T4 | 02 | 2 | BATC-04 | unit | `uv run pytest tests/unit/test_batch_runner.py::test_progress_columns -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_tune_cli.py` — CLI invoke stubs for `yowo tune` subcommand (TUNE-01)
- [ ] `tests/unit/test_sweep.py` — sweep loop unit tests with mock backends (TUNE-02, TUNE-04)
- [ ] `tests/unit/test_tune_profile.py` — fingerprint computation, YAML save/load, fingerprint mismatch warn (TUNE-03)
- [ ] `tests/unit/test_batch_cli.py` — CLI invoke stubs for `yowo batch` subcommand (BATC-01)
- [ ] `tests/unit/test_batch_runner.py` — checkpoint atomic write, resume, progress columns (BATC-02, BATC-03, BATC-04)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No OOM crash during actual GPU sweep | TUNE-04 | Requires real GPU hardware | Run `yowo tune --model yolo11n` on CUDA device; verify no crash even with large batch sizes |
| Profile loads and inference starts instantly on second run | TUNE-03 | Requires real hardware fingerprint | Run `yowo tune` twice; verify second run skips sweep and prints "Using cached profile" |
| Batch GPU utilization > streaming mode | BATC-02 | Requires GPU profiling tool | Run `nvidia-smi dmon` while batch processing; compare to streaming FPS |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
