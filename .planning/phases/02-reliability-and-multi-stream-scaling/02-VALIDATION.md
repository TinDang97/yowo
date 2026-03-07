---
phase: 2
slug: reliability-and-multi-stream-scaling
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/ -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -q` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | STRM-01 | unit | `uv run pytest tests/unit/test_pipeline_scaling.py -x -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | STRM-02 | unit | `uv run pytest tests/unit/test_frame_drop.py -x -q` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | STRM-03 | unit | `uv run pytest tests/unit/test_stream_isolation.py -x -q` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | STRM-04 | unit | `uv run pytest tests/unit/test_stream_isolation.py -x -q` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 2 | STRM-05 | unit | `uv run pytest tests/unit/test_stream_isolation.py -x -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | RELY-01 | unit | `uv run pytest tests/unit/test_oom_monitor.py -x -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | RELY-02 | unit | `uv run pytest tests/unit/test_gpu_recovery.py -x -q` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 2 | RELY-03 | unit | `uv run pytest tests/unit/test_health_check.py -x -q` | ❌ W0 | ⬜ pending |
| 02-02-04 | 02 | 2 | RELY-04 | unit | `uv run pytest tests/unit/test_health_check.py -x -q` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 1 | RELY-05 | unit | `uv run pytest tests/unit/test_metrics_export.py -x -q` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 1 | RELY-05 | unit | `uv run pytest tests/unit/test_metrics_export.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_pipeline_scaling.py` — stubs for STRM-01 (100+ stream concurrency, frame drop rate)
- [ ] `tests/unit/test_frame_drop.py` — stubs for STRM-02 (frame drop policy enforcement)
- [ ] `tests/unit/test_stream_isolation.py` — stubs for STRM-03, STRM-04, STRM-05 (stream isolation, memory bounds, auto-remove)
- [ ] `tests/unit/test_oom_monitor.py` — stubs for RELY-01 (OOM detection, batch-size reduction, precision fallback)
- [ ] `tests/unit/test_gpu_recovery.py` — stubs for RELY-02 (GPU error retry, no-restart recovery)
- [ ] `tests/unit/test_health_check.py` — stubs for RELY-03, RELY-04 (HealthReport, JSON logging, structured status)
- [ ] `tests/unit/test_metrics_export.py` — stubs for RELY-05 (Prometheus/JSON export, latency/throughput/memory/error metrics)

*Existing test infrastructure (pyproject.toml, conftest.py) covers all phase requirements — only new test files needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 100+ concurrent RTSP streams sustained 24h | STRM-01, STRM-04 | Requires real RTSP sources or long-running mock — too slow for CI | Launch 100 RTSP mock streams, monitor memory/crash for 24h |
| OOM batch reduction under real GPU pressure | RELY-01 | Requires actual GPU OOM — cannot safely trigger in unit tests | Fill GPU memory to near-limit, verify batch size auto-reduces |
| Engine restart-free GPU error recovery | RELY-02 | CUDA error injection not safely automatable | Inject device lost error via torch.cuda API, verify stream continues |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
