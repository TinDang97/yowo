---
phase: 1
slug: correctness-benchmarking-and-proactive-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 (existing) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/ -x -q` |
| **Full suite command** | `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q`
- **After every plan wave:** Run `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | CORR-08 | unit | `uv run pytest tests/unit/test_warmup_validation.py -x` | Wave 0 | pending |
| 01-01-02 | 01 | 1 | PFIX-01 | unit | `uv run pytest tests/unit/test_reader.py::test_rtsp_reconnect -x` | Wave 0 | pending |
| 01-01-03 | 01 | 1 | PFIX-02 | unit | `uv run pytest tests/unit/test_thread_safety.py -x` | Wave 0 | pending |
| 01-01-04 | 01 | 1 | PFIX-03 | unit | `uv run pytest tests/unit/test_nms.py::test_deterministic_ordering -x` | Wave 0 | pending |
| 01-01-05 | 01 | 1 | PFIX-04 | unit | `uv run pytest tests/unit/test_cli.py::test_info_compat -x` | Wave 0 | pending |
| 01-01-06 | 01 | 1 | PFIX-05 | unit | `uv run pytest tests/unit/test_selector.py::test_error_messages -x` | Partial | pending |
| 01-02-01 | 02 | 2 | BENCH-01 | unit | `uv run pytest tests/unit/test_benchmark_cli.py -x` | Wave 0 | pending |
| 01-02-02 | 02 | 2 | BENCH-04 | unit | `uv run pytest tests/unit/test_benchmark_cli.py::test_json_output -x` | Wave 0 | pending |
| 01-03-01 | 03 | 2 | CORR-01 | integration | `uv run pytest tests/unit/test_benchmark_evaluator.py -x` | Wave 0 | pending |
| 01-03-02 | 03 | 2 | CORR-02 | integration | `uv run pytest tests/unit/test_benchmark_evaluator.py::test_classification -x` | Wave 0 | pending |
| 01-03-03 | 03 | 2 | CORR-04-07 | manual | Hardware-dependent export validation | N/A | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_warmup_validation.py` — stubs for CORR-08
- [ ] `tests/unit/test_thread_safety.py` — stubs for PFIX-02
- [ ] `tests/unit/test_benchmark_cli.py` — stubs for BENCH-01, BENCH-02, BENCH-03, BENCH-04
- [ ] `tests/unit/test_benchmark_evaluator.py` — stubs for CORR-01, CORR-02
- [ ] Add `test_deterministic_ordering` to existing `tests/unit/test_nms.py` — covers PFIX-03
- [ ] Add `test_rtsp_reconnect` to existing `tests/unit/test_reader.py` — covers PFIX-01
- [ ] Add `test_info_compat` to existing `tests/unit/test_cli.py` — covers PFIX-04

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Export format correctness (ONNX, TensorRT, OpenVINO) | CORR-04, CORR-05, CORR-06, CORR-07 | Requires real hardware + exported models + COCO dataset | 1. Export model with `yowo export` 2. Run inference on target device 3. Compare mAP vs PyTorch baseline 4. Verify delta < 1% |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
