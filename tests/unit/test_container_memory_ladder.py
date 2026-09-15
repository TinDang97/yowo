"""A CPU container degrades instead of being OOM-killed.

`_start_oom_monitor` returned early unless the device was CUDA (`engine.py:589`),
so the most common containerised deployment — pytorch-cpu or onnxruntime on a
memory-limited container — had no ladder at all. It did not degrade; it was
killed. The ladder itself was never CUDA-specific: nothing in
`_apply_oom_recovery` touches a CUDA handle. Only its denominator was.

The denominator is the whole point. Without a cgroup limit there is no honest
fraction to compute — `/proc/meminfo` describes the host — so the monitor must
not start rather than degrade against a number it invented.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.config import InferenceConfig
from yowo.engine import DetectionEngine
from yowo.types import BackendType, HealthStatus

_LIMIT_MB = 512
_LIMIT_BYTES = _LIMIT_MB * 1024 * 1024

_OPEN: list[DetectionEngine] = []


@pytest.fixture(autouse=True)
def _close_engines():
    """Every engine owns an event-bus worker thread; leaking them breaks other files."""
    yield
    while _OPEN:
        with contextlib.suppress(Exception):
            _OPEN.pop().close()


def _mock_backend() -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.load.return_value = None
    mock.warmup.return_value = None
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.unload.return_value = None
    return mock


def _cgroup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **files: str) -> Path:
    root = tmp_path / "cgroup"
    root.mkdir(exist_ok=True)
    for name, content in files.items():
        (root / name.replace("__", ".")).write_text(content)
    monkeypatch.setattr("yowo.hardware._cgroup.DEFAULT_CGROUP_ROOT", root)
    return root


def _cpu_engine(batch_size: int = 8) -> DetectionEngine:
    engine = DetectionEngine(
        InferenceConfig(batch_size=batch_size), backend_instance=_mock_backend()
    )
    _OPEN.append(engine)
    return engine


def _load(engine: DetectionEngine) -> None:
    with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
        engine.load()


# ---------------------------------------------------------------------------
# M4 — the ladder runs on a non-CUDA backend
# ---------------------------------------------------------------------------


def test_a_non_cuda_engine_under_a_limit_runs_the_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M4, A2, A5 — a CPU engine at 85% of its container budget halves."""
    _cgroup(
        tmp_path,
        monkeypatch,
        memory__max=str(_LIMIT_BYTES),
        memory__current=str(int(_LIMIT_BYTES * 0.85)),
    )
    engine = _cpu_engine(batch_size=8)
    _load(engine)

    assert engine._is_cuda is False
    assert engine._oom_thread is not None, (
        "a CPU container with a memory limit must get a monitor; without one it "
        "is OOM-killed rather than degraded"
    )

    engine._apply_oom_recovery(engine._memory_pressure() or 0.0)

    assert engine._batch_size == 4
    assert engine.metrics.degradations_total >= 1
    # `health` derives DEGRADED from the error count and frame staleness, not
    # from this flag — a halved batch is invisible there. Named as a residual on
    # the box rather than changed here: that is a user-visible health-surface
    # change, and this node was not authorised to make one.
    assert engine._health_state is HealthStatus.DEGRADED


def test_the_pressure_is_usage_over_the_cgroup_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M4 — the fraction is the container's, not the host's."""
    _cgroup(
        tmp_path,
        monkeypatch,
        memory__max=str(_LIMIT_BYTES),
        memory__current=str(int(_LIMIT_BYTES * 0.5)),
    )
    engine = _cpu_engine()

    assert engine._memory_pressure() == pytest.approx(0.5)


def test_health_report_memory_pct_is_the_cgroup_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M4 — the health surface stops reporting None inside a container."""
    _cgroup(
        tmp_path,
        monkeypatch,
        memory__max=str(_LIMIT_BYTES),
        memory__current=str(int(_LIMIT_BYTES * 0.25)),
    )
    engine = _cpu_engine()
    _load(engine)

    assert engine.health_report().memory_pct == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# R:PHANTOMLADDER — no limit, no denominator, no monitor
# ---------------------------------------------------------------------------


def test_a_non_cuda_engine_with_no_limit_starts_no_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: R:PHANTOMLADDER, A7 — nothing to measure against, so nothing runs."""
    monkeypatch.setattr("yowo.hardware._cgroup.DEFAULT_CGROUP_ROOT", tmp_path / "absent")
    engine = _cpu_engine()
    _load(engine)

    assert engine._memory_pressure() is None
    assert engine._oom_thread is None
    assert engine.health is HealthStatus.READY
    assert engine.health_report().memory_pct is None
    assert engine.metrics.degradations_total == 0


def test_an_unlimited_cgroup_is_the_same_as_no_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: R:PHANTOMLADDER — `max` must not become a budget."""
    _cgroup(tmp_path, monkeypatch, memory__max="max", memory__current="1000000")
    engine = _cpu_engine()
    _load(engine)

    assert engine._memory_pressure() is None
    assert engine._oom_thread is None


# ---------------------------------------------------------------------------
# A10 / A12 / A14 — the loop's behaviour under a bad tick, and what it says
# ---------------------------------------------------------------------------


def test_an_unreadable_usage_mid_run_does_not_degrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: A10 — a failed reading is skipped, never escalated."""
    root = _cgroup(
        tmp_path,
        monkeypatch,
        memory__max=str(_LIMIT_BYTES),
        memory__current=str(int(_LIMIT_BYTES * 0.10)),
    )
    engine = _cpu_engine(batch_size=8)
    _load(engine)

    # The file goes unreadable after the monitor started.
    (root / "memory.current").write_text("garbage")
    assert engine._memory_pressure() is None

    # One real pass through the loop body, then exit. Setting the stop event
    # instead would skip the body entirely and assert nothing.
    def _one_pass() -> object:
        ticks = iter([False, True])
        return patch.object(engine._oom_stop, "wait", side_effect=lambda timeout: next(ticks))

    with _one_pass(), patch.object(engine, "_apply_oom_recovery") as recovery:
        engine._oom_monitor_loop()
    recovery.assert_not_called()

    assert engine._batch_size == 8
    assert engine._health_state is HealthStatus.READY

    # ...and the same harness DOES reach recovery once the reading is readable,
    # so "not called" above is a property of the bad tick, not of the harness.
    (root / "memory.current").write_text(str(int(_LIMIT_BYTES * 0.85)))
    with _one_pass(), patch.object(engine, "_apply_oom_recovery") as recovery:
        engine._oom_monitor_loop()
    recovery.assert_called_once()
    assert recovery.call_args.args[0] == pytest.approx(0.85)


def test_one_tier_fires_per_poll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """covers: A12 — a reading above every threshold takes exactly one action."""
    _cgroup(
        tmp_path,
        monkeypatch,
        memory__max=str(_LIMIT_BYTES),
        memory__current=str(int(_LIMIT_BYTES * 0.99)),
    )
    engine = _cpu_engine(batch_size=8)
    _load(engine)

    with (
        patch.object(engine, "_evict_lowest_activity_streams") as evict,
        patch.object(engine, "_halve_batch_size") as halve,
    ):
        engine._apply_oom_recovery(0.99)

    # The dead precision rung that used to sit between these two is gone
    # (/tasks/precision-plumbing.md A17): it recovered nothing and `return`ed,
    # which made tier-1 unreachable for the whole [0.90, 0.95) band.
    assert evict.call_count == 1
    assert halve.call_count == 0


def test_the_degradation_log_names_the_budget_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """covers: A14 — the operator can tell a cgroup ladder from a CUDA one."""
    _cgroup(
        tmp_path,
        monkeypatch,
        memory__max=str(_LIMIT_BYTES),
        memory__current=str(int(_LIMIT_BYTES * 0.85)),
    )
    engine = _cpu_engine(batch_size=8)
    _load(engine)

    with caplog.at_level(logging.WARNING, logger="yowo.engine"):
        engine._apply_oom_recovery(0.85)

    message = " ".join(record.getMessage() for record in caplog.records)
    assert "cgroup" in message.lower(), (
        f"the halving line must name the budget it read; got {message!r}. A "
        "batch size that drops with no stated cause is the defect metrics-truth closed."
    )
    assert f"{_LIMIT_MB} MB" in message
