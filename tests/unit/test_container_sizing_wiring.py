"""The call sites read the limit — not just the helper that knows it.

A helper that computes the right number and a call site that ignores it is the
exact shape of the defect this node exists to fix, and it is the shape a
mutation sweep caught one node ago: `metrics-truth` shipped a drop counter that
every check exercised directly while the engine wired nothing to it. So each
surface here is bound twice — once at the source, so deleting the call fails,
and once through the real path, so the number that arrives is the cgroup's.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

from yowo.hardware import detect_system_memory_mb, effective_cpu_count

# --cpus=2 on the measured 6-CPU host.
_QUOTA_2_CPUS = "200000 100000"
_LIMIT_512_MB = "536870912"
_HOST_MB = 12017


@pytest.fixture
def cgroup_2cpu_512mb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every default cgroup read at a container granting 2 CPUs / 512 MB."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.max").write_text(_QUOTA_2_CPUS)
    (root / "memory.max").write_text(_LIMIT_512_MB)
    (root / "memory.current").write_text(str(400 * 1024 * 1024))
    monkeypatch.setattr("yowo.hardware._cgroup.DEFAULT_CGROUP_ROOT", root)
    return root


@pytest.fixture
def cgroup_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every default cgroup read at a root that does not exist."""
    root = tmp_path / "no-cgroup-here"
    monkeypatch.setattr("yowo.hardware._cgroup.DEFAULT_CGROUP_ROOT", root)
    return root


# ---------------------------------------------------------------------------
# M1 — thread sizing
# ---------------------------------------------------------------------------

_SIZING_SOURCES = [
    "src/yowo/backends/_pytorch.py",
    "src/yowo/backends/_onnx.py",
    "src/yowo/backends/_tensorrt.py",
    "src/yowo/tune/_profile.py",
]

# `os.cpu_count()` and `len(os.sched_getaffinity(0))` both returned 6 inside a
# 2-CPU container. Either one, used to derive a budget, is the defect.
_HOST_PROBE = re.compile(r"os\.cpu_count\s*\(|sched_getaffinity")


@pytest.mark.parametrize("path", _SIZING_SOURCES, ids=[Path(p).stem for p in _SIZING_SOURCES])
def test_no_sizing_site_derives_a_budget_from_the_host(path: str) -> None:
    """covers: M1, R:HOSTCOUNT — the host probes are gone from every sizing site."""
    source = Path(path).read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if _HOST_PROBE.search(line) and not line.strip().startswith("#")
    ]

    assert not offenders, (
        f"{path} still sizes from the host: {offenders}. Measured in a 2-CPU "
        "container, both os.cpu_count() and sched_getaffinity report 6."
    )


@pytest.mark.parametrize("path", _SIZING_SOURCES, ids=[Path(p).stem for p in _SIZING_SOURCES])
def test_every_sizing_site_calls_the_effective_count(path: str) -> None:
    """covers: M1 — each site names the cgroup-aware helper."""
    assert "effective_cpu_count" in Path(path).read_text(), (
        f"{path} must derive its budget from effective_cpu_count()"
    )


def test_the_onnx_session_is_built_from_the_cgroup_budget(cgroup_2cpu_512mb: Path) -> None:
    """covers: M1 — the real _build_session_options, not a stand-in for it."""
    pytest.importorskip("onnxruntime")
    from yowo.backends._onnx import OnnxBackend
    from yowo.hardware import get_hardware_profile

    opts = OnnxBackend(get_hardware_profile())._build_session_options("cpu")

    assert opts.intra_op_num_threads == 1, (
        "2 CPUs of quota // 2 is 1 compute thread; the host's 6 would give 3"
    )
    assert opts.inter_op_num_threads == 1


def test_the_pytorch_backend_sizes_threads_from_the_cgroup(cgroup_2cpu_512mb: Path) -> None:
    """covers: M1 — torch actually ends up with the container's thread count."""
    torch = pytest.importorskip("torch")
    from yowo.backends._pytorch import PyTorchBackend
    from yowo.hardware import get_hardware_profile

    original = torch.get_num_threads()
    try:
        PyTorchBackend(get_hardware_profile())._configure_cpu_threads()
        assert torch.get_num_threads() == 1, (
            f"expected 2 // 2 = 1 compute thread under a 2-CPU quota, got {torch.get_num_threads()}"
        )
    finally:
        torch.set_num_threads(original)


def test_the_pytorch_load_path_configures_threads(cgroup_absent: Path) -> None:
    """covers: M1 — load() calls it, so the helper above is not dead code.

    This is the binding the previous node's mutation sweep proved was missing:
    a helper verified in isolation while nothing on the real path calls it.
    """
    from yowo.backends._pytorch import PyTorchBackend

    body = inspect.getsource(PyTorchBackend.load)

    assert "_configure_cpu_threads" in body, (
        "PyTorchBackend.load must configure CPU threads. Deleting the call "
        "leaves torch sizing itself from the host and every check above green."
    )


def test_the_tune_fingerprint_changes_with_the_cgroup_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M1 — two containers on one host do not share a tuned profile."""
    from yowo.tune._profile import compute_fingerprint

    class _HW:
        primary_gpu = None

    fingerprints = set()
    for cpus in ("200000 100000", "400000 100000"):
        root = tmp_path / cpus.replace(" ", "_")
        root.mkdir()
        (root / "cpu.max").write_text(cpus)
        monkeypatch.setattr("yowo.hardware._cgroup.DEFAULT_CGROUP_ROOT", root)
        fingerprints.add(compute_fingerprint(_HW()))  # type: ignore[arg-type]

    assert len(fingerprints) == 2, (
        "a 2-CPU and a 4-CPU container on the same host must not share a tune "
        "cache entry — their optimal batch size differs and the key did not"
    )


# ---------------------------------------------------------------------------
# M2 — memory sizing
# ---------------------------------------------------------------------------


def test_memory_total_is_capped_by_the_cgroup_limit(cgroup_2cpu_512mb: Path) -> None:
    """covers: M2, A4 — 512 MB granted, whatever /proc/meminfo says."""
    total_mb, _ = detect_system_memory_mb()

    assert total_mb == 512, (
        f"the container grants 512 MB; sizing reported {total_mb} MB. The host "
        f"this was measured on reports {_HOST_MB} MB — a 23x overstatement."
    )


def test_memory_available_is_limit_minus_current(cgroup_2cpu_512mb: Path) -> None:
    """covers: M2 — headroom comes from the budget, not from the host's free RAM."""
    _, available_mb = detect_system_memory_mb()

    assert available_mb == 512 - 400, (
        "400 MB of a 512 MB budget is in use, so 112 MB remain; the host's "
        "MemAvailable is irrelevant inside the limit"
    )


def test_usage_above_the_limit_clamps_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: E7 — a process over its limit reports no headroom, never negative."""
    root = tmp_path / "over"
    root.mkdir()
    (root / "memory.max").write_text(_LIMIT_512_MB)
    (root / "memory.current").write_text(str(600 * 1024 * 1024))
    monkeypatch.setattr("yowo.hardware._cgroup.DEFAULT_CGROUP_ROOT", root)

    total_mb, available_mb = detect_system_memory_mb()

    assert total_mb == 512
    assert available_mb == 0


def test_no_limit_returns_the_host_reading_unchanged(cgroup_absent: Path) -> None:
    """covers: M2, R:STARVE, E1, E2 — an unlimited process is not starved."""
    total_mb, available_mb = detect_system_memory_mb()

    assert total_mb > 512 or total_mb == 0, (
        "with no cgroup limit the host reading must survive untouched — 0 only "
        "on a platform whose probe failed, which is the pre-existing contract"
    )
    assert available_mb >= 0
    assert effective_cpu_count() == (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    )


def test_detect_system_memory_mb_takes_a_root(tmp_path: Path) -> None:
    """covers: M3, A1 — the caller can name the cgroup here too."""
    root = tmp_path / "explicit"
    root.mkdir()
    (root / "memory.max").write_text(_LIMIT_512_MB)
    (root / "memory.current").write_text(str(128 * 1024 * 1024))

    assert detect_system_memory_mb(root=root) == (512, 384)


# ---------------------------------------------------------------------------
# M5 — the box says what was measured, and what it costs
# ---------------------------------------------------------------------------

_MILESTONE = Path(".add/milestones/m2-survive-week-two.md")


def _box_9() -> str:
    """Locate m2's last box by its task citation, never by its own prose."""
    lines = [ln for ln in _MILESTONE.read_text().splitlines() if ln.startswith("- [")]
    matches = [ln for ln in lines if "← container-aware-sizing" in ln]
    assert len(matches) == 1, f"expected exactly one box citing this task, found {len(matches)}"
    return matches[0]


def test_box_9_records_the_measurement_and_the_invalidation() -> None:
    """covers: M5 — the numbers and the cost are on the box, not only in a commit."""
    box = _box_9()

    assert box.startswith("- [x]"), "the box is ticked"
    for measured in ("--cpus=2", "--memory=512m", "6", "12017", "512"):
        assert measured in box, f"the box must carry the measured value {measured!r}"
    assert "INVALIDATION" in box, (
        "switching the tune fingerprint invalidates every cached CPU-path profile; "
        "a user must not have to discover that from a cache miss"
    )
    assert "RESIDUAL" in box, "what this change does NOT cover is named on the box"


def test_the_box_does_not_claim_a_ladder_on_an_unlimited_host() -> None:
    """covers: M5, R:PHANTOMLADDER — the box's claim matches the shipped gate."""
    box = _box_9()

    assert "ONLY where a limit is readable" in box, (
        "the box must state the ladder's precondition; claiming a ladder for every "
        "non-CUDA deployment would be false on bare metal"
    )
