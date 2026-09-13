"""The cgroup readers: what the container granted, not what the host has.

Measured 2026-09-13 in ``docker run --cpus=2 --memory=512m`` (cgroup v2) on a
6-CPU / 12017 MB host:

    cpu.max      200000 100000   -> 2.0 CPUs      os.cpu_count()  -> 6
    memory.max   536870912       -> 512 MB        MemTotal        -> 12017 MB

``len(os.sched_getaffinity(0))`` also returned 6: affinity sees a cpuset, never
a quota, so the usual "use affinity instead" fix would not have fixed this.

Every case here is driven from a fixture directory rather than from the real
``/sys/fs/cgroup``, because macOS has no cgroups at all and a check that can
only run on Linux is a check that does not run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yowo.hardware import effective_cpu_count
from yowo.hardware._cgroup import (
    V1_UNLIMITED,
    read_cpu_quota,
    read_memory_limit_bytes,
    read_memory_usage_bytes,
)

# The numbers the real container reported, kept as constants so a reader can
# tie every assertion below back to the measurement in the milestone box.
_MEASURED_HOST_CPUS = 6
_MEASURED_QUOTA_CPUS = 2
_MEASURED_HOST_MB = 12017
_MEASURED_LIMIT_MB = 512
_MEASURED_LIMIT_BYTES = 536870912


def _v2(root: Path, **files: str) -> Path:
    """Write a cgroup v2 layout: every file sits directly under the root."""
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name.replace("__", ".")).write_text(content)
    return root


def _v1(root: Path, **files: str) -> Path:
    """Write a cgroup v1 layout: files live under per-controller subdirectories."""
    for name, content in files.items():
        controller, _, leaf = name.partition("___")
        target = root / controller / leaf.replace("__", ".")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def test_effective_cpu_count_reads_the_quota_not_the_host(tmp_path: Path) -> None:
    """covers: M1, A3 — 2.0 CPUs granted on a 6-CPU host must read as 2."""
    root = _v2(tmp_path, cpu__max=f"{_MEASURED_QUOTA_CPUS * 100000} 100000")

    got = effective_cpu_count(root=root, host_count=_MEASURED_HOST_CPUS)

    assert got == _MEASURED_QUOTA_CPUS, (
        f"the cgroup grants {_MEASURED_QUOTA_CPUS} CPUs and the host has "
        f"{_MEASURED_HOST_CPUS}; sizing read {got}. This is the measured defect: "
        "os.cpu_count() and sched_getaffinity both returned 6 inside that container."
    )


def test_a_fractional_quota_rounds_up_to_a_whole_cpu(tmp_path: Path) -> None:
    """covers: E4 — `--cpus=1.5` yields 2, never 0 and never a float."""
    root = _v2(tmp_path, cpu__max="150000 100000")

    got = effective_cpu_count(root=root, host_count=_MEASURED_HOST_CPUS)

    assert got == 2
    assert isinstance(got, int)


def test_a_narrower_cpuset_still_caps(tmp_path: Path) -> None:
    """covers: E8, A8 — affinity below the quota wins; no quota falls back to the host."""
    root = _v2(tmp_path, cpu__max="400000 100000")  # 4 CPUs of quota
    assert effective_cpu_count(root=root, host_count=2) == 2, (
        "a cpuset of 2 is a ceiling the process cannot exceed whatever the quota says"
    )

    bare = _v2(tmp_path / "bare")
    assert effective_cpu_count(root=bare, host_count=_MEASURED_HOST_CPUS) == _MEASURED_HOST_CPUS


def test_effective_cpu_count_is_exported_as_a_plain_int(tmp_path: Path) -> None:
    """covers: A13 — the contract os.cpu_count() had survives for every caller."""
    value = effective_cpu_count(root=tmp_path / "nothing-here")

    assert isinstance(value, int)
    assert value >= 1
    # Callable with no arguments at all, exactly like the function it replaces.
    assert isinstance(effective_cpu_count(), int)


@pytest.mark.parametrize(
    ("content", "why"),
    [
        ("max 100000", "the literal v2 form for 'no quota'"),
        ("", "an empty file"),
        ("200000", "one field where two are required"),
        ("abc 100000", "a non-numeric quota"),
        ("200000 0", "a zero period, which would divide by zero"),
        ("200000 100000 700000", "more fields than the format has"),
    ],
    ids=["max", "empty", "one-field", "non-numeric", "zero-period", "extra-field"],
)
def test_a_quota_that_is_not_a_quota_reads_as_no_limit(
    tmp_path: Path, content: str, why: str
) -> None:
    """covers: E1, E6, R:RAISES — every malformed form is 'no limit', not a crash."""
    root = _v2(tmp_path / content.replace(" ", "_") or "empty", cpu__max=content)

    assert read_cpu_quota(root) is None, why
    assert effective_cpu_count(root=root, host_count=_MEASURED_HOST_CPUS) == _MEASURED_HOST_CPUS


def test_cgroup_v1_quota_of_minus_one_is_no_limit(tmp_path: Path) -> None:
    """covers: E2 — v1 writes -1 rather than the word max."""
    root = _v1(tmp_path, cpu___cpu__cfs_quota_us="-1", cpu___cpu__cfs_period_us="100000")

    assert read_cpu_quota(root) is None
    assert effective_cpu_count(root=root, host_count=_MEASURED_HOST_CPUS) == _MEASURED_HOST_CPUS


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory_limit_reads_the_cgroup_not_the_host(tmp_path: Path) -> None:
    """covers: M2 — 512 MB granted where /proc/meminfo reports 12017 MB."""
    root = _v2(tmp_path, memory__max=str(_MEASURED_LIMIT_BYTES))

    assert read_memory_limit_bytes(root) == _MEASURED_LIMIT_BYTES
    assert read_memory_limit_bytes(root) // (1024 * 1024) == _MEASURED_LIMIT_MB


def test_the_v1_unlimited_sentinel_is_not_a_budget(tmp_path: Path) -> None:
    """covers: A9, E3 — PAGE_COUNTER_MAX is 'no limit', not an 8 EB allowance."""
    root = _v1(tmp_path, memory___memory__limit_in_bytes="9223372036854771712")

    assert read_memory_limit_bytes(root) is None, (
        "v1 writes a huge sentinel instead of the word 'max'. Read literally it "
        "becomes an 8-exabyte budget, and min(host, budget) then silently keeps "
        "the host reading this node exists to stop trusting."
    )
    assert V1_UNLIMITED <= 9223372036854771712


def test_memory_usage_reads_back(tmp_path: Path) -> None:
    """covers: M2 — current usage is the numerator the ladder needs."""
    root = _v2(tmp_path, memory__current="268435456")

    assert read_memory_usage_bytes(root) == 268435456


# ---------------------------------------------------------------------------
# Both readers, every failure shape
# ---------------------------------------------------------------------------


def test_readers_take_a_root_and_never_raise(tmp_path: Path) -> None:
    """covers: M3, R:RAISES, A1, E5 — a missing root is silence, not an exception."""
    missing = tmp_path / "no-such-cgroup-root"
    assert not missing.exists()

    assert read_cpu_quota(missing) is None
    assert read_memory_limit_bytes(missing) is None
    assert read_memory_usage_bytes(missing) is None
    assert effective_cpu_count(root=missing, host_count=4) == 4

    # A path that exists but is a file, and a file that cannot be parsed.
    a_file = tmp_path / "not-a-directory"
    a_file.write_text("")
    assert read_cpu_quota(a_file) is None
    assert read_memory_limit_bytes(a_file) is None

    garbage = _v2(tmp_path / "garbage", memory__max="not-a-number", memory__current="")
    assert read_memory_limit_bytes(garbage) is None
    assert read_memory_usage_bytes(garbage) is None


def test_the_reader_root_is_a_parameter_not_a_constant() -> None:
    """covers: A1 — the probe this assumption declared: the caller names the cgroup."""
    import inspect

    for fn in (read_cpu_quota, read_memory_limit_bytes, read_memory_usage_bytes):
        params = list(inspect.signature(fn).parameters)
        assert params and params[0] == "root", (
            f"{fn.__name__} must accept the cgroup root so a fixture — and a caller "
            "in a nested cgroup — can name it. Reading a module constant makes this "
            "untestable anywhere but inside a Linux container."
        )


def test_v1_and_v2_fixtures_agree(tmp_path: Path) -> None:
    """covers: M3, A6 — the same limit in either layout yields identical numbers."""
    v2 = _v2(
        tmp_path / "v2",
        cpu__max="200000 100000",
        memory__max=str(_MEASURED_LIMIT_BYTES),
        memory__current="100000000",
    )
    v1 = _v1(
        tmp_path / "v1",
        cpu___cpu__cfs_quota_us="200000",
        cpu___cpu__cfs_period_us="100000",
        memory___memory__limit_in_bytes=str(_MEASURED_LIMIT_BYTES),
        memory___memory__usage_in_bytes="100000000",
    )

    assert read_cpu_quota(v1) == read_cpu_quota(v2) == 2.0
    assert read_memory_limit_bytes(v1) == read_memory_limit_bytes(v2)
    assert read_memory_usage_bytes(v1) == read_memory_usage_bytes(v2)
    assert effective_cpu_count(root=v1, host_count=6) == effective_cpu_count(root=v2, host_count=6)


def test_v2_wins_when_both_layouts_are_present(tmp_path: Path) -> None:
    """covers: A6 — a hybrid host is read as v2, the layout the kernel prefers."""
    root = _v2(tmp_path, cpu__max="200000 100000")
    _v1(root, cpu___cpu__cfs_quota_us="600000", cpu___cpu__cfs_period_us="100000")

    assert read_cpu_quota(root) == 2.0


def test_the_default_root_is_the_real_one() -> None:
    """covers: A1 — the parameter defaults to /sys/fs/cgroup, not to a test path."""
    from yowo.hardware._cgroup import DEFAULT_CGROUP_ROOT

    assert Path("/sys/fs/cgroup") == DEFAULT_CGROUP_ROOT
    # And the real call must be harmless on a machine that has no cgroups.
    assert read_cpu_quota() is None or read_cpu_quota() > 0
    assert effective_cpu_count() >= 1


def test_no_limit_never_sizes_the_process_down(tmp_path: Path) -> None:
    """covers: R:STARVE — absent, `max` and `-1` all leave the host reading alone."""
    host = os.cpu_count() or 1
    for root in (
        tmp_path / "absent",
        _v2(tmp_path / "v2max", cpu__max="max 100000"),
        _v1(tmp_path / "v1neg", cpu___cpu__cfs_quota_us="-1", cpu___cpu__cfs_period_us="100000"),
    ):
        assert effective_cpu_count(root=root, host_count=host) == host
        assert read_memory_limit_bytes(root) is None
