"""Read the limit this process was given, not the machine it landed on.

Measured 2026-09-13 in ``docker run --cpus=2 --memory=512m`` on a 6-CPU,
12017 MB host (cgroup v2)::

    cpu.max     "200000 100000"  -> 2.0 CPUs     os.cpu_count()          -> 6
    memory.max  "536870912"      -> 512 MB       /proc/meminfo MemTotal  -> 12017 MB

``len(os.sched_getaffinity(0))`` also returned 6. Affinity describes a *cpuset*;
a CFS *quota* is invisible to it. Only the cgroup file carries the number.

Every reader here:

* takes its cgroup root as an argument, so a fixture directory can drive it on
  a machine with no cgroups at all (macOS has none, and a check that only runs
  on Linux is a check that does not run);
* understands cgroup v2 first and falls back to v1, because JetPack — this
  project's edge target — still ships v1;
* returns ``None`` for "no limit here" and never raises. A sizing probe that
  raises turns an unreadable file into a failed load.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

__all__ = [
    "DEFAULT_CGROUP_ROOT",
    "V1_UNLIMITED",
    "effective_cpu_count",
    "read_cpu_quota",
    "read_memory_limit_bytes",
    "read_memory_usage_bytes",
]

DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")

# cgroup v1 has no "max" keyword: it writes PAGE_COUNTER_MAX (or a page-rounded
# variant of it) to mean "unlimited". Read literally that is an 8-exabyte budget,
# and min(host, budget) would then quietly hand back the host reading this module
# exists to stop trusting. Anything at or above 2**62 is the sentinel, not a limit.
V1_UNLIMITED = 1 << 62

# cgroup v2: flat, under the root. cgroup v1: one directory per controller.
_V2_CPU_MAX = "cpu.max"
_V1_CPU_QUOTA = Path("cpu") / "cpu.cfs_quota_us"
_V1_CPU_PERIOD = Path("cpu") / "cpu.cfs_period_us"
_V2_MEMORY_MAX = "memory.max"
_V2_MEMORY_CURRENT = "memory.current"
_V1_MEMORY_LIMIT = Path("memory") / "memory.limit_in_bytes"
_V1_MEMORY_USAGE = Path("memory") / "memory.usage_in_bytes"


def _resolve(root: Path | None) -> Path:
    """Resolve the cgroup root at call time, so the module global stays live."""
    return root if root is not None else DEFAULT_CGROUP_ROOT


def _read(root: Path, name: str | Path) -> str | None:
    """Read one cgroup file, or None for any reason it cannot be read."""
    try:
        return (root / name).read_text().strip()
    except (OSError, ValueError):
        return None


def _as_int(text: str | None) -> int | None:
    """Parse a cgroup integer field, or None when it is not one."""
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def read_cpu_quota(root: Path | None = None) -> float | None:
    """Return the CPU quota in whole-CPU units, or None when unlimited.

    ``2.0`` means "this process may use two CPUs' worth of runtime per period",
    which is what ``--cpus=2`` grants. Returns None for an absent file, the v2
    ``max`` keyword, the v1 ``-1`` sentinel, and every malformed form.

    Args:
        root: Cgroup root to read; defaults to the real one, resolved here at
            call time rather than bound as a default argument at import.
    """
    root = _resolve(root)
    v2 = _read(root, _V2_CPU_MAX)
    if v2 is not None:
        fields = v2.split()
        # The format is exactly "<quota> <period>". A different field count is a
        # file we do not understand, and guessing at one is worse than declining.
        if len(fields) != 2:
            return None
        quota, period = _as_int(fields[0]), _as_int(fields[1])
        if quota is None or period is None or quota <= 0 or period <= 0:
            return None
        return quota / period

    quota = _as_int(_read(root, _V1_CPU_QUOTA))
    period = _as_int(_read(root, _V1_CPU_PERIOD))
    if quota is None or period is None or quota <= 0 or period <= 0:
        return None
    return quota / period


def read_memory_limit_bytes(root: Path | None = None) -> int | None:
    """Return the memory limit in bytes, or None when unlimited."""
    root = _resolve(root)
    v2 = _read(root, _V2_MEMORY_MAX)
    if v2 is not None:
        limit = _as_int(v2)  # "max" parses to None, which is the right answer
        return limit if limit is not None and 0 < limit < V1_UNLIMITED else None

    limit = _as_int(_read(root, _V1_MEMORY_LIMIT))
    return limit if limit is not None and 0 < limit < V1_UNLIMITED else None


def read_memory_usage_bytes(root: Path | None = None) -> int | None:
    """Return current memory usage in bytes, or None when it cannot be read."""
    root = _resolve(root)
    v2 = _as_int(_read(root, _V2_MEMORY_CURRENT))
    if v2 is not None:
        return v2
    return _as_int(_read(root, _V1_MEMORY_USAGE))


def _host_cpu_count() -> int:
    """The widest count the host itself admits, before any quota is applied."""
    # Affinity first: a cpuset is a ceiling the process cannot exceed, and
    # os.cpu_count() does not see it. It does not see a quota either — hence
    # everything above.
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return max(1, len(getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def effective_cpu_count(
    root: Path | None = None,
    host_count: int | None = None,
) -> int:
    """Return the number of CPUs this process may actually use.

    The minimum of what the host offers (affinity, then ``os.cpu_count()``) and
    what the cgroup quota grants. A fractional quota rounds **up**, matching the
    JVM's container support: 1.5 CPUs of runtime is spread across two of them.

    With no quota, this is exactly what ``os.cpu_count()`` returned before —
    an unlimited process must never be sized down.

    Args:
        root: Cgroup root to read. Defaults to the real one, resolved at call
            time so a caller (or a test) can redirect it.
        host_count: Override the host reading. For tests that must state the
            host's core count rather than inherit the machine running them.
    """
    host = host_count if host_count is not None else _host_cpu_count()
    quota = read_cpu_quota(root)
    if quota is None:
        return max(1, host)
    return max(1, min(host, math.ceil(quota)))
