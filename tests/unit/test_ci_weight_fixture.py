"""A test can obtain a real, verified weight anywhere — not just on one laptop.

Red-first for ADD task `ci-weight-fixture`.

The integration tier is not merely unrun, it is unrunnable, and it reports
green while being so:

    conftest.py:16                 /Users/tindang/Downloads/Ultralytics YOLO26.pt
    test_engine_integration.py:25  tmp/weights/yolo26n_statedict.pt  (gitignored, absent)
    conftest.py:52                 bus.jpg fetched per session, pytest.skip on failure

A skip is green. 33 tests collect, the weight-dependent ones never execute,
and nothing has ever said so.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import tomllib
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
INTEGRATION = REPO_ROOT / "tests/integration"
CI_YML = REPO_ROOT / ".github/workflows/ci.yml"
FIXTURES = REPO_ROOT / "tests/fixtures"

# An absolute path under someone's home directory, on any of the three platforms.
_MACHINE_PATH = re.compile(r'["\'](?:/Users/|/home/|C:\\\\Users\\\\)[^"\']+["\']')


def _integration_sources() -> dict[str, str]:
    return {p.name: p.read_text() for p in INTEGRATION.rglob("*.py")}


def test_no_fixture_depends_on_a_machine_specific_path() -> None:
    """covers: M1, R:LOCALPATH — a fixture that works only for its author."""
    offenders = {
        name: _MACHINE_PATH.findall(src)
        for name, src in _integration_sources().items()
        if _MACHINE_PATH.search(src)
    }
    assert not offenders, f"machine-specific paths in fixtures: {offenders}"


def test_weight_fixture_uses_the_verified_resolution_path() -> None:
    """covers: M2, R:UNVERIFIEDFIXTURE — reuse production, do not reimplement."""
    src = "\n".join(_integration_sources().values())
    assert "resolve_weights" in src, (
        "no fixture routes through resolve_weights, so nothing digest-verifies the weight"
    )


def _weight_pulling_suites() -> set[str]:
    """Integration modules that resolve a real weight, by what they call.

    Not by what they spell: a module that iterates `ModelSize` downloads every
    size without ever writing one down.
    """
    return {
        name
        for name, src in _integration_sources().items()
        if name != "conftest.py" and ("resolve_weights" in src or "verified_weight" in src)
    }


def test_the_shared_weight_fixture_targets_the_smallest_pinned_model() -> None:
    """covers: A2 — a 109 MB cold download invites someone to disable the tier.

    Narrowed 2026-09-16 to the shared session fixture, which is what A2 is about.
    This previously grepped every integration source for the literal strings
    `yolo11x` and `yolo26x`, and that proxy was already blind to the case it
    existed to catch: `test_export_parity.py` iterates `for size in ModelSize`
    and contains neither literal, so `parity-all` had been downloading both
    109 MB weights since 2026-09-15 with this check green. Meanwhile
    `test_arch_equivalence.py` compares all ten variants by design — an
    equivalence claim with two sizes missing is a claim about eight variants.

    A spelling grep sorts those two by spelling. The cost it was worried about
    is sorted by the cache, which the second half of this check now binds.
    """
    conftest = (INTEGRATION / "conftest.py").read_text()
    assert "_FIXTURE_SIZE = ModelSize.NANO" in conftest, (
        "the shared session fixture no longer pins the smallest weight. Every "
        "weight-dependent test that does not name its own variant pays this "
        "download on a cold cache."
    )


def test_every_suite_that_pulls_a_weight_runs_where_the_store_is_cached() -> None:
    """covers: A2 — the cost A2 names, bound by cache rather than by spelling.

    A suite may deliberately pull the largest weights; what it may not do is pay
    a cold ~1.42 GB download on every run. That is a property of the job, and it
    holds across every workflow — `parity.yml` escaped a `ci.yml`-only guard
    within the hour on 2026-09-15.
    """
    suites = _weight_pulling_suites()
    assert suites, "no integration suite resolves a weight — the tier stopped testing"

    offenders: dict[str, str] = {}
    for workflow in sorted((REPO_ROOT / ".github/workflows").glob("*.yml")):
        jobs = (yaml.safe_load(workflow.read_text()) or {}).get("jobs", {}) or {}
        for job_id, job in jobs.items():
            steps = job.get("steps", []) or []
            runs = " ".join(" ".join(str(s.get("run", "")).split()) for s in steps)
            named = sorted(s for s in suites if s in runs)
            if not named:
                continue
            cached = any(
                "yowo/weights" in str((s.get("with") or {}).get("path", "")) for s in steps
            )
            if not cached:
                offenders[f"{workflow.name}:{job_id}"] = ", ".join(named)

    assert not offenders, (
        f"these jobs run weight-pulling suites without restoring the weight store: "
        f"{offenders}. Each pays a cold download of every weight it touches, every run."
    )


def test_ci_caches_the_weight_store_keyed_on_the_digest() -> None:
    """covers: M3, A8, A9, E2 — a name-keyed cache serves old bytes after a re-pin."""
    ci = yaml.safe_load(CI_YML.read_text())
    cache_steps = [
        s
        for job in ci["jobs"].values()
        for s in job.get("steps", [])
        if "actions/cache" in s.get("uses", "")
    ]
    assert cache_steps, "no actions/cache step caches the weight store"
    key = str(cache_steps[0].get("with", {}).get("key", ""))
    assert "weights" in str(cache_steps[0].get("with", {}).get("path", ""))
    assert "sha256" in key or "digest" in key or "hashFiles" in key, (
        f"cache key is not bound to the pinned digest: {key!r}"
    )


def test_weight_is_not_committed_or_shipped() -> None:
    """covers: M4 — a licence-encumbered 5.4 MB file must not enter the repo."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    assert not [f for f in tracked if f.endswith(".pt")], (
        f"weights tracked in git: {[f for f in tracked if f.endswith('.pt')]}"
    )
    sdist = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    included = sdist["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert not any(str(i).endswith(".pt") for i in included)


def test_missing_input_fails_in_ci_and_skips_locally() -> None:
    """covers: M5, A4, E1, A16 — a green skip is how this tier died."""
    src = "\n".join(_integration_sources().values())
    assert 'environ.get("CI"' in src or "environ.get('CI'" in src, (
        "no fixture distinguishes CI from local; in CI a missing input must FAIL"
    )


def test_sample_image_is_cached_and_fails_loudly() -> None:
    """covers: A14, A15, A16 — cached like the weight, and NOT committed.

    Committing bus.jpg would put a third-party asset with no stated licence into
    an Apache-2.0 repository — the exact failure `licensing-provenance` documents.
    So it is cached rather than vendored, and a failure to obtain it raises in CI
    instead of skipping, because a green skip is how this tier died.
    """
    src = "\n".join(_integration_sources().values())
    assert 'pytest.skip(f"Cannot download' not in src, (
        "the image fixture still skips on network failure"
    )
    assert not list(FIXTURES.glob("*.jpg")), (
        "a third-party image with no stated licence must not be committed"
    )
