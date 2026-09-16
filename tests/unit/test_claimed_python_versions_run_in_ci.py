"""Every Python version the package claims must be executed by CI.

`pyproject.toml` declares `requires-python = ">=3.8"` and classifiers for 3.8,
3.9, 3.10, 3.11 and 3.12. Measured 2026-09-16 across all 18 jobs in all four
workflows: **every one of them ran python 3.11 on ubuntu-latest.** Four of the
five claimed versions had never executed anything. A classifier is a promise
to a user, and an unexecuted promise is a string in a metadata file.

The claimed set is DERIVED from pyproject here, never kept beside it. A list
maintained by hand next to the thing it describes is the drift this repository
has now found seven separate times.

The test suite cannot stand in for the claim. Measured on real interpreters:

    CPython 3.8.20   src/yowo compiles 96/96   tests compile 141/163
                     (22 modules use parenthesized context managers, 3.10+)
    CPython 3.10.20  src/yowo compiles 96/96   tests compile 163/163
                     but 6 modules `import tomllib`, which is 3.11+ stdlib
    CPython 3.12.13  unit tier: 2863 passed

So the test suite's floor is **3.11**, four versions above the package's, and
the claim is checked by installing the package as a consumer does and
exercising its documented surface (`scripts/check_public_surface.py`).

Authored red under ADD task `ci-matrix`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOC = REPO_ROOT / "docs" / "ci-required-checks.md"
PROTECTION = REPO_ROOT / "docs" / "branch-protection.json"
SURFACE_SCRIPT = REPO_ROOT / "scripts" / "check_public_surface.py"

CLAIM_JOB = "python-claim"
UNIT_NEWEST_JOB = "unit-newest"

#: The gaps this milestone deliberately does NOT close, and why. Each must
#: still be findable in the documentation — a limitation kept only in a
#: comment stops being read (M4).
RECORDED_GAPS = {
    "unit tier floor": "tomllib",
    "operating system": "ubuntu",
}


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text())


def claimed_versions() -> list[str]:
    """The Python versions the package claims, from its classifiers.

    Derived, never listed. `Programming Language :: Python :: 3.10` is the
    promise; a matrix constant beside it is a copy that can rot.
    """
    project = _pyproject().get("project") or {}
    found = []
    for classifier in project.get("classifiers", []):
        match = re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", classifier.strip())
        if match:
            found.append(match.group(1))
    return sorted(found, key=lambda v: tuple(int(p) for p in v.split(".")))


def requires_python_floor() -> str:
    spec = ((_pyproject().get("project") or {}).get("requires-python") or "").strip()
    match = re.search(r">=\s*(3\.\d+)", spec)
    assert match, f"cannot read a floor from requires-python={spec!r}"
    return match.group(1)


def _ci() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text())


def _job(job_id: str) -> dict[str, Any]:
    jobs = _ci()["jobs"]
    assert job_id in jobs, f"ci.yml has no {job_id!r} job. Jobs present: {sorted(jobs)}"
    return jobs[job_id]


def matrix_versions(job: dict[str, Any]) -> list[Any]:
    return list(((job.get("strategy") or {}).get("matrix") or {}).get("python-version") or [])


def test_every_claimed_python_version_is_executed_by_a_job() -> None:
    """binds R:CLAIM_UNRUN — four of five claimed versions had never run.

    The sets must be EQUAL, not merely overlapping: a leg for a version the
    package does not claim is as wrong as a claim with no leg.
    """
    claimed = set(claimed_versions())
    assert claimed, "pyproject declares no Python classifiers, so there is no claim to check"
    exercised = {str(v) for v in matrix_versions(_job(CLAIM_JOB))}
    assert exercised == claimed, (
        f"claimed but never executed: {sorted(claimed - exercised)}; "
        f"executed but not claimed: {sorted(exercised - claimed)}. "
        f"Add the matrix leg, or drop the classifier — a classifier is a promise."
    )


def test_the_matrix_is_derived_from_pyproject_not_a_hand_written_list() -> None:
    """binds M1 and R:HANDLIST.

    Also binds the YAML trap that makes a hand-written matrix silently wrong:
    unquoted `3.10` parses as the FLOAT 3.1, and `setup-python` would then
    install Python 3.1 or fail obscurely. Every entry must be a string.
    """
    raw = matrix_versions(_job(CLAIM_JOB))
    non_strings = [v for v in raw if not isinstance(v, str)]
    assert not non_strings, (
        f"these matrix entries are not strings: {non_strings!r}. Unquoted `3.10` "
        f"in YAML is the float 3.1, which is a different Python. Quote them."
    )
    # And the derivation itself: this test computes the expectation from the
    # manifest rather than restating it, so the two cannot drift apart.
    assert [str(v) for v in raw] == claimed_versions()


def test_the_requires_python_floor_agrees_with_the_lowest_classifier() -> None:
    """binds E3 — two claims that can disagree eventually will."""
    floor, lowest = requires_python_floor(), claimed_versions()[0]
    assert floor == lowest, (
        f"requires-python says >={floor} but the lowest classifier is {lowest}. "
        f"pip reads the first and humans read the second."
    )


def test_every_claim_job_is_registered_and_published() -> None:
    """Three places, or it gates nothing — and one name per version (A6)."""
    from tests.unit.test_ci_contract import FROZEN_JOB_IDS

    for job_id in (CLAIM_JOB, UNIT_NEWEST_JOB):
        assert job_id in FROZEN_JOB_IDS, f"{job_id!r} is not registered in FROZEN_JOB_IDS"

    template = str(_job(CLAIM_JOB).get("name", ""))
    assert "matrix.python-version" in template, (
        f"the {CLAIM_JOB!r} job must publish a name carrying the version, so a reader "
        f"can see WHICH Python broke. Got {template!r}."
    )
    doc, payload = DOC.read_text(), PROTECTION.read_text()
    for version in claimed_versions():
        name = f"Python Claim ({version})"
        assert name in doc, f"{name!r} is absent from {DOC.name}"
        assert f'"{name}"' in payload, f"{name!r} is absent from {PROTECTION.name}"

    newest = f"Unit Tests ({claimed_versions()[-1]})"
    assert _job(UNIT_NEWEST_JOB).get("name") == newest
    assert newest in doc and f'"{newest}"' in payload


def test_the_claim_job_exercises_the_documented_public_surface() -> None:
    """An import-only smoke would pass with the CLI broken."""
    runs = " ".join(
        " ".join(str(s.get("run", "")).split()) for s in _job(CLAIM_JOB).get("steps", []) or []
    )
    assert "scripts/check_public_surface.py" in runs, (
        f"{CLAIM_JOB!r} must run scripts/check_public_surface.py — the surface is "
        f"the claim, and `import yowo` alone is not the surface"
    )
    source = SURFACE_SCRIPT.read_text()
    for required in ("PUBLIC_API", "PUBLIC_SUBMODULES", "CLI_COMMANDS", "list_available"):
        assert required in source, f"the surface script no longer checks {required}"


def test_the_claim_job_installs_the_package_as_a_consumer_does() -> None:
    """binds M3 and R:DEVENV_PROVES_SHIP.

    The dev group cannot resolve on 3.8 at all — `pre-commit>=3.8.0` requires
    Python >=3.9 (measured 2026-09-16) — so a run that installed it would
    either fail or be measuring something other than the shipped package.
    """
    for step in _job(CLAIM_JOB).get("steps", []) or []:
        run = " ".join(str(step.get("run", "")).split())
        if "install" not in run and "sync" not in run and "pip" not in run:
            continue
        assert "--group dev" not in run and "--all-groups" not in run, (
            f"the {CLAIM_JOB!r} job installs the dev group ({run!r}). It must install "
            f"the package the way a consumer does, or it proves the wrong thing."
        )


def test_the_claim_matrix_does_not_stop_at_the_first_failure() -> None:
    """binds A11 — one bad version must not hide the other four."""
    strategy = _job(CLAIM_JOB).get("strategy") or {}
    assert strategy.get("fail-fast") is False, (
        "the claim matrix must set `fail-fast: false`; otherwise the first broken "
        "version cancels the rest and you learn about one problem per run"
    )


def test_the_unit_tier_runs_on_the_newest_claimed_version() -> None:
    """Measured 2026-09-16 on a real CPython 3.12.13: 2863 passed."""
    job = _job(UNIT_NEWEST_JOB)
    newest = claimed_versions()[-1]
    pys = {
        str((s.get("with") or {}).get("python-version"))
        for s in job.get("steps", []) or []
        if (s.get("with") or {}).get("python-version")
    }
    assert pys == {newest}, f"{UNIT_NEWEST_JOB!r} must run python {newest}, got {pys or 'none'}"
    runs = " ".join(" ".join(str(s.get("run", "")).split()) for s in job.get("steps", []) or [])
    assert "pytest tests/unit" in runs, f"{UNIT_NEWEST_JOB!r} does not run the unit tier"


def test_the_unit_tier_floor_is_recorded_with_its_cause() -> None:
    """binds M4, E5 and R:SILENT_GAP.

    The unit tier runs on 3.11 and 3.12 but not on 3.8-3.10, and the cause is
    not laziness: 6 test modules `import tomllib`, which is 3.11+ stdlib. A
    limitation recorded without its cause reads as an oversight and gets
    "fixed" by someone who then discovers why.
    """
    doc = DOC.read_text()
    assert RECORDED_GAPS["unit tier floor"] in doc, (
        f"{DOC.name} no longer records WHY the unit tier cannot run below 3.11. "
        f"The cause is tomllib (3.11+ stdlib), used by 6 test modules."
    )
    assert "3.11" in doc


def test_the_operating_system_gap_is_recorded_with_its_cause() -> None:
    """binds M4, E6 and R:SILENT_GAP.

    Human decision 2026-09-16: record the gap, add no OS job. `Backend
    Conformance` already carries a strict xfail scoped to macOS arm64 for an
    unexplained divergence owned by m4, so an OS matrix would go red on
    arrival — which this milestone's own wording warns against.
    """
    doc = DOC.read_text()
    assert RECORDED_GAPS["operating system"] in doc, (
        f"{DOC.name} no longer records that CI runs ubuntu only. The package "
        f"declares no OS classifiers, so it implicitly claims every OS."
    )


def test_the_public_surface_script_fails_when_the_surface_is_broken() -> None:
    """A smoke that cannot fail proves nothing (Q15)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_surface_probe", SURFACE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main() == 0, "the surface script must pass on the interpreter running the suite"

    module.EXPECTED_DETECTION_VARIANTS = 999
    assert module.main() == 1, (
        "the surface script returned 0 with a registry expectation it cannot meet, "
        "so it would report success over a broken surface"
    )
