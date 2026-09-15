"""The arch-equivalence oracle must run on a pull request, under a requirable name.

`src/yowo/arch/` reimplements two model families from scratch. The only thing
that ever proved that reimplementation correct was a harness in `tmp/` that no
longer exists. A suite that exists but never runs before a merge is prose with
a `.py` extension.

Authored red under ADD task `arch-equivalence-in-ci`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github" / "workflows" / "ci.yml"
DOC = ROOT / "docs" / "ci-required-checks.md"

#: The job id and the published name. The name is what branch protection binds;
#: the id is what `FROZEN_JOB_IDS` registers.
JOB_ID = "arch-equivalence"
JOB_NAME = "Arch Equivalence"

#: The comparison itself takes 8.87 s locally (measured 2026-09-16, all ten
#: variants, weights warm). The budget is dominated by restoring a ~1.42 GB
#: weight cache and by `uv sync`, which pulls torch AND ultralytics. A timeout
#: bounds a hang; it is not a performance assertion. Below this ceiling GitHub
#: cancels the job and the contributor gets a bare cancellation carrying none of
#: the guidance the deviation message holds.
TIMEOUT_MINUTES_REQUIRED = 30


def _ci() -> dict[str, Any]:
    if not CI.exists():
        pytest.fail("ci.yml does not exist — the PR gate is not wired")
    return yaml.safe_load(CI.read_text())


def _job() -> dict[str, Any]:
    jobs = _ci()["jobs"]
    assert JOB_ID in jobs, (
        f"ci.yml has no {JOB_ID!r} job, so the native architecture is compared "
        f"against ultralytics nowhere before a merge. Jobs present: {sorted(jobs)}"
    )
    return jobs[JOB_ID]


def test_ci_registers_the_arch_equivalence_job() -> None:
    """Registered in FROZEN_JOB_IDS in the same commit that adds the job.

    The frozen set is what makes an append deliberate rather than silent drift;
    a job added without registering it is exactly the restructure that set is for.
    """
    from tests.unit.test_ci_contract import FROZEN_JOB_IDS

    assert JOB_ID in FROZEN_JOB_IDS, (
        f"{JOB_ID!r} is not in FROZEN_JOB_IDS. Add the id in the same commit "
        f"that adds the job — that is what makes the append reviewable."
    )
    assert JOB_ID in _ci()["jobs"]


def test_the_arch_equivalence_job_publishes_a_requirable_name() -> None:
    """Branch protection binds the published name, not the job id.

    `Export Parity` shipped and blocked nothing because the name it published
    was in no protection payload. The doc row and the payload are checked
    together by `test_check_name_uniqueness.py`; this binds the name's existence.
    """
    assert _job().get("name") == JOB_NAME, (
        f"the {JOB_ID!r} job must publish `name: {JOB_NAME}` — that string is "
        f"what a required-status-check payload can name. Got {_job().get('name')!r}."
    )
    assert JOB_NAME in DOC.read_text(), (
        f"{JOB_NAME!r} is absent from {DOC.name}. A check nobody classified as "
        f"gating or advisory is a check nobody decided about."
    )


def test_the_arch_equivalence_step_runs_under_ci_true_within_its_budget() -> None:
    """An absent weight or oracle must fail the job, not skip it green (Q3)."""
    job = _job()
    steps = job.get("steps", [])
    target = [
        s for s in steps if "test_arch_equivalence.py" in " ".join(str(s.get("run", "")).split())
    ]
    assert target, (
        f"no step in {JOB_ID!r} runs tests/integration/test_arch_equivalence.py. "
        f"The job name would report success having compared nothing."
    )

    # `map-gate` declares its budget on the step rather than the job; this
    # follows that convention rather than inventing a second one.
    timeout = target[0].get("timeout-minutes", job.get("timeout-minutes"))
    assert isinstance(timeout, int) and timeout >= TIMEOUT_MINUTES_REQUIRED, (
        f"the {JOB_ID!r} oracle step declares timeout-minutes={timeout!r}; it needs at "
        f"least {TIMEOUT_MINUTES_REQUIRED} to restore the weight cache and install torch."
    )

    envs = [job.get("env", {})] + [s.get("env", {}) or {} for s in job.get("steps", [])]
    assert any(str(e.get("CI", "")).lower() in ("1", "true", "yes") for e in envs), (
        f"{JOB_ID!r} must set CI=true so an absent weight or an absent ultralytics "
        f"fails the job. Without it a missing input skips, and a skip is green."
    )
