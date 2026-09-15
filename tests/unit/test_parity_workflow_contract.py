"""The parity workflow: two entry points, one set of assertions.

m3 box 3 asks for the n/s variants on every pull request and all ten on a
schedule, and says the split "is deliberate, not a later quiet narrowing". The
danger in a split like that is not the variant list — it is that the two entry
points drift into asserting different things while wearing one name.

This repo had no `schedule:` or `cron:` anywhere (checked 2026-09-15), and
`ci.yml`'s job ids are frozen by `test_ci_contract.py`, so the schedule lives
in a new workflow file rather than extending an existing one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY = REPO_ROOT / ".github" / "workflows" / "parity.yml"

#: The four the pull-request entry point must run. "n/s" spans BOTH families:
#: read as one, a whole architecture goes unchecked on every pull request.
PR_VARIANTS = ("yolo11n", "yolo11s", "yolo26n", "yolo26s")

#: Every detection variant the registry ships.
ALL_VARIANTS = tuple(
    f"yolo{family}{size}" for family in ("11", "26") for size in ("n", "s", "m", "l", "x")
)

#: The x weights are 114 MB and 118 MB and their exports dominate the run.
PARITY_TIMEOUT_MINUTES_REQUIRED = 60


def _workflow() -> dict[str, Any]:
    if not PARITY.exists():
        pytest.fail(f"{PARITY.name} does not exist — the parity split is not wired")
    return yaml.safe_load(PARITY.read_text())


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML parses the bare key `on:` as the boolean True.
    return workflow.get("on") or workflow.get(True) or {}


def test_the_parity_workflow_declares_a_schedule() -> None:
    """covers: A15,A18,S3 — the first `schedule:` in this repo."""
    schedule = _triggers(_workflow()).get("schedule")
    assert schedule, "the parity workflow declares no schedule, so nothing runs all ten"
    assert any("cron" in entry for entry in schedule), (
        f"a schedule trigger needs a cron expression: {schedule!r}"
    )


def test_the_pull_request_trigger_runs_the_four_ns_variants() -> None:
    """covers: M4,A8,S2."""
    workflow = _workflow()
    assert "pull_request" in _triggers(workflow), (
        "the parity workflow does not run on pull requests, so the n/s half of "
        "the split never happens"
    )
    body = PARITY.read_text()
    pr_job = workflow["jobs"]["parity-pr"]
    run = " ".join(str(step.get("run", "")) for step in pr_job.get("steps", []))
    assert "--parity-scope pr" in run, (
        f"the pull-request job does not select the 'pr' scope: {run!r}"
    )
    for variant in PR_VARIANTS:
        assert variant in body, f"{variant} is named nowhere in the parity workflow"


def test_the_schedule_runs_all_ten_variants() -> None:
    """covers: M4,A14,S3 — a schedule that silently covers nine is worse than four."""
    scheduled = _workflow()["jobs"]["parity-all"]
    run = " ".join(str(step.get("run", "")) for step in scheduled.get("steps", []))
    assert "--parity-scope all" in run, (
        f"the scheduled job does not select the 'all' scope: {run!r}"
    )


def test_both_entry_points_share_one_parity_implementation() -> None:
    """covers: M4,E5 — the split is which variants, never which assertions."""
    workflow = _workflow()
    targets = {}
    for job_id in ("parity-pr", "parity-all"):
        run = " ".join(str(s.get("run", "")) for s in workflow["jobs"][job_id].get("steps", []))
        targets[job_id] = [tok for tok in run.split() if tok.endswith(".py")]
    assert targets["parity-pr"] == targets["parity-all"], (
        f"the two entry points run different test files: {targets}. A split that "
        f"changes what is asserted makes them two checks wearing one name."
    )
    assert targets["parity-pr"], "neither job names a test file to run"


def test_both_parity_jobs_declare_a_timeout() -> None:
    """covers: A9,A15 — without one, a hung export burns the default six hours."""
    workflow = _workflow()
    for job_id in ("parity-pr", "parity-all"):
        steps = workflow["jobs"][job_id].get("steps", [])
        running = [s for s in steps if "pytest" in str(s.get("run", ""))]
        assert running, f"{job_id} runs no pytest step"
        assert running[0].get("timeout-minutes") == PARITY_TIMEOUT_MINUTES_REQUIRED, (
            f"{job_id}'s parity step must declare "
            f"timeout-minutes: {PARITY_TIMEOUT_MINUTES_REQUIRED}"
        )


def test_the_parity_jobs_publish_explicit_names() -> None:
    """covers: A12 — a check nobody can name is a check nobody can require.

    GitHub falls back to the job id when `name:` is absent, and branch
    protection binds the published NAME.
    """
    workflow = _workflow()
    for job_id in ("parity-pr", "parity-all"):
        assert workflow["jobs"][job_id].get("name"), (
            f"{job_id} publishes no explicit name, so it cannot be required by one"
        )


def test_a_missing_weight_fails_under_ci_rather_than_skipping() -> None:
    """covers: A10,E4,R:GREEN_BY_SKIP — Q3: a skipped test is green."""
    workflow = _workflow()
    for job_id in ("parity-pr", "parity-all"):
        steps = workflow["jobs"][job_id].get("steps", [])
        running = [s for s in steps if "pytest" in str(s.get("run", ""))]
        assert str(running[0].get("env", {}).get("CI")) == "true", (
            f"{job_id} does not set CI=true, so an unobtainable weight would skip "
            f"green and the run would report parity it never measured"
        )


def test_the_scheduled_run_fails_visibly() -> None:
    """covers: A16 — a scheduled job that swallows failure is advisory by another name."""
    workflow = _workflow()
    for job_id, job in workflow["jobs"].items():
        assert not job.get("continue-on-error"), f"{job_id} continues on error"
        for step in job.get("steps", []):
            name = step.get("name", step.get("uses", "<unnamed>"))
            assert not step.get("continue-on-error"), f"{job_id}/{name} continues on error"
            assert "|| true" not in str(step.get("run", "")), f"{job_id}/{name} swallows failure"
