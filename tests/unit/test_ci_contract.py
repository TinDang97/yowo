"""The CI workflow contract.

`ci.yml` is the pre-merge gate, and its job ids are a frozen contract that six later
tasks append to. These checks bind that contract so an append is a deliberate edit
here rather than a silent restructure.

Authored red under ADD task `pr-ci-gate` — see `.add/tasks/pr-ci-gate.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
RELEASE = WORKFLOWS / "release.yml"

# The frozen job-id set. A later task ADDS an id here in the same commit that adds
# the job; it never reuses or renames one (task pr-ci-gate, A11).
# `sdist` appended by task sdist-manifest — registered here in the same commit,
# which is what makes an append deliberate rather than silent drift.
FROZEN_JOB_IDS = {"quality", "sdist", "reproducible", "weights", "backend-smoke"}

# The four quality commands both gates must run identically (task pr-ci-gate, M2).
_QUALITY_MARKERS = ("ruff check", "ruff format", "pyright", "pytest tests/unit")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.fail(f"{path.name} does not exist — the PR gate is not wired")
    return yaml.safe_load(path.read_text())


def _steps(workflow: dict[str, Any], job_id: str) -> list[dict[str, Any]]:
    return workflow["jobs"][job_id].get("steps", [])


def _quality_commands(steps: list[dict[str, Any]]) -> dict[str, str]:
    """Map each quality marker to the normalised `run:` command that carries it."""
    found: dict[str, str] = {}
    for step in steps:
        run = " ".join(str(step.get("run", "")).split())
        for marker in _QUALITY_MARKERS:
            if marker in run:
                found[marker] = run
    return found


def test_ci_workflow_triggers_on_pull_request() -> None:
    """covers: G1 — the trigger that makes the gate run before a merge.

    `yaml.safe_load` resolves the bare key `on` to the boolean True, so a typo'd
    `on: pull-request` still parses; this asserts the real trigger, not just the file.
    """
    ci = _load(CI)
    triggers = ci.get("on", ci.get(True))
    assert isinstance(triggers, dict), f"ci.yml `on:` must be a mapping, got {triggers!r}"
    assert "pull_request" in triggers, f"ci.yml must trigger on pull_request, got {list(triggers)}"
    assert "main" in (triggers["pull_request"] or {}).get("branches", [])


def test_ci_job_ids_are_the_frozen_set() -> None:
    """covers: G2 — the job-id namespace six later tasks append beside."""
    ci = _load(CI)
    assert set(ci["jobs"]) == FROZEN_JOB_IDS, (
        "ci.yml job ids drifted from the frozen contract. Adding a job is fine — "
        "add its id to FROZEN_JOB_IDS in the same commit. Renaming or reusing one is not."
    )


def test_ci_runs_the_same_four_quality_commands_as_release() -> None:
    """covers: G2 — binds R:DIVERGENCE.

    A PR must not be able to pass a weaker gate than the one `main` runs.
    """
    ci_cmds = _quality_commands(_steps(_load(CI), "quality"))
    release_cmds = _quality_commands(_steps(_load(RELEASE), "quality"))

    missing = [m for m in _QUALITY_MARKERS if m not in ci_cmds]
    assert not missing, f"ci.yml is missing quality steps present in release.yml: {missing}"
    for marker in _QUALITY_MARKERS:
        assert ci_cmds[marker] == release_cmds[marker], (
            f"{marker!r} differs between the two gates:\n"
            f"  ci.yml:      {ci_cmds[marker]}\n"
            f"  release.yml: {release_cmds[marker]}"
        )


def test_ci_has_no_silently_passing_step() -> None:
    """covers: G2 — binds R:SILENT_PASS."""
    ci = _load(CI)
    for job_id, job in ci["jobs"].items():
        assert not job.get("continue-on-error"), f"job {job_id} continues on error"
        for step in job.get("steps", []):
            name = step.get("name", step.get("uses", "<unnamed>"))
            assert not step.get("continue-on-error"), f"{job_id}/{name} continues on error"
            assert "|| true" not in str(step.get("run", "")), f"{job_id}/{name} swallows failure"


def test_required_check_name_matches_published_job_name() -> None:
    """covers: G3 — binds A6's probe.

    Branch protection is configured against the check NAME GitHub publishes. If the
    doc records a name the workflow never emits, protection blocks on a check that
    never reports and the branch is unguarded while appearing guarded.
    """
    doc = Path(__file__).resolve().parents[2] / "docs" / "ci-required-checks.md"
    if not doc.exists():
        pytest.fail("docs/ci-required-checks.md does not exist — the human step is unrecorded")
    published = _load(CI)["jobs"]["quality"].get("name", "quality")
    assert published in doc.read_text(), (
        f"the required-checks doc does not name the check the workflow publishes ({published!r})"
    )
