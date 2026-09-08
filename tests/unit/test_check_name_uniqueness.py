"""A required status check names exactly one workflow's job.

Red-first for ADD task `check-name-collision`.

`ci.yml` and `release.yml` both publish check-run names `Quality Gate` and
`Source Distribution`. A required status check is configured by NAME, so
GitHub cannot tell which workflow satisfied it — a gate intended to prove a
PR passed can be satisfied by a push-to-main run instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
DOC = REPO_ROOT / "docs/ci-required-checks.md"

# Configured on `main` by task pr-ci-gate. Changing these un-gates the branch.
REQUIRED_CONTEXTS = ("Quality Gate",)


def _effective_names(path: Path) -> dict[str, str]:
    """Check-run name per job. GitHub falls back to the job id when `name:` is absent (E1)."""
    wf = yaml.safe_load(path.read_text())
    return {jid: job.get("name", jid) for jid, job in wf["jobs"].items()}


def test_no_check_run_name_is_published_by_two_workflows() -> None:
    """covers: G1, R:AMBIGUOUS, A4, E1 — the collision itself."""
    owners: dict[str, list[str]] = {}
    for wf in sorted(WORKFLOW_DIR.glob("*.yml")):
        for name in _effective_names(wf).values():
            owners.setdefault(name, []).append(wf.name)
    collisions = {n: w for n, w in owners.items() if len(w) > 1}
    assert not collisions, f"check-run names published by more than one workflow: {collisions}"


def test_ci_keeps_the_names_branch_protection_is_configured_with() -> None:
    """covers: G2, A2 — renaming the protected context silently un-gates main."""
    names = set(_effective_names(WORKFLOW_DIR / "ci.yml").values())
    assert "Quality Gate" in names
    assert "Source Distribution" in names


def test_required_contexts_resolve_to_a_ci_job() -> None:
    """covers: G2, A16 — a context nothing publishes never blocks anything."""
    ci_names = set(_effective_names(WORKFLOW_DIR / "ci.yml").values())
    other = {
        n
        for wf in WORKFLOW_DIR.glob("*.yml")
        if wf.name != "ci.yml"
        for n in _effective_names(wf).values()
    }
    for context in REQUIRED_CONTEXTS:
        assert context in ci_names, f"required context {context!r} names no ci.yml job"
        assert context not in other, (
            f"required context {context!r} is also published by another workflow"
        )


def test_docs_name_the_owning_workflow() -> None:
    """covers: G3, A18 — a doc listing an ambiguous name is how this happened."""
    text = DOC.read_text()
    assert "ci.yml" in text and "release.yml" in text, (
        "the doc must say which workflow owns which check-run name"
    )
    assert "Source Distribution (release)" in text or "(release)" in text, (
        "the doc must record the disambiguated release-path names"
    )
