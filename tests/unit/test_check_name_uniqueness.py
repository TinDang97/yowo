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

PROTECTION = REPO_ROOT / "docs/branch-protection.json"


def required_contexts() -> tuple[str, ...]:
    """The contexts protection actually binds, read from the payload that applies it.

    Not a copy. This was a hardcoded `("Quality Gate",)` while protection had
    required four contexts since 2026-09-10, so the guard that exists to stop
    the branch being silently un-gated was itself checking a stale list.
    """
    import json

    payload = json.loads(PROTECTION.read_text())
    return tuple(payload["required_status_checks"]["contexts"])


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
    for context in required_contexts():
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


def _doc_rows() -> dict[str, bool]:
    """Check-run name -> whether the doc marks it required on `main`."""
    rows: dict[str, bool] = {}
    for line in DOC.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        name = cells[0].strip("`")
        rows[name] = "yes" in cells[3].lower()
    return rows


def test_the_doc_and_the_protection_payload_agree() -> None:
    """A doc that disagrees with the payload documents a gate nobody has.

    Both directions: a context bound but undocumented is an unexplained block,
    and a row marked required that protection does not bind is a gate that
    exists only on paper.
    """
    documented = {name for name, req in _doc_rows().items() if req}
    bound = set(required_contexts())
    assert documented == bound, (
        f"docs/ci-required-checks.md and docs/branch-protection.json disagree.\n"
        f"  marked required in the doc but not bound: {sorted(documented - bound)}\n"
        f"  bound but not marked required in the doc: {sorted(bound - documented)}"
    )


def test_every_ci_job_is_classified_as_gating_or_advisory() -> None:
    """R:ADVISORY_BY_ACCIDENT.

    `Real Backend Smoke`, `Backend Conformance` and `Accuracy Dataset` each ran
    on every pull request for weeks while blocking nothing. Backend Conformance
    caught a real design fault 2656 green unit tests missed, and could not have
    stopped it merging. Nothing reported that, because nothing compared the set
    of jobs against the set of required contexts.

    A new job must therefore land with a verdict: required, or recorded in the
    doc as advisory. Silence is what produced a gate everybody read as one.
    """
    ci_names = set(_effective_names(WORKFLOW_DIR / "ci.yml").values())
    rows = _doc_rows()
    unclassified = sorted(ci_names - set(rows))
    assert not unclassified, (
        f"these ci.yml jobs appear in no row of docs/ci-required-checks.md: "
        f"{unclassified}. A job that is neither required nor recorded as advisory "
        f"is a check contributors read as a gate while it blocks nothing."
    )
