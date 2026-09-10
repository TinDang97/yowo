"""The publish mechanism is proved against TestPyPI before it is trusted with PyPI.

Red-first for ADD task `testpypi-dry-run`.

WHAT THESE PROVE, AND WHAT THEY DO NOT. Every assertion here reads a YAML file and
a markdown file. Together they prove the dry-run path is CONFIGURED correctly and
cannot reach PyPI. NONE of them proves a dry run publishes — that is R:SHAPEONLY,
and it is not a caveat, it is the specific failure this task exists to expose. The
release workflow has been red since 2026-08-26 and `Publish to PyPI` has never run
once, while eleven checks in `test_release_contract.py` stayed green the whole time,
because they assert the shape of a job that cannot start. m1 box 2 clause (ii)
closes on a RUN, not on this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
DRY_RUN_YML = WORKFLOW_DIR / "release-dry-run.yml"
RELEASE_YML = WORKFLOW_DIR / "release.yml"
CI_YML = WORKFLOW_DIR / "ci.yml"
RUNBOOK = REPO_ROOT / "docs/release-setup.md"

TESTPYPI_UPLOAD_URL = "https://test.pypi.org/legacy/"

# major.minor.patch, or a full commit SHA. `@v4` is a floating tag: the same ref
# can point at different code tomorrow (M8).
_EXACT_PIN = re.compile(r"@(v?\d+\.\d+\.\d+|[0-9a-f]{40})$")


@pytest.fixture(scope="module")
def dry_run() -> dict:
    """The dry-run workflow, or an empty dict if it does not exist yet.

    Deliberately does not assert. A fixture that raises turns every dependent
    check into an ERROR, and an error is a wrong-reason red.
    """
    if not DRY_RUN_YML.is_file():
        return {}
    return yaml.safe_load(DRY_RUN_YML.read_text())


def _triggers(workflow: dict) -> dict:
    """The `on:` block.

    PyYAML resolves the bare key `on` to the boolean True (YAML 1.1), so the
    trigger block lands under `True`, not under `"on"`. Reading only `"on"` is
    how a trigger check silently passes on a workflow with no triggers at all.
    """
    for key in (True, "on", "true"):
        if key in workflow:
            return workflow[key] or {}
    return {}


def _jobs(workflow: dict) -> dict:
    return workflow.get("jobs", {}) or {}


def _steps(job: dict) -> list[dict]:
    return job.get("steps", []) or []


def _all_steps(workflow: dict) -> list[dict]:
    return [s for job in _jobs(workflow).values() for s in _steps(job)]


def _uses(workflow: dict) -> list[str]:
    return [s["uses"] for s in _all_steps(workflow) if "uses" in s]


def _job_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    loaded = yaml.safe_load(path.read_text()) or {}
    jobs = loaded.get("jobs", {}) or {}
    return {job.get("name", job_id) for job_id, job in jobs.items()}


def _upload_steps(workflow: dict) -> list[dict]:
    return [s for s in _all_steps(workflow) if "pypi-publish" in s.get("uses", "")]


# --------------------------------------------------------------------------
# M1, M8 — it exists, and something can start it
# --------------------------------------------------------------------------


def test_a_version_tag_push_triggers_the_dry_run(dry_run: dict) -> None:
    """covers: M1, E1, A2 — a `v*` tag starts it."""
    assert dry_run, f"{DRY_RUN_YML.name} does not exist — nothing can be dry-run"
    tags = (_triggers(dry_run).get("push", {}) or {}).get("tags", [])
    assert tags, (
        "no `push: tags:` trigger. This is the exact shape release.yml has: it is "
        "`on: push: branches: [main]`, so no tag can start it, which is why clause "
        "(ii) has never been satisfiable"
    )
    assert "v*" in tags, (
        f'tags {tags} does not include `v*`, and `tag_format = "v{{version}}"` in '
        "[tool.semantic_release] is what this repo actually produces (A2)"
    )


def test_it_can_also_be_run_on_demand(dry_run: dict) -> None:
    """covers: M1, E2, A1 — `workflow_dispatch`, so no throwaway tag is needed."""
    triggers = _triggers(dry_run)
    assert "workflow_dispatch" in triggers, (
        "no `workflow_dispatch`. Configuring a trusted publisher is a trial-and-error "
        "loop against four fields that must all match; requiring a tag per attempt "
        "makes that loop cost a tag each time (A1)"
    )


def test_a_non_version_tag_does_not_trigger_it(dry_run: dict) -> None:
    """covers: E5 — the tag filter is a filter, not a wildcard."""
    tags = (_triggers(dry_run).get("push", {}) or {}).get("tags", [])
    assert tags, "no tag filter to evaluate"
    assert "*" not in tags and "**" not in tags, (
        f"tag filter {tags} matches every tag, so any tag at all publishes to "
        "TestPyPI. E5 asks that a non-version tag does not trigger it"
    )
    assert all(t.startswith("v") for t in tags), (
        f"tag filter {tags} admits a pattern that is not version-shaped"
    )


def test_every_action_is_pinned_to_an_exact_version(dry_run: dict) -> None:
    """covers: M8, E7 — box 2 clause (iii) applies here too."""
    uses = _uses(dry_run)
    assert uses, "no actions to inspect"
    floating = [u for u in uses if not _EXACT_PIN.search(u)]
    assert not floating, (
        f"floating action refs on the release path: {floating}. A `@v4` can point at "
        "different code tomorrow, and every action here runs with this workflow's "
        "OIDC permission"
    )


# --------------------------------------------------------------------------
# M2, R:TOKEN — OIDC, no credential
# --------------------------------------------------------------------------


def test_it_publishes_over_oidc_with_no_credential_input(dry_run: dict) -> None:
    """covers: M2, R:TOKEN — trusted publishing, presenting nothing."""
    uploads = _upload_steps(dry_run)
    assert uploads, "no pypa/gh-action-pypi-publish step — nothing uploads anywhere"
    for step in uploads:
        assert "password" not in (step.get("with", {}) or {}), (
            "a `password:` input means token auth, not OIDC — and a dry run over a "
            "different mechanism than the real publish proves nothing about it (M2)"
        )
    id_token = [
        (job.get("permissions", {}) or {}).get("id-token")
        for job in _jobs(dry_run).values()
        if _upload_steps({"jobs": {"j": job}})
    ]
    assert id_token and all(v == "write" for v in id_token), (
        f"the uploading job does not declare `id-token: write` ({id_token}); OIDC "
        "cannot mint a token without it"
    )


def test_no_testpypi_credential_exists_anywhere_in_the_repository() -> None:
    """covers: R:TOKEN — no credential on this path, in any workflow.

    Written to assert the dry-run file is among what it scanned. Without that,
    this check passes vacuously before the file exists — a green scan of nothing.
    """
    # `*.yaml` too — GitHub accepts both, and a scan that misses half the extensions
    # is a scan whose name over-claims.
    scanned = sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])
    assert DRY_RUN_YML in scanned, (
        f"{DRY_RUN_YML.name} was not among the scanned workflows, so this check "
        "would be proving the absence of a credential in files that are not the "
        "one under test"
    )
    # Two rules, because an enumerated blocklist only catches the spellings someone
    # thought of. `secrets.PYPI_TOKEN` and `secrets.TESTPYPI` both slipped the
    # original list.
    env_carriers = ("TWINE_", "UV_PUBLISH_TOKEN", "HATCH_INDEX_AUTH", "FLIT_PASSWORD")
    offenders: list[str] = []
    for path in scanned:
        text = path.read_text()
        offenders += [f"{path.name}: {c}" for c in env_carriers if c in text]
        offenders += [f"{path.name}: __token__" for _ in range(1) if "__token__" in text]
        # Any secret whose name mentions an index at all.
        offenders += [
            f"{path.name}: secrets.{m}"
            for m in re.findall(r"secrets\.([A-Za-z0-9_]+)", text)
            if "PYPI" in m.upper() or "PUBLISH" in m.upper() or "TWINE" in m.upper()
        ]
    assert not offenders, f"a credential reference on the publish path: {offenders}"
    # Nothing on this path interpolates anything at all today; keep it that way, so
    # there is no expression-injection surface either.
    assert "${{" not in DRY_RUN_YML.read_text(), (
        "the dry run interpolates an expression. It needs none, and every `${{ }}` is "
        "somewhere a secret or untrusted input could later be spliced in (R:TOKEN)"
    )


# --------------------------------------------------------------------------
# M3 — it builds what the release path builds
# --------------------------------------------------------------------------


def test_it_builds_from_the_checkout_with_the_release_paths_build_command(
    dry_run: dict,
) -> None:
    """covers: M3 — same source, same `uv build`."""
    steps = _all_steps(dry_run)
    assert any("actions/checkout" in s.get("uses", "") for s in steps), (
        "nothing checks out the tagged commit, so whatever is uploaded did not come "
        "from the tag (M3)"
    )
    runs = " ".join(s.get("run", "") for s in steps)
    assert "uv build" in runs, (
        "the dry run does not use `uv build`, which is `build_command` in "
        "[tool.semantic_release] — a dry run of a different build proves the wrong "
        "thing (M3)"
    )


# --------------------------------------------------------------------------
# M4, R:REALINDEX — it cannot reach PyPI
# --------------------------------------------------------------------------


def test_every_upload_in_this_workflow_targets_testpypi(dry_run: dict) -> None:
    """covers: M4, R:REALINDEX — no branch of this can publish for real."""
    uploads = _upload_steps(dry_run)
    assert uploads, "no upload step"
    for step in uploads:
        url = (step.get("with", {}) or {}).get("repository-url", "")
        assert url == TESTPYPI_UPLOAD_URL, (
            f"upload targets {url!r}, not {TESTPYPI_UPLOAD_URL!r}. An omitted "
            "`repository-url` DEFAULTS TO PYPI — the dangerous case is the absent "
            "one, not a wrong one (R:REALINDEX)"
        )
    text = DRY_RUN_YML.read_text()
    assert "upload.pypi.org" not in text, "a real-index URL appears in the dry run"

    # Positively `testpypi`, not merely "not `release`". `environment` is half the
    # OIDC claim set the TestPyPI publisher binds to, so deleting it breaks the run
    # — and a `!= "release"` assertion is satisfied by the empty string and by the
    # key being absent altogether. Verified by mutation: before this line, deleting
    # `environment: testpypi` left all 13 checks green.
    for job in _jobs(dry_run).values():
        env = job.get("environment", "")
        name = env.get("name", "") if isinstance(env, dict) else env
        assert name == "testpypi", (
            f"the publishing job runs in environment {name!r}, not 'testpypi'. That "
            "is one of the five fields the TestPyPI trusted publisher matches on, and "
            "'release' is the one the PYPI publisher is bound to (M4)"
        )

    # R:REALINDEX says ANY branch that can upload, not any action-based one. A shell
    # upload is invisible to `_upload_steps`. Verified by mutation: before this line,
    # appending `run: uv publish` left all 13 checks green.
    uploaders = ("twine upload", "uv publish", "hatch publish", "flit publish", "poetry publish")
    shell = [
        f"{s.get('name', '?')}: {tool}"
        for s in _all_steps(dry_run)
        for tool in uploaders
        if tool in s.get("run", "")
    ]
    assert not shell, (
        f"a shell step uploads without going through the pinned publish action: "
        f"{shell}. `repository-url` cannot constrain what it does not run through "
        "(R:REALINDEX)"
    )


# --------------------------------------------------------------------------
# M5, M6 — it stands beside the release path, not in it
# --------------------------------------------------------------------------


def test_it_neither_gates_nor_is_gated_by_the_release_path(dry_run: dict) -> None:
    """covers: M5, A3 — a TestPyPI outage never blocks a real release."""
    assert dry_run, "no workflow"
    # Exhaustive, not additive. `workflow_call` would make this job — which holds
    # `id-token: write` and the `testpypi` environment — invocable from any other
    # workflow, including a `pull_request_target` one: the textbook fork-PR
    # escalation shape. Listing forbidden triggers one at a time never catches the
    # one nobody thought of.
    keys = {k for k in _triggers(dry_run)}
    assert keys == {"push", "workflow_dispatch"}, (
        f"triggers are {sorted(keys)}, not exactly ['push', 'workflow_dispatch'] (M5, A3)"
    )
    release_text = RELEASE_YML.read_text()
    assert "release-dry-run" not in release_text and "dry_run" not in release_text, (
        "release.yml references the dry run, so a TestPyPI failure can stand between "
        "a merge and a release. This proves a mechanism; it is not a quality gate (M5)"
    )


def test_its_job_names_collide_with_nothing_ci_or_release_publishes(
    dry_run: dict,
) -> None:
    """covers: M6, E6 — a required check can never be satisfied by this workflow."""
    mine = {job.get("name", jid) for jid, job in _jobs(dry_run).items()}
    assert mine, "no jobs"
    clash = mine & (_job_names(CI_YML) | _job_names(RELEASE_YML))
    assert not clash, (
        f"check-run name(s) {clash} are published by more than one workflow. Branch "
        "protection stores the NAME, so a required check meant to prove a pull "
        "request passed could be satisfied by this dry run instead (M6)"
    )


# --------------------------------------------------------------------------
# A4, A5, A6 — the failure mode, the repeat, the reader
# --------------------------------------------------------------------------


def test_a_publish_failure_names_the_missing_testpypi_publisher(dry_run: dict) -> None:
    """covers: M7, E4, A4 — an unconfigured publisher fails, saying so."""
    steps = _all_steps(dry_run)
    explainers = [
        s
        for s in steps
        if "failure()" in str(s.get("if", "")) and "release-setup" in s.get("run", "")
    ]
    assert explainers, (
        "no `if: failure()` step naming docs/release-setup.md. An OIDC refusal from "
        "an unconfigured publisher is unreadable, and the maintainer is looking at a "
        "fifteen-minute feedback loop against four fields that must all match (A6)"
    )
    for step in steps:
        assert "continue-on-error" not in step, (
            "a `continue-on-error` step makes a failed publish read green — which is "
            "precisely the failure this task exists to expose (A4)"
        )
    for job in _jobs(dry_run).values():
        assert "continue-on-error" not in job, "a job swallows its own failure (A4)"


def test_a_repeat_dry_run_of_the_same_version_is_a_no_op(dry_run: dict) -> None:
    """covers: A5, E3 — a re-run is green, matching the real publish step."""
    uploads = _upload_steps(dry_run)
    assert uploads, "no upload step"
    for step in uploads:
        assert (step.get("with", {}) or {}).get("skip-existing") is True, (
            "no `skip-existing: true`. A red job on a successful release is how the "
            "last five release failures went unread here (A5)"
        )


def test_the_runbook_names_the_four_testpypi_publisher_fields(dry_run: dict) -> None:
    """covers: A6 — the reader can tell WHICH field is wrong."""
    assert RUNBOOK.is_file(), f"{RUNBOOK} does not exist"
    text = RUNBOOK.read_text()
    assert "test.pypi.org" in text, (
        "the runbook says nothing about TestPyPI, so the dry run's own failure "
        "message points at a document that cannot answer it (A6)"
    )
    # No `if e` filter. Deriving the requirement from the workflow means a deleted
    # environment deletes its own requirement — the check weakens in exactly the
    # case it exists to catch. The five fields are stated here literally.
    required = {DRY_RUN_YML.name, "TinDang97", "yowo", "testpypi", "test.pypi.org"}
    missing = sorted(f for f in required if f not in text)
    assert not missing, (
        f"the runbook does not name {missing}. All four TestPyPI publisher fields "
        "(owner, repository, workflow filename, environment) must match exactly, and "
        "the OIDC refusal does not say which one is wrong (A6)"
    )


def test_the_publishing_job_grants_only_what_its_steps_need(dry_run: dict) -> None:
    """covers: M2, M4 — least privilege, computed from the steps, not copied.

    A job-level `permissions:` block REPLACES the workflow default, it does not
    merge: "if you specify the access for any of these permissions, all of those
    that are not specified are set to `none`". So declaring `id-token: write`
    alone leaves `contents: none` — while the first step is `actions/checkout`.

    `release.yml`'s publish job declares `id-token: write` alone and is CORRECT to,
    because it has no checkout; it downloads an artifact. Copying that permission
    block without copying the absence of a checkout is the defect this catches.
    """
    for job_id, job in _jobs(dry_run).items():
        perms = job.get("permissions", {}) or {}
        assert isinstance(perms, dict), f"job {job_id} uses a blanket permissions scope"
        needs_checkout = any("actions/checkout" in s.get("uses", "") for s in _steps(job))
        if needs_checkout:
            assert perms.get("contents") == "read", (
                f"job {job_id} checks out but declares contents={perms.get('contents')!r}. "
                "A job-level permissions block replaces rather than merges, so the "
                "workflow-level `contents: read` does NOT apply here"
            )
        for scope, level in perms.items():
            if scope == "id-token":
                continue
            assert level == "read", (
                f"job {job_id} grants {scope}: {level}. Nothing on this path writes "
                "anything but an OIDC token — a publish job that can write to the "
                "repository is a publish job that can rewrite what it publishes"
            )


def test_nothing_records_clause_ii_as_satisfied() -> None:
    """covers: R:SHAPEONLY — the box closes on a run, and nothing here says otherwise.

    This is the one rule in the node that is not a property of the workflow. It is a
    property of what the repository CLAIMS about the workflow, and that is exactly as
    assertable — and exactly as worth asserting, because the failure it guards is the
    one already sitting in this repo: eleven green checks over a release job that has
    never started once.
    """
    milestone = (REPO_ROOT / ".add/milestones/m1-trust-the-ship.md").read_text()
    box_2 = [
        line
        for line in milestone.splitlines()
        if line.startswith("- [") and "TestPyPI dry-run" in line
    ]
    assert len(box_2) == 1, f"m1 box 2 is not where it was ({len(box_2)} matches)"
    assert box_2[0].startswith("- [ ]"), (
        "m1 box 2 is ticked. Clause (ii) asks that a dry run PUBLISHES; this node "
        "ships only the mechanism, and a shape check cannot close it. Tick it when a "
        "version is visible at test.pypi.org/project/yowo/ — not before, and not "
        "because `release-dry-run.yml` exists (R:SHAPEONLY)"
    )

    runbook = RUNBOOK.read_text()
    assert "closes on a run" in runbook, (
        "the runbook no longer states that the box closes on a run rather than on the "
        "workflow file existing — which is the only thing standing between this and "
        "the failure at the top of that same page"
    )

    protection = (REPO_ROOT / "docs/branch-protection.json").read_text()
    assert "Publish to TestPyPI" not in protection, (
        "the dry run is a required status check on `main`. It is a canary, not a gate: "
        "making it required lets a TestPyPI outage block every merge, and dresses a "
        "shape check up as a guarantee (M5, R:SHAPEONLY)"
    )
