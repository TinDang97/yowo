"""The verifier must not carry its own copy of the thing it verifies.

`scripts/verify_branch_protection.py` is what `docs/ci-required-checks.md`
tells a maintainer to run to confirm the live configuration — "this is the
check bound to the task, not a claim". It carried a hardcoded four-context
tuple with a comment saying it was "kept in the SAME ORDER as
docs/branch-protection.json so a reader can diff the two by eye". A convention
that depends on someone diffing by eye is not a check, and it drifted: the
payload required eight while the verifier asserted four and reported
"requires all 4 checks" as a success.

The same drift had already happened in `test_check_name_uniqueness.py`, whose
`REQUIRED_CONTEXTS` sat at one entry while protection required four.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify_branch_protection.py"
PAYLOAD = REPO_ROOT / "docs" / "branch-protection.json"


@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_branch_protection", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_verifier_reads_the_payload_it_verifies(verifier: ModuleType) -> None:
    """One source of truth. A copy is a second thing to keep in step, and this one wasn't."""
    expected = tuple(json.loads(PAYLOAD.read_text())["required_status_checks"]["contexts"])
    assert verifier.required_contexts() == expected, (
        "the verifier's context list disagrees with docs/branch-protection.json — "
        "it would report a subset as 'all checks' and pass while the rest were dropped"
    )


def test_a_dropped_context_is_reported_as_missing(verifier: ModuleType) -> None:
    """The failure this drift hid: four of eight contexts silently removed.

    With a hardcoded four-entry list, dropping the other four left the verifier
    green — it only ever asked whether its own four were present.
    """
    payload = json.loads(PAYLOAD.read_text())
    kept = payload["required_status_checks"]["contexts"][:4]
    failures = verifier.missing_contexts(kept)
    assert failures, (
        f"dropping half the required contexts (kept {kept}) was not reported as a failure"
    )


def test_a_commit_with_no_ci_checks_says_so_rather_than_blaming_protection(
    verifier: ModuleType,
) -> None:
    """`ci.yml` runs on `pull_request`, so a commit on `main` never carries its checks.

    Run on main — which is what the doc's copy-pasteable command does — the
    verifier reported "GitHub has never reported a check named 'Quality Gate'
    ... protection requiring it would block every merge on a check that never
    reports". That diagnosis is false: the check reports fine, on pull
    requests, which is where it is required. A maintainer following the
    documented procedure was told their branch protection was broken when it
    was not.
    """
    observed, why = verifier.explain_observation([], "deadbeefcafe")
    assert not observed
    assert "--ref" in why, (
        f"the message must name the flag that fixes it, not blame protection: {why}"
    )
    assert "pull request" in why.lower(), (
        f"the message must say why a main commit carries no ci.yml checks: {why}"
    )
