"""The release path publishes to PyPI without holding a credential that can reach it.

Red-first for ADD task `pypi-trusted-publish`.

These are assertions about a YAML file. They prove the release path is CONFIGURED
correctly; they cannot prove it WORKS. Only an actual release can do that, and the
first one is the test. A green run here means "the contract we wrote is present".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github/workflows"
RELEASE_YML = WORKFLOW_DIR / "release.yml"

# An exact pin is major.minor.patch, or a full commit SHA. `@v4` is a floating tag:
# the same ref can point at different code tomorrow (A15).
_EXACT_PIN = re.compile(r"@(v?\d+\.\d+\.\d+|[0-9a-f]{40})$")


@pytest.fixture(scope="module")
def release() -> dict:
    return yaml.safe_load(RELEASE_YML.read_text())


@pytest.fixture(scope="module")
def publish_job(release: dict) -> dict:
    """The publish job, or an empty dict if it does not exist yet.

    Deliberately does not assert: a fixture that raises turns every dependent
    check into an ERROR, and an error is a wrong-reason red.
    """
    return release["jobs"].get("publish", {})


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def _uses(job: dict) -> list[str]:
    return [s["uses"] for s in _steps(job) if "uses" in s]


def test_publish_uses_oidc_with_no_password_input(publish_job: dict) -> None:
    """covers: M2 — trusted publishing presents no credential."""
    upload = [s for s in _steps(publish_job) if "pypi-publish" in s.get("uses", "")]
    assert upload, "no pypa/gh-action-pypi-publish step — nothing uploads to PyPI"
    assert "password" not in upload[0].get("with", {}), (
        "a `password:` input means this is token auth, not OIDC"
    )


def test_no_pypi_credential_in_any_workflow() -> None:
    """covers: R:SECRET — no PyPI credential anywhere in the workflows.

    Narrowed deliberately: a hermetic test cannot enumerate live repository
    secrets. That half is a recorded human attestation (`gh secret list`, empty).
    """
    forbidden = re.compile(r"PYPI_\w*(TOKEN|PASSWORD)|__token__|TWINE_PASSWORD", re.I)
    for wf in sorted(WORKFLOW_DIR.glob("*.yml")):
        hits = forbidden.findall(wf.read_text())
        assert not hits, f"{wf.name} references a PyPI credential: {hits}"


def test_publish_runs_only_after_quality_and_sdist(publish_job: dict) -> None:
    """covers: R:UNGATED — both gates must pass on this commit first."""
    assert publish_job, "no publish job exists to gate"
    needs = publish_job.get("needs", [])
    needs = [needs] if isinstance(needs, str) else needs
    assert "quality" in needs and "sdist" in needs, f"publish must need both gate jobs, got {needs}"


def test_release_push_uses_app_token_not_github_token(release: dict) -> None:
    """covers: M3, A8 — checkout and GH_TOKEN both read the minted App token."""
    job = release["jobs"]["release"]
    assert "secrets.GITHUB_TOKEN" not in yaml.dump(job), (
        "GITHUB_TOKEN cannot push to a protected branch — that is today's GH006 failure"
    )
    assert "steps.app-token.outputs.token" in yaml.dump(job), (
        "checkout and GH_TOKEN must read the minted App token"
    )


def test_app_token_is_minted_before_checkout(release: dict) -> None:
    """covers: A9 — App tokens expire in an hour; mint late, not early."""
    steps = _steps(release["jobs"]["release"])
    ids = [s.get("id") or s.get("uses", "") for s in steps]
    mint = [i for i, s in enumerate(steps) if s.get("id") == "app-token"]
    checkout = [i for i, s in enumerate(steps) if "checkout" in s.get("uses", "")]
    assert mint, f"no app-token step in the release job, got {ids}"
    assert mint[0] < checkout[0], f"App token must be minted before checkout, got {ids}"


def test_no_fallback_to_github_token_on_mint_failure(release: dict) -> None:
    """covers: E2 — a silent fallback re-creates GH006 while looking like a new bug."""
    steps = _steps(release["jobs"]["release"])
    mint = [s for s in steps if s.get("id") == "app-token"]
    assert mint, "no app-token step exists, so there is nothing to fall back from"
    assert "continue-on-error" not in mint[0], "minting must fail loudly, not continue"
    job = yaml.dump(release["jobs"]["release"])
    assert "||" not in job, "no fallback expression may substitute another token"


def test_every_action_and_tool_in_release_is_pinned(release: dict) -> None:
    """covers: M4, A14, A16 — one unpinned action is the whole supply chain."""
    floating = [
        u for job in release["jobs"].values() for u in _uses(job) if not _EXACT_PIN.search(u)
    ]
    assert not floating, f"floating action refs in release.yml: {floating}"
    text = RELEASE_YML.read_text()
    assert re.search(r"python-semantic-release==\d+\.\d+\.\d+", text), (
        "python-semantic-release must be installed at an exact version"
    )


def test_duplicate_upload_is_skipped_not_failed(publish_job: dict) -> None:
    """covers: E1, A4 — a red job on a successful release trains everyone to ignore it."""
    upload = [s for s in _steps(publish_job) if "pypi-publish" in s.get("uses", "")]
    assert upload, "no upload step to configure"
    assert upload[0].get("with", {}).get("skip-existing") is True


def test_pypi_upload_does_not_rebuild_the_artifact(publish_job: dict) -> None:
    """covers: M1 — what reaches PyPI is what the gates passed."""
    assert publish_job, "no publish job exists to check"
    runs = " ".join(s.get("run", "") for s in _steps(publish_job))
    assert "uv build" not in runs and "python -m build" not in runs, (
        "the publish job must upload the built artifact, never rebuild it"
    )
