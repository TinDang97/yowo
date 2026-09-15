"""The jobs that revive the unrun modules, and what each of them needs.

Three integration modules were named by no workflow (measured 2026-09-16):

    test_cli_e2e.py                    21 tests   -> new job `CLI End-to-End`
    test_chroma_gallery_persistence.py  8 tests   -> new job `Persistent Gallery`
    test_onnx_provider_honoured.py      2 tests   -> joins `backend-smoke`

`test_onnx_provider_honoured.py` joins `backend-smoke` because they assert the
same thing — that a real backend honours what the caller asked for — and that
job already restores the weight store. The other two take names of their own:
21 tests failing under somebody else's label is a diagnosis nobody can read.

A job that runs a suite without the fixtures the suite needs downloads them
cold every run, or fails. `test_cli_e2e.py` needs `sample_image_path` and
`sample_image_dir`; `test_onnx_provider_honoured.py` needs `verified_weight`.

Authored red under ADD task `integration-tier-revival`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOC = REPO_ROOT / "docs" / "ci-required-checks.md"
PROTECTION = REPO_ROOT / "docs" / "branch-protection.json"

#: job id -> (published name, the module it revives)
REVIVED = {
    "cli-e2e": ("CLI End-to-End", "tests/integration/test_cli_e2e.py"),
    "persistent-gallery": (
        "Persistent Gallery",
        "tests/integration/test_chroma_gallery_persistence.py",
    ),
}

#: The suite that joins an existing job rather than taking a name of its own.
PROVIDER_SUITE = "tests/integration/test_onnx_provider_honoured.py"
PROVIDER_HOST_JOB = "backend-smoke"

#: A bare cancellation carries none of the guidance a real failure holds.
MIN_TIMEOUT_MINUTES = 30


def _ci() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text())


def _job(job_id: str) -> dict[str, Any]:
    jobs = _ci()["jobs"]
    assert job_id in jobs, f"ci.yml has no {job_id!r} job. Jobs present: {sorted(jobs)}"
    return jobs[job_id]


def _steps_running(job: dict[str, Any], module: str) -> list[dict[str, Any]]:
    return [
        s for s in job.get("steps", []) or [] if module in " ".join(str(s.get("run", "")).split())
    ]


def _cached_paths(job: dict[str, Any]) -> str:
    return "\n".join(str((s.get("with") or {}).get("path", "")) for s in job.get("steps", []) or [])


def _assert_registered_and_published(job_id: str) -> None:
    """Three places, or it gates nothing.

    `Export Parity` shipped and blocked nothing because it appeared in one of
    them. FROZEN_JOB_IDS makes the append reviewable, the doc classifies it as
    gating or advisory, and the payload is what branch protection actually
    stores.
    """
    from tests.unit.test_ci_contract import FROZEN_JOB_IDS

    name, module = REVIVED[job_id]
    assert job_id in FROZEN_JOB_IDS, f"{job_id!r} is not registered in FROZEN_JOB_IDS"
    job = _job(job_id)
    assert job.get("name") == name, (
        f"{job_id!r} must publish `name: {name}`, got {job.get('name')!r}"
    )
    assert name in DOC.read_text(), f"{name!r} is absent from {DOC.name}"
    assert f'"{name}"' in PROTECTION.read_text(), f"{name!r} is absent from {PROTECTION.name}"
    assert _steps_running(job, module), f"no step in {job_id!r} runs {module}"


def test_cli_e2e_is_registered_and_published() -> None:
    """21 tests get a name of their own — failing under another label reads as that job."""
    _assert_registered_and_published("cli-e2e")


def test_persistent_gallery_is_registered_and_published() -> None:
    """8 tests over a shipped public class that CI has never executed."""
    _assert_registered_and_published("persistent-gallery")


def test_cli_e2e_restores_the_fixtures_its_tests_require() -> None:
    """It needs the image store, not only the weight store.

    `sample_image_path` and `sample_image_dir` both resolve through
    `~/.cache/yowo/test-assets`. A job that caches only weights re-downloads
    the image on every run, and fails outright when the fetch is unavailable.
    """
    cached = _cached_paths(_job("cli-e2e"))
    assert "yowo/test-assets" in cached, (
        "cli-e2e must restore ~/.cache/yowo/test-assets — test_cli_e2e.py uses "
        "sample_image_path and sample_image_dir"
    )
    assert "yowo/weights" in cached, "cli-e2e runs the CLI, which resolves a weight"


def test_the_provider_suite_runs_in_a_job_with_the_weight_store() -> None:
    """It joins `backend-smoke` by subject, and that job already caches weights."""
    job = _job(PROVIDER_HOST_JOB)
    assert _steps_running(job, PROVIDER_SUITE), (
        f"{PROVIDER_HOST_JOB!r} does not run {PROVIDER_SUITE}. It was built in PR #50 "
        f"as a gate and then run by nothing."
    )
    assert "yowo/weights" in _cached_paths(job), (
        f"{PROVIDER_HOST_JOB!r} must restore the weight store — the suite uses verified_weight"
    )


def test_every_revived_job_sets_ci_true() -> None:
    """binds R:GREEN_BY_SKIP — a missing input must fail, not skip.

    `tests/integration/conftest.py` fails in CI and skips locally, and it reads
    that distinction from the CI environment variable alone.
    """
    for job_id, (_, module) in sorted(REVIVED.items()):
        job = _job(job_id)
        for step in _steps_running(job, module):
            env = {**(job.get("env") or {}), **(step.get("env") or {})}
            assert str(env.get("CI", "")).lower() in ("1", "true", "yes"), (
                f"{job_id!r} runs {module} without CI=true; an absent fixture would skip, "
                f"and a skip is green"
            )


def test_every_revived_job_declares_a_budget() -> None:
    """A bare GitHub cancellation carries none of the guidance a failure holds."""
    for job_id, (_, module) in sorted(REVIVED.items()):
        job = _job(job_id)
        for step in _steps_running(job, module):
            timeout = step.get("timeout-minutes", job.get("timeout-minutes"))
            assert isinstance(timeout, int) and timeout >= MIN_TIMEOUT_MINUTES, (
                f"{job_id!r} declares timeout-minutes={timeout!r}; it needs at least "
                f"{MIN_TIMEOUT_MINUTES} to restore caches and install dependencies"
            )
