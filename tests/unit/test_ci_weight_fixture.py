"""A test can obtain a real, verified weight anywhere — not just on one laptop.

Red-first for ADD task `ci-weight-fixture`.

The integration tier is not merely unrun, it is unrunnable, and it reports
green while being so:

    conftest.py:16                 /Users/tindang/Downloads/Ultralytics YOLO26.pt
    test_engine_integration.py:25  tmp/weights/yolo26n_statedict.pt  (gitignored, absent)
    conftest.py:52                 bus.jpg fetched per session, pytest.skip on failure

A skip is green. 33 tests collect, the weight-dependent ones never execute,
and nothing has ever said so.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import tomllib
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
INTEGRATION = REPO_ROOT / "tests/integration"
CI_YML = REPO_ROOT / ".github/workflows/ci.yml"
FIXTURES = REPO_ROOT / "tests/fixtures"

# An absolute path under someone's home directory, on any of the three platforms.
_MACHINE_PATH = re.compile(r'["\'](?:/Users/|/home/|C:\\\\Users\\\\)[^"\']+["\']')


def _integration_sources() -> dict[str, str]:
    return {p.name: p.read_text() for p in INTEGRATION.rglob("*.py")}


def test_no_fixture_depends_on_a_machine_specific_path() -> None:
    """covers: M1, R:LOCALPATH — a fixture that works only for its author."""
    offenders = {
        name: _MACHINE_PATH.findall(src)
        for name, src in _integration_sources().items()
        if _MACHINE_PATH.search(src)
    }
    assert not offenders, f"machine-specific paths in fixtures: {offenders}"


def test_weight_fixture_uses_the_verified_resolution_path() -> None:
    """covers: M2, R:UNVERIFIEDFIXTURE — reuse production, do not reimplement."""
    src = "\n".join(_integration_sources().values())
    assert "resolve_weights" in src, (
        "no fixture routes through resolve_weights, so nothing digest-verifies the weight"
    )


def test_weight_fixture_targets_the_smallest_pinned_model() -> None:
    """covers: A2 — a 109 MB cold download invites someone to disable the tier."""
    src = "\n".join(_integration_sources().values())
    assert "yolo11n" in src or "ModelSize.NANO" in src, (
        "the fixture should resolve the smallest pinned weight"
    )
    assert "yolo11x" not in src and "yolo26x" not in src


def test_ci_caches_the_weight_store_keyed_on_the_digest() -> None:
    """covers: M3, A8, A9, E2 — a name-keyed cache serves old bytes after a re-pin."""
    ci = yaml.safe_load(CI_YML.read_text())
    cache_steps = [
        s
        for job in ci["jobs"].values()
        for s in job.get("steps", [])
        if "actions/cache" in s.get("uses", "")
    ]
    assert cache_steps, "no actions/cache step caches the weight store"
    key = str(cache_steps[0].get("with", {}).get("key", ""))
    assert "weights" in str(cache_steps[0].get("with", {}).get("path", ""))
    assert "sha256" in key or "digest" in key or "hashFiles" in key, (
        f"cache key is not bound to the pinned digest: {key!r}"
    )


def test_weight_is_not_committed_or_shipped() -> None:
    """covers: M4 — a licence-encumbered 5.4 MB file must not enter the repo."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    assert not [f for f in tracked if f.endswith(".pt")], (
        f"weights tracked in git: {[f for f in tracked if f.endswith('.pt')]}"
    )
    sdist = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    included = sdist["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert not any(str(i).endswith(".pt") for i in included)


def test_missing_input_fails_in_ci_and_skips_locally() -> None:
    """covers: M5, A4, E1, A16 — a green skip is how this tier died."""
    src = "\n".join(_integration_sources().values())
    assert 'environ.get("CI"' in src or "environ.get('CI'" in src, (
        "no fixture distinguishes CI from local; in CI a missing input must FAIL"
    )


def test_sample_image_is_cached_and_fails_loudly() -> None:
    """covers: A14, A15, A16 — cached like the weight, and NOT committed.

    Committing bus.jpg would put a third-party asset with no stated licence into
    an Apache-2.0 repository — the exact failure `licensing-provenance` documents.
    So it is cached rather than vendored, and a failure to obtain it raises in CI
    instead of skipping, because a green skip is how this tier died.
    """
    src = "\n".join(_integration_sources().values())
    assert 'pytest.skip(f"Cannot download' not in src, (
        "the image fixture still skips on network failure"
    )
    assert not list(FIXTURES.glob("*.jpg")), (
        "a third-party image with no stated licence must not be committed"
    )
