"""Two builds of the same commit produce byte-identical artifacts.

Red-first for ADD task `reproducible-sdist`, which owns the half of m1 box 1
that `sdist-manifest` did not build.

Honest reading of this suite: a probe during Direction showed the build is
ALREADY deterministic under a pinned SOURCE_DATE_EPOCH — hatchling honours it
and no backend change is needed. So the two determinism checks are guards
against regression, not descriptions of missing behaviour. What is genuinely
absent is the script, the CI job and the recorded digest.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import verify_reproducible_build as vrb

REPO_ROOT = Path(__file__).parent.parent.parent
NODE = REPO_ROOT / ".add/tasks/reproducible-sdist.md"


def _build(out_dir: Path) -> None:
    env = {**dict(__import__("os").environ), "SOURCE_DATE_EPOCH": vrb.SOURCE_DATE_EPOCH}
    proc = subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(f"uv build failed:\n{proc.stderr}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("repro")
    a, b = root / "a", root / "b"
    _build(a)
    _build(b)
    return a, b


def test_two_builds_produce_identical_sdist(two_builds: tuple[Path, Path]) -> None:
    """covers: M1, A8 — a guard: already true, must stay true."""
    a, b = two_builds
    assert _sha(next(a.glob("*.tar.gz"))) == _sha(next(b.glob("*.tar.gz")))


def test_two_builds_produce_identical_wheel(two_builds: tuple[Path, Path]) -> None:
    """covers: M1, A8 — a guard: comparing only the sdist would let the wheel drift."""
    a, b = two_builds
    assert _sha(next(a.glob("*.whl"))) == _sha(next(b.glob("*.whl")))


def test_script_sets_the_epoch_itself() -> None:
    """covers: A4 — a developer's unset environment must not change the answer."""
    assert vrb.SOURCE_DATE_EPOCH, "the script must pin the epoch, not trust the environment"
    source = (REPO_ROOT / "scripts/verify_reproducible_build.py").read_text()
    assert "SOURCE_DATE_EPOCH" in source and "env" in source, (
        "the script must pass the pinned epoch into the build environment"
    )


def test_missing_artifact_fails_not_passes(tmp_path: Path) -> None:
    """covers: E1, A10 — absence must never read as 'no difference found'."""
    with pytest.raises(vrb.ArtifactError):
        vrb.build_twice(REPO_ROOT, tmp_path / "nonexistent-and-unwritable\x00")


def test_comparison_covers_every_member() -> None:
    """covers: R:DRIFT, E2 — narrowing the comparison to make it pass is the failure."""
    source = (REPO_ROOT / "scripts/verify_reproducible_build.py").read_text()
    for banned in ("EXCLUDE", "SKIP_MEMBERS", "IGNORE_PATHS"):
        assert banned not in source, f"{banned} would narrow the comparison (R:DRIFT)"
    assert vrb.differing_members.__doc__, "differing_members must report what differs (A11)"
    a = next((REPO_ROOT / "dist").glob("*.tar.gz"), None) if (REPO_ROOT / "dist").exists() else None
    if a is not None:
        assert vrb.differing_members(a, a) == []


def test_ci_appends_a_reproducibility_job_and_registers_its_id() -> None:
    """covers: M3 — under the contract pr-ci-gate froze."""
    from tests.unit.test_ci_contract import FROZEN_JOB_IDS

    ci = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert "reproducible" in ci["jobs"], "no reproducibility job in ci.yml"
    assert "reproducible" in FROZEN_JOB_IDS, (
        "the appended job id must be registered in FROZEN_JOB_IDS in the same commit"
    )


def test_recorded_digest_names_its_toolchain() -> None:
    """covers: M2, A14 — a digest without its toolchain is reproducible by nobody."""
    text = NODE.read_text()
    assert "## EVIDENCE" in text
    evidence = text.split("## EVIDENCE", 1)[1]
    assert "sha256" in evidence.lower(), "no digest recorded in EVIDENCE"
    for token in ("hatchling", "python", "commit"):
        assert token in evidence.lower(), f"the record must name the {token} that produced it"
