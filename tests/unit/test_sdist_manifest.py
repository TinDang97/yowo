"""The published distributions contain what we chose to publish.

Red-first for ADD task `sdist-manifest`. These fail against the pre-change
pyproject.toml because it configures only the wheel target: the sdist is a
whole-repo sweep, which is how a 5.6 MB AGPL weight reached PyPI twice.

The artifact assertions build once per session; the config and CI assertions are
hermetic and cheap.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11; the sdist ships tests (A6)
    import tomli as tomllib  # type: ignore[no-redef]
import zipfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import verify_sdist_contents as vsc

REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The real sdist and wheel, built from the working tree."""
    out_dir = tmp_path_factory.mktemp("dist")
    return vsc.build(REPO_ROOT, out_dir)


@pytest.fixture(scope="session")
def sdist_entries(built: tuple[Path, Path]) -> set[str]:
    return vsc.top_level_entries(vsc.sdist_members(built[0]))


def test_sdist_contains_exactly_the_allowlisted_top_level_entries(
    sdist_entries: set[str],
) -> None:
    """covers: M1, A2 — the allowlist holds in both directions."""
    assert sdist_entries - vsc.ALLOWED_ENTRIES == set(), (
        "the sdist carries entries outside the allowlist"
    )
    assert vsc.REQUIRED_ENTRIES - sdist_entries == set(), "the sdist is missing a required entry"


def test_sdist_and_wheel_ship_the_same_package_payload(
    built: tuple[Path, Path],
) -> None:
    """covers: M2 — a wheel-only regression must not pass."""
    sdist, wheel = built
    from_sdist = vsc._package_payload_from_sdist(vsc.sdist_members(sdist))
    from_wheel = vsc._package_payload_from_wheel(vsc.wheel_members(wheel))
    assert from_sdist == from_wheel


def test_untracked_unignored_file_does_not_enter_the_sdist(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """covers: M3, E1 — the live defect: `?? yolo11n.pt`, neither tracked nor ignored."""
    planted = REPO_ROOT / "test_sdist_manifest_planted.pt"
    planted.write_bytes(b"\x00" * 64)
    try:
        assert (
            subprocess.run(
                ["git", "check-ignore", "-q", planted.name],
                cwd=REPO_ROOT,
            ).returncode
            != 0
        ), "fixture invalid: the planted file is gitignored"
        out_dir = tmp_path_factory.mktemp("dist-planted")
        sdist, _ = vsc.build(REPO_ROOT, out_dir)
        assert planted.name not in vsc.sdist_members(sdist)
    finally:
        planted.unlink(missing_ok=True)


def test_no_binary_artifact_in_either_distribution(built: tuple[Path, Path]) -> None:
    """covers: R:BINARY, A14 — both artifacts, package source included."""
    sdist, wheel = built
    offenders = [
        m
        for members in (vsc.sdist_members(sdist), vsc.wheel_members(wheel))
        for m in members
        if m.endswith(vsc.FORBIDDEN_SUFFIXES)
    ]
    assert offenders == []


def test_pyproject_declares_an_explicit_sdist_include_list() -> None:
    """covers: R:SWEEP — only `only-include` restricts; `include` does not.

    Probe evidence: with `include = [...]` the sdist still carried .add/, bench/,
    docs/ and .gitignore. `only-include` is the key that actually narrows it.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    sdist_target = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert sdist_target.get("only-include"), (
        "hatchling's `include` ADDS to a whole-repo sweep; only `only-include` "
        "restricts it. An empty or absent only-include re-widens the artifact."
    )


def _synthetic_sdist(path: Path, members: list[str]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for member in members:
            info = tarfile.TarInfo(f"yowo-0.0.0/{member}")
            info.size = 0
            tar.addfile(info)
    return path


def _synthetic_wheel(path: Path, members: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for member in members:
            zf.writestr(member, b"")
    return path


def test_verifier_rejects_a_planted_forbidden_file(tmp_path: Path) -> None:
    """covers: E2 — a checker never shown to reject anything asserts nothing."""
    sdist = _synthetic_sdist(
        tmp_path / "s.tar.gz",
        [*sorted(vsc.REQUIRED_ENTRIES), "src/yowo/__init__.py", "weights/yolo11n.pt"],
    )
    wheel = _synthetic_wheel(tmp_path / "w.whl", ["yowo/__init__.py"])
    found = vsc.violations(sdist, wheel)
    assert any("yolo11n.pt" in line and "forbidden binary" in line for line in found)


def test_verifier_fails_when_no_artifact_was_produced(tmp_path: Path) -> None:
    """covers: A16 — a build that produced nothing must never read as clean."""
    with pytest.raises(vsc.ArtifactError, match="no sdist"):
        vsc.find_artifacts(tmp_path)


def test_required_entry_missing_from_the_artifact_fails(tmp_path: Path) -> None:
    """covers: A4 — hatchling silently skips an include that matches nothing."""
    without_license = sorted(vsc.REQUIRED_ENTRIES - {"LICENSE"})
    sdist = _synthetic_sdist(tmp_path / "s.tar.gz", [*without_license, "src/yowo/__init__.py"])
    wheel = _synthetic_wheel(tmp_path / "w.whl", ["yowo/__init__.py"])
    found = vsc.violations(sdist, wheel)
    assert any(line.startswith("LICENSE:") and "absent" in line for line in found)


def test_ci_appends_an_sdist_job_and_registers_its_id() -> None:
    """covers: M4 — appended under the frozen ci.yml contract from `pr-ci-gate`."""
    from tests.unit.test_ci_contract import FROZEN_JOB_IDS

    ci = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert "sdist" in ci["jobs"], "the sdist job was not appended to ci.yml"
    assert "sdist" in FROZEN_JOB_IDS, (
        "the appended job id must be registered in FROZEN_JOB_IDS in the same commit"
    )
    assert "needs" not in ci["jobs"]["sdist"], (
        "appended jobs are independent by default (ci.yml contract, A14)"
    )
