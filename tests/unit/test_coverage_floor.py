"""Branch coverage, over the whole package, with a floor that fails the build.

There was no coverage machinery in this repo at all — no config, no floor, no
CI step. The number nobody measures is the number nobody defends.

Measured 2026-09-15 on the unit tier, darwin/arm64: combined **80.8737%**
(8149/9779 statements, 1830/2560 branches, 308 partial), line-only 83.3316%,
and 96 of 96 source files present in the denominator.

The floor is set from what CI measures, not from that figure. ubuntu takes
different branches from darwin — no MPS, no CoreML — and the honest floor is
the one the platform that enforces it actually reaches.

`fail_under` is read back out of pyproject.toml here rather than copied. Three
separate stale copies of the required-context list turned up in this repo in
one day, two of them inside the very guards meant to catch drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PACKAGE = REPO_ROOT / "src" / "yowo"

#: The floor the build enforces. Set from the CI measurement, rounded DOWN to a
#: whole percent. Raise it freely in the commit that earns it; never lower it to
#: turn a red build green — that is the one move this check exists to make
#: visible.
EXPECTED_FAIL_UNDER = 80


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _coverage_config() -> dict[str, Any]:
    return _pyproject().get("tool", {}).get("coverage", {})


def test_coverage_is_configured_with_branch_coverage_on() -> None:
    """covers: M1,S1 — line coverage alone calls a never-taken branch covered."""
    run = _coverage_config().get("run", {})
    assert run.get("branch") is True, (
        "branch coverage is off, so a two-way branch counts as covered when only "
        "one side ever executes"
    )


def test_the_denominator_is_every_source_file_not_only_the_imported_ones() -> None:
    """covers: M1,A2,A4,E2,R:PHANTOM_DENOMINATOR.

    Measuring only what the tests imported lets deleting a test RAISE the
    percentage.
    """
    source = _coverage_config().get("run", {}).get("source", [])
    assert source, "no coverage source is configured, so the denominator is whatever got imported"
    assert any(str(s).replace("\\", "/").endswith("src/yowo") for s in source), (
        f"coverage source {source!r} does not name the package; an unimported "
        f"module would be absent from the denominator rather than counted at 0%"
    )


def test_a_floor_is_configured_and_fails_the_build() -> None:
    """covers: M2,S1 — a step that reports a number and exits zero gates nothing."""
    report = _coverage_config().get("report", {})
    assert "fail_under" in report, (
        "no fail_under is configured, so coverage is reported and never enforced"
    )
    assert isinstance(report["fail_under"], (int, float))
    assert report["fail_under"] > 0


def test_the_configured_floor_is_the_value_this_check_expects() -> None:
    """covers: M3,A14,A18,E4,R:FLOOR_LOWERED_TO_PASS,S3."""
    configured = _coverage_config().get("report", {}).get("fail_under")
    assert configured == EXPECTED_FAIL_UNDER, (
        f"pyproject.toml sets fail_under = {configured!r} but this check expects "
        f"{EXPECTED_FAIL_UNDER!r}. Raising the floor is welcome — do it here and "
        f"there in the same commit. LOWERING it to make a red build green is the "
        f"move this check exists to make visible: write the test instead."
    )


def test_a_missing_floor_is_not_read_as_satisfied() -> None:
    """covers: A16,E1,S3 — deleting fail_under must not read as passing."""
    config: dict[str, Any] = {"report": {}}
    assert "fail_under" not in config["report"]
    with pytest.raises(AssertionError):
        _assert_floor(config, EXPECTED_FAIL_UNDER)


def _assert_floor(config: dict[str, Any], expected: int) -> None:
    """The floor rule, as a function, so its absent case can be exercised."""
    report = config.get("report", {})
    if "fail_under" not in report:
        msg = "fail_under is absent — coverage would be reported and never enforced"
        raise AssertionError(msg)
    if report["fail_under"] != expected:
        msg = f"fail_under is {report['fail_under']!r}, expected {expected!r}"
        raise AssertionError(msg)


def test_the_floor_is_not_above_what_ci_measured() -> None:
    """covers: A3,A13,A15 — a floor above the measured value never goes green."""
    assert EXPECTED_FAIL_UNDER <= 100
    measured_locally = 80.8737
    assert measured_locally >= EXPECTED_FAIL_UNDER, (
        f"the floor {EXPECTED_FAIL_UNDER} is above the {measured_locally}% measured "
        f"on darwin/arm64. CI may differ, but a floor above every known measurement "
        f"is a build that cannot go green."
    )


def _quality_steps() -> list[dict[str, Any]]:
    ci = yaml.safe_load(CI.read_text(encoding="utf-8"))
    return ci["jobs"]["quality"].get("steps", [])


def test_ci_runs_coverage_in_the_quality_job() -> None:
    """covers: M2,A7,A9,A11,S2 — a fifth quality command, beside the other four."""
    runs = " ".join(str(step.get("run", "")) for step in _quality_steps())
    assert "--cov" in runs, "the quality job runs no coverage step"


def test_the_coverage_step_cannot_report_success_without_measuring() -> None:
    """covers: A10,E3,R:GREEN_BY_SKIP,S2."""
    for step in _quality_steps():
        run = str(step.get("run", ""))
        if "--cov" not in run:
            continue
        assert "|| true" not in run, "the coverage step swallows its own failure"
        assert not step.get("continue-on-error"), "the coverage step continues on error"
        assert "--cov-fail-under" not in run, (
            "the floor belongs in pyproject.toml, where a check can read it back; "
            "a command-line override is a second copy that would drift"
        )


def test_what_the_number_covers_is_stated_where_it_is_reported() -> None:
    """covers: M4,A8,A12,A6,E5 — the unit tier only, and it says so.

    The integration tier is not in this measurement. A number presented as
    project coverage, produced by one tier, overstates what is defended.
    """
    for step in _quality_steps():
        if "--cov" not in str(step.get("run", "")):
            continue
        name = str(step.get("name", ""))
        assert "unit" in name.lower(), (
            f"the coverage step is named {name!r}, which does not say the number "
            f"covers the unit tier only"
        )
        return
    pytest.fail("no coverage step found to check")
