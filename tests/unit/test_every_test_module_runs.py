"""Every test module must be run by a job on a pull request.

Measured 2026-09-16, before this guard existed: **42 tests never ran in CI and
all of them passed locally.**

    tests/integration/test_cli_e2e.py                   21   named by no workflow
    tests/integration/test_chroma_gallery_persistence.py  8   named by no workflow
    tests/integration/test_onnx_provider_honoured.py      2   named by no workflow
    tests/unit/test_chroma_gallery.py                    11   importorskip'd chromadb

The third is the one that should sting: it was built in PR #50 *as a gate*,
and then nothing ran it.

All four look like coverage in a file listing. A test nobody runs is a claim
nobody checked, and the file listing is how everybody checks.

This guard computes coverage from **what jobs actually run** — the paths their
`run:` commands name — not from a literal a file happens to contain. That
distinction is not academic here: `test_ci_weight_fixture.py` matched sources
by the literals `yolo11x`/`yolo26x` and was blind to `test_export_parity.py`,
which iterates `for size in ModelSize` and downloads both without writing
either down. Five times now this repository has found a drifted copy of a list
inside the thing meant to notice drift.

Authored red under ADD task `integration-tier-revival`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TESTS = REPO_ROOT / "tests"

#: Files under `tests/` that are not test modules. Excluded by KIND — a name
#: added here excuses nothing else.
NOT_A_TEST_MODULE = ("conftest.py", "__init__.py")


def _workflows() -> dict[str, dict[str, Any]]:
    return {p.name: (yaml.safe_load(p.read_text()) or {}) for p in sorted(WORKFLOWS.glob("*.yml"))}


def runs_on_pull_request(workflow: dict[str, Any]) -> bool:
    """Whether this workflow triggers on a pull request.

    `yaml.safe_load` resolves the bare key `on` to the boolean True, so the
    trigger has to be read from both spellings or a real workflow reads as
    having none.
    """
    triggers = workflow.get("on", workflow.get(True))
    return isinstance(triggers, dict) and "pull_request" in triggers


def _run_commands(workflow: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps", []) or []:
            run = step.get("run")
            if run:
                out.append(" ".join(str(run).split()))
    return out


#: A pytest target inside a command: a path under `tests/`, file or directory.
_TARGET = re.compile(r"(?<![\w/])(tests/[\w./-]*)")


def pytest_targets(command: str) -> set[str]:
    """The `tests/` paths a command runs.

    Only from a command that actually invokes pytest. A `run:` step that merely
    mentions a path — an echo, a comment, a cache key — runs nothing, and
    treating the mention as coverage is exactly the spelling match this guard
    exists to avoid (R:GUARD_BY_SPELLING).
    """
    if "pytest" not in command:
        return set()
    return {m.rstrip("/") for m in _TARGET.findall(command)}


def covered_modules(targets: set[str]) -> set[Path]:
    """Every test module those targets run, resolved against the repo."""
    found: set[Path] = set()
    for target in targets:
        path = REPO_ROOT / target
        if path.is_dir():
            found.update(p for p in path.rglob("test_*.py"))
        elif path.is_file() and path.name.startswith("test_"):
            found.add(path)
    return found


def all_test_modules() -> set[Path]:
    return {
        p
        for p in TESTS.rglob("test_*.py")
        if p.name not in NOT_A_TEST_MODULE and "__pycache__" not in p.parts
    }


def unrun_modules() -> set[Path]:
    """Test modules no pull-request job runs. The whole subject, in one call."""
    covered: set[Path] = set()
    for workflow in _workflows().values():
        if not runs_on_pull_request(workflow):
            continue
        for command in _run_commands(workflow):
            covered |= covered_modules(pytest_targets(command))
    return all_test_modules() - covered


def unrun_message(modules: set[Path]) -> str:
    """Name each module and say what to do — a count teaches nobody which file."""
    listed = "\n".join(f"  - {p.relative_to(REPO_ROOT)}" for p in sorted(modules))
    return (
        f"{len(modules)} test module(s) are run by no job on a pull request:\n{listed}\n"
        f"Add each to a job in a workflow that triggers on `pull_request`, or delete it. "
        f"A module nobody runs looks like coverage in a file listing and checks nothing."
    )


def test_every_test_module_is_run_by_a_pull_request_job() -> None:
    """The guard. Unit and integration alike — the chromadb unit module sat
    inside Quality Gate's path and still never executed."""
    unrun = unrun_modules()
    assert not unrun, unrun_message(unrun)


def test_the_failure_names_the_module_and_the_fix() -> None:
    """Someone who just added a test file must learn which file, and what to do."""
    message = unrun_message({TESTS / "integration" / "test_made_up.py"})
    assert "tests/integration/test_made_up.py" in message
    assert "pull_request" in message
    assert "delete it" in message


def test_coverage_is_computed_from_what_jobs_run_not_from_a_literal() -> None:
    """binds R:GUARD_BY_SPELLING — a mention is not a run.

    A step that echoes a path, or names one in a cache key, executes nothing.
    A guard that counted those would report the tier covered while it sat idle.
    """
    assert pytest_targets("echo tests/integration/test_cli_e2e.py") == set()
    assert pytest_targets("uv run pytest tests/unit/ -x -q") == {"tests/unit"}
    assert pytest_targets("uv run pytest tests/integration/test_cli_e2e.py -q") == {
        "tests/integration/test_cli_e2e.py"
    }
    # A directory target covers what is under it, which is how `tests/unit/` works.
    assert (TESTS / "unit" / "test_every_test_module_runs.py") in covered_modules({"tests/unit"})


def test_a_workflow_without_a_pull_request_trigger_does_not_count() -> None:
    """A scheduled job does not gate a pull request.

    `parity.yml` runs `parity-all` on a schedule; on a pull request it reports
    `skipping`, and GitHub treats a skipped required check as satisfied.
    """
    assert runs_on_pull_request({"on": {"pull_request": {"branches": ["main"]}}})
    assert not runs_on_pull_request({"on": {"schedule": [{"cron": "0 4 * * *"}]}})
    assert not runs_on_pull_request({"on": {"push": {"tags": ["v*"]}}})
    # the bare key `on` parses as the boolean True — read both spellings
    assert runs_on_pull_request({True: {"pull_request": None}})


def test_conftest_and_dunder_init_are_not_test_modules() -> None:
    """Excluded by kind. The exclusion list excuses nothing else."""
    modules = {p.name for p in all_test_modules()}
    assert "conftest.py" not in modules
    assert "__init__.py" not in modules
    assert "test_every_test_module_runs.py" in modules


def test_a_module_named_by_a_renamed_job_is_not_covered() -> None:
    """Coverage follows the command, not the label.

    Renaming a job changes the published check name and nothing about what it
    runs; renaming what it runs is what uncovers a module. This asserts the
    guard reads the latter.
    """
    workflow = {
        "on": {"pull_request": {"branches": ["main"]}},
        "jobs": {
            "whatever-it-is-called-now": {
                "steps": [{"run": "uv run pytest tests/integration/test_cli_e2e.py -q"}]
            }
        },
    }
    targets: set[str] = set()
    for command in _run_commands(workflow):
        targets |= pytest_targets(command)
    assert targets == {"tests/integration/test_cli_e2e.py"}

    dropped = {"on": {"pull_request": None}, "jobs": {"same-name": {"steps": [{"run": "echo hi"}]}}}
    assert not {t for c in _run_commands(dropped) for t in pytest_targets(c)}
