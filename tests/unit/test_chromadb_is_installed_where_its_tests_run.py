"""A dependency CI never installs turns its tests into a green skip.

`tests/unit/test_chroma_gallery.py` (11 tests) and
`tests/integration/test_chroma_gallery_persistence.py` (8 tests) both open with
`pytest.importorskip("chromadb")`. `tests/unit/` is inside Quality Gate's path,
so that module has been *collected* on every pull request since it was written
— and skipped every time, because no workflow installs chromadb.

They cover `yowo.tracking.ChromaEmbeddingGallery`, which `yowo/tracking/__init__.py`
exports publicly. Nineteen tests, a shipped public class, zero executions.

Measured 2026-09-16: chromadb 1.5.2 resolves in 176 ms warm (78 packages), and
with it installed all 19 pass — 11 unit in 2.73 s, 8 integration inside a
10.52 s run. The extra already exists in `pyproject.toml` (line 48), so nothing
here declares a new dependency.

Authored red under ADD task `integration-tier-revival`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml

from tests.unit.test_every_test_module_runs import (
    REPO_ROOT,
    covered_modules,
    pytest_targets,
    runs_on_pull_request,
)

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TESTS = REPO_ROOT / "tests"

#: The extra `pyproject.toml` already declares. Not a new dependency.
EXTRA = "chromadb"


def _importorskip_arguments(source: str) -> set[str]:
    """The dependencies a module actually calls `importorskip` on.

    Parsed, not matched. A regex over the source found this very module, whose
    docstring quotes `pytest.importorskip("chromadb")` as an example — prose
    that executes nothing. Matching text rather than behaviour is the defect
    this task exists to remove, and it reappeared inside the guard against it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a file that cannot parse cannot run
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "importorskip" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    return found


def modules_importorskipping(dependency: str) -> set[Path]:
    """Test modules whose execution is conditional on `dependency` being present."""
    return {
        p
        for p in TESTS.rglob("test_*.py")
        if "__pycache__" not in p.parts and dependency in _importorskip_arguments(p.read_text())
    }


def _jobs(workflow_path: Path) -> dict[str, dict[str, Any]]:
    return (yaml.safe_load(workflow_path.read_text()) or {}).get("jobs", {}) or {}


def installs_extra(job: dict[str, Any], extra: str) -> bool:
    """Whether this job's install step actually brings the extra in."""
    for step in job.get("steps", []) or []:
        run = " ".join(str(step.get("run", "")).split())
        if "uv sync" in run and (f"--extra {extra}" in run or "--all-extras" in run):
            return True
    return False


def test_every_job_running_a_chromadb_module_installs_the_extra() -> None:
    """binds M3 and R:GREEN_BY_SKIP — importorskip turns an absent dep into green."""
    conditional = modules_importorskipping(EXTRA)
    assert conditional, (
        "no module importorskips chromadb any more — if the construct is gone this "
        "check should be reconsidered deliberately, not deleted to go green"
    )

    offenders: dict[str, list[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text()) or {}
        for job_id, job in _jobs(path).items():
            run_here: set[Path] = set()
            for step in job.get("steps", []) or []:
                run_here |= covered_modules(
                    pytest_targets(" ".join(str(step.get("run", "")).split()))
                )
            at_risk = sorted(p.name for p in (run_here & conditional))
            if at_risk and not installs_extra(job, EXTRA):
                offenders[f"{path.name}:{job_id}"] = at_risk
        del workflow

    assert not offenders, (
        f"these jobs run chromadb-dependent modules without installing the extra, so "
        f"the modules skip and the job reports success: {offenders}. "
        f"Add `--extra {EXTRA}` to the job's `uv sync`."
    )


def test_the_release_gate_installs_what_the_pull_request_gate_installs() -> None:
    """binds M4 and R:DIVERGENCE — neither path may be the weaker of the two.

    `ci.yml` and `release.yml` already run four identical quality COMMANDS. The
    same command over a different install set is a different gate wearing the
    same name, and the weaker one would be whichever installs less.
    """
    ci_quality = _jobs(WORKFLOWS / "ci.yml")["quality"]
    release_quality = _jobs(WORKFLOWS / "release.yml")["quality"]

    assert installs_extra(ci_quality, EXTRA) == installs_extra(release_quality, EXTRA), (
        f"ci.yml quality installs {EXTRA}={installs_extra(ci_quality, EXTRA)} but "
        f"release.yml quality installs {EXTRA}={installs_extra(release_quality, EXTRA)}. "
        f"Whichever installs less runs fewer tests under the same command."
    )
    assert installs_extra(ci_quality, EXTRA), (
        f"neither gate installs {EXTRA}, so the 11 unit tests covering the shipped "
        f"ChromaEmbeddingGallery skip on both paths"
    )


def test_an_absent_chromadb_fails_under_ci_rather_than_skipping() -> None:
    """Q3: a skipped test is green.

    Every job that runs a conditional module must set `CI=true`, so that if the
    install ever stops working the job reds instead of quietly reporting a pass
    over zero executed tests.
    """
    conditional = modules_importorskipping(EXTRA)
    missing: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text()) or {}
        if not runs_on_pull_request(workflow):
            continue
        for job_id, job in _jobs(path).items():
            for step in job.get("steps", []) or []:
                run = " ".join(str(step.get("run", "")).split())
                if not (covered_modules(pytest_targets(run)) & conditional):
                    continue
                env = {**(job.get("env") or {}), **(step.get("env") or {})}
                if str(env.get("CI", "")).lower() not in ("1", "true", "yes"):
                    missing.append(f"{path.name}:{job_id}")

    assert not missing, (
        f"these steps run chromadb-dependent modules without CI=true: {sorted(set(missing))}. "
        f"Without it an absent dependency skips, and a skip is green."
    )
