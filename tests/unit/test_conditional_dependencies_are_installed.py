"""A dependency no job installs turns its tests into a green skip.

This generalises and REPLACES the chromadb-specific guard that shipped with
`integration-tier-revival`. That guard closed one instance and could not see
the next: its own first CI run logged a skip it had no concept of —
`tests/unit/test_backend_roster.py` `importorskip`s openvino, the `quality` job
did not install it, and the test that skipped guards a defect that shipped
("exactly what happened on openvino 2026.3.1 for every load", where a malformed
IR file was reported as a missing package and sent users to install something
they already had).

Measured 2026-09-16: 18 modules use `importorskip` across 8 dependencies —
chromadb, torch, onnxruntime, onnx, onnxruntime.quantization,
yowo.backends._onnx, openvino, pycocotools. Exactly one was unprovided.

Two design rules, both learned by getting them wrong first:

**Resolve provision from `pyproject.toml`, never from a hand-written map.** The
first version of this analysis mapped pycocotools to the `benchmark` extra and
reported the `quality` job at risk for two modules. `pycocotools>=2.0.4` is in
the dev group DIRECTLY as well, so both were fine — CI's log showed a single
skip. The hand-map produced two false positives out of three, which is the
same drift this repository has now found seven times in a list maintained by
hand inside the thing meant to notice drift.

**Detect the call by parsing, never by matching text.** A regex for
`importorskip(...)` matched a docstring that quotes the call as an example —
prose that executes nothing.

Authored red under ADD task `conditional-dep-install`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TESTS = REPO_ROOT / "tests"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Gaps a human accepted, each with its reason. EMPTY is valid and is the
#: strongest state — never read an empty mapping as a disabled check. An entry
#: here is a decision on the record; omitting one is not acceptance (M3).
#: Keyed by (job, module, dependency).
ACCEPTED_GAPS: dict[tuple[str, str, str], str] = {}

#: Dependencies that resolve to the package under test itself.
_FIRST_PARTY = "yowo"


def importorskip_dependencies(source: str) -> set[str]:
    """The dependencies a module actually calls `importorskip` on.

    Parsed, not matched — binds R:TEXT_MATCH.
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


def top_level(dependency: str) -> str:
    """`onnxruntime.quantization` -> `onnxruntime`; `yowo.backends._onnx` -> `yowo`."""
    return dependency.split(".", 1)[0]


def _distribution_names(requirements: list[str]) -> set[str]:
    """Package names from PEP 508 requirement strings, markers and specifiers dropped."""
    names: set[str] = set()
    for req in requirements:
        head = re.split(r"[;\[<>=!~ ]", req.strip(), maxsplit=1)[0]
        if head:
            names.add(head.replace("-", "_").lower())
    return names


def provided_by(extras: set[str]) -> set[str]:
    """What a `uv sync --group dev` plus these extras actually provides.

    Read from pyproject.toml — binds M1 and R:HANDMAP.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    provided = _distribution_names((data.get("dependency-groups") or {}).get("dev", []))
    optional = (data.get("project") or {}).get("optional-dependencies") or {}
    for extra in extras:
        provided |= _distribution_names(optional.get(extra, []))
    provided |= _distribution_names((data.get("project") or {}).get("dependencies", []))
    provided.add(_FIRST_PARTY)
    return provided


_TARGET = re.compile(r"(?<![\w/])(tests/[\w./-]*)")


def _pytest_targets(command: str) -> set[str]:
    if "pytest" not in command:
        return set()
    return {m.rstrip("/") for m in _TARGET.findall(command)}


def _modules_under(targets: set[str]) -> set[Path]:
    found: set[Path] = set()
    for target in targets:
        path = REPO_ROOT / target
        if path.is_dir():
            found.update(path.rglob("test_*.py"))
        elif path.is_file() and path.name.startswith("test_"):
            found.add(path)
    return found


def _job_extras(job: dict[str, Any]) -> set[str]:
    extras: set[str] = set()
    for step in job.get("steps", []) or []:
        run = " ".join(str(step.get("run", "")).split())
        if "uv sync" in run:
            extras |= set(re.findall(r"--extra (\S+)", run))
    return extras


def gaps() -> dict[tuple[str, str, str], None]:
    """Every (job, module, dependency) the job runs but does not provide."""
    found: dict[tuple[str, str, str], None] = {}
    for workflow_path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(workflow_path.read_text()) or {}
        triggers = workflow.get("on", workflow.get(True))
        if not (isinstance(triggers, dict) and "pull_request" in triggers):
            continue
        for job_id, job in (workflow.get("jobs") or {}).items():
            extras = _job_extras(job)
            provided = provided_by(extras)
            for step in job.get("steps", []) or []:
                run = " ".join(str(step.get("run", "")).split())
                for module in _modules_under(_pytest_targets(run)):
                    for dependency in importorskip_dependencies(module.read_text()):
                        if top_level(dependency).replace("-", "_").lower() in provided:
                            continue
                        key = (f"{workflow_path.name}:{job_id}", module.name, dependency)
                        found[key] = None
    return found


def gap_message(found: dict[tuple[str, str, str], None]) -> str:
    """Name module, dependency AND job, and both ways out.

    A message that names only a count leaves deleting the check as the cheapest
    move available to whoever reads it.
    """
    lines = [
        f"  - {module} importorskips {dependency!r}, but {job} does not provide it"
        for job, module, dependency in sorted(found)
    ]
    body = "\n".join(lines)
    return (
        f"{len(found)} conditional dependenc(ies) are not provided by the job that runs "
        f"the module, so the module skips and the job reports success:\n{body}\n"
        f"Either add the extra to that job's `uv sync`, or record the gap in "
        f"ACCEPTED_GAPS with the reason it is acceptable."
    )


def test_every_conditional_dependency_is_provided_by_the_job_that_runs_it() -> None:
    """The guard, over every dependency rather than one — binds R:GREEN_BY_SKIP."""
    unprovided = {k: v for k, v in gaps().items() if k not in ACCEPTED_GAPS}
    assert not unprovided, gap_message(unprovided)


def test_provision_is_read_from_pyproject_not_from_a_hand_written_map() -> None:
    """binds M1 and R:HANDMAP.

    A hand-map gave two false positives out of three on its first run: it said
    pycocotools came only from the `benchmark` extra, when the dev group
    declares it too.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    dev = _distribution_names((data.get("dependency-groups") or {}).get("dev", []))
    assert "pycocotools" in dev, (
        "this check's premise is that the dev group is read, not assumed. If "
        "pycocotools left the dev group, re-derive rather than editing a constant."
    )
    # openvino is an extra only; naming it must change the answer.
    assert "openvino" not in provided_by(set())
    assert "openvino" in provided_by({"openvino"})


def test_a_dependency_from_the_dev_group_counts_as_provided() -> None:
    """binds E2 — pycocotools ships in the dev group AND an extra."""
    provided = provided_by(set())
    for dependency in ("torch", "onnx", "onnxruntime", "pycocotools"):
        assert dependency in provided, f"{dependency} is in the dev group and must count"


def test_a_submodule_resolves_through_its_top_level_distribution() -> None:
    """binds E3 — `onnxruntime.quantization` is onnxruntime, not a package of its own."""
    assert top_level("onnxruntime.quantization") == "onnxruntime"
    assert top_level("yowo.backends._onnx") == "yowo"
    assert top_level("chromadb") == "chromadb"
    assert top_level("onnxruntime.quantization") in provided_by(set())


def test_importorskip_is_detected_by_parsing_not_by_matching_text() -> None:
    """binds R:TEXT_MATCH — prose that quotes the call executes nothing."""
    real = 'import pytest\npytest.importorskip("chromadb")\n'
    assert importorskip_dependencies(real) == {"chromadb"}

    prose = '"""Modules open with pytest.importorskip("chromadb") — an example."""\n'
    assert importorskip_dependencies(prose) == set(), (
        "a docstring quoting the call is not a call. A regex over source flagged "
        "this module's own documentation on 2026-09-16."
    )

    commented = "# pytest.importorskip('openvino')\nx = 1\n"
    assert importorskip_dependencies(commented) == set()


def test_the_failure_names_the_module_the_dependency_and_the_job() -> None:
    """binds A6 and E1 — otherwise deleting the check is the cheapest fix."""
    message = gap_message({("ci.yml:quality", "test_made_up.py", "openvino"): None})
    assert "test_made_up.py" in message
    assert "openvino" in message
    assert "ci.yml:quality" in message
    assert "ACCEPTED_GAPS" in message
    assert "uv sync" in message


def test_an_accepted_gap_requires_a_reason() -> None:
    """binds M3 and R:SILENT_ACCEPT — omission is not acceptance."""
    for key, reason in ACCEPTED_GAPS.items():
        assert isinstance(key, tuple) and len(key) == 3, (
            f"an accepted gap is keyed by (job, module, dependency); got {key!r}"
        )
        assert isinstance(reason, str) and len(reason.strip()) >= 20, (
            f"the accepted gap {key!r} carries no usable reason. An entry without "
            f"one records that somebody gave up, not that somebody decided."
        )


def test_an_accepted_gap_that_is_no_longer_real_fails() -> None:
    """binds E6 — the list must not outlive its reason.

    A gap that was closed leaves an entry asserting a hole that no longer
    exists, and the next reader trusts it.
    """
    real = set(gaps())
    stale = sorted(k for k in ACCEPTED_GAPS if k not in real)
    assert not stale, (
        f"these accepted gaps no longer correspond to a real one: {stale}. "
        f"Remove them — an acceptance that outlives its reason misleads."
    )


def test_the_accepted_gap_list_may_be_empty() -> None:
    """binds E5 and A10 — empty is the strongest state, never a disabled check.

    This asserts the guard still computes something when nothing is accepted,
    so that emptying the list can never be mistaken for switching it off.
    """
    assert isinstance(ACCEPTED_GAPS, dict)
    computed = gaps()
    assert isinstance(computed, dict)
    # The guard runs over a real, non-trivial population regardless.
    conditional = [
        p
        for p in TESTS.rglob("test_*.py")
        if "__pycache__" not in p.parts and importorskip_dependencies(p.read_text())
    ]
    assert len(conditional) >= 10, (
        f"only {len(conditional)} modules use importorskip; if that collapsed, this "
        f"guard is passing over an empty population and proves nothing"
    )


def test_both_quality_gates_provide_the_same_dependencies() -> None:
    """binds M4 and R:DIVERGENCE — neither path may be the weaker.

    `ci.yml` and `release.yml` already run four identical quality COMMANDS. The
    same command over a smaller install set is a different gate wearing the
    same name.
    """
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]["quality"]
    release = yaml.safe_load((WORKFLOWS / "release.yml").read_text())["jobs"]["quality"]
    ci_provided, release_provided = provided_by(_job_extras(ci)), provided_by(_job_extras(release))
    assert ci_provided == release_provided, (
        f"the gates provide different dependencies. Only in ci.yml: "
        f"{sorted(ci_provided - release_provided)}; only in release.yml: "
        f"{sorted(release_provided - ci_provided)}. Whichever provides less runs "
        f"fewer tests under the same command."
    )
