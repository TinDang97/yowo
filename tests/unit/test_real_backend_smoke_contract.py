"""The parts of the real-backend smoke that are not the inference itself.

The smoke lives in `tests/integration/test_real_backend_smoke.py` and executes a
backend. Three things about it are not executable there and are held here, in
the fast gate:

* CI must actually invoke it — a test file no workflow names proves nothing,
  which is what `test_cli_e2e.py` demonstrates today;
* no leg of it may report green without executing a backend — the
  `importorskip` hole this node closed;
* m2 box 7 must say what the smoke establishes, and no more.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

from yowo.types import BackendType, BoundingBox

_ROOT = Path(__file__).resolve().parents[2]
_SMOKE = _ROOT / "tests" / "integration" / "test_real_backend_smoke.py"
_CONFTEST = _ROOT / "tests" / "integration" / "conftest.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_MILESTONE = _ROOT / ".add" / "milestones" / "m2-survive-week-two.md"
_ENGINE = _ROOT / "src" / "yowo" / "engine.py"
_NODE = _ROOT / ".add" / "tasks" / "real-backend-smoke.md"

# The box this node owns, located by a phrase only it contains.
_BOX_7 = "asked for by name"


def _box_7() -> str:
    for line in _MILESTONE.read_text(encoding="utf-8").splitlines():
        if _BOX_7 in line:
            return line
    raise AssertionError(f"m2 box 7 not found by marker {_BOX_7!r} in {_MILESTONE}")


def _claim_half(box: str) -> str:
    """The box's own claim, with its AMENDED history excluded.

    The history quotes the wording being replaced, so scanning the whole line
    for a word finds the quote and reports the box still says it.
    """
    return box.partition("AMENDED")[0]


# ---------------------------------------------------------------------------
# The smoke is wired into CI and cannot skip its way green
# ---------------------------------------------------------------------------


def test_ci_invokes_the_smoke_file() -> None:
    """covers: M5 — a test no workflow names proves nothing."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert "tests/integration/test_real_backend_smoke.py" in workflow, (
        "ci.yml does not name the smoke file, so CI never runs it. This is exactly "
        "the state test_cli_e2e.py is in: real assertions nothing invokes."
    )


def test_no_leg_of_the_smoke_can_skip_its_way_green() -> None:
    """covers: M4, A8, R:GREENSKIP — the backend, the weight and the image share one policy."""
    smoke = _SMOKE.read_text(encoding="utf-8")
    fixture = smoke.partition("def smoke_engine")[2].partition("\ndef ")[0]
    assert "importorskip" not in fixture, (
        "the smoke's engine fixture guards an input with importorskip. A skip is green, "
        "so a CI job missing that input would report success having executed no backend — "
        "the hole this node exists to close."
    )
    assert "torch_available" in fixture, (
        "the smoke's engine fixture must take the torch_available guard, which fails in CI "
        "and skips locally, rather than deciding the policy for itself."
    )


def _conftest_module() -> object:
    spec = importlib.util.spec_from_file_location("_integration_conftest", _CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_missing_input_fails_in_ci() -> None:
    """covers: M4, A7, A10, E1, E2, R:GREENSKIP — in CI an unobtainable input raises."""
    unavailable = _conftest_module()._unavailable
    previous = os.environ.get("CI")
    os.environ["CI"] = "true"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            unavailable("torch", "ImportError: No module named 'torch'")
    finally:
        if previous is None:
            os.environ.pop("CI", None)
        else:
            os.environ["CI"] = previous
    assert "must not skip" in str(excinfo.value), (
        "the CI failure must say why skipping is not acceptable, or the next person "
        "reinstates the skip to get a green build"
    )


def test_a_missing_input_skips_locally_with_a_reason() -> None:
    """covers: A7, A9, A12, E1, E2 — outside CI it skips, naming what was unavailable."""
    unavailable = _conftest_module()._unavailable
    previous = os.environ.get("CI")
    os.environ.pop("CI", None)
    try:
        with pytest.raises(pytest.skip.Exception) as excinfo:
            unavailable("torch", "ImportError: No module named 'torch'")
    finally:
        if previous is not None:
            os.environ["CI"] = previous
    message = str(excinfo.value)
    assert "torch" in message and "unavailable" in message, (
        f"a skip must name what was unavailable; got {message!r}"
    )


def test_the_fallback_reason_prefix_the_smoke_guards_on_still_exists() -> None:
    """covers: M2 — the smoke's fallback guard is a string contract; pin it to the source.

    `test_the_smoke_executes_the_backend_it_asked_for` proves the backend was not
    reached by fallback by reading `selection.reason`. Measured: asserting on
    `selection.backend` alone does NOT discriminate — the engine rewrites the
    selection with the backend it fell back to. So the guard rests on this prefix,
    and a reword must redden here rather than silently disarm it.
    """
    engine_src = _ENGINE.read_text(encoding="utf-8")
    assert 'reason=f"Fallback from' in engine_src, (
        "engine.load no longer writes a reason beginning 'Fallback from'. The smoke's "
        "guard against regressing on a fallback-reached backend now passes vacuously — "
        "update _FALLBACK_REASON in tests/integration/test_real_backend_smoke.py to match."
    )
    smoke = _SMOKE.read_text(encoding="utf-8")
    assert '_FALLBACK_REASON = "Fallback from"' in smoke, (
        "the smoke's fallback guard no longer matches the prefix engine.py writes"
    )


# ---------------------------------------------------------------------------
# Box 7 says what the smoke establishes, and no more
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("required", ["date", "reason"], ids=["date", "reason"])
def test_box_7_records_that_it_was_amended(required: str) -> None:
    """covers: M6, A21, R:SILENTREWRITE — an undated rewrite reads as the original."""
    box = _box_7()
    assert "AMENDED" in box, "box 7 was rewritten without recording that it was amended"
    if required == "date":
        assert re.search(r"AMENDED \d{4}-\d{2}-\d{2}", box), (
            "box 7's amendment carries no date, so a reader cannot tell it ever changed"
        )
    else:
        assert "by human decision" in box, (
            "box 7's amendment records no authority; a box that changes on nobody's say-so "
            "teaches a reader that boxes are negotiable"
        )
        assert "already satisfied" in box, (
            "box 7's amendment must say WHY it changed — that the original words were "
            "already true and bought nothing"
        )


def test_box_7_names_the_backends_it_does_not_cover() -> None:
    """covers: A22, R:WIDEN, E7 — four backends are constructed by no unmocked check."""
    box = _box_7()
    assert "NOT COVERED" in box, (
        "box 7 executes one backend. Without naming the four it does not, a reader takes "
        "it as a whole-matrix guarantee."
    )
    uncovered = {b for b in BackendType if b is not BackendType.PYTORCH}
    for backend in uncovered:
        assert f"`{backend.value}`" in box, (
            f"box 7 does not name {backend.value} among the backends it leaves uncovered"
        )


def test_box_7_keeps_the_clause_saying_what_it_is_a_baseline_for() -> None:
    """covers: A20 — the purpose clause is why the box exists."""
    box = _box_7()
    assert "regress against" in _claim_half(box), (
        "box 7 lost the clause naming what it is a baseline for. Without it the box reads "
        "as a smoke test for its own sake, and the tasks below it lose their stated reason."
    )


def test_box_7_claims_nothing_a_probe_refutes() -> None:
    """covers: M6 — every symbol the box names is resolved, not trusted."""
    box = _box_7()

    for field in ("x1", "x2", "y1", "y2", "confidence", "class_name"):
        assert f"`{field}`" in box, f"box 7 does not name the {field} invariant"
        assert hasattr(BoundingBox, "__dataclass_fields__")
        assert field in BoundingBox.__dataclass_fields__, (
            f"box 7 names `{field}` as a box invariant, but BoundingBox has no such field"
        )

    named = set(re.findall(r"`(pytorch|onnx|tensorrt|openvino|coreml)`", box))
    assert named, "box 7 names no backend at all"
    values = {b.value for b in BackendType}
    for backend in named:
        assert backend in values, (
            f"box 7 names backend `{backend}`, which is not a BackendType value"
        )

    from yowo.backends import create_backend  # noqa: F401

    assert "`create_backend`" in box, (
        "box 7 must name the factory the smoke constructs through, so the claim is "
        "traceable to code rather than to the word 'real'"
    )


def test_no_selection_or_fallback_source_changed() -> None:
    """covers: M7, R:BEHAVIOUR — this node makes behaviour visible; it does not move it."""
    node = _NODE.read_text(encoding="utf-8")
    scope = node.partition("scope:")[2].partition("gives:")[0]
    assert "src/" not in scope, (
        "real-backend-smoke declares src/ in scope. Whether a backend that cannot load the "
        "given weight should warn or raise is degraded-mode-correctness's call; this node "
        "only makes today's answer visible."
    )
    engine_src = _ENGINE.read_text(encoding="utf-8")
    assert "backends_to_try = [self._selection.backend, *get_fallback_backends(" in engine_src, (
        "engine.load's fallback chain changed shape. That is a behaviour change, and it "
        "belongs to degraded-mode-correctness, not here."
    )
