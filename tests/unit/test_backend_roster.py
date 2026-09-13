"""Every backend `create_backend()` can return has a verdict, and the verdict is honest.

Measured 2026-09-13. `create_backend()` returns FIVE backends. Exactly one —
`pytorch` — is executed unmocked anywhere, by `real-backend-smoke`. Coverage of
the rest, from `pytest tests/unit --cov=src/yowo/backends`:

    _openvino.py   21%      _tensorrt.py   73%
    _pytorch.py    58%      _onnx.py       77%

and there is no `test_pytorch_backend.py` at all. A backend nobody can run is
not "untested", it is *unshipped* — and the only thing distinguishing the two,
from the outside, is whether someone wrote it down.

So the roster is committed, and it is TOTAL over the enum: a sixth backend
cannot be added without a verdict. `UNVERIFIED` entries carry the reason and the
runner they would need, because "unverified" alone is indistinguishable from
"nobody looked".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from yowo.backends._roster import (
    EXECUTED,
    UNVERIFIED,
    Verdict,
    verdict_for,
)
from yowo.types import BackendType

_REPO_ROOT = Path(__file__).parent.parent.parent


def test_the_roster_is_total_over_the_backend_enum() -> None:
    """covers: M1, A1, E2 — enumerated at runtime, so a new member cannot slip in."""
    missing = [b for b in BackendType if verdict_for(b) is None]

    assert not missing, (
        f"these backends have no roster verdict: {[b.value for b in missing]}. "
        "A hand-written list would have stayed silent here — the enum is the source."
    )
    assert len(EXECUTED) + len(UNVERIFIED) == len(list(BackendType))


def test_every_backend_has_exactly_one_verdict() -> None:
    """covers: M1, A1 — executed and unverified are disjoint."""
    overlap = set(EXECUTED) & set(UNVERIFIED)
    assert not overlap, f"{[b.value for b in overlap]} is both executed and unverified"


def test_every_unverified_entry_names_a_reason_and_a_runner() -> None:
    """covers: M1, A13, A8, E1 — no bare 'unverified'."""
    for backend, entry in UNVERIFIED.items():
        assert entry.reason.strip(), f"{backend.value} has no reason"
        assert entry.runner.strip(), f"{backend.value} names no runner"
        # A reason that merely restates the verdict tells a reader nothing.
        assert entry.reason.strip().lower() not in {"unverified", "not tested", "untested"}, (
            f"{backend.value}'s reason restates the verdict instead of explaining it"
        )
        assert len(entry.reason) >= 20, (
            f"{backend.value}'s reason is too short to be a reason: {entry.reason!r}"
        )


def test_the_executed_set_is_what_the_conformance_suite_drives() -> None:
    """covers: R:MOCKED — the roster's claim and the suite's parametrisation are one list.

    Declaring a backend EXECUTED and then not driving it is the failure this
    whole node exists to make impossible, so the suite reads the roster rather
    than carrying its own copy.
    """
    suite = (_REPO_ROOT / "tests/integration/test_backend_conformance.py").read_text()

    assert "from yowo.backends._roster import" in suite, (
        "the conformance suite must parametrise over the roster's EXECUTED set; "
        "a second hand-maintained list is how the two drift apart"
    )
    assert not re.search(r"\bMagicMock\b|\bmock\.patch\b|unittest\.mock", suite), (
        "no mock may appear in the conformance suite — a mocked backend satisfies "
        "every assertion here while executing nothing"
    )


def test_a_skipped_backend_is_not_recorded_as_covered() -> None:
    """covers: R:SKIPGREEN, A8 — a skip maps to unverified, never to executed.

    Lesson Q3: a skipped test is green. Two fixtures once pointed at one
    machine's filesystem and an entire tier reported success for months.
    """
    for backend in EXECUTED:
        entry = verdict_for(backend)
        assert entry is Verdict.EXECUTED, f"{backend.value} claims execution without being executed"

    # The suite must fail rather than skip when CI cannot obtain an input.
    conftest = (_REPO_ROOT / "tests/integration/conftest.py").read_text()
    assert "CI must not skip" in conftest, (
        "the integration tier's unavailable-input policy must still fail in CI"
    )


@pytest.mark.parametrize(
    "backend",
    [BackendType.TENSORRT, BackendType.COREML],
    ids=["tensorrt", "coreml"],
)
def test_the_backends_ci_cannot_reach_say_which_runner_they_need(backend: BackendType) -> None:
    """covers: M1, A3, A13 — the reason is a runner, not a shrug.

    openvino is deliberately NOT here: it is installable on ubuntu CPU
    (openvino==2026.3.1, measured), so listing it unverified would record a
    choice as a limitation.
    """
    entry = UNVERIFIED[backend]
    assert re.search(r"runner|GPU|macOS|NVIDIA", entry.runner), (
        f"{backend.value}'s runner field must name the machine it needs: {entry.runner!r}"
    )


def test_openvino_is_executed_not_excused() -> None:
    """covers: M1, A3 — measured installable, so it is executed."""
    assert BackendType.OPENVINO in EXECUTED, (
        "openvino>=2024.0 is a declared extra and installs on ubuntu CPU "
        "(measured 2026-09-13: openvino==2026.3.1, 2 packages). Listing it "
        "unverified would record a choice as a limitation."
    )


# ---------------------------------------------------------------------------
# M4 — OpenVINO imports from the module this version publishes
# ---------------------------------------------------------------------------


def test_openvino_imports_core_from_the_published_module() -> None:
    """covers: M4, E7 — the modern path first, the pre-2025 path as fallback.

    `openvino.runtime` was REMOVED in OpenVINO 2025. Measured 2026-09-13 on
    openvino 2026.3.1: the old import raises, while `ov.Core()` reports
    `['CPU']` and both read_model and compile_model succeed in the same process.
    """
    source = (_REPO_ROOT / "src/yowo/backends/_openvino.py").read_text()

    assert "from openvino import Core" in source, (
        "OpenVINO >=2025 publishes Core at `openvino`, not `openvino.runtime`"
    )
    assert "from openvino.runtime import Core" in source, (
        "the pre-2025 path must stay as a fallback — dropping it would break "
        "every deployment still on openvino 2024"
    )
    assert source.index("from openvino import Core") < source.index(
        "from openvino.runtime import Core"
    ), "the modern import must be tried first"


def test_the_core_import_is_used_by_load_not_duplicated() -> None:
    """covers: M4 — load() goes through the helper, so the fallback is live there.

    The wiring, not the mechanism: a correct helper that `load` does not call is
    the shape that survived a mutation sweep two nodes ago.
    """
    import inspect

    from yowo.backends._openvino import OpenVinoBackend, _import_core

    body = inspect.getsource(OpenVinoBackend.load)
    assert "_import_core()" in body, "load() must obtain Core through the helper"
    assert callable(_import_core)


def test_an_openvino_load_failure_is_not_reported_as_a_missing_dependency(
    tmp_path: Path,
) -> None:
    """covers: M4, R:FALSEDEP, E8 — a real failure stops wearing a false label."""
    pytest.importorskip("openvino")
    from yowo.backends._openvino import OpenVinoBackend
    from yowo.errors import BackendLoadError, DependencyError
    from yowo.hardware import get_hardware_profile

    junk = tmp_path / "model.xml"
    junk.write_text("<not-an-ir-file/>")

    backend = OpenVinoBackend(get_hardware_profile())
    with pytest.raises(BackendLoadError) as excinfo:
        backend.load(junk, device="cpu")

    assert not isinstance(excinfo.value, DependencyError), (
        "a malformed IR file is not a missing package. Reporting it as one sent "
        "users to install a package they already had — which is exactly what "
        "happened on openvino 2026.3.1 for every load."
    )
    assert "uv add openvino" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# M5 — the roster's claims are re-proved on every PR
# ---------------------------------------------------------------------------

_CI = _REPO_ROOT / ".github/workflows/ci.yml"


def test_ci_installs_the_openvino_extra_and_runs_this_suite() -> None:
    """covers: M5, A7 — a roster proved once rots exactly like the coverage did."""
    import yaml

    workflow = yaml.safe_load(_CI.read_text())
    job = workflow["jobs"].get("conformance")

    assert job is not None, "ci.yml must carry a job that runs the conformance suite"

    steps = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "--extra openvino" in steps, (
        "the roster claims OpenVINO is EXECUTED; CI must install the extra that "
        "makes that true, or the claim is aspirational"
    )
    assert "tests/integration/test_backend_conformance.py" in steps, (
        "the job must run the suite it exists for"
    )

    env = {}
    for step in job["steps"]:
        env.update(step.get("env", {}))
    assert str(env.get("CI", "")).lower() == "true", (
        "without CI=true an unobtainable weight becomes a green skip, and the "
        "job reports success having executed no backend at all (lesson Q3)"
    )


def test_the_conformance_job_is_in_the_frozen_ci_contract() -> None:
    """covers: M5 — added under pr-ci-gate's APPEND rule, not smuggled in."""
    from tests.unit.test_ci_contract import FROZEN_JOB_IDS

    assert "conformance" in FROZEN_JOB_IDS
