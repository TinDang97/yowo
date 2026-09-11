"""The CI smoke: a real backend, a real image, and assertions a broken one fails.

m2 box 7 asks that "at least one real backend is constructed and executed by a
test — no mock". That was already true before this file existed, and it bought
nothing, for three measured reasons:

* the backend that ran was not chosen — ``select_backend`` picks ONNX here,
  ``OnnxBackend.load`` is handed a ``.pt`` and raises, and the engine falls back
  to PyTorch. The old assertion ``det.backend == PYTORCH`` passed *because* of
  that failure;
* the input was a 640x640 black frame, which yields **0 boxes**, and the
  assertions were ``isinstance(result, list)`` and ``inference_time_ms > 0`` —
  all of which a backend returning nothing forever satisfies;
* the fixture guarded torch with ``importorskip``, so the whole thing could
  report green having executed no backend at all.

So this file executes the backend it *asked* for, on ``bus.jpg``, and asserts
box invariants a broken backend cannot satisfy. ``test_a_black_frame_does_not
_satisfy_this_smoke`` is the control: if the old input still passed these
assertions, the new ones would prove no more than the old ones did.

Behaviour is not changed here. Whether selecting a backend that cannot load the
given weight should warn or raise belongs to ``degraded-mode-correctness``; this
file makes today's answer visible so it cannot change unnoticed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from yowo.backends._selector import select_backend
from yowo.engine import InferenceEngine
from yowo.errors import BackendError, BackendLoadError
from yowo.hardware import get_hardware_profile
from yowo.postprocess._nms import COCO_CLASSES
from yowo.types import BackendType, Detection, Frame

pytestmark = pytest.mark.integration

# The backend this smoke asks for by name. PyTorch is the only one that loads a
# `.pt`; the box asks for "at least one", and claiming more than one backend is
# executed would be false (R:WIDEN).
_SMOKE_BACKEND = BackendType.PYTORCH

# `engine.load` announces a fallback by REWRITING `self._selection` with the
# backend it landed on and this reason prefix. So `selection.backend` alone
# cannot tell "asked for PyTorch" from "fell back to PyTorch" — measured: with
# the override removed, an assertion on `.backend` still passed. The prefix is
# pinned to the source by
# `test_the_fallback_reason_prefix_the_smoke_guards_on_still_exists`, so a
# reword reddens there rather than silently disarming this guard.
_FALLBACK_REASON = "Fallback from"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_engine(torch_available: None, verified_weight: Path) -> Iterator[InferenceEngine]:
    """A real engine pinned to the backend this smoke names.

    The ``backend=`` override is what makes M2 checkable: without it the engine takes
    whatever survives the fallback chain, and asserting on the result tells you
    nothing about which backend produced it.
    """
    with InferenceEngine(weights_path=verified_weight, backend=_SMOKE_BACKEND) as eng:
        yield eng


@pytest.fixture(scope="module")
def real_frame(sample_image_path: Path) -> Frame:
    """bus.jpg — an image with a bus and people in it, not a synthesised blank."""
    pixels = cv2.imread(str(sample_image_path))
    assert pixels is not None, f"cv2 could not decode the sample image at {sample_image_path}"
    return Frame(pixels=pixels, source_id="real-backend-smoke", frame_index=0)


@pytest.fixture(scope="module")
def black_frame() -> Frame:
    """The input the old smoke used, kept only as the control."""
    return Frame(
        pixels=np.zeros((640, 640, 3), dtype=np.uint8),
        source_id="real-backend-smoke-control",
        frame_index=0,
    )


def _boxes(results: list[Detection]) -> list[object]:
    return [box for result in results for box in result.boxes]


# ---------------------------------------------------------------------------
# The smoke
# ---------------------------------------------------------------------------


def test_the_smoke_executes_the_backend_it_asked_for(
    smoke_engine: InferenceEngine, real_frame: Frame
) -> None:
    """covers: M2, A2 — the result carries the backend named, not a fallback's leftovers."""
    selection = smoke_engine.selection
    assert selection.backend is _SMOKE_BACKEND, (
        f"asked for {_SMOKE_BACKEND.value} by override, engine resolved "
        f"{selection.backend.value} (reason: {selection.reason})"
    )
    assert not selection.reason.startswith(_FALLBACK_REASON), (
        f"the smoke reached {selection.backend.value} by FALLBACK ({selection.reason!r}), "
        "not by asking for it. The backend under test must be the one named, or this "
        "smoke regresses against whatever happens to survive."
    )
    results = smoke_engine.detect([real_frame])
    assert results, "detect() returned no results for a single frame"
    for result in results:
        assert result.backend is _SMOKE_BACKEND, (
            f"the smoke asked for {_SMOKE_BACKEND.value}; the detection reports "
            f"{result.backend.value}, so the executed backend was not the one under test"
        )


def test_a_real_image_yields_boxes(smoke_engine: InferenceEngine, real_frame: Frame) -> None:
    """covers: M1, A3, A4, E4, R:VACUOUS — a backend returning nothing fails here."""
    boxes = _boxes(smoke_engine.detect([real_frame]))
    assert len(boxes) >= 1, (
        f"{_SMOKE_BACKEND.value} found nothing in an image containing a bus and people. "
        "Zero boxes here is a broken backend, not an empty scene — which is exactly what "
        "the old black-frame smoke could never tell you."
    )


def test_every_box_satisfies_its_invariants(
    smoke_engine: InferenceEngine, real_frame: Frame
) -> None:
    """covers: M1, A6, E5 — each failure names the box, the invariant and the value."""
    boxes = _boxes(smoke_engine.detect([real_frame]))
    assert boxes, "no boxes to check invariants against"
    for i, box in enumerate(boxes):
        where = f"{_SMOKE_BACKEND.value} box {i}"
        assert box.x1 < box.x2, f"{where}: x1 must be < x2, got x1={box.x1} x2={box.x2}"
        assert box.y1 < box.y2, f"{where}: y1 must be < y2, got y1={box.y1} y2={box.y2}"
        assert 0.0 <= box.confidence <= 1.0, (
            f"{where}: confidence must be in [0, 1], got {box.confidence}"
        )
        assert box.class_name in COCO_CLASSES, (
            f"{where}: class_name {box.class_name!r} (id {box.class_id}) is not a COCO class"
        )


def test_a_black_frame_does_not_satisfy_this_smoke(
    smoke_engine: InferenceEngine, black_frame: Frame
) -> None:
    """covers: R:VACUOUS, A4 — the control.

    The frame the old smoke ran on must FAIL the assertion the new smoke makes.
    If a blank image also produced boxes, `test_a_real_image_yields_boxes` would
    be satisfiable without a working backend and would prove no more than
    `isinstance(result, list)` did.
    """
    boxes = _boxes(smoke_engine.detect([black_frame]))
    assert len(boxes) == 0, (
        f"a 640x640 black frame produced {len(boxes)} box(es). The real-image assertion is "
        "then not discriminating, and this smoke needs a harder input."
    )


def test_a_backend_that_cannot_load_the_weight_says_so(verified_weight: Path) -> None:
    """covers: E6 — a load failure is attributed to the backend, not seen as an empty result."""
    onnx = pytest.importorskip(
        "yowo.backends._onnx", reason="ONNX backend module unavailable"
    )
    backend = onnx.OnnxBackend(get_hardware_profile())
    with pytest.raises((BackendLoadError, BackendError)) as excinfo:
        backend.load(verified_weight)
    assert "onnx" in str(excinfo.value).lower(), (
        f"a load failure must name the backend that failed; got {excinfo.value!r}"
    )


def test_the_default_selection_falls_back_on_a_pt_weight(
    torch_available: None, verified_weight: Path
) -> None:
    """covers: M3, A14, A15, A16, A17, A18, E3 — a tripwire, not a bug report.

    Today the backend the DEFAULT selection runs is not the backend it picked:
    `select_backend` prefers ONNX, `OnnxBackend.load` cannot read a `.pt`, and
    the engine falls back. Read from the resolved selection rather than the log
    line, because a log line is not a contract.

    If this goes RED you may have FIXED something — a backend that loads a `.pt`
    is now selected first, or the selector stopped offering one that cannot.
    Look before reverting.
    """
    primary = select_backend(get_hardware_profile(), model_size="n").backend
    if primary is _SMOKE_BACKEND:
        pytest.skip(
            f"this host selects {primary.value} first, which loads a .pt — there is no "
            "fallback to observe. Host difference, not a defect."
        )
    with InferenceEngine(weights_path=verified_weight) as eng:
        ran = eng.selection.backend
    assert ran is not primary, (
        f"the selector picked {primary.value} and it loaded the weight after all. "
        "That is a real change in behaviour — look at it before changing this check."
    )
