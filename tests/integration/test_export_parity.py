"""PyTorch and ONNX must agree, per variant, on coordinates AND confidence.

`test_backend_conformance.py` binds coordinates for ONE variant (yolo26n) and
binds confidence NOWHERE. Nine of ten exports and the entire confidence axis
have never been checked by CI.

Both bounds are declared BEFORE the run and are not moved to fit the result:

    COORD_TOLERANCE_PX = 1e-3     absolute, on box coordinates
    CONF_TOLERANCE     = 1e-4     absolute, on confidence

Measured 2026-09-15, all ten variants, provider pinned to
`CPUExecutionProvider`, bus.jpg, fp32, 5 detections each:

    coordinates   9.155e-05 .. 3.0899e-04 px   worst case 3.2x inside the bound
    confidence    1.8e-07   .. 2.09e-06        worst case  48x inside the bound

Those figures are DOCUMENTATION, not assertions. A number measured on one
machine is not a property of the code (Q11), and the first version of the
conformance suite pinned one and CI refuted it. The only assertion is against
the declared bound.

YOLO26 deviates about twice as much as YOLO11 on coordinates at every size.
That is a real family difference rather than noise, and every variant is still
comfortably inside one shared bound — so the per-variant tolerance the
milestone's wording implies stays unnecessary.

The provider is pinned AND read back from the live session. An unpinned
provider measures the runtime, not the export: `_select_providers` once served
an explicit `device="cpu"` from the CoreML EP, which computes in FP16, and that
alone moved the measured deviation from 0.00012207 px to 0.33050537 px.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import pytest

from yowo import InferenceEngine
from yowo.types import (
    BackendType,
    ExportFormat,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
)

pytestmark = pytest.mark.integration

#: Absolute bound on box coordinates, in pixels. From m3 box 3.
COORD_TOLERANCE_PX = 1e-3

#: Absolute bound on detection confidence. From m3 box 3. Never before checked.
CONF_TOLERANCE = 1e-4

#: The provider the comparison is pinned to, and must confirm it ran on.
PINNED_PROVIDER = "CPUExecutionProvider"

#: Documentation of what was measured, NOT asserted. See the module docstring.
OBSERVED_COORD_RANGE_PX = (9.155e-05, 3.0899e-04)
OBSERVED_CONF_RANGE = (1.8e-07, 2.09e-06)

#: The pull-request subset. "n/s" spans BOTH families: read as one, a whole
#: architecture would go unchecked on every pull request.
PR_VARIANTS = (
    (ModelFamily.YOLO11, ModelSize.NANO),
    (ModelFamily.YOLO11, ModelSize.SMALL),
    (ModelFamily.YOLO26, ModelSize.NANO),
    (ModelFamily.YOLO26, ModelSize.SMALL),
)
PR_VARIANT_IDS = [f"{f.value}{s.value}" for f, s in PR_VARIANTS]


@dataclasses.dataclass
class _FakeBox:
    """A box-shaped stand-in, so the rules can be exercised without an export."""

    confidence: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0
    class_id: int = 0


class _Box(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


def variants_for_scope(scope: str) -> tuple[tuple[ModelFamily, ModelSize], ...]:
    """Which variants an entry point runs. The ONLY thing the split changes.

    Both entry points share every assertion; a split that also changed what was
    asserted would make the scheduled run and the pull-request run two
    different checks wearing one name.
    """
    if scope == "pr":
        return PR_VARIANTS
    if scope == "all":
        return tuple(
            (family, size)
            for family in (ModelFamily.YOLO11, ModelFamily.YOLO26)
            for size in ModelSize
        )
    msg = f"unknown parity scope {scope!r}; expected 'pr' or 'all'"
    raise ValueError(msg)


def deviation_message(variant: str, axis: str, value: float, bound: float, unit: str = "") -> str:
    """Name the variant, the axis, the value and the bound.

    A bare `assert dev <= 1e-3` tells the reader neither which export moved nor
    by how much, so they cannot tell a regressed exporter from a bound that was
    always too tight.
    """
    return (
        f"{variant}: PyTorch vs ONNX max {axis} deviation {value:.8f}{unit} "
        f"exceeds the declared bound of {bound}{unit}. Provider pinned to "
        f"{PINNED_PROVIDER}. Either the export path regressed for this variant, "
        f"or the bound was never right — do not widen it to make this pass."
    )


def provider_failure(providers: tuple[str, ...], variant: str) -> str | None:
    """Why these numbers cannot be trusted to the pinned provider, or None.

    A pure verdict so the empty case can be exercised without standing up a
    session — `active_providers` returns `()` when none is bound, and `()` must
    read as "could not confirm", never as confirmed.
    """
    if not providers:
        return (
            f"{variant}: the ONNX session reported no execution providers. That is "
            f"'could not confirm', which must never read as confirmed."
        )
    if providers[0] != PINNED_PROVIDER:
        return (
            f"{variant}: the numbers just compared were produced by {providers[0]!r}, "
            f"not the pinned {PINNED_PROVIDER!r} (session reported {providers}). The "
            f"comparison measured that runtime, not this export."
        )
    return None


def by_confidence(boxes: list[_Box]) -> list[_Box]:
    """Pair by confidence, not by position: output order is not part of the contract."""
    return sorted(boxes, key=lambda b: -b.confidence)


def paired(pytorch: list[_Box], onnx: list[_Box], variant: str) -> list[tuple[_Box, _Box]]:
    """Pair the two backends' detections, refusing to compare unequal counts.

    `zip` truncates to the shorter list without a word, so a backend that
    dropped a detection would be compared only on the ones it kept — and pass.
    """
    if len(pytorch) != len(onnx):
        msg = (
            f"{variant}: PyTorch returned {len(pytorch)} detections and ONNX "
            f"returned {len(onnx)}. A count mismatch is a parity failure in its "
            f"own right; pairing them would silently compare only the overlap."
        )
        raise AssertionError(msg)
    if not pytorch:
        msg = (
            f"{variant}: both backends returned zero detections. Agreement over "
            f"an empty set proves nothing — this is a broken fixture, not parity."
        )
        raise AssertionError(msg)
    return list(zip(by_confidence(pytorch), by_confidence(onnx)))


@pytest.fixture(scope="module")
def sample_frame(sample_image_path: Path) -> Frame:
    pixels = cv2.imread(str(sample_image_path))
    assert pixels is not None, f"cv2 could not decode {sample_image_path}"
    return Frame(pixels=pixels, source_id="parity", frame_index=0)


def _run(
    weights: Path,
    backend: BackendType,
    family: ModelFamily,
    size: ModelSize,
    frame: Frame,
) -> tuple[list[_Box], tuple[str, ...]]:
    """Drive one backend through the real engine; return its boxes and providers.

    The providers are read from the engine's own backend rather than from a
    separately-constructed session: the execution provider that matters is the
    one that produced THESE numbers, and confirming a different session proves
    nothing about them.
    """
    with InferenceEngine(
        weights_path=weights,
        backend=backend,
        model_family=family,
        model_size=size,
        device="cpu",
        precision=Precision.FP32,
    ) as engine:
        assert engine.selection.backend is backend, (
            f"asked for {backend.value}, ran {engine.selection.backend.value} "
            f"({engine.selection.reason!r}) — a fallback would compare a backend "
            f"against itself"
        )
        boxes = [box for result in engine.detect([frame]) for box in result.boxes]
        providers = tuple(getattr(engine._backend, "active_providers", ()))
    return boxes, providers


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise from the scope the entry point asked for.

    Both entry points reach the same checks; only the variant list differs.
    Default is "pr", so a bare `pytest` run does the four-variant version.
    """
    if "measured" not in metafunc.fixturenames:
        return
    scope = metafunc.config.getoption("--parity-scope")
    variants = variants_for_scope(scope)
    metafunc.parametrize(
        "measured",
        variants,
        ids=[f"{family.value}{size.value}" for family, size in variants],
        indirect=True,
        scope="module",
    )


@pytest.fixture(scope="module")
def measured(request: pytest.FixtureRequest, sample_frame: Frame) -> Iterator[dict[str, object]]:
    """Export one variant and run both backends over the same frame, once."""
    from yowo.models._weights import resolve_weights

    family, size = request.param
    variant = f"{family.value}{size.value}"
    spec = ModelSpec(family=family, size=size)
    try:
        weights = resolve_weights(spec)
    except Exception as exc:
        from tests.integration.conftest import _unavailable

        _unavailable(f"weight {variant}", f"{type(exc).__name__}: {exc}")
        raise

    from yowo.export import export_model

    with tempfile.TemporaryDirectory() as tmp:
        onnx_dir = Path(tmp) / "onnx"
        export_model(spec, ExportFormat.ONNX, onnx_dir, precision=Precision.FP32, imgsz=640)
        onnx_path = next(onnx_dir.glob("*.onnx"))
        pytorch_boxes, _ = _run(weights, BackendType.PYTORCH, family, size, sample_frame)
        onnx_boxes, providers = _run(onnx_path, BackendType.ONNX, family, size, sample_frame)

    yield {
        "variant": variant,
        "pytorch": pytorch_boxes,
        "onnx": onnx_boxes,
        "providers": providers,
    }


def test_pytorch_and_onnx_agree_on_coordinates(measured: dict[str, object]) -> None:
    """covers: M1,A3,S1,S2 — 1e-3 px, declared before the run."""
    variant = str(measured["variant"])
    pairs = paired(measured["pytorch"], measured["onnx"], variant)  # type: ignore[arg-type]
    deviation = max(
        max(abs(a.x1 - b.x1), abs(a.y1 - b.y1), abs(a.x2 - b.x2), abs(a.y2 - b.y2))
        for a, b in pairs
    )
    assert deviation <= COORD_TOLERANCE_PX, deviation_message(
        variant, "box-coordinate", deviation, COORD_TOLERANCE_PX, " px"
    )


def test_pytorch_and_onnx_agree_on_confidence(measured: dict[str, object]) -> None:
    """covers: M1,A3,S1,S2 — 1e-4, an axis nothing in CI had ever checked."""
    variant = str(measured["variant"])
    pairs = paired(measured["pytorch"], measured["onnx"], variant)  # type: ignore[arg-type]
    deviation = max(abs(a.confidence - b.confidence) for a, b in pairs)
    assert deviation <= CONF_TOLERANCE, deviation_message(
        variant, "confidence", deviation, CONF_TOLERANCE
    )


def test_the_compared_numbers_came_from_the_pinned_provider(
    measured: dict[str, object],
) -> None:
    """covers: M2,R:UNPINNED_PROVIDER,E2,S1 — read from the live session.

    An unpinned provider measures the runtime, not the export. Serving an
    explicit `device="cpu"` from the CoreML EP, which computes in FP16, moved
    the measured deviation from 0.00012207 px to 0.33050537 px — and nothing
    about the exported graph had changed.
    """
    failure = provider_failure(tuple(measured["providers"]), str(measured["variant"]))  # type: ignore[arg-type]
    assert failure is None, failure


# ---------------------------------------------------------------------------
# The rules themselves, exercised without needing an export
# ---------------------------------------------------------------------------


def test_the_declared_bounds_are_the_numbers_the_box_states() -> None:
    """covers: M1,A3,R:SELF_RATIFYING_BOUND.

    The bounds come from the milestone, not from what the run happened to
    measure. Binding them to a check makes widening one a visible diff against
    an assertion rather than a quiet edit to a constant.
    """
    assert COORD_TOLERANCE_PX == 1e-3
    assert CONF_TOLERANCE == 1e-4
    low, high = OBSERVED_COORD_RANGE_PX
    assert high < COORD_TOLERANCE_PX, (
        "the recorded observation is outside the declared bound, so one of them "
        "is wrong — and it is not the bound that moves"
    )
    conf_low, conf_high = OBSERVED_CONF_RANGE
    assert conf_high < CONF_TOLERANCE
    assert low <= high and conf_low <= conf_high


def test_a_count_mismatch_fails_before_any_pairing() -> None:
    """covers: A2,R:COUNT_BLIND,E1.

    Measured 2026-09-15: every one of the ten variants returns 5 and 5, so this
    guard is inert today — which is exactly when it is cheapest to get wrong.
    """
    three = [_FakeBox(confidence=c) for c in (0.9, 0.8, 0.7)]
    two = [_FakeBox(confidence=c) for c in (0.9, 0.8)]
    with pytest.raises(AssertionError, match="3 detections and ONNX returned 2"):
        paired(three, two, "yolo11n")  # type: ignore[arg-type]


def test_zero_detections_on_both_sides_is_not_agreement() -> None:
    """covers: E6 — agreement over an empty set proves nothing."""
    with pytest.raises(AssertionError, match="zero detections"):
        paired([], [], "yolo11n")


def test_detections_are_paired_by_confidence_not_position() -> None:
    """covers: A5 — output order is not part of the contract."""
    ascending = [_FakeBox(confidence=c) for c in (0.1, 0.5, 0.9)]
    descending = [_FakeBox(confidence=c) for c in (0.9, 0.5, 0.1)]
    pairs = paired(ascending, descending, "yolo11n")  # type: ignore[arg-type]
    assert [a.confidence for a, _ in pairs] == [0.9, 0.5, 0.1]
    assert all(a.confidence == b.confidence for a, b in pairs), (
        "pairing by position would have compared 0.1 against 0.9"
    )


def test_an_empty_provider_list_is_not_a_pass() -> None:
    """covers: A4,E3 — `()` means could-not-confirm."""
    failure = provider_failure((), "yolo11n")
    assert failure is not None
    assert "could not confirm" in failure


def test_a_disagreement_names_the_variant_axis_value_and_bound() -> None:
    """covers: M3,A6 — the reader must not have to open the source."""
    message = deviation_message("yolo26n", "box-coordinate", 0.33050537, COORD_TOLERANCE_PX, " px")
    assert "yolo26n" in message
    assert "box-coordinate" in message
    assert "0.33050537" in message
    assert str(COORD_TOLERANCE_PX) in message
    assert PINNED_PROVIDER in message, (
        "a deviation report that omits which provider ran invites the reader to "
        "blame the exporter for a runtime difference"
    )

    wrong = provider_failure(("CoreMLExecutionProvider", "CPUExecutionProvider"), "yolo11n")
    assert wrong is not None and "CoreMLExecutionProvider" in wrong
