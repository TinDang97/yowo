"""`src/yowo/arch/` must still compute what ultralytics computes.

The native architecture was validated against ultralytics ONCE, by a harness in
`tmp/` that no longer exists (confirmed absent 2026-09-11). Since then nothing
in the repository could tell whether the reimplementation still matches its
reference — the invariants in `CLAUDE.md` (`Conv.bn eps=1e-3 momentum=0.03`,
`SPPF.cv1 act=not shortcut`, `Attention.qkv act=False`) are prose, and prose
does not fail a build.

This runs the oracle. Both sides are handed the SAME weight file and the same
tensor; the comparison is against what ultralytics computes now, not against a
number recorded when someone last looked.

Two bounds, one per axis, declared BEFORE the run:

    BOX_TOLERANCE_PX = 5e-3     absolute, on box geometry in pixels
    CLASS_TOLERANCE  = 1e-4     absolute, on class score / confidence

They are separate because the axes differ by two orders of magnitude, and one
bound over both would be set entirely by the looser. Measured 2026-09-16 on
bus.jpg, fp32, same weight file both sides, on TWO platforms:

    YOLO11 raw, every one of 8400 anchors
        box     darwin/arm64    5.6458e-04 .. 1.4038e-03 px  (worst: yolo11l)
                ubuntu/x86-64   4.2725e-04 .. 2.2583e-03 px  (worst: yolo11x)
        class   darwin/arm64    1.3188e-06 .. 3.5942e-05
                ubuntu/x86-64   2.3842e-07 .. 1.3351e-05
    YOLO26 end2end, real detections only
        box     3.0518e-05 .. 6.1035e-05 px   both platforms
        conf    1.1921e-07 .. 2.7418e-06      both platforms

The second row is why this suite prints its margins on every run. The first
version of this docstring recorded only the darwin figures and called the box
headroom 3.6x. CI measured 2.2583e-03 on x86-64 within the hour — 1.6x larger
than the worst number darwin had produced, leaving the real headroom at 2.2x.
Nothing was wrong with the code; a one-machine measurement was being written
down as a property of it (Q11).

Note also that the worst VARIANT is not the same variant on the two platforms:
yolo11l on darwin, yolo11x on x86-64. Per-variant bounds would have encoded one
machine's ranking and reddened on the other.

RESIDUAL, stated rather than hidden: two platforms have been sampled, and the
box bound now carries 2.2x headroom rather than the 3.6x first claimed. A third
runner could plausibly exceed it without any architectural drift. If that
happens, the fix is to record the new measurement here and reconsider the bound
DELIBERATELY — not to widen the constant so the build goes green.

**A single 1e-3 bound — the one `test_export_parity.py` uses — would FAIL
yolo11l at 1.4038e-03.** That is the trap for anyone who reuses the parity
suite's constant here. The two suites measure different things: parity measures
one exporter against its own source model, this measures two independent
implementations of the same paper against each other.

Those figures are DOCUMENTATION, never assertions. A number measured on one
machine is not a property of the code (Q11), and CI has already refuted one
pinned per-machine figure in this repository.

The two families are paired differently because their outputs are shaped
differently. YOLO11 emits `(1, 84, 8400)` pre-NMS in anchor order, so position
pairing is exact and every anchor is compared. YOLO26 emits `(1, 300, 6)`
end2end, whose tail below the confidence floor is junk whose ordering is not
part of any contract: position-pairing that full list on a noise image reports a
max deviation of 6.479e+02, which is not a defect but rows reordering. The same
comparison restricted to real detections gives 3.0518e-05. Pairing is the whole
difference between a phantom bug and a measurement.

The input tensor is built here with plain cv2/numpy rather than through
`yowo`'s own preprocessing. That is deliberate: routing the input through
`yowo.io` would put yowo code on BOTH sides of the comparison, and a
preprocessing bug would then cancel itself out and read as agreement. The
subject of this suite is the model, so only the model differs between the sides.

ultralytics is a DEV dependency (AGPL-3.0, `python_version >= '3.11'`) and the
oracle here runs in CI only — nothing on the inference path imports it. Note
that the package is not import-free of it: `yowo.benchmark._comparison` lazily
imports `ultralytics.YOLO` inside `run_ultralytics_benchmark`, which
`yowo.benchmark.__init__` re-exports. That is a pre-existing Apache-2.0 /
AGPL-3.0 question this suite neither creates nor resolves; it is recorded here
because the sentence "yowo ships no import of ultralytics" is false and was
about to be written down as true.

Authored under ADD task `arch-equivalence-in-ci`.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
import torch

from yowo.arch import build_model, load_weights
from yowo.models._registry import list_available
from yowo.models._weights import resolve_weights
from yowo.types import ModelFamily, ModelSize, ModelSpec

pytestmark = pytest.mark.integration

#: Absolute bound on box geometry, in pixels. Worst measured across both
#: sampled platforms: 2.2583e-03 (yolo11x, ubuntu/x86-64) — 2.2x headroom.
BOX_TOLERANCE_PX = 5e-3

#: Absolute bound on class score / confidence. Worst measured across both
#: sampled platforms: 3.5942e-05 (yolo11x, darwin/arm64) — 2.8x headroom.
CLASS_TOLERANCE = 1e-4

#: Detections below this are the junk tail of a fixed-length top-300 list.
YOLO26_CONF_FLOOR = 0.25

#: Documentation of what was measured. NOT asserted — see the module docstring.
#: Spans BOTH sampled platforms — see the docstring. A range recorded from one
#: machine understated the box maximum by 1.6x.
OBSERVED_YOLO11_BOX_RANGE_PX = (4.2725e-04, 2.2583e-03)
OBSERVED_YOLO11_CLASS_RANGE = (2.3842e-07, 3.5942e-05)
OBSERVED_YOLO26_BOX_RANGE_PX = (3.0518e-05, 6.1035e-05)
OBSERVED_YOLO26_CONF_RANGE = (1.1921e-07, 2.7418e-06)

#: The smallest headroom a declared bound must keep over the worst measurement,
#: and the largest it may keep. The ceiling is what makes widening a bound to
#: admit a fresh deviation a visible diff against an assertion rather than a
#: one-character edit (R:BOUND_FITTED_TO_RESULT).
MIN_HEADROOM = 2.0
MAX_HEADROOM = 10.0

VARIANTS = tuple(
    (family, size) for family in (ModelFamily.YOLO11, ModelFamily.YOLO26) for size in ModelSize
)
YOLO11_VARIANTS = tuple(v for v in VARIANTS if v[0] is ModelFamily.YOLO11)
YOLO26_VARIANTS = tuple(v for v in VARIANTS if v[0] is ModelFamily.YOLO26)


def variant_id(family: ModelFamily, size: ModelSize) -> str:
    """`yolo11n`, `yolo26x` — the spelling the weights and the CHECKS block use."""
    return f"{family.value}{size.value}"


YOLO11_IDS = [variant_id(*v) for v in YOLO11_VARIANTS]
YOLO26_IDS = [variant_id(*v) for v in YOLO26_VARIANTS]


def absent_oracle_is_fatal(ci_value: str | None) -> bool:
    """Whether an absent oracle must fail rather than skip.

    A pure verdict so the CI branch can be exercised without an absent
    ultralytics. In CI an unobtainable oracle means the run proved nothing, and
    a skip is green (Q3) — indistinguishable from equivalence that was checked.
    """
    return (ci_value or "").lower() in ("1", "true", "yes")


def _require_oracle() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by the CI branch
        detail = (
            f"ultralytics is unavailable ({exc}). It is a dev dependency gated on "
            f"python_version >= '3.11'; without it NOTHING here is compared and "
            f"reporting that as green is the failure this suite exists to prevent."
        )
        if absent_oracle_is_fatal(os.environ.get("CI")):
            raise RuntimeError(detail) from exc
        pytest.skip(detail)


def deviation_message(variant: str, axis: str, value: float, bound: float, unit: str = "") -> str:
    """Name the variant, the axis, the value, the bound and the oracle version.

    Whoever reads this failure is deciding whether the architecture drifted or
    the oracle moved under it, and cannot tell those apart without the version.
    """
    try:
        import ultralytics

        version = ultralytics.__version__
    except Exception:  # pragma: no cover - the message must never be the failure
        version = "unknown"
    return (
        f"{variant}: native arch vs ultralytics {version} max {axis} deviation "
        f"{value:.8e}{unit} exceeds the declared bound of {bound}{unit}. Either "
        f"`src/yowo/arch/` drifted from the reference, or ultralytics changed the "
        f"computation — do not widen the bound to make this pass."
    )


def require_same_shape(oracle: torch.Tensor, native: torch.Tensor, variant: str) -> None:
    """A shape mismatch is structural, not a deviation.

    Broadcasting two differently-shaped outputs into a comparison produces a
    number, and a number looks like a measurement. It is not one.
    """
    if tuple(oracle.shape) != tuple(native.shape):
        msg = (
            f"{variant}: the oracle returned {tuple(oracle.shape)} and the native "
            f"model returned {tuple(native.shape)}. That is a structural mismatch, "
            f"not a numeric deviation, and must not be coerced into a comparison."
        )
        raise AssertionError(msg)


def paired_end2end(
    oracle: torch.Tensor, native: torch.Tensor, variant: str, floor: float = YOLO26_CONF_FLOOR
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real detections from both sides, confidence-ranked, refusing what proves nothing.

    Takes `(300, 6)` rows `[x1, y1, x2, y2, conf, cls]` from each side. Rows below
    the floor are the fixed-length list's junk tail; their ORDER is not part of
    any contract, so comparing them measures sorting, not architecture.
    """
    keep_o = oracle[oracle[:, 4] >= floor]
    keep_n = native[native[:, 4] >= floor]
    if len(keep_o) == 0 or len(keep_n) == 0:
        msg = (
            f"{variant}: {len(keep_o)} oracle and {len(keep_n)} native detections "
            f"above the {floor} confidence floor. Agreement over an empty set "
            f"proves nothing, so this is a failure, not a pass."
        )
        raise AssertionError(msg)
    if len(keep_o) != len(keep_n):
        msg = (
            f"{variant}: the oracle found {len(keep_o)} detections above {floor} and "
            f"the native model found {len(keep_n)}. Unequal counts cannot be paired; "
            f"truncating to the shorter would compare a subset and call it agreement."
        )
        raise AssertionError(msg)
    order_o = torch.argsort(keep_o[:, 4], descending=True)
    order_n = torch.argsort(keep_n[:, 4], descending=True)
    return keep_o[order_o], keep_n[order_n]


@pytest.fixture(scope="session")
def input_tensor(sample_image_path: Path) -> torch.Tensor:
    """One real image, letterbox-free 640x640, fp32 — identical for both sides.

    A real image and not noise: on noise, YOLO26 produces no detection above the
    floor, and an empty comparison is what `paired_end2end` refuses. The image
    arrives through the tier's digest-pinned fixture, so this suite inherits the
    same fail-in-CI-skip-locally rule as every other integration test.
    """
    image = cv2.imread(str(sample_image_path))
    if image is None:  # pragma: no cover - a corrupt cache, not a code path
        pytest.fail(f"could not decode {sample_image_path}")
    resized = cv2.resize(image, (640, 640))[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
    return torch.from_numpy(chw)[None]


class _Comparison:
    """Both sides' raw output for one variant, plus the weight each was handed."""

    def __init__(self, oracle: torch.Tensor, native: torch.Tensor, weight: Path) -> None:
        self.oracle = oracle
        self.native = native
        self.weight = weight


_CACHE: dict[str, _Comparison] = {}

#: variant -> axis -> deviation, reported at teardown. See `_report_margins`.
_MARGINS: dict[str, dict[str, float]] = {}


def _compare(family: ModelFamily, size: ModelSize, tensor: torch.Tensor) -> _Comparison:
    """Run both implementations on the same bytes. Cached: each weight loads once."""
    name = variant_id(family, size)
    if name in _CACHE:
        return _CACHE[name]
    _require_oracle()
    from ultralytics import YOLO

    weight = resolve_weights(ModelSpec(family=family, size=size))

    oracle_model = YOLO(str(weight)).model.float().eval()
    with torch.no_grad():
        oracle_out = oracle_model(tensor)
    oracle_out = oracle_out[0] if isinstance(oracle_out, (list, tuple)) else oracle_out

    native_model = build_model(family, size)
    load_weights(native_model, weight)
    native_model.eval()
    with torch.no_grad():
        native_out = native_model(tensor)
    native_out = native_out[0] if isinstance(native_out, (list, tuple)) else native_out

    require_same_shape(oracle_out, native_out, name)
    result = _Comparison(oracle_out.detach(), native_out.detach(), Path(weight))
    _CACHE[name] = result
    return result


@pytest.fixture(scope="session", autouse=True)
def _report_margins() -> Iterator[None]:
    """Print what this machine actually measured, then drop the cached tensors.

    The bounds assert only that the deviation is inside them. Without this, a
    green run on a runner says the bound held and nothing about the margin —
    and the headroom figures in the module docstring are darwin/arm64 numbers
    (Q11: a measurement from one machine is not a property of the code). The
    mAP gate prints for the same reason; both CI steps run with `-s`.
    """
    yield
    if _MARGINS:
        print("\n--- arch equivalence, measured on this machine ---")
        print(f"{'variant':10} {'box (px)':>13} {'/ bound':>9} {'class':>13} {'/ bound':>9}")
        for name in sorted(_MARGINS):
            axes = _MARGINS[name]
            box, cls = axes.get("box"), axes.get("class")
            box_s = f"{box:.4e}" if box is not None else "-"
            box_p = f"{box / BOX_TOLERANCE_PX:6.1%}" if box is not None else "-"
            cls_s = f"{cls:.4e}" if cls is not None else "-"
            cls_p = f"{cls / CLASS_TOLERANCE:6.1%}" if cls is not None else "-"
            print(f"{name:10} {box_s:>13} {box_p:>9} {cls_s:>13} {cls_p:>9}")
    _CACHE.clear()


@pytest.mark.parametrize(("family", "size"), YOLO11_VARIANTS, ids=YOLO11_IDS)
def test_yolo11_box_geometry_matches_the_oracle(
    family: ModelFamily, size: ModelSize, input_tensor: torch.Tensor
) -> None:
    """Every one of the 8400 anchors, position-paired — the layout permits it."""
    name = variant_id(family, size)
    c = _compare(family, size, input_tensor)
    deviation = (c.oracle[:, :4] - c.native[:, :4]).abs().max().item()
    _MARGINS.setdefault(name, {})["box"] = deviation
    assert deviation <= BOX_TOLERANCE_PX, deviation_message(
        name, "box geometry", deviation, BOX_TOLERANCE_PX, " px"
    )


@pytest.mark.parametrize(("family", "size"), YOLO11_VARIANTS, ids=YOLO11_IDS)
def test_yolo11_class_scores_match_the_oracle(
    family: ModelFamily, size: ModelSize, input_tensor: torch.Tensor
) -> None:
    """The tighter axis, bounded separately — 80 class rows over every anchor."""
    name = variant_id(family, size)
    c = _compare(family, size, input_tensor)
    deviation = (c.oracle[:, 4:] - c.native[:, 4:]).abs().max().item()
    _MARGINS.setdefault(name, {})["class"] = deviation
    assert deviation <= CLASS_TOLERANCE, deviation_message(
        name, "class score", deviation, CLASS_TOLERANCE
    )


@pytest.mark.parametrize(("family", "size"), YOLO26_VARIANTS, ids=YOLO26_IDS)
def test_yolo26_detections_match_the_oracle(
    family: ModelFamily, size: ModelSize, input_tensor: torch.Tensor
) -> None:
    """End2end output, confidence-ranked. Both axes, since both live in one tensor."""
    name = variant_id(family, size)
    c = _compare(family, size, input_tensor)
    oracle, native = paired_end2end(c.oracle[0], c.native[0], name)

    box = (oracle[:, :4] - native[:, :4]).abs().max().item()
    _MARGINS.setdefault(name, {})["box"] = box
    assert box <= BOX_TOLERANCE_PX, deviation_message(
        name, "box geometry", box, BOX_TOLERANCE_PX, " px"
    )
    conf = (oracle[:, 4] - native[:, 4]).abs().max().item()
    _MARGINS.setdefault(name, {})["class"] = conf
    assert conf <= CLASS_TOLERANCE, deviation_message(name, "confidence", conf, CLASS_TOLERANCE)


def test_every_registered_variant_is_compared() -> None:
    """A variant absent from the set must fail, not vanish quietly.

    `list_available()` is the registry the shipped package serves. If a variant
    is registered and this suite does not compare it, the equivalence claim has
    a hole exactly the shape of the variant nobody noticed.
    """
    registered = {f"{m.family.value}{m.size.value}" for m in list_available()}
    compared = set(YOLO11_IDS) | set(YOLO26_IDS)
    assert registered == compared, (
        f"registered but never compared: {sorted(registered - compared)}; "
        f"compared but not registered: {sorted(compared - registered)}"
    )


def test_both_sides_are_handed_the_same_weight_file(input_tensor: torch.Tensor) -> None:
    """The oracle runs on the bytes under test — binds R:ORACLE_NOT_RUN.

    An oracle handed a different checkpoint would agree or disagree for reasons
    that have nothing to do with the architecture.
    """
    family, size = YOLO11_VARIANTS[0]
    expected = Path(resolve_weights(ModelSpec(family=family, size=size)))
    c = _compare(family, size, input_tensor)
    assert c.weight == expected, (
        f"the comparison used {c.weight}, but the production resolution path "
        f"serves {expected}. Both sides must be handed the same bytes."
    )
    assert c.weight.exists()


def test_the_declared_bounds_are_the_measured_ones_with_headroom() -> None:
    """A bound is measured first and kept in a band — binds R:BOUND_FITTED_TO_RESULT.

    The floor stops a bound so tight that ordinary float noise reds the build.
    The CEILING is the point: without it, widening a bound to admit whatever the
    run just produced is a one-character edit nobody reviews. With it, widening
    is a diff against an assertion that says what the bound is for.
    """
    worst_box = max(OBSERVED_YOLO11_BOX_RANGE_PX[1], OBSERVED_YOLO26_BOX_RANGE_PX[1])
    worst_class = max(OBSERVED_YOLO11_CLASS_RANGE[1], OBSERVED_YOLO26_CONF_RANGE[1])

    for label, bound, worst in (
        ("BOX_TOLERANCE_PX", BOX_TOLERANCE_PX, worst_box),
        ("CLASS_TOLERANCE", CLASS_TOLERANCE, worst_class),
    ):
        headroom = bound / worst
        assert headroom >= MIN_HEADROOM, (
            f"{label}={bound} leaves only {headroom:.2f}x over the worst measured "
            f"{worst:.4e}; ordinary float noise would red the build."
        )
        assert headroom <= MAX_HEADROOM, (
            f"{label}={bound} is {headroom:.2f}x the worst measured {worst:.4e}. "
            f"A bound that loose stops constraining anything. If a real deviation "
            f"grew, record the new measurement here — do not widen the bound alone."
        )


def test_the_recorded_measurements_are_documentation_not_assertions() -> None:
    """Q11: a number measured on one machine is not a property of the code.

    The OBSERVED_ constants say what one machine saw on one date. They must not
    appear in any comparison against a freshly measured value, or CI becomes a
    referendum on whose CPU reorders floats.
    """
    offenders = []
    for name, obj in list(globals().items()):
        if not name.startswith("test_yolo") or not callable(obj):
            continue
        source = inspect.getsource(obj)
        if "OBSERVED_" in source:
            offenders.append(name)
    assert not offenders, (
        f"{offenders} compare against a recorded measurement. Assert against the "
        f"declared bound; the recorded figures are documentation."
    )


def test_a_shape_mismatch_is_a_structural_failure() -> None:
    """Never coerced into a comparison that yields a plausible-looking number."""
    with pytest.raises(AssertionError, match="structural mismatch"):
        require_same_shape(torch.zeros(1, 84, 8400), torch.zeros(1, 300, 6), "made-up")
    require_same_shape(torch.zeros(1, 84, 8400), torch.zeros(1, 84, 8400), "made-up")


def test_zero_detections_above_the_floor_is_not_agreement() -> None:
    """Agreement over an empty set proves nothing — and unequal counts cannot pair."""
    empty = torch.zeros(300, 6)  # every confidence is 0.0, below the floor
    with pytest.raises(AssertionError, match="proves nothing"):
        paired_end2end(empty, empty, "made-up")

    one = torch.zeros(300, 6)
    one[0, 4] = 0.9
    two = torch.zeros(300, 6)
    two[0, 4] = 0.9
    two[1, 4] = 0.8
    with pytest.raises(AssertionError, match="Unequal counts"):
        paired_end2end(one, two, "made-up")

    ranked_o, ranked_n = paired_end2end(two, two, "made-up")
    assert len(ranked_o) == len(ranked_n) == 2
    assert ranked_o[0, 4] >= ranked_o[1, 4], "pairs must be confidence-ranked, not slice-ordered"


def test_an_absent_oracle_fails_under_ci_rather_than_skipping() -> None:
    """Q3: a skipped test is green — binds R:GREEN_BY_SKIP.

    ultralytics is gated on `python_version >= '3.11'`, so a 3.10 runner would
    otherwise report equivalence it never checked.
    """
    assert absent_oracle_is_fatal("true")
    assert absent_oracle_is_fatal("1")
    assert absent_oracle_is_fatal("YES")
    assert not absent_oracle_is_fatal("")
    assert not absent_oracle_is_fatal(None)
    assert not absent_oracle_is_fatal("false")
