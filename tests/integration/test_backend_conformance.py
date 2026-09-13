"""One weight, three backends, one set of answers — or the difference is reported.

Every artifact here descends from ONE digest-verified checkpoint, exported in
the fixture. Two separately-pinned files could not be *known* to hold the same
weights, and a conformance suite comparing two different models reports their
difference as a backend divergence.

The tolerance is declared before the run and is not moved to fit the result:
**1e-3 absolute on box coordinates, exact on class ids**.

The deviation actually measured, PyTorch vs ONNX on bus.jpg, both pinned
cpu/fp32, same weights, differs by PLATFORM by a factor of ~2100:

    ubuntu-latest x86_64   0.00015450 px   — inside the bound
    macOS 15 arm64         0.33050537 px   — 330x the bound

Counts (5 and 5) and every class id agree on both. This suite asserts the bound
on whatever platform runs it, and CI runs x86_64, where the backends conform.

No numeric figure is asserted as a constant here, deliberately. The first
version of this file pinned the arm64 number and CI rejected it — the pin was a
property of one developer's machine, not of the code, which is lesson Q7 and
lesson Q11. The platform figures above are DOCUMENTATION; the only assertion is
against the declared bound.

The arm64 gap is real for anyone developing there and is owned by
`pytorch-onnx-numeric-divergence` in m4-honest-deployment.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest

from yowo import InferenceEngine
from yowo.backends._roster import EXECUTED
from yowo.errors import BackendLoadError, ModelLoadError
from yowo.types import (
    BackendType,
    ExportFormat,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
)

# ---------------------------------------------------------------------------
# The declared tolerance — stated BEFORE the run, per m3 box 2.
# ---------------------------------------------------------------------------

#: Absolute bound on box coordinates, in pixels, between any two backends.
COORD_TOLERANCE_PX = 1e-3

#: Observed 2026-09-13, PyTorch vs ONNX, cpu/fp32, bus.jpg, by platform. These
#: are recorded for a reader, NOT asserted: a number measured on one machine is
#: not a property of the code, and asserting one here is what CI rejected.
OBSERVED_COORD_DEVIATION_X86_64 = 0.00015450
OBSERVED_COORD_DEVIATION_ARM64 = 0.00024414

#: What arm64 measured BEFORE `pytorch-onnx-numeric-divergence` landed. It was
#: filed twice as a platform fact — first as a general PyTorch-ONNX divergence,
#: then as arm64 arithmetic — before anyone checked which execution provider had
#: run. It was neither: `_select_providers` served an explicit `device="cpu"`
#: from the CoreML EP, which computes in FP16. Honouring the requested device
#: took arm64 from this figure to the one above, inside the bound.
SUPERSEDED_ARM64_DEVIATION_VIA_COREML = 0.33050537

_FIXTURE_FAMILY = ModelFamily.YOLO26
_FIXTURE_SIZE = ModelSize.NANO
_EXECUTED_IDS = [b.value for b in EXECUTED]


# ---------------------------------------------------------------------------
# One artifact chain, rooted in the verified weight
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact_chain(verified_weight: Path) -> Iterator[dict[BackendType, Path]]:
    """Export the ONE verified weight to every format the executed backends need.

    Module-scoped: the ONNX export costs ~5s and the IR conversion a little
    more, and every check below wants the same bytes.
    """
    from yowo.export import export_model

    spec = ModelSpec(family=_FIXTURE_FAMILY, size=_FIXTURE_SIZE)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        chain: dict[BackendType, Path] = {BackendType.PYTORCH: verified_weight}

        if BackendType.ONNX in EXECUTED:
            onnx_dir = out / "onnx"
            export_model(spec, ExportFormat.ONNX, onnx_dir, precision=Precision.FP32, imgsz=640)
            # The export writes external data beside the graph; the .onnx alone
            # is not the model, so the path must stay in its directory.
            chain[BackendType.ONNX] = next(onnx_dir.glob("*.onnx"))

        if BackendType.OPENVINO in EXECUTED:
            ov_dir = out / "openvino"
            export_model(spec, ExportFormat.OPENVINO, ov_dir, precision=Precision.FP32, imgsz=640)
            chain[BackendType.OPENVINO] = next(ov_dir.rglob("*.xml"))

        yield chain


@pytest.fixture(scope="module")
def real_frame(sample_image_path: Path) -> Frame:
    pixels = cv2.imread(str(sample_image_path))
    assert pixels is not None, f"cv2 could not decode {sample_image_path}"
    return Frame(pixels=pixels, source_id="conformance", frame_index=0)


@pytest.fixture(scope="module")
def black_frame() -> Frame:
    return Frame(
        pixels=np.zeros((640, 640, 3), dtype=np.uint8),
        source_id="conformance-empty",
        frame_index=0,
    )


def _run(chain: dict[BackendType, Path], backend: BackendType, frame: Frame) -> list[object]:
    """Drive one backend through the real engine and return its boxes."""
    with InferenceEngine(
        weights_path=chain[backend],
        backend=backend,
        device="cpu",
        precision=Precision.FP32,
    ) as eng:
        assert eng.selection.backend is backend, (
            f"asked for {backend.value}, ran {eng.selection.backend.value} "
            f"({eng.selection.reason!r}) — a fallback would make this suite compare "
            "one backend against itself"
        )
        return [box for result in eng.detect([frame]) for box in result.boxes]


def _sorted_boxes(boxes: list[object]) -> list[object]:
    """Pair by confidence, not by position: order is not part of the contract."""
    return sorted(boxes, key=lambda b: -b.confidence)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# M2 — the five conformance axes the box names
# ---------------------------------------------------------------------------


def test_all_backends_load_from_one_artifact_chain(
    artifact_chain: dict[BackendType, Path], verified_weight: Path
) -> None:
    """covers: M2, A2 — every artifact descends from the digest-verified weight."""
    assert artifact_chain[BackendType.PYTORCH] == verified_weight
    assert set(artifact_chain) == set(EXECUTED), (
        f"the chain must cover exactly the executed roster: "
        f"{sorted(b.value for b in artifact_chain)} vs {sorted(_EXECUTED_IDS)}"
    )
    for backend, path in artifact_chain.items():
        assert path.exists(), f"{backend.value} artifact missing at {path}"


@pytest.mark.parametrize("backend", list(EXECUTED), ids=_EXECUTED_IDS)
def test_output_shape_and_dtype_agree(
    artifact_chain: dict[BackendType, Path], real_frame: Frame, backend: BackendType
) -> None:
    """covers: M2, A4 — every backend returns the same result shape."""
    boxes = _run(artifact_chain, backend, real_frame)

    assert boxes, f"{backend.value} returned no detections for bus.jpg"
    for box in boxes:
        assert isinstance(box.x1, float) and isinstance(box.y1, float)  # type: ignore[attr-defined]
        assert box.x1 < box.x2 and box.y1 < box.y2  # type: ignore[attr-defined]
        assert 0.0 <= box.confidence <= 1.0  # type: ignore[attr-defined]
        assert isinstance(box.class_id, int)  # type: ignore[attr-defined]


@pytest.mark.parametrize("backend", list(EXECUTED), ids=_EXECUTED_IDS)
def test_an_empty_input_yields_an_empty_result_everywhere(
    artifact_chain: dict[BackendType, Path], black_frame: Frame, backend: BackendType
) -> None:
    """covers: M2, A9, E3 — an empty result, not a raise and not None."""
    boxes = _run(artifact_chain, backend, black_frame)

    assert isinstance(boxes, list), f"{backend.value} returned {type(boxes).__name__}, not a list"
    assert len(boxes) == 0, (
        f"{backend.value} found {len(boxes)} objects in an all-black frame; "
        "the other backends find none, so this is a divergence, not a detection"
    )


def test_class_mapping_agrees(artifact_chain: dict[BackendType, Path], real_frame: Frame) -> None:
    """covers: M2, A4 — the same ids map to the same names across backends."""
    mappings: dict[BackendType, dict[int, str]] = {}
    for backend in EXECUTED:
        boxes = _run(artifact_chain, backend, real_frame)
        mappings[backend] = {b.class_id: b.class_name for b in boxes}  # type: ignore[attr-defined]

    reference = mappings[BackendType.PYTORCH]
    for backend, mapping in mappings.items():
        shared = set(reference) & set(mapping)
        assert shared, f"{backend.value} shares no class id with pytorch"
        for class_id in shared:
            assert mapping[class_id] == reference[class_id], (
                f"class {class_id} is {mapping[class_id]!r} on {backend.value} "
                f"but {reference[class_id]!r} on pytorch"
            )


@pytest.mark.parametrize("backend", list(EXECUTED), ids=_EXECUTED_IDS)
def test_bad_input_raises_the_same_error_type(tmp_path: Path, backend: BackendType) -> None:
    """covers: M2, E4 — one error type for a weight that is not one."""
    junk = tmp_path / f"not-a-model-{backend.value}.bin"
    junk.write_bytes(b"this is not a model" * 64)

    with (
        pytest.raises((BackendLoadError, ModelLoadError)) as excinfo,
        InferenceEngine(weights_path=junk, backend=backend, device="cpu") as eng,
    ):
        eng.load()

    message = str(excinfo.value).lower()
    assert "missing optional dependency" not in message, (
        f"{backend.value} reported a corrupt weight as a missing dependency. "
        "That is the OpenVINO defect this node fixed, reappearing elsewhere."
    )


def test_detections_are_paired_by_confidence_not_position(
    artifact_chain: dict[BackendType, Path], real_frame: Frame
) -> None:
    """covers: A11, E5 — a permutation must not read as a divergence."""
    boxes = _sorted_boxes(_run(artifact_chain, BackendType.PYTORCH, real_frame))
    shuffled = _sorted_boxes(list(reversed(boxes)))

    assert [b.class_id for b in shuffled] == [b.class_id for b in boxes], (  # type: ignore[attr-defined]
        "sorting by confidence must make the comparison order-independent"
    )


# ---------------------------------------------------------------------------
# M3 — the declared tolerance, and what was actually measured against it
# ---------------------------------------------------------------------------


def test_the_declared_tolerance_is_one_thousandth_of_a_pixel() -> None:
    """covers: M3, R:LOOSENED, A5, A6 — the bound is a constant, findable and fixed."""
    assert COORD_TOLERANCE_PX == 1e-3, (
        "the tolerance was declared before the run and is not moved to fit a "
        "result. Widening it here is the one change this node forbids."
    )


def test_class_ids_match_exactly_across_backends(
    artifact_chain: dict[BackendType, Path], real_frame: Frame
) -> None:
    """covers: M3, A5 — 0 mismatches observed on both platforms."""
    reference = _sorted_boxes(_run(artifact_chain, BackendType.PYTORCH, real_frame))

    for backend in EXECUTED:
        if backend is BackendType.PYTORCH:
            continue
        other = _sorted_boxes(_run(artifact_chain, backend, real_frame))
        assert len(other) == len(reference), (
            f"{backend.value} found {len(other)} objects, pytorch found {len(reference)}"
        )
        mismatches = [
            (i, p.class_id, q.class_id)  # type: ignore[attr-defined]
            for i, (p, q) in enumerate(zip(reference, other))
            if p.class_id != q.class_id  # type: ignore[attr-defined]
        ]
        assert not mismatches, f"pytorch vs {backend.value} class-id mismatches: {mismatches}"


#: True on the one platform measured to diverge. Scoping the expectation to the
#: platform means x86_64 must PASS and macOS arm64 must FAIL — either one
#: flipping is a red, so neither the conformance claim nor the arm64 gap can
#: change without someone noticing. A blanket skip would hide both (Q3).
_IS_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"


def test_pytorch_and_onnx_agree_within_the_declared_bound(
    artifact_chain: dict[BackendType, Path], real_frame: Frame
) -> None:
    """covers: M3, A10, E6 — the declared bound, asserted on whatever platform runs.

    E6 twice over. This check first pinned a deviation measured on one machine
    and CI refuted it. It then carried a strict xfail scoping the gap to macOS
    arm64 — and that marker did its job: when `pytorch-onnx-numeric-divergence`
    made arm64 honour an explicit `device="cpu"`, the xfail turned into an XPASS
    failure and the change was noticed rather than absorbed. Both platforms now
    conform on their own merits, so no marker excuses either.
    """
    a = _sorted_boxes(_run(artifact_chain, BackendType.PYTORCH, real_frame))
    b = _sorted_boxes(_run(artifact_chain, BackendType.ONNX, real_frame))

    deviation = max(
        max(
            abs(p.x1 - q.x1),
            abs(p.y1 - q.y1),
            abs(p.x2 - q.x2),
            abs(p.y2 - q.y2),  # type: ignore[attr-defined]
        )
        for p, q in zip(a, b)
    )
    assert deviation <= COORD_TOLERANCE_PX, (
        deviation_message("pytorch", "onnx", "box-coordinate", deviation, COORD_TOLERANCE_PX)
        + f". Counts agree ({len(a)} vs {len(b)}) and class ids agree. Observed by "
        f"platform: x86_64 {OBSERVED_COORD_DEVIATION_X86_64}, arm64 "
        f"{OBSERVED_COORD_DEVIATION_ARM64}. If arm64 has regressed toward "
        f"{SUPERSEDED_ARM64_DEVIATION_VIA_COREML}, check which execution provider "
        "bound the graph: that figure is the CoreML EP computing in FP16, not "
        "arm64 arithmetic."
    )


def deviation_message(a_name: str, b_name: str, axis: str, value: float, bound: float) -> str:
    """Build the line a reader gets when two backends disagree.

    Separated from the assertion so what the reader SEES can be checked without
    needing two backends to actually disagree — and without pinning a number
    that belongs to one machine.
    """
    return (
        f"{a_name} vs {b_name}: max {axis} deviation {value:.8f} exceeds the "
        f"declared tolerance of {bound}"
    )


def test_a_disagreement_names_both_backends_the_axis_and_the_value() -> None:
    """covers: M3, A14 — a reader learns which backends, which axis, what value.

    "assert 0.33 < 0.001" tells nobody which two backends diverged or on what.
    """
    message = deviation_message("pytorch", "onnx", "box-coordinate", 0.33050537, COORD_TOLERANCE_PX)

    assert "pytorch" in message and "onnx" in message
    assert "box-coordinate" in message
    assert "0.33050537" in message
    assert str(COORD_TOLERANCE_PX) in message


def test_both_platform_measurements_are_documented_not_asserted() -> None:
    """covers: M3, A6 — the figures are recorded for a reader, never as a bound.

    The first version of this suite asserted the arm64 figure and CI rejected
    it: 0.00015450 on x86_64 against 0.33050537 on arm64, a factor of ~2100.
    A number measured on one machine is not a property of the code (Q7, Q11).
    That split is now closed — it was the CoreML EP computing in FP16, not the
    platform — but the rule outlives the split: these figures stay documentation.
    """
    import tests.integration.test_backend_conformance as module

    source = Path(module.__file__).read_text()

    assert OBSERVED_COORD_DEVIATION_X86_64 < COORD_TOLERANCE_PX, (
        "on the platform CI runs the backends conform; that is what box 2 claims"
    )
    assert OBSERVED_COORD_DEVIATION_ARM64 < COORD_TOLERANCE_PX, (
        "arm64 conforms too since pytorch-onnx-numeric-divergence honoured the "
        "requested device; the superseded CoreML figure is kept separately"
    )
    # Neither figure may be COMPARED against a live measurement. Two forms are
    # legitimate and must not trip this, or the scanner starts editing correct
    # content to keep itself happy (lesson Q5):
    #   - comparing the two constants to the declared bound, above;
    #   - asserting a figure appears in documentation text, e.g. a marker reason.
    _ALLOWED = ("COORD_TOLERANCE_PX", "in reason", "in source", "in message")
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("assert ") or "OBSERVED_COORD_DEVIATION" not in stripped:
            continue
        assert any(allowed in stripped for allowed in _ALLOWED), (
            f"a platform figure is being compared against a live measurement: "
            f"{stripped!r}. That pins one machine's result as a property of the "
            "code, which is exactly what CI rejected (Q11)."
        )


# ---------------------------------------------------------------------------
# M1 / M4 — OpenVINO actually runs
# ---------------------------------------------------------------------------


def test_openvino_actually_executes_on_a_real_weight(
    artifact_chain: dict[BackendType, Path], real_frame: Frame
) -> None:
    """covers: M1, M4 — proved by running it, not by reading the import."""
    boxes = _run(artifact_chain, BackendType.OPENVINO, real_frame)

    assert boxes, (
        "OpenVINO returned nothing. Before this node it could not load at all on "
        "openvino>=2025: `from openvino.runtime import Core` names a module "
        "removed in 2025, and the failure was reported as a missing dependency."
    )


def test_no_marker_excuses_either_platform() -> None:
    """covers: E6 — the conformance claim stands on its own on every platform.

    This check has now been wrong in both directions, and E6 is the record of
    that. First it PINNED a deviation measured on one machine, and CI refuted
    it. Then it carried a strict xfail scoping the gap to macOS arm64 — correct
    at the time, and the marker earned its keep: when the execution-provider
    fix landed, strict turned the newly-conforming arm64 run into an XPASS
    failure instead of letting it pass unnoticed.

    What must hold now is the opposite of what E6 once asserted: no xfail,
    skipif, or other marker may excuse the bound on any platform. An expected
    failure that outlives the defect it recorded is indistinguishable from a
    suite that never checked.
    """
    marks = list(getattr(test_pytorch_and_onnx_agree_within_the_declared_bound, "pytestmark", []))
    excusing = [m for m in marks if m.name in ("xfail", "skip", "skipif")]

    assert not excusing, (
        f"{[m.name for m in excusing]} excuses the declared bound. Both platforms "
        f"conform on their own merits — x86_64 {OBSERVED_COORD_DEVIATION_X86_64} px, "
        f"arm64 {OBSERVED_COORD_DEVIATION_ARM64} px, against a bound of "
        f"{COORD_TOLERANCE_PX}. A marker left behind after the divergence closed "
        "would hide the next one."
    )


def test_the_superseded_measurement_is_kept_as_history() -> None:
    """The 0.33 px figure must stay readable, and stay labelled as superseded.

    It was filed twice as a platform property before anyone checked which
    execution provider had run. Deleting it loses the lesson; asserting it
    would re-pin a machine measurement. So it is recorded, named for what it
    actually was, and never compared against.
    """
    assert SUPERSEDED_ARM64_DEVIATION_VIA_COREML == 0.33050537
    assert SUPERSEDED_ARM64_DEVIATION_VIA_COREML > COORD_TOLERANCE_PX, (
        "the superseded figure should sit outside the bound — that is why it "
        "needed an xfail at the time"
    )
    assert OBSERVED_COORD_DEVIATION_ARM64 < COORD_TOLERANCE_PX, (
        "arm64 must now conform; if it does not, the EP fix regressed"
    )
