"""A recorded mAP baseline, and the two-sided band that defends it.

A bare float is not a baseline: it cannot tell "the model regressed" from
"the weights were re-pinned" or "the subset was re-selected". Every input that
determines the number is recorded beside it, and the gate refuses to compare
across a mismatch rather than averaging two different measurements together.

The band is two-sided on purpose. Repeat runs on one machine are bit-identical
(measured 2026-09-15: 0.41577614647708055, twice), so for a fixed model, fixed
weights and a fixed image set the number should not move at all. A rise past
the band means something changed that nobody intended — which is exactly the
shape of the defect this node was created to fix, where a broken evaluation
scope moved the number by 10x.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yowo.benchmark._baseline import (
    BaselineError,
    MapRegression,
    check_against_baseline,
    load_baseline,
)

_RECORD = {
    "model": "yolo11n",
    "weights_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "backend": "pytorch",
    "device": "cpu",
    "confidence_threshold": 0.001,
    "iou_threshold": 0.45,
    "images": 500,
    "subset_manifest_sha256": "872ac411cf65f03234a725df9711e9ced36d1c8156161ab0b7ae63bd23336aad",
    "map_50_95": 0.4158,
    "map_50": 0.585,
    "map_75": 0.4509,
    "tolerance": 0.005,
    "measured_on": "ubuntu-24.04 x86_64",
    "measured_at": "2026-09-15",
}


def _write(tmp_path: Path, **overrides: object) -> Path:
    record = dict(_RECORD)
    for key, value in overrides.items():
        if value is None:
            record.pop(key, None)
        else:
            record[key] = value
    p = tmp_path / "coco_map_baseline.json"
    p.write_text(json.dumps(record, indent=2))
    return p


def _observed(**overrides: object) -> dict[str, object]:
    seen = {
        "weights_sha256": _RECORD["weights_sha256"],
        "backend": "pytorch",
        "device": "cpu",
        "confidence_threshold": 0.001,
        "iou_threshold": 0.45,
        "images_evaluated": 500,
        "subset_manifest_sha256": _RECORD["subset_manifest_sha256"],
        "map_50_95": 0.4158,
        "map_50": 0.585,
        "map_75": 0.4509,
    }
    seen.update(overrides)
    return seen


# --- M2: what makes a record a baseline ----------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "weights_sha256",
        "backend",
        "device",
        "confidence_threshold",
        "iou_threshold",
        "images",
        "subset_manifest_sha256",
        "map_50_95",
        "tolerance",
        "measured_on",
        "measured_at",
    ],
)
def test_the_baseline_records_every_input_that_determines_the_number(
    tmp_path: Path, field: str
) -> None:
    """Dropping any one field must be refused, by name.

    Q4: a loader with a default for every field cannot detect a dropped one.
    """
    with pytest.raises(BaselineError, match=field):
        load_baseline(_write(tmp_path, **{field: None}))


def test_a_complete_baseline_loads(tmp_path: Path) -> None:
    baseline = load_baseline(_write(tmp_path))
    assert baseline.map_50_95 == pytest.approx(0.4158)
    assert baseline.tolerance == pytest.approx(0.005)
    assert baseline.images == 500


# --- A10/E5: absence is a failure, never a pass ---------------------------


def test_a_missing_baseline_fails_rather_than_passes(tmp_path: Path) -> None:
    """R:GREEN_BY_SKIP — 'no baseline, therefore fine' defends nothing."""
    with pytest.raises(BaselineError, match="not found"):
        load_baseline(tmp_path / "absent.json")


def test_an_unparseable_baseline_fails(tmp_path: Path) -> None:
    p = tmp_path / "coco_map_baseline.json"
    p.write_text("{ this is not json")
    with pytest.raises(BaselineError):
        load_baseline(p)


# --- M3: the band, both ways ---------------------------------------------


def test_a_measurement_on_the_baseline_passes(tmp_path: Path) -> None:
    check_against_baseline(load_baseline(_write(tmp_path)), _observed())


def test_a_measurement_inside_the_band_passes(tmp_path: Path) -> None:
    baseline = load_baseline(_write(tmp_path))
    check_against_baseline(baseline, _observed(map_50_95=0.4158 - 0.004))
    check_against_baseline(baseline, _observed(map_50_95=0.4158 + 0.004))


def test_a_measurement_below_the_band_fails(tmp_path: Path) -> None:
    baseline = load_baseline(_write(tmp_path))
    with pytest.raises(MapRegression) as exc:
        check_against_baseline(baseline, _observed(map_50_95=0.4158 - 0.006))
    message = str(exc.value)
    assert "0.4158" in message, "the failure must name the baseline"
    assert "0.4098" in message, "the failure must name the measurement"
    assert "-0.006" in message, "the failure must name the signed delta"


def test_a_measurement_above_the_band_also_fails(tmp_path: Path) -> None:
    """A rise is not automatically good news.

    Nothing about the model, the weights or the images changed, so a number
    that went up went up for a reason nobody declared.
    """
    baseline = load_baseline(_write(tmp_path))
    with pytest.raises(MapRegression) as exc:
        check_against_baseline(baseline, _observed(map_50_95=0.4158 + 0.006))
    assert "+0.006" in str(exc.value), "the failure must name the signed delta"


def test_zero_predictions_fail_rather_than_passing_vacuously(tmp_path: Path) -> None:
    """E4: a model that detects nothing must not reach a 'nothing to compare' pass."""
    baseline = load_baseline(_write(tmp_path))
    with pytest.raises(MapRegression):
        check_against_baseline(baseline, _observed(map_50_95=0.0))


# --- E5/E6: refuse to compare across a mismatch --------------------------


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("weights_sha256", "0" * 64),
        ("backend", "onnx"),
        ("device", "cuda"),
        ("confidence_threshold", 0.25),
        ("iou_threshold", 0.7),
        ("images_evaluated", 499),
        ("subset_manifest_sha256", "f" * 64),
    ],
)
def test_a_mismatched_input_fails_instead_of_being_compared(
    tmp_path: Path, field: str, wrong: object
) -> None:
    """Two numbers measured under different conditions are not comparable.

    Failing here is the point: it sends the reader to the input that changed
    rather than letting a backend swap read as a model regression.
    """
    baseline = load_baseline(_write(tmp_path))
    with pytest.raises(MapRegression, match=field.replace("_", ".")):
        check_against_baseline(baseline, _observed(**{field: wrong}))


def test_the_gate_never_writes_the_baseline(tmp_path: Path) -> None:
    """R:SELF_UPDATING_BASELINE — a gate that re-records prevents nothing."""
    path = _write(tmp_path)
    before = path.read_bytes()
    baseline = load_baseline(path)
    with pytest.raises(MapRegression):
        check_against_baseline(baseline, _observed(map_50_95=0.0))
    assert path.read_bytes() == before, "the gate rewrote the baseline it was gating against"


# --- the shipped record itself -------------------------------------------
#
# Every test above writes its own record to tmp_path. None of them loads the
# file the gate actually reads, so until this block the shipped fixture was
# unconstrained: editing `"tolerance": 0.005` to `1.0` widened the band to
# unfalsifiable and left all 2798 unit tests, all four quality gates and
# pre-commit green — the only job that reads the file being the one the
# widened band could no longer fail.

SHIPPED = Path(__file__).resolve().parents[1] / "fixtures" / "coco_map_baseline.json"

#: A band wider than this is not defending anything. yolo11n scores ~0.4158 on
#: the pinned 500, so 0.01 is already 2.4% of the number; cross-machine drift
#: measured between darwin/arm64 and ubuntu/x86-64 sits inside 0.005.
MAX_DEFENSIBLE_TOLERANCE = 0.01


def test_the_shipped_baseline_loads() -> None:
    """The file the gate reads must satisfy the loader it is read by."""
    assert load_baseline(SHIPPED).model == "yolo11n"


def test_the_shipped_band_is_narrow_enough_to_fail_something() -> None:
    """R:SELF_UPDATING_BASELINE, by the back door.

    Widening the tolerance is re-recording the baseline in everything but
    name: it makes the gate pass without the number improving, and nothing
    else in the suite reads this file.
    """
    tolerance = load_baseline(SHIPPED).tolerance
    assert 0.0 < tolerance <= MAX_DEFENSIBLE_TOLERANCE, (
        f"the shipped band is +/-{tolerance}, which is not a band a regression "
        f"could fail. A gate whose tolerance was widened to accommodate a drop "
        f"is a gate that was switched off."
    )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("tolerance", "0.005"),
        ("map_50_95", "0.4158"),
        ("images", "500"),
        ("confidence_threshold", None),
    ],
)
def test_a_field_of_the_wrong_type_is_refused(tmp_path: Path, field: str, wrong: object) -> None:
    """Presence is not validity.

    `"tolerance": "0.005"` loaded fine and raised a bare TypeError from the
    comparison, so the loader's documented contract — missing, unreadable or
    incomplete — was untrue for a malformed record.
    """
    record = dict(_RECORD)
    record[field] = wrong
    p = tmp_path / "coco_map_baseline.json"
    p.write_text(json.dumps(record))
    with pytest.raises(BaselineError, match=field):
        load_baseline(p)


def test_a_non_positive_tolerance_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BaselineError, match="tolerance"):
        load_baseline(_write(tmp_path, tolerance=0.0))
    with pytest.raises(BaselineError, match="tolerance"):
        load_baseline(_write(tmp_path, tolerance=-0.1))


# --- a measurement that is not a number ----------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_measurement_fails_rather_than_passes(tmp_path: Path, value: float) -> None:
    """NaN walked through the band check.

    `abs(nan) > tolerance` is False, so a NaN measurement returned normally
    from the one function that decides whether the gate passed — past an
    isinstance guard whose stated purpose is that a gate with no measurement
    must not report success.
    """
    baseline = load_baseline(_write(tmp_path))
    with pytest.raises(MapRegression):
        check_against_baseline(baseline, _observed(map_50_95=value))


# --- every recorded number is enforced, not just one ---------------------


@pytest.mark.parametrize("field", ["map_50_95", "map_50", "map_75"])
def test_all_three_recorded_scores_are_defended(tmp_path: Path, field: str) -> None:
    """`map_50` and `map_75` were required by the loader and read by nothing.

    Editing either in the shipped fixture left every test in the repository
    green, including the gate — two of the three numbers in a file whose whole
    argument is that a bare float is not a baseline were decorative.
    """
    baseline = load_baseline(_write(tmp_path))
    drifted = dict(_RECORD)[field] + 0.02  # type: ignore[operator]
    with pytest.raises(MapRegression, match=field):
        check_against_baseline(baseline, _observed(**{field: drifted}))
