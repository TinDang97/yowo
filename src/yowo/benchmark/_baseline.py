"""A recorded mAP baseline, and the two-sided band that defends it.

A bare float is not a baseline. It cannot tell "the model regressed" from "the
weights were re-pinned", "the backend changed" or "the subset was re-selected",
and a gate that cannot tell them apart teaches contributors to re-record the
number rather than investigate it. Everything that determines the measurement
is recorded beside it, and a mismatch is refused rather than compared.

The band is two-sided on purpose. Repeat runs on one machine are bit-identical
(measured 2026-09-15 on the pinned COCO val2017 500: 0.41577614647708055,
twice), so for a fixed model, fixed weights and a fixed image set the number
should not move at all. The band exists to absorb cross-machine float drift and
nothing else — and a rise past it means something changed that nobody declared,
which is exactly the shape of the defect this module was written for: a broken
evaluation scope moved the reported number by a factor of ten.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from yowo.errors import YowoError

__all__ = [
    "BaselineError",
    "MapBaseline",
    "MapRegression",
    "check_against_baseline",
    "load_baseline",
]


class BaselineError(YowoError):
    """The baseline record is missing, unreadable, or incomplete.

    Never a pass. "No baseline, therefore fine" defends nothing, and a gate
    that reports green because its own reference was absent is the same defect
    as a test that passes because it never ran.
    """


class MapRegression(YowoError):
    """The measured mAP left the recorded band, or was measured differently."""


@dataclass(frozen=True)
class MapBaseline:
    """Every input that determines the number, recorded beside the number.

    Each field is required. A loader with a default for every field cannot
    detect a dropped one, so the record would quietly shrink over time and the
    gate would compare across a difference it no longer knew about.
    """

    model: str
    weights_sha256: str
    backend: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    images: int
    subset_manifest_sha256: str
    map_50_95: float
    map_50: float
    map_75: float
    tolerance: float
    measured_on: str
    measured_at: str


def load_baseline(path: str | Path) -> MapBaseline:
    """Read a baseline record, refusing anything incomplete.

    Args:
        path: Path to the baseline JSON.

    Returns:
        The parsed record.

    Raises:
        BaselineError: The file is absent, unparseable, or missing a field.
    """
    p = Path(path)
    if not p.is_file():
        raise BaselineError(
            f"mAP baseline not found at {p}. The gate has nothing to compare "
            f"against; it fails rather than passing, because a gate with no "
            f"reference defends nothing."
        )
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"mAP baseline at {p} could not be read: {exc}") from exc

    if not isinstance(record, Mapping):
        raise BaselineError(f"mAP baseline at {p} is not a JSON object")

    required = [f.name for f in fields(MapBaseline)]
    missing = [name for name in required if name not in record]
    if missing:
        raise BaselineError(
            f"mAP baseline at {p} is missing {', '.join(missing)}. Every field is "
            f"required: without it the gate cannot tell a model regression from a "
            f"change to how the number was measured."
        )
    return MapBaseline(**{name: record[name] for name in required})


#: Recorded field -> the key the gate's own run reports it under. The names
#: differ for the image count on purpose: the baseline records what it covered,
#: the run reports what it actually evaluated, and conflating them is how a
#: clamped subset misattributes its denominator.
_MUST_MATCH: tuple[tuple[str, str], ...] = (
    ("weights_sha256", "weights_sha256"),
    ("backend", "backend"),
    ("device", "device"),
    ("confidence_threshold", "confidence_threshold"),
    ("iou_threshold", "iou_threshold"),
    ("images", "images_evaluated"),
    ("subset_manifest_sha256", "subset_manifest_sha256"),
)


def check_against_baseline(
    baseline: MapBaseline,
    observed: Mapping[str, object],
) -> None:
    """Compare a measurement against the recorded band. Never writes.

    Args:
        baseline: The recorded reference.
        observed: What this run measured — the same inputs, plus ``map_50_95``
            and ``images_evaluated``.

    Raises:
        MapRegression: An input differs from the recorded one, or the measured
            mAP is further than ``tolerance`` from the recorded mAP in either
            direction.
    """
    for recorded_name, observed_name in _MUST_MATCH:
        recorded = getattr(baseline, recorded_name)
        seen = observed.get(observed_name)
        if isinstance(recorded, float) and isinstance(seen, (int, float)):
            same = abs(float(seen) - recorded) < 1e-12
        else:
            same = seen == recorded
        if not same:
            raise MapRegression(
                f"{observed_name} does not match the baseline: measured {seen!r}, "
                f"recorded {recorded!r} (baseline field {recorded_name!r}). Two "
                f"numbers measured under different conditions are not comparable — "
                f"this is refused rather than reported as a regression, so the "
                f"reader is sent to the input that changed."
            )

    measured = observed.get("map_50_95")
    if not isinstance(measured, (int, float)):
        raise MapRegression(
            f"the run reported no map_50_95 to compare (got {measured!r}); "
            f"a gate with no measurement must not report success"
        )

    delta = float(measured) - baseline.map_50_95
    if abs(delta) > baseline.tolerance:
        direction = "below" if delta < 0 else "above"
        raise MapRegression(
            f"mAP@0.5:0.95 is {abs(delta):.4f} {direction} the recorded baseline: "
            f"measured {float(measured):.4f}, baseline {baseline.map_50_95:.4f}, "
            f"delta {delta:+.4f}, tolerance +/-{baseline.tolerance:.4f}.\n"
            f"The baseline was measured on {baseline.measured_on} at "
            f"{baseline.measured_at} over {baseline.images} images with "
            f"{baseline.model} / {baseline.backend} / {baseline.device}, "
            f"confidence {baseline.confidence_threshold}, IoU "
            f"{baseline.iou_threshold}.\n"
            f"A rise is not automatically good news: nothing about the model, the "
            f"weights or the images changed, so check what did before re-recording "
            f"the number. The gate never rewrites its own baseline."
        )
