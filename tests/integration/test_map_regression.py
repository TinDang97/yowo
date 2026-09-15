"""The mAP regression gate: a real number, on real images, against a pinned band.

This is the only place in the suite where an accuracy claim is measured rather
than asserted. `tests/unit/test_coco_map_evaluator.py` proves the evaluator
scopes correctly on synthetic ground truth; `tests/unit/test_map_baseline.py`
proves the band refuses a mismatch. Neither proves yolo11n still detects
anything. This does, and it needs the real 1.07 GB and the real weight.

Under CI a missing dataset or weight FAILS (see ``require_coco_val2017``);
locally it skips. A skip is green, so a gate that fetched nothing must not
report one.

Measured 2026-09-15, yolo11n / PyTorch / CPU, confidence 0.001, over the pinned
500: mAP_50_95 = 0.4158, bit-identical across repeat runs on one machine. The
band exists only to absorb cross-machine float drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.support.datasets import SUBSET_MANIFEST_PATH, file_digest, pinned_subset_ids
from yowo.benchmark._baseline import check_against_baseline, load_baseline
from yowo.types import BackendType, ModelFamily, ModelSize

pytestmark = pytest.mark.integration

BASELINE = Path(__file__).resolve().parents[1] / "fixtures" / "coco_map_baseline.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def measured(coco_val2017_root: Path) -> dict[str, object]:
    """Infer over the pinned 500 and evaluate — once, for every check below."""
    import cv2

    from yowo.benchmark._evaluator import detections_to_coco_results, evaluate_coco_map
    from yowo.engine import DetectionEngine
    from yowo.models._weights import resolve_weights
    from yowo.types import Frame, ModelSpec

    baseline = load_baseline(BASELINE)
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    weights = resolve_weights(spec)

    image_ids = list(pinned_subset_ids())
    annotations = coco_val2017_root / "annotations" / "instances_val2017.json"
    images_dir = coco_val2017_root / "val2017"

    import contextlib
    import io

    from pycocotools.coco import COCO  # type: ignore[import-untyped]

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotations))
    paths = [images_dir / info["file_name"] for info in coco.loadImgs(image_ids)]

    engine = DetectionEngine(
        model_family=ModelFamily.YOLO11,
        model_size=ModelSize.NANO,
        backend=BackendType.PYTORCH,
        device="cpu",
        confidence_threshold=baseline.confidence_threshold,
        iou_threshold=baseline.iou_threshold,
        weights_path=weights,
    )
    detections = []
    with engine:
        device = engine.selection.device_type.value
        backend = engine.selection.backend.value
        for index, path in enumerate(paths):
            pixels = cv2.imread(str(path))
            assert pixels is not None, (
                f"{path} did not decode. A black-frame fallback would score 0.0 and "
                "read as a model regression rather than a broken dataset."
            )
            detections.extend(
                engine.detect([Frame(pixels=pixels, source_id=str(path), frame_index=index)])
            )

    results = detections_to_coco_results(detections, image_ids)
    scores = evaluate_coco_map(annotations, results, image_ids=image_ids)
    return {
        "weights_sha256": _sha256(weights),
        "backend": backend,
        "device": device,
        "confidence_threshold": baseline.confidence_threshold,
        "iou_threshold": baseline.iou_threshold,
        "images_evaluated": int(scores["images_evaluated"]),
        "subset_manifest_sha256": file_digest(SUBSET_MANIFEST_PATH),
        "map_50_95": scores["mAP_50_95"],
        "map_50": scores["mAP_50"],
        "map_75": scores["mAP_75"],
        "detections": len(results),
    }


def test_the_measured_map_is_inside_the_recorded_band(measured: dict[str, object]) -> None:
    """covers: M3,M4,E4,E5,E6 — the gate itself.

    Fails on a drop AND on an unexplained rise: for a fixed model, fixed
    weights and a fixed image set the number should not move at all, so a rise
    means something changed that nobody declared. That is the exact shape of
    the defect this node was created to fix, where a broken evaluation scope
    moved the number by 10x.
    """
    baseline = load_baseline(BASELINE)
    # Print before asserting. A gate that reports only pass/fail leaves no
    # record of WHAT it measured, so the CI log cannot answer "how much drift
    # does this machine actually have" without perturbing the baseline to
    # force a failure. `pytest -s` or a failure shows it; either way the
    # number is in the run rather than only in the verdict.
    print(
        f"\nmAP@0.5:0.95 measured {float(measured['map_50_95']):.6f} "
        f"| baseline {baseline.map_50_95:.6f} "
        f"| delta {float(measured['map_50_95']) - baseline.map_50_95:+.6f} "
        f"| tolerance +/-{baseline.tolerance} "
        f"| {measured['images_evaluated']} images, {measured['detections']} detections "
        f"| baseline measured on {baseline.measured_on}"
    )
    check_against_baseline(baseline, measured)


def test_the_gate_evaluated_every_pinned_image(measured: dict[str, object]) -> None:
    """covers: M1,A3 — the denominator is the pinned 500, not 5000 and not 2."""
    assert measured["images_evaluated"] == len(pinned_subset_ids())


def test_the_model_actually_detected_something(measured: dict[str, object]) -> None:
    """covers: E4 — a zero-detection run must not reach the band check silently."""
    assert int(measured["detections"]) > 0, (
        "yolo11n produced no detections at all over 500 real images"
    )
