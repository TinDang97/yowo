"""COCO mAP must count a missed image as a recall miss.

Measured 2026-09-13: `evaluate_coco_map` restricted `COCOeval.params.imgIds`
to the images that HAVE predictions, deleting every missed image from the
denominator. A model detecting its object in 1 of 10 images scored
0.99999999 against a perfect model's 1.0 — so a regression gate on the number
REWARDED a model that stopped detecting.

These checks drive REAL `COCOeval`. The defect survived review because
`test_benchmark_evaluator.py:219-256` mocks `pycocotools` entirely, so the
library never executed in the suite; a mocked check cannot bind this rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")

from yowo.benchmark._evaluator import evaluate_coco_map

_BOX = [100.0, 100.0, 200.0, 200.0]  # xywh
_N_IMAGES = 10


@pytest.fixture
def gt_path(tmp_path: Path) -> Path:
    """Synthetic COCO ground truth: N images, one identical annotation each."""
    gt = {
        "images": [
            {"id": i, "width": 640, "height": 640, "file_name": f"{i}.jpg"}
            for i in range(1, _N_IMAGES + 1)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": list(_BOX),
                "area": _BOX[2] * _BOX[3],
                "iscrowd": 0,
            }
            for i in range(1, _N_IMAGES + 1)
        ],
        "categories": [{"id": 1, "name": "thing", "supercategory": "thing"}],
    }
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(gt))
    return p


def _preds(image_ids: list[int]) -> list[dict[str, object]]:
    """Perfect predictions for exactly these image ids."""
    return [
        {"image_id": i, "category_id": 1, "bbox": list(_BOX), "score": 0.99}
        for i in image_ids
    ]


def test_a_model_that_misses_images_scores_lower(gt_path: Path) -> None:
    """The inversion, stated as a check: 1-of-10 must score BELOW 10-of-10."""
    perfect = evaluate_coco_map(gt_path, _preds(list(range(1, _N_IMAGES + 1))))
    partial = evaluate_coco_map(gt_path, _preds([1]))

    assert perfect["mAP_50_95"] > partial["mAP_50_95"], (
        f"a model detecting 1 of {_N_IMAGES} images scored "
        f"{partial['mAP_50_95']!r} against a perfect model's "
        f"{perfect['mAP_50_95']!r}. Scoping the evaluation to images that have "
        "predictions removes every miss from the denominator, so degrading "
        "recall raises the score and a gate on it rewards a model that stops "
        "detecting."
    )
    # And it must be lower by roughly the miss rate, not by a rounding error.
    assert partial["mAP_50_95"] < 0.2, (
        f"1-of-{_N_IMAGES} scored {partial['mAP_50_95']!r}; with misses counted "
        "it should sit near the 0.1 detection rate, not near 1.0"
    )


def test_subset_selects_from_the_ground_truth(gt_path: Path) -> None:
    """subset=N scores N ground-truth images regardless of prediction count."""
    # Predict perfectly on 2 images, but evaluate 10: the score must reflect
    # the 8 misses, not the 2 hits.
    scoped = evaluate_coco_map(gt_path, _preds([1, 2]), subset=_N_IMAGES)
    assert scoped["mAP_50_95"] < 0.4, (
        f"predicting on 2 of {_N_IMAGES} images scored {scoped['mAP_50_95']!r} "
        "with subset=10 — the 8 unpredicted images were not counted"
    )


def test_subset_is_deterministic(gt_path: Path) -> None:
    """A pinned baseline requires the same images every run."""
    a = evaluate_coco_map(gt_path, _preds([1, 2, 3]), subset=5)
    b = evaluate_coco_map(gt_path, _preds([1, 2, 3]), subset=5)
    assert a == b, (
        f"subset=5 produced {a} then {b} — a non-deterministic subset makes "
        "every recorded baseline incomparable"
    )


def test_subset_none_evaluates_every_image(gt_path: Path) -> None:
    full = evaluate_coco_map(gt_path, _preds([1]), subset=None)
    explicit = evaluate_coco_map(gt_path, _preds([1]), subset=_N_IMAGES)
    assert full["mAP_50_95"] == pytest.approx(explicit["mAP_50_95"]), (
        "subset=None must score the whole ground truth"
    )


def test_a_subset_larger_than_the_dataset_is_clamped(gt_path: Path) -> None:
    r = evaluate_coco_map(gt_path, _preds([1, 2]), subset=_N_IMAGES * 100)
    assert 0.0 <= r["mAP_50_95"] <= 1.0


def test_predictions_outside_the_subset_do_not_inflate_the_score(gt_path: Path) -> None:
    """Predicting on images outside the evaluated subset must not help."""
    inside_only = evaluate_coco_map(gt_path, _preds([1, 2]), subset=2)
    plus_outside = evaluate_coco_map(gt_path, _preds([1, 2, 9, 10]), subset=2)
    assert inside_only["mAP_50_95"] == pytest.approx(plus_outside["mAP_50_95"]), (
        "predictions on images outside the subset changed the score"
    )


def test_empty_predictions_score_zero_by_evaluation(gt_path: Path) -> None:
    """0.0 must be a measurement, not a sentinel that means 'did not run'."""
    empty = evaluate_coco_map(gt_path, [])
    # A prediction that matches nothing scores 0.0 through the real path.
    nonmatching = evaluate_coco_map(
        gt_path,
        [{"image_id": 1, "category_id": 1, "bbox": [600.0, 600.0, 10.0, 10.0], "score": 0.99}],
    )
    assert empty["mAP_50_95"] == pytest.approx(nonmatching["mAP_50_95"]) == 0.0


def test_a_perfect_model_still_scores_one(gt_path: Path) -> None:
    """A7: the repair moves the image scope and nothing about annotations.

    If it had also changed which annotations count (iscrowd handling, category
    filtering), every historical number would move for a second, unrelated
    reason and the regression this node fixes would be indistinguishable from
    that one.
    """
    r = evaluate_coco_map(gt_path, _preds(list(range(1, _N_IMAGES + 1))))
    assert r["mAP_50_95"] == pytest.approx(1.0), (
        f"a model matching every ground-truth box exactly scored {r['mAP_50_95']!r}, "
        "not 1.0 — the repair changed annotation matching, not just image scope"
    )


def test_subset_zero_is_an_empty_set_not_a_full_run(gt_path: Path) -> None:
    """A8: 0 must not be read as None, or a typo becomes a full-dataset run."""
    zero = evaluate_coco_map(gt_path, _preds(list(range(1, _N_IMAGES + 1))), subset=0)
    full = evaluate_coco_map(gt_path, _preds(list(range(1, _N_IMAGES + 1))), subset=None)
    assert zero["mAP_50_95"] != pytest.approx(full["mAP_50_95"]), (
        f"subset=0 scored {zero['mAP_50_95']!r}, the same as the full run "
        f"({full['mAP_50_95']!r}) — 0 is being read as None, so a typo would "
        "silently become a full-dataset evaluation"
    )
