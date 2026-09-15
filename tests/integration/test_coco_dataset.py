"""The pinned subset, bound to the real COCO val2017 annotations.

Everything in ``tests/unit/test_coco_dataset_fixture.py`` proves the acquisition
machinery behaves — against a local HTTP server and hand-built zips. Nothing
there proves the committed manifest actually describes COCO. This does, and it
is the only place that can: it needs the real 1.07 GB.

Under CI a missing dataset FAILS (see ``require_coco_val2017``); locally it
skips. A skip is green, so a CI box that fetched nothing must not report one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import datasets

pytestmark = pytest.mark.integration


class TestThePinnedSubsetDescribesRealCoco:
    def test_pinned_subset_matches_the_real_annotations(self, coco_val2017_root: Path) -> None:
        """covers: A5,M6,R:RESELECT

        The manifest must be exactly what ``evaluate_coco_map(subset=500)``
        selects — ``sorted(getImgIds())[:500]`` over GROUND TRUTH. If these ever
        diverge, the repo documents one set of images and the evaluator scores
        another, and every published mAP means something other than what the
        manifest says.
        """
        annotations = coco_val2017_root / "annotations" / "instances_val2017.json"
        with annotations.open(encoding="utf-8") as handle:
            ground_truth = json.load(handle)

        all_ids = [image["id"] for image in ground_truth["images"]]
        expected = datasets.select_subset_ids(all_ids, datasets.SUBSET_SIZE)

        assert list(datasets.pinned_subset_ids()) == expected, (
            "the committed manifest is not the first 500 sorted ground-truth "
            "image ids of instances_val2017.json"
        )

    def test_the_selection_agrees_with_the_evaluator_itself(self, coco_val2017_root: Path) -> None:
        """covers: M6,R:RESELECT

        Not a re-implementation of the rule — pycocotools' own ``getImgIds`` is
        asked, through the same sorted-prefix selection the evaluator applies,
        so a change to either side shows up here.
        """
        try:
            from pycocotools.coco import COCO  # type: ignore[import-untyped]
        except ImportError as exc:
            # importorskip would SKIP, and a skip is green. The accuracy tier
            # without pycocotools evaluated nothing and must say so.
            raise AssertionError(
                f"pycocotools is required by the accuracy tier: {exc}. "
                "Install with: pip install yowo[benchmark]"
            ) from exc

        import contextlib
        import io

        annotations = coco_val2017_root / "annotations" / "instances_val2017.json"
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(annotations))

        expected = datasets.select_subset_ids(coco.getImgIds(), datasets.SUBSET_SIZE)
        assert list(datasets.pinned_subset_ids()) == expected

    def test_every_pinned_image_is_present_on_disk(self, coco_val2017_root: Path) -> None:
        """covers: M5,M6

        A manifest naming images the tree does not hold would fail later as an
        opaque missing-JPEG error in the middle of an evaluation.
        """
        images = coco_val2017_root / "val2017"
        missing = [
            file_name
            for _, file_name in datasets.pinned_subset_entries()
            if not (images / file_name).is_file()
        ]
        assert missing == [], f"{len(missing)} pinned images are absent: {missing[:5]}"

    def test_the_file_names_match_the_ids_they_are_pinned_against(
        self, coco_val2017_root: Path
    ) -> None:
        """covers: M6

        COCO names every val2017 file after its own image id. Pinning both means
        a row can be checked by eye; this asserts the two halves agree with the
        annotations rather than merely with each other.
        """
        annotations = coco_val2017_root / "annotations" / "instances_val2017.json"
        with annotations.open(encoding="utf-8") as handle:
            by_id = {image["id"]: image["file_name"] for image in json.load(handle)["images"]}

        for image_id, file_name in datasets.pinned_subset_entries():
            assert by_id[image_id] == file_name


class TestTheDatasetRootIsUsableByTheEvaluator:
    def test_load_coco_dataset_reads_the_tree_this_fixture_builds(
        self, coco_val2017_root: Path
    ) -> None:
        """covers: M5

        The layout `ensure_coco_val2017` produces must be the layout
        `load_coco_dataset` expects, or the dataset is reachable and unusable.
        """
        try:
            from yowo.benchmark._evaluator import load_coco_dataset
        except ImportError as exc:  # pragma: no cover
            raise AssertionError(f"benchmark evaluator unavailable: {exc}") from exc

        paths, image_ids, ann_file = load_coco_dataset(
            coco_val2017_root, subset=datasets.SUBSET_SIZE
        )

        assert image_ids == list(datasets.pinned_subset_ids())
        assert len(paths) == datasets.SUBSET_SIZE
        assert all(p.is_file() for p in paths)
        assert Path(ann_file).is_file()
