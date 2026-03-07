"""Tests for yowo.benchmark._evaluator module."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from yowo.benchmark._evaluator import (
    YOLO_TO_COCO,
    detections_to_coco_results,
    evaluate_coco_map,
    evaluate_imagenet_accuracy,
    load_coco_dataset,
    load_imagenet_dataset,
)
from yowo.types import (
    BackendType,
    BoundingBox,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)

# ---------------------------------------------------------------------------
# YOLO_TO_COCO mapping tests
# ---------------------------------------------------------------------------


class TestYoloToCocoMapping:
    def test_mapping_has_80_entries(self) -> None:
        assert len(YOLO_TO_COCO) == 80

    def test_first_entry_is_person(self) -> None:
        # YOLO class 0 (person) -> COCO category 1
        assert YOLO_TO_COCO[0] == 1

    def test_last_entry_is_toothbrush(self) -> None:
        # YOLO class 79 (toothbrush) -> COCO category 90
        assert YOLO_TO_COCO[79] == 90

    def test_all_values_are_positive_integers(self) -> None:
        for idx, coco_id in enumerate(YOLO_TO_COCO):
            assert isinstance(coco_id, int), f"Index {idx} is not int: {type(coco_id)}"
            assert coco_id > 0, f"Index {idx} has non-positive value: {coco_id}"

    def test_no_duplicate_values(self) -> None:
        assert len(set(YOLO_TO_COCO)) == 80


# ---------------------------------------------------------------------------
# detections_to_coco_results tests
# ---------------------------------------------------------------------------


def _make_detection(
    boxes: list[tuple[float, float, float, float, float, int]],
) -> Detection:
    """Helper: create a Detection with given bounding boxes."""
    frame = Frame(pixels=np.zeros((480, 640, 3), dtype=np.uint8), source_id="test")
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    return Detection(
        frame=frame,
        boxes=tuple(
            BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=b[4], class_id=b[5])
            for b in boxes
        ),
        inference_time_ms=10.0,
        backend=BackendType.PYTORCH,
        model_spec=spec,
    )


class TestDetectionsToCocoResults:
    def test_converts_xyxy_to_xywh(self) -> None:
        det = _make_detection([(10.0, 20.0, 50.0, 80.0, 0.9, 0)])
        results = detections_to_coco_results([det], [1])
        assert len(results) == 1
        bbox = results[0]["bbox"]
        assert bbox == [10.0, 20.0, 40.0, 60.0]  # w=50-10, h=80-20

    def test_maps_class_id_0_to_coco_category_1(self) -> None:
        det = _make_detection([(0.0, 0.0, 10.0, 10.0, 0.5, 0)])
        results = detections_to_coco_results([det], [42])
        assert results[0]["category_id"] == 1

    def test_maps_class_id_79_to_coco_category_90(self) -> None:
        det = _make_detection([(0.0, 0.0, 10.0, 10.0, 0.5, 79)])
        results = detections_to_coco_results([det], [42])
        assert results[0]["category_id"] == 90

    def test_image_id_is_set_correctly(self) -> None:
        det = _make_detection([(0.0, 0.0, 10.0, 10.0, 0.5, 0)])
        results = detections_to_coco_results([det], [999])
        assert results[0]["image_id"] == 999

    def test_score_is_confidence(self) -> None:
        det = _make_detection([(0.0, 0.0, 10.0, 10.0, 0.87654, 0)])
        results = detections_to_coco_results([det], [1])
        assert results[0]["score"] == 0.87654

    def test_multiple_boxes_per_detection(self) -> None:
        det = _make_detection(
            [
                (0.0, 0.0, 10.0, 10.0, 0.9, 0),
                (20.0, 30.0, 40.0, 50.0, 0.8, 1),
            ]
        )
        results = detections_to_coco_results([det], [1])
        assert len(results) == 2

    def test_empty_detections(self) -> None:
        det = _make_detection([])
        results = detections_to_coco_results([det], [1])
        assert results == []

    def test_multiple_images(self) -> None:
        det1 = _make_detection([(0.0, 0.0, 10.0, 10.0, 0.9, 0)])
        det2 = _make_detection([(5.0, 5.0, 15.0, 15.0, 0.8, 1)])
        results = detections_to_coco_results([det1, det2], [100, 200])
        assert len(results) == 2
        assert results[0]["image_id"] == 100
        assert results[1]["image_id"] == 200


# ---------------------------------------------------------------------------
# evaluate_imagenet_accuracy tests
# ---------------------------------------------------------------------------


class TestEvaluateImagenetAccuracy:
    def test_perfect_accuracy(self) -> None:
        preds = [(i, i) for i in range(10)]
        result = evaluate_imagenet_accuracy(preds)
        assert result["top1_accuracy"] == 1.0
        assert result["total"] == 10.0
        assert result["correct"] == 10.0

    def test_zero_accuracy(self) -> None:
        preds = [(0, 1), (1, 2), (2, 3)]
        result = evaluate_imagenet_accuracy(preds)
        assert result["top1_accuracy"] == pytest.approx(0.0)
        assert result["correct"] == 0.0

    def test_half_accuracy(self) -> None:
        preds = [(i, i) for i in range(10)] + [(i, i + 1) for i in range(10)]
        result = evaluate_imagenet_accuracy(preds)
        assert result["top1_accuracy"] == pytest.approx(0.5)
        assert result["total"] == 20.0
        assert result["correct"] == 10.0

    def test_empty_predictions(self) -> None:
        result = evaluate_imagenet_accuracy([])
        assert result["top1_accuracy"] == 0.0
        assert result["total"] == 0.0


# ---------------------------------------------------------------------------
# load_coco_dataset error tests
# ---------------------------------------------------------------------------


class TestLoadCocoDataset:
    def test_raises_on_missing_val_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="val2017"):
            load_coco_dataset(tmp_path)

    def test_raises_on_missing_ann_file(self, tmp_path: Path) -> None:
        (tmp_path / "val2017").mkdir()
        with pytest.raises(FileNotFoundError, match="instances_val2017.json"):
            load_coco_dataset(tmp_path)


# ---------------------------------------------------------------------------
# load_imagenet_dataset error tests
# ---------------------------------------------------------------------------


class TestLoadImagenetDataset:
    def test_raises_on_empty_dir(self, tmp_path: Path) -> None:
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No class folders"):
            load_imagenet_dataset(tmp_path)

    def test_loads_simple_structure(self, tmp_path: Path) -> None:
        val_dir = tmp_path / "val"
        cls_a = val_dir / "n01"
        cls_b = val_dir / "n02"
        cls_a.mkdir(parents=True)
        cls_b.mkdir(parents=True)
        (cls_a / "img1.JPEG").write_text("fake")
        (cls_a / "img2.JPEG").write_text("fake")
        (cls_b / "img3.JPEG").write_text("fake")

        paths, labels = load_imagenet_dataset(tmp_path)
        assert len(paths) == 3
        assert labels == [0, 0, 1]  # n01=0, n02=1 (alphabetical)

    def test_subset_limits_results(self, tmp_path: Path) -> None:
        val_dir = tmp_path / "val"
        cls_a = val_dir / "n01"
        cls_a.mkdir(parents=True)
        for i in range(5):
            (cls_a / f"img{i}.JPEG").write_text("fake")

        paths, labels = load_imagenet_dataset(tmp_path, subset=2)
        assert len(paths) == 2
        assert len(labels) == 2


# ---------------------------------------------------------------------------
# evaluate_coco_map tests (mocked pycocotools)
# ---------------------------------------------------------------------------


class TestEvaluateCocoMap:
    def test_empty_predictions_returns_zero(self) -> None:
        result = evaluate_coco_map("/fake/path.json", [])
        assert result["mAP_50_95"] == 0.0
        assert result["mAP_50"] == 0.0
        assert result["mAP_75"] == 0.0

    def test_calls_cocoeval_correctly(self) -> None:
        """Verify COCOeval is instantiated and called with evaluate/accumulate/summarize."""
        mock_coco_gt = MagicMock()
        mock_coco_dt = MagicMock()
        mock_coco_gt.loadRes.return_value = mock_coco_dt

        mock_eval_instance = MagicMock()
        mock_eval_instance.stats = [0.35, 0.55, 0.38, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Create mock pycocotools modules
        mock_coco_mod = ModuleType("pycocotools.coco")
        mock_coco_mod.COCO = MagicMock(return_value=mock_coco_gt)  # type: ignore[attr-defined]
        mock_cocoeval_mod = ModuleType("pycocotools.cocoeval")
        mock_cocoeval_mod.COCOeval = MagicMock(return_value=mock_eval_instance)  # type: ignore[attr-defined]
        mock_pycocotools = ModuleType("pycocotools")

        saved = {
            k: sys.modules.get(k)
            for k in ("pycocotools", "pycocotools.coco", "pycocotools.cocoeval")
        }
        try:
            sys.modules["pycocotools"] = mock_pycocotools
            sys.modules["pycocotools.coco"] = mock_coco_mod
            sys.modules["pycocotools.cocoeval"] = mock_cocoeval_mod

            predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
            result = evaluate_coco_map("/fake/ann.json", predictions)

            # Verify COCO was loaded with the annotation path
            mock_coco_mod.COCO.assert_called_once_with("/fake/ann.json")  # type: ignore[attr-defined]
            # Verify loadRes was called with predictions
            mock_coco_gt.loadRes.assert_called_once_with(predictions)
            # Verify COCOeval was called with correct args
            mock_cocoeval_mod.COCOeval.assert_called_once_with(mock_coco_gt, mock_coco_dt, "bbox")  # type: ignore[attr-defined]
            # Verify evaluate/accumulate/summarize were called
            mock_eval_instance.evaluate.assert_called_once()
            mock_eval_instance.accumulate.assert_called_once()
            mock_eval_instance.summarize.assert_called_once()
            # Verify returned values from stats
            assert result["mAP_50_95"] == pytest.approx(0.35)
            assert result["mAP_50"] == pytest.approx(0.55)
            assert result["mAP_75"] == pytest.approx(0.38)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
