"""A subset benchmark must score only the images it evaluated.

Measured 2026-09-15 on the pinned COCO val2017 500-image subset, yolo11n /
PyTorch / CPU, confidence 0.001:

    evaluate_coco_map(ann, preds, subset=500)  ->  0.4158   (the truth)
    what `run_benchmark(subset=500)` reported  ->  0.0419   (10x too low)

`evaluate_coco_map`'s scope repair landed in PR #49 and stopped at the
runner's door: `_runner.py` received the evaluated image ids and dropped them,
so 4500 ground-truth images the run never saw were counted as total misses.
The number the shipped `yowo benchmark --subset` CLI prints is wrong, and a
regression gate built on that path would gate on 0.0419.

These checks drive REAL `COCOeval` through the public runner. A mocked
evaluator cannot bind this rule — that is precisely how the defect survived.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from yowo.benchmark._runner import BenchmarkResult, run_single_backend
from yowo.types import (
    BackendType,
    BoundingBox,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)

_N_IMAGES = 10
_BOX_XYWH = [100.0, 100.0, 200.0, 200.0]
_BOX_XYXY = (100.0, 100.0, 300.0, 300.0)
_SPEC = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)


@pytest.fixture
def gt_path(tmp_path: Path) -> Path:
    """Synthetic COCO ground truth: 10 images, one identical annotation each."""
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
                "bbox": list(_BOX_XYWH),
                "area": _BOX_XYWH[2] * _BOX_XYWH[3],
                "iscrowd": 0,
            }
            for i in range(1, _N_IMAGES + 1)
        ],
        "categories": [{"id": 1, "name": "thing", "supercategory": "thing"}],
    }
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(gt))
    return p


def _perfect_detection() -> Detection:
    """One detection that exactly matches the synthetic ground-truth box."""
    x1, y1, x2, y2 = _BOX_XYXY
    return Detection(
        frame=Frame(pixels=np.zeros((640, 640, 3), dtype=np.uint8), source_id="bench"),
        boxes=(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.99, class_id=0),),
        inference_time_ms=1.0,
        backend=BackendType.PYTORCH,
        model_spec=_SPEC,
    )


def _run_over(gt_path: Path, image_ids: list[int]) -> float:
    """Benchmark a perfect model over exactly *image_ids*, return mAP_50_95."""
    with patch("yowo.benchmark._runner.DetectionEngine") as engine_cls:
        engine = MagicMock()
        engine_cls.return_value = engine
        engine.__enter__ = MagicMock(return_value=engine)
        engine.__exit__ = MagicMock(return_value=False)
        engine.detect.return_value = [_perfect_detection()]
        result = run_single_backend(
            model_spec=_SPEC,
            backend_type=BackendType.PYTORCH,
            images=[Path(f"/nonexistent/{i}.jpg") for i in image_ids],
            image_ids=image_ids,
            gt_ann_path=str(gt_path),
            task="detect",
            warmup_passes=0,
        )
    assert result.map_50_95 is not None
    return result.map_50_95


def test_a_subset_benchmark_scores_only_the_images_it_evaluated(gt_path: Path) -> None:
    """M1 on the public path: a perfect model over 2 of 10 images scores 1.0.

    Before the repair this scored ~0.2 — the eight images the run never looked
    at were counted as recall misses against it.
    """
    scored = _run_over(gt_path, [1, 2])
    assert scored == pytest.approx(1.0), (
        f"a perfect model benchmarked over 2 of {_N_IMAGES} images scored {scored!r}, "
        f"not 1.0 — the runner is scoring it against ground truth it never saw. "
        f"Measured on real COCO: this is the difference between 0.4158 and 0.0419."
    )


def test_a_full_run_is_unchanged(gt_path: Path) -> None:
    """E2: a run that does cover every image still scores the same as before."""
    scored = _run_over(gt_path, list(range(1, _N_IMAGES + 1)))
    assert scored == pytest.approx(1.0)


def test_a_model_that_misses_images_inside_its_own_subset_still_scores_lower(
    gt_path: Path,
) -> None:
    """The repair must not make recall free.

    Scoping to the evaluated images is correct; scoping to the images that
    happened to produce a prediction is the original defect. A run that
    evaluated four images and detected in only two must score below one that
    detected in all four.
    """
    with patch("yowo.benchmark._runner.DetectionEngine") as engine_cls:
        engine = MagicMock()
        engine_cls.return_value = engine
        engine.__enter__ = MagicMock(return_value=engine)
        engine.__exit__ = MagicMock(return_value=False)
        empty = Detection(
            frame=Frame(pixels=np.zeros((640, 640, 3), dtype=np.uint8), source_id="bench"),
            boxes=(),
            inference_time_ms=1.0,
            backend=BackendType.PYTORCH,
            model_spec=_SPEC,
        )
        engine.detect.side_effect = [
            [_perfect_detection()],
            [_perfect_detection()],
            [empty],
            [empty],
        ]
        partial = run_single_backend(
            model_spec=_SPEC,
            backend_type=BackendType.PYTORCH,
            images=[Path(f"/nonexistent/{i}.jpg") for i in (1, 2, 3, 4)],
            image_ids=[1, 2, 3, 4],
            gt_ann_path=str(gt_path),
            task="detect",
            warmup_passes=0,
        )
    assert partial.map_50_95 is not None
    assert partial.map_50_95 < 1.0, (
        f"a model detecting in 2 of the 4 images it evaluated scored "
        f"{partial.map_50_95!r} — degrading recall must lower the score"
    )


def test_the_table_shows_the_denominator_beside_the_map(capsys: pytest.CaptureFixture[str]) -> None:
    """A6: a reader must be able to see what the mAP was measured over.

    The 0.0419 defect shipped because a wrong number looks exactly like a right
    one. `mAP@0.5:0.95 0.0419` on its own is indistinguishable from a model
    that regressed; `0.0419 over 500 images` against a 5000-image dataset is
    a question a reader can ask.
    """
    from yowo.benchmark._report import render_table

    render_table(
        [
            BenchmarkResult(
                format="pytorch",
                requested_format="pytorch",
                map_50_95=0.4158,
                map_50=0.585,
                fps_avg=24.0,
                latency_p50_ms=41.0,
                latency_p95_ms=45.0,
                latency_p99_ms=48.0,
                model_size_mb=5.4,
                device="cpu",
                num_images=500,
            )
        ],
        model_name="yolo11n",
    )
    rendered = " ".join(capsys.readouterr().out.split())
    assert "500" in rendered, (
        f"the benchmark table reports an mAP without saying how many images it covers:\n{rendered}"
    )
