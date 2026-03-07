"""Benchmark evaluation: COCO mAP via pycocotools and ImageNet top-1 accuracy.

Provides official mAP computation using ``pycocotools.cocoeval.COCOeval``
and straightforward top-1 accuracy for classification models.

The YOLO-to-COCO category mapping (``YOLO_TO_COCO``) translates the
contiguous 0-79 YOLO class indices to the non-contiguous COCO category IDs
used by the official evaluation toolkit.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

from yowo.types import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOLO class index (0-79) -> COCO category ID mapping
# COCO has 80 categories but uses non-contiguous IDs (1-90).
# This maps the contiguous YOLO index to the official COCO category_id.
# ---------------------------------------------------------------------------
YOLO_TO_COCO: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)


def detections_to_coco_results(
    detections: list[Detection],
    image_ids: list[int],
) -> list[dict[str, object]]:
    """Convert YOWO Detection objects to COCO result format.

    Each detection is converted to a dict with keys:
    ``image_id``, ``category_id``, ``bbox`` (xywh), ``score``.

    Args:
        detections: One Detection per image (parallel with *image_ids*).
        image_ids: COCO image IDs corresponding to each detection.

    Returns:
        List of COCO-format result dicts ready for ``coco_gt.loadRes()``.
    """
    results: list[dict[str, object]] = []
    for det, img_id in zip(detections, image_ids, strict=True):
        for box in det.boxes:
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            w = x2 - x1
            h = y2 - y1
            coco_cat_id = (
                YOLO_TO_COCO[box.class_id] if box.class_id < len(YOLO_TO_COCO) else box.class_id
            )
            results.append(
                {
                    "image_id": img_id,
                    "category_id": coco_cat_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(box.confidence, 6),
                }
            )
    return results


def evaluate_coco_map(
    gt_ann_path: str | Path,
    predictions: list[dict[str, object]],
    subset: int | None = None,
) -> dict[str, float]:
    """Compute COCO mAP using official pycocotools COCOeval.

    Args:
        gt_ann_path: Path to COCO ``instances_val2017.json`` annotations file.
        predictions: List of COCO-format result dicts (from
            :func:`detections_to_coco_results`).
        subset: Unused (kept for API compatibility). Image scope is determined
            by the unique image IDs present in *predictions*.

    Returns:
        Dict with keys ``mAP_50_95``, ``mAP_50``, ``mAP_75``.

    Raises:
        ImportError: If pycocotools is not installed.
    """
    try:
        from pycocotools.coco import COCO  # type: ignore[import-untyped]
        from pycocotools.cocoeval import COCOeval  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO mAP evaluation. "
            "Install with: pip install pycocotools>=2.0.4 "
            "or: pip install yowo[benchmark]"
        ) from exc

    if not predictions:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_ann_path))

    # Restrict evaluation to only the images that have predictions so that
    # unscored images don't drag down recall across the full val set.
    evaluated_img_ids = sorted({int(p["image_id"]) for p in predictions})  # type: ignore[arg-type]

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)  # type: ignore[arg-type]

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = evaluated_img_ids

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
    }


def evaluate_imagenet_accuracy(
    predictions: list[tuple[int, int]],
    top_k: int = 1,
) -> dict[str, float]:
    """Compute ImageNet top-1 accuracy from (predicted, ground_truth) pairs.

    Args:
        predictions: List of ``(predicted_class_id, ground_truth_class_id)`` tuples.
        top_k: Unused, reserved for future top-k accuracy support.

    Returns:
        Dict with keys ``top1_accuracy``, ``total``, ``correct``.
    """
    if not predictions:
        return {"top1_accuracy": 0.0, "total": 0.0, "correct": 0.0}

    total = len(predictions)
    correct = sum(1 for pred, gt in predictions if pred == gt)
    return {
        "top1_accuracy": correct / total,
        "total": float(total),
        "correct": float(correct),
    }


def load_coco_dataset(
    data_path: str | Path,
    subset: int | None = None,
) -> tuple[list[Path], list[int], str]:
    """Load COCO val2017 dataset structure for benchmark evaluation.

    Expects the standard COCO layout::

        data_path/
            val2017/
                000000000139.jpg
                ...
            annotations/
                instances_val2017.json

    Args:
        data_path: Root directory of the COCO dataset.
        subset: If set, return only the first *subset* images.

    Returns:
        Tuple of (image_paths, image_ids, annotation_file_path).

    Raises:
        FileNotFoundError: If the expected directory structure is missing.
    """
    root = Path(data_path)
    val_dir = root / "val2017"
    ann_file = root / "annotations" / "instances_val2017.json"

    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"COCO val2017 images directory not found at: {val_dir}\n"
            f"Expected layout:\n"
            f"  {root}/\n"
            f"    val2017/\n"
            f"      000000000139.jpg\n"
            f"      ...\n"
            f"    annotations/\n"
            f"      instances_val2017.json"
        )

    if not ann_file.is_file():
        raise FileNotFoundError(
            f"COCO annotations file not found at: {ann_file}\n"
            f"Expected layout:\n"
            f"  {root}/\n"
            f"    val2017/\n"
            f"    annotations/\n"
            f"      instances_val2017.json"
        )

    # Load annotation file to get valid image IDs and filenames
    try:
        from pycocotools.coco import COCO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required. Install with: pip install yowo[benchmark]"
        ) from exc

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann_file))

    img_ids = sorted(coco.getImgIds())
    if subset is not None:
        img_ids = img_ids[:subset]

    img_infos = coco.loadImgs(img_ids)
    image_paths = [val_dir / info["file_name"] for info in img_infos]

    return image_paths, img_ids, str(ann_file)


def load_imagenet_dataset(
    data_path: str | Path,
    subset: int | None = None,
) -> tuple[list[Path], list[int]]:
    """Load ImageNet validation dataset from folder structure.

    Expects the standard torchvision ImageFolder layout::

        data_path/
            val/
                n01440764/
                    ILSVRC2012_val_00000293.JPEG
                    ...
                n01443537/
                    ...

    Class indices are assigned by sorting folder names alphabetically
    (standard torchvision convention).

    Args:
        data_path: Root directory containing the ``val/`` subfolder,
            or the ``val/`` directory itself.
        subset: If set, return only the first *subset* images.

    Returns:
        Tuple of (image_paths, class_labels).

    Raises:
        FileNotFoundError: If the expected directory structure is missing.
    """
    root = Path(data_path)

    # Accept both root/ (with val/ inside) and val/ directly
    val_dir = root / "val" if (root / "val").is_dir() else root

    class_dirs = sorted(
        [d for d in val_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in: {val_dir}\n"
            f"Expected ImageNet val layout:\n"
            f"  {root}/\n"
            f"    val/\n"
            f"      n01440764/\n"
            f"        ILSVRC2012_val_00000293.JPEG\n"
            f"        ...\n"
            f"      n01443537/\n"
            f"        ..."
        )

    class_to_idx: dict[str, int] = {d.name: idx for idx, d in enumerate(class_dirs)}
    image_paths: list[Path] = []
    class_labels: list[int] = []

    for cls_dir in class_dirs:
        cls_idx = class_to_idx[cls_dir.name]
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                image_paths.append(img_path)
                class_labels.append(cls_idx)

    if subset is not None:
        image_paths = image_paths[:subset]
        class_labels = class_labels[:subset]

    return image_paths, class_labels


__all__ = [
    "YOLO_TO_COCO",
    "detections_to_coco_results",
    "evaluate_coco_map",
    "evaluate_imagenet_accuracy",
    "load_coco_dataset",
    "load_imagenet_dataset",
]
