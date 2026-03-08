"""Public surface for the postprocess package.

Exports:
    COCO_CLASSES        — list of 80 standard COCO class name strings.
    DOTA_CLASSES        — list of 15 DOTA v1 class name strings.
    PostprocessBuffer   — reusable scratch buffer for postprocess hot path.
    postprocess         — decode raw model output into list[Detection].
    postprocess_classify — decode classification logits into list[ClassificationResult].
    postprocess_obb     — decode raw OBBHead output into list[OBBDetection].
    probiou_matrix      — pairwise probabilistic IoU for rotated boxes.
"""

from yowo.postprocess._classify import postprocess_classify
from yowo.postprocess._nms import COCO_CLASSES, PostprocessBuffer, postprocess
from yowo.postprocess._obb_nms import DOTA_CLASSES, postprocess_obb, probiou_matrix

__all__ = [
    "COCO_CLASSES",
    "DOTA_CLASSES",
    "PostprocessBuffer",
    "postprocess",
    "postprocess_classify",
    "postprocess_obb",
    "probiou_matrix",
]
