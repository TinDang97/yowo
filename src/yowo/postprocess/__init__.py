"""Public surface for the postprocess package.

Exports:
    COCO_CLASSES      — list of 80 standard COCO class name strings.
    PostprocessBuffer — reusable scratch buffer for postprocess hot path.
    postprocess       — decode raw model output into list[Detection].
"""

from yowo.postprocess._nms import COCO_CLASSES, PostprocessBuffer, postprocess

__all__ = [
    "COCO_CLASSES",
    "PostprocessBuffer",
    "postprocess",
]
