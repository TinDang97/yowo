"""Public surface for the postprocess package.

Exports:
    COCO_CLASSES — list of 80 standard COCO class name strings.
    postprocess  — decode raw model output into list[Detection].
"""

from yowo.postprocess._nms import COCO_CLASSES, postprocess

__all__ = [
    "COCO_CLASSES",
    "postprocess",
]
