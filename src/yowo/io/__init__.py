"""Public surface for the io package.

Exports:
    FrameSource          — Protocol for all input sources.
    open_source          — factory that returns a FrameSource by dispatch.
    TensorMeta           — letterbox transform metadata for postprocessing.
    preprocess           — convert list[Frame] -> PreprocessedTensor.
    make_tensor_meta     — extract TensorMeta from a PreprocessedTensor.
    write_json           — serialize Detection list to JSON.
    write_annotated_frames — draw boxes on frames and save as JPEG.
"""

from yowo.io._decode import TensorMeta, make_tensor_meta, preprocess
from yowo.io._sink import write_annotated_frames, write_json
from yowo.io._source import FrameSource, open_source

__all__ = [
    "FrameSource",
    "TensorMeta",
    "make_tensor_meta",
    "open_source",
    "preprocess",
    "write_annotated_frames",
    "write_json",
]
