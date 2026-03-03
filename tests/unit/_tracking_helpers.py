"""Shared test helpers for tracking unit tests."""

from __future__ import annotations

import numpy as np

from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._strack import STrack
from yowo.types import BackendType, BoundingBox, Detection, Frame, ModelFamily, ModelSize, ModelSpec

SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
KF = KalmanFilterXYAH()


def make_frame(frame_index: int = 0) -> Frame:
    return Frame(
        pixels=np.zeros((480, 640, 3), dtype=np.uint8),
        source_id="test",
        frame_index=frame_index,
    )


def make_box(
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 200.0,
    conf: float = 0.9,
    cls_id: int = 0,
    cls_name: str = "person",
) -> BoundingBox:
    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
    )


def make_detection(
    boxes: tuple[BoundingBox, ...] = (),
    frame_index: int = 0,
) -> Detection:
    return Detection(
        frame=make_frame(frame_index=frame_index),
        boxes=boxes,
        inference_time_ms=5.0,
        backend=BackendType.ONNX,
        model_spec=SPEC,
    )


def make_strack(
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 200.0,
    track_id: int = 1,
    min_hits: int = 3,
) -> STrack:
    return STrack(
        track_id=track_id,
        box_xyxy=(x1, y1, x2, y2),
        confidence=0.9,
        class_id=0,
        class_name="person",
        kalman=KF,
        min_hits=min_hits,
    )
