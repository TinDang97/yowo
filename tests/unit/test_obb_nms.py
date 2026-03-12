"""Tests for OBB NMS: probiou_matrix and postprocess_obb."""

from __future__ import annotations

import numpy as np
import torch

from yowo.postprocess._obb_nms import postprocess_obb, probiou_matrix
from yowo.types import Frame, ModelFamily, ModelSize, ModelSpec, OBBDetection, PreprocessedTensor


def _make_frame(frame_index: int = 0, source_id: str = "test") -> Frame:
    return Frame(
        pixels=np.zeros((640, 640, 3), dtype=np.uint8),
        source_id=source_id,
        frame_index=frame_index,
    )


class TestProbiouMatrix:
    def test_identical_boxes_iou_one(self) -> None:
        """Two identical xywhr boxes must have IoU >= 0.99."""
        box = torch.tensor([[320.0, 320.0, 100.0, 50.0, 0.0]])  # (1, 5): cx,cy,w,h,angle
        iou = probiou_matrix(box, box)
        assert iou.shape == (1, 1)
        assert float(iou[0, 0]) > 0.99, f"Expected IoU>=0.99, got {float(iou[0, 0])}"

    def test_non_overlapping_boxes_iou_near_zero(self) -> None:
        """Two far-apart xywhr boxes must have IoU < 0.01."""
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
        box2 = torch.tensor([[10000.0, 10000.0, 10.0, 10.0, 0.0]])
        iou = probiou_matrix(box1, box2)
        assert iou.shape == (1, 1)
        assert float(iou[0, 0]) < 0.01, f"Expected IoU<0.01, got {float(iou[0, 0])}"

    def test_output_shape_n_m(self) -> None:
        """probiou_matrix returns (N, M) for N and M input boxes."""
        boxes1 = torch.randn(3, 5).abs()
        boxes1[:, 4] = 0.0  # zero angle
        boxes2 = torch.randn(5, 5).abs()
        boxes2[:, 4] = 0.0
        iou = probiou_matrix(boxes1, boxes2)
        assert iou.shape == (3, 5)

    def test_iou_bounded_zero_one(self) -> None:
        """probiou values must be in [0, 1]."""
        boxes = torch.rand(4, 5)
        boxes[:, 2:4] = boxes[:, 2:4] * 100 + 1.0  # positive w, h
        boxes[:, 4] = boxes[:, 4] * 1.57  # angle in [0, pi/2]
        iou = probiou_matrix(boxes, boxes)
        assert iou.min() >= -1e-5
        assert iou.max() <= 1.0 + 1e-5


def _make_tensor_meta(batch: int = 1) -> PreprocessedTensor:
    """Identity transform tensor meta (no rescaling, no padding)."""
    return PreprocessedTensor(
        data=np.zeros((batch, 3, 640, 640), dtype=np.float32),
        original_shapes=tuple((640, 640) for _ in range(batch)),
        input_shape=(640, 640),
        scale_factors=tuple((1.0, 1.0) for _ in range(batch)),
        pad_offsets=tuple((0, 0) for _ in range(batch)),
    )


class TestPostprocessOBB:
    def _make_spec(self, nc: int = 15) -> ModelSpec:
        return ModelSpec(
            family=ModelFamily.YOLO11,
            size=ModelSize.NANO,
            task="obb",
            num_classes=nc,
        )

    def _make_raw(self, nc: int = 15, na: int = 8400, batch: int = 1) -> torch.Tensor:
        """Create raw OBBHead output (B, 4+nc+1, na) with all near-zero confidence."""
        return torch.zeros(batch, 4 + nc + 1, na)

    def test_no_detections_returns_empty_obb_detection(self) -> None:
        raw = self._make_raw()
        frames = [_make_frame()]
        spec = self._make_spec()
        results = postprocess_obb(raw, frames, spec, _make_tensor_meta())
        assert len(results) == 1
        det = results[0]
        assert isinstance(det, OBBDetection)
        assert det.boxes == ()
        assert det.frame_index == 0
        assert det.source_id == "test"

    def test_high_confidence_box_survives(self) -> None:
        """A single high-confidence box at anchor 0 should survive postprocess."""
        nc = 15
        raw = self._make_raw(nc=nc)
        # Set class 0 confidence to high at anchor 0 (raw logit=10 → sigmoid≈1.0)
        raw[0, 4, 0] = 10.0  # class 0 high logit
        # Set box coordinates at anchor 0
        raw[0, 0, 0] = 100.0  # cx
        raw[0, 1, 0] = 200.0  # cy
        raw[0, 2, 0] = 50.0  # w
        raw[0, 3, 0] = 30.0  # h

        frames = [_make_frame()]
        spec = self._make_spec(nc=nc)
        results = postprocess_obb(raw, frames, spec, _make_tensor_meta(), conf_threshold=0.5)
        assert len(results) == 1
        det = results[0]
        assert len(det.boxes) >= 1

    def test_nms_suppresses_duplicate_boxes(self) -> None:
        """Two boxes at same location (IoU>0.45) with same class → only 1 survives."""
        nc = 15
        raw = self._make_raw(nc=nc, na=8400)
        # Place two high-confidence boxes at anchors 0 and 1 with near-identical coords
        for anchor in [0, 1]:
            raw[0, 4, anchor] = 10.0  # class 0 high confidence
            raw[0, 0, anchor] = 320.0  # cx
            raw[0, 1, anchor] = 320.0  # cy
            raw[0, 2, anchor] = 100.0  # w
            raw[0, 3, anchor] = 80.0  # h

        frames = [_make_frame()]
        spec = self._make_spec(nc=nc)
        results = postprocess_obb(
            raw,
            frames,
            spec,
            _make_tensor_meta(),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )
        assert len(results) == 1
        # After NMS: two identical boxes → only 1 survives
        assert len(results[0].boxes) == 1

    def test_two_separated_boxes_both_survive(self) -> None:
        """Two boxes in different class lanes both survive NMS."""
        nc = 15
        raw = self._make_raw(nc=nc, na=8400)
        # Box 1: class 0, anchor 0
        raw[0, 4 + 0, 0] = 10.0  # class 0
        raw[0, 0, 0] = 50.0
        raw[0, 1, 0] = 50.0
        raw[0, 2, 0] = 20.0
        raw[0, 3, 0] = 20.0
        # Box 2: class 1, anchor 100 (different class → class-offset trick separates)
        raw[0, 4 + 1, 100] = 10.0  # class 1
        raw[0, 0, 100] = 60.0
        raw[0, 1, 100] = 60.0
        raw[0, 2, 100] = 20.0
        raw[0, 3, 100] = 20.0

        frames = [_make_frame()]
        spec = self._make_spec(nc=nc)
        results = postprocess_obb(
            raw,
            frames,
            spec,
            _make_tensor_meta(),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )
        assert len(results) == 1
        assert len(results[0].boxes) == 2

    def test_batch_results_match_frames(self) -> None:
        """postprocess_obb returns one OBBDetection per frame in batch."""
        nc = 15
        b_size = 3
        raw = self._make_raw(nc=nc, batch=b_size)
        frames = [_make_frame(frame_index=i, source_id=f"cam{i}") for i in range(b_size)]
        spec = self._make_spec(nc=nc)
        results = postprocess_obb(raw, frames, spec, _make_tensor_meta(batch=b_size))
        assert len(results) == b_size
        for i, det in enumerate(results):
            assert det.frame_index == i
            assert det.source_id == f"cam{i}"
