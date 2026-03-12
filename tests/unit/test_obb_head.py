"""Tests for OBBHead, OBBBox, OBBDetection, and dist2rbox."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from yowo.arch._heads import OBBHead, dist2rbox
from yowo.types import OBBBox, OBBDetection


class TestOBBBox:
    def test_frozen_dataclass(self) -> None:
        box = OBBBox(cx=10.0, cy=20.0, w=5.0, h=3.0, angle=0.0, confidence=0.9, class_id=0)
        with pytest.raises(FrozenInstanceError):
            box.cx = 99.0  # type: ignore[misc]

    def test_has_angle_field(self) -> None:
        box = OBBBox(cx=10.0, cy=20.0, w=5.0, h=3.0, angle=0.5, confidence=0.9, class_id=1)
        assert isinstance(box.angle, float)
        assert box.angle == pytest.approx(0.5)

    def test_class_name_defaults_empty(self) -> None:
        box = OBBBox(cx=0.0, cy=0.0, w=1.0, h=1.0, angle=0.0, confidence=0.5, class_id=0)
        assert box.class_name == ""

    def test_full_construction(self) -> None:
        box = OBBBox(
            cx=100.0,
            cy=200.0,
            w=50.0,
            h=30.0,
            angle=-0.25 * math.pi,
            confidence=0.95,
            class_id=5,
            class_name="plane",
        )
        assert box.class_name == "plane"
        assert box.class_id == 5


class TestOBBDetection:
    def test_frozen_dataclass(self) -> None:
        det = OBBDetection(frame_index=0, source_id="test", boxes=())
        with pytest.raises(FrozenInstanceError):
            det.frame_index = 1  # type: ignore[misc]

    def test_empty_boxes(self) -> None:
        det = OBBDetection(frame_index=0, source_id="cam0", boxes=())
        assert det.boxes == ()
        assert det.inference_time_ms == 0.0

    def test_with_boxes(self) -> None:
        box = OBBBox(cx=10.0, cy=20.0, w=5.0, h=3.0, angle=0.0, confidence=0.8, class_id=2)
        det = OBBDetection(frame_index=5, source_id="cam1", boxes=(box,), inference_time_ms=12.5)
        assert len(det.boxes) == 1
        assert det.inference_time_ms == pytest.approx(12.5)


class TestDist2rbox:
    def test_zero_angle_identity(self) -> None:
        """At angle=0, rotation is identity: dist2rbox should return cx=anchor_x, cy=anchor_y."""
        B, A = 1, 4
        # pred_dist (B, 4, A): [left, top, right, bottom] distances
        pred_dist = torch.ones(B, 4, A)  # all 1.0: lt=(1,1), rb=(1,1)
        # angle=0 means cos=1, sin=0
        pred_angle = torch.zeros(B, 1, A)
        anchor_points = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]
        )  # (A, 2)

        result = dist2rbox(pred_dist, pred_angle, anchor_points)

        assert result.shape == (B, 4, A)
        # At angle=0: cx = (rb_x - lt_x)/2 * cos(0) - (rb_y - lt_y)/2 * sin(0) + anchor_x
        # = 0 * 1 - 0 * 0 + anchor_x = anchor_x
        # Similarly cy = anchor_y, w = lt_x + rb_x = 2, h = lt_y + rb_y = 2
        cx = result[0, 0, :]  # (A,)
        cy = result[0, 1, :]
        w = result[0, 2, :]
        h = result[0, 3, :]
        expected_cx = anchor_points[:, 0]
        expected_cy = anchor_points[:, 1]
        assert torch.allclose(cx, expected_cx, atol=1e-5), f"cx={cx} != {expected_cx}"
        assert torch.allclose(cy, expected_cy, atol=1e-5), f"cy={cy} != {expected_cy}"
        assert torch.allclose(w, torch.full((A,), 2.0), atol=1e-5)
        assert torch.allclose(h, torch.full((A,), 2.0), atol=1e-5)

    def test_output_shape(self) -> None:
        B, A = 2, 10
        pred_dist = torch.randn(B, 4, A)
        pred_angle = torch.randn(B, 1, A)
        anchor_points = torch.randn(A, 2)
        result = dist2rbox(pred_dist, pred_angle, anchor_points)
        assert result.shape == (B, 4, A)

    def test_pre_transposed_anchors(self) -> None:
        """anchors_t kwarg produces same result as computed internally."""
        B, A = 1, 5
        pred_dist = torch.randn(B, 4, A)
        pred_angle = torch.zeros(B, 1, A)
        anchor_points = torch.randn(A, 2)
        anchors_t = anchor_points.T.unsqueeze(0)  # (1, 2, A)
        r1 = dist2rbox(pred_dist, pred_angle, anchor_points)
        r2 = dist2rbox(pred_dist, pred_angle, anchor_points, anchors_t=anchors_t)
        assert torch.allclose(r1, r2)


class TestOBBHead:
    def _make_feats(self, ch: tuple[int, ...], input_size: int = 640) -> list[torch.Tensor]:
        """Create fake multi-scale feature maps."""
        # For 640 input: strides 8, 16, 32 → spatial 80, 40, 20
        strides = [8, 16, 32]
        feats = []
        for c, s in zip(ch, strides):
            h = w = input_size // s
            feats.append(torch.zeros(1, c, h, w))
        return feats

    def test_init_creates_cv4_with_nl_entries(self) -> None:
        ch = (64, 128, 256)
        head = OBBHead(nc=15, ch=ch)
        assert hasattr(head, "cv4")
        assert len(head.cv4) == len(ch)

    def test_ne_attribute(self) -> None:
        head = OBBHead(nc=15, ne=1, ch=(64, 128, 256))
        assert head.ne == 1

    def test_forward_output_shape(self) -> None:
        """OBBHead forward produces (1, 4+nc+ne, 8400) for 640 input."""
        ch = (64, 128, 256)
        nc = 15
        ne = 1
        head = OBBHead(nc=nc, ne=ne, ch=ch)
        head.eval()
        feats = self._make_feats(ch)
        with torch.no_grad():
            out = head(feats)
        # total anchors = 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
        expected_anchors = 80 * 80 + 40 * 40 + 20 * 20
        assert out.shape == (1, 4 + nc + ne, expected_anchors), (
            f"Expected (1, {4 + nc + ne}, {expected_anchors}), got {out.shape}"
        )

    def test_forward_output_shape_nano(self) -> None:
        """Nano scale channels produce correct output."""
        # Nano: width=0.25 → 256*0.25=64, 512*0.25=128, 1024*0.25=256
        ch = (64, 128, 256)
        head = OBBHead(nc=15, ch=ch)
        head.eval()
        feats = self._make_feats(ch)
        with torch.no_grad():
            out = head(feats)
        assert out.shape == (1, 20, 8400)

    def test_angle_encoding_range(self) -> None:
        """Angle branch output must be in [-pi/4, 3pi/4] range after encoding."""
        ch = (64, 128, 256)
        head = OBBHead(nc=15, ch=ch)
        head.eval()
        feats = self._make_feats(ch)
        with torch.no_grad():
            out = head(feats)
        # Last channel is angle: (sigmoid(x) - 0.25) * pi ∈ [-pi/4, 3pi/4]
        angle_out = out[0, -1, :]  # (8400,)
        assert angle_out.min() >= -math.pi / 4 - 1e-5
        assert angle_out.max() <= 3 * math.pi / 4 + 1e-5
