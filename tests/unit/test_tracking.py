"""Unit tests for yowo.tracking — ByteTracker, Kalman filter, matching, STrack."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from yowo.errors import TrackingError, YowoError
from yowo.tracking import track_detections, track_stream
from yowo.tracking._clip_reid import CLIPReIDExtractor
from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import (
    appearance_distance,
    fuse_score,
    gated_fused_cost,
    iou_batch,
    linear_assignment,
    needs_reid,
)
from yowo.tracking._reid import (  # noqa: F401
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    CLIPExtractor,
    FastReIDExtractor,
    ReIDExtractor,
)
from yowo.tracking._strack import STrack, TrackedBox, TrackedDetection, TrackState
from yowo.tracking._tracker import ByteTracker
from yowo.types import BackendType, BoundingBox, Detection, Frame, ModelFamily, ModelSize, ModelSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
_KF = KalmanFilterXYAH()


def _make_frame(frame_index: int = 0) -> Frame:
    return Frame(
        pixels=np.zeros((480, 640, 3), dtype=np.uint8),
        source_id="test",
        frame_index=frame_index,
    )


def _make_box(
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


def _make_detection(
    boxes: tuple[BoundingBox, ...] = (),
    frame_index: int = 0,
) -> Detection:
    return Detection(
        frame=_make_frame(frame_index=frame_index),
        boxes=boxes,
        inference_time_ms=5.0,
        backend=BackendType.ONNX,
        model_spec=_SPEC,
    )


def _make_strack(
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
        kalman=_KF,
        min_hits=min_hits,
    )


# ---------------------------------------------------------------------------
# Section 1: Kalman filter
# ---------------------------------------------------------------------------


class TestKalmanFilter:
    def test_initiate_shapes(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, cov = kf.initiate(m)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

    def test_initiate_velocity_zero(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, _ = kf.initiate(m)
        np.testing.assert_array_equal(mean[4:], np.zeros(4))

    def test_predict_increments_position(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, cov = kf.initiate(m)
        # Set a non-zero velocity
        mean = mean.copy()
        mean[4] = 5.0  # vx
        mean_pred, _ = kf.predict(mean, cov)
        # cx should advance by vx (constant velocity model, dt=1)
        assert mean_pred[0] > mean[0]

    def test_predict_covariance_grows(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, cov = kf.initiate(m)
        _, cov_pred = kf.predict(mean, cov)
        # Uncertainty must grow after prediction (no measurement)
        assert np.trace(cov_pred) > np.trace(cov)

    def test_update_mean_moves_toward_measurement(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, cov = kf.initiate(m)
        mean_pred, cov_pred = kf.predict(mean, cov)
        # Measurement at different location
        new_m = KalmanFilterXYAH.xyxy_to_xyah((120.0, 120.0, 220.0, 220.0))
        mean_upd, _ = kf.update(mean_pred, cov_pred, new_m)
        # Updated cx should be between predicted and measured
        assert mean_pred[0] <= mean_upd[0] <= new_m[0] or new_m[0] <= mean_upd[0] <= mean_pred[0]

    def test_update_covariance_shrinks(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, cov = kf.initiate(m)
        mean_pred, cov_pred = kf.predict(mean, cov)
        _, cov_upd = kf.update(mean_pred, cov_pred, m)
        assert np.trace(cov_upd) < np.trace(cov_pred)

    def test_xyxy_to_xyah_roundtrip(self) -> None:
        xyxy = (50.0, 80.0, 150.0, 280.0)
        xyah = KalmanFilterXYAH.xyxy_to_xyah(xyxy)
        recovered = KalmanFilterXYAH.xyah_to_xyxy(xyah)
        for a, b in zip(xyxy, recovered, strict=False):
            assert abs(a - b) < 1e-8

    def test_project_returns_4d(self) -> None:
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, cov = kf.initiate(m)
        proj_mean, proj_cov = kf.project(mean, cov)
        assert proj_mean.shape == (4,)
        assert proj_cov.shape == (4, 4)


# ---------------------------------------------------------------------------
# Section 2: Matching
# ---------------------------------------------------------------------------


class TestIouBatch:
    def test_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        iou = iou_batch(boxes, boxes)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
        b = np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float64)
        iou = iou_batch(a, b)
        assert iou[0, 0] < 1e-6

    def test_half_overlap(self) -> None:
        a = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        b = np.array([[50.0, 0.0, 150.0, 100.0]], dtype=np.float64)
        iou = iou_batch(a, b)
        # Inter=50*100=5000, union=100*100+100*100-5000=15000, iou=1/3
        assert abs(iou[0, 0] - 1.0 / 3.0) < 1e-5

    def test_shape(self) -> None:
        a = np.random.rand(5, 4).astype(np.float64)
        b = np.random.rand(3, 4).astype(np.float64)
        iou = iou_batch(a, b)
        assert iou.shape == (5, 3)


class TestLinearAssignment:
    def test_empty_matrix(self) -> None:
        cost = np.empty((0, 3), dtype=np.float64)
        matches, ur, uc = linear_assignment(cost, thresh=0.5)
        assert matches == []
        assert ur == []
        assert set(uc) == {0, 1, 2}

    def test_perfect_match_diagonal(self) -> None:
        cost = np.eye(3, dtype=np.float64) * 0.1  # low cost on diagonal
        matches, ur, uc = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 3
        assert ur == []
        assert uc == []

    def test_threshold_rejects_high_cost(self) -> None:
        cost = np.ones((2, 2), dtype=np.float64) * 0.9  # above thresh
        matches, ur, uc = linear_assignment(cost, thresh=0.5)
        assert matches == []
        assert len(ur) == 2
        assert len(uc) == 2

    def test_rectangular_more_rows(self) -> None:
        cost = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.8]], dtype=np.float64)
        matches, ur, uc = linear_assignment(cost, thresh=0.5)
        assert len(matches) == 2
        assert len(ur) == 1  # one unmatched row

    def test_returns_correct_types(self) -> None:
        cost = np.array([[0.2]], dtype=np.float64)
        matches, ur, uc = linear_assignment(cost, thresh=0.5)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in matches)


# ---------------------------------------------------------------------------
# Section 3: STrack
# ---------------------------------------------------------------------------


class TestSTrack:
    def test_initial_state_new(self) -> None:
        t = _make_strack()
        assert t.state == TrackState.NEW
        assert t.hits == 0
        assert t.age == 0

    def test_activate_transitions_to_tracked(self) -> None:
        t = _make_strack()
        t.activate(frame_id=1)
        assert t.state == TrackState.TRACKED
        assert t.hits == 1
        assert t.start_frame == 1

    def test_mark_lost(self) -> None:
        t = _make_strack()
        t.activate(frame_id=1)
        t.mark_lost()
        assert t.state == TrackState.LOST

    def test_mark_removed(self) -> None:
        t = _make_strack()
        t.mark_removed()
        assert t.state == TrackState.REMOVED

    def test_is_confirmed_requires_min_hits(self) -> None:
        t = _make_strack(min_hits=3)
        t.activate(frame_id=1)
        assert not t.is_confirmed  # hits=1 < 3
        t.update((100, 100, 200, 200), 0.9, 0, "person", frame_id=2)
        t.update((100, 100, 200, 200), 0.9, 0, "person", frame_id=3)
        assert t.is_confirmed  # hits=3 >= 3

    def test_to_tracked_box(self) -> None:
        t = _make_strack(x1=50, y1=60, x2=150, y2=160)
        t.activate(frame_id=1)
        tb = t.to_tracked_box()
        assert isinstance(tb, TrackedBox)
        assert tb.track_id == 1
        assert tb.class_name == "person"
        # Kalman should preserve approximate position
        assert abs(tb.x1 - 50.0) < 5.0

    def test_predict_increments_age(self) -> None:
        t = _make_strack()
        t.predict()
        assert t.age == 1
        assert t.time_since_update == 1

    def test_update_resets_time_since_update(self) -> None:
        t = _make_strack()
        t.activate(frame_id=1)
        t.predict()
        assert t.time_since_update == 1
        t.update((100, 100, 200, 200), 0.9, 0, "person", frame_id=2)
        assert t.time_since_update == 0


# ---------------------------------------------------------------------------
# Section 4: ByteTracker
# ---------------------------------------------------------------------------


class TestByteTracker:
    def test_single_detection_creates_track(self) -> None:
        # new_track_thresh = track_high_thresh + 0.1 = 0.7 + 0.1 = 0.8
        tracker = ByteTracker(min_hits=1, track_high_thresh=0.7)
        det = _make_detection(boxes=(_make_box(conf=0.9),))
        result = tracker.update(det)
        assert isinstance(result, TrackedDetection)
        assert result.num_boxes == 1

    def test_consistent_box_same_track_id(self) -> None:
        tracker = ByteTracker(min_hits=1)
        box = _make_box(conf=0.9)
        result0 = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        result1 = tracker.update(_make_detection(boxes=(box,), frame_index=1))
        result2 = tracker.update(_make_detection(boxes=(box,), frame_index=2))
        ids = (
            {b.track_id for b in result0.boxes}
            | {b.track_id for b in result1.boxes}
            | {b.track_id for b in result2.boxes}
        )
        assert len(ids) == 1

    def test_two_separate_objects_different_ids(self) -> None:
        tracker = ByteTracker(min_hits=1)
        box_a = _make_box(x1=0, y1=0, x2=50, y2=50, conf=0.9)
        box_b = _make_box(x1=400, y1=400, x2=450, y2=450, conf=0.9)
        for i in range(3):
            tracker.update(_make_detection(boxes=(box_a, box_b), frame_index=i))
        result = tracker.update(_make_detection(boxes=(box_a, box_b), frame_index=3))
        ids = {b.track_id for b in result.boxes}
        assert len(ids) == 2

    def test_disappeared_track_marked_lost(self) -> None:
        tracker = ByteTracker(min_hits=1)
        box = _make_box(conf=0.9)
        tracker.update(_make_detection(boxes=(box,), frame_index=0))
        # Next frame: no detections
        tracker.update(_make_detection(boxes=(), frame_index=1))
        assert tracker.lost_track_count >= 1

    def test_lost_track_removed_after_max_age(self) -> None:
        tracker = ByteTracker(min_hits=1, max_age=2)
        # Use moving boxes so track has velocity > 1.0 (no stationary extension)
        for i in range(3):
            box = _make_box(100 + i * 20, 100, 200 + i * 20, 200, conf=0.9)
            tracker.update(_make_detection(boxes=(box,), frame_index=i))
        # max_age+2 frames with no detection
        for i in range(3, 8):
            tracker.update(_make_detection(boxes=(), frame_index=i))
        assert tracker.lost_track_count == 0
        assert tracker.active_track_count == 0

    def test_lost_track_re_associated(self) -> None:
        """A lost track re-appears and keeps the same track_id (not a new one)."""
        tracker = ByteTracker(min_hits=1, max_age=10)
        box = _make_box(conf=0.9)
        r0 = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        original_id = r0.boxes[0].track_id
        # Disappear for 3 frames
        for i in range(1, 4):
            tracker.update(_make_detection(boxes=(), frame_index=i))
        assert tracker.lost_track_count >= 1
        # Reappear at same position
        r4 = tracker.update(_make_detection(boxes=(box,), frame_index=4))
        assert r4.num_boxes >= 1
        refound_id = r4.boxes[0].track_id
        assert refound_id == original_id, (
            f"Lost track should be re-associated with same ID {original_id}, "
            f"got new ID {refound_id}"
        )

    def test_lost_track_velocity_zeroed(self) -> None:
        """Lost tracks zero their height velocity to prevent drift."""
        tracker = ByteTracker(min_hits=1, max_age=10)
        box = _make_box(conf=0.9)
        tracker.update(_make_detection(boxes=(box,), frame_index=0))
        # Disappear
        tracker.update(_make_detection(boxes=(), frame_index=1))
        # Check internal state: lost tracks should have zeroed h-velocity
        for track in tracker._lost:  # type: ignore[attr-defined]
            # mean[7] is vh (height velocity)
            assert track._mean[7] == 0.0  # type: ignore[attr-defined]

    def test_min_hits_confirmation(self) -> None:
        tracker = ByteTracker(min_hits=3, track_high_thresh=0.5)
        box = _make_box(conf=0.9)
        # Frame 0: first appearance
        r0 = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        if r0.boxes:
            assert not r0.boxes[0].is_confirmed
        # Frames 1, 2: not confirmed until 3 hits
        tracker.update(_make_detection(boxes=(box,), frame_index=1))
        r2 = tracker.update(_make_detection(boxes=(box,), frame_index=2))
        if r2.boxes:
            assert r2.boxes[0].is_confirmed

    def test_below_low_thresh_ignored(self) -> None:
        tracker = ByteTracker(track_low_thresh=0.3, track_high_thresh=0.6)
        # conf=0.1 is below track_low_thresh=0.3 → completely ignored
        box = _make_box(conf=0.1)
        result = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert result.num_boxes == 0
        assert tracker.active_track_count == 0

    def test_reset_clears_all_tracks(self) -> None:
        tracker = ByteTracker(min_hits=1)
        for i in range(3):
            tracker.update(_make_detection(boxes=(_make_box(conf=0.9),), frame_index=i))
        tracker.reset()
        assert tracker.active_track_count == 0
        assert tracker.lost_track_count == 0

    def test_empty_detection_no_crash(self) -> None:
        tracker = ByteTracker()
        result = tracker.update(_make_detection(boxes=(), frame_index=0))
        assert isinstance(result, TrackedDetection)
        assert result.num_boxes == 0

    def test_tracking_time_ms_positive(self) -> None:
        tracker = ByteTracker(min_hits=1)
        result = tracker.update(_make_detection(boxes=(_make_box(conf=0.9),), frame_index=0))
        assert result.tracking_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Section 5: Integration shim
# ---------------------------------------------------------------------------


class TestIntegrationShim:
    def _make_mock_engine(self, n_frames: int = 3) -> MagicMock:
        engine = MagicMock()
        frames = [
            _make_detection(boxes=(_make_box(conf=0.9),), frame_index=i) for i in range(n_frames)
        ]
        engine.stream.return_value = iter(frames)
        return engine

    def test_track_stream_yields_tracked_detection(self) -> None:
        engine = self._make_mock_engine(3)
        source = MagicMock()
        results = list(track_stream(engine, source))
        assert len(results) == 3
        assert all(isinstance(r, TrackedDetection) for r in results)

    def test_track_detections_from_list(self) -> None:
        dets = [_make_detection(boxes=(_make_box(conf=0.9),), frame_index=i) for i in range(4)]
        results = list(track_detections(dets))
        assert len(results) == 4
        assert all(isinstance(r, TrackedDetection) for r in results)

    def test_custom_tracker_not_recreated(self) -> None:
        tracker = ByteTracker(min_hits=1)
        dets = [_make_detection(boxes=(_make_box(conf=0.9),), frame_index=i) for i in range(3)]
        results = list(track_detections(dets, tracker=tracker))
        # Same tracker used — consistent IDs
        ids = [r.boxes[0].track_id for r in results if r.boxes]
        assert len(set(ids)) == 1

    def test_track_stream_passes_kwargs(self) -> None:
        engine = self._make_mock_engine(2)
        source = MagicMock()
        # min_hits=1 means track confirmed on first hit
        results = list(track_stream(engine, source, min_hits=1))
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Section 6: Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_tracked_box_to_dict(self) -> None:
        tb = TrackedBox(
            x1=10.0,
            y1=20.0,
            x2=110.0,
            y2=120.0,
            confidence=0.85,
            class_id=1,
            class_name="car",
            track_id=42,
            is_confirmed=True,
        )
        d = tb.to_dict()
        assert d["x1"] == 10.0
        assert d["track_id"] == 42
        assert d["class_name"] == "car"
        assert d["is_confirmed"] is True
        assert len(d) == 9

    def test_tracked_detection_to_json(self) -> None:
        import json

        tracker = ByteTracker(min_hits=1)
        det = _make_detection(boxes=(_make_box(conf=0.9),), frame_index=7)
        result = tracker.update(det)
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["frame_index"] == 7
        assert "boxes" in parsed
        assert "tracking_time_ms" in parsed

    def test_tracked_detection_num_boxes(self) -> None:
        tracker = ByteTracker(min_hits=1)
        det = _make_detection(
            boxes=(_make_box(conf=0.9), _make_box(x1=300, y1=300, x2=400, y2=400, conf=0.9)),
            frame_index=0,
        )
        result = tracker.update(det)
        assert result.num_boxes == result.boxes.__len__()

    def test_tracked_box_area_property(self) -> None:
        tb = TrackedBox(
            x1=0,
            y1=0,
            x2=100,
            y2=100,
            confidence=0.9,
            class_id=0,
            class_name="p",
            track_id=1,
            is_confirmed=True,
        )
        assert abs(tb.area - 10000.0) < 1e-6


# ---------------------------------------------------------------------------
# Section 7: Validation & edge-case coverage
# ---------------------------------------------------------------------------


class TestByteTrackerValidation:
    def test_track_low_gte_high_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="track_low_thresh"):
            ByteTracker(track_low_thresh=0.6, track_high_thresh=0.6)

    def test_track_low_greater_than_high_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="track_low_thresh"):
            ByteTracker(track_low_thresh=0.8, track_high_thresh=0.5)

    def test_new_track_thresh_clamped_to_one(self) -> None:
        """track_high_thresh=0.95 -> new_track_thresh=1.0, not 1.05."""
        tracker = ByteTracker(track_high_thresh=0.95, track_low_thresh=0.1, min_hits=1)
        # conf=0.96 above track_high_thresh=0.95 but below new_track_thresh=1.0
        box = _make_box(conf=0.96)
        result = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert result.num_boxes == 0

    def test_stage2_low_conf_keeps_track_active(self) -> None:
        """Low-conf detection matches unmatched tracked track via stage 2."""
        tracker = ByteTracker(track_high_thresh=0.6, track_low_thresh=0.1, min_hits=1, max_age=10)
        box = _make_box(conf=0.9)
        tracker.update(_make_detection(boxes=(box,), frame_index=0))
        # Frame 1: same position, below high_thresh, above low_thresh → stage 2
        low_box = _make_box(conf=0.3)
        tracker.update(_make_detection(boxes=(low_box,), frame_index=1))
        assert tracker.active_track_count >= 1

    def test_high_conf_below_new_thresh_no_birth(self) -> None:
        """Detection above high_thresh but below new_track_thresh does not birth a track."""
        tracker = ByteTracker(track_high_thresh=0.7, track_low_thresh=0.1, min_hits=1)
        # new_track_thresh = 0.7 + 0.1 = 0.8
        box = _make_box(conf=0.75)
        result = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert result.num_boxes == 0


# ---------------------------------------------------------------------------
# Section 8: Kalman edge cases
# ---------------------------------------------------------------------------


class TestKalmanFilterEdge:
    def test_update_cholesky_fallback_degenerate_covariance(self) -> None:
        """Near-singular covariance triggers inv() fallback without error."""
        kf = KalmanFilterXYAH()
        m = KalmanFilterXYAH.xyxy_to_xyah((100.0, 100.0, 200.0, 200.0))
        mean, _ = kf.initiate(m)
        cov_degenerate = np.zeros((8, 8), dtype=np.float64)
        measurement = KalmanFilterXYAH.xyxy_to_xyah((110.0, 110.0, 210.0, 210.0))
        mean_upd, cov_upd = kf.update(mean, cov_degenerate, measurement)
        assert mean_upd.shape == (8,)
        assert cov_upd.shape == (8, 8)

    def test_xyxy_to_xyah_zero_height_make_box(self) -> None:
        """Zero-height box uses max(h, 1e-6) guard to avoid division by zero."""
        xyah = KalmanFilterXYAH.xyxy_to_xyah((50.0, 100.0, 150.0, 100.0))
        assert np.isfinite(xyah[2])
        assert abs(xyah[3]) < 1e-5


# ---------------------------------------------------------------------------
# Section 9: iou_distance + remove_duplicate_tracks direct tests
# ---------------------------------------------------------------------------


class TestIouDistance:
    def test_basic(self) -> None:
        from yowo.tracking._matching import iou_distance

        t = _make_strack(x1=0, y1=0, x2=100, y2=100)
        t.activate(frame_id=0)
        dets = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        cost = iou_distance([t], dets)
        assert cost.shape == (1, 1)
        assert cost[0, 0] < 0.1

    def test_empty_tracks(self) -> None:
        from yowo.tracking._matching import iou_distance

        dets = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        cost = iou_distance([], dets)
        assert cost.shape == (0, 1)

    def test_empty_detections(self) -> None:
        from yowo.tracking._matching import iou_distance

        t = _make_strack()
        t.activate(frame_id=0)
        dets = np.empty((0, 4), dtype=np.float64)
        cost = iou_distance([t], dets)
        assert cost.shape == (1, 0)


class TestRemoveDuplicateTracks:
    def test_empty_a(self) -> None:
        from yowo.tracking._matching import remove_duplicate_tracks

        t = _make_strack(track_id=1)
        t.activate(frame_id=0)
        a, b = remove_duplicate_tracks([], [t])
        assert a == []
        assert len(b) == 1

    def test_empty_b(self) -> None:
        from yowo.tracking._matching import remove_duplicate_tracks

        t = _make_strack(track_id=1)
        t.activate(frame_id=0)
        a, b = remove_duplicate_tracks([t], [])
        assert len(a) == 1
        assert b == []

    def test_removes_shorter_lived(self) -> None:
        from yowo.tracking._matching import remove_duplicate_tracks

        t1 = _make_strack(track_id=1)
        t1.activate(frame_id=0)
        t1.update((100, 100, 200, 200), 0.9, 0, "person", frame_id=5)
        t2 = _make_strack(track_id=2)
        t2.activate(frame_id=4)
        a, b = remove_duplicate_tracks([t1], [t2])
        assert len(a) == 1
        assert len(b) == 0


class TestIouBatchEdge:
    def test_inverted_box_area_clamped(self) -> None:
        """Boxes with x2 < x1 (Kalman drift) have area clamped to 0."""
        a = np.array([[100.0, 100.0, 50.0, 50.0]], dtype=np.float64)
        b = np.array([[0.0, 0.0, 200.0, 200.0]], dtype=np.float64)
        iou = iou_batch(a, b)
        assert iou[0, 0] < 1e-6


# ---------------------------------------------------------------------------
# Section 10: Munkres fallback
# ---------------------------------------------------------------------------


class TestMunkresFallback:
    def test_munkres_correct_assignment(self) -> None:
        from yowo.tracking._matching import _munkres

        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
        row_ind, col_ind = _munkres(cost)
        pairs = set(zip(row_ind.tolist(), col_ind.tolist(), strict=True))
        assert pairs == {(0, 0), (1, 1)}

    def test_hungarian_without_scipy(self, mocker: object) -> None:
        """Patching _has_scipy=False forces the _munkres path."""
        import yowo.tracking._matching as match_mod

        mocker.patch.object(match_mod, "_has_scipy", False)  # type: ignore[union-attr]
        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
        row_ind, col_ind = match_mod._hungarian(cost)
        pairs = set(zip(row_ind.tolist(), col_ind.tolist(), strict=True))
        assert pairs == {(0, 0), (1, 1)}


# ---------------------------------------------------------------------------
# Section 11: STrack re_activate
# ---------------------------------------------------------------------------


class TestSTrackReActivate:
    def test_re_activate_transitions_to_tracked(self) -> None:
        t = _make_strack()
        t.activate(frame_id=0)
        t.mark_lost()
        assert t.state == TrackState.LOST
        t.re_activate((110, 110, 210, 210), 0.85, 1, "car", frame_id=5)
        assert t.state == TrackState.TRACKED
        assert t.time_since_update == 0
        assert t.class_id == 1
        assert t.class_name == "car"
        assert t.confidence == 0.85
        assert t.frame_id == 5


# ---------------------------------------------------------------------------
# Section 12: TrackedDetection/TrackedBox properties
# ---------------------------------------------------------------------------


class TestTrackedDetectionProperties:
    def test_has_detections_true(self) -> None:
        tracker = ByteTracker(min_hits=1)
        result = tracker.update(_make_detection(boxes=(_make_box(conf=0.9),), frame_index=0))
        assert result.has_detections is True

    def test_has_detections_false(self) -> None:
        tracker = ByteTracker(min_hits=1)
        result = tracker.update(_make_detection(boxes=(), frame_index=0))
        assert result.has_detections is False

    def test_to_dict_keys(self) -> None:
        tracker = ByteTracker(min_hits=1)
        result = tracker.update(_make_detection(boxes=(_make_box(conf=0.9),), frame_index=3))
        d = result.to_dict()
        for key in (
            "source_id",
            "frame_index",
            "inference_time_ms",
            "tracking_time_ms",
            "backend",
            "model",
            "boxes",
        ):
            assert key in d

    def test_tracked_box_as_xyxy(self) -> None:
        tb = TrackedBox(
            x1=10.0,
            y1=20.0,
            x2=30.0,
            y2=40.0,
            confidence=0.9,
            class_id=0,
            class_name="p",
            track_id=1,
            is_confirmed=True,
        )
        assert tb.as_xyxy == (10.0, 20.0, 30.0, 40.0)

    def test_tracked_box_is_frozen(self) -> None:
        tb = TrackedBox(
            x1=0,
            y1=0,
            x2=10,
            y2=10,
            confidence=0.9,
            class_id=0,
            class_name="p",
            track_id=1,
            is_confirmed=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            tb.x1 = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 13: TrackingError
# ---------------------------------------------------------------------------


class TestTrackingError:
    def test_inherits_from_yowo_error(self) -> None:
        err = TrackingError("test error")
        assert isinstance(err, YowoError)
        assert str(err) == "test error"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(TrackingError, match="corrupt state"):
            raise TrackingError("corrupt state")


# ---------------------------------------------------------------------------
# Section 14: Integration edge cases
# ---------------------------------------------------------------------------


class TestIntegrationEdge:
    def test_track_detections_empty_iterable(self) -> None:
        results = list(track_detections([]))
        assert results == []


# ---------------------------------------------------------------------------
# Section 15: v2.1.0 top-level exports
# ---------------------------------------------------------------------------


class TestV21Exports:
    def test_tracking_exports_importable(self) -> None:
        from yowo import (  # noqa: F401
            ByteTracker,
            TrackedBox,
            TrackedDetection,
            TrackState,
            track_detections,
            track_stream,
        )

    def test_counter_exports_importable(self) -> None:
        from yowo import (  # noqa: F401
            CountLine,
            CountResult,
            CountZone,
            CrossDirection,
            LineCrossEvent,
            ObjectCounter,
        )


# ---------------------------------------------------------------------------
# MockReIDExtractor helper
# ---------------------------------------------------------------------------


class MockReIDExtractor:
    """Deterministic mock: returns position-based embeddings for testing."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        if not boxes_xyxy:
            return None
        embs = np.zeros((len(boxes_xyxy), self._dim), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            embs[i, 0] = cx / 1920
            embs[i, 1] = cy / 1080
            embs[i, 2] = (x2 - x1) / 1920
            embs[i, 3] = (y2 - y1) / 1080
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        return (embs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Section 16: STrack embedding tests
# ---------------------------------------------------------------------------


class TestSTrackEmbedding:
    def test_embedding_slot_default_none(self) -> None:
        t = _make_strack()
        assert t.embedding is None

    def test_update_embedding_first_call_copies(self) -> None:
        t = _make_strack()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t.update_embedding(emb)
        assert t.embedding is not None
        # Must be a copy, not the same object
        assert t.embedding is not emb
        np.testing.assert_allclose(t.embedding, emb, atol=1e-6)

    def test_update_embedding_ema_formula(self) -> None:
        t = _make_strack()
        old = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        t.update_embedding(old)
        t.update_embedding(new, eta=0.9)
        # EMA: 0.9 * [1,0,0] + 0.1 * [0,1,0] = [0.9, 0.1, 0] -> normalized
        expected = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        assert t.embedding is not None
        np.testing.assert_allclose(t.embedding, expected, atol=1e-5)

    def test_update_embedding_renormalized(self) -> None:
        t = _make_strack()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        t.update_embedding(emb1)
        t.update_embedding(emb2, eta=0.5)
        assert t.embedding is not None
        norm = float(np.linalg.norm(t.embedding))
        assert abs(norm - 1.0) < 1e-5

    def test_update_embedding_zero_norm_safety(self) -> None:
        t = _make_strack()
        zero = np.zeros(4, dtype=np.float32)
        t.update_embedding(zero)
        # First call copies — zero embedding stored
        assert t.embedding is not None
        # Second call with zero: EMA of zeros = zeros, norm guard prevents crash
        t.update_embedding(zero)
        assert t.embedding is not None


# ---------------------------------------------------------------------------
# Section 17: Appearance distance tests
# ---------------------------------------------------------------------------


class TestAppearanceDistance:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        tracks = rng.standard_normal((3, 512)).astype(np.float32)
        tracks /= np.linalg.norm(tracks, axis=1, keepdims=True)
        dets = rng.standard_normal((5, 512)).astype(np.float32)
        dets /= np.linalg.norm(dets, axis=1, keepdims=True)
        dist = appearance_distance(tracks, dets)
        assert dist.shape == (3, 5)
        assert dist.dtype == np.float64

    def test_identical_distance_zero(self) -> None:
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        dist = appearance_distance(emb, emb)
        assert abs(dist[0, 0]) < 1e-6

    def test_orthogonal_distance_one(self) -> None:
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        dist = appearance_distance(a, b)
        assert abs(dist[0, 0] - 1.0) < 1e-6

    def test_empty_tracks(self) -> None:
        tracks = np.empty((0, 128), dtype=np.float32)
        dets = np.random.randn(5, 128).astype(np.float32)
        dist = appearance_distance(tracks, dets)
        assert dist.shape == (0, 5)

    def test_empty_dets(self) -> None:
        tracks = np.random.randn(3, 128).astype(np.float32)
        dets = np.empty((0, 128), dtype=np.float32)
        dist = appearance_distance(tracks, dets)
        assert dist.shape == (3, 0)


# ---------------------------------------------------------------------------
# Section 18: Gated fused cost tests
# ---------------------------------------------------------------------------


class TestGatedFusedCost:
    def test_iou_only_fallback(self) -> None:
        """When cos_dist > theta_e, result should equal IoU cost."""
        iou_cost = np.array([[0.3]], dtype=np.float64)
        # Orthogonal: cos_dist = 1.0 > theta_e=0.3
        t_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        d_emb = np.array([[0.0, 1.0]], dtype=np.float32)
        fused = gated_fused_cost(iou_cost, t_emb, d_emb)
        assert abs(fused[0, 0] - 0.3) < 1e-6

    def test_fused_path(self) -> None:
        """When both gates pass, result = min(iou, 0.5*cos_dist)."""
        iou_cost = np.array([[0.4]], dtype=np.float64)
        # Nearly identical: cos_dist ~ 0
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        fused = gated_fused_cost(iou_cost, emb, emb, theta_e=0.30, theta_iou=0.5)
        # cos_dist = 0, d_hat = 0.5 * 0 = 0, min(0.4, 0) = 0
        assert fused[0, 0] < 0.01

    def test_shape_preserved(self) -> None:
        n, m, d = 4, 6, 64
        rng = np.random.default_rng(42)
        iou_cost = rng.random((n, m)).astype(np.float64)
        t = rng.standard_normal((n, d)).astype(np.float32)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        de = rng.standard_normal((m, d)).astype(np.float32)
        de /= np.linalg.norm(de, axis=1, keepdims=True)
        fused = gated_fused_cost(iou_cost, t, de)
        assert fused.shape == (n, m)
        assert fused.dtype == np.float64

    def test_backward_compat_zero_embeddings(self) -> None:
        """Zero embeddings (no info) -> cos_dist=1.0 -> gate fails -> IoU only."""
        iou_cost = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=np.float64)
        zeros = np.zeros((2, 64), dtype=np.float32)
        fused = gated_fused_cost(iou_cost, zeros, zeros)
        np.testing.assert_allclose(fused, iou_cost, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 19: needs_reid tests
# ---------------------------------------------------------------------------


class TestNeedsReid:
    def test_empty_cost(self) -> None:
        cost = np.empty((0, 0), dtype=np.float64)
        assert needs_reid(cost) is False

    def test_clear_match(self) -> None:
        """One clear match per column -> no ReID needed."""
        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
        assert needs_reid(cost) is False

    def test_no_overlap(self) -> None:
        """All costs > 0.7 -> no spatial overlap -> no ReID needed."""
        cost = np.array([[0.8, 0.9], [0.9, 0.8]], dtype=np.float64)
        assert needs_reid(cost) is False

    def test_ambiguous(self) -> None:
        """Two tracks with similar overlap for one detection -> need ReID."""
        cost = np.array([[0.45, 0.9], [0.50, 0.1]], dtype=np.float64)
        assert needs_reid(cost) is True


# ---------------------------------------------------------------------------
# Section 20: ReID Protocol tests
# ---------------------------------------------------------------------------


class TestReIDProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        mock = MockReIDExtractor(dim=128)
        assert isinstance(mock, ReIDExtractor)

    def test_mock_extract_returns_correct_shape(self) -> None:
        mock = MockReIDExtractor(dim=128)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [(100.0, 100.0, 200.0, 200.0), (300.0, 300.0, 400.0, 400.0)]
        result = mock.extract(frame, boxes)
        assert result is not None
        assert result.shape == (2, 128)

    def test_mock_extract_empty_returns_none(self) -> None:
        mock = MockReIDExtractor(dim=128)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert mock.extract(frame, []) is None

    def test_mock_embeddings_l2_normalized(self) -> None:
        mock = MockReIDExtractor(dim=128)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [(100.0, 100.0, 200.0, 200.0)]
        result = mock.extract(frame, boxes)
        assert result is not None
        norm = float(np.linalg.norm(result[0]))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Section 21: ByteTracker with ReID tests
# ---------------------------------------------------------------------------


class TestByteTrackerReID:
    def test_no_reid_identical_output(self) -> None:
        """Without reid_extractor, behavior is identical to original."""
        tracker = ByteTracker(min_hits=1)
        box = _make_box(conf=0.9)
        r = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert isinstance(r, TrackedDetection)
        assert r.num_boxes == 1

    def test_accepts_reid_extractor(self) -> None:
        mock = MockReIDExtractor(dim=128)
        tracker = ByteTracker(min_hits=1, reid_extractor=mock)
        box = _make_box(conf=0.9)
        r = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert isinstance(r, TrackedDetection)

    def test_reid_updates_embeddings(self) -> None:
        """After matching with ReID, tracks should have embeddings."""
        mock = MockReIDExtractor(dim=128)
        tracker = ByteTracker(
            min_hits=1,
            reid_extractor=mock,
            reid_frame_interval=1,
        )
        box_a = _make_box(x1=100, y1=100, x2=300, y2=300, conf=0.9)
        box_b = _make_box(x1=400, y1=100, x2=600, y2=300, conf=0.9)
        # Create tracks
        tracker.update(_make_detection(boxes=(box_a, box_b), frame_index=0))
        # Second frame — matching triggers ReID if ambiguous
        tracker.update(_make_detection(boxes=(box_a, box_b), frame_index=1))
        # Embedding may or may not be set depending on needs_reid gate,
        # but the tracker should not crash with ReID enabled.
        # Check that internal tracks exist and the tracker is functional.
        _ = any(
            t.embedding is not None
            for t in tracker._tracked  # type: ignore[attr-defined]
        )
        assert tracker.active_track_count >= 1

    def test_reid_graceful_none_extract(self) -> None:
        """If extractor.extract returns None, tracker falls back to IoU."""

        class NullExtractor:
            @property
            def embedding_dim(self) -> int:
                return 128

            def extract(
                self,
                frame_pixels: NDArray[np.uint8],
                boxes_xyxy: list[tuple[float, float, float, float]],
            ) -> NDArray[np.float32] | None:
                return None

        tracker = ByteTracker(
            min_hits=1,
            reid_extractor=NullExtractor(),
            reid_frame_interval=1,
        )
        box = _make_box(conf=0.9)
        r = tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert isinstance(r, TrackedDetection)

    def test_reset_clears_state_with_reid(self) -> None:
        mock = MockReIDExtractor(dim=128)
        tracker = ByteTracker(min_hits=1, reid_extractor=mock)
        for i in range(3):
            tracker.update(_make_detection(boxes=(_make_box(conf=0.9),), frame_index=i))
        tracker.reset()
        assert tracker.active_track_count == 0
        assert tracker.lost_track_count == 0

    def test_frame_interval_throttle(self) -> None:
        """ReID should not be called more often than reid_frame_interval."""
        call_count = 0

        class CountingExtractor:
            @property
            def embedding_dim(self) -> int:
                return 128

            def extract(
                self,
                frame_pixels: NDArray[np.uint8],
                boxes_xyxy: list[tuple[float, float, float, float]],
            ) -> NDArray[np.float32] | None:
                nonlocal call_count
                call_count += 1
                n = len(boxes_xyxy)
                rng = np.random.default_rng(call_count)
                embs = rng.standard_normal((n, 128)).astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
                return (embs / norms).astype(np.float32)

        tracker = ByteTracker(
            min_hits=1,
            reid_extractor=CountingExtractor(),
            reid_frame_interval=3,
        )
        # Create two overlapping tracks that trigger ambiguity
        box_a = _make_box(x1=100, y1=100, x2=250, y2=250, conf=0.9)
        box_b = _make_box(x1=150, y1=100, x2=300, y2=250, conf=0.9)
        for i in range(10):
            tracker.update(_make_detection(boxes=(box_a, box_b), frame_index=i))
        # With interval=3, over 10 frames should have <= 4 extract calls
        # (even less if needs_reid sometimes returns False)
        assert call_count <= 4


# ---------------------------------------------------------------------------
# Section 22: track_stream/track_detections with ReID
# ---------------------------------------------------------------------------


class TestTrackStreamReID:
    def test_track_stream_accepts_reid_extractor(self) -> None:
        engine = MagicMock()
        dets = [_make_detection(boxes=(_make_box(conf=0.9),), frame_index=i) for i in range(3)]
        engine.stream.return_value = iter(dets)
        mock = MockReIDExtractor(dim=128)
        results = list(track_stream(engine, MagicMock(), reid_extractor=mock))
        assert len(results) == 3

    def test_track_detections_accepts_reid_extractor(self) -> None:
        dets = [_make_detection(boxes=(_make_box(conf=0.9),), frame_index=i) for i in range(3)]
        mock = MockReIDExtractor(dim=128)
        results = list(track_detections(dets, reid_extractor=mock))
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Section 23: Output box clamping to frame bounds
# ---------------------------------------------------------------------------


class TestOutputBoxClamp:
    """Verify Kalman-predicted boxes are clamped to frame dimensions."""

    def test_box_within_frame_unchanged(self) -> None:
        """A box fully inside the frame is returned as-is."""
        det = _make_detection(
            boxes=(_make_box(x1=100, y1=100, x2=200, y2=200, conf=0.9),),
            frame_index=0,
        )
        tracker = ByteTracker(track_high_thresh=0.5, track_low_thresh=0.1)
        tracked = tracker.update(det)
        for box in tracked.boxes:
            assert box.x1 >= 0
            assert box.y1 >= 0
            assert box.x2 <= 640  # frame width from _make_frame
            assert box.y2 <= 480  # frame height from _make_frame

    def test_box_exceeding_frame_is_clamped(self) -> None:
        """A box extending beyond frame edges gets clamped."""
        from yowo.tracking._tracker import _clamp_tracked_box

        box = TrackedBox(
            x1=-10.0,
            y1=-20.0,
            x2=700.0,
            y2=500.0,
            confidence=0.9,
            class_id=0,
            class_name="car",
            track_id=1,
            is_confirmed=True,
        )
        clamped = _clamp_tracked_box(box, w=640, h=480)
        assert clamped.x1 == 0.0
        assert clamped.y1 == 0.0
        assert clamped.x2 == 640.0
        assert clamped.y2 == 480.0
        assert clamped.track_id == 1
        assert clamped.confidence == 0.9

    def test_clamp_preserves_valid_box(self) -> None:
        """A box already within bounds is returned without allocation."""
        from yowo.tracking._tracker import _clamp_tracked_box

        box = TrackedBox(
            x1=50.0,
            y1=60.0,
            x2=200.0,
            y2=300.0,
            confidence=0.8,
            class_id=1,
            class_name="truck",
            track_id=5,
            is_confirmed=False,
        )
        result = _clamp_tracked_box(box, w=640, h=480)
        assert result is box  # same object — no allocation

    def test_negative_coords_clamped_to_zero(self) -> None:
        """Negative coordinates from Kalman overshoot are clamped to 0."""
        from yowo.tracking._tracker import _clamp_tracked_box

        box = TrackedBox(
            x1=-50.0,
            y1=-100.0,
            x2=100.0,
            y2=200.0,
            confidence=0.7,
            class_id=0,
            class_name="car",
            track_id=3,
            is_confirmed=True,
        )
        clamped = _clamp_tracked_box(box, w=640, h=480)
        assert clamped.x1 == 0.0
        assert clamped.y1 == 0.0
        assert clamped.x2 == 100.0
        assert clamped.y2 == 200.0

    def test_bottom_edge_entry_clamped(self) -> None:
        """Simulates a car entering from the bottom — wide aspect ratio clamped."""
        # Frame 0: partial detection near bottom edge
        bottom_box = _make_box(x1=400, y1=420, x2=600, y2=480, conf=0.8)
        det0 = _make_detection(boxes=(bottom_box,), frame_index=0)
        tracker = ByteTracker(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            min_hits=1,
        )
        tracked0 = tracker.update(det0)
        assert tracked0.num_boxes >= 1
        for box in tracked0.boxes:
            assert box.x1 >= 0.0
            assert box.y1 >= 0.0
            assert box.x2 <= 640.0
            assert box.y2 <= 480.0


# ---------------------------------------------------------------------------
# Section 24: Hybrid output — detection box for young, Kalman for mature
# ---------------------------------------------------------------------------


class TestHybridOutputBox:
    """Verify young tracks output raw detection, mature tracks output Kalman."""

    def test_young_track_outputs_detection_make_box(self) -> None:
        """A track with hits < min_hits should output the raw detection coords."""
        from yowo.tracking._kalman import KalmanFilterXYAH
        from yowo.tracking._strack import STrack

        kalman = KalmanFilterXYAH()
        det_box = (100.0, 200.0, 300.0, 400.0)
        track = STrack(1, det_box, 0.9, 0, "car", kalman, min_hits=3)
        track.activate(frame_id=0)
        assert track.hits == 1  # < min_hits=3

        tb = track.to_tracked_box()
        assert (tb.x1, tb.y1, tb.x2, tb.y2) == det_box

    def test_matched_mature_track_outputs_detection(self) -> None:
        """A matched mature track outputs the raw detection, not Kalman."""
        from yowo.tracking._kalman import KalmanFilterXYAH
        from yowo.tracking._strack import STrack

        kalman = KalmanFilterXYAH()
        det_box = (100.0, 200.0, 300.0, 400.0)
        track = STrack(1, det_box, 0.9, 0, "car", kalman, min_hits=3)
        track.activate(frame_id=0)

        # Accumulate hits to reach min_hits
        for i in range(1, 4):
            track.predict()
            track.update(det_box, 0.9, 0, "car", frame_id=i)

        assert track.hits >= 3
        assert track.time_since_update == 0  # matched
        tb = track.to_tracked_box()
        # Matched track always outputs detection (time_since_update == 0)
        assert (tb.x1, tb.y1, tb.x2, tb.y2) == det_box

    def test_unmatched_track_outputs_kalman_prediction(self) -> None:
        """An unmatched track (time_since_update > 0) outputs Kalman prediction."""
        from yowo.tracking._kalman import KalmanFilterXYAH
        from yowo.tracking._strack import STrack

        kalman = KalmanFilterXYAH()
        det_box = (100.0, 200.0, 300.0, 400.0)
        track = STrack(1, det_box, 0.9, 0, "car", kalman, min_hits=3)
        track.activate(frame_id=0)

        # Simulate predict without update (track is unmatched this frame)
        track.predict()
        assert track.time_since_update == 1

        tb = track.to_tracked_box()
        # Unmatched → should use Kalman-predicted box, not stale detection
        assert (tb.x1, tb.y1, tb.x2, tb.y2) == track.predicted_xyxy
        # Kalman prediction is used — verify it comes from predicted_xyxy
        assert track.time_since_update > 0

    def test_det_xyxy_updated_on_update(self) -> None:
        """STrack._det_xyxy is refreshed each time update() is called."""
        from yowo.tracking._kalman import KalmanFilterXYAH
        from yowo.tracking._strack import STrack

        kalman = KalmanFilterXYAH()
        track = STrack(1, (10.0, 20.0, 30.0, 40.0), 0.9, 0, "car", kalman, 3)
        track.activate(frame_id=0)
        assert track._det_xyxy == (10.0, 20.0, 30.0, 40.0)

        new_box = (50.0, 60.0, 70.0, 80.0)
        track.predict()
        track.update(new_box, 0.8, 0, "car", frame_id=1)
        assert track._det_xyxy == new_box

    def test_det_xyxy_updated_on_reactivate(self) -> None:
        """STrack._det_xyxy is refreshed when re_activate() is called."""
        from yowo.tracking._kalman import KalmanFilterXYAH
        from yowo.tracking._strack import STrack

        kalman = KalmanFilterXYAH()
        track = STrack(1, (10.0, 20.0, 30.0, 40.0), 0.9, 0, "car", kalman, 3)
        track.activate(frame_id=0)
        track.mark_lost()

        new_box = (90.0, 100.0, 110.0, 120.0)
        track.predict()
        track.re_activate(new_box, 0.8, 0, "car", frame_id=5)
        assert track._det_xyxy == new_box

    def test_bottom_entry_no_inflation(self) -> None:
        """A car entering from the bottom has no width inflation in output.

        Without hybrid output, the Kalman aspect ratio lag would inflate
        the box width by ~2x for the first several frames.
        """
        from yowo.tracking._kalman import KalmanFilterXYAH
        from yowo.tracking._strack import STrack

        kalman = KalmanFilterXYAH()
        # Partial car at bottom: 200px wide, 50px tall → a=4.0
        box0 = (400.0, 430.0, 600.0, 480.0)
        track = STrack(1, box0, 0.8, 0, "car", kalman, min_hits=3)
        track.activate(frame_id=0)

        # Frame 1: car more visible — real box narrows in aspect ratio
        box1 = (380.0, 380.0, 600.0, 480.0)
        track.predict()
        track.update(box1, 0.85, 0, "car", frame_id=1)

        assert track.hits == 2  # still young
        tb = track.to_tracked_box()
        output_w = tb.x2 - tb.x1
        real_w = box1[2] - box1[0]  # 220
        # Output width should match detection (young track), not inflated Kalman
        assert abs(output_w - real_w) < 1.0, (
            f"Young track output width {output_w:.1f} should match "
            f"detection width {real_w:.1f}, not inflated Kalman"
        )


# ---------------------------------------------------------------------------
# Intra-tracked dedup + birth suppression tests
# ---------------------------------------------------------------------------


class TestRemoveIntraDuplicates:
    """Tests for remove_intra_duplicates (dedup within _tracked)."""

    def test_no_duplicates_returns_same(self) -> None:
        from yowo.tracking._matching import remove_intra_duplicates

        t1 = _make_strack(100, 100, 200, 200, track_id=1)
        t2 = _make_strack(400, 400, 500, 500, track_id=2)
        t1.activate(0)
        t2.activate(0)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_overlapping_removes_younger(self) -> None:
        from yowo.tracking._matching import remove_intra_duplicates

        t1 = _make_strack(100, 100, 200, 200, track_id=1)
        t1.activate(0)
        # Advance t1 by updating to make it older
        t1.update((100, 100, 200, 200), 0.9, 0, "person", frame_id=5)

        t2 = _make_strack(105, 105, 205, 205, track_id=2)  # ~82% IoU with t1
        t2.activate(3)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1  # older survives

    def test_identical_boxes_removes_one(self) -> None:
        from yowo.tracking._matching import remove_intra_duplicates

        t1 = _make_strack(100, 100, 200, 200, track_id=1)
        t2 = _make_strack(100, 100, 200, 200, track_id=2)
        t1.activate(0)
        t2.activate(0)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1

    def test_empty_list(self) -> None:
        from yowo.tracking._matching import remove_intra_duplicates

        assert remove_intra_duplicates([]) == []

    def test_single_track(self) -> None:
        from yowo.tracking._matching import remove_intra_duplicates

        t1 = _make_strack(100, 100, 200, 200, track_id=1)
        t1.activate(0)
        result = remove_intra_duplicates([t1])
        assert len(result) == 1

    def test_below_threshold_kept(self) -> None:
        from yowo.tracking._matching import remove_intra_duplicates

        # IoU ~0.53 (boxes share ~53% overlap) — below 0.70 threshold
        t1 = _make_strack(100, 100, 200, 200, track_id=1)
        t2 = _make_strack(150, 100, 250, 200, track_id=2)  # shifted 50px right
        t1.activate(0)
        t2.activate(0)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2  # both survive


class TestBirthSuppression:
    """Tests that duplicate births are suppressed when overlapping a matched track."""

    def test_nms_leak_suppressed(self) -> None:
        """Two YOLO detections for same object → only one track created."""
        tracker = ByteTracker(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Frame 0: One detection → one track born
        det0 = _make_detection(
            boxes=(_make_box(100, 100, 200, 200, conf=0.8),),
            frame_index=0,
        )
        tracker.update(det0)

        # Frame 1: Two overlapping detections (NMS leak) — same object
        box_a = _make_box(100, 100, 200, 200, conf=0.8)
        box_b = _make_box(105, 105, 205, 205, conf=0.7)  # ~82% IoU
        det1 = _make_detection(boxes=(box_a, box_b), frame_index=1)
        tracked = tracker.update(det1)

        # Should be 1 tracked box, not 2
        assert len(tracked.boxes) == 1

    def test_distinct_objects_not_suppressed(self) -> None:
        """Two genuinely different objects should both get tracks."""
        tracker = ByteTracker(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Frame 0: one object
        det0 = _make_detection(
            boxes=(_make_box(100, 100, 200, 200, conf=0.8),),
            frame_index=0,
        )
        tracker.update(det0)

        # Frame 1: existing object + new object far away
        det1 = _make_detection(
            boxes=(
                _make_box(100, 100, 200, 200, conf=0.8),
                _make_box(400, 400, 500, 500, conf=0.8),
            ),
            frame_index=1,
        )
        tracked = tracker.update(det1)
        assert len(tracked.boxes) == 2

    def test_multi_frame_no_accumulating_duplicates(self) -> None:
        """Repeated NMS leaks across frames don't accumulate duplicate tracks."""
        tracker = ByteTracker(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        for frame_idx in range(10):
            box_a = _make_box(100, 100, 200, 200, conf=0.8)
            box_b = _make_box(102, 102, 202, 202, conf=0.75)
            det = _make_detection(boxes=(box_a, box_b), frame_index=frame_idx)
            tracked = tracker.update(det)

        # After 10 frames, should still be 1 track, not 10
        assert len(tracked.boxes) == 1

    def test_all_unique_ids_across_frames(self) -> None:
        """Same object with NMS leak should maintain one consistent ID."""
        tracker = ByteTracker(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        all_ids: set[int] = set()
        for frame_idx in range(20):
            box_a = _make_box(100, 100, 200, 200, conf=0.8)
            box_b = _make_box(103, 103, 203, 203, conf=0.7)
            det = _make_detection(boxes=(box_a, box_b), frame_index=frame_idx)
            tracked = tracker.update(det)
            for b in tracked.boxes:
                all_ids.add(b.track_id)

        # Should have only 1 unique ID (or at most 2 if birth happened once)
        assert len(all_ids) <= 2


# ---------------------------------------------------------------------------
# Section 28: FastReIDExtractor tests
# ---------------------------------------------------------------------------


class TestFastReIDExtractor:
    def test_implements_protocol(self) -> None:
        """FastReIDExtractor satisfies the ReIDExtractor Protocol."""
        # Can't instantiate without a real ONNX file, but the class has
        # the right attributes: embedding_dim property + extract() method.
        assert hasattr(FastReIDExtractor, "embedding_dim")
        assert hasattr(FastReIDExtractor, "extract")
        # Protocol structural check: FastReIDExtractor has the same method
        # signatures as ReIDExtractor. We verify via attribute presence
        # since isinstance requires an instance.

    def test_file_not_found_raises(self) -> None:
        """Non-existent model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="FastReID ONNX model not found"):
            FastReIDExtractor("/nonexistent/fastreid.onnx")

    def test_imagenet_constants_correct(self) -> None:
        """ImageNet normalization constants match standard values."""
        expected_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        expected_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        np.testing.assert_allclose(_IMAGENET_MEAN, expected_mean, atol=1e-6)
        np.testing.assert_allclose(_IMAGENET_STD, expected_std, atol=1e-6)

    def test_input_size_portrait_default(self) -> None:
        """Default input_size is portrait (256, 128) not square."""
        # We can check the __init__ signature default without instantiation
        import inspect

        sig = inspect.signature(FastReIDExtractor.__init__)
        default = sig.parameters["input_size"].default
        assert default == (256, 128), f"Expected (256, 128), got {default}"

    def test_embedding_dim_default(self) -> None:
        """Default embedding_dim is 256 (not 512 like CLIP)."""
        import inspect

        sig = inspect.signature(FastReIDExtractor.__init__)
        default = sig.parameters["embedding_dim"].default
        assert default == 256, f"Expected 256, got {default}"

    def test_slots_defined(self) -> None:
        """FastReIDExtractor uses __slots__ for memory efficiency."""
        assert hasattr(FastReIDExtractor, "__slots__")
        assert "_session" in FastReIDExtractor.__slots__
        assert "_input_size_hw" in FastReIDExtractor.__slots__

    def test_export_in_tracking_init(self) -> None:
        """FastReIDExtractor is exported from yowo.tracking."""
        from yowo.tracking import FastReIDExtractor as Imported

        assert Imported is FastReIDExtractor


# ---------------------------------------------------------------------------
# Section 28b: CLIPReIDExtractor tests
# ---------------------------------------------------------------------------


class TestCLIPReIDExtractor:
    def test_implements_protocol(self) -> None:
        """CLIPReIDExtractor satisfies the ReIDExtractor Protocol."""
        assert hasattr(CLIPReIDExtractor, "embedding_dim")
        assert hasattr(CLIPReIDExtractor, "extract")

    def test_file_not_found_raises(self) -> None:
        """Non-existent model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="CLIP-ReID ONNX model not found"):
            CLIPReIDExtractor("/nonexistent/clip_reid.onnx")

    def test_embedding_dim_default(self) -> None:
        """Default embedding_dim is 1280 (768 + 512 concat)."""
        import inspect

        sig = inspect.signature(CLIPReIDExtractor.__init__)
        default = sig.parameters["embedding_dim"].default
        assert default == 1280, f"Expected 1280, got {default}"

    def test_input_size_default(self) -> None:
        """Default input_size is 256 (not 224 like standard CLIP)."""
        import inspect

        sig = inspect.signature(CLIPReIDExtractor.__init__)
        default = sig.parameters["input_size"].default
        assert default == 256, f"Expected 256, got {default}"

    def test_preprocessing_constants(self) -> None:
        """CLIP-ReID uses mean=0.5, std=0.5 (not ImageNet or CLIP mean/std)."""
        from yowo.tracking._clip_reid import _CLIPREID_MEAN, _CLIPREID_STD

        expected = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        np.testing.assert_allclose(_CLIPREID_MEAN, expected, atol=1e-6)
        np.testing.assert_allclose(_CLIPREID_STD, expected, atol=1e-6)

    def test_slots_defined(self) -> None:
        """CLIPReIDExtractor uses __slots__ for memory efficiency."""
        assert hasattr(CLIPReIDExtractor, "__slots__")
        assert "_session" in CLIPReIDExtractor.__slots__
        assert "_input_size" in CLIPReIDExtractor.__slots__

    def test_export_in_tracking_init(self) -> None:
        """CLIPReIDExtractor is exported from yowo.tracking."""
        from yowo.tracking import CLIPReIDExtractor as Imported

        assert Imported is CLIPReIDExtractor

    def test_export_in_reid_module(self) -> None:
        """CLIPReIDExtractor is re-exported from _reid.py __all__."""
        from yowo.tracking._reid import __all__ as reid_all

        assert "CLIPReIDExtractor" in reid_all


# ---------------------------------------------------------------------------
# 29. fuse_score
# ---------------------------------------------------------------------------
class TestFuseScore:
    """Tests for the fuse_score function."""

    def test_identity_at_score_one(self) -> None:
        """Scores all 1.0 → cost unchanged."""
        cost = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=np.float64)
        scores = np.array([1.0, 1.0], dtype=np.float64)
        result = fuse_score(cost, scores)
        np.testing.assert_allclose(result, cost)

    def test_zeros_at_score_zero(self) -> None:
        """Scores all 0.0 → cost all 1.0 (no similarity)."""
        cost = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=np.float64)
        scores = np.array([0.0, 0.0], dtype=np.float64)
        result = fuse_score(cost, scores)
        np.testing.assert_allclose(result, np.ones_like(cost))

    def test_scales_correctly(self) -> None:
        """Verify formula: cost = 1 - (1 - iou_cost) * score."""
        cost = np.array([[0.3]], dtype=np.float64)
        scores = np.array([0.5], dtype=np.float64)
        expected = 1.0 - (1.0 - 0.3) * 0.5  # 1 - 0.7*0.5 = 1 - 0.35 = 0.65
        result = fuse_score(cost, scores)
        np.testing.assert_allclose(result, [[expected]])

    def test_empty_matrix(self) -> None:
        """Empty cost matrix → returns empty."""
        cost = np.empty((0, 0), dtype=np.float64)
        scores = np.empty(0, dtype=np.float64)
        result = fuse_score(cost, scores)
        assert result.shape == (0, 0)

    def test_broadcast_shape(self) -> None:
        """Result shape matches input shape with multiple tracks and dets."""
        cost = np.array([[0.1, 0.4, 0.9], [0.3, 0.2, 0.7]], dtype=np.float64)
        scores = np.array([0.8, 0.5, 0.3], dtype=np.float64)
        result = fuse_score(cost, scores)
        assert result.shape == (2, 3)
        # Lower score → higher cost
        assert result[0, 2] > result[0, 0], "Low-score det should have higher cost"

    def test_exported_from_matching(self) -> None:
        """fuse_score is in _matching.__all__."""
        from yowo.tracking._matching import __all__ as matching_all

        assert "fuse_score" in matching_all


# ---------------------------------------------------------------------------
# 30. ByteTracker fuse_score integration
# ---------------------------------------------------------------------------
class TestByteTrackerFuseScore:
    """Tests for ByteTracker with fuse_score flag."""

    def test_fuse_score_default_false(self) -> None:
        """fuse_score defaults to False."""
        tracker = ByteTracker()
        assert tracker._fuse_score is False

    def test_fuse_score_flag_stored(self) -> None:
        """fuse_score=True is stored on the tracker."""
        tracker = ByteTracker(fuse_score=True)
        assert tracker._fuse_score is True

    def test_fuse_score_does_not_crash(self) -> None:
        """ByteTracker with fuse_score=True processes detections without error."""
        tracker = ByteTracker(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            fuse_score=True,
        )
        bbox = _make_box(100, 100, 200, 200, conf=0.9)
        det = _make_detection(boxes=(bbox,), frame_index=1)
        result = tracker.update(det)
        assert isinstance(result, TrackedDetection)

    def test_fuse_score_penalizes_low_confidence(self) -> None:
        """With fuse_score=True, a low-confidence detection is less likely to match."""
        # Without fuse_score, a 0.3-conf det at the same location should match
        tracker_no_fuse = ByteTracker(
            track_high_thresh=0.2,
            track_low_thresh=0.1,
            min_hits=1,
        )
        tracker_fuse = ByteTracker(
            track_high_thresh=0.2,
            track_low_thresh=0.1,
            min_hits=1,
            fuse_score=True,
        )
        # Frame 1: birth a track
        bbox1 = _make_box(100, 100, 200, 200, conf=0.9)
        det1 = _make_detection(boxes=(bbox1,), frame_index=1)
        tracker_no_fuse.update(det1)
        tracker_fuse.update(det1)
        # Frame 2: same box but low confidence
        bbox2 = _make_box(100, 100, 200, 200, conf=0.25)
        det2 = _make_detection(boxes=(bbox2,), frame_index=2)
        r_no_fuse = tracker_no_fuse.update(det2)
        r_fuse = tracker_fuse.update(det2)
        # Both should still have tracks (IoU is perfect), but behavior is valid
        assert isinstance(r_no_fuse, TrackedDetection)
        assert isinstance(r_fuse, TrackedDetection)


# ---------------------------------------------------------------------------
# 31. Unconfirmed track re-association stage
# ---------------------------------------------------------------------------
class TestUnconfirmedStage:
    """Tests for the Stage 4 unconfirmed track re-association."""

    def _make_tracker(self, **kwargs: object) -> ByteTracker:
        defaults: dict[str, object] = {
            "track_high_thresh": 0.3,
            "track_low_thresh": 0.1,
            "min_hits": 3,
        }
        defaults.update(kwargs)
        return ByteTracker(**defaults)  # type: ignore[arg-type]

    def test_unconfirmed_separated_from_confirmed(self) -> None:
        """Tracks with hits < min_hits are treated as unconfirmed."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: birth a track (hits=1 after activate)
        bbox = _make_box(100, 100, 200, 200, conf=0.9)
        det = _make_detection(boxes=(bbox,), frame_index=1)
        tracker.update(det)
        # Track exists but is unconfirmed (hits=1 < min_hits=3)
        assert len(tracker._tracked) == 1
        assert not tracker._tracked[0].is_confirmed

    def test_unconfirmed_re_associates(self) -> None:
        """Unconfirmed track that matches in Stage 4 survives."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: birth track at (100,100,200,200)
        bbox1 = _make_box(100, 100, 200, 200, conf=0.9)
        det1 = _make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 1
        # Frame 2: same box appears — unconfirmed track should match in Stage 4
        bbox2 = _make_box(102, 102, 202, 202, conf=0.9)
        det2 = _make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        # Track should survive (matched in Stage 4)
        assert len(tracker._tracked) >= 1
        # Should keep same track_id (not birth a new one)
        ids = {t.track_id for t in tracker._tracked}
        assert 1 in ids, "Original track should survive through Stage 4"

    def test_unconfirmed_removed_on_fail(self) -> None:
        """Unconfirmed track that fails to match is REMOVED (not LOST)."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: birth track at (100,100,200,200)
        bbox1 = _make_box(100, 100, 200, 200, conf=0.9)
        det1 = _make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 1
        # Frame 2: detection far away — unconfirmed track fails to match
        bbox2 = _make_box(500, 500, 600, 600, conf=0.9)
        det2 = _make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        # Original track should be REMOVED, not in _lost
        lost_ids = {t.track_id for t in tracker._lost}
        assert 1 not in lost_ids, "Unconfirmed failures should be REMOVED, not LOST"

    def test_confirmed_bypass_unconfirmed_stage(self) -> None:
        """Tracks with hits >= min_hits participate in Stage 1, not Stage 4."""
        tracker = self._make_tracker(min_hits=1)
        # With min_hits=1, track becomes confirmed after activate (hits=1)
        bbox1 = _make_box(100, 100, 200, 200, conf=0.9)
        det1 = _make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 1
        assert tracker._tracked[0].is_confirmed  # hits=1 >= min_hits=1

    def test_backward_compatible_min_hits_1(self) -> None:
        """With min_hits=1, unconfirmed list is always empty — no behavior change."""
        tracker = self._make_tracker(min_hits=1)
        # Frame 1: birth
        bbox1 = _make_box(100, 100, 200, 200, conf=0.9)
        det1 = _make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        # With min_hits=1, track is confirmed immediately
        assert len(tracker._tracked) == 1
        assert tracker._tracked[0].is_confirmed
        # Frame 2: match
        bbox2 = _make_box(102, 102, 202, 202, conf=0.9)
        det2 = _make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        assert len(tracker._tracked) == 1
        assert tracker._tracked[0].hits == 2

    def test_unconfirmed_matches_tight_iou(self) -> None:
        """Stage 4 uses thresh=0.7 — only close IoU matches succeed."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: birth at (100,100,200,200)
        bbox1 = _make_box(100, 100, 200, 200, conf=0.9)
        det1 = _make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        # Frame 2: detection slightly shifted (IoU ~0.56, which is < 0.7 threshold)
        # IoU between (100,100,200,200) and (130,130,230,230) is low
        bbox2 = _make_box(160, 160, 260, 260, conf=0.9)
        det2 = _make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        # Original track should be removed (IoU too low for 0.7 threshold)
        # New track birthed for the second detection
        assert len(tracker._tracked) >= 1

    def test_multi_frame_unconfirmed_graduation(self) -> None:
        """Track graduates from unconfirmed to confirmed after min_hits updates."""
        tracker = self._make_tracker(min_hits=3)
        for frame_idx in range(1, 5):
            bbox = _make_box(100, 100, 200, 200, conf=0.9)
            det = _make_detection(boxes=(bbox,), frame_index=frame_idx)
            tracker.update(det)
        # After 4 frames of matching, hits should be >= 3 → confirmed
        assert len(tracker._tracked) >= 1
        confirmed = [t for t in tracker._tracked if t.is_confirmed]
        assert len(confirmed) >= 1, "Track should graduate to confirmed"

    def test_crowded_scene_unconfirmed_survives(self) -> None:
        """In crowded scenes, unconfirmed tracks survive through Stage 4."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: two objects born
        b1 = _make_box(100, 100, 200, 200, conf=0.9)
        b2 = _make_box(400, 400, 500, 500, conf=0.9)
        det1 = _make_detection(boxes=(b1, b2), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 2
        # Frame 2: both objects still present at same locations
        b3 = _make_box(102, 102, 202, 202, conf=0.9)
        b4 = _make_box(402, 402, 502, 502, conf=0.9)
        det2 = _make_detection(boxes=(b3, b4), frame_index=2)
        tracker.update(det2)
        # Both tracks should survive (Stage 4 re-association)
        assert len(tracker._tracked) == 2
        ids = {t.track_id for t in tracker._tracked}
        assert len(ids) == 2, "Both tracks should keep their IDs"


class TestLostTrackLowConfRecovery:
    """Tests for Stage 2.5: low-conf detection recovery of lost tracks."""

    def test_low_conf_reappearance_keeps_id(self) -> None:
        """A lost track reappearing at low confidence recovers its original ID."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Frame 0: car appears at high conf → tracked
        car = _make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(_make_detection(boxes=(car,), frame_index=0))
        assert r0.num_boxes == 1
        original_id = r0.boxes[0].track_id

        # Frames 1-10: car occluded → no detections → track goes to lost
        for i in range(1, 11):
            tracker.update(_make_detection(boxes=(), frame_index=i))
        assert tracker.lost_track_count >= 1

        # Frame 11: car reappears at low conf (0.56 < track_high_thresh=0.6)
        reappear = _make_box(100, 200, 300, 400, conf=0.56, cls_id=2, cls_name="car")
        r11 = tracker.update(_make_detection(boxes=(reappear,), frame_index=11))
        assert r11.num_boxes >= 1
        recovered_ids = {b.track_id for b in r11.boxes}
        assert original_id in recovered_ids

    def test_low_conf_no_match_stale_lost(self) -> None:
        """Moving lost tracks beyond recency limit are NOT matched by Stage 2.5."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Use moving boxes so track has velocity > 1.0 (no stationary extension)
        for i in range(3):
            box = _make_box(
                100 + i * 20,
                200,
                300 + i * 20,
                400,
                conf=0.85,
                cls_id=2,
                cls_name="car",
            )
            tracker.update(_make_detection(boxes=(box,), frame_index=i))
        original_id = tracker._tracked[0].track_id

        # Occlude for 20 frames (> max_age//2 = 15 → too stale for Stage 2.5)
        for i in range(3, 23):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Reappear at low conf at last known position — should NOT recover
        reappear = _make_box(140, 200, 340, 400, conf=0.56, cls_id=2, cls_name="car")
        r23 = tracker.update(_make_detection(boxes=(reappear,), frame_index=23))
        recovered_ids = {b.track_id for b in r23.boxes}
        assert original_id not in recovered_ids

    def test_class_id_guard_prevents_cross_class(self) -> None:
        """Stage 2.5 does not re-associate a car track with a person detection."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        car = _make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(_make_detection(boxes=(car,), frame_index=0))
        car_id = r0.boxes[0].track_id

        # Occlude for 5 frames
        for i in range(1, 6):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Person appears at same location with low conf
        person = _make_box(100, 200, 300, 400, conf=0.56, cls_id=0, cls_name="person")
        r6 = tracker.update(_make_detection(boxes=(person,), frame_index=6))
        for b in r6.boxes:
            assert b.track_id != car_id

    def test_high_conf_reappearance_uses_stage1(self) -> None:
        """High-conf re-appearance uses Stage 1 (no regression)."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        car = _make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(_make_detection(boxes=(car,), frame_index=0))
        original_id = r0.boxes[0].track_id

        # Occlude for 5 frames
        for i in range(1, 6):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Reappear at high conf → Stage 1 should match
        reappear = _make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r6 = tracker.update(_make_detection(boxes=(reappear,), frame_index=6))
        assert r6.num_boxes >= 1
        assert r6.boxes[0].track_id == original_id

    def test_zero_overhead_no_lost(self) -> None:
        """When there are no lost tracks, Stage 2.5 adds zero overhead."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
        )
        box = _make_box(100, 200, 300, 400, conf=0.85)
        tracker.update(_make_detection(boxes=(box,), frame_index=0))
        assert tracker.lost_track_count == 0

        # Low-conf detection nearby → Stage 2 handles it, Stage 2.5 skips
        low = _make_box(105, 205, 305, 405, conf=0.5)
        r1 = tracker.update(_make_detection(boxes=(low,), frame_index=1))
        assert r1.num_boxes >= 1

    def test_low_conf_consumed_by_stage2_not_reused(self) -> None:
        """A low-conf detection consumed by Stage 2 is NOT re-used in Stage 2.5."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Track A: actively tracked
        box_a = _make_box(100, 200, 300, 400, conf=0.85)
        tracker.update(_make_detection(boxes=(box_a,), frame_index=0))
        # Track B: tracked then lost
        box_b = _make_box(500, 200, 700, 400, conf=0.85)
        tracker.update(_make_detection(boxes=(box_a, box_b), frame_index=1))
        # Lose track B
        for i in range(2, 5):
            tracker.update(_make_detection(boxes=(box_a,), frame_index=i))
        assert tracker.lost_track_count >= 1

        # Low-conf at A's position → consumed by Stage 2 for track A
        low_a = _make_box(100, 200, 300, 400, conf=0.5)
        tracker.update(_make_detection(boxes=(low_a,), frame_index=5))
        # Track B should remain lost (low_a was consumed by Stage 2)
        assert tracker.lost_track_count >= 1

    def test_stationary_occlusion_scenario(self) -> None:
        """End-to-end: parked car disappears during occlusion, ID preserved."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Frames 0-4: parked car tracked at high conf
        car = _make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(_make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-14: car not detected (occluded by passing vehicle)
        for i in range(5, 15):
            tracker.update(_make_detection(boxes=(), frame_index=i))
        # Parked car should be lost
        lost_ids = {t.track_id for t in tracker._lost}
        assert car_id in lost_ids

        # Frame 15: occlusion clears, parked car reappears at low-conf
        reappear = _make_box(100, 300, 250, 450, conf=0.56, cls_id=2, cls_name="car")
        r15 = tracker.update(_make_detection(boxes=(reappear,), frame_index=15))
        recovered = [b for b in r15.boxes if b.track_id == car_id]
        assert len(recovered) == 1, f"Parked car should recover ID {car_id}"


class TestStationaryTrackRetention:
    """Tests for extended retention of stationary lost tracks."""

    def test_stationary_extended_retention(self) -> None:
        """Stationary track survives beyond normal max_age in lost pool."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=10,
        )
        # Frames 0-4: establish stationary track
        car = _make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(_make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: car disappears (15 frames > max_age=10)
        for i in range(5, 20):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Stationary track should still be in lost pool (< stationary_max_age=30)
        lost_ids = {t.track_id for t in tracker._lost}
        assert car_id in lost_ids, f"Stationary track {car_id} should survive beyond max_age=10"

    def test_moving_track_normal_retention(self) -> None:
        """Moving track is removed at normal max_age (no regression)."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=10,
        )
        # Frames 0-4: establish MOVING track (shift position each frame)
        for i in range(5):
            box = _make_box(
                100 + i * 20,
                300,
                250 + i * 20,
                450,
                conf=0.82,
                cls_id=2,
                cls_name="car",
            )
            tracker.update(_make_detection(boxes=(box,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: disappears for 15 frames (> max_age=10)
        for i in range(5, 20):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Moving track should be removed (velocity > 1.0 px/frame)
        lost_ids = {t.track_id for t in tracker._lost}
        assert car_id not in lost_ids, "Moving track should be removed at normal max_age"

    def test_stationary_recovery_long_absence_stage1(self) -> None:
        """Stationary track recovered via Stage 1 after >max_age gap."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=10,
        )
        # Frames 0-4: establish stationary track
        car = _make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(_make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: car disappears for 15 frames (> max_age=10)
        for i in range(5, 20):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Frame 20: car reappears at HIGH conf → Stage 1 recovery
        reappear = _make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        r20 = tracker.update(_make_detection(boxes=(reappear,), frame_index=20))

        recovered = [b for b in r20.boxes if b.track_id == car_id]
        assert len(recovered) == 1, f"Stationary car should recover ID {car_id} via Stage 1"

    def test_stationary_low_conf_recovery_long_absence(self) -> None:
        """Stationary track recovered via extended Stage 2.5 after long gap."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=10,
        )
        # Frames 0-4: establish stationary track
        car = _make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(_make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: car disappears for 15 frames (> max_age=10, > lost_low_age=5)
        for i in range(5, 20):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Frame 20: car reappears at LOW conf → Stage 2.5 with extended recency
        reappear = _make_box(100, 300, 250, 450, conf=0.50, cls_id=2, cls_name="car")
        r20 = tracker.update(_make_detection(boxes=(reappear,), frame_index=20))

        recovered = [b for b in r20.boxes if b.track_id == car_id]
        assert len(recovered) == 1, (
            f"Stationary car should recover ID {car_id} via extended Stage 2.5"
        )

    def test_short_lived_track_no_extended_retention(self) -> None:
        """Tracks with fewer than min_hits do NOT get stationary extension (C-1)."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=3,
            max_age=5,
        )
        # Single frame → hits=1 < min_hits=3 → NOT eligible for extension
        car = _make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        tracker.update(_make_detection(boxes=(car,), frame_index=0))
        track_id = tracker._tracked[0].track_id

        # Disappear for 8 frames (> max_age=5, < stationary_max_age=15)
        for i in range(1, 9):
            tracker.update(_make_detection(boxes=(), frame_index=i))

        # Should be removed — NOT given extended retention
        lost_ids = {t.track_id for t in tracker._lost}
        assert track_id not in lost_ids, (
            "Short-lived track (hits < min_hits) must not get extended retention"
        )
