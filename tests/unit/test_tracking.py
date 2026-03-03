"""Unit tests for yowo.tracking — ByteTracker, Kalman filter, matching, STrack."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from unit._tracking_helpers import make_box as _make_box
from unit._tracking_helpers import make_detection as _make_detection
from unit._tracking_helpers import make_strack as _make_strack
from yowo.errors import TrackingError, YowoError
from yowo.tracking import track_detections, track_stream
from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import (
    iou_batch,
    linear_assignment,
)
from yowo.tracking._strack import TrackedBox, TrackedDetection, TrackState
from yowo.tracking._tracker import ByteTracker


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


class TestTrackingError:
    def test_inherits_from_yowo_error(self) -> None:
        err = TrackingError("test error")
        assert isinstance(err, YowoError)
        assert str(err) == "test error"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(TrackingError, match="corrupt state"):
            raise TrackingError("corrupt state")


class TestIntegrationEdge:
    def test_track_detections_empty_iterable(self) -> None:
        results = list(track_detections([]))
        assert results == []


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
