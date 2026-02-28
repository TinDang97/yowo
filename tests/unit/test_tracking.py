"""Unit tests for yowo.tracking — ByteTracker, Kalman filter, matching, STrack."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from yowo.tracking import track_detections, track_stream
from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import (
    iou_batch,
    linear_assignment,
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
        box = _make_box(conf=0.9)
        tracker.update(_make_detection(boxes=(box,), frame_index=0))
        # max_age+2 frames with no detection
        for i in range(1, 5):
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
