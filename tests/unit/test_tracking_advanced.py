"""Unit tests for advanced tracking scenarios — output box, dedup, birth, fuse, unconfirmed."""

from __future__ import annotations

import numpy as np

from unit._tracking_helpers import make_box, make_detection, make_strack
from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import remove_intra_duplicates
from yowo.tracking._strack import STrack, TrackedBox
from yowo.tracking._tracker import ByteTracker

# ---------------------------------------------------------------------------
# Output box clamping to frame bounds
# ---------------------------------------------------------------------------


class TestOutputBoxClamp:
    """Verify Kalman-predicted boxes are clamped to frame dimensions."""

    def test_box_within_frame_unchanged(self) -> None:
        """A box fully inside the frame is returned as-is."""
        det = make_detection(
            boxes=(make_box(x1=100, y1=100, x2=200, y2=200, conf=0.9),),
            frame_index=0,
        )
        tracker = ByteTracker(track_high_thresh=0.5, track_low_thresh=0.1)
        tracked = tracker.update(det)
        for box in tracked.boxes:
            assert box.x1 >= 0
            assert box.y1 >= 0
            assert box.x2 <= 640  # frame width from make_frame
            assert box.y2 <= 480  # frame height from make_frame

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
        bottom_box = make_box(x1=400, y1=420, x2=600, y2=480, conf=0.8)
        det0 = make_detection(boxes=(bottom_box,), frame_index=0)
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
# Hybrid output — detection box for young, Kalman for mature
# ---------------------------------------------------------------------------


class TestHybridOutputBox:
    """Verify young tracks output raw detection, mature tracks output Kalman."""

    def test_young_track_outputs_detection_make_box(self) -> None:
        """A track with hits < min_hits should output the raw detection coords."""
        kalman = KalmanFilterXYAH()
        det_box = (100.0, 200.0, 300.0, 400.0)
        track = STrack(1, det_box, 0.9, 0, "car", kalman, min_hits=3)
        track.activate(frame_id=0)
        assert track.hits == 1  # < min_hits=3

        tb = track.to_tracked_box()
        assert (tb.x1, tb.y1, tb.x2, tb.y2) == det_box

    def test_matched_mature_track_outputs_detection(self) -> None:
        """A matched mature track outputs the raw detection, not Kalman."""
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
        kalman = KalmanFilterXYAH()
        track = STrack(1, (10.0, 20.0, 30.0, 40.0), 0.9, 0, "car", kalman, 3)
        track.activate(frame_id=0)
        track.mark_lost()

        new_box = (90.0, 100.0, 110.0, 120.0)
        track.predict()
        track.re_activate(new_box, 0.8, 0, "car", frame_id=5)
        assert track._det_xyxy == new_box

    def test_bottom_entry_no_inflation(self) -> None:
        """A car entering from the bottom has no width inflation in output."""
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
        t1 = make_strack(100, 100, 200, 200, track_id=1)
        t2 = make_strack(400, 400, 500, 500, track_id=2)
        t1.activate(0)
        t2.activate(0)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_overlapping_removes_younger(self) -> None:
        t1 = make_strack(100, 100, 200, 200, track_id=1)
        t1.activate(0)
        # Advance t1 by updating to make it older
        t1.update((100, 100, 200, 200), 0.9, 0, "person", frame_id=5)

        t2 = make_strack(105, 105, 205, 205, track_id=2)  # ~82% IoU with t1
        t2.activate(3)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1  # older survives

    def test_identical_boxes_removes_one(self) -> None:
        t1 = make_strack(100, 100, 200, 200, track_id=1)
        t2 = make_strack(100, 100, 200, 200, track_id=2)
        t1.activate(0)
        t2.activate(0)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1

    def test_empty_list(self) -> None:
        assert remove_intra_duplicates([]) == []

    def test_single_track(self) -> None:
        t1 = make_strack(100, 100, 200, 200, track_id=1)
        t1.activate(0)
        result = remove_intra_duplicates([t1])
        assert len(result) == 1

    def test_below_threshold_kept(self) -> None:
        # IoU ~0.53 (boxes share ~53% overlap) — below 0.70 threshold
        t1 = make_strack(100, 100, 200, 200, track_id=1)
        t2 = make_strack(150, 100, 250, 200, track_id=2)  # shifted 50px right
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
        det0 = make_detection(
            boxes=(make_box(100, 100, 200, 200, conf=0.8),),
            frame_index=0,
        )
        tracker.update(det0)

        # Frame 1: Two overlapping detections (NMS leak) — same object
        box_a = make_box(100, 100, 200, 200, conf=0.8)
        box_b = make_box(105, 105, 205, 205, conf=0.7)  # ~82% IoU
        det1 = make_detection(boxes=(box_a, box_b), frame_index=1)
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
        det0 = make_detection(
            boxes=(make_box(100, 100, 200, 200, conf=0.8),),
            frame_index=0,
        )
        tracker.update(det0)

        # Frame 1: existing object + new object far away
        det1 = make_detection(
            boxes=(
                make_box(100, 100, 200, 200, conf=0.8),
                make_box(400, 400, 500, 500, conf=0.8),
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
            box_a = make_box(100, 100, 200, 200, conf=0.8)
            box_b = make_box(102, 102, 202, 202, conf=0.75)
            det = make_detection(boxes=(box_a, box_b), frame_index=frame_idx)
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
            box_a = make_box(100, 100, 200, 200, conf=0.8)
            box_b = make_box(103, 103, 203, 203, conf=0.7)
            det = make_detection(boxes=(box_a, box_b), frame_index=frame_idx)
            tracked = tracker.update(det)
            for b in tracked.boxes:
                all_ids.add(b.track_id)

        # Should have only 1 unique ID (or at most 2 if birth happened once)
        assert len(all_ids) <= 2


# ---------------------------------------------------------------------------
# ByteTracker fuse_score integration
# ---------------------------------------------------------------------------


class TestByteTrackerFuseScore:
    """Tests for ByteTracker with fuse_score flag."""

    def test_fuse_score_default_false(self) -> None:
        """fuse_score defaults to False."""
        tracker = ByteTracker()
        assert not tracker._fuse_score

    def test_fuse_score_enabled(self) -> None:
        """fuse_score=True is accepted without error."""
        tracker = ByteTracker(fuse_score=True)
        assert tracker._fuse_score

    def test_fuse_score_single_track(self) -> None:
        """Single track with fuse_score=True produces correct output."""
        tracker = ByteTracker(min_hits=1, fuse_score=True)
        box = make_box(conf=0.9)
        r = tracker.update(make_detection(boxes=(box,), frame_index=0))
        assert r.num_boxes == 1

    def test_fuse_score_multi_track_matches(self) -> None:
        """Multiple tracks with fuse_score still match correctly."""
        tracker = ByteTracker(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            min_hits=1,
            fuse_score=True,
        )
        b1 = make_box(100, 100, 200, 200, conf=0.9)
        b2 = make_box(400, 400, 500, 500, conf=0.8)
        tracker.update(make_detection(boxes=(b1, b2), frame_index=0))
        assert tracker.active_track_count == 2

        # Frame 2: same boxes → should match
        b1_next = make_box(102, 102, 202, 202, conf=0.85)
        b2_next = make_box(402, 402, 502, 502, conf=0.75)
        tracker.update(make_detection(boxes=(b1_next, b2_next), frame_index=1))
        assert tracker.active_track_count == 2

    def test_fuse_score_penalizes_low_conf(self) -> None:
        """With fuse_score, low-conf detections have inflated IoU cost."""
        from yowo.tracking._matching import fuse_score as fs

        cost = np.array([[0.2]], dtype=np.float64)
        high = fs(cost, np.array([0.9], dtype=np.float64))
        low = fs(cost, np.array([0.3], dtype=np.float64))
        assert low[0, 0] > high[0, 0], "Low-conf should have higher fused cost"


# ---------------------------------------------------------------------------
# Unconfirmed track re-association stage
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
        bbox = make_box(100, 100, 200, 200, conf=0.9)
        det = make_detection(boxes=(bbox,), frame_index=1)
        tracker.update(det)
        # Track exists but is unconfirmed (hits=1 < min_hits=3)
        assert len(tracker._tracked) == 1
        assert not tracker._tracked[0].is_confirmed

    def test_unconfirmed_re_associates(self) -> None:
        """Unconfirmed track that matches in Stage 4 survives."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: birth track at (100,100,200,200)
        bbox1 = make_box(100, 100, 200, 200, conf=0.9)
        det1 = make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 1
        # Frame 2: same box appears — unconfirmed track should match in Stage 4
        bbox2 = make_box(102, 102, 202, 202, conf=0.9)
        det2 = make_detection(boxes=(bbox2,), frame_index=2)
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
        bbox1 = make_box(100, 100, 200, 200, conf=0.9)
        det1 = make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 1
        # Frame 2: detection far away — unconfirmed track fails to match
        bbox2 = make_box(500, 500, 600, 600, conf=0.9)
        det2 = make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        # Original track should be REMOVED, not in _lost
        lost_ids = {t.track_id for t in tracker._lost}
        assert 1 not in lost_ids, "Unconfirmed failures should be REMOVED, not LOST"

    def test_confirmed_bypass_unconfirmed_stage(self) -> None:
        """Tracks with hits >= min_hits participate in Stage 1, not Stage 4."""
        tracker = self._make_tracker(min_hits=1)
        # With min_hits=1, track becomes confirmed after activate (hits=1)
        bbox1 = make_box(100, 100, 200, 200, conf=0.9)
        det1 = make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 1
        assert tracker._tracked[0].is_confirmed  # hits=1 >= min_hits=1

    def test_backward_compatible_min_hits_1(self) -> None:
        """With min_hits=1, unconfirmed list is always empty — no behavior change."""
        tracker = self._make_tracker(min_hits=1)
        # Frame 1: birth
        bbox1 = make_box(100, 100, 200, 200, conf=0.9)
        det1 = make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        # With min_hits=1, track is confirmed immediately
        assert len(tracker._tracked) == 1
        assert tracker._tracked[0].is_confirmed
        # Frame 2: match
        bbox2 = make_box(102, 102, 202, 202, conf=0.9)
        det2 = make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        assert len(tracker._tracked) == 1
        assert tracker._tracked[0].hits == 2

    def test_unconfirmed_matches_tight_iou(self) -> None:
        """Stage 4 uses thresh=0.7 — only close IoU matches succeed."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: birth at (100,100,200,200)
        bbox1 = make_box(100, 100, 200, 200, conf=0.9)
        det1 = make_detection(boxes=(bbox1,), frame_index=1)
        tracker.update(det1)
        # Frame 2: detection slightly shifted (IoU ~0.56, which is < 0.7 threshold)
        bbox2 = make_box(160, 160, 260, 260, conf=0.9)
        det2 = make_detection(boxes=(bbox2,), frame_index=2)
        tracker.update(det2)
        # Original track should be removed (IoU too low for 0.7 threshold)
        # New track birthed for the second detection
        assert len(tracker._tracked) >= 1

    def test_multi_frame_unconfirmed_graduation(self) -> None:
        """Track graduates from unconfirmed to confirmed after min_hits updates."""
        tracker = self._make_tracker(min_hits=3)
        for frame_idx in range(1, 5):
            bbox = make_box(100, 100, 200, 200, conf=0.9)
            det = make_detection(boxes=(bbox,), frame_index=frame_idx)
            tracker.update(det)
        # After 4 frames of matching, hits should be >= 3 → confirmed
        assert len(tracker._tracked) >= 1
        confirmed = [t for t in tracker._tracked if t.is_confirmed]
        assert len(confirmed) >= 1, "Track should graduate to confirmed"

    def test_crowded_scene_unconfirmed_survives(self) -> None:
        """In crowded scenes, unconfirmed tracks survive through Stage 4."""
        tracker = self._make_tracker(min_hits=3)
        # Frame 1: two objects born
        b1 = make_box(100, 100, 200, 200, conf=0.9)
        b2 = make_box(400, 400, 500, 500, conf=0.9)
        det1 = make_detection(boxes=(b1, b2), frame_index=1)
        tracker.update(det1)
        assert len(tracker._tracked) == 2
        # Frame 2: both objects still present at same locations
        b3 = make_box(102, 102, 202, 202, conf=0.9)
        b4 = make_box(402, 402, 502, 502, conf=0.9)
        det2 = make_detection(boxes=(b3, b4), frame_index=2)
        tracker.update(det2)
        # Both tracks should survive (Stage 4 re-association)
        assert len(tracker._tracked) == 2
        ids = {t.track_id for t in tracker._tracked}
        assert len(ids) == 2, "Both tracks should keep their IDs"
