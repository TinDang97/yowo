"""Unit tests for Stage 2.5 lost-track recovery and stationary track retention."""

from __future__ import annotations

from unit._tracking_helpers import make_box, make_detection
from yowo.tracking._tracker import ByteTracker


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
        car = make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(make_detection(boxes=(car,), frame_index=0))
        assert r0.num_boxes == 1
        original_id = r0.boxes[0].track_id

        # Frames 1-10: car occluded → no detections → track goes to lost
        for i in range(1, 11):
            tracker.update(make_detection(boxes=(), frame_index=i))
        assert tracker.lost_track_count >= 1

        # Frame 11: car reappears at low conf (0.56 < track_high_thresh=0.6)
        reappear = make_box(100, 200, 300, 400, conf=0.56, cls_id=2, cls_name="car")
        r11 = tracker.update(make_detection(boxes=(reappear,), frame_index=11))
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
            box = make_box(
                100 + i * 20,
                200,
                300 + i * 20,
                400,
                conf=0.85,
                cls_id=2,
                cls_name="car",
            )
            tracker.update(make_detection(boxes=(box,), frame_index=i))
        original_id = tracker._tracked[0].track_id

        # Occlude for 20 frames (> max_age//2 = 15 → too stale for Stage 2.5)
        for i in range(3, 23):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Reappear at low conf at last known position — should NOT recover
        reappear = make_box(140, 200, 340, 400, conf=0.56, cls_id=2, cls_name="car")
        r23 = tracker.update(make_detection(boxes=(reappear,), frame_index=23))
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
        car = make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(make_detection(boxes=(car,), frame_index=0))
        car_id = r0.boxes[0].track_id

        # Occlude for 5 frames
        for i in range(1, 6):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Person appears at same location with low conf
        person = make_box(100, 200, 300, 400, conf=0.56, cls_id=0, cls_name="person")
        r6 = tracker.update(make_detection(boxes=(person,), frame_index=6))
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
        car = make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(make_detection(boxes=(car,), frame_index=0))
        original_id = r0.boxes[0].track_id

        # Occlude for 5 frames
        for i in range(1, 6):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Reappear at high conf → Stage 1 should match
        reappear = make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r6 = tracker.update(make_detection(boxes=(reappear,), frame_index=6))
        assert r6.num_boxes >= 1
        assert r6.boxes[0].track_id == original_id

    def test_zero_overhead_no_lost(self) -> None:
        """When there are no lost tracks, Stage 2.5 adds zero overhead."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
        )
        box = make_box(100, 200, 300, 400, conf=0.85)
        tracker.update(make_detection(boxes=(box,), frame_index=0))
        assert tracker.lost_track_count == 0

        # Low-conf detection nearby → Stage 2 handles it, Stage 2.5 skips
        low = make_box(105, 205, 305, 405, conf=0.5)
        r1 = tracker.update(make_detection(boxes=(low,), frame_index=1))
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
        box_a = make_box(100, 200, 300, 400, conf=0.85)
        tracker.update(make_detection(boxes=(box_a,), frame_index=0))
        # Track B: tracked then lost
        box_b = make_box(500, 200, 700, 400, conf=0.85)
        tracker.update(make_detection(boxes=(box_a, box_b), frame_index=1))
        # Lose track B
        for i in range(2, 5):
            tracker.update(make_detection(boxes=(box_a,), frame_index=i))
        assert tracker.lost_track_count >= 1

        # Low-conf at A's position → consumed by Stage 2 for track A
        low_a = make_box(100, 200, 300, 400, conf=0.5)
        tracker.update(make_detection(boxes=(low_a,), frame_index=5))
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
        car = make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-14: car not detected (occluded by passing vehicle)
        for i in range(5, 15):
            tracker.update(make_detection(boxes=(), frame_index=i))
        # Parked car should be lost
        lost_ids = {t.track_id for t in tracker._lost}
        assert car_id in lost_ids

        # Frame 15: occlusion clears, parked car reappears at low-conf
        reappear = make_box(100, 300, 250, 450, conf=0.56, cls_id=2, cls_name="car")
        r15 = tracker.update(make_detection(boxes=(reappear,), frame_index=15))
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
        car = make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: car disappears (15 frames > max_age=10)
        for i in range(5, 20):
            tracker.update(make_detection(boxes=(), frame_index=i))

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
            box = make_box(
                100 + i * 20,
                300,
                250 + i * 20,
                450,
                conf=0.82,
                cls_id=2,
                cls_name="car",
            )
            tracker.update(make_detection(boxes=(box,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: disappears for 15 frames (> max_age=10)
        for i in range(5, 20):
            tracker.update(make_detection(boxes=(), frame_index=i))

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
        car = make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: car disappears for 15 frames (> max_age=10)
        for i in range(5, 20):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Frame 20: car reappears at HIGH conf → Stage 1 recovery
        reappear = make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        r20 = tracker.update(make_detection(boxes=(reappear,), frame_index=20))

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
        car = make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        for i in range(5):
            tracker.update(make_detection(boxes=(car,), frame_index=i))
        car_id = tracker._tracked[0].track_id

        # Frames 5-19: car disappears for 15 frames (> max_age=10, > lost_low_age=5)
        for i in range(5, 20):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Frame 20: car reappears at LOW conf → Stage 2.5 with extended recency
        reappear = make_box(100, 300, 250, 450, conf=0.50, cls_id=2, cls_name="car")
        r20 = tracker.update(make_detection(boxes=(reappear,), frame_index=20))

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
        car = make_box(100, 300, 250, 450, conf=0.82, cls_id=2, cls_name="car")
        tracker.update(make_detection(boxes=(car,), frame_index=0))
        track_id = tracker._tracked[0].track_id

        # Disappear for 8 frames (> max_age=5, < stationary_max_age=15)
        for i in range(1, 9):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Should be removed — NOT given extended retention
        lost_ids = {t.track_id for t in tracker._lost}
        assert track_id not in lost_ids, (
            "Short-lived track (hits < min_hits) must not get extended retention"
        )
