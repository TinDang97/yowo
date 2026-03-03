"""Tests for devil-advocate review fixes: veto gates, velocity copy, lost cap, etc."""

from __future__ import annotations

import numpy as np

from unit._tracking_helpers import make_box, make_detection
from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import remove_duplicate_tracks, remove_intra_duplicates
from yowo.tracking._strack import STrack
from yowo.tracking._tracker import ByteTracker

# ---------------------------------------------------------------------------
# C-1: remove_duplicate_tracks now has class gate
# ---------------------------------------------------------------------------


class TestCrossListClassGate:
    """Verify remove_duplicate_tracks respects class/embedding/velocity veto."""

    def _make_pair(
        self,
        cls_a: int = 0,
        cls_b: int = 0,
        name_a: str = "person",
        name_b: str = "person",
    ) -> tuple[list[STrack], list[STrack]]:
        """Create two overlapping tracks with different class IDs."""
        kf = KalmanFilterXYAH()
        ta = STrack(1, (100, 200, 300, 400), 0.9, cls_a, name_a, kf, 1)
        ta.activate(0)
        ta.state = ta.state  # TRACKED
        tb = STrack(2, (100, 200, 300, 400), 0.9, cls_b, name_b, kf, 1)
        tb.activate(0)
        return [ta], [tb]

    def test_same_class_deduplicates(self) -> None:
        """Same class, perfect IoU overlap → shorter-lived track removed."""
        a_list, b_list = self._make_pair(cls_a=0, cls_b=0)
        fa, fb = remove_duplicate_tracks(a_list, b_list)
        total = len(fa) + len(fb)
        assert total == 1, "Same-class overlap should deduplicate"

    def test_different_class_preserves_both(self) -> None:
        """Different class IDs → both tracks survive despite high IoU."""
        a_list, b_list = self._make_pair(cls_a=0, cls_b=2, name_a="person", name_b="car")
        fa, fb = remove_duplicate_tracks(a_list, b_list)
        assert len(fa) == 1
        assert len(fb) == 1

    def test_cross_list_class_gate_tracked_vs_lost(self) -> None:
        """A lost car at same position as tracked person is NOT removed by dedup."""
        kf = KalmanFilterXYAH()
        # Person: actively tracked, younger
        person = STrack(1, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        person.activate(0)
        # Car: lost, older (higher age = more frames of history)
        car = STrack(2, (105, 205, 305, 405), 0.9, 2, "car", kf, 1)
        car.activate(0)
        car.frame_id = 10  # older track
        car.start_frame = 0

        tracked, lost = remove_duplicate_tracks([person], [car])
        # Both should survive — different class IDs
        assert len(tracked) == 1 and tracked[0].track_id == 1
        assert len(lost) == 1 and lost[0].track_id == 2


# ---------------------------------------------------------------------------
# C-2: velocity property returns a copy (mutation safety)
# ---------------------------------------------------------------------------


class TestVelocityCopySafety:
    """Verify that STrack.velocity returns a copy, not a view."""

    def test_mutation_does_not_corrupt_state(self) -> None:
        """Mutating the returned velocity array must not affect Kalman state."""
        kf = KalmanFilterXYAH()
        t = STrack(1, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t.activate(0)
        # Advance a few predict/update cycles to get non-zero velocity
        t.update((105, 205, 305, 405), 0.9, 0, "person", 1)
        t.predict()

        v = t.velocity
        original = v.copy()
        # Mutate returned array
        v[0] = 999.0
        v[1] = -999.0

        # Internal state should be unaffected
        v2 = t.velocity
        np.testing.assert_array_equal(v2, original)

    def test_velocity_is_not_view(self) -> None:
        """Velocity should not share memory with internal state."""
        kf = KalmanFilterXYAH()
        t = STrack(1, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t.activate(0)
        v = t.velocity
        assert v.base is not t._mean, "velocity must not be a view of _mean"


# ---------------------------------------------------------------------------
# W-1: Lost pool cap
# ---------------------------------------------------------------------------


class TestLostPoolCap:
    """Verify max_lost caps the lost pool size."""

    def test_lost_pool_capped(self) -> None:
        """Lost pool does not exceed max_lost."""
        max_lost = 5
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=100,  # large so tracks stay lost a long time
            max_lost=max_lost,
        )
        # Create 10 tracks by spawning them across frames
        for i in range(10):
            box = make_box(
                x1=float(i * 50),
                y1=100,
                x2=float(i * 50 + 40),
                y2=200,
                conf=0.85,
            )
            tracker.update(make_detection(boxes=(box,), frame_index=i))

        # All 10 tracked — now lose them all
        for i in range(10, 20):
            tracker.update(make_detection(boxes=(), frame_index=i))

        assert tracker.lost_track_count <= max_lost

    def test_evicts_oldest_unseen(self) -> None:
        """Eviction removes tracks with highest time_since_update first."""
        max_lost = 2
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=100,
            max_lost=max_lost,
        )
        # Track A: appears at frame 0, disappears at frame 1
        box_a = make_box(0, 0, 50, 50, conf=0.85)
        tracker.update(make_detection(boxes=(box_a,), frame_index=0))

        # Track B: appears at frame 1, disappears at frame 2
        box_b = make_box(200, 200, 250, 250, conf=0.85)
        tracker.update(make_detection(boxes=(box_a, box_b), frame_index=1))

        # Track C: appears at frame 2, disappears at frame 3
        box_c = make_box(400, 400, 450, 450, conf=0.85)
        tracker.update(make_detection(boxes=(box_a, box_b, box_c), frame_index=2))

        # Now lose all three by only providing nothing
        for i in range(3, 8):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Should have at most max_lost=2 in lost pool
        assert tracker.lost_track_count <= max_lost


# ---------------------------------------------------------------------------
# W-2: np.inf class gate in Stage 2.5
# ---------------------------------------------------------------------------


class TestStage25InfGate:
    """Verify Stage 2.5 class gate is unconditional (np.inf, not 1.0)."""

    def test_cross_class_blocked_regardless_of_threshold(self) -> None:
        """Different class is never matched even with very lenient IoU."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            max_age=30,
        )
        # Track a car
        car = make_box(100, 200, 300, 400, conf=0.85, cls_id=2, cls_name="car")
        r0 = tracker.update(make_detection(boxes=(car,), frame_index=0))
        car_id = r0.boxes[0].track_id

        # Lose it
        for i in range(1, 6):
            tracker.update(make_detection(boxes=(), frame_index=i))

        # Person detection at exact same location, low conf → Stage 2.5
        person = make_box(100, 200, 300, 400, conf=0.56, cls_id=0, cls_name="person")
        r6 = tracker.update(make_detection(boxes=(person,), frame_index=6))

        # Person must NOT steal the car's track ID
        for b in r6.boxes:
            assert b.track_id != car_id


# ---------------------------------------------------------------------------
# W-3: Configurable stationary threshold
# ---------------------------------------------------------------------------


class TestConfigurableStationaryThresh:
    """Verify stationary_thresh flows through to STrack.is_stationary."""

    def test_custom_threshold_on_strack(self) -> None:
        """STrack created with custom threshold uses it for is_stationary."""
        kf = KalmanFilterXYAH()
        # Very strict threshold (almost nothing is stationary)
        strict = STrack(1, (100, 200, 300, 400), 0.9, 0, "p", kf, 1, stationary_thresh=0.0001)
        strict.activate(0)
        # Very lenient threshold
        lenient = STrack(3, (100, 200, 300, 400), 0.9, 0, "p", kf, 1, stationary_thresh=1.0)
        lenient.activate(0)

        # Give them noticeable velocity with predict→update cycles
        for i in range(1, 10):
            shift = i * 30
            for t in (strict, lenient):
                t.predict()
                t.update(
                    (100 + shift, 200 + shift, 300 + shift, 400 + shift),
                    0.9,
                    0,
                    "p",
                    i,
                )

        # Lenient (thresh=1.0) should still consider it stationary
        assert lenient.is_stationary
        # Strict (thresh=0.0001) should NOT consider it stationary
        assert not strict.is_stationary

    def test_tracker_passes_threshold_to_new_tracks(self) -> None:
        """ByteTracker with custom stationary_thresh creates tracks that use it."""
        tracker = ByteTracker(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            min_hits=1,
            stationary_thresh=0.5,  # very lenient
        )
        box = make_box(100, 200, 300, 400, conf=0.85)
        tracker.update(make_detection(boxes=(box,), frame_index=0))
        assert tracker._tracked[0]._stationary_thresh == 0.5


# ---------------------------------------------------------------------------
# W-4: Configurable veto gate thresholds
# ---------------------------------------------------------------------------


class TestConfigurableVetoThresholds:
    """Verify embedding/velocity veto thresholds are passed through."""

    def test_very_strict_veto_preserves_more(self) -> None:
        """Very low veto thresholds (strict) → more pairs vetoed."""
        kf = KalmanFilterXYAH()
        # Two same-class overlapping tracks
        t1 = STrack(1, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t1.activate(0)
        t2 = STrack(2, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t2.activate(0)
        # Give t2 slightly different velocity by updating with offset box
        t2.update((102, 202, 302, 402), 0.9, 0, "person", 1)

        # Default thresholds — should deduplicate (small velocity diff)
        result_default = remove_intra_duplicates(
            [t1, t2],
            velocity_veto_thresh=0.10,
        )
        # Very strict threshold — any velocity diff vetoes removal
        result_strict = remove_intra_duplicates(
            [t1, t2],
            velocity_veto_thresh=0.0001,
        )
        assert len(result_strict) >= len(result_default)


# ---------------------------------------------------------------------------
# N-3: Embedding normalization (functional test)
# ---------------------------------------------------------------------------


class TestEmbeddingNormalization:
    """Verify embedding veto gate works correctly with normalised embeddings."""

    def test_orthogonal_embeddings_veto(self) -> None:
        """Orthogonal L2-normalised embeddings (cos_dist=1.0) → veto fires."""
        kf = KalmanFilterXYAH()
        t1 = STrack(1, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t1.activate(0)
        t2 = STrack(2, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t2.activate(0)

        # Orthogonal unit vectors
        dim = 128
        e1 = np.zeros(dim, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(dim, dtype=np.float32)
        e2[1] = 1.0
        t1.update_embedding(e1)
        t2.update_embedding(e2)

        # Should preserve both due to embedding veto
        result = remove_intra_duplicates([t1, t2], embedding_veto_thresh=0.40)
        assert len(result) == 2

    def test_identical_embeddings_no_veto(self) -> None:
        """Identical embeddings (cos_dist=0.0) → veto does NOT fire."""
        kf = KalmanFilterXYAH()
        t1 = STrack(1, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t1.activate(0)
        t2 = STrack(2, (100, 200, 300, 400), 0.9, 0, "person", kf, 1)
        t2.activate(0)

        dim = 128
        e = np.zeros(dim, dtype=np.float32)
        e[0] = 1.0
        t1.update_embedding(e)
        t2.update_embedding(e.copy())

        # Same appearance, same class → should deduplicate
        result = remove_intra_duplicates([t1, t2], embedding_veto_thresh=0.40)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Infeasible cost matrix guard in _hungarian
# ---------------------------------------------------------------------------


class TestHungarianInfeasible:
    """Verify _hungarian handles all-inf cost matrices gracefully."""

    def test_all_inf_returns_empty(self) -> None:
        """All-inf cost matrix → empty assignment, no ValueError."""
        from yowo.tracking._matching import linear_assignment

        cost = np.full((3, 3), np.inf, dtype=np.float64)
        matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)
        assert matches == []
        assert len(unmatched_rows) == 3
        assert len(unmatched_cols) == 3
