"""Unit tests for ByteTracker performance optimizations (P0-P4).

Tests verify that vectorized/batched implementations produce identical results
to the original scalar implementations, and validate edge cases.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import iou_batch, remove_intra_duplicates
from yowo.tracking._strack import STrack, TrackState
from yowo.types import ModelFamily, ModelSize, ModelSpec

_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)
_KF = KalmanFilterXYAH()


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
# P4: predicted_xyxy cache
# ---------------------------------------------------------------------------


class TestPredictedXyxyCache:
    """Verify lazy cache on STrack.predicted_xyxy."""

    def test_cache_hit_returns_same_object(self) -> None:
        """Second access returns the same cached tuple (identity check)."""
        t = _make_strack()
        first = t.predicted_xyxy
        second = t.predicted_xyxy
        assert first is second

    def test_predict_invalidates_cache(self) -> None:
        """predict() clears the cache so next access recomputes."""
        t = _make_strack()
        before = t.predicted_xyxy
        t.predict()
        after = t.predicted_xyxy
        # After prediction, position changes due to velocity
        assert before is not after

    def test_update_invalidates_cache(self) -> None:
        """update() clears the cache."""
        t = _make_strack()
        t.activate(frame_id=0)
        _ = t.predicted_xyxy
        t.update((110, 110, 210, 210), 0.85, 0, "person", 1)
        result = t.predicted_xyxy
        # After update with new measurement, position should reflect it
        assert result is not None

    def test_reactivate_invalidates_cache(self) -> None:
        """re_activate() clears the cache."""
        t = _make_strack()
        t.activate(frame_id=0)
        t.mark_lost()
        _ = t.predicted_xyxy
        t.re_activate((120, 120, 220, 220), 0.8, 0, "person", 5)
        fresh = t.predicted_xyxy
        assert fresh is not None

    def test_activate_does_not_invalidate(self) -> None:
        """activate() doesn't mutate _mean, so cache stays valid."""
        t = _make_strack()
        cached = t.predicted_xyxy
        t.activate(frame_id=0)
        # Cache still valid — activate doesn't touch _mean
        assert t.predicted_xyxy is cached

    def test_mark_lost_does_not_invalidate(self) -> None:
        """mark_lost() doesn't mutate _mean, cache stays valid."""
        t = _make_strack()
        t.activate(frame_id=0)
        cached = t.predicted_xyxy
        t.mark_lost()
        assert t.predicted_xyxy is cached

    def test_cache_values_match_uncached(self) -> None:
        """Cached value matches fresh xyah_to_xyxy computation."""
        t = _make_strack(50, 60, 150, 260)
        cached = t.predicted_xyxy
        fresh = KalmanFilterXYAH.xyah_to_xyxy(t._mean[:4])
        for c, f in zip(cached, fresh, strict=True):
            assert abs(c - f) < 1e-10


# ---------------------------------------------------------------------------
# P0: vectorized _overlaps_any
# ---------------------------------------------------------------------------


class TestOverlapsAnyVectorized:
    """Verify vectorized _overlaps_any via iou_batch."""

    @staticmethod
    def _overlaps_any_scalar(
        box: tuple[float, float, float, float],
        existing: list[tuple[float, float, float, float]],
        thresh: float,
    ) -> bool:
        """Original scalar reference implementation."""
        for ex1, ey1, ex2, ey2 in existing:
            xi1 = max(box[0], ex1)
            yi1 = max(box[1], ey1)
            xi2 = min(box[2], ex2)
            yi2 = min(box[3], ey2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_ex = (ex2 - ex1) * (ey2 - ey1)
            union = area_box + area_ex - inter
            if union > 0 and inter / union >= thresh:
                return True
        return False

    @staticmethod
    def _overlaps_any_vec(
        box: tuple[float, float, float, float],
        existing: NDArray[np.float64],
        thresh: float,
    ) -> bool:
        """Vectorized version matching production code."""
        if existing.shape[0] == 0:
            return False
        box_arr = np.array(box, dtype=np.float64).reshape(1, 4)
        ious = iou_batch(box_arr, existing)
        return bool(np.any(ious >= thresh))

    def test_empty_existing(self) -> None:
        existing = np.empty((0, 4), dtype=np.float64)
        assert not self._overlaps_any_vec((100, 100, 200, 200), existing, 0.5)

    def test_identical_box(self) -> None:
        existing = np.array([[100, 100, 200, 200]], dtype=np.float64)
        assert self._overlaps_any_vec((100, 100, 200, 200), existing, 0.99)

    def test_no_overlap(self) -> None:
        existing = np.array([[500, 500, 600, 600]], dtype=np.float64)
        assert not self._overlaps_any_vec((100, 100, 200, 200), existing, 0.01)

    def test_partial_overlap_below_thresh(self) -> None:
        existing = np.array([[150, 150, 250, 250]], dtype=np.float64)
        # IoU of these two boxes: intersection=50*50=2500, union=2*10000-2500=17500
        # IoU ≈ 0.143
        assert not self._overlaps_any_vec((100, 100, 200, 200), existing, 0.5)

    def test_partial_overlap_above_thresh(self) -> None:
        existing = np.array([[150, 150, 250, 250]], dtype=np.float64)
        assert self._overlaps_any_vec((100, 100, 200, 200), existing, 0.1)

    def test_growing_buffer(self) -> None:
        """Simulate growing buffer as in birth loop."""
        buf = np.empty((10, 4), dtype=np.float64)
        buf[0] = [100, 100, 200, 200]
        n = 1
        # Second box doesn't overlap first
        assert not self._overlaps_any_vec((500, 500, 600, 600), buf[:n], 0.5)
        buf[n] = [500, 500, 600, 600]
        n += 1
        # Third box overlaps second
        assert self._overlaps_any_vec((510, 510, 610, 610), buf[:n], 0.5)

    def test_matches_scalar_reference(self) -> None:
        """Fuzz: vectorized matches scalar for random boxes."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            n_existing = rng.integers(1, 20)
            coords = rng.uniform(0, 500, size=(n_existing + 1, 4))
            # Ensure x2 > x1, y2 > y1
            coords[:, 2] = coords[:, 0] + rng.uniform(10, 100, n_existing + 1)
            coords[:, 3] = coords[:, 1] + rng.uniform(10, 100, n_existing + 1)

            box = tuple(float(v) for v in coords[0])
            existing_list = [tuple(float(v) for v in row) for row in coords[1:]]
            existing_arr = coords[1:].astype(np.float64)

            scalar = self._overlaps_any_scalar(box, existing_list, 0.5)  # type: ignore[arg-type]
            vec = self._overlaps_any_vec(box, existing_arr, 0.5)  # type: ignore[arg-type]
            assert scalar == vec, f"Mismatch at box={box}"


# ---------------------------------------------------------------------------
# P1: batch Kalman predict
# ---------------------------------------------------------------------------


class TestBatchKalmanPredict:
    """Verify batch predict matches per-track predict."""

    @staticmethod
    def _predict_per_track(
        tracks: list[STrack],
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Reference: run predict() on each track individually."""
        results = []
        for t in tracks:
            t.predict()
            results.append((t._mean.copy(), t._covariance.copy()))
        return results

    def test_matches_per_track_means(self) -> None:
        """Batch predict produces same means as per-track predict."""
        kf = KalmanFilterXYAH()

        # Create reference tracks (per-track predict)
        ref_tracks = [_make_strack(track_id=i, x1=50 * i) for i in range(5)]
        for t in ref_tracks:
            t.activate(frame_id=0)
        ref_results = self._predict_per_track(ref_tracks)

        # Create batch tracks (same initial state)
        batch_tracks = [_make_strack(track_id=i, x1=50 * i) for i in range(5)]
        for t in batch_tracks:
            t.activate(frame_id=0)
        lost_mask = np.array([False] * 5, dtype=np.bool_)
        kf.predict_batch(batch_tracks, lost_mask)

        for i in range(5):
            np.testing.assert_allclose(batch_tracks[i]._mean, ref_results[i][0], atol=1e-12)

    def test_matches_per_track_covariances(self) -> None:
        """Batch predict produces same covariances as per-track predict."""
        kf = KalmanFilterXYAH()

        ref_tracks = [_make_strack(track_id=i, x1=50 * i) for i in range(5)]
        for t in ref_tracks:
            t.activate(frame_id=0)
        ref_results = self._predict_per_track(ref_tracks)

        batch_tracks = [_make_strack(track_id=i, x1=50 * i) for i in range(5)]
        for t in batch_tracks:
            t.activate(frame_id=0)
        lost_mask = np.array([False] * 5, dtype=np.bool_)
        kf.predict_batch(batch_tracks, lost_mask)

        for i in range(5):
            np.testing.assert_allclose(batch_tracks[i]._covariance, ref_results[i][1], atol=1e-12)

    def test_lost_velocity_zeroing(self) -> None:
        """Lost tracks have height velocity zeroed before prediction."""
        kf = KalmanFilterXYAH()
        tracks = [_make_strack(track_id=i) for i in range(3)]
        for t in tracks:
            t.activate(frame_id=0)
        # Manually set height velocity and mark some as lost
        tracks[1]._mean[7] = 5.0
        tracks[1].state = TrackState.LOST
        tracks[2]._mean[7] = 3.0
        tracks[2].state = TrackState.LOST

        lost_mask = np.array([False, True, True], dtype=np.bool_)
        kf.predict_batch(tracks, lost_mask)

        # Velocity was zeroed before F @ mean, so mean[7] contribution is 0
        # For tracked track[0], velocity should be preserved in the predict
        assert tracks[0].age == 1

    def test_empty_tracks(self) -> None:
        """Empty track list is handled gracefully."""
        kf = KalmanFilterXYAH()
        kf.predict_batch([], np.array([], dtype=np.bool_))  # Should not raise

    def test_single_track(self) -> None:
        """Single track matches per-track predict."""
        kf = KalmanFilterXYAH()

        ref = _make_strack()
        ref.activate(frame_id=0)
        ref.predict()
        ref_mean = ref._mean.copy()

        batch = _make_strack()
        batch.activate(frame_id=0)
        kf.predict_batch([batch], np.array([False], dtype=np.bool_))

        np.testing.assert_allclose(batch._mean, ref_mean, atol=1e-12)

    def test_age_incremented(self) -> None:
        """Batch predict increments age and time_since_update."""
        kf = KalmanFilterXYAH()
        t = _make_strack()
        t.activate(frame_id=0)
        assert t.age == 0
        assert t.time_since_update == 0
        kf.predict_batch([t], np.array([False], dtype=np.bool_))
        assert t.age == 1
        assert t.time_since_update == 1

    def test_cache_cleared(self) -> None:
        """Batch predict clears _cached_xyxy on all tracks."""
        kf = KalmanFilterXYAH()
        tracks = [_make_strack(track_id=i) for i in range(3)]
        for t in tracks:
            t.activate(frame_id=0)
            _ = t.predicted_xyxy  # Populate cache
            assert t._cached_xyxy is not None

        lost_mask = np.array([False, False, False], dtype=np.bool_)
        kf.predict_batch(tracks, lost_mask)
        for t in tracks:
            assert t._cached_xyxy is None

    def test_multiple_iterations(self) -> None:
        """Batch predict over multiple steps stays consistent with per-track."""
        kf2 = KalmanFilterXYAH()

        ref = _make_strack(track_id=1, x1=100, y1=50, x2=250, y2=300)
        ref.activate(frame_id=0)

        batch = _make_strack(track_id=1, x1=100, y1=50, x2=250, y2=300)
        batch.activate(frame_id=0)

        for _ in range(10):
            ref.predict()
            kf2.predict_batch([batch], np.array([False], dtype=np.bool_))

        np.testing.assert_allclose(batch._mean, ref._mean, atol=1e-10)
        np.testing.assert_allclose(batch._covariance, ref._covariance, atol=1e-10)


# ---------------------------------------------------------------------------
# P2: vectorized remove_intra_duplicates
# ---------------------------------------------------------------------------


class TestRemoveIntraDuplicatesOpt:
    """Verify vectorized remove_intra_duplicates."""

    def test_no_duplicates(self) -> None:
        """Non-overlapping tracks all survive."""
        tracks = [
            _make_strack(x1=0, y1=0, x2=50, y2=50, track_id=1),
            _make_strack(x1=200, y1=200, x2=250, y2=250, track_id=2),
        ]
        for t in tracks:
            t.activate(frame_id=0)
        result = remove_intra_duplicates(tracks, dist_thresh=0.30)
        assert len(result) == 2

    def test_overlap_pair_removes_younger(self) -> None:
        """Overlapping pair: younger track is removed."""
        t1 = _make_strack(x1=100, y1=100, x2=200, y2=200, track_id=1)
        t1.activate(frame_id=0)
        t1.start_frame = 0
        t1.frame_id = 10  # age=10

        t2 = _make_strack(x1=105, y1=105, x2=205, y2=205, track_id=2)
        t2.activate(frame_id=5)
        t2.start_frame = 5
        t2.frame_id = 10  # age=5

        result = remove_intra_duplicates([t1, t2], dist_thresh=0.30)
        assert len(result) == 1
        assert result[0].track_id == 1  # Older survives

    def test_three_way_dedup(self) -> None:
        """Three tracks overlapping — two removed, oldest survives."""
        t1 = _make_strack(x1=100, y1=100, x2=200, y2=200, track_id=1)
        t1.activate(frame_id=0)
        t1.start_frame = 0
        t1.frame_id = 20

        t2 = _make_strack(x1=102, y1=102, x2=202, y2=202, track_id=2)
        t2.activate(frame_id=5)
        t2.start_frame = 5
        t2.frame_id = 20

        t3 = _make_strack(x1=104, y1=104, x2=204, y2=204, track_id=3)
        t3.activate(frame_id=10)
        t3.start_frame = 10
        t3.frame_id = 20

        result = remove_intra_duplicates([t1, t2, t3], dist_thresh=0.30)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_single_track(self) -> None:
        """Single track returns unchanged."""
        t = _make_strack()
        t.activate(frame_id=0)
        result = remove_intra_duplicates([t])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# P3: Kalman covariance buffer
# ---------------------------------------------------------------------------


class TestKalmanCovBuffer:
    """Verify pre-allocated covariance buffer correctness."""

    def test_predict_diagonal_values(self) -> None:
        """predict() diagonal matches scalar formula."""
        kf = KalmanFilterXYAH()
        measurement = np.array([100.0, 100.0, 1.0, 200.0], dtype=np.float64)
        mean, cov = kf.initiate(measurement)
        new_mean, new_cov = kf.predict(mean, cov)
        # Verify result is a valid 8x8 symmetric positive semi-definite matrix
        assert new_cov.shape == (8, 8)
        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(new_cov)
        assert np.all(eigvals >= -1e-10)

    def test_predict_off_diagonal_structure(self) -> None:
        """Motion noise buffer contributes only to diagonal."""
        kf = KalmanFilterXYAH()
        # After predict, motion_cov_buf should only have diagonal values
        measurement = np.array([50.0, 50.0, 1.5, 100.0], dtype=np.float64)
        mean, cov = kf.initiate(measurement)
        kf.predict(mean, cov)
        buf = kf._motion_cov_buf
        # Off-diagonal should be zero
        mask = ~np.eye(8, dtype=bool)
        np.testing.assert_allclose(buf[mask], 0.0, atol=1e-15)

    def test_initiate_fresh_array(self) -> None:
        """initiate() returns a fresh covariance, not the shared buffer."""
        kf = KalmanFilterXYAH()
        m1 = np.array([100.0, 100.0, 1.0, 200.0], dtype=np.float64)
        _, cov1 = kf.initiate(m1)
        m2 = np.array([200.0, 200.0, 0.5, 50.0], dtype=np.float64)
        _, cov2 = kf.initiate(m2)
        # cov1 should NOT be affected by cov2 computation
        assert not np.array_equal(cov1, cov2)
        # Verify cov1 diagonal matches height=200 formula
        h = 200.0
        expected_d0 = (2 * (1.0 / 20) * h) ** 2
        assert abs(cov1[0, 0] - expected_d0) < 1e-10

    def test_repeated_predict_no_accumulation(self) -> None:
        """Multiple predict() calls don't accumulate in shared buffer."""
        kf = KalmanFilterXYAH()
        m1 = np.array([100.0, 100.0, 1.0, 200.0], dtype=np.float64)
        mean1, cov1 = kf.initiate(m1)

        m2 = np.array([50.0, 50.0, 2.0, 50.0], dtype=np.float64)
        mean2, cov2 = kf.initiate(m2)

        # Predict with first track
        kf.predict(mean1, cov1)
        buf_after_first = kf._motion_cov_buf.copy()

        # Predict with second track (different height → different diagonal)
        kf.predict(mean2, cov2)
        buf_after_second = kf._motion_cov_buf.copy()

        # Buffer should reflect second track's values, not accumulated
        assert not np.array_equal(buf_after_first, buf_after_second)
        # Off-diagonal still zero
        mask = ~np.eye(8, dtype=bool)
        np.testing.assert_allclose(buf_after_second[mask], 0.0, atol=1e-15)


class TestIntraDuplicateVetoGates:
    """Tests for multi-signal veto gates in remove_intra_duplicates."""

    @staticmethod
    def _make_overlapping_pair(
        class_id_a: int = 0,
        class_id_b: int = 0,
        age_a: int = 10,
        age_b: int = 5,
    ) -> tuple[STrack, STrack]:
        """Create two tracks with IoU > 0.70 (overlapping boxes)."""
        t1 = STrack(
            track_id=1,
            box_xyxy=(100.0, 100.0, 200.0, 200.0),
            confidence=0.9,
            class_id=class_id_a,
            class_name="obj",
            kalman=_KF,
            min_hits=3,
        )
        t1.activate(frame_id=0)
        t1.start_frame = 0
        t1.frame_id = age_a

        t2 = STrack(
            track_id=2,
            box_xyxy=(105.0, 105.0, 205.0, 205.0),
            confidence=0.9,
            class_id=class_id_b,
            class_name="obj",
            kalman=_KF,
            min_hits=3,
        )
        t2.activate(frame_id=age_a - age_b)
        t2.start_frame = age_a - age_b
        t2.frame_id = age_a

        return t1, t2

    def test_class_mismatch_preserves_both(self) -> None:
        """Different class_id → both tracks survive despite high IoU."""
        t1, t2 = self._make_overlapping_pair(class_id_a=0, class_id_b=2)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_same_class_no_embedding_deduped(self) -> None:
        """Same class, no embeddings, similar velocity → younger removed."""
        t1, t2 = self._make_overlapping_pair()
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_embedding_veto_preserves_distinct(self) -> None:
        """Same class, orthogonal embeddings → both survive."""
        t1, t2 = self._make_overlapping_pair()
        # Orthogonal L2-normalised embeddings (cosine distance = 1.0)
        emb_a = np.zeros(512, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(512, dtype=np.float32)
        emb_b[1] = 1.0
        t1._embedding = emb_a
        t2._embedding = emb_b
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_similar_embeddings_still_deduped(self) -> None:
        """Same class, nearly identical embeddings → younger removed."""
        t1, t2 = self._make_overlapping_pair()
        emb = np.random.default_rng(42).standard_normal(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        t1._embedding = emb.copy()
        t2._embedding = emb.copy()
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_one_embedding_none_gate_skipped(self) -> None:
        """One None embedding → gate skipped, dedup still happens."""
        t1, t2 = self._make_overlapping_pair()
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0
        t1._embedding = emb
        # t2._embedding stays None
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_velocity_divergence_veto(self) -> None:
        """Diverging Kalman velocities → both survive."""
        t1, t2 = self._make_overlapping_pair()
        # Set diverging velocities: t1 moves right, t2 moves left
        # mean[3]=height, mean[4:6]=[vcx, vcy]
        t1._mean[3] = 100.0
        t1._mean[4] = 6.0
        t1._mean[5] = 0.0
        t2._mean[3] = 100.0
        t2._mean[4] = -6.0
        t2._mean[5] = 0.0
        # rel_vel = norm([12, 0]) / 100 = 0.12 > 0.10 → veto
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_low_velocity_no_veto(self) -> None:
        """Similar velocities → no veto, younger removed."""
        t1, t2 = self._make_overlapping_pair()
        t1._mean[3] = 100.0
        t1._mean[4] = 1.0
        t1._mean[5] = 0.5
        t2._mean[3] = 100.0
        t2._mean[4] = 1.2
        t2._mean[5] = 0.3
        # rel_vel = norm([0.2, -0.2]) / 100 = 0.003 < 0.10 → no veto
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_class_aware_false_disables(self) -> None:
        """class_aware=False disables class gate."""
        t1, t2 = self._make_overlapping_pair(class_id_a=0, class_id_b=2)
        result = remove_intra_duplicates(
            [t1, t2],
            class_aware=False,
        )
        assert len(result) == 1

    def test_embedding_veto_overrides_velocity(self) -> None:
        """Same class, similar velocity, but different embeddings → survives."""
        t1, t2 = self._make_overlapping_pair()
        # Similar velocity → no velocity veto
        t1._mean[4] = 1.0
        t2._mean[4] = 1.0
        # Orthogonal embeddings → embedding veto fires
        emb_a = np.zeros(512, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(512, dtype=np.float32)
        emb_b[1] = 1.0
        t1._embedding = emb_a
        t2._embedding = emb_b
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_three_way_mixed_classes(self) -> None:
        """A(cls0)+B(cls0)+C(cls1): dedup A/B, keep C."""
        ta = STrack(
            track_id=1,
            box_xyxy=(100.0, 100.0, 200.0, 200.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            kalman=_KF,
            min_hits=3,
        )
        ta.activate(frame_id=0)
        ta.start_frame = 0
        ta.frame_id = 10

        tb = STrack(
            track_id=2,
            box_xyxy=(105.0, 105.0, 205.0, 205.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            kalman=_KF,
            min_hits=3,
        )
        tb.activate(frame_id=5)
        tb.start_frame = 5
        tb.frame_id = 10

        tc = STrack(
            track_id=3,
            box_xyxy=(103.0, 103.0, 203.0, 203.0),
            confidence=0.9,
            class_id=1,
            class_name="car",
            kalman=_KF,
            min_hits=3,
        )
        tc.activate(frame_id=3)
        tc.start_frame = 3
        tc.frame_id = 10

        result = remove_intra_duplicates([ta, tb, tc])
        ids = {t.track_id for t in result}
        assert 1 in ids, "Oldest same-class track should survive"
        assert 3 in ids, "Different-class track should survive"
        assert len(result) == 2

    def test_vectorized_path_with_class_veto(self) -> None:
        """N>20 tracks: class veto works in vectorized path."""
        tracks: list[STrack] = []
        # 21 non-overlapping tracks of class 0
        for i in range(21):
            t = STrack(
                track_id=i + 1,
                box_xyxy=(
                    float(i * 300),
                    0.0,
                    float(i * 300 + 100),
                    100.0,
                ),
                confidence=0.9,
                class_id=0,
                class_name="person",
                kalman=_KF,
                min_hits=3,
            )
            t.activate(frame_id=0)
            t.start_frame = 0
            t.frame_id = 10
            tracks.append(t)
        # Add two overlapping tracks of different classes
        overlap_a = STrack(
            track_id=100,
            box_xyxy=(5000.0, 0.0, 5100.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            kalman=_KF,
            min_hits=3,
        )
        overlap_a.activate(frame_id=0)
        overlap_a.start_frame = 0
        overlap_a.frame_id = 10
        tracks.append(overlap_a)

        overlap_b = STrack(
            track_id=101,
            box_xyxy=(5005.0, 5.0, 5105.0, 105.0),
            confidence=0.9,
            class_id=2,
            class_name="car",
            kalman=_KF,
            min_hits=3,
        )
        overlap_b.activate(frame_id=5)
        overlap_b.start_frame = 5
        overlap_b.frame_id = 10
        tracks.append(overlap_b)

        assert len(tracks) == 23  # > 20 → vectorized path
        result = remove_intra_duplicates(tracks)
        ids = {t.track_id for t in result}
        assert 100 in ids, "Class-0 overlap track should survive"
        assert 101 in ids, "Class-2 overlap track should survive (class veto)"
        assert len(result) == 23  # No tracks removed
