"""Unit tests for veto gates in remove_intra_duplicates.

Tests verify class, embedding, and velocity veto gates correctly prevent
false-positive deduplication of distinct overlapping tracks.
"""

from __future__ import annotations

import numpy as np

from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import remove_intra_duplicates
from yowo.tracking._strack import STrack

_KF = KalmanFilterXYAH()


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
        """Different class_id -> both tracks survive despite high IoU."""
        t1, t2 = self._make_overlapping_pair(class_id_a=0, class_id_b=2)
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_same_class_no_embedding_deduped(self) -> None:
        """Same class, no embeddings, similar velocity -> younger removed."""
        t1, t2 = self._make_overlapping_pair()
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_embedding_veto_preserves_distinct(self) -> None:
        """Same class, orthogonal embeddings -> both survive."""
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
        """Same class, nearly identical embeddings -> younger removed."""
        t1, t2 = self._make_overlapping_pair()
        emb = np.random.default_rng(42).standard_normal(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        t1._embedding = emb.copy()
        t2._embedding = emb.copy()
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_one_embedding_none_gate_skipped(self) -> None:
        """One None embedding -> gate skipped, dedup still happens."""
        t1, t2 = self._make_overlapping_pair()
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0
        t1._embedding = emb
        # t2._embedding stays None
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_velocity_divergence_veto(self) -> None:
        """Diverging Kalman velocities -> both survive."""
        t1, t2 = self._make_overlapping_pair()
        # Set diverging velocities: t1 moves right, t2 moves left
        # mean[3]=height, mean[4:6]=[vcx, vcy]
        t1._mean[3] = 100.0
        t1._mean[4] = 6.0
        t1._mean[5] = 0.0
        t2._mean[3] = 100.0
        t2._mean[4] = -6.0
        t2._mean[5] = 0.0
        # rel_vel = norm([12, 0]) / 100 = 0.12 > 0.10 -> veto
        result = remove_intra_duplicates([t1, t2])
        assert len(result) == 2

    def test_low_velocity_no_veto(self) -> None:
        """Similar velocities -> no veto, younger removed."""
        t1, t2 = self._make_overlapping_pair()
        t1._mean[3] = 100.0
        t1._mean[4] = 1.0
        t1._mean[5] = 0.5
        t2._mean[3] = 100.0
        t2._mean[4] = 1.2
        t2._mean[5] = 0.3
        # rel_vel = norm([0.2, -0.2]) / 100 = 0.003 < 0.10 -> no veto
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
        """Same class, similar velocity, but different embeddings -> survives."""
        t1, t2 = self._make_overlapping_pair()
        # Similar velocity -> no velocity veto
        t1._mean[4] = 1.0
        t2._mean[4] = 1.0
        # Orthogonal embeddings -> embedding veto fires
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

        assert len(tracks) == 23  # > 20 -> vectorized path
        result = remove_intra_duplicates(tracks)
        ids = {t.track_id for t in result}
        assert 100 in ids, "Class-0 overlap track should survive"
        assert 101 in ids, "Class-2 overlap track should survive (class veto)"
        assert len(result) == 23  # No tracks removed
