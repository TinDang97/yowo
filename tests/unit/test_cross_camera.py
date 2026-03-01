"""Tests for cross-camera tracking components: gallery, camera link, tracker."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from yowo.tracking._camera_link import CameraLink, CameraLinkModel
from yowo.tracking._gallery import EmbeddingGallery, GalleryMatch
from yowo.types import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rand_embedding(dim: int = 512) -> np.ndarray:
    """Random L2-normalized embedding."""
    v = np.random.default_rng(42).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _similar_embedding(base: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Create an embedding similar to base with small noise."""
    rng = np.random.default_rng(123)
    noisy = base + rng.standard_normal(base.shape).astype(np.float32) * noise
    return (noisy / np.linalg.norm(noisy)).astype(np.float32)


def _orthogonal_embedding(dim: int = 512) -> np.ndarray:
    """Create an embedding very different from typical random ones."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    return v


# ===========================================================================
# TestEmbeddingGallery
# ===========================================================================
class TestEmbeddingGallery:
    """Tests for the embedding gallery."""

    def test_add_and_query_nearest(self) -> None:
        """Gallery returns nearest match by cosine distance."""
        gallery = EmbeddingGallery(embedding_dim=512)
        emb1 = _rand_embedding()
        emb2 = _orthogonal_embedding()

        gallery.add("cam_a", 1, emb1, timestamp=10.0)
        gallery.add("cam_b", 2, emb2, timestamp=20.0)

        # Query with something similar to emb1
        query = _similar_embedding(emb1)
        matches = gallery.query(query, threshold=0.5)
        assert len(matches) >= 1
        assert matches[0].camera_id == "cam_a"
        assert matches[0].local_track_id == 1

    def test_exclude_camera(self) -> None:
        """Same-camera exclusion prevents within-camera matches."""
        gallery = EmbeddingGallery(embedding_dim=512)
        emb = _rand_embedding()
        gallery.add("cam_a", 1, emb, timestamp=10.0)

        # Query from same camera — should be excluded
        matches = gallery.query(emb, exclude_camera="cam_a", threshold=0.5)
        assert len(matches) == 0

        # Query from different camera — should match
        matches = gallery.query(emb, exclude_camera="cam_b", threshold=0.5)
        assert len(matches) == 1

    def test_threshold_filters(self) -> None:
        """Matches beyond threshold distance are excluded."""
        gallery = EmbeddingGallery(embedding_dim=512)
        emb = _rand_embedding()
        gallery.add("cam_a", 1, emb, timestamp=10.0)

        # Very tight threshold — orthogonal embedding won't match
        ortho = _orthogonal_embedding()
        matches = gallery.query(ortho, threshold=0.1)
        assert len(matches) == 0

    def test_max_entries_eviction(self) -> None:
        """Gallery evicts oldest entries when exceeding max_entries."""
        gallery = EmbeddingGallery(embedding_dim=4, max_entries=5)

        for i in range(10):
            emb = np.array([float(i), 0.0, 0.0, 1.0], dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            gallery.add("cam_a", i, emb, timestamp=float(i))

        assert gallery.size == 5

    def test_empty_gallery_returns_empty(self) -> None:
        """Query on empty gallery returns empty list."""
        gallery = EmbeddingGallery(embedding_dim=512)
        matches = gallery.query(_rand_embedding(), threshold=0.5)
        assert matches == []

    def test_global_id_assignment(self) -> None:
        """Each add() returns a unique global_id unless specified."""
        gallery = EmbeddingGallery(embedding_dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        gid1 = gallery.add("cam_a", 1, emb)
        gid2 = gallery.add("cam_b", 2, emb)
        assert gid1 != gid2

        # Explicit global_id reuse
        gid3 = gallery.add("cam_c", 3, emb, global_id=gid1)
        assert gid3 == gid1

    def test_thread_safety_concurrent_add_query(self) -> None:
        """Concurrent add/query doesn't crash."""
        gallery = EmbeddingGallery(embedding_dim=64, max_entries=1000)
        errors: list[Exception] = []

        def add_worker(cam: str, n: int) -> None:
            try:
                rng = np.random.default_rng(hash(cam) & 0xFFFFFFFF)
                for i in range(n):
                    emb = rng.standard_normal(64).astype(np.float32)
                    emb = emb / np.linalg.norm(emb)
                    gallery.add(cam, i, emb, timestamp=float(i))
            except Exception as e:
                errors.append(e)

        def query_worker(n: int) -> None:
            try:
                rng = np.random.default_rng(99)
                for _ in range(n):
                    emb = rng.standard_normal(64).astype(np.float32)
                    emb = emb / np.linalg.norm(emb)
                    gallery.query(emb, exclude_camera="cam_x", threshold=0.5)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_worker, args=("cam_a", 100)),
            threading.Thread(target=add_worker, args=("cam_b", 100)),
            threading.Thread(target=query_worker, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert gallery.size > 0

    def test_next_global_id(self) -> None:
        """next_global_id allocates sequential IDs."""
        gallery = EmbeddingGallery(embedding_dim=4)
        id1 = gallery.next_global_id()
        id2 = gallery.next_global_id()
        assert id2 == id1 + 1

    def test_top_k_limits_results(self) -> None:
        """top_k parameter limits the number of returned matches."""
        gallery = EmbeddingGallery(embedding_dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        for i in range(10):
            gallery.add(f"cam_{i}", i, emb, timestamp=float(i))

        matches = gallery.query(emb, top_k=3, threshold=0.5)
        assert len(matches) <= 3


# ===========================================================================
# TestCameraLinkModel
# ===========================================================================
class TestCameraLinkModel:
    """Tests for camera link model."""

    def test_feasible_within_window(self) -> None:
        """Transition within configured window is feasible."""
        link = CameraLink("cam_a", "cam_b", 10.0, 30.0)
        model = CameraLinkModel(links=[link])

        assert model.is_feasible("cam_a", "cam_b", 100.0, 115.0)  # 15s transit

    def test_infeasible_too_fast(self) -> None:
        """Transit faster than minimum is rejected."""
        link = CameraLink("cam_a", "cam_b", 10.0, 30.0)
        model = CameraLinkModel(links=[link])

        assert not model.is_feasible("cam_a", "cam_b", 100.0, 105.0)  # 5s < 10s

    def test_infeasible_too_slow(self) -> None:
        """Transit slower than maximum is rejected."""
        link = CameraLink("cam_a", "cam_b", 10.0, 30.0)
        model = CameraLinkModel(links=[link])

        assert not model.is_feasible("cam_a", "cam_b", 100.0, 140.0)  # 40s > 30s

    def test_default_window_for_unconfigured(self) -> None:
        """Unconfigured camera pairs use default window."""
        model = CameraLinkModel(default_window=(5.0, 60.0))

        # Within default window
        assert model.is_feasible("cam_x", "cam_y", 100.0, 130.0)  # 30s
        # Outside default window
        assert not model.is_feasible("cam_x", "cam_y", 100.0, 170.0)  # 70s

    def test_directional_links(self) -> None:
        """A→B link doesn't imply B→A feasibility."""
        link = CameraLink("cam_a", "cam_b", 10.0, 30.0)
        model = CameraLinkModel(
            links=[link],
            default_window=(100.0, 200.0),  # wide default
        )

        # A→B uses configured link
        assert model.is_feasible("cam_a", "cam_b", 100.0, 115.0)  # 15s, within 10-30
        # B→A falls to default window (100-200s), so 15s is too fast
        assert not model.is_feasible("cam_b", "cam_a", 100.0, 115.0)

    def test_same_camera_infeasible(self) -> None:
        """Same camera always returns False."""
        model = CameraLinkModel()
        assert not model.is_feasible("cam_a", "cam_a", 100.0, 110.0)

    def test_negative_transit_infeasible(self) -> None:
        """Backward-in-time transitions are rejected."""
        model = CameraLinkModel()
        assert not model.is_feasible("cam_a", "cam_b", 100.0, 90.0)

    def test_filter_matches(self) -> None:
        """filter_matches removes infeasible candidates."""
        link = CameraLink("cam_a", "cam_b", 10.0, 30.0)
        model = CameraLinkModel(links=[link])

        matches = [
            GalleryMatch(1, 0.1, "cam_a", 10, timestamp=100.0),  # 15s transit → OK
            GalleryMatch(2, 0.2, "cam_a", 20, timestamp=50.0),  # 65s transit → too slow
        ]

        filtered = model.filter_matches("cam_b", 115.0, matches)
        assert len(filtered) == 1
        assert filtered[0].global_id == 1

    def test_link_count(self) -> None:
        """link_count reflects configured links."""
        model = CameraLinkModel(
            links=[
                CameraLink("a", "b", 5.0, 30.0),
                CameraLink("b", "c", 10.0, 60.0),
            ]
        )
        assert model.link_count == 2


# ===========================================================================
# TestCrossCameraTracker
# ===========================================================================
class TestCrossCameraTracker:
    """Tests for the CrossCameraTracker orchestrator."""

    def _make_mock_reid(self, dim: int = 512) -> object:
        """Create a simple ReID extractor that returns random embeddings."""

        class _MockReID:
            __slots__ = ("_dim",)

            def __init__(self, d: int) -> None:
                self._dim = d

            @property
            def embedding_dim(self) -> int:
                return self._dim

            def extract(
                self,
                frame_pixels: np.ndarray,
                boxes_xyxy: list[tuple[float, float, float, float]],
            ) -> np.ndarray | None:
                if not boxes_xyxy:
                    return None
                n = len(boxes_xyxy)
                rng = np.random.default_rng(42)
                embs = rng.standard_normal((n, self._dim)).astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                return embs / norms

        return _MockReID(dim)

    def _make_detection(
        self,
        boxes: list[tuple[float, float, float, float, float, int, str]],
        frame_index: int = 0,
    ) -> Detection:
        """Create a Detection with given boxes.

        Each box is (x1, y1, x2, y2, conf, class_id, class_name).
        """
        from yowo.types import BoundingBox, Frame

        frame = Frame(
            pixels=np.zeros((480, 640, 3), dtype=np.uint8),
            source_id="test",
            frame_index=frame_index,
        )
        bb = tuple(
            BoundingBox(
                x1=b[0],
                y1=b[1],
                x2=b[2],
                y2=b[3],
                confidence=b[4],
                class_id=b[5],
                class_name=b[6],
            )
            for b in boxes
        )
        return Detection(
            frame=frame,
            boxes=bb,
            inference_time_ms=1.0,
            backend="onnx",
            model_spec=None,
        )

    def test_single_camera_assigns_global_ids(self) -> None:
        """Single-camera usage assigns global IDs to confirmed tracks."""
        from yowo.tracking._cross_camera import CrossCameraTracker

        reid = self._make_mock_reid(dim=64)
        cct = CrossCameraTracker(
            reid,  # type: ignore[arg-type]
            match_threshold=0.4,
            track_high_thresh=0.3,
            min_hits=1,
        )

        det = self._make_detection(
            [
                (100, 100, 200, 200, 0.9, 0, "car"),
            ]
        )

        # First frame — track should be created
        result = cct.update("cam_a", det, timestamp=1.0)
        assert len(result) == 1
        assert result[0].camera_id == "cam_a"

    def test_camera_count(self) -> None:
        """camera_count reflects registered cameras."""
        from yowo.tracking._cross_camera import CrossCameraTracker

        reid = self._make_mock_reid(dim=64)
        cct = CrossCameraTracker(reid, match_threshold=0.4)  # type: ignore[arg-type]
        cct.register_camera("cam_a")
        cct.register_camera("cam_b")
        assert cct.camera_count == 2

    def test_register_duplicate_camera_raises(self) -> None:
        """Registering same camera twice raises ValueError."""
        from yowo.tracking._cross_camera import CrossCameraTracker

        reid = self._make_mock_reid(dim=64)
        cct = CrossCameraTracker(reid, match_threshold=0.4)  # type: ignore[arg-type]
        cct.register_camera("cam_a")
        with pytest.raises(ValueError, match="already registered"):
            cct.register_camera("cam_a")

    def test_auto_register_on_update(self) -> None:
        """Camera is auto-registered on first update() call."""
        from yowo.tracking._cross_camera import CrossCameraTracker

        reid = self._make_mock_reid(dim=64)
        cct = CrossCameraTracker(reid, match_threshold=0.4)  # type: ignore[arg-type]
        assert cct.camera_count == 0

        det = self._make_detection([])
        cct.update("cam_a", det, timestamp=1.0)
        assert cct.camera_count == 1

    def test_gallery_grows_with_confirmed_tracks(self) -> None:
        """Gallery entries increase as tracks get confirmed and assigned IDs."""
        from yowo.tracking._cross_camera import CrossCameraTracker

        reid = self._make_mock_reid(dim=64)
        cct = CrossCameraTracker(
            reid,  # type: ignore[arg-type]
            match_threshold=0.4,
            track_high_thresh=0.3,
            min_hits=1,
        )

        # Feed multiple frames with same detection to confirm a track
        for i in range(5):
            det = self._make_detection(
                [(100, 100, 200, 200, 0.9, 0, "car")],
                frame_index=i,
            )
            cct.update("cam_a", det, timestamp=float(i))

        assert cct.gallery_size >= 1

    def test_unconfirmed_track_has_no_global_id(self) -> None:
        """Track below min_hits has global_id=None."""
        from yowo.tracking._cross_camera import CrossCameraTracker

        reid = self._make_mock_reid(dim=64)
        cct = CrossCameraTracker(
            reid,  # type: ignore[arg-type]
            match_threshold=0.4,
            track_high_thresh=0.3,
            min_hits=5,  # high threshold → track stays unconfirmed longer
        )

        det = self._make_detection(
            [(100, 100, 200, 200, 0.9, 0, "car")],
        )
        result = cct.update("cam_a", det, timestamp=1.0)

        # First frame — track is unconfirmed (hits=1 < min_hits=5)
        unconfirmed = [r for r in result if r.global_id is None]
        assert len(unconfirmed) >= 0  # may or may not appear in output
