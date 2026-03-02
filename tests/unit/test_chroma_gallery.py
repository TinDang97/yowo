"""Tests for ChromaEmbeddingGallery — ChromaDB-backed persistent gallery."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from yowo.tracking._chroma_gallery import ChromaEmbeddingGallery  # noqa: E402
from yowo.tracking._gallery import EmbeddingGallery, GalleryProtocol  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_cross_camera.py)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gallery(tmp_path: Path) -> ChromaEmbeddingGallery:
    """Fresh gallery backed by a temp directory."""
    return ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "gallery")


@pytest.fixture
def small_gallery(tmp_path: Path) -> ChromaEmbeddingGallery:
    """Small 4-dim gallery for eviction tests."""
    return ChromaEmbeddingGallery(embedding_dim=4, max_entries=5, persist_path=tmp_path / "small")


# ===========================================================================
# TestChromaEmbeddingGallery — Mirrored from TestEmbeddingGallery
# ===========================================================================
class TestChromaEmbeddingGallery:
    """Tests for the ChromaDB-backed embedding gallery."""

    def test_add_and_query_nearest(self, gallery: ChromaEmbeddingGallery) -> None:
        """Gallery returns nearest match by cosine distance."""
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

    def test_exclude_camera(self, gallery: ChromaEmbeddingGallery) -> None:
        """Same-camera exclusion prevents within-camera matches."""
        emb = _rand_embedding()
        gallery.add("cam_a", 1, emb, timestamp=10.0)

        # Query from same camera — should be excluded
        matches = gallery.query(emb, exclude_camera="cam_a", threshold=0.5)
        assert len(matches) == 0

        # Query from different camera — should match
        matches = gallery.query(emb, exclude_camera="cam_b", threshold=0.5)
        assert len(matches) == 1

    def test_threshold_filters(self, gallery: ChromaEmbeddingGallery) -> None:
        """Matches beyond threshold distance are excluded."""
        emb = _rand_embedding()
        gallery.add("cam_a", 1, emb, timestamp=10.0)

        # Very tight threshold — orthogonal embedding won't match
        ortho = _orthogonal_embedding()
        matches = gallery.query(ortho, threshold=0.1)
        assert len(matches) == 0

    def test_max_entries_eviction(self, tmp_path: Path) -> None:
        """Gallery evicts oldest entries when exceeding max_entries."""
        gallery = ChromaEmbeddingGallery(
            embedding_dim=4, max_entries=5, persist_path=tmp_path / "evict"
        )

        for i in range(10):
            emb = np.array([float(i), 0.0, 0.0, 1.0], dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            gallery.add("cam_a", i, emb, timestamp=float(i))

        assert gallery.size == 5

    def test_empty_gallery_returns_empty(self, gallery: ChromaEmbeddingGallery) -> None:
        """Query on empty gallery returns empty list."""
        matches = gallery.query(_rand_embedding(), threshold=0.5)
        assert matches == []

    def test_global_id_assignment(self, small_gallery: ChromaEmbeddingGallery) -> None:
        """Each add() returns a unique global_id unless specified."""
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        gid1 = small_gallery.add("cam_a", 1, emb)
        gid2 = small_gallery.add("cam_b", 2, emb)
        assert gid1 != gid2

        # Explicit global_id reuse
        gid3 = small_gallery.add("cam_c", 3, emb, global_id=gid1)
        assert gid3 == gid1

    def test_thread_safety_concurrent_add_query(self, tmp_path: Path) -> None:
        """Concurrent add/query doesn't crash."""
        gallery = ChromaEmbeddingGallery(
            embedding_dim=64, max_entries=1000, persist_path=tmp_path / "threads"
        )
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
            threading.Thread(target=add_worker, args=("cam_a", 50)),
            threading.Thread(target=add_worker, args=("cam_b", 50)),
            threading.Thread(target=query_worker, args=(50,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert gallery.size > 0

    def test_next_global_id(self, gallery: ChromaEmbeddingGallery) -> None:
        """next_global_id allocates sequential IDs."""
        id1 = gallery.next_global_id()
        id2 = gallery.next_global_id()
        assert id2 == id1 + 1

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        """top_k parameter limits the number of returned matches."""
        gallery = ChromaEmbeddingGallery(embedding_dim=4, persist_path=tmp_path / "topk")
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        for i in range(10):
            gallery.add(f"cam_{i}", i, emb, timestamp=float(i))

        matches = gallery.query(emb, top_k=3, threshold=0.5)
        assert len(matches) <= 3

    def test_embedding_dim_property(self, gallery: ChromaEmbeddingGallery) -> None:
        """embedding_dim returns the configured dimensionality."""
        assert gallery.embedding_dim == 512


# ===========================================================================
# Persistence-specific tests
# ===========================================================================
class TestChromaGalleryPersistence:
    """Tests for persistence across gallery restarts."""

    def test_persistence_survives_restart(self, tmp_path: Path) -> None:
        """Entries added in one instance survive after destruction and reinit."""
        persist = tmp_path / "persist_test"
        emb = _rand_embedding(dim=64)

        g1 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        gid = g1.add("cam_a", 1, emb, timestamp=42.0)
        del g1

        g2 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        assert g2.size == 1
        matches = g2.query(emb, threshold=0.1)
        assert len(matches) == 1
        assert matches[0].global_id == gid
        assert matches[0].camera_id == "cam_a"

    def test_global_id_recovery_after_restart(self, tmp_path: Path) -> None:
        """next_global_id after restart does not collide with pre-restart IDs."""
        persist = tmp_path / "recovery_test"
        emb = _rand_embedding(dim=64)

        g1 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        gid1 = g1.add("cam_a", 1, emb)
        gid2 = g1.add("cam_b", 2, emb)
        max_before = max(gid1, gid2)
        del g1

        g2 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        new_gid = g2.next_global_id()
        assert new_gid > max_before

    def test_insertion_counter_recovery_after_restart(self, tmp_path: Path) -> None:
        """Insertion counter resumes correctly after restart (eviction preserved)."""
        persist = tmp_path / "counter_test"
        emb = _rand_embedding(dim=64)

        g1 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist, max_entries=3)
        for i in range(3):
            g1.add(f"cam_{i}", i, emb)
        del g1

        # After restart, add one more — should evict oldest (not fail)
        g2 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist, max_entries=3)
        assert g2.size == 3
        g2.add("cam_new", 99, emb)
        assert g2.size == 3  # evicted oldest, still bounded

    def test_eviction_removes_oldest(self, tmp_path: Path) -> None:
        """Eviction removes entries with lowest insertion_order."""
        persist = tmp_path / "eviction_test"
        emb_old = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_new = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        g = ChromaEmbeddingGallery(embedding_dim=4, persist_path=persist, max_entries=2)
        g.add("cam_a", 0, emb_old)  # insertion_order=0 — will be evicted
        g.add("cam_b", 1, emb_new)  # insertion_order=1

        # Add third entry — should evict insertion_order=0
        g.add("cam_c", 2, emb_new)
        assert g.size == 2

        # emb_old should no longer be retrievable (evicted)
        matches = g.query(emb_old, threshold=0.01)
        assert len(matches) == 0

    def test_split_brain_recovery(self, tmp_path: Path) -> None:
        """Counters recover correctly when sidecar is stale (crash simulation).

        Simulates: ChromaDB wrote entry but process crashed before sidecar
        update. On restart, max(sidecar, scan) yields correct counters.
        """
        persist = tmp_path / "splitbrain"
        emb = _rand_embedding(dim=64)

        g1 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        g1.add("cam_a", 1, emb)
        g1.add("cam_b", 2, emb)
        # Sidecar says next_global_id=3, insertion_counter=2

        # Simulate crash: manually write stale sidecar (as if crash happened
        # after ChromaDB wrote entry 2 but before sidecar was updated)
        state_path = persist / "_gallery_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "next_global_id": 2,  # stale — should be 3
                    "insertion_counter": 1,  # stale — should be 2
                }
            )
        )
        del g1

        # Recovery should use max(sidecar=2, scan=2+1=3) = 3 for global_id
        g2 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        new_gid = g2.next_global_id()
        assert new_gid >= 3  # no collision with existing IDs

        # Add should not crash with duplicate insertion_order
        g2.add("cam_c", 3, emb)
        assert g2.size == 3

    def test_corrupted_sidecar_recovery(self, tmp_path: Path) -> None:
        """Gallery recovers from corrupted sidecar file via collection scan."""
        persist = tmp_path / "corrupt"
        emb = _rand_embedding(dim=64)

        g1 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        gid = g1.add("cam_a", 1, emb)
        del g1

        # Corrupt the sidecar
        state_path = persist / "_gallery_state.json"
        state_path.write_text("NOT VALID JSON{{{")

        # Should recover via collection scan
        g2 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        new_gid = g2.next_global_id()
        assert new_gid > gid


# ===========================================================================
# Protocol conformance test
# ===========================================================================
class TestGalleryProtocol:
    """Verify both gallery implementations satisfy GalleryProtocol."""

    def test_in_memory_gallery_satisfies_protocol(self) -> None:
        """EmbeddingGallery satisfies GalleryProtocol."""
        gallery = EmbeddingGallery(embedding_dim=64)
        assert isinstance(gallery, GalleryProtocol)

    def test_chroma_gallery_satisfies_protocol(self, tmp_path: Path) -> None:
        """ChromaEmbeddingGallery satisfies GalleryProtocol."""
        gallery = ChromaEmbeddingGallery(embedding_dim=64, persist_path=tmp_path / "proto")
        assert isinstance(gallery, GalleryProtocol)
