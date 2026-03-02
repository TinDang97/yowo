"""Unit tests for ChromaEmbeddingGallery — ChromaDB-backed gallery."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from tests.utils import orthogonal_embedding, rand_embedding, similar_embedding  # noqa: E402
from yowo.tracking._chroma_gallery import ChromaEmbeddingGallery  # noqa: E402


# ===========================================================================
# TestChromaEmbeddingGallery — Mirrored from TestEmbeddingGallery
# ===========================================================================
class TestChromaEmbeddingGallery:
    """Tests for the ChromaDB-backed embedding gallery."""

    def test_add_and_query_nearest(self, tmp_path: Path) -> None:
        """Gallery returns nearest match by cosine distance."""
        gallery = ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "g")
        emb1 = rand_embedding()
        emb2 = orthogonal_embedding()

        gallery.add("cam_a", 1, emb1, timestamp=10.0)
        gallery.add("cam_b", 2, emb2, timestamp=20.0)

        query = similar_embedding(emb1)
        matches = gallery.query(query, threshold=0.5)
        assert len(matches) >= 1
        assert matches[0].camera_id == "cam_a"
        assert matches[0].local_track_id == 1

    def test_exclude_camera(self, tmp_path: Path) -> None:
        """Same-camera exclusion prevents within-camera matches."""
        gallery = ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "g")
        emb = rand_embedding()
        gallery.add("cam_a", 1, emb, timestamp=10.0)

        matches = gallery.query(emb, exclude_camera="cam_a", threshold=0.5)
        assert len(matches) == 0

        matches = gallery.query(emb, exclude_camera="cam_b", threshold=0.5)
        assert len(matches) == 1

    def test_threshold_filters(self, tmp_path: Path) -> None:
        """Matches beyond threshold distance are excluded."""
        gallery = ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "g")
        emb = rand_embedding()
        gallery.add("cam_a", 1, emb, timestamp=10.0)

        ortho = orthogonal_embedding()
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

    def test_empty_gallery_returns_empty(self, tmp_path: Path) -> None:
        """Query on empty gallery returns empty list."""
        gallery = ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "g")
        matches = gallery.query(rand_embedding(), threshold=0.5)
        assert matches == []

    def test_global_id_assignment(self, tmp_path: Path) -> None:
        """Each add() returns a unique global_id unless specified."""
        gallery = ChromaEmbeddingGallery(
            embedding_dim=4, max_entries=5, persist_path=tmp_path / "g"
        )
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        gid1 = gallery.add("cam_a", 1, emb)
        gid2 = gallery.add("cam_b", 2, emb)
        assert gid1 != gid2

        gid3 = gallery.add("cam_c", 3, emb, global_id=gid1)
        assert gid3 == gid1

    def test_explicit_global_id_advances_counter(self, tmp_path: Path) -> None:
        """add(global_id=N) advances counter past N to prevent collisions."""
        gallery = ChromaEmbeddingGallery(
            embedding_dim=4, max_entries=100, persist_path=tmp_path / "g"
        )
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        gallery.add("cam_a", 1, emb, global_id=100)
        # Next auto-assigned ID must be > 100
        gid = gallery.add("cam_b", 2, emb)
        assert gid > 100

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

    def test_next_global_id(self, tmp_path: Path) -> None:
        """next_global_id allocates sequential IDs."""
        gallery = ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "g")
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

    def test_embedding_dim_property(self, tmp_path: Path) -> None:
        """embedding_dim returns the configured dimensionality."""
        gallery = ChromaEmbeddingGallery(embedding_dim=512, persist_path=tmp_path / "g")
        assert gallery.embedding_dim == 512
