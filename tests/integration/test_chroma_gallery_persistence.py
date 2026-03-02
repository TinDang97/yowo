"""Integration tests for ChromaEmbeddingGallery persistence + protocol conformance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from tests.utils import rand_embedding  # noqa: E402
from yowo.tracking._chroma_gallery import ChromaEmbeddingGallery  # noqa: E402
from yowo.tracking._gallery import EmbeddingGallery, GalleryProtocol  # noqa: E402


# ===========================================================================
# Persistence tests — destroy + recreate gallery from disk
# ===========================================================================
class TestChromaGalleryPersistence:
    """Tests for persistence across gallery restarts."""

    def test_persistence_survives_restart(self, tmp_path: Path) -> None:
        """Entries added in one instance survive after destruction and reinit."""
        persist = tmp_path / "persist_test"
        emb = rand_embedding(dim=64)

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
        emb = rand_embedding(dim=64)

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
        emb = rand_embedding(dim=64)

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
        emb = rand_embedding(dim=64)

        g1 = ChromaEmbeddingGallery(embedding_dim=64, persist_path=persist)
        g1.add("cam_a", 1, emb)
        g1.add("cam_b", 2, emb)

        # Simulate crash: manually write stale sidecar
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
        emb = rand_embedding(dim=64)

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
# Protocol conformance — both gallery implementations satisfy GalleryProtocol
# ===========================================================================
class TestGalleryProtocol:
    """Verify both gallery implementations satisfy GalleryProtocol."""

    def test_in_memory_gallery_satisfies_protocol(self) -> None:
        """EmbeddingGallery is a structural subtype of GalleryProtocol."""
        gallery = EmbeddingGallery(embedding_dim=64)
        assert isinstance(gallery, GalleryProtocol)

    def test_chroma_gallery_satisfies_protocol(self, tmp_path: Path) -> None:
        """ChromaEmbeddingGallery is a structural subtype of GalleryProtocol."""
        gallery = ChromaEmbeddingGallery(embedding_dim=64, persist_path=tmp_path / "proto")
        assert isinstance(gallery, GalleryProtocol)
