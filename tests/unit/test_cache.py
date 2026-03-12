"""Unit tests for yowo.cache — feature map caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yowo.cache._similarity import frame_similarity
from yowo.cache._store import FeatureStore


def _make_features(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 3 small feature maps with deterministic values."""
    rng = np.random.RandomState(seed)
    return (
        rng.randn(1, 64, 80, 80).astype(np.float32),
        rng.randn(1, 128, 40, 40).astype(np.float32),
        rng.randn(1, 256, 20, 20).astype(np.float32),
    )


# ─── frame_similarity ───────────────────────────────────────────────


class TestFrameSimilarity:
    def test_identical_frames_zero(self) -> None:
        a = np.ones((1, 3, 4, 4), dtype=np.float32) * 0.5
        assert frame_similarity(a, a) == pytest.approx(0.0)

    def test_different_frames_positive(self) -> None:
        a = np.zeros((1, 3, 4, 4), dtype=np.float32)
        b = np.ones((1, 3, 4, 4), dtype=np.float32)
        assert frame_similarity(a, b) == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        a = np.zeros((1, 3, 2, 2), dtype=np.float32)
        b = np.ones((1, 3, 2, 2), dtype=np.float32) * 0.1
        result = frame_similarity(a, b)
        assert isinstance(result, float)
        assert 0.0 < result < 1.0


# ─── FeatureStore (in-memory) ────────────────────────────────────────


class TestFeatureStoreInMemory:
    def test_store_and_load(self) -> None:
        store = FeatureStore()
        features = _make_features(seed=42)
        store.store("cam1", features)

        loaded = store.load("cam1")
        assert loaded is not None
        for orig, cached in zip(features, loaded):
            np.testing.assert_array_equal(orig, cached)

    def test_load_missing_key_returns_none(self) -> None:
        store = FeatureStore()
        assert store.load("nonexistent") is None

    def test_size_tracks_entries(self) -> None:
        store = FeatureStore()
        assert store.size == 0

        store.store("a", _make_features(0))
        assert store.size == 1

        store.store("b", _make_features(1))
        assert store.size == 2

    def test_fifo_eviction_at_capacity(self) -> None:
        store = FeatureStore(max_entries=2)
        store.store("a", _make_features(0))
        store.store("b", _make_features(1))
        store.store("c", _make_features(2))  # should evict "a"

        assert store.size == 2
        assert store.load("a") is None  # evicted
        assert store.load("b") is not None
        assert store.load("c") is not None

    def test_clear_removes_all(self) -> None:
        store = FeatureStore()
        store.store("a", _make_features(0))
        store.store("b", _make_features(1))
        store.clear()

        assert store.size == 0
        assert store.load("a") is None


# ─── FeatureStore (mmap) ─────────────────────────────────────────────


class TestFeatureStoreMmap:
    def test_store_and_load(self, tmp_path: Path) -> None:
        store = FeatureStore(cache_dir=tmp_path / "cache")
        features = _make_features(seed=42)
        store.store("cam1", features)

        loaded = store.load("cam1")
        assert loaded is not None
        for orig, cached in zip(features, loaded):
            np.testing.assert_array_equal(orig, cached)

    def test_fifo_eviction_at_capacity(self, tmp_path: Path) -> None:
        store = FeatureStore(max_entries=2, cache_dir=tmp_path / "cache")
        store.store("a", _make_features(0))
        store.store("b", _make_features(1))
        store.store("c", _make_features(2))

        assert store.size == 2
        assert store.load("a") is None
        assert store.load("c") is not None

    def test_clears_stale_entries_on_startup(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        store1 = FeatureStore(cache_dir=cache_dir)
        store1.store("old", _make_features(0))
        assert store1.size == 1

        store2 = FeatureStore(cache_dir=cache_dir)
        assert store2.size == 0

    def test_hash_key_no_collision(self, tmp_path: Path) -> None:
        """Keys with similar characters don't collide after hashing."""
        store = FeatureStore(cache_dir=tmp_path / "cache")
        store.store("cam/1", _make_features(0))
        store.store("cam_1", _make_features(1))
        assert store.size == 2

    def test_special_chars_in_key(self, tmp_path: Path) -> None:
        store = FeatureStore(cache_dir=tmp_path / "cache")
        store.store("rtsp://cam:554/live", _make_features(0))
        assert store.size == 1
        assert store.load("rtsp://cam:554/live") is not None


# ─── FeatureCache coordinator ────────────────────────────────────────


class TestFeatureCache:
    def test_miss_on_first_frame(self) -> None:
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)
        result = cache.check_and_load("cam1", tensor, "cpu")
        assert result is None

    def test_miss_after_update_with_different_frame(self) -> None:
        from yowo.cache import FeatureCache

        cache = FeatureCache(similarity_threshold=0.01)
        t1 = np.zeros((1, 3, 4, 4), dtype=np.float32)
        t2 = np.ones((1, 3, 4, 4), dtype=np.float32)

        cache.update("cam1", t1, _make_features(0))
        result = cache.check_and_load("cam1", t2, "cpu")
        assert result is None

    def test_hit_after_update_with_similar_frame(self) -> None:
        import torch

        from yowo.cache import FeatureCache

        cache = FeatureCache(similarity_threshold=0.1)
        t1 = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        t2 = np.full((1, 3, 4, 4), 0.505, dtype=np.float32)

        features = _make_features(42)
        cache.update("cam1", t1, features)
        result = cache.check_and_load("cam1", t2, "cpu")

        assert result is not None
        assert len(result) == 3
        for tensor in result:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == "cpu"

    def test_clear(self) -> None:
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        t = np.zeros((1, 3, 4, 4), dtype=np.float32)
        cache.update("cam1", t, _make_features(0))
        assert cache.size == 1

        cache.clear()
        assert cache.size == 0

    def test_multiple_sources(self) -> None:
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        for i in range(3):
            t = np.full((1, 3, 4, 4), float(i) * 0.1, dtype=np.float32)
            cache.update(f"cam{i}", t, _make_features(i))

        assert cache.size == 3

    def test_last_fingerprints_bounded(self) -> None:
        """_last_fingerprints never exceeds max_entries."""
        from yowo.cache import FeatureCache

        cache = FeatureCache(max_entries=3)
        for i in range(10):
            t = np.full((1, 3, 4, 4), float(i) * 0.01, dtype=np.float32)
            cache.update(f"src{i}", t, _make_features(i))

        # Both store and _last_fingerprints should be capped at 3
        assert cache.size == 3
        assert len(cache._last_fingerprints) == 3

    def test_mmap_mode_via_cache_dir(self, tmp_path: Path) -> None:
        """FeatureCache works with mmap when cache_dir is provided."""
        from yowo.cache import FeatureCache

        cache = FeatureCache(cache_dir=tmp_path / "cache")
        t1 = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        cache.update("cam1", t1, _make_features(0))
        assert cache.size == 1

        # Verify files exist on disk
        dirs = list((tmp_path / "cache").iterdir())
        assert len(dirs) == 1
