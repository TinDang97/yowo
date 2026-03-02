"""Shared test helpers for embedding-based tests."""

from __future__ import annotations

import numpy as np


def rand_embedding(dim: int = 512) -> np.ndarray:
    """Random L2-normalized embedding."""
    v = np.random.default_rng(42).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def similar_embedding(base: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Create an embedding similar to base with small noise."""
    rng = np.random.default_rng(123)
    noisy = base + rng.standard_normal(base.shape).astype(np.float32) * noise
    return (noisy / np.linalg.norm(noisy)).astype(np.float32)


def orthogonal_embedding(dim: int = 512) -> np.ndarray:
    """Create an embedding very different from typical random ones."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    return v
