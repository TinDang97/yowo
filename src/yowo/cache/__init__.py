"""Feature map caching for sequential inference.

Provides transparent caching of YOLO neck feature maps across sequential
frames (e.g., video streams, multi-camera feeds). When consecutive frames
are similar (below a configurable threshold), the backbone and neck are
skipped entirely — only the lightweight detection head runs.

In-memory by default. When ``cache_dir`` is provided, features are
persisted via numpy mmap so the OS manages memory pressure transparently.

Usage::

    from yowo.cache import FeatureCache

    cache = FeatureCache()  # in-memory (default)

    # In inference loop:
    cached = cache.check_and_load(source_id, preprocessed_tensor, device)
    if cached is not None:
        output = model.forward_head(cached)  # skip backbone + neck
    else:
        output = model(preprocessed_tensor)
        neck_features = ...  # captured via hook
        cache.update(source_id, preprocessed_tensor, neck_features)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from yowo.cache._store import FeatureStore

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

__all__ = ["FeatureCache"]


class FeatureCache:
    """Coordinator for feature map caching.

    Combines frame similarity checking with bounded feature storage.
    Designed for sequential inference scenarios (video streams, multi-camera).

    Args:
        similarity_threshold: Maximum L1 mean pixel difference to consider
            frames "similar enough" for cache reuse. Lower = stricter.
            Default 0.01 works well for surveillance/static-background video.
        max_entries: Maximum cached sources before FIFO eviction.
        cache_dir: Optional directory for mmap persistence. When None
            (default), features are cached in-memory only.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.01,
        max_entries: int = 32,
        cache_dir: Path | None = None,
    ) -> None:
        self._store = FeatureStore(max_entries=max_entries, cache_dir=cache_dir)
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._last_fingerprints: dict[str, NDArray[np.float32]] = {}
        self._lock = threading.Lock()
        self._pending_fingerprint: NDArray[np.float32] | None = None

    def check_and_load(
        self,
        source_id: str,
        current_tensor: NDArray[np.float32],
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Check if cached features can be reused for the current frame.

        Compares the current preprocessed tensor to the last cached tensor
        for this source. If similar enough, loads cached neck features
        and returns them as PyTorch tensors on the target device.

        Args:
            source_id: Identifies the input source (camera, file, etc.).
            current_tensor: Current preprocessed BCHW float32 numpy array.
            device: Target device for the returned tensors.

        Returns:
            Tuple of 3 tensors (P3', P4'', P5'') if cache hit, else None.
        """
        import torch

        with self._lock:
            # Compute fingerprint once — reused by update() if cache miss
            current_fp = current_tensor.mean(axis=(2, 3))  # (B, C)
            self._pending_fingerprint = current_fp

            last_fp = self._last_fingerprints.get(source_id)
            if last_fp is None:
                return None

            diff = float(np.abs(current_fp - last_fp).mean())
            if diff >= self._threshold:
                return None

            cached = self._store.load(source_id)
            if cached is None:
                return None

        logger.debug(
            "Cache hit for '%s' (diff=%.4f < threshold=%.4f)",
            source_id,
            diff,
            self._threshold,
        )
        return (
            torch.from_numpy(cached[0]).to(device, non_blocking=True),
            torch.from_numpy(cached[1]).to(device, non_blocking=True),
            torch.from_numpy(cached[2]).to(device, non_blocking=True),
        )

    def update(
        self,
        source_id: str,
        input_tensor: NDArray[np.float32],
        neck_features: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    ) -> None:
        """Store input fingerprint and neck features for future cache checks.

        Args:
            source_id: Identifies the input source.
            input_tensor: The preprocessed BCHW tensor (spatial-mean fingerprint
                stored for similarity comparison on next frame).
            neck_features: Tuple of 3 numpy arrays (P3', P4'', P5'') to cache.
        """
        with self._lock:
            # Cap fingerprints to max_entries (prevent unbounded growth)
            if source_id not in self._last_fingerprints:
                while len(self._last_fingerprints) >= self._max_entries:
                    oldest = next(iter(self._last_fingerprints))
                    del self._last_fingerprints[oldest]
            # Reuse fingerprint from check_and_load() if available
            fp = self._pending_fingerprint
            self._pending_fingerprint = None
            if fp is None:
                fp = input_tensor.mean(axis=(2, 3))
            self._last_fingerprints[source_id] = fp
            self._store.store(source_id, neck_features)

    def clear(self) -> None:
        """Remove all cached data."""
        with self._lock:
            self._store.clear()
            self._last_fingerprints.clear()
            self._pending_fingerprint = None

    @property
    def size(self) -> int:
        """Number of cached sources."""
        return self._store.size
