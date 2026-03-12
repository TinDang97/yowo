"""Bounded feature map cache with optional mmap persistence."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_FEATURE_NAMES = ("p3", "p4", "p5")

_Features = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]


@dataclass()
class _MmapMeta:
    """Metadata for a mmap-backed cache entry."""

    path: Path
    shapes: dict[str, tuple[int, ...]]
    dtype: str


class FeatureStore:
    """Bounded feature map cache.

    In-memory by default (stores numpy arrays in a dict).
    When ``cache_dir`` is provided, features are persisted via numpy mmap
    and the OS manages memory pressure transparently (page cache).

    FIFO eviction: when at capacity, the oldest entry is evicted.

    Args:
        max_entries: Maximum cached entries. FIFO eviction when exceeded.
        cache_dir: Optional directory for mmap persistence. When None,
            features are stored in-memory only.
    """

    def __init__(
        self,
        max_entries: int = 32,
        cache_dir: Path | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._cache_dir = cache_dir
        self._mem: dict[str, _Features] = {}
        self._mmap: dict[str, _MmapMeta] = {}
        self._lock = threading.Lock()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Only clean up directories matching the expected hash pattern
            for d in cache_dir.iterdir():
                if d.is_dir() and _is_hash_dir(d.name):
                    shutil.rmtree(d, ignore_errors=True)

    def store(self, key: str, features: _Features) -> None:
        """Store 3 feature maps (P3', P4'', P5'').

        Args:
            key: Cache key (typically ``source_id``).
            features: Tuple of 3 numpy arrays (P3', P4'', P5'').
        """
        with self._lock:
            entries = self._active_entries()
            while len(entries) >= self._max_entries:
                self._remove_entry(next(iter(entries)))

            if self._cache_dir is not None:
                self._store_mmap(key, features)
            else:
                self._mem[key] = features

    def load(self, key: str) -> _Features | None:
        """Load cached feature maps.

        Returns None if key is not cached.
        """
        with self._lock:
            if self._cache_dir is not None:
                return self._load_mmap(key)
            return self._mem.get(key)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            if self._cache_dir is not None:
                for key in list(self._mmap):
                    self._remove_entry(key)
            else:
                self._mem.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._active_entries())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _active_entries(self) -> dict[str, _Features] | dict[str, _MmapMeta]:
        return self._mmap if self._cache_dir is not None else self._mem

    def _remove_entry(self, key: str) -> None:
        if self._cache_dir is not None:
            meta = self._mmap.pop(key, None)
            if meta is not None and meta.path.exists():
                shutil.rmtree(meta.path, ignore_errors=True)
        else:
            self._mem.pop(key, None)

    def _store_mmap(self, key: str, features: _Features) -> None:
        assert self._cache_dir is not None
        entry_dir = self._cache_dir / _hash_key(key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        shapes: dict[str, tuple[int, ...]] = {}
        dtype_str = ""
        try:
            for name, arr in zip(_FEATURE_NAMES, features, strict=True):
                mm = np.memmap(
                    entry_dir / f"{name}.dat",
                    dtype=arr.dtype,
                    mode="w+",
                    shape=arr.shape,
                )
                mm[:] = arr
                del mm
                shapes[name] = arr.shape
                dtype_str = str(arr.dtype)
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            raise
        self._mmap[key] = _MmapMeta(path=entry_dir, shapes=shapes, dtype=dtype_str)

    def _load_mmap(self, key: str) -> _Features | None:
        meta = self._mmap.get(key)
        if meta is None:
            return None
        try:
            arrays: list[NDArray[np.float32]] = []
            dt = np.dtype(meta.dtype)
            for name in _FEATURE_NAMES:
                mm = np.memmap(
                    meta.path / f"{name}.dat",
                    dtype=dt,
                    mode="r",
                    shape=meta.shapes[name],
                )
                arrays.append(np.array(mm))
                del mm
            return (arrays[0], arrays[1], arrays[2])
        except Exception:
            logger.debug("Failed to load mmap entry '%s', removing", key)
            self._remove_entry(key)
            return None


_HASH_DIR_RE = re.compile(r"^[0-9a-f]{16}$")


def _is_hash_dir(name: str) -> bool:
    """Return True if ``name`` matches the 16-char hex hash pattern."""
    return _HASH_DIR_RE.match(name) is not None


def _hash_key(key: str) -> str:
    """Hash key for collision-free filesystem directory names."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]
