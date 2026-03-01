"""Embedding gallery for cross-camera vehicle re-identification."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["EmbeddingGallery", "GalleryEntry", "GalleryMatch"]


@dataclass(slots=True)
class GalleryEntry:
    """A single entry in the embedding gallery."""

    global_id: int
    camera_id: str
    local_track_id: int
    embedding: NDArray[np.float32]
    class_id: int = 0
    timestamp: float = 0.0
    insertion_order: int = 0


@dataclass(frozen=True, slots=True)
class GalleryMatch:
    """Result of a gallery query — a candidate cross-camera match."""

    global_id: int
    distance: float
    camera_id: str
    local_track_id: int
    timestamp: float = 0.0


class EmbeddingGallery:
    """Bounded gallery of track embeddings for cross-camera ReID.

    Stores finalized track embeddings (from tracks that left the scene or
    expired) indexed by (camera_id, local_track_id). Provides efficient
    nearest-neighbor lookup for cross-camera identity matching.

    Thread-safe: multiple cameras can add/query concurrently.
    """

    __slots__ = (
        "_embedding_dim",
        "_entries",
        "_insertion_counter",
        "_lock",
        "_max_entries",
        "_next_global_id",
    )

    def __init__(
        self,
        embedding_dim: int,
        max_entries: int = 10_000,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._max_entries = max_entries
        self._entries: list[GalleryEntry] = []
        self._next_global_id = 1
        self._insertion_counter = 0
        self._lock = threading.Lock()

    def add(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: NDArray[np.float32],
        *,
        class_id: int = 0,
        timestamp: float = 0.0,
        global_id: int | None = None,
    ) -> int:
        """Add finalized track embedding to the gallery.

        Args:
            camera_id: Source camera identifier.
            local_track_id: Per-camera track ID.
            embedding: L2-normalized (D,) float32 embedding.
            class_id: Object class index.
            timestamp: Time when track was finalized.
            global_id: If provided, reuse this global ID. Otherwise assign new.

        Returns:
            The global_id assigned to this entry.
        """
        with self._lock:
            if global_id is None:
                global_id = self._next_global_id
                self._next_global_id += 1

            entry = GalleryEntry(
                global_id=global_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
                embedding=embedding.copy(),
                class_id=class_id,
                timestamp=timestamp,
                insertion_order=self._insertion_counter,
            )
            self._insertion_counter += 1
            self._entries.append(entry)

            if len(self._entries) > self._max_entries:
                self._evict_oldest(self._max_entries)

            return global_id

    def query(
        self,
        embedding: NDArray[np.float32],
        *,
        exclude_camera: str | None = None,
        top_k: int = 5,
        threshold: float = 0.4,
    ) -> list[GalleryMatch]:
        """Find nearest gallery entries by cosine distance.

        Args:
            embedding: L2-normalized (D,) query embedding.
            exclude_camera: Skip entries from this camera (same-camera exclusion).
            top_k: Maximum number of matches to return.
            threshold: Maximum cosine distance (1 - dot) for a match.

        Returns:
            List of GalleryMatch sorted by distance (ascending).
        """
        with self._lock:
            if not self._entries:
                return []

            # Filter candidates
            candidates = self._entries
            if exclude_camera is not None:
                candidates = [e for e in candidates if e.camera_id != exclude_camera]

            if not candidates:
                return []

            # Batch cosine distance: 1 - dot(query, gallery)
            gallery_matrix = np.stack([e.embedding for e in candidates], axis=0)  # (M, D)
            dots = gallery_matrix @ embedding  # (M,)
            distances = 1.0 - dots

            # Filter by threshold and sort
            matches: list[GalleryMatch] = []
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices[:top_k]:
                dist = float(distances[idx])
                if dist > threshold:
                    break
                entry = candidates[int(idx)]
                matches.append(
                    GalleryMatch(
                        global_id=entry.global_id,
                        distance=dist,
                        camera_id=entry.camera_id,
                        local_track_id=entry.local_track_id,
                        timestamp=entry.timestamp,
                    )
                )

            return matches

    def _evict_oldest(self, keep: int) -> int:
        """Remove oldest entries to maintain bounded size.

        Must be called with lock held.

        Args:
            keep: Number of entries to retain.

        Returns:
            Number of entries evicted.
        """
        if len(self._entries) <= keep:
            return 0
        evicted = len(self._entries) - keep
        self._entries = self._entries[evicted:]
        return evicted

    def next_global_id(self) -> int:
        """Allocate and return a new global ID."""
        with self._lock:
            gid = self._next_global_id
            self._next_global_id += 1
            return gid

    @property
    def size(self) -> int:
        """Current number of entries in the gallery."""
        with self._lock:
            return len(self._entries)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of stored embeddings."""
        return self._embedding_dim
