"""ChromaDB-backed persistent EmbeddingGallery for cross-camera ReID.

Requires: ``uv add yowo[chromadb]``

Drop-in replacement for :class:`EmbeddingGallery`. Embeddings survive
process restarts; global IDs and insertion counters are recovered
automatically from the persisted collection on initialization.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.errors import DependencyError
from yowo.tracking._gallery import GalleryMatch

__all__ = ["ChromaEmbeddingGallery"]


def _make_noop_embedding_fn() -> Any:
    """Build a registered no-op embedding function (pre-computed embeddings)."""
    import chromadb  # already imported by caller
    from chromadb.utils.embedding_functions import register_embedding_function

    @register_embedding_function
    class _NoOp(chromadb.EmbeddingFunction):  # type: ignore[type-arg]
        def __init__(self) -> None:
            pass  # skip base class deprecation warning

        def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]
            msg = "Pre-computed embeddings must be supplied"
            raise NotImplementedError(msg)

        @staticmethod
        def name() -> str:
            return "noop_precomputed"

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> Any:
            return _NoOp()

        def get_config(self) -> dict[str, Any]:
            return {}

    return _NoOp()


class ChromaEmbeddingGallery:
    """Persistent embedding gallery backed by ChromaDB.

    Uses ``chromadb.PersistentClient`` with cosine distance space.
    Cosine distance = ``1 - dot(query, gallery)`` for L2-normalized
    vectors — exactly matching the in-memory gallery formula.

    Performance characteristics:
    - ``add()``: O(1) amortized — no disk I/O for sidecar (collection scan
      handles crash recovery). Eviction is O(1) via monotonic counter tracking.
    - ``query()``: O(log N) via HNSW index. No lock contention with add().
    - ``next_global_id()``: O(1) with sidecar persist (for IDs not stored
      in ChromaDB).

    Thread-safe: ChromaDB ``PersistentClient`` is thread-safe within a
    single process. A write lock guards counter increments + chroma
    writes to ensure atomicity.

    Args:
        embedding_dim: Dimensionality of stored embeddings.
        max_entries: Maximum gallery size. Oldest entries evicted on overflow.
        persist_path: Directory for ChromaDB on-disk storage.
        collection_name: Name of the ChromaDB collection.
    """

    __slots__ = (
        "_client",
        "_collection",
        "_embedding_dim",
        "_insertion_counter",
        "_lock",
        "_max_entries",
        "_next_global_id",
        "_oldest_counter",
        "_size",
        "_state_path",
    )

    def __init__(
        self,
        embedding_dim: int,
        max_entries: int = 10_000,
        *,
        persist_path: str | Path = ".yowo_gallery",
        collection_name: str = "embeddings",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise DependencyError("chromadb", "uv add yowo[chromadb]") from exc

        self._embedding_dim = embedding_dim
        self._max_entries = max_entries
        self._lock = threading.Lock()

        persist = Path(persist_path)
        self._state_path = persist / "_gallery_state.json"
        self._client: Any = chromadb.PersistentClient(path=str(persist))
        self._collection: Any = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=_make_noop_embedding_fn(),
        )

        # Recover monotonic counters — max(sidecar, collection scan)
        # guards against split-brain if crash occurs between ChromaDB
        # write and sidecar write.
        sidecar_gid, sidecar_counter = 1, 0
        if self._state_path.exists():
            try:
                state = json.loads(self._state_path.read_text())
                sidecar_gid = int(state["next_global_id"])
                sidecar_counter = int(state["insertion_counter"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # corrupted sidecar — rely on collection scan

        self._size = self._collection.count()
        scan_gid, scan_counter, oldest = 1, 0, 0
        if self._size > 0:
            metadatas: list[dict[str, Any]] = self._collection.get(
                include=["metadatas"],
            )["metadatas"]
            scan_gid = max(int(m["global_id"]) for m in metadatas) + 1
            orders = [int(m["insertion_order"]) for m in metadatas]
            scan_counter = max(orders) + 1
            oldest = min(orders)

        self._next_global_id = max(sidecar_gid, scan_gid)
        self._insertion_counter = max(sidecar_counter, scan_counter)
        self._oldest_counter = oldest if self._size > 0 else self._insertion_counter

    def _persist_state(self) -> None:
        """Write counter state to sidecar file atomically. Must be called with lock held."""
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "next_global_id": self._next_global_id,
                    "insertion_counter": self._insertion_counter,
                }
            )
        )
        tmp.replace(self._state_path)

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
                The counter advances past this value to prevent future collisions.

        Returns:
            The global_id assigned to this entry.
        """
        with self._lock:
            if global_id is None:
                global_id = self._next_global_id
                self._next_global_id += 1
            else:
                # Advance counter past explicit ID to prevent future collisions
                self._next_global_id = max(self._next_global_id, global_id + 1)

            order = self._insertion_counter
            self._insertion_counter += 1

            self._collection.add(
                ids=[f"entry_{order}"],
                embeddings=[embedding.tolist()],
                metadatas=[
                    {
                        "global_id": int(global_id),
                        "camera_id": camera_id,
                        "local_track_id": int(local_track_id),
                        "class_id": int(class_id),
                        "timestamp": float(timestamp),
                        "insertion_order": int(order),
                    }
                ],
            )
            self._size += 1

            if self._size > self._max_entries:
                self._evict_oldest()

            # No _persist_state() here — collection scan recovers counters
            # on restart via max(sidecar, scan). Only next_global_id() needs
            # sidecar persist (for IDs allocated but never stored in ChromaDB).
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
            size = self._size

        if size == 0:
            return []

        where: dict[str, Any] | None = (
            {"camera_id": {"$ne": exclude_camera}} if exclude_camera is not None else None
        )

        result = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, size),
            where=where,
            include=["metadatas", "distances"],
        )

        if not result["distances"] or not result["distances"][0]:
            return []

        distances: list[float] = result["distances"][0]
        metadatas: list[dict[str, Any]] = result["metadatas"][0]

        return [
            GalleryMatch(
                global_id=int(meta["global_id"]),
                distance=float(dist),
                camera_id=str(meta["camera_id"]),
                local_track_id=int(meta["local_track_id"]),
                timestamp=float(meta["timestamp"]),
            )
            for dist, meta in zip(distances, metadatas, strict=True)
            if dist <= threshold
        ]

    def _evict_oldest(self) -> None:
        """O(1) eviction using monotonic insertion counter.

        IDs follow ``entry_{counter}`` format with contiguous counters,
        so the oldest entries can be deleted by ID without scanning.
        Must be called with ``_lock`` held.
        """
        to_evict = self._size - self._max_entries
        if to_evict <= 0:
            return
        ids = [f"entry_{self._oldest_counter + i}" for i in range(to_evict)]
        self._collection.delete(ids=ids)
        self._oldest_counter += to_evict
        self._size -= to_evict

    def next_global_id(self) -> int:
        """Allocate and return a new global ID."""
        with self._lock:
            gid = self._next_global_id
            self._next_global_id += 1
            # Must persist — these IDs are NOT stored in ChromaDB, so
            # collection scan alone cannot recover them after crash.
            self._persist_state()
            return gid

    @property
    def size(self) -> int:
        """Current number of entries in the gallery."""
        with self._lock:
            return self._size

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of stored embeddings."""
        return self._embedding_dim
