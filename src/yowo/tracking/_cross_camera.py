"""Cross-camera tracker for multi-camera multi-target vehicle tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.tracking._camera_link import CameraLinkModel
from yowo.tracking._gallery import EmbeddingGallery
from yowo.tracking._reid import ReIDExtractor
from yowo.tracking._strack import TrackedBox
from yowo.tracking._tracker import ByteTracker
from yowo.types import Detection

__all__ = ["CrossCameraTracker", "GlobalTrackedBox"]


@dataclass(frozen=True, slots=True)
class GlobalTrackedBox:
    """Extends TrackedBox with global cross-camera identity.

    Attributes:
        box: Original single-camera tracked box.
        camera_id: Source camera identifier.
        global_id: Cross-camera global identity. None if not yet matched.
        local_track_id: Per-camera track ID from ByteTracker.
    """

    box: TrackedBox
    camera_id: str
    global_id: int | None
    local_track_id: int


class CrossCameraTracker:
    """Manages per-camera ByteTrackers and cross-camera identity assignment.

    Each camera gets its own ByteTracker instance for single-camera tracking.
    When tracks become confirmed, their ReID embeddings are extracted and
    queried against a shared gallery to find cross-camera matches. Expired
    tracks are added to the gallery for future matching.

    Thread-safe: update() can be called concurrently from multiple camera
    threads (each camera thread only touches its own ByteTracker).
    """

    __slots__ = (
        "_gallery",
        "_link_model",
        "_local_to_global",
        "_lock",
        "_match_threshold",
        "_prev_track_ids",
        "_reid",
        "_tracker_kwargs",
        "_trackers",
    )

    def __init__(
        self,
        reid_extractor: ReIDExtractor,
        *,
        camera_link_model: CameraLinkModel | None = None,
        gallery_max_entries: int = 10_000,
        match_threshold: float = 0.4,
        **tracker_kwargs: Any,
    ) -> None:
        """Initialize cross-camera tracker.

        Args:
            reid_extractor: Shared ReID model for embedding extraction.
            camera_link_model: Optional spatial-temporal constraints.
            gallery_max_entries: Max embeddings in gallery.
            match_threshold: Cosine distance threshold for cross-camera match.
            **tracker_kwargs: Passed to each per-camera ByteTracker.
        """
        self._reid = reid_extractor
        self._gallery = EmbeddingGallery(
            embedding_dim=reid_extractor.embedding_dim,
            max_entries=gallery_max_entries,
        )
        self._link_model = camera_link_model
        self._match_threshold = match_threshold
        self._tracker_kwargs = tracker_kwargs
        self._trackers: dict[str, ByteTracker] = {}
        # Maps (camera_id, local_track_id) → global_id
        self._local_to_global: dict[tuple[str, int], int] = {}
        # Per-camera set of track IDs seen in previous frame output
        self._prev_track_ids: dict[str, set[int]] = {}
        self._lock = threading.Lock()

    def register_camera(self, camera_id: str) -> None:
        """Create a new ByteTracker for a camera.

        Args:
            camera_id: Unique camera identifier.

        Raises:
            ValueError: If camera_id is already registered.
        """
        with self._lock:
            if camera_id in self._trackers:
                msg = f"Camera '{camera_id}' already registered"
                raise ValueError(msg)
            self._trackers[camera_id] = ByteTracker(
                reid_extractor=self._reid,
                **self._tracker_kwargs,
            )
            self._prev_track_ids[camera_id] = set()

    def update(
        self,
        camera_id: str,
        detection: Detection,
        timestamp: float | None = None,
    ) -> list[GlobalTrackedBox]:
        """Process one frame from one camera.

        Steps:
          1. Run ByteTracker.update() for this camera (single-camera tracking)
          2. For confirmed tracks without global ID: extract embedding, query gallery
          3. Assign global IDs (matched or new)
          4. Clean up mappings for tracks that disappeared from output
          5. Return boxes with global IDs

        Args:
            camera_id: Camera that produced this detection.
            detection: Detection result from InferenceEngine.
            timestamp: Frame timestamp in seconds. Defaults to time.time().

        Returns:
            List of GlobalTrackedBox with cross-camera global IDs.
        """
        if timestamp is None:
            timestamp = time.time()

        # Auto-register camera on first use
        with self._lock:
            if camera_id not in self._trackers:
                self._trackers[camera_id] = ByteTracker(
                    reid_extractor=self._reid,
                    **self._tracker_kwargs,
                )
                self._prev_track_ids[camera_id] = set()

        tracker = self._trackers[camera_id]

        # Run single-camera tracking
        tracked_result = tracker.update(detection)

        # Collect current output track IDs
        curr_ids: set[int] = set()
        frame_pixels = detection.frame.pixels
        output: list[GlobalTrackedBox] = []

        for tbox in tracked_result.boxes:
            curr_ids.add(tbox.track_id)
            key = (camera_id, tbox.track_id)

            with self._lock:
                existing_gid = self._local_to_global.get(key)

            if existing_gid is not None:
                output.append(
                    GlobalTrackedBox(
                        box=tbox,
                        camera_id=camera_id,
                        global_id=existing_gid,
                        local_track_id=tbox.track_id,
                    )
                )
                continue

            if not tbox.is_confirmed:
                output.append(
                    GlobalTrackedBox(
                        box=tbox,
                        camera_id=camera_id,
                        global_id=None,
                        local_track_id=tbox.track_id,
                    )
                )
                continue

            # Confirmed track without global ID — extract embedding and query
            global_id = self._assign_global_id(
                camera_id,
                tbox,
                frame_pixels,
                timestamp,
            )
            output.append(
                GlobalTrackedBox(
                    box=tbox,
                    camera_id=camera_id,
                    global_id=global_id,
                    local_track_id=tbox.track_id,
                )
            )

        # Clean up mappings for tracks no longer in output
        prev_ids = self._prev_track_ids.get(camera_id, set())
        expired_ids = prev_ids - curr_ids
        for expired_id in expired_ids:
            key = (camera_id, expired_id)
            with self._lock:
                self._local_to_global.pop(key, None)

        self._prev_track_ids[camera_id] = curr_ids

        return output

    def _assign_global_id(
        self,
        camera_id: str,
        tbox: TrackedBox,
        frame_pixels: NDArray[np.uint8],
        timestamp: float,
    ) -> int:
        """Extract embedding and assign global ID via gallery query.

        Args:
            camera_id: Source camera.
            tbox: Confirmed tracked box.
            frame_pixels: Current frame pixels for embedding extraction.
            timestamp: Current timestamp.

        Returns:
            Assigned global_id.
        """
        box_xyxy = (tbox.x1, tbox.y1, tbox.x2, tbox.y2)
        embedding = self._reid.extract(frame_pixels, [box_xyxy])

        if embedding is not None and float(np.linalg.norm(embedding[0])) > 1e-8:
            emb = embedding[0]

            # Query gallery for cross-camera match
            matches = self._gallery.query(
                emb,
                exclude_camera=camera_id,
                top_k=5,
                threshold=self._match_threshold,
            )

            # Apply camera link filtering
            if matches and self._link_model is not None:
                matches = self._link_model.filter_matches(
                    camera_id,
                    timestamp,
                    matches,
                )

            global_id = matches[0].global_id if matches else self._gallery.next_global_id()

            # Add to gallery for future cross-camera matching
            self._gallery.add(
                camera_id=camera_id,
                local_track_id=tbox.track_id,
                embedding=emb,
                class_id=tbox.class_id,
                timestamp=timestamp,
                global_id=global_id,
            )
        else:
            # No valid embedding — assign new global ID without gallery entry
            global_id = self._gallery.next_global_id()

        with self._lock:
            self._local_to_global[(camera_id, tbox.track_id)] = global_id

        return global_id

    @property
    def gallery_size(self) -> int:
        """Current number of entries in the embedding gallery."""
        return self._gallery.size

    @property
    def camera_count(self) -> int:
        """Number of registered cameras."""
        with self._lock:
            return len(self._trackers)
