"""Internal track representation and public tracked result types for ByteTrack."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.types import BackendType, Frame, ModelSpec

if TYPE_CHECKING:
    pass


class TrackState(enum.IntEnum):
    """Lifecycle state of a tracked object."""

    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(frozen=True, slots=True)
class TrackedBox:
    """A bounding box annotated with a persistent track ID.

    Attributes:
        x1: Left edge in pixel coordinates.
        y1: Top edge in pixel coordinates.
        x2: Right edge in pixel coordinates.
        y2: Bottom edge in pixel coordinates.
        confidence: Detection confidence in [0, 1].
        class_id: Integer class index.
        class_name: Human-readable class label.
        track_id: Unique persistent track identifier.
        is_confirmed: True when the track has enough hit history.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int
    is_confirmed: bool

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        width = max(0.0, self.x2 - self.x1)
        height = max(0.0, self.y2 - self.y1)
        return width * height

    @property
    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Coordinates as a plain (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Serialize to a plain dict with JSON-safe primitive values."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "is_confirmed": self.is_confirmed,
        }


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """Tracking result for a single frame.

    Attributes:
        frame: The source frame that was processed.
        boxes: Tuple of tracked bounding boxes.
        inference_time_ms: Wall-clock time for the inference call only.
        tracking_time_ms: Wall-clock time for the tracking update step.
        backend: Backend that produced the detections.
        model_spec: Model that produced the detections.
    """

    frame: Frame
    boxes: tuple[TrackedBox, ...]
    inference_time_ms: float
    tracking_time_ms: float
    backend: BackendType
    model_spec: ModelSpec

    @property
    def num_boxes(self) -> int:
        """Number of tracked bounding boxes."""
        return len(self.boxes)

    @property
    def has_detections(self) -> bool:
        """True when at least one tracked box is present."""
        return len(self.boxes) > 0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict.

        Pixel data and local paths are excluded — only portable metadata
        is included so the result is safe for logging and wire transport.
        """
        return {
            "source_id": self.frame.source_id,
            "frame_index": self.frame.frame_index,
            "timestamp_ms": self.frame.timestamp_ms,
            "inference_time_ms": self.inference_time_ms,
            "tracking_time_ms": self.tracking_time_ms,
            "backend": str(self.backend),
            "model": f"{self.model_spec.family.value}{self.model_spec.size.value}",
            "boxes": [box.to_dict() for box in self.boxes],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class STrack:
    """Single object track maintained by ByteTracker.

    Mutable internal state — not part of the public API.
    """

    __slots__ = (
        "_covariance",
        "_det_xyxy",
        "_embedding",
        "_kalman",
        "_mean",
        "_min_hits",
        "age",
        "class_id",
        "class_name",
        "confidence",
        "frame_id",
        "hits",
        "start_frame",
        "state",
        "time_since_update",
        "track_id",
    )

    def __init__(
        self,
        track_id: int,
        box_xyxy: tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
        kalman: KalmanFilterXYAH,
        min_hits: int,
    ) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.state = TrackState.NEW
        self._kalman = kalman
        self._min_hits = min_hits
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.start_frame = 0
        self.frame_id = 0
        self._embedding: NDArray[np.float32] | None = None
        self._det_xyxy = box_xyxy

        measurement = KalmanFilterXYAH.xyxy_to_xyah(box_xyxy)
        self._mean, self._covariance = kalman.initiate(measurement)

    def predict(self) -> None:
        """Advance the Kalman filter prediction one time step.

        Lost tracks have their height velocity zeroed to prevent unchecked
        drift while they await re-association (matches reference impl).
        """
        if self.state != TrackState.TRACKED:
            self._mean[7] = 0  # zero height velocity
        self._mean, self._covariance = self._kalman.predict(self._mean, self._covariance)
        self.age += 1
        self.time_since_update += 1

    def activate(self, frame_id: int) -> None:
        """Transition track from NEW to TRACKED on first confirmed association.

        Args:
            frame_id: Current frame index.
        """
        self.state = TrackState.TRACKED
        self.hits = 1
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.time_since_update = 0

    def update(
        self,
        box_xyxy: tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
        frame_id: int,
    ) -> None:
        """Update track state from a new matched detection.

        Args:
            box_xyxy: New detection bounding box.
            confidence: New detection confidence.
            class_id: Detected class index.
            class_name: Detected class name.
            frame_id: Current frame index.
        """
        measurement = KalmanFilterXYAH.xyxy_to_xyah(box_xyxy)
        self._mean, self._covariance = self._kalman.update(
            self._mean, self._covariance, measurement
        )
        self._det_xyxy = box_xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.frame_id = frame_id
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED

    def re_activate(
        self,
        box_xyxy: tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str,
        frame_id: int,
    ) -> None:
        """Re-activate a previously lost track from a new matched detection.

        Args:
            box_xyxy: Matched detection bounding box.
            confidence: Matched detection confidence.
            class_id: Detected class index.
            class_name: Detected class name.
            frame_id: Current frame index.
        """
        measurement = KalmanFilterXYAH.xyxy_to_xyah(box_xyxy)
        self._mean, self._covariance = self._kalman.update(
            self._mean, self._covariance, measurement
        )
        self._det_xyxy = box_xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.frame_id = frame_id
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED

    def mark_lost(self) -> None:
        """Mark track as lost when unmatched for one or more frames."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Mark track for removal after exceeding max_age."""
        self.state = TrackState.REMOVED

    @property
    def is_confirmed(self) -> bool:
        """True when track has sufficient hit history and is actively tracked."""
        return self.hits >= self._min_hits and self.state == TrackState.TRACKED

    @property
    def embedding(self) -> NDArray[np.float32] | None:
        """Current appearance embedding, or None if not yet extracted."""
        return self._embedding

    def update_embedding(self, new_embedding: NDArray[np.float32], eta: float = 0.9) -> None:
        """Update track appearance embedding via EMA with L2 renormalization.

        Args:
            new_embedding: (D,) L2-normalized embedding from ReID extractor.
            eta: EMA momentum -- higher retains more history. Default 0.9.
        """
        if self._embedding is None:
            self._embedding = new_embedding.copy()
            return
        blended = eta * self._embedding + (1.0 - eta) * new_embedding
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            blended = blended / np.float32(norm)
        self._embedding = blended.astype(np.float32, copy=False)

    @property
    def predicted_xyxy(self) -> tuple[float, float, float, float]:
        """Current predicted position as (x1, y1, x2, y2) pixel coordinates."""
        return KalmanFilterXYAH.xyah_to_xyxy(self._mean[:4])

    @property
    def output_xyxy(self) -> tuple[float, float, float, float]:
        """Box for output: raw detection when matched, Kalman when unmatched.

        Matched tracks (time_since_update == 0) use the raw detection — always
        more accurate than Kalman prediction, especially for edge-entering
        objects whose aspect ratio changes rapidly.  Kalman prediction is only
        used when the track has no fresh detection (lost/occluded frames).
        """
        if self.time_since_update == 0:
            return self._det_xyxy
        return self.predicted_xyxy

    def to_tracked_box(self) -> TrackedBox:
        """Convert current track state to an immutable TrackedBox."""
        x1, y1, x2, y2 = self.output_xyxy
        return TrackedBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            track_id=self.track_id,
            is_confirmed=self.is_confirmed,
        )


__all__ = ["TrackState", "TrackedBox", "TrackedDetection"]
