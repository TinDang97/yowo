"""Object tracking for yowo — ByteTrack multi-object tracker.

Provides ByteTrack multi-object tracking as a composable post-processing
layer over Detection. Assigns stable track IDs across frames without
modifying the InferenceEngine.

Example::

    from yowo import InferenceEngine, open_source
    from yowo.tracking import track_stream

    with InferenceEngine() as engine:
        for tracked in track_stream(engine, open_source("video.mp4")):
            for box in tracked.boxes:
                print(f"Track {box.track_id}: {box.class_name} ({box.confidence:.2f})")
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from yowo.tracking._camera_link import CameraLink, CameraLinkModel
from yowo.tracking._clip_reid import CLIPReIDExtractor
from yowo.tracking._cross_camera import CrossCameraTracker, GlobalTrackedBox
from yowo.tracking._gallery import EmbeddingGallery, GalleryEntry, GalleryMatch
from yowo.tracking._reid import (
    CLIPExtractor,
    FastReIDExtractor,
    ReIDExtractor,
    VehicleReIDExtractor,
)
from yowo.tracking._strack import TrackedBox, TrackedDetection, TrackState
from yowo.tracking._tracker import ByteTracker


def track_stream(
    engine: Any,
    source: Any,
    *,
    tracker: ByteTracker | None = None,
    reid_extractor: ReIDExtractor | None = None,
    **tracker_kwargs: Any,
) -> Iterator[TrackedDetection]:
    """Wrap engine.stream() to yield TrackedDetection with track IDs.

    Creates a ByteTracker internally or uses the provided instance.
    The same tracker is reused across all frames — do NOT share a tracker
    across multiple concurrent streams.

    Args:
        engine: A loaded InferenceEngine.
        source: Any FrameSource compatible with engine.stream().
        tracker: Optional pre-configured ByteTracker. If None, one is
            created from tracker_kwargs.
        reid_extractor: Optional ReID feature extractor (e.g. CLIPExtractor).
            Forwarded to ByteTracker if tracker is None.
        **tracker_kwargs: Forwarded to ByteTracker() if tracker is None.

    Yields:
        TrackedDetection for each frame.
    """
    if tracker is None:
        tracker = ByteTracker(reid_extractor=reid_extractor, **tracker_kwargs)
    for detection in engine.stream(source):
        yield tracker.update(detection)


def track_detections(
    detections: Iterable[Any],
    *,
    tracker: ByteTracker | None = None,
    reid_extractor: ReIDExtractor | None = None,
    **tracker_kwargs: Any,
) -> Iterator[TrackedDetection]:
    """Apply tracking to an iterable of Detections.

    Args:
        detections: Any iterable of Detection objects.
        tracker: Optional pre-configured ByteTracker.
        reid_extractor: Optional ReID feature extractor (e.g. CLIPExtractor).
            Forwarded to ByteTracker if tracker is None.
        **tracker_kwargs: Forwarded to ByteTracker() if tracker is None.

    Yields:
        TrackedDetection for each input Detection.
    """
    if tracker is None:
        tracker = ByteTracker(reid_extractor=reid_extractor, **tracker_kwargs)
    for detection in detections:
        yield tracker.update(detection)


__all__ = [
    "ByteTracker",
    "CLIPExtractor",
    "CLIPReIDExtractor",
    "CameraLink",
    "CameraLinkModel",
    "CrossCameraTracker",
    "EmbeddingGallery",
    "FastReIDExtractor",
    "GalleryEntry",
    "GalleryMatch",
    "GlobalTrackedBox",
    "ReIDExtractor",
    "TrackState",
    "TrackedBox",
    "TrackedDetection",
    "VehicleReIDExtractor",
    "track_detections",
    "track_stream",
]
