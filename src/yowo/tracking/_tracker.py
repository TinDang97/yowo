"""ByteTrack multi-object tracker.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022. https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

import time

import numpy as np

from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import iou_distance, linear_assignment, remove_duplicate_tracks
from yowo.tracking._strack import STrack, TrackedDetection, TrackState
from yowo.types import Detection


class ByteTracker:
    """ByteTrack multi-object tracker (Zhang et al., ECCV 2022).

    Implements two-stage detection association:
    1. High-confidence detections matched to active tracks via IoU.
    2. Low-confidence detections matched to remaining unmatched tracks.

    Each instance is independent — create one per stream for multi-stream use.
    Not thread-safe; protect with a lock if shared across threads.

    Args:
        track_high_thresh: Stage-1 confidence gate. Detections above this
            are used in the first association pass. Default 0.6.
        track_low_thresh: Lower bound for stage-2 detections. Detections
            between this and track_high_thresh are used in the second pass.
            Default 0.1.
        match_thresh: IoU distance threshold for stage-1 association.
            Pairs with cost > match_thresh are rejected. Default 0.8.
        max_age: Frames a lost track survives before removal. Default 30.
        min_hits: Consecutive hits before a track is marked confirmed.
            Default 3.
    """

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        if track_low_thresh >= track_high_thresh:
            raise ValueError(
                f"track_low_thresh ({track_low_thresh}) must be less than "
                f"track_high_thresh ({track_high_thresh})"
            )
        self._track_high_thresh = track_high_thresh
        self._track_low_thresh = track_low_thresh
        self._match_thresh = match_thresh
        self._stage2_thresh = 0.5  # hardcoded per paper
        self._max_age = max_age
        self._min_hits = min_hits
        # Clamped to 1.0: high_thresh=0.9 → new_thresh=1.0, not 1.05 (which would block all births)
        self._new_track_thresh = min(track_high_thresh + 0.1, 1.0)
        self._next_id = 1
        self._tracked: list[STrack] = []  # actively tracked
        self._lost: list[STrack] = []  # lost, within max_age
        self._kalman = KalmanFilterXYAH()

    def update(self, detection: Detection) -> TrackedDetection:
        """Process one Detection frame and return tracked result.

        Runs Kalman prediction, two-stage association, track birth/death,
        and duplicate removal. Call once per frame in temporal order.

        Args:
            detection: A Detection from engine.detect() or engine.stream().

        Returns:
            TrackedDetection with a TrackedBox per active track.
        """
        t0 = time.perf_counter()
        frame_id: int = detection.frame.frame_index

        # --- Split detections by confidence tier ---
        high_boxes: list[tuple[float, float, float, float]] = []
        high_confs: list[float] = []
        high_cls_ids: list[int] = []
        high_cls_names: list[str] = []
        low_boxes: list[tuple[float, float, float, float]] = []
        low_confs: list[float] = []
        low_cls_ids: list[int] = []
        low_cls_names: list[str] = []

        for box in detection.boxes:
            conf = box.confidence
            xyxy = box.as_xyxy
            if conf >= self._track_high_thresh:
                high_boxes.append(xyxy)
                high_confs.append(conf)
                high_cls_ids.append(box.class_id)
                high_cls_names.append(box.class_name)
            elif conf >= self._track_low_thresh:
                low_boxes.append(xyxy)
                low_confs.append(conf)
                low_cls_ids.append(box.class_id)
                low_cls_names.append(box.class_name)

        # --- Kalman predict all tracks ---
        for track in self._tracked + self._lost:
            track.predict()

        # --- STAGE 1: match high-conf dets → active tracked tracks ---
        high_arr = (
            np.array(high_boxes, dtype=np.float64)
            if high_boxes
            else np.empty((0, 4), dtype=np.float64)
        )
        cost1 = iou_distance(self._tracked, high_arr)
        matches1, unmatched_track_idxs, unmatched_high_idxs = linear_assignment(
            cost1, self._match_thresh
        )

        # Update matched tracks
        for ti, di in matches1:
            self._tracked[ti].update(
                high_boxes[di], high_confs[di], high_cls_ids[di], high_cls_names[di], frame_id
            )

        # Collect tracks not matched in stage 1 — candidates for stage 2
        # These were TRACKED before but got no high-conf match.
        unmatched_tracked: list[STrack] = []
        newly_lost: list[STrack] = []
        for idx in unmatched_track_idxs:
            track = self._tracked[idx]
            if track.state == TrackState.TRACKED:
                unmatched_tracked.append(track)
            else:
                newly_lost.append(track)

        # --- STAGE 2: match low-conf dets → unmatched tracked tracks ---
        low_arr = (
            np.array(low_boxes, dtype=np.float64)
            if low_boxes
            else np.empty((0, 4), dtype=np.float64)
        )
        still_unmatched_tracked: list[STrack] = []
        if unmatched_tracked and low_boxes:
            cost2 = iou_distance(unmatched_tracked, low_arr)
            matches2, still_unmatched_idxs, _ = linear_assignment(cost2, self._stage2_thresh)
            for ti, di in matches2:
                unmatched_tracked[ti].re_activate(
                    low_boxes[di], low_confs[di], low_cls_ids[di], low_cls_names[di], frame_id
                )
            for idx in still_unmatched_idxs:
                still_unmatched_tracked.append(unmatched_tracked[idx])
        else:
            still_unmatched_tracked = list(unmatched_tracked)

        # Mark all remaining unmatched tracked tracks as lost
        for track in still_unmatched_tracked + newly_lost:
            track.mark_lost()

        # --- NEW TRACKS from unmatched high-conf dets ---
        for idx in unmatched_high_idxs:
            if high_confs[idx] >= self._new_track_thresh:
                track = STrack(
                    self._next_id,
                    high_boxes[idx],
                    high_confs[idx],
                    high_cls_ids[idx],
                    high_cls_names[idx],
                    self._kalman,
                    self._min_hits,
                )
                self._next_id += 1
                track.activate(frame_id)
                self._tracked.append(track)

        # --- UPDATE LOST pool ---
        # Add newly lost + age existing lost; remove expired
        new_lost: list[STrack] = []
        for track in self._lost:
            if track.time_since_update <= self._max_age:
                new_lost.append(track)
            else:
                track.mark_removed()

        for track in still_unmatched_tracked + newly_lost:
            if track.state == TrackState.LOST:
                new_lost.append(track)

        self._lost = new_lost

        # Rebuild active tracked list (only TRACKED state)
        self._tracked = [t for t in self._tracked if t.state == TrackState.TRACKED]

        # --- Duplicate removal across tracked / lost ---
        self._tracked, self._lost = remove_duplicate_tracks(self._tracked, self._lost)

        # --- Build output ---
        output_boxes = tuple(t.to_tracked_box() for t in self._tracked)
        tracking_time_ms = (time.perf_counter() - t0) * 1000.0

        return TrackedDetection(
            frame=detection.frame,
            boxes=output_boxes,
            inference_time_ms=detection.inference_time_ms,
            tracking_time_ms=tracking_time_ms,
            backend=detection.backend,
            model_spec=detection.model_spec,
        )

    def reset(self) -> None:
        """Clear all tracks. Does NOT reset the track ID counter."""
        self._tracked.clear()
        self._lost.clear()

    @property
    def active_track_count(self) -> int:
        """Number of actively tracked (TRACKED state) tracks."""
        return len(self._tracked)

    @property
    def lost_track_count(self) -> int:
        """Number of lost tracks still within max_age."""
        return len(self._lost)


__all__ = ["ByteTracker"]
