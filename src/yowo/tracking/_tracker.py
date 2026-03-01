"""ByteTrack multi-object tracker.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022. https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from yowo.tracking._kalman import KalmanFilterXYAH
from yowo.tracking._matching import (
    appearance_distance,
    gated_fused_cost,
    iou_distance,
    linear_assignment,
    needs_reid,
    remove_duplicate_tracks,
    remove_intra_duplicates,
)
from yowo.tracking._matching import (
    fuse_score as _fuse_score,
)
from yowo.tracking._reid import ReIDExtractor
from yowo.tracking._strack import STrack, TrackedBox, TrackedDetection, TrackState
from yowo.types import Detection


class ByteTracker:
    """ByteTrack multi-object tracker (Zhang et al., ECCV 2022).

    Implements two-stage detection association with optional CLIP-based ReID:
    1. High-confidence detections matched to active tracks via IoU (+ appearance).
    2. Low-confidence detections matched to remaining unmatched tracks.
    3. (Optional) Appearance-only rescue for long-lost tracks.

    Each instance is independent — create one per stream for multi-stream use.
    Not thread-safe; protect with a lock if shared across threads.

    Args:
        track_high_thresh: Stage-1 confidence gate. Default 0.6.
        track_low_thresh: Lower bound for stage-2 detections. Default 0.1.
        match_thresh: IoU distance threshold for stage-1 association. Default 0.8.
        max_age: Frames a lost track survives before removal. Default 30.
        min_hits: Consecutive hits before a track is marked confirmed. Default 3.
        reid_extractor: Optional ReID feature extractor (e.g. CLIPExtractor).
            When None, tracker uses pure IoU matching (backward compatible).
        reid_lost_age: Minimum frames a track must be lost before Stage 3
            appearance-only rescue is attempted. Default 5.
        reid_frame_interval: Minimum frames between ReID extractions to
            control compute budget. Default 3.
    """

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
        fuse_score: bool = False,
        reid_extractor: ReIDExtractor | None = None,
        reid_lost_age: int = 5,
        reid_frame_interval: int = 3,
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
        self._fuse_score = fuse_score
        # Clamped to 1.0: high_thresh=0.9 → new_thresh=1.0, not 1.05 (which would block all births)
        self._new_track_thresh = min(track_high_thresh + 0.1, 1.0)
        self._next_id = 1
        self._tracked: list[STrack] = []  # actively tracked
        self._lost: list[STrack] = []  # lost, within max_age
        self._kalman = KalmanFilterXYAH()
        # ReID state
        self._reid = reid_extractor
        self._reid_lost_age = reid_lost_age
        self._reid_frame_interval = reid_frame_interval
        self._frames_since_reid: int = reid_frame_interval  # start ready to extract

    def update(self, detection: Detection) -> TrackedDetection:
        """Process one Detection frame and return tracked result.

        Runs Kalman prediction, multi-stage association, track birth/death,
        and duplicate removal. Call once per frame in temporal order.

        Association stages:
          1. High-conf dets → confirmed tracked + lost (IoU + optional fuse_score + ReID)
          2. Low-conf dets → unmatched confirmed tracked (IoU only, thresh=0.5)
          3. Appearance rescue for long-lost tracks (ReID only)
          4. Remaining high-conf dets → unconfirmed tracks (IoU, thresh=0.7)
          5. Birth new tracks from remaining unmatched high-conf dets

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

        # --- Separate unconfirmed from confirmed tracks ---
        # Unconfirmed tracks (hits < min_hits) are excluded from Stage 1
        # and get a second chance in Stage 4 at a tighter threshold.
        unconfirmed: list[STrack] = []
        confirmed_tracked: list[STrack] = []
        for track in self._tracked:
            if track.is_confirmed:
                confirmed_tracked.append(track)
            else:
                unconfirmed.append(track)

        # --- STAGE 1: match high-conf dets → confirmed tracked + lost pool ---
        strack_pool = confirmed_tracked + self._lost
        high_arr = (
            np.array(high_boxes, dtype=np.float64)
            if high_boxes
            else np.empty((0, 4), dtype=np.float64)
        )
        cost1 = iou_distance(strack_pool, high_arr)

        # Optional: fuse detection confidence into IoU cost
        if self._fuse_score and high_confs:
            cost1 = _fuse_score(
                cost1,
                np.array(high_confs, dtype=np.float64),
            )

        # --- Conditional ReID: fuse appearance when IoU is ambiguous ---
        det_embeddings: NDArray[np.float32] | None = None
        if self._reid is not None:
            if self._frames_since_reid >= self._reid_frame_interval and needs_reid(cost1):
                det_embeddings = self._reid.extract(
                    detection.frame.pixels,
                    high_boxes,
                )
                self._frames_since_reid = 0
                if det_embeddings is not None:
                    track_embs = _gather_track_embeddings(
                        strack_pool,
                        self._reid.embedding_dim,
                    )
                    if track_embs is not None:
                        cost1 = gated_fused_cost(cost1, track_embs, det_embeddings)
            else:
                self._frames_since_reid += 1

        matches1, unmatched_track_idxs, unmatched_high_idxs = linear_assignment(
            cost1, self._match_thresh
        )

        # Update matched tracks (re-activate if they were lost)
        refound: list[STrack] = []
        for ti, di in matches1:
            track = strack_pool[ti]
            if track.state == TrackState.TRACKED:
                track.update(
                    high_boxes[di], high_confs[di], high_cls_ids[di], high_cls_names[di], frame_id
                )
            else:
                track.re_activate(
                    high_boxes[di], high_confs[di], high_cls_ids[di], high_cls_names[di], frame_id
                )
                refound.append(track)
            # Update appearance embedding for matched tracks
            if det_embeddings is not None:
                emb = det_embeddings[di]
                if float(np.linalg.norm(emb)) > 1e-8:
                    track.update_embedding(emb)

        # Collect TRACKED tracks not matched in stage 1 — candidates for stage 2.
        # Lost tracks that weren't re-associated stay in self._lost (no action).
        unmatched_tracked: list[STrack] = []
        for idx in unmatched_track_idxs:
            track = strack_pool[idx]
            if track.state == TrackState.TRACKED:
                unmatched_tracked.append(track)

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
                unmatched_tracked[ti].update(
                    low_boxes[di], low_confs[di], low_cls_ids[di], low_cls_names[di], frame_id
                )
            for idx in still_unmatched_idxs:
                still_unmatched_tracked.append(unmatched_tracked[idx])
        else:
            still_unmatched_tracked = list(unmatched_tracked)

        # Mark all remaining unmatched tracked tracks as lost
        for track in still_unmatched_tracked:
            track.mark_lost()

        # --- STAGE 3: Appearance rescue for long-lost tracks ---
        rescued_det_idxs: set[int] = set()
        if self._reid is not None and det_embeddings is not None and unmatched_high_idxs:
            stage3_rescued, rescued_det_idxs = _appearance_rescue(
                self._lost,
                unmatched_high_idxs,
                det_embeddings,
                high_boxes,
                high_confs,
                high_cls_ids,
                high_cls_names,
                frame_id,
                self._reid_lost_age,
            )
            refound.extend(stage3_rescued)

        # --- STAGE 4: Re-associate unconfirmed tracks with remaining dets ---
        # Unconfirmed tracks get a second chance to match before removal.
        matched_unconf: list[STrack] = []
        remaining_high_idxs = [i for i in unmatched_high_idxs if i not in rescued_det_idxs]
        if unconfirmed and remaining_high_idxs:
            remaining_arr = np.array(
                [high_boxes[i] for i in remaining_high_idxs],
                dtype=np.float64,
            )
            cost_unconf = iou_distance(unconfirmed, remaining_arr)
            if self._fuse_score:
                remaining_scores = np.array(
                    [high_confs[i] for i in remaining_high_idxs],
                    dtype=np.float64,
                )
                cost_unconf = _fuse_score(cost_unconf, remaining_scores)
            matches_uc, unmatched_uc_idxs, _ = linear_assignment(
                cost_unconf,
                0.7,
            )
            for ti, di in matches_uc:
                orig_di = remaining_high_idxs[di]
                unconfirmed[ti].update(
                    high_boxes[orig_di],
                    high_confs[orig_di],
                    high_cls_ids[orig_di],
                    high_cls_names[orig_di],
                    frame_id,
                )
                if det_embeddings is not None:
                    emb = det_embeddings[orig_di]
                    if float(np.linalg.norm(emb)) > 1e-8:
                        unconfirmed[ti].update_embedding(emb)
                matched_unconf.append(unconfirmed[ti])
                rescued_det_idxs.add(orig_di)
            for idx in unmatched_uc_idxs:
                unconfirmed[idx].mark_removed()
        else:
            for track in unconfirmed:
                track.mark_removed()

        # --- NEW TRACKS from unmatched high-conf dets ---
        new_tracks: list[STrack] = []
        matched_boxes = _collect_matched_boxes(matches1, high_boxes)
        for idx in unmatched_high_idxs:
            if idx in rescued_det_idxs:
                continue  # already rescued or consumed by unconfirmed
            if high_confs[idx] >= self._new_track_thresh:
                if _overlaps_any(high_boxes[idx], matched_boxes, thresh=0.7):
                    continue  # suppress — overlaps an already-matched track
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
                # Initialize embedding for new track
                if det_embeddings is not None:
                    emb = det_embeddings[idx]
                    if float(np.linalg.norm(emb)) > 1e-8:
                        track.update_embedding(emb)
                matched_boxes.append(high_boxes[idx])
                new_tracks.append(track)

        # --- UPDATE LOST pool ---
        refound_ids = {t.track_id for t in refound}
        new_lost: list[STrack] = []
        for track in self._lost:
            if track.track_id in refound_ids:
                continue
            if track.time_since_update <= self._max_age:
                new_lost.append(track)
            else:
                track.mark_removed()

        for track in still_unmatched_tracked:
            if track.state == TrackState.LOST:
                new_lost.append(track)

        self._lost = new_lost

        # Rebuild active tracked list:
        # confirmed TRACKED + re-found + matched unconfirmed + newly birthed
        self._tracked = (
            [t for t in confirmed_tracked if t.state == TrackState.TRACKED]
            + refound
            + matched_unconf
            + new_tracks
        )

        # --- Duplicate removal ---
        self._tracked = remove_intra_duplicates(self._tracked)
        self._tracked, self._lost = remove_duplicate_tracks(self._tracked, self._lost)

        # --- Build output (clamp to frame bounds) ---
        frame_h, frame_w = detection.frame.pixels.shape[:2]
        output_boxes = tuple(
            _clamp_tracked_box(t.to_tracked_box(), frame_w, frame_h) for t in self._tracked
        )
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


def _gather_track_embeddings(
    tracks: list[STrack],
    embedding_dim: int,
) -> NDArray[np.float32] | None:
    """Stack track embeddings into a matrix, zeros for tracks without.

    Tracks without embeddings get zero vectors, which produce high cosine
    distance (1.0) and are naturally rejected by gated fusion.
    """
    if not tracks:
        return None
    embs: list[NDArray[np.float32]] = []
    for t in tracks:
        if t.embedding is not None:
            embs.append(t.embedding)
        else:
            embs.append(np.zeros(embedding_dim, dtype=np.float32))
    return np.stack(embs, axis=0)


def _appearance_rescue(
    lost_tracks: list[STrack],
    unmatched_det_idxs: list[int],
    det_embeddings: NDArray[np.float32],
    boxes: list[tuple[float, float, float, float]],
    confs: list[float],
    cls_ids: list[int],
    cls_names: list[str],
    frame_id: int,
    reid_lost_age: int,
    theta_e: float = 0.30,
) -> tuple[list[STrack], set[int]]:
    """Stage 3: Re-activate long-lost tracks via appearance-only matching.

    Returns:
        (rescued_tracks, rescued_det_indices) — tracks re-activated and
        detection indices consumed (should not birth new tracks).
    """
    long_lost = [
        t for t in lost_tracks if t.time_since_update > reid_lost_age and t.embedding is not None
    ]
    if not long_lost or not unmatched_det_idxs:
        return [], set()

    lost_embs = np.stack([t.embedding for t in long_lost], axis=0)  # type: ignore[misc]
    det_subset = det_embeddings[unmatched_det_idxs]
    cos_cost = appearance_distance(lost_embs, det_subset)
    matches, _, _ = linear_assignment(cos_cost, theta_e)

    rescued: list[STrack] = []
    rescued_det_idxs: set[int] = set()
    for ti, di in matches:
        orig_di = unmatched_det_idxs[di]
        long_lost[ti].re_activate(
            boxes[orig_di],
            confs[orig_di],
            cls_ids[orig_di],
            cls_names[orig_di],
            frame_id,
        )
        long_lost[ti].update_embedding(det_embeddings[orig_di])
        rescued.append(long_lost[ti])
        rescued_det_idxs.add(orig_di)

    return rescued, rescued_det_idxs


def _clamp_tracked_box(box: TrackedBox, w: int, h: int) -> TrackedBox:
    """Clamp box coordinates to frame bounds [0, w] x [0, h].

    Kalman-predicted boxes can extend outside the frame (e.g. when a car
    enters from the edge with a distorted initial aspect ratio). Clamping
    the output prevents oversized drawn boxes while leaving the internal
    Kalman state untouched for correct velocity tracking.
    """
    cx1 = max(0.0, min(box.x1, float(w)))
    cy1 = max(0.0, min(box.y1, float(h)))
    cx2 = max(0.0, min(box.x2, float(w)))
    cy2 = max(0.0, min(box.y2, float(h)))
    if cx1 == box.x1 and cy1 == box.y1 and cx2 == box.x2 and cy2 == box.y2:
        return box  # no change — avoid allocation
    return TrackedBox(
        x1=cx1,
        y1=cy1,
        x2=cx2,
        y2=cy2,
        confidence=box.confidence,
        class_id=box.class_id,
        class_name=box.class_name,
        track_id=box.track_id,
        is_confirmed=box.is_confirmed,
    )


def _collect_matched_boxes(
    matches: list[tuple[int, int]],
    det_boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Collect detection boxes that were matched in stage-1.

    Returns the DETECTION boxes (not predicted), because they represent
    the actual object position this frame. Used to suppress duplicate births.
    """
    return [det_boxes[di] for _, di in matches]


def _overlaps_any(
    box: tuple[float, float, float, float],
    existing: list[tuple[float, float, float, float]],
    thresh: float,
) -> bool:
    """Check if box overlaps any existing box above the IoU threshold."""
    if not existing:
        return False
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return False
    for ex1, ey1, ex2, ey2 in existing:
        ix1 = max(x1, ex1)
        iy1 = max(y1, ey1)
        ix2 = min(x2, ex2)
        iy2 = min(y2, ey2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_e = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
        union = area + area_e - inter
        if union > 0 and inter / union >= thresh:
            return True
    return False


__all__ = ["ByteTracker"]
