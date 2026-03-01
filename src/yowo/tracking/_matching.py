"""IoU distance matrix and linear assignment for ByteTrack."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from yowo.tracking._strack import STrack

# Use scipy's C-extension linear_sum_assignment when available (~500x faster than
# the pure-numpy fallback for N>=20). Install with: pip install yowo[tracking]
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore[import-untyped]

    _has_scipy = True
except ImportError:
    _has_scipy = False


def iou_batch(
    boxes_a: NDArray[np.float64],
    boxes_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized IoU between (N, 4) and (M, 4) xyxy boxes.

    Args:
        boxes_a: (N, 4) xyxy bounding boxes.
        boxes_b: (M, 4) xyxy bounding boxes.

    Returns:
        (N, M) IoU matrix.
    """
    # Expand dims for broadcast: (N,1,4) vs (1,M,4)
    a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = np.maximum(a[:, :, 1], b[:, :, 1])
    inter_x2 = np.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = np.minimum(a[:, :, 3], b[:, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    # Clamp to zero: Kalman-predicted boxes can have inverted coords (x2 < x1).
    area_a = np.maximum(0.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = np.maximum(0.0, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter + 1e-7
    return inter / union


def iou_distance(
    tracks: Sequence[STrack],
    detections: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cost matrix (1 - IoU) between track predicted positions and detection boxes.

    Args:
        tracks: Sequence of active STrack objects.
        detections: (M, 4) detection boxes in xyxy format.

    Returns:
        (N, M) cost matrix where N = len(tracks).
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float64)

    track_boxes = np.array([t.predicted_xyxy for t in tracks], dtype=np.float64)
    return 1.0 - iou_batch(track_boxes, detections)


def linear_assignment(
    cost_matrix: NDArray[np.float64],
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with threshold gating.

    Args:
        cost_matrix: (N, M) cost matrix.
        thresh: Maximum cost for a valid match.

    Returns:
        (matches, unmatched_rows, unmatched_cols) where matches is a list of
        (row_idx, col_idx) pairs and unmatched lists contain remaining indices.
    """
    if cost_matrix.size == 0:
        rows, cols = cost_matrix.shape
        return [], list(range(rows)), list(range(cols))

    row_ind, col_ind = _hungarian(cost_matrix)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for r, c in zip(row_ind, col_ind, strict=True):
        if cost_matrix[r, c] <= thresh:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]

    return matches, unmatched_rows, unmatched_cols


def _hungarian(
    cost: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Optimal assignment for rectangular cost matrices.

    Uses scipy.optimize.linear_sum_assignment when available (recommended;
    ~500x faster than the pure-numpy fallback for N>=20).
    Falls back to pure-numpy Munkres otherwise.

    Args:
        cost: (N, M) cost matrix.

    Returns:
        (row_indices, col_indices) of optimal assignments.
    """
    if _has_scipy:
        row_ind, col_ind = _scipy_lsa(cost)  # type: ignore[possibly-undefined]
        return row_ind.astype(np.intp), col_ind.astype(np.intp)

    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)

    # Pad to square with a large cost to prevent spurious assignments.
    padded = np.full((n, n), fill_value=1e9, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    # Run Munkres on the padded square matrix.
    row_ind, col_ind = _munkres(padded)

    # Filter out padding assignments.
    valid = (row_ind < n_rows) & (col_ind < n_cols)
    return row_ind[valid].astype(np.intp), col_ind[valid].astype(np.intp)


def _munkres(
    cost: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pure-numpy Munkres (Hungarian) algorithm for square cost matrices.

    Implements the six-step algorithm. Operates on a copy to avoid
    mutating the caller's array.

    Args:
        cost: (n, n) square cost matrix.

    Returns:
        (row_indices, col_indices) arrays of optimal assignment pairs.
    """
    n = cost.shape[0]
    c = cost.copy()

    # Step 1: Row reduction — subtract row minimum.
    c -= c.min(axis=1, keepdims=True)

    # Step 2: Column reduction — subtract column minimum.
    c -= c.min(axis=0, keepdims=True)

    # Starred and primed zeros: 0=none, 1=starred, 2=primed.
    mask = np.zeros((n, n), dtype=np.int8)

    # Mark initial starred zeros (one per row, one per col).
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if c[i, j] < 1e-9 and not row_covered[i] and not col_covered[j]:
                mask[i, j] = 1
                row_covered[i] = True
                col_covered[j] = True

    row_covered[:] = False
    col_covered[:] = False

    # Main loop: cover columns with starred zeros, then iterate.
    while True:
        # Cover columns containing starred zeros.
        col_covered[:] = np.any(mask == 1, axis=0)

        if col_covered.sum() >= n:
            break  # Optimal assignment found.

        # Find an uncovered zero and prime it.
        path_row_0 = path_col_0 = -1
        done = False

        while not done:
            # Find an uncovered zero.
            zero_row, zero_col = -1, -1
            for i in range(n):
                if row_covered[i]:
                    continue
                for j in range(n):
                    if not col_covered[j] and c[i, j] < 1e-9:
                        zero_row, zero_col = i, j
                        break
                if zero_row >= 0:
                    break

            if zero_row < 0:
                # No uncovered zeros: adjust the matrix.
                uncovered_vals = c[~row_covered][:, ~col_covered]
                min_val = uncovered_vals.min()
                c[~row_covered] -= min_val
                c[:, col_covered] += min_val
                # Do not break — re-scan for uncovered zeros.
                continue

            # Prime the uncovered zero.
            mask[zero_row, zero_col] = 2

            # Find a starred zero in the same row.
            star_col = -1
            star_cols = np.where(mask[zero_row] == 1)[0]
            if len(star_cols) > 0:
                star_col = int(star_cols[0])

            if star_col >= 0:
                # Cover the row, uncover the column.
                row_covered[zero_row] = True
                col_covered[star_col] = False
            else:
                # Build augmenting path starting at (zero_row, zero_col).
                path_row_0, path_col_0 = zero_row, zero_col
                done = True

        if path_row_0 < 0:
            continue

        # Augmenting path: alternate star/prime zeros.
        path: list[tuple[int, int]] = [(path_row_0, path_col_0)]

        while True:
            _, c_col = path[-1]
            # Find starred zero in the column of the last path element.
            star_rows = np.where(mask[:, c_col] == 1)[0]
            if len(star_rows) == 0:
                break
            star_row = int(star_rows[0])
            path.append((star_row, c_col))
            # Find primed zero in the row of the starred zero.
            prime_cols = np.where(mask[star_row] == 2)[0]
            prime_col = int(prime_cols[0])
            path.append((star_row, prime_col))

        # Flip starred and primed along the augmenting path.
        for pr, pc in path:
            if mask[pr, pc] == 1:
                mask[pr, pc] = 0
            else:
                mask[pr, pc] = 1

        # Erase all primes and uncover all rows.
        mask[mask == 2] = 0
        row_covered[:] = False
        col_covered[:] = False

    row_ind, col_ind = np.where(mask == 1)
    return row_ind.astype(np.intp), col_ind.astype(np.intp)


def appearance_distance(
    track_embeddings: NDArray[np.float32],
    det_embeddings: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Cosine distance matrix between track and detection embeddings.

    Args:
        track_embeddings: (N, D) L2-normalized track appearance vectors.
        det_embeddings: (M, D) L2-normalized detection appearance vectors.

    Returns:
        (N, M) cosine distance matrix in [0, 2]. Lower = more similar.
    """
    if track_embeddings.size == 0 or det_embeddings.size == 0:
        return np.zeros(
            (track_embeddings.shape[0], det_embeddings.shape[0]),
            dtype=np.float64,
        )
    similarity = track_embeddings @ det_embeddings.T  # (N, M)
    return (1.0 - similarity).astype(np.float64)


def gated_fused_cost(
    iou_cost: NDArray[np.float64],
    track_embeddings: NDArray[np.float32],
    det_embeddings: NDArray[np.float32],
    theta_e: float = 0.30,
    theta_iou: float = 0.5,
) -> NDArray[np.float64]:
    """BoT-SORT gated min-cost fusion of IoU and appearance distance.

    For each (track, detection) pair:
    - Compute cosine distance from embeddings
    - If cos_dist < theta_e AND iou_cost < theta_iou (both gates pass):
        d_hat = 0.5 * cos_dist (scaled appearance cost)
    - Else:
        d_hat = 1.0 (reject appearance, fall back to IoU)
    - Final cost = min(iou_cost, d_hat)

    Args:
        iou_cost: (N, M) IoU distance matrix.
        track_embeddings: (N, D) L2-normalized track embeddings.
        det_embeddings: (M, D) L2-normalized detection embeddings.
        theta_e: Appearance gate threshold. Default 0.30.
        theta_iou: IoU gate threshold. Default 0.5.

    Returns:
        (N, M) fused cost matrix.
    """
    cos_dist = 1.0 - track_embeddings @ det_embeddings.T  # (N, M)
    gate = (cos_dist < theta_e) & (iou_cost < theta_iou)
    d_hat = np.where(gate, 0.5 * cos_dist, 1.0)
    return np.minimum(iou_cost, d_hat).astype(np.float64)


def needs_reid(
    iou_cost: NDArray[np.float64],
    clear_thresh: float = 0.4,
    no_overlap_thresh: float = 0.7,
) -> bool:
    """Check whether ReID extraction is needed based on IoU ambiguity.

    Returns True when IoU cost matrix has ambiguous assignments
    (multiple tracks with moderate overlap for a detection).
    Returns False when all assignments are clear or absent.

    Args:
        iou_cost: (N, M) IoU distance matrix.
        clear_thresh: Cost below which match is unambiguous.
        no_overlap_thresh: Cost above which no spatial association exists.

    Returns:
        True if appearance features would help disambiguation.
    """
    if iou_cost.size == 0:
        return False
    for col in range(iou_cost.shape[1]):
        col_costs = iou_cost[:, col]
        clear_matches = int(np.sum(col_costs < clear_thresh))
        if clear_matches == 1:
            continue
        if np.all(col_costs > no_overlap_thresh):
            continue
        ambiguous = int(np.sum((col_costs >= clear_thresh) & (col_costs <= no_overlap_thresh)))
        if ambiguous >= 2:
            return True
    return False


def remove_duplicate_tracks(
    tracks_a: list[STrack],
    tracks_b: list[STrack],
) -> tuple[list[STrack], list[STrack]]:
    """Remove duplicate tracks across two track lists.

    When two tracks have IoU distance < 0.15, the shorter-lived one is removed.

    Args:
        tracks_a: First list of tracks (typically active tracks).
        tracks_b: Second list of tracks (typically re-activated or new tracks).

    Returns:
        (filtered_a, filtered_b) with duplicates removed.
    """
    if not tracks_a or not tracks_b:
        return tracks_a, tracks_b

    dist_matrix = iou_distance(
        tracks_a, np.array([t.predicted_xyxy for t in tracks_b], dtype=np.float64)
    )

    remove_a: set[int] = set()
    remove_b: set[int] = set()

    pairs = np.argwhere(dist_matrix < 0.15)
    for i, j in pairs:
        age_a = tracks_a[i].frame_id - tracks_a[i].start_frame
        age_b = tracks_b[j].frame_id - tracks_b[j].start_frame
        if age_a > age_b:
            remove_b.add(int(j))
        else:
            remove_a.add(int(i))

    filtered_a = [t for idx, t in enumerate(tracks_a) if idx not in remove_a]
    filtered_b = [t for idx, t in enumerate(tracks_b) if idx not in remove_b]
    return filtered_a, filtered_b


def remove_intra_duplicates(
    tracks: list[STrack],
    dist_thresh: float = 0.30,
) -> list[STrack]:
    """Remove duplicate tracks within a single list.

    When two tracks in the same list have IoU distance below ``dist_thresh``
    (i.e., IoU above ``1 - dist_thresh``), the shorter-lived track is removed.

    This handles the case where YOLO NMS lets two boxes through for one
    object and both land in ``_tracked`` (which cross-list dedup cannot catch).

    Args:
        tracks: List of active tracks to deduplicate.
        dist_thresh: IoU distance below which two tracks are considered
            duplicates.  Default 0.30 (IoU > 0.70).

    Returns:
        Filtered list with duplicates removed.
    """
    if len(tracks) < 2:
        return tracks

    boxes = np.array([t.predicted_xyxy for t in tracks], dtype=np.float64)
    ious = iou_batch(boxes, boxes)
    n = len(tracks)

    remove: set[int] = set()
    for i in range(n):
        if i in remove:
            continue
        for j in range(i + 1, n):
            if j in remove:
                continue
            if (1.0 - ious[i, j]) < dist_thresh:
                age_i = tracks[i].frame_id - tracks[i].start_frame
                age_j = tracks[j].frame_id - tracks[j].start_frame
                if age_i >= age_j:
                    remove.add(j)
                else:
                    remove.add(i)

    if not remove:
        return tracks
    return [t for idx, t in enumerate(tracks) if idx not in remove]


def fuse_score(
    cost_matrix: NDArray[np.float64],
    detection_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fuse detection confidence scores into IoU cost matrix.

    Penalizes low-confidence detections by scaling the IoU similarity
    by the detection score: ``cost = 1 - (1 - iou_cost) * score``.

    Args:
        cost_matrix: (N, M) IoU distance matrix.
        detection_scores: (M,) detection confidence scores.

    Returns:
        (N, M) fused cost matrix.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    sim = 1.0 - cost_matrix
    scores = detection_scores[np.newaxis, :]
    fused_sim = sim * scores
    return 1.0 - fused_sim


__all__ = [
    "appearance_distance",
    "fuse_score",
    "gated_fused_cost",
    "iou_batch",
    "iou_distance",
    "linear_assignment",
    "needs_reid",
    "remove_duplicate_tracks",
    "remove_intra_duplicates",
]
