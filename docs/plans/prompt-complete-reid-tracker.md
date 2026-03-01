# Prompt: Complete ReID Tracker Features — Unconfirmed Stage + Score Fusion

You are a Senior Multi-Object Tracking Engineer with 15+ years specializing in production ByteTrack/BoT-SORT systems. You have deep expertise in:
- ByteTrack two-stage association: high-conf IoU → low-conf IoU → unconfirmed re-association
- BoT-SORT cost fusion: detection score weighting, appearance gating, min-cost selection
- Kalman filter lifecycle: NEW → TRACKED → LOST → REMOVED state machines
- Python 3.12+, strict pyright, Protocol-based interfaces, frozen dataclasses
- Production CCTV constraints: fixed cameras, 24/7 stability, crowded scenes

## Stakes (P6)

This closes the final two discrepancies found in the ultralytics validation audit (2026-03-01). D-5 (unconfirmed track re-association) is HIGH priority — without it, newly born tracks that briefly fail to match are immediately lost, causing ID fragmentation in crowded scenes. D-1 (score fusion) is MEDIUM priority — without it, low-confidence ghost detections can steal associations from legitimate tracks. I'll tip you $200 for a fully working, backward-compatible, tested implementation.

---

## Context: What Exists

### Current Tracker Architecture (all 1226 tests pass, 0 pyright errors)

```
src/yowo/tracking/
├── __init__.py       (96 lines)  — exports: ByteTracker, CLIPExtractor, FastReIDExtractor, ReIDExtractor, track_stream, track_detections
├── _tracker.py       (445 lines) — ByteTracker with 3-stage association + ReID
├── _strack.py        (340 lines) — STrack with _embedding, output_xyxy hybrid
├── _matching.py      (460 lines) — iou_batch, iou_distance, linear_assignment, appearance_distance, gated_fused_cost, needs_reid, remove_duplicate_tracks, remove_intra_duplicates
├── _kalman.py        (195 lines) — KalmanFilterXYAH — NO CHANGES
├── _reid.py          (400 lines) — ReIDExtractor Protocol + CLIPExtractor + FastReIDExtractor
```

### Validation Report Findings (from `docs/experiments/2026-03-01-tracker-ultralytics-validation.md`)

**D-5 (HIGH): No unconfirmed track re-association stage**

ultralytics BYTETracker separates tracks into `unconfirmed` (not yet activated) and `tracked` (confirmed). After Stage 1+2, unconfirmed tracks get a second chance to match remaining high-conf detections at thresh=0.7. yowo skips this — newly born tracks that miss in the next frame immediately go to LOST.

**D-1 (MEDIUM): No `fuse_score` function**

ultralytics fuses detection confidence into the IoU cost matrix: `cost = 1 - (1 - iou_cost) * detection_score`. This penalizes low-confidence detections, reducing false matches. yowo uses raw IoU cost only.

### Current ByteTracker.update() Flow (simplified)

```python
def update(self, detection):
    # Split detections by confidence
    high_boxes, low_boxes = split_by_threshold(detection)

    # Kalman predict all tracks
    for track in self._tracked + self._lost:
        track.predict()

    # STAGE 1: high-conf dets → tracked + lost pool (IoU + optional ReID)
    strack_pool = self._tracked + self._lost
    cost1 = iou_distance(strack_pool, high_arr)
    # ... conditional ReID fusion ...
    matches1, unmatched_tracks, unmatched_dets = linear_assignment(cost1, match_thresh)

    # STAGE 2: low-conf dets → unmatched TRACKED tracks (IoU only)
    cost2 = iou_distance(unmatched_tracked, low_arr)
    matches2, still_unmatched, _ = linear_assignment(cost2, 0.5)

    # Mark remaining unmatched tracked → LOST

    # STAGE 3: appearance rescue for long-lost tracks (ReID only)
    # ... appearance_rescue() ...

    # NEW TRACKS from unmatched high-conf dets
    for idx in unmatched_high_idxs:
        if conf >= new_track_thresh:
            track = STrack(...)
            track.activate(frame_id)
            self._tracked.append(track)

    # Duplicate removal + output
```

### ultralytics Unconfirmed Stage (reference implementation)

```python
# In BYTETracker.update():
# Before Stage 1, separate unconfirmed from tracked:
unconfirmed = []
tracked_stracks = []
for track in self.tracked_stracks:
    if not track.is_activated:
        unconfirmed.append(track)
    else:
        tracked_stracks.append(track)

# Stage 1+2 use tracked_stracks (not unconfirmed)
# ...

# After Stage 2, match unconfirmed to remaining high-conf dets:
detections = [detections[i] for i in u_detection]  # remaining unmatched dets
dists = self.get_dists(unconfirmed, detections)
matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
for itracked, idet in matches:
    unconfirmed[itracked].update(detections[idet], self.frame_id)
    activated_stracks.append(unconfirmed[itracked])
for it in u_unconfirmed:
    track = unconfirmed[it]
    track.mark_removed()  # unconfirmed tracks that fail → REMOVED (not LOST)
    removed_stracks.append(track)
```

### ultralytics `fuse_score` (reference implementation)

```python
def fuse_score(cost_matrix, detections):
    """Fuse detection confidence into IoU cost matrix."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix           # convert cost → similarity
    det_scores = np.array([d.score for d in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores      # multiply by confidence
    return 1 - fuse_sim                  # convert back to cost
```

### Key STrack Properties

```python
class STrack:
    # Track is "confirmed" when hits >= min_hits AND state == TRACKED
    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self._min_hits and self.state == TrackState.TRACKED

    # Track lifecycle:
    # - __init__: state=NEW, hits=0
    # - activate(): state=TRACKED, hits=1
    # - update(): state=TRACKED, hits+=1
    # - mark_lost(): state=LOST
    # - mark_removed(): state=REMOVED
```

---

## Task Decomposition (P3)

Take a deep breath and work through this step by step.

### Step 1: Add `fuse_score()` to `_matching.py`

Add a new function after `iou_distance()`:

```python
def fuse_score(
    cost_matrix: NDArray[np.float64],
    detection_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fuse detection confidence scores into IoU cost matrix.

    Penalizes low-confidence detections by scaling the IoU similarity
    by the detection score: cost = 1 - (1 - iou_cost) * score.

    Args:
        cost_matrix: (N, M) IoU distance matrix.
        detection_scores: (M,) detection confidence scores.

    Returns:
        (N, M) fused cost matrix.
    """
```

**Implementation:**
- Convert cost → similarity: `sim = 1 - cost_matrix`
- Broadcast scores: `scores = detection_scores[np.newaxis, :]`
- Fuse: `fused_sim = sim * scores`
- Convert back: `return 1 - fused_sim`
- Handle empty: if `cost_matrix.size == 0`, return as-is

**Update `__all__` in `_matching.py`** to include `fuse_score`.

### Step 2: Add `fuse_score` parameter to `ByteTracker`

Add `fuse_score: bool = False` parameter to `ByteTracker.__init__()`:

```python
def __init__(
    self,
    *,
    track_high_thresh: float = 0.6,
    track_low_thresh: float = 0.1,
    match_thresh: float = 0.8,
    max_age: int = 30,
    min_hits: int = 3,
    fuse_score: bool = False,  # NEW
    reid_extractor: ReIDExtractor | None = None,
    reid_lost_age: int = 5,
    reid_frame_interval: int = 3,
) -> None:
```

Store as `self._fuse_score = fuse_score`.

In `update()`, after computing `cost1 = iou_distance(strack_pool, high_arr)`, conditionally apply:

```python
if self._fuse_score and high_confs:
    from yowo.tracking._matching import fuse_score as _fuse_score
    cost1 = _fuse_score(cost1, np.array(high_confs, dtype=np.float64))
```

**Note:** Apply fuse_score BEFORE ReID fusion (same order as ultralytics).

### Step 3: Add unconfirmed track re-association stage

This is the core change. Modify `ByteTracker.update()`:

**3a. Separate unconfirmed from confirmed tracks before Stage 1**

Currently `strack_pool = self._tracked + self._lost`. Change to:

```python
# Separate unconfirmed (recently born, not yet confirmed) from confirmed
unconfirmed: list[STrack] = []
confirmed_tracked: list[STrack] = []
for track in self._tracked:
    if not track.is_confirmed:
        unconfirmed.append(track)
    else:
        confirmed_tracked.append(track)

# Stage 1 pool: confirmed tracked + lost (NOT unconfirmed)
strack_pool = confirmed_tracked + self._lost
```

**3b. After Stage 2 (and before new track birth), add unconfirmed stage**

After Stage 2 marks unmatched tracked as lost, and after Stage 3 (appearance rescue), add:

```python
# --- STAGE 4: Re-associate unconfirmed tracks with remaining high-conf dets ---
# Unconfirmed tracks get a second chance to match before being removed.
remaining_high_idxs = [
    i for i in unmatched_high_idxs if i not in rescued_det_idxs
]
if unconfirmed and remaining_high_idxs:
    remaining_high_arr = np.array(
        [high_boxes[i] for i in remaining_high_idxs], dtype=np.float64
    )
    cost_unconf = iou_distance(unconfirmed, remaining_high_arr)
    if self._fuse_score:
        remaining_scores = np.array(
            [high_confs[i] for i in remaining_high_idxs], dtype=np.float64
        )
        cost_unconf = _fuse_score(cost_unconf, remaining_scores)
    matches_unconf, unmatched_unconf_idxs, unmatched_rem_idxs = linear_assignment(
        cost_unconf, 0.7  # ultralytics uses 0.7 for unconfirmed
    )
    for ti, di in matches_unconf:
        orig_di = remaining_high_idxs[di]
        unconfirmed[ti].update(
            high_boxes[orig_di], high_confs[orig_di],
            high_cls_ids[orig_di], high_cls_names[orig_di], frame_id,
        )
        # Update embedding if available
        if det_embeddings is not None:
            emb = det_embeddings[orig_di]
            if float(np.linalg.norm(emb)) > 1e-8:
                unconfirmed[ti].update_embedding(emb)
        rescued_det_idxs.add(orig_di)  # consumed — don't birth new track
    for idx in unmatched_unconf_idxs:
        # Unconfirmed tracks that fail to re-associate → REMOVED (not LOST)
        unconfirmed[idx].mark_removed()
else:
    # No remaining dets → all unconfirmed tracks are removed
    for track in unconfirmed:
        track.mark_removed()
```

**3c. Fix the tracked list rebuild**

After the unconfirmed stage, the `_tracked` list rebuild needs to include successfully matched unconfirmed tracks:

```python
# Rebuild active tracked list:
# Keep confirmed TRACKED + refound + successfully matched unconfirmed
matched_unconf = [unconfirmed[ti] for ti, _ in matches_unconf] if unconfirmed and remaining_high_idxs else []
self._tracked = (
    [t for t in confirmed_tracked if t.state == TrackState.TRACKED]
    + refound
    + matched_unconf
)
```

**Critical:** Unconfirmed tracks that fail are `mark_removed()`, NOT `mark_lost()`. This matches ultralytics behavior — unconfirmed tracks don't get a third chance in the lost pool.

### Step 4: Add unit tests

Add tests to `tests/unit/test_tracking.py`:

**4a. `fuse_score` tests:**
- `test_fuse_score_identity_at_score_one` — scores all 1.0 → cost unchanged
- `test_fuse_score_zeros_at_score_zero` — scores all 0.0 → cost all 1.0
- `test_fuse_score_scales_correctly` — verify formula `1 - (1-cost) * score`
- `test_fuse_score_empty_matrix` — empty → empty
- `test_tracker_fuse_score_flag` — ByteTracker with `fuse_score=True` doesn't crash

**4b. Unconfirmed stage tests:**
- `test_unconfirmed_track_re_associates` — born track misses frame 2, re-associates frame 3 at thresh=0.7
- `test_unconfirmed_track_removed_on_fail` — born track fails re-association → REMOVED (not LOST)
- `test_confirmed_tracks_bypass_unconfirmed_stage` — tracks with hits >= min_hits are in strack_pool, not unconfirmed list
- `test_unconfirmed_stage_backward_compatible` — default behavior (min_hits=3) with standard detections produces same results as before

**4c. Integration test:**
- `test_crowded_scene_unconfirmed_helps` — simulate 2 objects entering same area, verify unconfirmed stage reduces ID fragmentation vs previous behavior

### Step 5: Run quality gates and verify backward compatibility

```bash
uv run ruff check src/ tests/ --quiet
uv run pyright src/yowo/
uv run pytest tests/unit/ -x -q
```

Verify:
1. All existing 1226 tests still pass
2. New tests pass
3. Default behavior (fuse_score=False) produces identical results to before
4. Annotated video experiment still produces same numbers (run `tmp/experiment_reid_comparison.py --mode baseline`)

---

## Constraints (Non-Negotiable)

1. **Backward compatible**: `fuse_score=False` default, unconfirmed stage changes internal flow but same observable behavior for `min_hits=1`
2. **No modifications to _matching.py beyond adding fuse_score()** — existing functions unchanged
3. **No modifications to _strack.py, _kalman.py, _reid.py** — only _tracker.py and _matching.py get production changes
4. **File sizes**: `_tracker.py` stays under 550 lines, `_matching.py` under 500 lines
5. **Quality gates must pass**: ruff + pyright + pytest all green
6. **`__init__.py` exports updated** if new public functions added

---

## Chain-of-Thought Guidance (P12, P19)

- **fuse_score placement**: Apply fuse_score to `cost1` BEFORE ReID fusion. This ensures that when `needs_reid()` checks cost ambiguity, it sees the score-fused costs. ultralytics applies fuse_score inside `get_dists()` before embedding fusion.

- **Unconfirmed separation**: The key insight is that `is_confirmed` already checks `hits >= min_hits`. A track with `hits=1` (just activated) is NOT confirmed. These tracks should be separated BEFORE Stage 1 so they don't compete with established tracks for detections.

- **REMOVED vs LOST for unconfirmed**: ultralytics removes unconfirmed failures outright — they never enter the lost pool. This prevents phantom tracks from accumulating. yowo should match this behavior.

- **Stage ordering**: The correct order is:
  1. Stage 1: high-conf → confirmed tracked + lost (IoU + optional ReID + optional fuse_score)
  2. Stage 2: low-conf → unmatched confirmed tracked (IoU only, thresh=0.5)
  3. Stage 3: appearance rescue for long-lost tracks (ReID only)
  4. **Stage 4 (NEW):** remaining high-conf → unconfirmed tracks (IoU, thresh=0.7)
  5. Birth new tracks from remaining unmatched high-conf dets

- **Edge case — min_hits=1**: When `min_hits=1`, a track becomes confirmed on its first hit. The unconfirmed list will always be empty (since `activate()` sets `hits=1` and `state=TRACKED`). This means **no behavioral change** for `min_hits=1` — backward compatible by design.

- **Edge case — min_hits=3 (default)**: Tracks with hits=1 or hits=2 are unconfirmed. They get separated before Stage 1, given a second chance in Stage 4 at thresh=0.7, and removed if they fail. This improves tracking in crowded scenes where brief occlusions happen right after track birth.

- **fuse_score impact on needs_reid**: With fuse_score=True, low-confidence detections get higher costs. This may cause `needs_reid()` to trigger less often (fewer ambiguous assignments). This is correct behavior — if a detection is low-confidence, ReID shouldn't spend compute on it.

---

## Self-Evaluation Framework (P15)

After completing all steps, verify:

1. **fuse_score formula correct**: `1 - (1-cost) * score` matches ultralytics
2. **Unconfirmed stage correctly separates**: `is_confirmed` is False for hits < min_hits
3. **REMOVED not LOST**: Unconfirmed failures are mark_removed(), not mark_lost()
4. **Stage ordering correct**: fuse_score → ReID → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Birth
5. **Backward compatible**: Default params produce same results as before
6. **All tests pass**: Existing 1226 + new tests
7. **No regressions**: Baseline experiment numbers unchanged

Rate confidence (0-1) on each. If any < 0.9, fix before presenting.

---

## Reference Documents

| Document | Location |
|----------|----------|
| Validation report | `docs/experiments/2026-03-01-tracker-ultralytics-validation.md` |
| Current tracker | `src/yowo/tracking/_tracker.py` |
| Matching module | `src/yowo/tracking/_matching.py` |
| STrack | `src/yowo/tracking/_strack.py` |
| Tests | `tests/unit/test_tracking.py` |
| ultralytics byte_tracker.py | `github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py` |
| ultralytics bot_sort.py | `github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py` |
| ultralytics matching.py | `github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/matching.py` |
| 3-way comparison | `docs/experiments/2026-03-01-reid-method-comparison.md` |
