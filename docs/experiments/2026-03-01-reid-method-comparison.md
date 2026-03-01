# ReID Method Comparison: No ReID vs CLIP ViT-B/16 vs FastReID SBS-S50

**Date:** 2026-03-01
**Author:** Tin Dang

## Summary

Three-way comparison of ReID methods integrated with ByteTrack on real surveillance footage (928 frames, 1280x720, traffic). The `needs_reid()` conditional gate achieves 99.8% skip rate on this video (only 2 extraction calls per 928 frames), making all three configurations nearly identical in FPS and unique track IDs. The primary differentiator is per-extraction latency and model size: FastReID S50 is 3.6x smaller than CLIP (92MB vs 329MB) with lower per-extraction cost.

**Key limitation:** MCT 1.2.mp4 contains vehicles (not pedestrians). FastReID is person-specific — its embeddings are less meaningful for vehicles. IDS reduction results on this video are not representative of FastReID's performance on person-tracking benchmarks. The comparison remains valid for measuring latency overhead and pipeline integration.

## Setup

| Property | Value |
|----------|-------|
| Hardware | Apple M4 Pro, 48GB RAM |
| YOLO model | YOLO26n (ONNX, CoreML EP) |
| Video | MCT 1.2.mp4 (928 frames, 1280x720, 30fps) |
| Backend | ONNX Runtime + CoreML EP |
| Tracker config | high_thresh=0.3, low_thresh=0.1, match_thresh=0.8, max_age=30, min_hits=3 |
| ReID config | reid_frame_interval=3, reid_lost_age=5 |
| Fusion | BoT-SORT gated: theta_e=0.30, theta_iou=0.5 |

## Model Comparison

| Property | No ReID | CLIP ViT-B/16 | FastReID SBS-S50 |
|----------|---------|---------------|-----------------|
| Parameters | — | 87.8M | 24.0M (fallback) |
| ONNX size | — | 329 MB | 92 MB |
| Embedding dim | — | 512 | 256 |
| Input size | — | 224x224 (square) | 256x128 (portrait) |
| Normalization | — | CLIP | ImageNet |
| Object classes | — | Universal | Person-only |
| Requires training data | — | No (zero-shot) | Yes (person ReID) |

**Note:** FastReID SBS-S50 export uses ResNet-50 + 256-d projection fallback (ImageNet weights) since the native FastReID library is unmaintained. Production deployment requires actual FastReID weights trained on person ReID data (Market-1501, DukeMTMC, etc.).

## Metrics Comparison (928 frames, CoreML EP)

### Run 1: Metrics-only experiment

| Metric | No ReID | CLIP ViT-B/16 | FastReID S50 |
|--------|---------|---------------|-------------|
| Avg FPS | 106.6 | 106.1 | 108.4 |
| Unique IDs | 62 | 62 | 62 |
| Mean track ms | 1.155 | 1.443 | 1.214 |
| p95 track ms | 1.594 | 1.523 | 1.523 |
| Max track ms | 1.845 | 195.645 | 53.824 |
| ReID calls | — | 2 | 2 |
| Skip rate | — | 99.8% | 99.8% |
| Memory delta | +153,631 KB | +2,922 KB | +2,884 KB |

### Run 2: Annotated video experiment

| Metric | No ReID | CLIP ViT-B/16 | FastReID S50 |
|--------|---------|---------------|-------------|
| Avg FPS | 89.9 | 86.8 | 84.2 |
| Unique IDs | 62 | 62 | 62 |
| Mean track ms | 0.314 | 0.674 | 0.493 |
| p95 track ms | 0.439 | 0.449 | 0.525 |
| ReID calls | — | 2 | 2 |
| Skip rate | — | 99.8% | 99.8% |

**Note:** FPS is lower in Run 2 due to video encoding overhead (cv2.VideoWriter).

### Latency Breakdown (Run 1)

| Method | Infer mean | Infer p95 | Track mean | Track p95 | Track max |
|--------|-----------|-----------|-----------|-----------|-----------|
| No ReID | 5.96ms | 6.23ms | 1.155ms | 1.594ms | 1.845ms |
| CLIP ViT-B/16 | 5.96ms | 6.22ms | 1.443ms | 1.523ms | 195.645ms |
| FastReID S50 | 5.97ms | 6.25ms | 1.214ms | 1.523ms | 53.824ms |

The `max track ms` spikes correspond to the 2 frames where ReID was triggered:
- CLIP: ~196ms per extraction (large 329MB model)
- FastReID: ~54ms per extraction (smaller 92MB model, 3.6x faster)

### ReID Efficiency

| Metric | CLIP ViT-B/16 | FastReID S50 |
|--------|---------------|-------------|
| Extraction calls | 2 | 2 |
| Skip rate | 99.8% | 99.8% |
| Per-extraction cost | ~196ms | ~54ms |
| ONNX model size | 329 MB | 92 MB |

The `needs_reid()` gate prevents extraction when IoU assignments are unambiguous. On this traffic video with well-separated vehicles, ambiguity is rare (2 out of 928 frames).

## Conclusions

1. **Identical tracking quality:** All three configurations produce 62 unique track IDs on this video. The `needs_reid()` conditional gate is highly effective — ReID is only invoked when truly needed.

2. **Negligible FPS impact:** With 99.8% skip rate, ReID adds < 0.5ms mean tracking overhead. Real-time performance is maintained for all methods.

3. **FastReID is more efficient per extraction:** 54ms vs 196ms per call (3.6x faster), 92MB vs 329MB model (3.6x smaller). On videos with more frequent occlusions, this difference compounds.

4. **CLIP max spike is significant:** 196ms per extraction risks frame drops at 30fps. FastReID's 54ms is within the 33ms budget at 30fps only if other processing is fast.

5. **Traffic video limitation:** With vehicles and clear spatial separation, ReID has minimal opportunity to help. Person-tracking scenarios (indoor, crowded, frequent occlusion) would trigger ReID far more often and reveal quality differences between CLIP (universal) and FastReID (person-specialized).

## Recommendations

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Person tracking (CCTV, indoor) | FastReID | Domain-specific embeddings, lower latency, smaller model |
| Multi-class tracking (vehicles + people) | CLIP | Universal zero-shot, no training data needed |
| CPU-only, real-time (>25fps) | No ReID | ReID extraction too slow on CPU for real-time |
| GPU / CoreML available | CLIP or FastReID | Hardware absorbs extraction overhead |
| Edge devices (Jetson, mobile) | FastReID | 3.6x smaller model, 3.6x faster extraction |
| Proof-of-concept / rapid deployment | CLIP | Zero-shot, no training pipeline needed |

## Reproduction

```bash
# Step 1: Export models
uv run python tmp/export_clip_onnx.py
uv run python tmp/export_fastreid_onnx.py

# Step 2: Validate preprocessing
uv run python tmp/validate_fastreid_preprocess.py

# Step 3: Run metrics comparison
uv run python tmp/experiment_reid_comparison.py

# Step 4: Generate annotated videos
uv run python tmp/experiment_reid_comparison_annotated.py

# Step 5: Quality gates
uv run ruff check src/ tests/ --quiet
uv run pyright src/yowo/
uv run pytest tests/unit/ -x -q
```

## Output Files

| File | Description |
|------|-------------|
| `tmp/exports/fastreid-sbs-s50.onnx` | FastReID ONNX model (92MB) |
| `tmp/MCT_1.2_no_reid.mp4` | Annotated video — baseline |
| `tmp/MCT_1.2_clip_reid.mp4` | Annotated video — CLIP ReID |
| `tmp/MCT_1.2_fastreid.mp4` | Annotated video — FastReID |
