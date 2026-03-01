# VeRi-776 Cross-Camera Vehicle ReID Benchmark

**Date:** 2026-03-01
**Author:** Tin Dang
**Dataset:** VeRi-776 (Liu et al., ICME 2016)
**Hardware:** Apple M4 Pro, ONNX Runtime (CoreML EP)

## Overview

Evaluates yowo's ReID extractors on VeRi-776, the standard vehicle re-identification benchmark. VeRi-776 contains 776 vehicles captured across 20 real-world traffic cameras with varying viewpoints, lighting, and occlusion conditions. This experiment measures how well yowo's current ReID models can match the same vehicle across different camera views — the core capability needed for `CrossCameraTracker`.

## Dataset

| Split | Images | Vehicles | Cameras |
|-------|--------|----------|---------|
| Query | 1,678 | 200 | 19 |
| Gallery (test) | 11,579 | 200 | 19 |
| Train | 37,778 | 576 | 20 |

- **Naming:** `{vehicleID}_c{cameraID}_{timestamp}_{flag}.jpg`
- **Images:** Pre-cropped vehicle patches (no detection needed), variable resolution (125x129 to 411x299)
- **Average:** 57.9 images/vehicle, 8.4 cameras/vehicle, max 18 cameras for one vehicle
- **Protocol:** For each query, rank all gallery images excluding same-camera (junk) images

## Extractors Tested

| Extractor | Architecture | Dim | Input Size | Training | Domain |
|-----------|-------------|-----|------------|----------|--------|
| **CLIP ViT-B/16** | Vision Transformer | 512 | 224x224 | LAION-2B (400M image-text pairs) | Zero-shot general |
| **FastReID SBS-S50** | ResNet50 (SBS) | 256 | 256x128 (portrait) | Person ReID datasets | Person-specific |

## Results

### Retrieval Metrics

| Extractor | Rank-1 | Rank-5 | Rank-10 | mAP | Speed (img/s) |
|-----------|--------|--------|---------|-----|---------------|
| **CLIP ViT-B/16** | **31.82%** | **52.98%** | **63.41%** | **9.32%** | 27 |
| **FastReID SBS-S50** | 30.21% | 48.15% | 57.21% | 8.43% | **89** (3.3x) |
| VeRi-776 SOTA (reference) | ~97% | ~99% | ~99% | ~80% | — |

### Viewpoint Analysis (CLIP)

**Intra-camera vs cross-camera embedding similarity:**

| Comparison | Mean Cosine Similarity | Std |
|------------|----------------------|-----|
| Same camera (same vehicle) | **0.9157** | 0.0524 |
| Cross camera (same vehicle) | **0.8656** | 0.0527 |
| **Viewpoint gap** | **0.0501** | — |

**Hardest camera pairs** (viewpoint change degrades similarity most):

| Camera Pair | Mean Similarity | Samples |
|-------------|----------------|---------|
| c005-c018 | 0.7612 | 49 |
| c007-c018 | 0.8204 | 49 |
| c002-c018 | 0.8231 | 182 |
| c005-c016 | 0.8239 | 683 |
| c007-c014 | 0.8255 | 636 |

**Easiest camera pairs** (similar viewpoint):

| Camera Pair | Mean Similarity | Samples |
|-------------|----------------|---------|
| c003-c010 | 0.9059 | 6,557 |
| c013-c019 | 0.9028 | 5,110 |
| c001-c018 | 0.9006 | 182 |
| c015-c018 | 0.8959 | 329 |
| c016-c018 | 0.8952 | 378 |

Camera c018 appears in both hardest and easiest pairs — it likely has a unique viewpoint that pairs well with some cameras (c001, c015, c016) but poorly with others (c005, c007).

### Threshold Sweep (CLIP — for EmbeddingGallery tuning)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|-----------|-----------|--------|-----|-----|-----|-----|
| **0.10** | 0.0186 | 0.2930 | **0.0350** | 31,386 | 1,656,718 | 75,720 |
| 0.20 | 0.0080 | 0.8959 | 0.0160 | 95,956 | 11,824,922 | 11,150 |
| 0.30 | 0.0060 | 0.9886 | 0.0119 | 105,881 | 17,618,692 | 1,225 |
| 0.40 | 0.0055 | 0.9999 | 0.0110 | 107,092 | 19,208,075 | 14 |
| 0.50 | 0.0055 | 1.0000 | 0.0110 | 107,106 | 19,310,416 | 0 |

Best F1 at threshold 0.10 — CLIP embeddings are too clustered to separate individual vehicles from same-type vehicles. The precision is extremely low at all thresholds because many different vehicles of the same color/type produce high similarity.

## Visual Results

Visual grids saved to `tmp/veri_results/`:
- `veri_reid_CLIP_ViT-B-16.jpg` — 30 queries × 10 matches
- `veri_reid_FastReID_SBS-S50.jpg` — 30 queries × 10 matches

**Format:** Each row = `[Blue: Query] → [Green: correct match | Red: wrong match]`

**Key visual observations:**

1. **Distinctive vehicles succeed:** Yellow buses, red trucks, uniquely colored vehicles get many green (correct) matches across cameras.

2. **Common colors fail:** White/silver sedans (the majority) produce mostly red (wrong) matches — CLIP matches "white sedan" as a concept, not the specific vehicle.

3. **FastReID portrait distortion:** Vehicle crops are forced into 256x128 portrait aspect ratio, losing spatial detail. Vehicles are naturally landscape (~1.5:1), so portrait compression (2:1) discards discriminative width information.

4. **Color is the dominant feature:** Both models primarily match by color → type → rough shape. Fine-grained features (license plate, minor body differences) are not captured at 224x224 / 256x128 resolution.

## Analysis

### Why CLIP beats person-trained FastReID on vehicles

1. **Aspect ratio mismatch:** FastReID's 256x128 portrait input is designed for standing humans (2:1 tall). Vehicles are landscape (~1.5:1 wide). Resizing distorts spatial relationships.

2. **Semantic understanding:** CLIP's vision-language pretraining gives it concept-level vehicle understanding ("yellow bus", "red sedan"). FastReID only learned texture patterns for person discrimination.

3. **Embedding dimension:** CLIP's 512-dim has 2x the capacity of FastReID's 256-dim, potentially encoding more discriminative features.

### Why both are far from SOTA

VeRi-776 SOTA uses models specifically trained on vehicle datasets:

| Aspect | Zero-shot (CLIP/FastReID) | VeRi-trained |
|--------|--------------------------|-------------|
| Training data | General/person images | VeRi-776 train (37K vehicle images) |
| Input resolution | 224x224 / 256x128 | Often 256x256 or higher |
| Features learned | Color, type, rough shape | License plate, fine body detail, viewpoint-invariant features |
| Rank-1 | ~31% | ~97% |
| mAP | ~9% | ~80% |

The 60+ point gap in Rank-1 demonstrates that **vehicle-specific training is essential for production cross-camera ReID**.

### Viewpoint gap interpretation

The 0.0501 viewpoint gap (same-camera: 0.9157, cross-camera: 0.8656) means:
- CLIP embeddings are **relatively viewpoint-robust** — only 5% similarity drop across cameras
- But the **inter-vehicle gap is even smaller** — different vehicles of the same type have similarity ~0.85+, overlapping with cross-camera same-vehicle similarity
- This means CLIP cannot reliably distinguish "same vehicle, different camera" from "different vehicle, same type" without extremely tight thresholds

## Implications for yowo CrossCameraTracker

### EmbeddingGallery threshold recommendations

| Extractor | Recommended threshold | Rationale |
|-----------|----------------------|-----------|
| CLIP (current) | 0.08-0.12 | Very tight — accepts only near-identical embeddings |
| VeRi-trained (future) | 0.30-0.40 | Wider — trained embeddings have better inter-vehicle separation |

### Deployment scenarios

| Scenario | CLIP sufficient? | Recommendation |
|----------|-----------------|----------------|
| Low-density traffic (rural road, highway) | Yes | Few vehicles of same type → CLIP works |
| Medium-density (suburban intersection) | Marginal | Color-unique vehicles match; common colors fail |
| High-density (urban, many white sedans) | No | Needs VeRi-trained model |
| Multi-type (buses, trucks, cars mixed) | Yes | Type diversity = high CLIP discrimination |

### Next steps

1. **VehicleReIDExtractor:** Train or fine-tune a ResNet50 on VeRi-776 train set, export to ONNX. Expected: Rank-1 ~90%+, mAP ~70%+.

2. **Input aspect ratio:** Vehicle ReID should use square or landscape input (224x224 or 256x256), not portrait.

3. **Gallery threshold auto-tuning:** Use the threshold sweep data to auto-select threshold based on extractor type.

4. **License plate feature:** For high-density same-type scenarios, combine appearance ReID with license plate OCR as secondary discriminator.

## Reproduction

```bash
# Full benchmark (both extractors, all analyses)
uv run python tmp/experiment_veri_reid.py --extractor all

# CLIP only, fast (skip viewpoint + sweep)
uv run python tmp/experiment_veri_reid.py --extractor clip --skip-viewpoint --skip-sweep

# Limited queries for quick test
uv run python tmp/experiment_veri_reid.py --max-query 100
```

**Required files:**
- VeRi dataset at `/Users/tindang/Downloads/VeRi/`
- `tmp/exports/clip-vit-b16-visual.onnx` (CLIP visual encoder)
- `tmp/exports/fastreid-sbs-s50.onnx` (FastReID SBS-S50)

## Sources

- Liu, X., Liu, W., Ma, H., Fu, H. (2016). "Large-scale vehicle re-identification in urban surveillance videos." IEEE ICME.
- VeRi-776 dataset: 776 vehicles, 20 cameras, 49,357 images
- CLIP: Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision."
- FastReID: He et al. (2020). "FastReID: A Pytorch Toolbox for General Instance Re-identification."
