# VeRi-776 Cross-Camera Vehicle ReID Benchmark

**Date:** 2026-03-01
**Author:** Tin Dang
**Dataset:** VeRi-776 (200 vehicles, 19 cameras, 11,579 gallery images, 1,678 queries)

## 1. Overview

Evaluated yowo's ReID extractors on the VeRi-776 benchmark to measure cross-camera vehicle re-identification accuracy. VeRi-776 is the standard benchmark for vehicle ReID in urban traffic surveillance — 20 cameras covering real intersections with diverse viewpoints, lighting, and occlusion.

**Goal:** Establish baseline performance and identify the path to production-grade cross-camera vehicle matching for yowo's `CrossCameraTracker`.

---

## 2. Experiment Setup

### Dataset
- **Train:** 37,778 images, 576 vehicles (not used — zero-shot evaluation)
- **Test/Gallery:** 11,579 images, 200 vehicles, 19 cameras
- **Query:** 1,678 images, 200 vehicles, 19 cameras
- **Protocol:** For each query, rank all gallery images excluding same-camera (junk) images
- **Avg cameras/vehicle:** 8.4 | **Max cameras/vehicle:** 18

### Extractors Tested
| Extractor | Dim | Input Size | Training | Speed |
|---|---|---|---|---|
| CLIP ViT-B/16 | 512 | 224×224 | Zero-shot (no vehicle training) | 27 img/s |
| FastReID SBS-S50 | 256 | 256×128 | Person ReID (Market-1501) | 89 img/s |
| **CLIP-ReID ViT-B/16** | **1280** | **256×256** | **VeRi-776 fine-tuned (prompt learning)** | **15.5 img/s** |

### Metrics
- **Rank-K:** % of queries where correct vehicle appears in top-K results
- **mAP:** Mean Average Precision (standard ReID metric)
- All embeddings L2-normalized, cosine distance = `1 - dot(query, gallery)`

---

## 3. Results

### 3.1 Retrieval Accuracy

| Extractor | Rank-1 | Rank-5 | Rank-10 | mAP | Speed |
|---|---|---|---|---|---|
| **CLIP-ReID VeRi** | **96.66%** | **98.45%** | **98.93%** | **82.28%** | 15.5 img/s |
| CLIP ViT-B/16 | 31.82% | 52.98% | 63.41% | 9.32% | 27 img/s |
| FastReID SBS-S50 | 30.21% | 48.15% | 57.21% | 8.43% | **89 img/s** |

**CLIP-ReID vs reference:** mAP=82.28% vs 83.3% reference (-1.02%), Rank-1=96.66% vs 97.4% (-0.74%). Gap due to skipping SIE (camera/view) embeddings — the checkpoint is the stride-12 SIE-OLP variant (reference mAP=84.5%) run without camera labels.

### 3.2 Cross-Camera Viewpoint Analysis (CLIP)

| Metric | Mean Similarity | Std | Count |
|---|---|---|---|
| **Same-camera** | 0.9157 | 0.0524 | 53,300 pairs |
| **Cross-camera** | 0.8656 | 0.0527 | 520,111 pairs |
| **Viewpoint gap** | **0.0501** | — | — |

#### Hardest Camera Pairs (lowest similarity — largest viewpoint change)
| Pair | Mean Sim | Std | Samples |
|---|---|---|---|
| c005-c018 | 0.7612 | 0.0454 | 49 |
| c007-c018 | 0.8204 | 0.0216 | 49 |
| c002-c018 | 0.8231 | 0.0628 | 182 |
| c005-c016 | 0.8239 | 0.0666 | 683 |
| c007-c014 | 0.8255 | 0.0356 | 636 |

#### Easiest Camera Pairs (highest similarity — similar viewpoint)
| Pair | Mean Sim | Std | Samples |
|---|---|---|---|
| c003-c010 | 0.9059 | 0.0485 | 6,557 |
| c013-c019 | 0.9028 | 0.0390 | 5,110 |
| c001-c018 | 0.9006 | 0.0253 | 182 |
| c015-c018 | 0.8959 | 0.0250 | 329 |
| c016-c018 | 0.8952 | 0.0207 | 378 |

### 3.3 Gallery Threshold Sweep (CLIP)

For `EmbeddingGallery` threshold tuning in `CrossCameraTracker`:

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.10 | 0.0186 | 0.2930 | **0.0350** | 31,386 | 1,656,718 | 75,720 |
| 0.20 | 0.0080 | 0.8959 | 0.0160 | 95,956 | 11,824,922 | 11,150 |
| 0.30 | 0.0060 | 0.9886 | 0.0119 | 105,881 | 17,618,692 | 1,225 |
| 0.40 | 0.0055 | 0.9999 | 0.0110 | 107,092 | 19,208,075 | 14 |

**Best F1 at threshold 0.10** — CLIP embeddings cluster too tightly; even the tightest threshold produces many false positives because same-type vehicles (e.g., two white sedans) have cosine distance < 0.10.

### 3.4 Visual Results

Grouped image grids saved to:
- `tmp/veri_results/veri_reid_CLIP_ViT-B-16.jpg` — 30 queries × 10 gallery matches
- `tmp/veri_results/veri_reid_FastReID_SBS-S50.jpg` — 30 queries × 10 gallery matches

Format: `[Blue=Query] [Green=Correct match] [Red=Wrong match]`

**Observations from visual inspection:**
- Distinctive vehicles (yellow buses, red cars, green trucks) → many green matches across cameras
- Common vehicles (white/silver sedans) → mostly red matches — CLIP matches type+color, not individual identity
- FastReID worse on vehicles because it uses portrait crops (256×128) designed for persons

---

## 4. Comparison with State of the Art

### 4.1 VeRi-776 Leaderboard (published results)

| Method | mAP | Rank-1 | Year | Open Weights? | Training |
|---|---|---|---|---|---|
| **CLIP-SENet** | **92.9%** | **98.7%** | 2025 | **No** (paper only) | VeRi train + CLIP fine-tune |
| CLIP-ReID (ViT-SIE-OLP) | ~90% | ~98% | 2023 | **Yes** (GitHub) | VeRi train + prompt learning |
| LKA-ReID | 86.7% | 98.0% | 2024 | Unknown | VeRi train |
| MDFE-Net | ~85% | ~97% | 2024 | Unknown | VeRi train |
| ResNet50-IBN + triplet | ~78% | ~96% | 2020 | Yes (fast-reid) | VeRi train |
| **CLIP-ReID VeRi (ours)** | **82.3%** | **96.7%** | **2026** | **Yes** | **VeRi train (prompt learning)** |
| CLIP ViT-B/16 (ours) | 9.3% | 31.8% | — | Yes | None (zero-shot) |
| FastReID SBS-S50 (ours) | 8.4% | 30.2% | — | Yes | Person ReID only |

### 4.2 CLIP-SENet (SOTA, closed)

**Paper:** [CLIP-SENet: CLIP-based Semantic Enhancement Network for Vehicle Re-identification](https://arxiv.org/html/2502.16815v1) (Feb 2025)

**Architecture:**
- CLIP ViT-B/16 backbone (image encoder)
- Semantic Enhancement Module: extracts vehicle attributes (color, type, viewpoint) from CLIP text encoder in unsupervised manner
- Fine-grained fusion: merges semantic features with appearance features at patch level
- Trained on VeRi-776 train set with ID loss + triplet loss + semantic loss

**Results:**
| Dataset | mAP | Rank-1 |
|---|---|---|
| VeRi-776 | **92.9%** | **98.7%** |
| VehicleID (small) | — | **90.4%** |
| VeRi-Wild | **89.1%** | **97.9%** |

**Why it's best:**
- Uses CLIP's text encoder to generate semantic attribute descriptions without manual labels
- Partial-level alignment captures fine-grained differences (headlights, grille pattern, license plate area)
- Surpasses previous SOTA (MBR) by +1% mAP and +0.5% Rank-1

**Status:** Paper only. No code, no weights, no GitHub repository as of 2026-03-01. Authors have not responded to code release inquiries from the community.

### 4.3 CLIP-ReID (near-SOTA, fully open)

**Paper:** [CLIP-ReID: Exploiting Vision-Language Model for Image Re-identification without Concrete Text Labels](https://arxiv.org/pdf/2211.13977v3) (AAAI 2023)

**GitHub:** [github.com/Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID)

**Architecture:**
- Two-stage training:
  1. Freeze CLIP text + image encoders. Optimize learnable text tokens per vehicle ID
  2. Freeze text encoder. Fine-tune image encoder using learned text features as supervision
- Variants: CNN-baseline, CNN-CLIP-ReID, ViT-baseline, **ViT-CLIP-ReID-SIE-OLP** (strongest)
- SIE = Side Information Embedding (camera ID, viewpoint)
- OLP = Overlapping Patches (local feature extraction)

**Pretrained weights available for VeRi-776:**

| Variant | Architecture | VeRi Weights |
|---|---|---|
| CNN-baseline | ResNet50 | Google Drive |
| CNN-CLIP-ReID | ResNet50 + prompt learning | Google Drive |
| ViT-baseline | ViT-B/16 | Google Drive |
| **ViT-CLIP-ReID-SIE-OLP** | ViT-B/16 + SIE + OLP | Google Drive |

**Integration path for yowo:**
1. Download ViT-CLIP-ReID-SIE-OLP checkpoint from GitHub
2. Load in PyTorch, export to ONNX
3. Plug into `CLIPExtractor` (same 512-dim L2-normalized embeddings)
4. Expected: mAP ~88-90%, Rank-1 ~98% — **10x improvement over current zero-shot**

---

## 5. Gap Analysis

### Why our mAP is 9.3% vs SOTA 92.9%

| Factor | Our approach | SOTA approach | Impact |
|---|---|---|---|
| **Training** | Zero-shot (no VeRi data) | Fine-tuned on VeRi train | **~70% of the gap** |
| **Feature type** | Generic visual (color, shape) | Instance-discriminative (individual vehicle) | Matches type not identity |
| **Local features** | Global embedding only | Part-level patches (headlights, grille) | Misses fine-grained cues |
| **Camera awareness** | None | SIE (camera/viewpoint encoding) | Same vehicle looks different per camera |
| **Re-ranking** | None | k-reciprocal post-processing | Free +5-10% mAP boost |

### Similarity distribution problem

CLIP zero-shot embeddings:
- Same vehicle, same camera: **0.9157** similarity
- Same vehicle, cross camera: **0.8656** similarity
- **Different vehicle, same type/color:** ~0.85 similarity (overlaps with true matches!)

The overlap between "same vehicle cross-camera" and "different vehicle same type" is the core problem. Training on VeRi data teaches the model to push apart vehicles of the same type.

---

## 6. Recommended Improvement Path

### Phase 1: Post-processing (no training, immediate)
- **k-reciprocal re-ranking** on `EmbeddingGallery.query()` results
- Expected: mAP 9% → ~15-20%
- Effort: ~50 lines of code in `_gallery.py`

### Phase 2: CLIP-ReID weights (DONE)
- Downloaded ViT-CLIP-ReID-SIE-OLP checkpoint, exported to ONNX (346MB)
- `CLIPReIDExtractor` in `src/yowo/tracking/_clip_reid.py` (ONNX-based, 1280-dim)
- **Actual: mAP=82.28%, Rank-1=96.66%** (8.8x mAP improvement over zero-shot)
- Export script: `tmp/export_clip_reid_onnx.py`, ONNX model: `tmp/exports/clip-reid-veri-vit-b16.onnx`

### Phase 3: Custom training (maximum accuracy)
- Train CLIP-ReID on VeRi-776 + domain-specific data
- Add camera-aware loss for specific deployment cameras
- Expected: mAP → 92%+
- Effort: GPU training pipeline, dataset preparation

---

## 7. Implications for yowo CrossCameraTracker

| Config | CLIP zero-shot | CLIP-ReID (Phase 2 — DONE) |
|---|---|---|
| `match_threshold` | 0.10 (very tight) | 0.30-0.40 (relaxed) |
| False positive rate | High (same type/color) | Low (instance-discriminative) |
| Use case | Low-density traffic | **Production traffic CCTV** |
| `EmbeddingGallery.max_entries` | 1,000 (limit noise) | 10,000 (accurate matches) |
| mAP on VeRi-776 | 9.32% | **82.28%** |
| Rank-1 on VeRi-776 | 31.82% | **96.66%** |

---

## 8. Experiment Artifacts

| Artifact | Path |
|---|---|
| Experiment script | `tmp/experiment_veri_reid.py` |
| CLIP-ReID visual grid | `tmp/veri_results/veri_reid_CLIP-ReID_VeRi.jpg` |
| CLIP visual grid | `tmp/veri_results/veri_reid_CLIP_ViT-B-16.jpg` |
| FastReID visual grid | `tmp/veri_results/veri_reid_FastReID_SBS-S50.jpg` |
| CLIP-ReID ONNX model | `tmp/exports/clip-reid-veri-vit-b16.onnx` (346MB, 86.4M params) |
| CLIP-ReID export script | `tmp/export_clip_reid_onnx.py` |
| CLIP ONNX model | `tmp/exports/clip-vit-b16-visual.onnx` |
| FastReID ONNX model | `tmp/exports/fastreid-sbs-s50.onnx` |
| CLIP-ReID extractor | `src/yowo/tracking/_clip_reid.py` (203 lines) |

---

## 9. References

- [CLIP-SENet: CLIP-based Semantic Enhancement Network for Vehicle Re-identification](https://arxiv.org/html/2502.16815v1) (2025, paper only, no code)
- [CLIP-ReID: Exploiting Vision-Language Model for Image Re-identification](https://github.com/Syliz517/CLIP-ReID) (AAAI 2023, open source + VeRi weights)
- [VeRi-776 Dataset](https://github.com/JDAI-CV/VeRidataset) (Liu et al., ICME 2016)
- [fast-reid: SOTA Re-identification Toolbox](https://github.com/JDAI-CV/fast-reid) (supports vehicle ReID training)
- [Person ReID Baseline PyTorch](https://github.com/layumi/Person_reID_baseline_pytorch) (supports VeRi training)
- [Prototypical Contrastive Learning CLIP Fine-tuning for ReID](https://arxiv.org/abs/2310.17218)
- [Transformer-based Vehicle ReID with View Information](https://www.nature.com/articles/s41598-025-24392-y) (2025)
- [Multiple Discriminative Features with Non-local Attention](https://www.nature.com/articles/s41598-024-82755-3) (2024)
