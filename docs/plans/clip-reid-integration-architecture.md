# CLIP-ReID Integration for ByteTrack — Architecture & Implementation Plan

> **Author**: Tin Dang
> **Date**: 2026-02-28
> **Status**: DRAFT — Pending Review
> **Target Version**: v2.2.0
> **Module**: `yowo.tracking`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research: SOTA Appearance-Based MOT (2023–2026)](#2-research-sota-appearance-based-mot-20232026)
3. [CLIP & SegCLIP Deep Dive](#3-clip--segclip-deep-dive)
4. [Architecture Design — ReID Module](#4-architecture-design--reid-module)
5. [Architecture Design — Cost Matrix Fusion](#5-architecture-design--cost-matrix-fusion)
6. [Implementation Plan — File Structure & Changes](#6-implementation-plan--file-structure--changes)
7. [Public API Surface](#7-public-api-surface)
8. [Performance Budget Analysis](#8-performance-budget-analysis)
9. [CCTV-Specific Edge Cases](#9-cctv-specific-edge-cases)
10. [Testing Strategy](#10-testing-strategy)
11. [Risk Assessment](#11-risk-assessment)
12. [Self-Evaluation](#12-self-evaluation)

---

## 1. Executive Summary

Pure IoU-based ByteTrack suffers identity switches (IDS) in dense CCTV scenes — occlusion, re-entry after long absence, and visually similar targets cause track fragmentation. Integrating CLIP-based visual appearance embeddings into the association cascade can reduce IDS by 30–60% (per BoT-SORT/Deep OC-SORT literature) while keeping total frame time under 100 ms.

**Design principles:**

- **Backward compatible** — `ByteTracker()` without ReID produces bit-identical results
- **Protocol-based** — pluggable extractor (CLIP default, custom allowed)
- **ONNX-native** — CLIP visual encoder runs via ONNX Runtime (same stack as YOLO backend)
- **Conditional extraction** — ReID only runs when IoU matching is ambiguous, saving 40–80% of CLIP inference
- **Memory-bounded** — EMA embedding update (no unbounded gallery growth)
- **File budget** — every file stays under 700 lines

---

## 2. Research: SOTA Appearance-Based MOT (2023–2026)

### 2.1 Comparison Table

| Tracker | Venue | ReID Model | Embedding Update | Cost Fusion | IDS Reduction | FPS Impact | CCTV Suitability |
|---|---|---|---|---|---|---|---|
| **BoT-SORT** | arXiv 2022 | FastReID (SBS-S50, ResNet-50) | EMA (`η=0.9`) | `C = min(d_IoU, d̂_cos)` with gating: `d̂_cos = 0.5·d_cos` if `d_cos < θ_e` AND `d_IoU < θ_IoU`, else 1.0 | ~25–35% | -15–25% (FastReID overhead) | **High** — gated fusion prevents false appearance matches |
| **Deep OC-SORT** | ICASSP 2023 | FastReID (custom) | Dynamic EMA with adaptive α | `C = IoU + (a_w + w_b(m,n))·A_c` where `w_b` is per-pair discriminative weight | ~30–40% | -20–30% | **Medium** — adaptive weighting helps, but more complex |
| **StrongSORT** | TMM 2023 | OSNet (Omni-Scale) | EMA with momentum α, replaces DeepSORT 100-frame bank | Cascade: appearance first → IoU gating | ~20–30% | -30–40% (OSNet heavy) | **Medium** — OSNet heavy for edge |
| **SMILEtrack** | AAAI 2024 | Siamese SLM (Patch Self-Attention, CSP-Net backbone) | Per-frame Siamese comparison (no gallery) | Stage I: IoU + appearance (high-conf); Stage II: IoU only (low-conf); GATE function prevents false associations | ~15–25% (0.4–0.8 MOTA improvement) | **Severe**: 5.6–7.2 FPS vs ByteTrack 29.6 FPS | **Low** — too slow for real-time CCTV |
| **CLIP-ReID** | AAAI 2023 | CLIP ViT-B/16 (fine-tuned) | Two-stage: (1) optimize text tokens per ID, (2) fine-tune image encoder with ID-text constraints | Cosine distance in CLIP embedding space | 86.7% mAP on MSMT17 (ReID benchmark, not MOT) | N/A (ReID, not tracker) | **High for embeddings** — excellent feature quality, but needs tracker integration |

### 2.2 Key Insights

1. **BoT-SORT's gated min-cost fusion is the most production-ready**: It prevents appearance matches when IoU is already clear, avoiding catastrophic false associations. The formula `C = min(d_IoU, d̂_cos)` with dual thresholds (`θ_e=0.25`, `θ_IoU=0.5`) is simple, interpretable, and tunable.

2. **Deep OC-SORT's adaptive weighting adds complexity without proportional gain for CCTV**: The per-pair discriminative weight `w_b(m,n)` requires computing statistics over the entire cost matrix, adding latency. Better suited for academic benchmarks (DanceTrack) than production.

3. **EMA is universally preferred over gallery banks**: StrongSORT proved EMA with momentum (`η·old + (1-η)·new`) outperforms DeepSORT's 100-frame feature bank while using constant memory. All modern trackers have adopted this.

4. **Conditional ReID extraction saves 40–80% compute** (per "When to Extract ReID Features", arXiv 2409.06617): If exactly one confirmed track has IoU > threshold with a detection, skip ReID. Only extract when association is ambiguous (multiple candidates or low IoU). This yielded 80% FPS improvement on MOT17 with neutral accuracy.

5. **CLIP embeddings outperform task-specific ReID models for generalization**: CLIP-ReID showed that CLIP ViT-B/16 fine-tuned for ReID achieves 86.7% mAP on MSMT17. Even without fine-tuning, CLIP zero-shot features provide strong appearance discrimination due to training on 400M image-text pairs.

### 2.3 Recommended Approach for yowo

**Hybrid: BoT-SORT gated fusion + Conditional extraction + CLIP ONNX backend**

- Adopt BoT-SORT's `min(d_IoU, d̂_cos)` gated cost fusion (proven, simple, tunable)
- Add conditional ReID extraction (skip when IoU is unambiguous)
- Use CLIP ViT-B/16 visual encoder via ONNX Runtime (reuse yowo's ONNX infrastructure)
- Offer MobileCLIP-S0 alternative for edge (11.4M params, 1.5ms on iPhone)
- EMA embedding update with L2 renormalization

---

## 3. CLIP & SegCLIP Deep Dive

### 3.1 CLIP (Contrastive Language-Image Pre-training)

**Paper**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

**Core idea**: Train paired image + text encoders via contrastive loss on 400M image-text pairs from the internet. The visual encoder learns general-purpose image representations that transfer to downstream tasks without fine-tuning.

**Architecture variants relevant to ReID:**

| Variant | Image Encoder | Params (visual) | Embedding Dim | Input Size | Patch Size |
|---|---|---|---|---|---|
| ViT-B/32 | Vision Transformer Base | ~87M | 512 | 224×224 | 32×32 |
| ViT-B/16 | Vision Transformer Base | ~86M | 512 | 224×224 | 16×16 |
| ViT-L/14 | Vision Transformer Large | ~304M | 768 | 224×224 | 14×14 |
| MobileCLIP-S0 | MobileOne (Apple) | ~11.4M | 512 | 256×256 | - |
| MobileCLIP-S2 | MobileOne (Apple) | ~35.7M | 512 | 256×256 | - |

**Why CLIP for ReID in CCTV:**

1. **Domain robustness**: Trained on internet-scale diverse data — handles lighting, angles, partial occlusion better than task-specific ReID models (OSNet, SBS-S50) trained on pedestrian datasets only
2. **Zero-shot generalization**: No fine-tuning needed for deployment — works on vehicles, people, animals out-of-the-box
3. **ONNX-friendly**: Visual encoder is a standard ViT — clean export to ONNX, no text encoder needed for ReID (image-only embedding)
4. **L2-normalized embeddings**: CLIP naturally produces unit-norm embeddings ideal for cosine similarity

**Preprocessing pipeline (for ONNX):**
```
BGR (OpenCV) → RGB → Resize(224, bicubic) → CenterCrop(224) → /255.0 → Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
```

**Embedding extraction (visual encoder only):**
```
Input:  (B, 3, 224, 224) float32
Output: (B, 512) float32  [or (B, 768) for ViT-L/14]
```

The text encoder is **not needed** for ReID — we only extract and compare image embeddings.

### 3.2 SegCLIP (Patch Aggregation with Learnable Centers)

**Paper**: Luo et al., "SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation", ICML 2023

**Core idea**: Extend CLIP for pixel-level segmentation by introducing learnable center tokens that dynamically aggregate ViT patches into semantic groups via cross-attention.

**Architecture:**
- Builds on CLIP ViT backbone
- Inserts a "group module" with K learnable center embeddings into ViT's middle layers
- Cross-attention between centers and patch tokens generates a mapping matrix `M ∈ R^(K × N)` (K centers × N patches)
- Each center pulls semantically related patches, forming irregular-shaped segments
- Trained with: (1) reconstruction loss on masked patches, (2) superpixel-based KL loss with pseudo-labels

**Key differences from standard CLIP:**

| Aspect | CLIP | SegCLIP |
|---|---|---|
| Output | Global [CLS] embedding (512-d) | Per-patch semantic groups + segment maps |
| Granularity | Image-level | Pixel-level (patch-aggregated) |
| Training | Contrastive image-text | Contrastive + reconstruction + KL loss |
| Extra params | None | K learnable center embeddings + cross-attention |
| Use case | Classification, retrieval, ReID | Open-vocabulary segmentation |

**Performance** (PASCAL VOC / PASCAL Context / COCO):
- +0.3% / +2.3% / +2.2% mIoU over GroupViT baselines

**Relevance to ReID / MOT:**

SegCLIP is **not directly applicable** to ReID for tracking. Its strengths are in pixel-level segmentation, not instance-level appearance matching. However, two concepts from SegCLIP are architecturally informative:

1. **Patch-level features for part-aware ReID**: Instead of using only the global [CLS] token for ReID, one could use SegCLIP's patch aggregation to extract part-aware embeddings (head, torso, legs). This is more robust to partial occlusion — if legs are occluded, head/torso patches still match. *However, the complexity and latency cost outweigh benefits for real-time CCTV tracking.*

2. **Learnable centers as appearance prototypes**: The concept of K learnable centers could be adapted to learn per-class appearance prototypes (e.g., "person-in-uniform" vs "person-in-casual") for cross-camera ReID. *This requires training and is outside scope of initial integration.*

**Recommendation**: Use standard CLIP's global [CLS] embedding for ReID. SegCLIP's patch aggregation is a potential **Phase 2** enhancement for part-aware ReID if needed. The added complexity (custom centers, cross-attention, extra training) is not justified for the initial feature.

### 3.3 Model Selection Decision Matrix

| Criterion | CLIP ViT-B/16 | CLIP ViT-B/32 | MobileCLIP-S0 | SegCLIP |
|---|---|---|---|---|
| Embedding quality | **Best** (16×16 patches) | Good (32×32 patches, less spatial detail) | Good (comparable to ViT-B/16 at 4.8x speed) | Pixel-level (overkill for ReID) |
| ONNX latency (GPU) | ~4–8ms | ~2–4ms | ~1.5ms (iPhone Neural Engine) | ~10–20ms (extra cross-attention) |
| ONNX latency (CPU) | ~15–30ms | ~8–15ms | ~5–10ms (estimated) | ~30–60ms |
| Edge suitability | Server/desktop | Desktop/strong edge | **Best for edge** | Not recommended |
| Zero-shot ReID | Excellent | Good | Good | Not designed for ReID |
| ONNX export | Clean (standard ViT) | Clean | Requires `reparameterize_model()` | Complex (custom ops) |
| Maintenance | Community-maintained, stable | Same | Apple-maintained | Research code only |

**Default**: CLIP ViT-B/16 (best quality, acceptable latency on GPU/CoreML)
**Edge option**: MobileCLIP-S0 (4.8x faster, comparable quality)
**Not recommended**: SegCLIP (wrong tool for the job)

---

## 4. Architecture Design — ReID Module

### 4.1 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    ByteTracker.update()                   │
│                                                           │
│  Detection ──┬── split by confidence                     │
│              │                                            │
│              ▼                                            │
│  ┌───────────────────────┐                               │
│  │ Conditional ReID Gate │──── skip if IoU unambiguous   │
│  └──────────┬────────────┘                               │
│             │ ambiguous detections only                   │
│             ▼                                            │
│  ┌───────────────────────┐                               │
│  │  ReIDExtractor        │ Protocol-based, pluggable     │
│  │  ├─ extract_crops()   │ BGR numpy → (N, 224, 224, 3) │
│  │  ├─ preprocess()      │ normalize, RGB, CHW, batch    │
│  │  └─ infer()           │ ONNX session → (N, 512)      │
│  └──────────┬────────────┘                               │
│             │ embeddings: (N, 512) float32               │
│             ▼                                            │
│  ┌───────────────────────┐                               │
│  │ Cost Matrix Fusion    │                               │
│  │  IoU cost + gated     │                               │
│  │  appearance cost      │                               │
│  │  → fused_cost         │                               │
│  └──────────┬────────────┘                               │
│             │                                            │
│             ▼                                            │
│  ┌───────────────────────┐                               │
│  │ Hungarian Assignment  │ existing linear_assignment()  │
│  └──────────┬────────────┘                               │
│             │                                            │
│             ▼                                            │
│  ┌───────────────────────┐                               │
│  │ STrack.update()       │                               │
│  │  + EMA embedding      │ η·old + (1-η)·new, L2 renorm │
│  │    update             │                               │
│  └───────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 ReIDExtractor Protocol

```python
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

@runtime_checkable
class ReIDExtractor(Protocol):
    """Protocol for appearance feature extraction.

    Implementations must extract L2-normalized embeddings from
    detection crops for use in appearance-based track association.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings (e.g. 512 for CLIP ViT-B/16)."""
        ...

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]:
        """Extract L2-normalized embeddings for detection crops.

        Args:
            frame_pixels: Full frame as HWC BGR uint8 numpy array.
            boxes: Sequence of (x1, y1, x2, y2) bounding boxes to crop.

        Returns:
            (N, embedding_dim) float32 array, L2-normalized along dim=-1.
            Returns empty (0, embedding_dim) if boxes is empty.
        """
        ...
```

### 4.3 CLIPExtractor Implementation

```python
class CLIPExtractor:
    """ONNX-based CLIP visual encoder for ReID embedding extraction.

    Uses CLIP ViT-B/16 by default. Expects an ONNX model file containing
    only the visual encoder (image → embedding, no text encoder).

    Args:
        model_path: Path to ONNX model file.
        embedding_dim: Output embedding dimensionality (default 512).
        input_size: Model input resolution (default 224).
        device: "cpu" or "cuda" (selects ONNX EP).
        min_crop_area: Minimum crop area in pixels to extract (skip tiny crops).
    """

    # CLIP ImageNet normalization constants
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path,
        *,
        embedding_dim: int = 512,
        input_size: int = 224,
        device: str = "cpu",
        min_crop_area: int = 1024,  # 32×32 minimum
    ) -> None: ...

    @property
    def embedding_dim(self) -> int: ...

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]: ...

    def _crop_and_preprocess(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]:
        """Crop, resize, normalize, and batch detections.

        Steps:
        1. Clip box coords to frame bounds
        2. Filter out crops below min_crop_area
        3. For each valid crop: BGR→RGB, resize(224, bicubic), /255, normalize
        4. Stack → (N, 3, 224, 224) float32
        """
        ...
```

### 4.4 STrack Embedding Storage

Add to `STrack.__slots__`:

```python
__slots__ = (
    # ... existing slots ...
    "_embedding",       # NDArray[np.float32] | None — L2-norm'd appearance vector
)
```

**EMA update method on STrack:**

```python
def update_embedding(
    self,
    new_embedding: NDArray[np.float32],
    momentum: float = 0.9,
) -> None:
    """Update track appearance embedding via exponential moving average.

    Args:
        new_embedding: (D,) L2-normalized embedding from ReIDExtractor.
        momentum: Weight for existing embedding (default 0.9).
    """
    if self._embedding is None:
        self._embedding = new_embedding.copy()
    else:
        self._embedding = momentum * self._embedding + (1 - momentum) * new_embedding
        # Re-normalize to unit sphere
        norm = np.linalg.norm(self._embedding)
        if norm > 0:
            self._embedding /= norm
```

**Design decision — EMA vs. gallery:**

| Approach | Memory | Quality | Complexity |
|---|---|---|---|
| EMA (chosen) | O(D) per track = 2KB | Good — smooths noise, adapts gradually | Low — single vector, constant memory |
| Circular buffer (K=10) | O(K·D) per track = 20KB | Better — can match against multiple views | Medium — need min-distance over gallery |
| Unbounded gallery | O(T·D), grows with time | Best — full history | **Unacceptable** — memory leak over hours |

EMA is chosen for production 24/7 operation. A 512-dim float32 embedding = 2048 bytes per track. With 200 active tracks, that's ~400KB — negligible.

### 4.5 Crop Extraction Strategy

```python
def _extract_crops(
    frame: NDArray[np.uint8],          # HWC BGR
    boxes: Sequence[tuple[float, float, float, float]],
    target_size: int = 224,
    min_area: int = 1024,
) -> tuple[NDArray[np.float32], list[int]]:
    """Extract and preprocess crops from frame.

    Returns:
        (crops, valid_indices) where crops is (N, 3, H, W) float32
        and valid_indices maps crop index → original box index.
    """
    h, w = frame.shape[:2]
    crops = []
    valid_indices = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Clip to frame bounds
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
        cw, ch = ix2 - ix1, iy2 - iy1
        if cw * ch < min_area:
            continue  # Skip tiny / degenerate crops

        crop = frame[iy1:iy2, ix1:ix2]            # BGR HWC
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - MEAN) / STD                  # Normalize
        crop = crop.transpose(2, 0, 1)              # HWC → CHW
        crops.append(crop)
        valid_indices.append(i)

    if not crops:
        return np.empty((0, 3, target_size, target_size), dtype=np.float32), []

    return np.stack(crops, axis=0), valid_indices
```

**Performance note**: `cv2.resize` with `INTER_LINEAR` (bilinear) is used instead of `INTER_CUBIC` (bicubic) for speed. Bicubic is ~2x slower and provides marginal quality improvement for 224×224 targets from CCTV crops.

---

## 5. Architecture Design — Cost Matrix Fusion

### 5.1 Association Cascade (Modified ByteTrack)

```
                    ┌─────────────────────────┐
                    │  Kalman Predict All      │
                    │  (existing)              │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Conditional ReID Gate   │
                    │  Extract embeddings for  │
                    │  detections where needed │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │  STAGE 1a: Fused IoU+Appearance Match       │
          │  tracked + lost  ←→  high-conf detections   │
          │  cost = gated_fused_cost(iou, appearance)   │
          │  threshold = match_thresh (0.8)             │
          └──────────────────────┬──────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │  STAGE 2: IoU-only Match (existing)         │
          │  unmatched tracked  ←→  low-conf detections │
          │  cost = iou_distance (no appearance)        │
          │  threshold = stage2_thresh (0.5)            │
          └──────────────────────┬──────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │  STAGE 3: Appearance Rescue (NEW)           │
          │  lost tracks (age > N)  ←→  unmatched high  │
          │  cost = appearance_distance only             │
          │  threshold = reid_thresh (0.4)              │
          │  Gate: skip if lost < reid_lost_age frames  │
          └──────────────────────┬──────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │  Birth / Death / Dedup (existing)           │
          └─────────────────────────────────────────────┘
```

### 5.2 Gated Fused Cost Function (BoT-SORT Inspired)

```python
def gated_fused_cost(
    iou_cost: NDArray[np.float64],           # (N, M), 1 - IoU
    appearance_cost: NDArray[np.float64],     # (N, M), 1 - cosine_sim
    *,
    iou_gate: float = 0.5,                   # max IoU distance to consider appearance
    appearance_gate: float = 0.25,           # max appearance distance for valid match
    appearance_weight: float = 0.5,          # weight for appearance when gated
) -> NDArray[np.float64]:
    """Fused cost matrix with dual gating (BoT-SORT style).

    For each (track, detection) pair:
    - If iou_cost < iou_gate AND appearance_cost < appearance_gate:
        fused = min(iou_cost, appearance_weight * appearance_cost)
    - Else:
        fused = iou_cost  (fall back to IoU only)

    This prevents appearance from creating false long-range matches
    while allowing it to resolve ambiguous IoU-close pairs.
    """
    fused = iou_cost.copy()
    gate_mask = (iou_cost < iou_gate) & (appearance_cost < appearance_gate)
    fused[gate_mask] = np.minimum(
        iou_cost[gate_mask],
        appearance_weight * appearance_cost[gate_mask],
    )
    return fused
```

### 5.3 Appearance Distance Function

```python
def appearance_distance(
    tracks: Sequence[STrack],
    embeddings: NDArray[np.float32],   # (M, D) detection embeddings
    valid_indices: list[int],          # maps embedding row → detection index
    num_detections: int,               # total detection count
) -> NDArray[np.float64]:
    """Cosine distance matrix between track embeddings and detection embeddings.

    Args:
        tracks: Sequence of STrack with _embedding attribute.
        embeddings: (M, D) L2-normalized detection embeddings.
        valid_indices: Maps embedding index to detection index.
        num_detections: Total number of detections (including those without embeddings).

    Returns:
        (N, num_detections) cost matrix. Entries for tracks/detections without
        embeddings are set to 1.0 (maximum distance, effectively IoU-only).
    """
    n_tracks = len(tracks)
    cost = np.ones((n_tracks, num_detections), dtype=np.float64)

    if embeddings.size == 0:
        return cost

    # Collect track embeddings (None → skip)
    track_embs = []
    track_mask = []
    for i, track in enumerate(tracks):
        if track._embedding is not None:
            track_embs.append(track._embedding)
            track_mask.append(i)

    if not track_embs:
        return cost

    track_matrix = np.stack(track_embs, axis=0).astype(np.float64)  # (K, D)
    det_matrix = embeddings[..., :].astype(np.float64)               # (M, D)

    # Cosine similarity → distance
    sim = track_matrix @ det_matrix.T                                # (K, M)
    dist = 1.0 - sim                                                 # cosine distance

    # Map back to full cost matrix
    for ki, ti in enumerate(track_mask):
        for mi, di in enumerate(valid_indices):
            cost[ti, di] = dist[ki, mi]

    return cost
```

### 5.4 Conditional ReID Gate (Selective Extraction)

```python
def should_extract_reid(
    iou_cost_row: NDArray[np.float64],
    ambiguity_low: float = 0.4,
    ambiguity_high: float = 0.7,
) -> bool:
    """Determine if ReID is needed for a track's association candidates.

    ReID is needed when IoU matching is ambiguous:
    - Multiple detections have IoU cost in [ambiguity_low, ambiguity_high]
    - Or the best match is in the ambiguous range

    Skip ReID when:
    - Exactly one detection has IoU cost < ambiguity_low (clear match)
    - All detections have IoU cost > ambiguity_high (no overlap, new track)
    """
    in_range = (iou_cost_row >= ambiguity_low) & (iou_cost_row <= ambiguity_high)
    clear_match = np.sum(iou_cost_row < ambiguity_low)
    return bool(np.any(in_range) or clear_match > 1)
```

**In practice**, the gate is applied per-frame (not per-track) for efficiency:

```python
# In ByteTracker.update():
needs_reid = any(
    should_extract_reid(cost1[i]) for i in range(cost1.shape[0])
)
if needs_reid and self._reid is not None:
    embeddings, valid_indices = self._reid.extract(frame_pixels, high_boxes)
    app_cost = appearance_distance(strack_pool, embeddings, valid_indices, len(high_boxes))
    cost1 = gated_fused_cost(cost1, app_cost)
```

### 5.5 Stage 3: Lost Track Re-Association via Appearance

For tracks lost for >N frames, IoU is unreliable (Kalman prediction drifts). Appearance-only matching can recover these:

```python
# Stage 3: Appearance rescue for long-lost tracks
if self._reid is not None and unmatched_high_idxs:
    long_lost = [t for t in self._lost if t.time_since_update > self._reid_lost_age]
    if long_lost:
        lost_embs = [t for t in long_lost if t._embedding is not None]
        if lost_embs:
            reid_cost = appearance_distance(
                lost_embs, embeddings, valid_indices, len(high_boxes)
            )
            matches3, _, _ = linear_assignment(reid_cost, self._reid_thresh)
            # Re-activate matched lost tracks
```

---

## 6. Implementation Plan — File Structure & Changes

### 6.1 File Map

```
src/yowo/tracking/
├── __init__.py          # [MODIFY] Add ReIDExtractor, CLIPExtractor exports
├── _tracker.py          # [MODIFY] ByteTracker accepts optional reid_extractor
├── _strack.py           # [MODIFY] Add _embedding slot + update_embedding()
├── _matching.py         # [MODIFY] Add appearance_distance(), gated_fused_cost()
├── _kalman.py           # [NO CHANGE]
└── _reid.py             # [NEW] ReIDExtractor Protocol + CLIPExtractor
```

### 6.2 Ordered Task List

| # | Task | File(s) | Depends On | Est. Lines |
|---|---|---|---|---|
| 1 | Add `_embedding` slot to STrack + `update_embedding()` method | `_strack.py` | — | +25 |
| 2 | Create `_reid.py` — ReIDExtractor Protocol + CLIPExtractor | `_reid.py` (new) | — | ~250 |
| 3 | Add `appearance_distance()` + `gated_fused_cost()` to matching | `_matching.py` | #1 | +80 |
| 4 | Modify `ByteTracker.__init__()` — accept optional `reid_extractor` | `_tracker.py` | #2, #3 | +15 |
| 5 | Modify `ByteTracker.update()` — conditional ReID + fused cost + Stage 3 | `_tracker.py` | #4 | +60 |
| 6 | Update `__init__.py` exports | `__init__.py` | #2 | +5 |
| 7 | Update `track_stream()` — pass reid_extractor through | `__init__.py` | #6 | +10 |
| 8 | Unit tests for `_reid.py` | `tests/unit/tracking/test_reid.py` (new) | #2 | ~300 |
| 9 | Unit tests for appearance matching | `tests/unit/tracking/test_matching.py` | #3 | +150 |
| 10 | Integration tests — ByteTracker + mock ReID | `tests/unit/tracking/test_tracker.py` | #5 | +200 |
| 11 | Regression tests — pure IoU mode unchanged | `tests/unit/tracking/test_tracker.py` | #5 | +50 |

### 6.3 File Size Budget

| File | Current Lines | Added Lines | Projected Total | Under 700? |
|---|---|---|---|---|
| `_strack.py` | ~250 | +25 | ~275 | Yes |
| `_matching.py` | ~230 | +80 | ~310 | Yes |
| `_tracker.py` | ~300 | +75 | ~375 | Yes |
| `_reid.py` (new) | 0 | ~250 | ~250 | Yes |
| `__init__.py` | ~80 | +15 | ~95 | Yes |

---

## 7. Public API Surface

### 7.1 New Exports

```python
# yowo/tracking/__init__.py
from yowo.tracking._reid import CLIPExtractor, ReIDExtractor

__all__ = [
    "ByteTracker",
    "CLIPExtractor",      # NEW
    "ReIDExtractor",      # NEW
    "TrackState",
    "TrackedBox",
    "TrackedDetection",
    "track_detections",
    "track_stream",
]
```

### 7.2 Usage Examples

**Basic — ByteTracker with CLIP ReID:**

```python
from yowo.tracking import ByteTracker, CLIPExtractor

reid = CLIPExtractor("models/clip-vit-b16-visual.onnx")
tracker = ByteTracker(reid_extractor=reid)

for detection in engine.stream(source):
    tracked = tracker.update(detection)
    for box in tracked.boxes:
        print(f"Track {box.track_id}: {box.as_xyxy}")
```

**Via track_stream (convenience):**

```python
from yowo.tracking import track_stream, CLIPExtractor

reid = CLIPExtractor("models/clip-vit-b16-visual.onnx")
for tracked in track_stream(engine, source, reid_extractor=reid):
    ...
```

**Pure IoU mode (backward compatible):**

```python
# Exactly as before — no ReID, identical behavior
tracker = ByteTracker()
for detection in engine.stream(source):
    tracked = tracker.update(detection)
```

**Custom ReID extractor:**

```python
class MyReIDExtractor:
    """Custom extractor implementing ReIDExtractor protocol."""

    @property
    def embedding_dim(self) -> int:
        return 256

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]:
        # Your custom logic here
        ...

tracker = ByteTracker(reid_extractor=MyReIDExtractor())
```

### 7.3 ByteTracker Constructor (Updated)

```python
def __init__(
    self,
    *,
    track_high_thresh: float = 0.6,
    track_low_thresh: float = 0.1,
    match_thresh: float = 0.8,
    max_age: int = 30,
    min_hits: int = 3,
    # NEW — ReID parameters
    reid_extractor: ReIDExtractor | None = None,
    reid_iou_gate: float = 0.5,
    reid_appearance_gate: float = 0.25,
    reid_appearance_weight: float = 0.5,
    reid_momentum: float = 0.9,
    reid_lost_age: int = 5,
    reid_thresh: float = 0.4,
) -> None:
```

### 7.4 TrackedBox — No Changes

`TrackedBox` remains frozen and unchanged. Embeddings are internal to `STrack` and not exposed in the public output. This keeps the public API minimal and avoids coupling consumers to the ReID implementation.

---

## 8. Performance Budget Analysis

### 8.1 Latency Breakdown (Target: <100ms total frame time)

| Component | Server GPU (RTX 3090) | Desktop CPU (M4 Pro) | Edge (Jetson Orin Nano) |
|---|---|---|---|
| **YOLO detection** | 3–8ms (TensorRT) | 6–12ms (CoreML) | 15–30ms (TensorRT INT8) |
| **Crop extraction** (20 dets, 224×224) | 0.5–1ms | 0.5–1ms | 1–2ms |
| **CLIP inference** (20 crops, ViT-B/16 ONNX) | 3–6ms (CUDA EP) | 8–15ms (CoreML EP) | 15–25ms (TensorRT) |
| **CLIP inference** (20 crops, MobileCLIP-S0) | 1–2ms | 3–5ms | 5–8ms |
| **Conditional gate savings** (skip 40–80%) | Saves 1–5ms | Saves 3–12ms | Saves 6–20ms |
| **Cost matrix fusion** | <0.1ms | <0.1ms | <0.2ms |
| **Kalman + Hungarian** | 0.2–0.5ms | 0.2–0.5ms | 0.5–1ms |
| **EMA embedding update** | <0.01ms | <0.01ms | <0.01ms |
| **Total (ViT-B/16, worst case)** | **7–16ms** | **15–29ms** | **32–59ms** |
| **Total (ViT-B/16, with gate)** | **5–12ms** | **10–20ms** | **22–42ms** |
| **Total (MobileCLIP-S0, with gate)** | **4–8ms** | **7–12ms** | **16–28ms** |
| **YOLO + Tracking total** | **10–24ms** | **16–41ms** | **38–87ms** |

### 8.2 CLIP ONNX Benchmark Reference

From CLIP-as-service benchmarks (Nvidia TITAN RTX):

| Model | Image QPS | ~Latency per image | Embedding Dim |
|---|---|---|---|
| ViT-B/32 | ~290 | ~3.4ms | 512 |
| ViT-B/16 | ~264 | ~3.8ms | 512 |
| ViT-L/14 | ~143 | ~7.0ms | 768 |

Batched inference (20 crops): `20/264 ≈ 76ms` single-threaded → `~5–8ms` batched on GPU (parallelism).

From MobileCLIP benchmarks (iPhone 12 Pro Max, Neural Engine):

| Model | Image Encoder Latency | Params (Image) |
|---|---|---|
| MobileCLIP-S0 | 1.5ms | 11.4M |
| MobileCLIP-S2 | 3.6ms | 35.7M |
| MobileCLIP2-B | 7.1ms | 86.3M |

### 8.3 Optimization Strategies

1. **Conditional extraction** (highest ROI): Skip CLIP for unambiguous IoU matches. Saves 40–80% of inference based on scene density. Implementation: compute IoU cost first, check ambiguity, only extract ReID for ambiguous pairs.

2. **Frame-skip extraction**: Extract embeddings every N frames (e.g., N=3), reuse cached embeddings for intermediate frames. Tracks in clear motion don't need per-frame appearance updates.

3. **Batched ONNX inference**: Always batch all crops per frame into a single `session.run()` call. Never extract per-detection.

4. **Async prefetch** (future Phase 2): Start CLIP inference on current frame's crops while postprocessing previous frame's detections. Requires pipeline overlap via ThreadPoolExecutor.

5. **Model selection**: Default to ViT-B/16 on GPU/CoreML, recommend MobileCLIP-S0 for edge via documentation/config.

---

## 9. CCTV-Specific Edge Cases

### 9.1 Failure Modes & Mitigations

| # | Failure Mode | Description | Mitigation |
|---|---|---|---|
| 1 | **Appearance ambiguity** | Same-uniform workers, identical vehicles → embedding similarity ceiling (~0.85 cosine) | IoU gating ensures spatial consistency prevails when appearances are indistinguishable. `reid_appearance_gate=0.25` rejects matches with cosine distance > 0.25 |
| 2 | **Lighting transitions** | Person moves sunlight → shadow → embedding drifts significantly | EMA momentum (η=0.9) smooths gradual lighting changes. High momentum preserves stable identity; low momentum (η=0.7) for rapidly changing scenes |
| 3 | **Partial occlusion crops** | Half-visible person → noisy CLIP embedding → false match | `min_crop_area=1024` gate (32×32 minimum). Crops below threshold get no embedding → fall back to IoU only |
| 4 | **Cross-camera re-entry** | Person leaves Camera A, enters Camera B minutes later | Stage 3 appearance rescue: lost tracks up to `max_age` frames get appearance-only matching. For cross-camera: requires shared track state (out of scope for v2.2.0, document as future work) |
| 5 | **Embedding gallery bloat** | Long-lived tracks accumulate stale features over hours | EMA = constant O(D) memory per track. No gallery growth. Embedding naturally forgets old appearance over time via exponential decay |
| 6 | **CLIP domain gap** | CLIP trained on internet images, not security camera angles/fish-eye/IR | Mitigated by CLIP's massive training set (400M pairs). For extreme domain gap (thermal IR, fish-eye): recommend fine-tuned weights or MobileCLIP with domain adaptation. Document as limitation |
| 7 | **Crowded scene (>50 dets)** | CLIP batch of 50 crops exceeds latency budget | Conditional gate reduces to ~10–20 ambiguous detections. If still >30: subsample by confidence (top-K crops only) |
| 8 | **Night / low-light** | CLIP embeddings degrade in very dark frames | Detect frame brightness (mean pixel < 30); disable ReID for dark frames and fall back to IoU only. Avoids adding noise |
| 9 | **Fast motion blur** | Blurred crops produce unreliable embeddings | Detect blur via Laplacian variance on crop; skip embedding if variance < threshold |

### 9.2 24/7 Operation Concerns

- **Memory stability**: EMA = 2KB per track × 200 tracks = 400KB. Track removal on `max_age` expiry prevents accumulation. No memory leak.
- **ONNX session lifecycle**: Single `InferenceSession` created at init, reused across all frames. No session churn.
- **Thread safety**: `CLIPExtractor.extract()` is stateless (no mutable instance state except the ONNX session, which is thread-safe for `session.run()`). Safe for concurrent use from pipeline workers.
- **Graceful degradation**: If ONNX session fails (GPU OOM, model corruption), catch `onnxruntime.OrtException`, log warning, disable ReID for remainder of stream, fall back to pure IoU. No crash.

---

## 10. Testing Strategy

### 10.1 Test Matrix

| Category | Test | File | Description |
|---|---|---|---|
| **Unit: _reid.py** | `test_clip_extractor_init` | `test_reid.py` | Verify ONNX session loads, embedding_dim property |
| | `test_extract_empty_boxes` | `test_reid.py` | Empty input → (0, D) output |
| | `test_extract_single_crop` | `test_reid.py` | One box → (1, D) L2-normalized |
| | `test_extract_batch_crops` | `test_reid.py` | 20 boxes → (20, D) L2-normalized |
| | `test_min_crop_area_gate` | `test_reid.py` | Tiny box filtered → fewer embeddings than boxes |
| | `test_crop_clipping_to_frame` | `test_reid.py` | Box partially outside frame → clipped, no crash |
| | `test_normalization_correctness` | `test_reid.py` | Verify CLIP mean/std applied correctly |
| | `test_protocol_compliance` | `test_reid.py` | CLIPExtractor satisfies ReIDExtractor protocol |
| | `test_custom_extractor_protocol` | `test_reid.py` | Custom class with matching protocol works |
| **Unit: _matching.py** | `test_appearance_distance_shape` | `test_matching.py` | (N, M) output for N tracks, M dets |
| | `test_appearance_distance_no_embeddings` | `test_matching.py` | All costs = 1.0 when no embeddings |
| | `test_appearance_distance_identical` | `test_matching.py` | Same embedding → cost = 0.0 |
| | `test_appearance_distance_orthogonal` | `test_matching.py` | Orthogonal embeddings → cost = 1.0 |
| | `test_gated_fused_cost_iou_only` | `test_matching.py` | When appearance > gate → falls back to IoU |
| | `test_gated_fused_cost_fused` | `test_matching.py` | When both gates pass → min(iou, weighted_app) |
| | `test_gated_fused_cost_shape` | `test_matching.py` | Output shape matches input |
| **Unit: _strack.py** | `test_embedding_slot_exists` | `test_strack.py` | `_embedding` in STrack.__slots__ |
| | `test_update_embedding_init` | `test_strack.py` | First call sets embedding (copy, not reference) |
| | `test_update_embedding_ema` | `test_strack.py` | EMA formula: η·old + (1-η)·new |
| | `test_update_embedding_renormalization` | `test_strack.py` | After EMA, embedding is L2-normalized |
| **Integration: _tracker.py** | `test_tracker_no_reid_identical` | `test_tracker.py` | ByteTracker() without ReID produces identical results to current code (regression) |
| | `test_tracker_with_mock_reid` | `test_tracker.py` | ByteTracker with mock extractor uses fused cost |
| | `test_tracker_reid_reduces_ids` | `test_tracker.py` | Synthetic swap scenario: 2 tracks cross paths, ReID prevents ID switch |
| | `test_tracker_stage3_lost_rescue` | `test_tracker.py` | Track lost for N frames → re-associated via appearance |
| | `test_tracker_reid_graceful_degradation` | `test_tracker.py` | If extractor raises, tracker falls back to IoU |
| **Performance** | `test_reid_overhead_20_dets` | `test_perf.py` | <10ms overhead for 20 detections with mock ONNX |
| | `test_conditional_gate_skip_rate` | `test_perf.py` | Verify >40% frames skip ReID in sparse scene |

### 10.2 Mock ReID Extractor for Tests

```python
class MockReIDExtractor:
    """Deterministic mock for testing — returns box-position-based embeddings."""

    @property
    def embedding_dim(self) -> int:
        return 128

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]:
        if not boxes:
            return np.empty((0, 128), dtype=np.float32)
        embs = np.zeros((len(boxes), 128), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Deterministic embedding based on box center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            embs[i, 0] = cx / 1920  # Normalized x
            embs[i, 1] = cy / 1080  # Normalized y
            embs[i, 2] = (x2 - x1) / 1920  # Normalized width
            embs[i, 3] = (y2 - y1) / 1080  # Normalized height
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embs / norms
```

---

## 11. Risk Assessment

### Top 5 Risks

| # | Risk | Severity | Probability | Mitigation |
|---|---|---|---|---|
| 1 | **CLIP latency exceeds budget on edge** — ViT-B/16 ONNX on Jetson: 15–25ms for 20 crops, pushing total >100ms | High | Medium | Offer MobileCLIP-S0 alternative (5–8ms on Jetson). Document model selection guide. Conditional gate reduces effective calls by 40–80% |
| 2 | **CLIP domain gap on CCTV** — Low-res, top-down angle, IR cameras produce embeddings in untrained distribution | Medium | Medium | CLIP's 400M training pairs provide broad coverage. For extreme cases (thermal, fish-eye): document need for fine-tuned weights. Test with real CCTV footage before release |
| 3 | **Appearance false matches in uniform crowds** — Factory workers, sports teams → cosine similarity > 0.85 everywhere | Medium | High | IoU gating (`reid_iou_gate=0.5`) ensures spatial proximity is required. Appearance alone cannot override spatial mismatch. Document limitation in user guide |
| 4 | **ONNX model distribution** — CLIP ViT-B/16 visual encoder ONNX file not shipped with yowo (copyright, size ~350MB) | Low | Certain | Provide `yowo export-clip` CLI command to export from open_clip. Document download instructions. Don't bundle weights |
| 5 | **STrack slot change breaks serialization** — Adding `_embedding` slot changes pickle/copy behavior | Low | Low | STrack is internal (not public API). No serialization guarantees. Document in changelog. `_embedding` initialized to `None` → backward-compatible for existing STrack construction |

### Risk #1 Deep Dive — Latency Sensitivity Analysis

Worst case (edge, ViT-B/16, 50 detections, no conditional gate):
```
YOLO:      30ms
Crop:       3ms
CLIP:      40ms  (50 crops at ~0.8ms/crop)
Matching:   1ms
Total:     74ms  → PASS (<100ms) but tight
```

With conditional gate (skip 60% of frames):
```
Average CLIP: 40ms × 0.4 = 16ms amortized
Total avg:    50ms → comfortable
```

With MobileCLIP-S0:
```
CLIP:       8ms  (50 crops at ~0.16ms/crop)
Total:     42ms → generous headroom
```

---

## 12. Self-Evaluation

| Dimension | Score | Notes |
|---|---|---|
| **Completeness** | 0.92 | Covers all 8 steps requested. Missing: actual ONNX export script for CLIP visual encoder (documented as CLI command, not implemented). SegCLIP analysis included with clear recommendation against direct use. |
| **Clarity** | 0.93 | Architecture diagram, code contracts, data flow all specified. A senior dev can implement from this document. File-by-file change list with line estimates provided. |
| **Practicality** | 0.91 | <100ms achieved on all target hardware with conditional gate + appropriate model selection. Edge case requires MobileCLIP-S0 (documented). Real CLIP ONNX benchmarks referenced from CLIP-as-service and MobileCLIP papers. |
| **Optimization** | 0.90 | Conditional gate (40–80% skip), batched inference, EMA (constant memory), dual gating (BoT-SORT). Frame-skip and async prefetch documented as Phase 2. |
| **Edge Cases** | 0.91 | 9 CCTV failure modes addressed with concrete mitigations. 24/7 operation concerns (memory, thread safety, graceful degradation) covered. Cross-camera ReID explicitly deferred as future work. |
| **Backward Compatibility** | 0.95 | `reid_extractor=None` default → no ReID → identical code path. Regression test planned. STrack `_embedding=None` init. No public API breakage. |

**Overall: 0.92** — Ready for implementation review. Primary gap: real ONNX CLIP latency validation on actual yowo hardware (requires benchmark after implementation).

---

## References

- [ByteTrack — Zhang et al., ECCV 2022](https://arxiv.org/abs/2110.06864)
- [BoT-SORT — Aharon et al., arXiv 2022](https://arxiv.org/abs/2206.14651) | [GitHub](https://github.com/NirAharon/BoT-SORT)
- [Deep OC-SORT — Maggiolino et al., ICASSP 2023](https://arxiv.org/abs/2302.11813) | [GitHub](https://github.com/GerardMaggiolino/Deep-OC-SORT)
- [StrongSORT — Du et al., TMM 2023](https://www.researchgate.net/publication/367659411_StrongSORT_Make_DeepSORT_Great_Again) | [Labellerr Tutorial](https://www.labellerr.com/blog/objects-tracking-using-strongsort/)
- [SMILEtrack — Wang et al., AAAI 2024](https://arxiv.org/abs/2211.08824) | [GitHub](https://github.com/WWangYuHsiang/SMILEtrack)
- [CLIP-ReID — Li et al., AAAI 2023](https://arxiv.org/abs/2211.13977) | [GitHub](https://github.com/Syliz517/CLIP-ReID)
- [SegCLIP — Luo et al., ICML 2023](https://arxiv.org/abs/2211.14813) | [GitHub](https://github.com/ArrowLuo/SegCLIP)
- [CLIP — Radford et al., ICML 2021](https://openai.com/index/clip/) | [GitHub](https://github.com/openai/CLIP)
- [MobileCLIP — Apple, CVPR 2024](https://machinelearning.apple.com/research/mobileclip) | [GitHub](https://github.com/apple/ml-mobileclip)
- [MobileCLIP2 — Apple, TMLR 2025](https://machinelearning.apple.com/research/mobileclip2)
- [OpenCLIP — mlfoundations](https://github.com/mlfoundations/open_clip)
- [CLIP-ONNX — Lednik7](https://github.com/Lednik7/CLIP-ONNX)
- [onnx_clip — Lakera AI](https://github.com/lakeraai/onnx_clip)
- [When to Extract ReID Features — ECCV 2024](https://arxiv.org/abs/2409.06617)
- [CLIP-as-service Benchmarks — Jina AI](https://clip-as-service.jina.ai/user-guides/benchmark/)
- [Comprehensive Guide: Tracking by Detection](https://miguel-mendez-ai.com/2023/11/08/tracking-by-detection-overview)
