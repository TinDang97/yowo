# Method 1: FastReID SBS-S50 Baseline — Integration into yowo ByteTrack

> **Author**: Tin Dang
> **Date**: 2026-02-28
> **Status**: DRAFT — Pending Review
> **Target Version**: v2.2.0
> **Module**: `yowo.tracking`
> **Prerequisite**: yowo v2.1.0 (ByteTrack + ObjectCounter)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [FastReID Architecture Deep Dive](#2-fastreid-architecture-deep-dive)
3. [ONNX Export Pipeline](#3-onnx-export-pipeline)
4. [Cost Matrix Fusion — BoT-SORT Formula](#4-cost-matrix-fusion--bot-sort-formula)
5. [Embedding Update Strategy](#5-embedding-update-strategy)
6. [Association Cascade Integration](#6-association-cascade-integration)
7. [Implementation Specification](#7-implementation-specification)
8. [Performance Analysis](#8-performance-analysis)
9. [Strengths and Limitations](#9-strengths-and-limitations)
10. [Testing Strategy](#10-testing-strategy)
11. [CCTV-Specific Considerations](#11-cctv-specific-considerations)
12. [Benchmark Results from Literature](#12-benchmark-results-from-literature)
13. [References](#13-references)

---

## 1. Executive Summary

### What is FastReID SBS-S50?

FastReID is a research toolkit for person re-identification (ReID), developed and open-sourced by MEGVII (now Megvii / 旷视科技). Within FastReID, **SBS-S50** (Short Baseline Strong, ResNet-50 variant) is the standard backbone configuration that achieves state-of-the-art person ReID accuracy with a practical compute footprint. The model extracts a compact 256-dimensional L2-normalized embedding vector from a cropped person image, enabling fast cosine-distance matching between tracklets across frames.

### Why SBS-S50 is the proven MOT baseline

Two of the highest-performing multi-object trackers in the MOT Challenge (2022-2023) use FastReID as their appearance extractor:

- **BoT-SORT** (Aharon et al., 2022) — **1st place on MOT17 and MOT20 private detection**. Uses FastReID SBS-S50 with gated cost matrix fusion. HOTA 65.0, IDF1 80.2, IDS 1212 on MOT17 (vs. ByteTrack's IDS 2196 — a 45% reduction in identity switches).
- **Deep OC-SORT** (Maggiolino et al., ICASSP 2023) — **1st place on MOT20**. Uses a custom FastReID variant with adaptive appearance weighting. HOTA 64.9 on MOT17, HOTA 63.9 on MOT20.

Both trackers build upon the ByteTrack association cascade and add FastReID embeddings as an appearance cue. This makes SBS-S50 the natural baseline ReID model for enhancing yowo's existing ByteTracker — no architectural innovations are required, only the proven BoT-SORT/Deep OC-SORT integration pattern.

### Why this document

yowo v2.1.0 ships a pure IoU-based ByteTracker (`src/yowo/tracking/`). It works well in sparse scenes but suffers from identity switches (IDS) when:

- Objects occlude each other and re-emerge
- Objects move through dense crowds with overlapping bounding boxes
- Kalman filter predictions drift during extended occlusion (>10 frames)
- Visually distinct objects occupy similar spatial positions

Adding FastReID appearance embeddings addresses all four failure modes. This document specifies the full integration plan — architecture, math, code contracts, performance budget, and test strategy — to serve as the baseline Method 1 against which alternative ReID approaches (CLIP-ReID, OSNet, custom models) can be evaluated.

---

## 2. FastReID Architecture Deep Dive

### 2.1 Model Architecture

SBS-S50 is a standard ResNet-50 modified with non-local attention blocks and a BN (Batch Normalization) neck for domain-adaptive feature extraction.

```
Input Image (256 x 128 x 3, RGB)
    |
    v
[ResNet-50 Backbone]
    |-- Conv1: 7x7, stride 2, 64 channels
    |-- BN + ReLU + MaxPool(3x3, stride 2)
    |-- Layer1: 3 x Bottleneck(64 -> 256)
    |-- Layer2: 4 x Bottleneck(128 -> 512)
    |-- [Non-Local Block inserted after Layer2]
    |-- Layer3: 6 x Bottleneck(256 -> 1024)
    |-- [Non-Local Block inserted after Layer3]
    |-- Layer4: 3 x Bottleneck(512 -> 2048)
    |
    v
[GeM Pooling] — Generalized Mean Pooling
    |-- Pool(x) = (1/HW * sum(x^p))^(1/p), learnable p (init p=3.0)
    |-- Output: 2048-d vector
    |
    v
[BN Neck] — Batch Normalization Neck
    |-- BatchNorm1d(2048)
    |-- Output: 2048-d normalized backbone feature (used for training losses)
    |
    v
[Projection Head] (FC layer, training only — or retained for compact embeddings)
    |-- Linear(2048 -> 256)
    |-- L2 Normalize
    |-- Output: 256-d unit-norm embedding vector
```

**Total parameters**: ~25.6M (ResNet-50 base) + ~0.5M (non-local blocks) + ~0.5M (projection head) = ~26.6M

### 2.2 Component Design Rationale

#### Input: 256 x 128 (2:1 portrait aspect ratio)

The 2:1 height-to-width ratio is not arbitrary. Pedestrian bounding boxes in surveillance video have a characteristic portrait aspect ratio:

- A standing person's height is typically 1.5-2.5x their shoulder width
- CCTV cameras are usually mounted above head height, looking down at 15-45 degrees, which compresses width relative to height
- The Market-1501 and MSMT17 ReID training datasets consist of person crops with mean aspect ratio approximately 2:1

Using 256x128 instead of a square 256x256 input provides two benefits: (1) it avoids distorting the person's appearance by stretching horizontally, which would degrade discriminative features like body proportions, clothing patterns, and gait characteristics; (2) it reduces compute by 50% compared to 256x256 since the total pixel count is 32,768 vs 65,536.

#### Output: 256-d L2-normalized embedding

The choice of 256 dimensions (rather than the backbone's native 2048) is driven by deployment constraints:

- **Cosine distance computation**: For N tracks and M detections, computing the full NxM distance matrix requires `N * M * D` multiply-accumulate operations. With D=256, matching 50 tracks against 20 detections takes 256,000 operations (sub-microsecond on modern CPUs). With D=2048, this becomes 2,048,000 operations — an 8x increase that does not improve matching accuracy after proper projection training.
- **Memory footprint**: 256-d float32 = 1,024 bytes per track. At 200 active tracks, total embedding storage is 200 KB. With 2048-d, this would be 1.6 MB — manageable, but unnecessary.
- **L2 normalization**: Constraining embeddings to the unit hypersphere converts Euclidean distance to cosine distance: `d_cos(a, b) = 1 - a^T b = 0.5 * ||a - b||^2` when `||a|| = ||b|| = 1`. This means cosine similarity becomes a simple dot product, and distance thresholds have a fixed, interpretable range of [0, 2].

#### Non-Local Blocks (after Layer2 and Layer3)

Non-local blocks (Wang et al., CVPR 2018) compute self-attention over spatial feature maps, capturing long-range dependencies within the person crop. This helps distinguish people by their global body structure (e.g., relative position of bag, hat, shoes) rather than relying solely on local texture patches. In ReID, this is critical for handling partial occlusion — the model learns to associate visible parts with the complete identity.

The non-local operation for position i:

```
y_i = (1/C(x)) * sum_j( f(x_i, x_j) * g(x_j) )
```

where `f` is a pairwise affinity function (embedded Gaussian), `g` is a linear projection, and `C(x)` is a normalization factor.

#### GeM Pooling (Generalized Mean Pooling)

Standard Global Average Pooling (GAP) treats all spatial positions equally, diluting discriminative regions with background pixels inside the bounding box. GeM pooling (Radenovic et al., TPAMI 2019) introduces a learnable exponent `p`:

```
GeM(x) = (1/(H*W) * sum_{h,w} x_{h,w}^p)^{1/p)
```

- When `p = 1`: equivalent to GAP (uniform weighting)
- When `p → inf`: equivalent to Global Max Pooling (selects most activated feature)
- Learned `p` (typically converges to 3-4): emphasizes high-activation regions (distinctive body parts) while suppressing low-response background

This allows the model to focus on the most discriminative spatial regions without explicit attention masks.

#### BN Neck (Batch Normalization Neck)

The BN neck, introduced in the "Bag of Tricks" paper (Luo et al., CVPRW 2019), is a single `BatchNorm1d(2048)` layer inserted between the backbone output and the classification/metric heads during training:

- **For triplet loss**: Uses the pre-BN feature (backbone output) — triplet loss benefits from the non-normalized space where distance magnitudes carry information
- **For cross-entropy loss**: Uses the post-BN feature — BN centers features per-class, improving classifier convergence
- **At inference**: The post-BN feature (or its 256-d projection) is used as the embedding, because BN implicitly performs a form of domain normalization that improves generalization to unseen cameras/environments

This dual-use design is the key insight of BN neck: it decouples the optimization landscapes of metric learning (triplet) and classification (cross-entropy) losses without requiring two separate feature paths.

### 2.3 Training Regime

FastReID SBS-S50 is trained on large-scale pedestrian ReID datasets with a multi-task loss:

**Datasets** (combined for SBS):
- Market-1501: 32,668 images, 1,501 identities, 6 cameras
- DukeMTMC-reID: 36,411 images, 1,404 identities, 8 cameras
- MSMT17: 126,441 images, 4,101 identities, 15 cameras

**Loss function**:
```
L = L_triplet + L_ce + lambda * L_center
```

- **Triplet loss** (`L_triplet`): Learns relative distances — pulls same-identity pairs closer, pushes different-identity pairs apart. Uses hard mining (hardest positive, hardest negative within a batch).
- **Cross-entropy loss** (`L_ce`): Treats each identity as a class. Provides strong gradient signal early in training when triplet loss is noisy.
- **Center loss** (`L_center`): Penalizes distance of each feature from its class center, reducing intra-class variance. Weight `lambda = 0.0005` (small, acts as regularizer).

**Training hyperparameters** (from FastReID SBS config):
- Optimizer: Adam, lr=3.5e-4, weight decay=5e-4
- Batch: 64 images (16 identities x 4 images per identity — PK sampling)
- Epochs: 120, with cosine LR decay
- Augmentation: random horizontal flip, random erasing (p=0.5), auto-augment
- Warmup: 10 epochs linear warmup

---

## 3. ONNX Export Pipeline

### 3.1 Export Procedure

FastReID models are PyTorch-native. Exporting to ONNX decouples the inference dependency from the FastReID codebase (which pulls in ~500MB of dependencies including PyTorch, detectron2, faiss, etc.).

**Export script** (run once, offline):

```python
import torch
import onnx
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

# 1. Load FastReID model
cfg = get_cfg()
cfg.merge_from_file("configs/Market1501/sbs_R50.yml")
cfg.MODEL.BACKBONE.PRETRAIN = False
model = build_model(cfg)
Checkpointer(model).load("market_sbs_R50.pth")
model.eval()

# 2. Extract the visual backbone + pooling + BN neck + projection
# FastReID's forward() returns dict with 'features' key
# We need to trace just the visual pipeline

# 3. Trace with dummy input — dynamic batch dimension
dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(
    model.backbone,        # backbone only (up to 2048-d)
    dummy,
    "fastreid_sbs_s50.onnx",
    input_names=["images"],
    output_names=["embeddings"],
    dynamic_axes={
        "images": {0: "batch"},
        "embeddings": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)

# 4. Add projection head as a separate step or bake it into the ONNX graph
# If baked in: output is 256-d; if not: output is 2048-d and projection is done in numpy

# 5. Simplify graph
import onnxsim
model_onnx = onnx.load("fastreid_sbs_s50.onnx")
model_simplified, check = onnxsim.simplify(model_onnx)
assert check, "ONNX simplification failed"
onnx.save(model_simplified, "fastreid_sbs_s50.onnx")
```

**Alternative**: Use FastReID's built-in export utility:
```bash
python tools/deploy/onnx_export.py \
    --config-file configs/Market1501/sbs_R50.yml \
    --name fastreid_sbs_s50 \
    --output outputs/onnx \
    --opts MODEL.WEIGHTS market_sbs_R50.pth
```

### 3.2 ONNX Model Specifications

| Property | Value |
|---|---|
| Input name | `images` |
| Input shape | `(batch, 3, 256, 128)` — NCHW, float32, normalized |
| Output name | `embeddings` |
| Output shape | `(batch, 2048)` or `(batch, 256)` if projection baked in |
| Dynamic axes | Batch dimension (axis 0) |
| Opset version | 17 |
| File size | ~100 MB (float32), ~25 MB (float16) |
| Params | ~25.6M (backbone) + ~0.5M (non-local) |

### 3.3 Preprocessing in NumPy

The ONNX model expects ImageNet-normalized float32 tensors. Preprocessing must be done in the caller (not baked into the ONNX graph) to maintain flexibility:

```python
import numpy as np
import cv2

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_crops(
    frame: np.ndarray,               # Full frame (H, W, 3), BGR, uint8
    boxes: list[tuple[float, float, float, float]],  # xyxy
    target_size: tuple[int, int] = (256, 128),        # (height, width)
) -> np.ndarray:
    """Extract, resize, normalize person crops for FastReID.

    Returns:
        (N, 3, 256, 128) float32 array, ImageNet-normalized.
    """
    if not boxes:
        return np.empty((0, 3, *target_size), dtype=np.float32)

    crops = []
    h_frame, w_frame = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        # Clamp to frame boundaries
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(w_frame, int(x2))
        iy2 = min(h_frame, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            # Degenerate box — use zeros
            crops.append(np.zeros((*target_size, 3), dtype=np.float32))
            continue

        crop = frame[iy1:iy2, ix1:ix2]
        # BGR -> RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Resize to (256, 128) with bilinear interpolation
        crop = cv2.resize(crop, (target_size[1], target_size[0]),
                          interpolation=cv2.INTER_LINEAR)
        # uint8 -> float32 [0, 1]
        crop = crop.astype(np.float32) / 255.0
        # ImageNet normalize
        crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
        crops.append(crop)

    # Stack and transpose: (N, H, W, 3) -> (N, 3, H, W)
    batch = np.stack(crops, axis=0)
    batch = np.transpose(batch, (0, 3, 1, 2))
    return np.ascontiguousarray(batch)
```

### 3.4 ONNX Runtime Execution Providers

The FastReID ONNX model runs through the same ORT infrastructure as the YOLO backend. Provider selection follows yowo's existing priority chain:

| Priority | Provider | Expected Latency (20 crops) | Notes |
|---|---|---|---|
| 1 | TensorRTExecutionProvider | 1-2 ms | Best GPU throughput; requires TRT engine build |
| 2 | CUDAExecutionProvider | 2-4 ms | Standard CUDA path |
| 3 | CoreMLExecutionProvider | 3-5 ms | macOS Neural Engine acceleration |
| 4 | CPUExecutionProvider | 5-10 ms | Fallback; acceptable for <30 FPS streams |

Thread tuning for CPU EP follows the same convention as yowo's ONNX backend: `intra_op = cpu_count // 2`, `inter_op = cpu_count // 4`.

### 3.5 Post-Processing: L2 Normalization + Optional Projection

If the ONNX model outputs 2048-d backbone features (no baked-in projection):

```python
def project_and_normalize(features: np.ndarray, proj_weight: np.ndarray) -> np.ndarray:
    """Project 2048-d to 256-d and L2 normalize.

    Args:
        features: (N, 2048) backbone features.
        proj_weight: (256, 2048) projection matrix.

    Returns:
        (N, 256) L2-normalized embeddings.
    """
    projected = features @ proj_weight.T  # (N, 256)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # prevent division by zero
    return projected / norms
```

If the ONNX model outputs 256-d features (projection baked in), only L2 normalization is needed:

```python
def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms
```

---

## 4. Cost Matrix Fusion — BoT-SORT Formula

### 4.1 Mathematical Formulation

For each candidate pair of track `i` and detection `j`, the fused cost is computed as:

**Step 1 — Compute individual distances:**

```
d_IoU(i, j) = 1 - IoU(predicted_box_i, detected_box_j)

d_cos(i, j) = 1 - (e_i^T * e_j)
```

where `e_i` is the track's current embedding (256-d, L2-normalized) and `e_j` is the detection's freshly extracted embedding.

**Step 2 — Gated appearance distance:**

```
        | 0.5 * d_cos(i, j)   if d_cos(i, j) < theta_e  AND  d_IoU(i, j) < theta_IoU
d_hat_cos(i, j) = |
        | 1.0                 otherwise
```

Default thresholds: `theta_e = 0.25`, `theta_IoU = 0.5`.

**Step 3 — Fused cost:**

```
C(i, j) = min(d_IoU(i, j), d_hat_cos(i, j))
```

### 4.2 Why Dual-Gating (Not Weighted Sum)

The design choice of `min()` over a weighted combination `alpha * d_IoU + (1 - alpha) * d_cos` is deliberate:

1. **Failure mode isolation**: A weighted sum always contaminates both signals. If appearance matching fails completely (wrong person, different clothing), the cosine distance injects a wrong gradient into the cost. With `min()`, the IoU distance acts as an independent fallback — a spatially close match is accepted regardless of appearance when the appearance gate fails (d_hat_cos = 1.0).

2. **Threshold interpretability**: The dual gate `d_cos < theta_e AND d_IoU < theta_IoU` ensures appearance matching only activates when *both* signals agree the match is plausible. This prevents catastrophic false associations in two specific failure modes:
   - **Appearance-only trap**: Two different people wearing identical uniforms (d_cos < theta_e) but spatially separated (d_IoU > theta_IoU). The IoU gate blocks the false appearance match.
   - **IoU-only trap**: Two different people crossing paths (d_IoU < theta_IoU) but visually distinct (d_cos > theta_e). The appearance gate blocks, and the cost falls back to d_IoU, which may or may not produce a match depending on the exact IoU value.

3. **Scaling factor 0.5**: When the dual gate passes, d_hat_cos = 0.5 * d_cos. This scales the appearance distance into a range where it can actually "win" the `min()` against d_IoU. Without this scaling, d_cos values (typically 0.05-0.20 for same-identity pairs) would rarely be smaller than d_IoU values (typically 0.1-0.4 for nearby tracks), making appearance information dead weight.

### 4.3 Comparison with Deep OC-SORT Adaptive Weighting

Deep OC-SORT uses a different fusion strategy:

```
C(i, j) = d_IoU(i, j) + (a_w + w_b(m, n)) * A_c(i, j)
```

where:
- `a_w` is a global appearance weight (hyperparameter)
- `w_b(m, n)` is a per-pair discriminative weight computed from the variance of appearance distances within the cost matrix row/column
- `A_c(i, j)` is the appearance cost (cosine distance)

**Why BoT-SORT's approach is preferred for yowo:**

| Criterion | BoT-SORT (min + gate) | Deep OC-SORT (adaptive sum) |
|---|---|---|
| Compute overhead | O(1) per pair (two comparisons + min) | O(N+M) per pair (row/column variance) |
| Tunable params | 2 thresholds (theta_e, theta_IoU) | 3+ params (a_w, w_b formula, scaling) |
| Failure containment | Appearance failure -> automatic IoU fallback | Appearance failure contaminates total cost |
| Implementation complexity | ~15 lines | ~40 lines + matrix statistics |
| MOT17 HOTA | 65.0 | 64.9 |

The marginal HOTA difference (0.1) does not justify the additional complexity for a production system. BoT-SORT's gated fusion is chosen as the baseline.

### 4.4 Vectorized Implementation

```python
def gated_fused_cost(
    iou_cost: np.ndarray,         # (N, M), d_IoU
    appearance_cost: np.ndarray,  # (N, M), d_cos
    theta_e: float = 0.25,
    theta_iou: float = 0.5,
) -> np.ndarray:
    """BoT-SORT gated cost matrix fusion.

    Returns:
        (N, M) fused cost matrix: C(i,j) = min(d_IoU, d_hat_cos).
    """
    # Dual gate: both appearance and IoU must be below thresholds
    gate = (appearance_cost < theta_e) & (iou_cost < theta_iou)

    # Gated appearance distance: 0.5 * d_cos if gate passes, else 1.0
    gated_appearance = np.where(gate, 0.5 * appearance_cost, 1.0)

    # Fused cost: element-wise minimum
    return np.minimum(iou_cost, gated_appearance)
```

---

## 5. Embedding Update Strategy

### 5.1 EMA (Exponential Moving Average)

When a track is matched to a detection, its stored embedding is updated via EMA:

```
e_track = eta * e_track + (1 - eta) * e_new
e_track = e_track / ||e_track||_2     # Re-normalize to unit sphere
```

Default `eta = 0.9` (high momentum, favoring the existing track embedding).

### 5.2 Why EMA, Not Gallery Bank

**DeepSORT approach (gallery bank)**: Each track stores the last 100 detection embeddings in a FIFO buffer. The matching distance is computed as the minimum distance between any stored embedding and the new detection:

```
d(track, det) = min_{k=1..100} d_cos(gallery_k, e_det)
```

**Problems with gallery bank for 24/7 CCTV:**

1. **Unbounded memory growth**: 100 embeddings x 256-d x 4 bytes = 102 KB per track. With 200 tracks, this is 20 MB. If tracks live for thousands of frames (re-appearing person in a store), the gallery grows without bound unless explicitly capped, and capping loses old appearance information.

2. **Stale embedding pollution**: In a gallery of 100 embeddings collected over 100 frames, early embeddings may represent the person under different lighting, angle, or partial occlusion. When the person re-appears, the minimum-distance matching may select a stale embedding that happens to be close to an unrelated person, causing a false association.

3. **Compute scaling**: Computing minimum distance against a gallery of K embeddings requires K dot products per pair, turning the NxM cost matrix into an NxMxK operation. At K=100, N=50 tracks, M=20 detections, this is 100,000 dot products per frame vs. 1,000 for EMA.

**EMA advantages:**

1. **Constant memory**: 1 embedding per track, always. 200 tracks = 200 KB, permanently.
2. **Temporal smoothing**: EMA with eta=0.9 means the embedding is a weighted average where the last 10 frames contribute ~65% of the signal and frames older than 30 contribute <5%. This naturally adapts to gradual appearance changes (lighting, pose) while forgetting stale information.
3. **O(1) update**: A single weighted average + L2 normalization per matched track per frame.

### 5.3 Momentum Value Selection

The momentum `eta = 0.9` is the default from BoT-SORT and works well for 24-30 FPS video where appearance changes are gradual between consecutive frames. The effective memory window is:

```
Weight of frame t-k in current embedding: (1 - eta) * eta^k
Half-life: k_{1/2} = -ln(2) / ln(eta) ≈ 6.6 frames (at eta=0.9)
95% decay: k_{95} = -ln(20) / ln(eta) ≈ 28.4 frames (~1 second at 30 FPS)
```

For lower frame rates (10-15 FPS) or rapid appearance changes (rotating person, changing lighting), a lower `eta = 0.8` may be appropriate. This is exposed as a configurable parameter.

### 5.4 Memory Analysis

| Component | Per Track | 200 Tracks | 500 Tracks |
|---|---|---|---|
| 256-d float32 embedding | 1,024 bytes | 200 KB | 500 KB |
| EMA momentum scalar | 4 bytes | 800 bytes | 2 KB |
| L2 norm (cached) | 4 bytes | 800 bytes | 2 KB |
| **Total** | **1,032 bytes** | **~201 KB** | **~502 KB** |

This is negligible compared to the ONNX model itself (~100 MB) and the YOLO backbone's runtime memory.

---

## 6. Association Cascade Integration

### 6.1 Current ByteTracker Flow (IoU-only, v2.1.0)

```
Frame N Detection
    |
    v
[Split by confidence]
    |-- High-conf (>= track_high_thresh)
    |-- Low-conf  (>= track_low_thresh, < track_high_thresh)
    |
    v
[Kalman Predict all tracks (tracked + lost)]
    |
    v
[STAGE 1] High-conf dets <-> (tracked + lost) pool
    |-- Cost matrix: iou_distance(pool, high_dets)
    |-- Hungarian assignment (thresh = match_thresh = 0.8)
    |-- Matched: update() or re_activate()
    |-- Unmatched tracks (TRACKED only) -> stage 2 candidates
    |-- Unmatched dets -> new track candidates
    |
    v
[STAGE 2] Low-conf dets <-> unmatched tracked tracks
    |-- Cost matrix: iou_distance(unmatched_tracked, low_dets)
    |-- Hungarian assignment (thresh = 0.5)
    |-- Matched: update()
    |-- Unmatched tracks -> mark_lost()
    |
    v
[Birth/Death]
    |-- Unmatched high-conf dets (conf >= new_track_thresh) -> new STrack
    |-- Lost tracks exceeding max_age -> mark_removed()
    |
    v
[Duplicate Removal]
    |-- remove_duplicate_tracks(tracked, lost)
    |
    v
TrackedDetection output
```

### 6.2 Modified Flow with FastReID (Proposed)

```
Frame N Detection
    |
    v
[Split by confidence]
    |-- High-conf (>= track_high_thresh)
    |-- Low-conf  (>= track_low_thresh, < track_high_thresh)
    |
    v
[Kalman Predict all tracks (tracked + lost)]
    |
    v
[EXTRACT EMBEDDINGS]  <-- NEW
    |-- Crop high-conf detection boxes from frame
    |-- Run FastReID ONNX: crops -> 256-d embeddings
    |-- (Low-conf dets: NO embedding extraction — unreliable crops)
    |
    v
[STAGE 1] High-conf dets <-> (tracked + lost) pool     <-- MODIFIED
    |-- IoU cost: iou_distance(pool, high_dets)
    |-- Appearance cost: cosine_distance(pool_embeddings, det_embeddings)
    |-- Fused cost: gated_fused_cost(iou_cost, appearance_cost)
    |-- Hungarian assignment (thresh = match_thresh = 0.8)
    |-- Matched: update() + update_embedding(EMA)      <-- MODIFIED
    |-- Unmatched tracks (TRACKED only) -> stage 2 candidates
    |-- Unmatched dets -> stage 3 or new track candidates
    |
    v
[STAGE 2] Low-conf dets <-> unmatched tracked tracks    (UNCHANGED)
    |-- Cost matrix: iou_distance(unmatched_tracked, low_dets)
    |-- Hungarian assignment (thresh = 0.5)
    |-- Matched: update() (NO embedding update — low-conf crop unreliable)
    |-- Unmatched tracks -> mark_lost()
    |
    v
[STAGE 3] Unmatched high-conf dets <-> lost tracks      <-- NEW
    |-- Appearance cost only: cosine_distance(lost_embeddings, unmatched_det_embeddings)
    |-- Gate: only pairs with d_cos < theta_e
    |-- Hungarian assignment (thresh = theta_e = 0.25)
    |-- Matched: re_activate() + update_embedding()
    |-- Unmatched dets -> new track candidates
    |
    v
[Birth/Death]
    |-- New STrack: initialize _embedding from detection embedding
    |-- Lost tracks exceeding max_age -> mark_removed()
    |
    v
[Duplicate Removal]
    |-- remove_duplicate_tracks(tracked, lost)
    |
    v
TrackedDetection output
```

### 6.3 Design Decisions

**Why Stage 2 remains IoU-only:**

Low-confidence detections (between `track_low_thresh=0.1` and `track_high_thresh=0.6`) are frequently partial detections — clipped by frame edges, heavily occluded, or small/far away. The resulting crops produce unreliable embeddings because:
- Partial crops may contain majority background pixels
- Small crops (e.g., 20x40 pixels) produce severe aliasing when resized to 256x128
- Low-confidence detections are disproportionately false positives

Using IoU-only for Stage 2 matches the original ByteTrack design and avoids contaminating track embeddings with noisy features.

**Why Stage 3 (appearance rescue) uses appearance-only:**

Long-lost tracks (>10 frames) have Kalman predictions that have drifted significantly from the true position. Their predicted bounding boxes are inaccurate, making IoU unreliable. However, their stored embeddings (from the last confident match) are still valid for appearance comparison. Stage 3 exploits this asymmetry:
- IoU between a drifted Kalman prediction and a re-appearing person is near zero
- Cosine distance between stored embedding and fresh embedding is still small (<0.25) for the same identity

This stage specifically targets the re-entry-after-long-occlusion failure mode.

**Minimum crop area gate:**

Crops smaller than 1,024 pixels (e.g., 32x32) are too small for meaningful feature extraction. A minimum area gate skips embedding extraction for tiny detections:

```python
MIN_CROP_AREA = 1024  # pixels (e.g., 32x32)

def should_extract_embedding(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return area >= MIN_CROP_AREA
```

Tracks matching small detections have their embedding unchanged (EMA update skipped).

---

## 7. Implementation Specification

### 7.1 File-by-File Changes

All files within `src/yowo/tracking/`. Every file stays under 700 lines.

#### 7.1.1 `_reid.py` (NEW) — FastReID Extractor

**Purpose**: ONNX Runtime-based ReID feature extractor implementing a `ReIDExtractor` Protocol.

**Estimated lines**: ~180

```python
"""FastReID SBS-S50 appearance feature extractor for ByteTrack."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ReIDExtractor(Protocol):
    """Protocol for pluggable ReID embedding extractors.

    Any class implementing this protocol can be used with ByteTracker
    for appearance-based association.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings."""
        ...

    def extract(
        self,
        frame: NDArray[np.uint8],
        boxes: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]:
        """Extract embeddings for cropped detections.

        Args:
            frame: Full frame (H, W, 3), BGR, uint8.
            boxes: List of (x1, y1, x2, y2) detection boxes.

        Returns:
            (N, embedding_dim) L2-normalized float32 embeddings.
            If boxes is empty, returns shape (0, embedding_dim).
        """
        ...


# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Minimum crop area (pixels) for reliable embedding extraction
MIN_CROP_AREA: int = 1024


class FastReIDExtractor:
    """FastReID SBS-S50 ONNX Runtime embedding extractor.

    Loads an ONNX-exported FastReID model and extracts 256-d L2-normalized
    embeddings from person crops. Uses the same ORT session management
    pattern as yowo's ONNX backend.

    Args:
        model_path: Path to .onnx file (FastReID SBS-S50 export).
        embedding_dim: Expected output embedding dimensionality. Default 256.
        providers: ORT execution providers. If None, auto-selects.
        target_size: (height, width) for crop resize. Default (256, 128).
        min_crop_area: Minimum crop area in pixels. Smaller crops get
            zero embeddings. Default 1024.
    """

    __slots__ = (
        "_embedding_dim",
        "_min_crop_area",
        "_session",
        "_input_name",
        "_output_name",
        "_target_size",
    )

    def __init__(
        self,
        model_path: str,
        *,
        embedding_dim: int = 256,
        providers: list[str] | None = None,
        target_size: tuple[int, int] = (256, 128),
        min_crop_area: int = MIN_CROP_AREA,
    ) -> None: ...

    @property
    def embedding_dim(self) -> int: ...

    def extract(
        self,
        frame: NDArray[np.uint8],
        boxes: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32]: ...

    def _preprocess_crops(
        self,
        frame: NDArray[np.uint8],
        boxes: list[tuple[float, float, float, float]],
    ) -> tuple[NDArray[np.float32], list[bool]]: ...
        # Returns (batch_tensor, valid_mask) where invalid crops get zeros
```

**Key contracts:**
- `extract()` always returns `(len(boxes), embedding_dim)` — invalid/small crops get zero vectors
- Zero-vector embeddings produce `d_cos = 1.0` (maximum distance), ensuring they never match by appearance
- Thread-safe: ORT session is thread-safe for inference after creation
- `isinstance(extractor, ReIDExtractor)` returns `True` (runtime_checkable Protocol)

#### 7.1.2 `_strack.py` (MODIFIED) — Add Embedding Slot

**Changes**: Add `_embedding` slot and `update_embedding()` method.

**New lines**: ~25

```python
# In STrack.__slots__, add:
"_embedding",  # NDArray[np.float32] | None

# In STrack.__init__(), add:
self._embedding: NDArray[np.float32] | None = None

# New method:
def update_embedding(
    self,
    new_embedding: NDArray[np.float32],
    momentum: float = 0.9,
) -> None:
    """Update track embedding via EMA, then L2 renormalize.

    Args:
        new_embedding: (D,) L2-normalized detection embedding.
        momentum: EMA weight for existing embedding. Default 0.9.
    """
    if self._embedding is None:
        self._embedding = new_embedding.copy()
        return
    self._embedding = momentum * self._embedding + (1.0 - momentum) * new_embedding
    norm = np.linalg.norm(self._embedding)
    if norm > 1e-12:
        self._embedding /= norm

@property
def embedding(self) -> NDArray[np.float32] | None:
    """Current track embedding, or None if not yet initialized."""
    return self._embedding
```

**File size after change**: ~320 lines (within 700-line limit).

#### 7.1.3 `_matching.py` (MODIFIED) — Add Appearance Matching

**Changes**: Add `cosine_distance()`, `gated_fused_cost()`, and `appearance_distance()` functions.

**New lines**: ~80

```python
def cosine_distance(
    track_embeddings: NDArray[np.float32],  # (N, D)
    det_embeddings: NDArray[np.float32],    # (M, D)
) -> NDArray[np.float64]:
    """Cosine distance matrix between track and detection embeddings.

    Args:
        track_embeddings: (N, D) L2-normalized track embeddings.
        det_embeddings: (M, D) L2-normalized detection embeddings.

    Returns:
        (N, M) distance matrix where d(i,j) = 1 - dot(e_i, e_j).
        Range [0, 2] for unit-norm vectors (0 = identical, 2 = opposite).
    """
    ...

def appearance_distance(
    tracks: Sequence[STrack],
    det_embeddings: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Cosine distance between track embeddings and detection embeddings.

    Tracks without embeddings get distance 1.0 (no appearance info).

    Args:
        tracks: Sequence of STrack objects.
        det_embeddings: (M, D) detection embeddings.

    Returns:
        (N, M) appearance cost matrix.
    """
    ...

def gated_fused_cost(
    iou_cost: NDArray[np.float64],
    appearance_cost: NDArray[np.float64],
    *,
    theta_e: float = 0.25,
    theta_iou: float = 0.5,
) -> NDArray[np.float64]:
    """BoT-SORT gated cost matrix fusion.

    C(i,j) = min(d_IoU(i,j), d_hat_cos(i,j))
    where d_hat_cos = 0.5 * d_cos if d_cos < theta_e AND d_IoU < theta_iou, else 1.0.

    Args:
        iou_cost: (N, M) IoU distance matrix.
        appearance_cost: (N, M) cosine distance matrix.
        theta_e: Appearance distance threshold. Default 0.25.
        theta_iou: IoU distance threshold for gate. Default 0.5.

    Returns:
        (N, M) fused cost matrix.
    """
    ...
```

**File size after change**: ~400 lines (within 700-line limit).

#### 7.1.4 `_tracker.py` (MODIFIED) — Accept ReIDExtractor, Modify update()

**Changes**: Add `reid_extractor` parameter to `__init__`, modify `update()` to perform 3-stage association with appearance when extractor is provided.

**New lines**: ~60

```python
class ByteTracker:
    def __init__(
        self,
        *,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
        reid_extractor: ReIDExtractor | None = None,  # NEW
        embedding_momentum: float = 0.9,               # NEW
        appearance_thresh: float = 0.25,                # NEW (theta_e)
        appearance_iou_gate: float = 0.5,               # NEW (theta_IoU)
    ) -> None:
        ...
        self._reid = reid_extractor
        self._embedding_momentum = embedding_momentum
        self._appearance_thresh = appearance_thresh
        self._appearance_iou_gate = appearance_iou_gate
```

**Critical backward compatibility guarantee**: When `reid_extractor is None`, `update()` produces **bit-identical** results to v2.1.0. The appearance code paths are gated behind `if self._reid is not None:` checks. No existing tests break.

**update() pseudocode with ReID:**

```python
def update(self, detection: Detection) -> TrackedDetection:
    # ... (existing split by confidence, Kalman predict) ...

    # --- NEW: Extract embeddings for high-conf detections ---
    det_embeddings: NDArray[np.float32] | None = None
    if self._reid is not None and high_boxes:
        det_embeddings = self._reid.extract(detection.frame.pixels, high_boxes)

    # --- STAGE 1 (MODIFIED): Fused IoU + appearance ---
    cost1 = iou_distance(strack_pool, high_arr)
    if det_embeddings is not None:
        app_cost = appearance_distance(strack_pool, det_embeddings)
        cost1 = gated_fused_cost(cost1, app_cost,
                                  theta_e=self._appearance_thresh,
                                  theta_iou=self._appearance_iou_gate)
    matches1, unmatched_track_idxs, unmatched_high_idxs = linear_assignment(
        cost1, self._match_thresh
    )

    # Update matched tracks + EMA embedding update
    for ti, di in matches1:
        track = strack_pool[ti]
        # ... (existing update/re_activate logic) ...
        if det_embeddings is not None:
            track.update_embedding(det_embeddings[di], self._embedding_momentum)

    # --- STAGE 2 (UNCHANGED): IoU-only for low-conf ---
    # ... (existing code, no modification) ...

    # --- STAGE 3 (NEW): Appearance rescue for long-lost tracks ---
    if self._reid is not None and unmatched_high_after_stage1:
        # Filter: only lost tracks with embeddings + unmatched high-conf dets with embeddings
        lost_with_emb = [t for t in self._lost
                         if t.embedding is not None and t.track_id not in refound_ids]
        if lost_with_emb and unmatched_high_after_stage1:
            unmatched_det_embs = det_embeddings[unmatched_high_after_stage1_indices]
            app_cost3 = appearance_distance(lost_with_emb, unmatched_det_embs)
            matches3, _, still_unmatched_dets = linear_assignment(
                app_cost3, self._appearance_thresh
            )
            for ti, di in matches3:
                track = lost_with_emb[ti]
                det_idx = unmatched_high_after_stage1_indices[di]
                track.re_activate(high_boxes[det_idx], ...)
                track.update_embedding(det_embeddings[det_idx], self._embedding_momentum)
                refound.append(track)

    # ... (existing birth/death, duplicate removal) ...
```

**File size after change**: ~300 lines (within 700-line limit).

#### 7.1.5 `__init__.py` (MODIFIED) — Add Exports

**Changes**: Export `ReIDExtractor` and `FastReIDExtractor`.

```python
from yowo.tracking._reid import FastReIDExtractor, ReIDExtractor

__all__ = [
    "ByteTracker",
    "FastReIDExtractor",    # NEW
    "ReIDExtractor",        # NEW
    "TrackState",
    "TrackedBox",
    "TrackedDetection",
    "track_detections",
    "track_stream",
]
```

### 7.2 File Size Summary

| File | Current Lines | Added Lines | Total | Under 700? |
|---|---|---|---|---|
| `_reid.py` (NEW) | 0 | ~180 | ~180 | Yes |
| `_strack.py` | 296 | ~25 | ~321 | Yes |
| `_matching.py` | 317 | ~80 | ~397 | Yes |
| `_tracker.py` | 240 | ~60 | ~300 | Yes |
| `__init__.py` | 85 | ~5 | ~90 | Yes |
| **Total** | 938 | ~350 | ~1,288 | N/A |

### 7.3 Dependency Management

FastReID ONNX integration requires only `onnxruntime`, which yowo already depends on. No new dependencies are needed at runtime.

For exporting the ONNX model (one-time offline step), the user needs the `fastreid` package. This is NOT a yowo runtime dependency:

```toml
# pyproject.toml — optional dependency group
[project.optional-dependencies]
reid = ["onnxruntime>=1.16.0"]  # ORT already in base deps
# No fastreid dependency — export is a one-time offline step
```

### 7.4 Public API Surface

```python
# User-facing API for ReID-enhanced tracking
from yowo.tracking import ByteTracker, FastReIDExtractor

# Initialize extractor
extractor = FastReIDExtractor("fastreid_sbs_s50.onnx")

# Create tracker with ReID
tracker = ByteTracker(reid_extractor=extractor)

# Use in stream
from yowo.tracking import track_stream
for tracked in track_stream(engine, source, tracker=tracker):
    ...

# Or: use Protocol for custom extractors
from yowo.tracking import ReIDExtractor

class MyCustomExtractor:
    @property
    def embedding_dim(self) -> int:
        return 128

    def extract(self, frame, boxes):
        ...  # Custom implementation

tracker = ByteTracker(reid_extractor=MyCustomExtractor())
```

---

## 8. Performance Analysis

### 8.1 Latency Breakdown

#### Server GPU (NVIDIA A100/V100, CUDA EP)

| Operation | Latency (20 dets) | Latency (50 dets) | Notes |
|---|---|---|---|
| Crop extraction + preprocess | 0.3 ms | 0.8 ms | cv2.resize + numpy normalize |
| FastReID ONNX inference | 1.5 ms | 2.5 ms | Batched, CUDA EP, 20-50 crops |
| Cost matrix fusion | 0.05 ms | 0.1 ms | Vectorized numpy |
| EMA embedding update | 0.01 ms | 0.02 ms | Per-matched-track |
| Kalman + association (existing) | 0.3 ms | 0.7 ms | Unchanged from v2.1.0 |
| **Total tracking overhead** | **2.2 ms** | **4.1 ms** | |
| YOLO detect (reference) | 5-15 ms | 5-15 ms | Detection-count independent |
| **Total frame time** | **7-17 ms** | **9-19 ms** | Well under 100 ms target |

#### Desktop CPU (Intel i7/AMD Ryzen, CPU EP)

| Operation | Latency (20 dets) | Latency (50 dets) | Notes |
|---|---|---|---|
| Crop extraction + preprocess | 0.5 ms | 1.2 ms | cv2.resize + numpy normalize |
| FastReID ONNX inference | 8 ms | 15 ms | CPU EP, multi-threaded |
| Cost matrix fusion | 0.05 ms | 0.1 ms | Vectorized numpy |
| EMA embedding update | 0.01 ms | 0.02 ms | Per-matched-track |
| Kalman + association (existing) | 0.5 ms | 1.2 ms | Unchanged |
| **Total tracking overhead** | **9 ms** | **17 ms** | |
| YOLO detect (reference) | 30-80 ms | 30-80 ms | CPU-dependent |
| **Total frame time** | **39-89 ms** | **47-97 ms** | Near 100 ms target at high det counts |

#### Apple Silicon (M4 Pro, CoreML EP)

| Operation | Latency (20 dets) | Latency (50 dets) | Notes |
|---|---|---|---|
| Crop extraction + preprocess | 0.4 ms | 1.0 ms | cv2.resize + numpy normalize |
| FastReID ONNX inference | 3 ms | 5 ms | CoreML EP, Neural Engine |
| Cost matrix fusion | 0.05 ms | 0.1 ms | Vectorized numpy |
| EMA embedding update | 0.01 ms | 0.02 ms | Per-matched-track |
| Kalman + association (existing) | 0.3 ms | 0.7 ms | Unchanged |
| **Total tracking overhead** | **3.8 ms** | **6.8 ms** | |
| YOLO detect (reference) | 6-37 ms | 6-37 ms | CoreML backend (v2.1.0 benchmark) |
| **Total frame time** | **10-41 ms** | **13-44 ms** | Comfortable under 100 ms |

#### Edge Device (Jetson Orin Nano, TensorRT EP)

| Operation | Latency (20 dets) | Latency (50 dets) | Notes |
|---|---|---|---|
| Crop extraction + preprocess | 0.8 ms | 2.0 ms | ARM CPU, cv2.resize |
| FastReID ONNX inference | 3 ms | 6 ms | TensorRT EP, FP16 |
| Cost matrix fusion | 0.1 ms | 0.2 ms | Vectorized numpy |
| EMA embedding update | 0.02 ms | 0.05 ms | Per-matched-track |
| Kalman + association (existing) | 0.5 ms | 1.2 ms | Unchanged |
| **Total tracking overhead** | **4.4 ms** | **9.5 ms** | |
| YOLO detect (reference) | 15-50 ms | 15-50 ms | TensorRT backend |
| **Total frame time** | **19-55 ms** | **25-60 ms** | Within 100 ms target |

### 8.2 Throughput Estimates

| Platform | FPS (no ReID) | FPS (with ReID, 20 dets) | Overhead % |
|---|---|---|---|
| Server GPU (A100) | 150-200 | 80-120 | 15-25% |
| Desktop CPU (i7) | 12-30 | 10-25 | 10-30% |
| Apple M4 Pro (CoreML) | 82-147 | 60-100 | 15-25% |
| Jetson Orin Nano (TRT) | 20-60 | 15-45 | 15-30% |

### 8.3 Memory Footprint

| Component | Size | Notes |
|---|---|---|
| FastReID ONNX model (FP32) | ~100 MB | Loaded once, shared across streams |
| FastReID ONNX model (FP16) | ~50 MB | If quantized |
| ORT session workspace | ~50-100 MB | Execution provider dependent |
| Per-track embeddings (200 tracks) | 200 KB | 256-d x float32 x 200 |
| Per-track embeddings (500 tracks) | 500 KB | Upper bound for dense scenes |
| Crop preprocessing buffer | ~6 MB | 50 crops x 256 x 128 x 3 x float32 |
| **Total runtime overhead** | **~160-210 MB** | Model + session + buffers |

---

## 9. Strengths and Limitations

### 9.1 Strengths

1. **Proven on pedestrian tracking benchmarks**: FastReID SBS-S50 was the exact model used in BoT-SORT (1st MOT17, 1st MOT20) and Deep OC-SORT (1st MOT20). Its pedestrian ReID accuracy (mAP 91.3% Market-1501, 83.1% MSMT17) is the highest among models with <30M parameters.

2. **Quantified IDS reduction**: BoT-SORT with FastReID reduces identity switches from 2196 (ByteTrack) to 1212 (BoT-SORT) on MOT17 — a 44.8% reduction. HOTA improves from 63.1 to 65.0 (+3.0%). These are real-world benchmarks on crowded surveillance sequences.

3. **Compact 256-d embeddings**: After projection, embeddings are only 1,024 bytes per track. Cosine distance between N tracks and M detections is a single matrix multiply — no iterative search or approximate nearest-neighbor structures needed.

4. **Low latency**: At ~25M parameters, SBS-S50 is significantly smaller than transformer-based alternatives (CLIP ViT-B/16 at ~86M, CLIP ViT-L/14 at ~304M). Batch inference for 20 crops takes 1.5-8 ms depending on hardware — well within the 100 ms frame budget.

5. **Mature ONNX export path**: FastReID provides official export scripts. The ResNet-50 backbone is fully supported by all ONNX Runtime execution providers without custom ops or dynamic shapes (except the batch dimension).

6. **Backward-compatible integration**: The `ReIDExtractor` Protocol design means `ByteTracker()` without `reid_extractor` produces identical results to v2.1.0. No existing tests break. ReID is purely additive.

7. **EMA embedding update**: Constant O(1) memory and compute per track update. No unbounded gallery growth. Natural temporal smoothing adapts to gradual appearance changes.

### 9.2 Limitations

1. **Domain-locked to pedestrians**: SBS-S50 is trained exclusively on person ReID datasets (Market-1501, DukeMTMC-reID, MSMT17). It produces meaningless embeddings for vehicles, animals, drones, or other object categories. Deploying on a vehicle-tracking CCTV system requires retraining on a vehicle ReID dataset (VeRi-776, VehicleID) — which requires labeled data and training infrastructure.

2. **Requires supervised ReID training data for new domains**: Unlike CLIP-based approaches that offer zero-shot embedding quality, FastReID requires (identity_label, camera_id, image) triplets for fine-tuning. Collecting and annotating this data for a new deployment scenario (e.g., retail store customer tracking) is expensive and time-consuming (typically 100+ hours of annotation for a production-quality dataset).

3. **256x128 aspect ratio assumption**: The portrait aspect ratio is optimal for standing pedestrians but produces distorted crops for:
   - Seated/crouching people (closer to square)
   - Cyclists (wider than tall)
   - People carrying large objects (irregular silhouette)
   - Non-person objects (vehicles are landscape-oriented)

4. **FastReID is a heavy development dependency**: While yowo only needs the exported ONNX model at runtime, the export step requires installing `fastreid` which pulls in ~500 MB of dependencies including specific versions of PyTorch, detectron2, and faiss. The FastReID repository (JDAI-CV/fast-reid) has seen declining maintenance:
   - Last major commit: 2022
   - Open issues: 400+
   - Python 3.10+ compatibility issues reported
   - No official PyTorch 2.x support

5. **FastReID repository maintenance risk**: If the FastReID repo becomes unmaintained, users cannot easily export new model variants or reproduce training. The ONNX model itself is self-contained, but the training pipeline depends on the upstream codebase.

6. **No zero-shot capability**: FastReID cannot be deployed on object categories it has not been explicitly trained on. A user tracking "shopping carts" or "forklifts" in a warehouse gets no benefit from SBS-S50 without collecting a domain-specific ReID dataset and retraining.

7. **Night/IR camera gap**: Training datasets are collected under visible-spectrum lighting. Infrared or night-vision cameras produce grayscale or false-color images that are out-of-distribution for the model. Cross-modality person ReID is an active research area but not solved by SBS-S50.

8. **Model file distribution**: The ~100 MB ONNX file must be distributed separately from the yowo package. Users need to download and provide the model path. There is no "batteries-included" experience without a model download step.

---

## 10. Testing Strategy

### 10.1 Unit Tests — `tests/unit/test_reid.py`

**Extractor initialization and configuration:**

```python
def test_fastreid_extractor_init_valid_model():
    """FastReIDExtractor loads ONNX model and reports correct embedding_dim."""

def test_fastreid_extractor_init_invalid_path():
    """FastReIDExtractor raises FileNotFoundError for missing model."""

def test_fastreid_extractor_implements_protocol():
    """FastReIDExtractor passes isinstance check against ReIDExtractor Protocol."""

def test_fastreid_extractor_embedding_dim():
    """embedding_dim property returns configured value (default 256)."""
```

**Extraction behavior:**

```python
def test_extract_empty_boxes_returns_empty():
    """extract(frame, []) returns shape (0, 256)."""

def test_extract_single_crop():
    """extract(frame, [one_box]) returns shape (1, 256), L2 normalized."""

def test_extract_batch_crops():
    """extract(frame, [20 boxes]) returns shape (20, 256), all L2 normalized."""

def test_extract_degenerate_box_returns_zero_vector():
    """Box with zero area produces zero embedding (d_cos = 1.0)."""

def test_extract_small_crop_below_min_area():
    """Crop smaller than MIN_CROP_AREA produces zero embedding."""

def test_extract_clamps_to_frame_boundaries():
    """Boxes extending beyond frame edges are clamped, not errored."""

def test_extract_output_l2_normalized():
    """All non-zero output embeddings have L2 norm == 1.0 (within 1e-6)."""
```

**Preprocessing:**

```python
def test_preprocess_bgr_to_rgb_conversion():
    """Preprocessing converts BGR frame to RGB before normalization."""

def test_preprocess_imagenet_normalization():
    """Pixel values are divided by 255 then normalized with ImageNet mean/std."""

def test_preprocess_resize_to_target_size():
    """Crops are resized to (256, 128) via bilinear interpolation."""

def test_preprocess_output_shape():
    """Preprocessed batch has shape (N, 3, 256, 128), dtype float32."""
```

### 10.2 Unit Tests — `tests/unit/test_matching_appearance.py`

**Cosine distance:**

```python
def test_cosine_distance_identical_embeddings():
    """Identical embeddings produce distance 0.0."""

def test_cosine_distance_orthogonal_embeddings():
    """Orthogonal embeddings produce distance 1.0."""

def test_cosine_distance_opposite_embeddings():
    """Opposite embeddings produce distance 2.0."""

def test_cosine_distance_batch_shape():
    """cosine_distance(N_tracks, M_dets) returns (N, M) matrix."""

def test_cosine_distance_empty_inputs():
    """Empty track or detection array returns correctly shaped zeros."""
```

**Gated fusion:**

```python
def test_gated_fused_cost_both_below_thresholds():
    """When d_cos < theta_e AND d_IoU < theta_IoU: C = min(d_IoU, 0.5*d_cos)."""

def test_gated_fused_cost_appearance_above_threshold():
    """When d_cos >= theta_e: d_hat_cos = 1.0, C = d_IoU (IoU fallback)."""

def test_gated_fused_cost_iou_above_threshold():
    """When d_IoU >= theta_IoU: d_hat_cos = 1.0, C = d_IoU (gate blocks)."""

def test_gated_fused_cost_empty_matrices():
    """Empty cost matrices return empty result."""

def test_gated_fused_cost_custom_thresholds():
    """Non-default theta_e, theta_iou are applied correctly."""

def test_appearance_distance_tracks_without_embeddings():
    """Tracks with None embedding get distance 1.0 for all detections."""
```

### 10.3 Unit Tests — `tests/unit/test_strack_embedding.py`

```python
def test_strack_embedding_initially_none():
    """New STrack has _embedding == None."""

def test_update_embedding_first_call_copies():
    """First update_embedding() sets embedding directly (no EMA)."""

def test_update_embedding_ema_momentum():
    """Second+ update applies EMA: new = 0.9*old + 0.1*new, then L2 norm."""

def test_update_embedding_l2_renormalized():
    """After EMA update, embedding L2 norm == 1.0."""

def test_update_embedding_custom_momentum():
    """momentum=0.5 applies equal weighting."""
```

### 10.4 Integration Tests — `tests/unit/test_tracker_reid_integration.py`

**Mock-based integration** (no actual ONNX model needed):

```python
class MockReIDExtractor:
    """Returns deterministic embeddings based on box position for testing."""

    @property
    def embedding_dim(self) -> int:
        return 256

    def extract(self, frame, boxes):
        # Deterministic: embedding[0] = hash of box center
        ...

def test_tracker_with_reid_reduces_ids_on_crossing_sequence():
    """Two objects crossing paths: IoU-only produces IDS, ReID-fused does not."""

def test_tracker_with_reid_recovers_lost_track_by_appearance():
    """Object disappears for 15 frames, re-appears: Stage 3 re-associates."""

def test_tracker_without_reid_bit_identical_to_v210():
    """ByteTracker(reid_extractor=None) produces identical output to v2.1.0."""

def test_tracker_reid_stage2_remains_iou_only():
    """Low-conf detections do not trigger embedding extraction."""

def test_tracker_reid_embedding_updated_on_match():
    """After Stage 1 match, track embedding is updated via EMA."""

def test_tracker_reid_no_embedding_update_on_low_conf_match():
    """Stage 2 low-conf matches do not update track embedding."""
```

### 10.5 Regression Tests

```python
def test_pure_iou_mode_identical_to_v210():
    """Run 100-frame synthetic sequence with ByteTracker():
    assert tracked_boxes == tracked_boxes_v210 (element-wise)."""

def test_reid_mode_superset_of_iou_mode():
    """ReID mode tracks at least as many objects as IoU-only mode
    (appearance rescue recovers lost tracks)."""
```

### 10.6 Performance Tests

```python
@pytest.mark.benchmark
def test_reid_overhead_20_detections():
    """Full tracking cycle with 20 dets: measure total_ms, assert < 15ms on CPU."""

@pytest.mark.benchmark
def test_reid_overhead_50_detections():
    """Full tracking cycle with 50 dets: measure total_ms, assert < 25ms on CPU."""

@pytest.mark.benchmark
def test_reid_overhead_100_detections():
    """Full tracking cycle with 100 dets: measure total_ms, assert < 50ms on CPU."""

@pytest.mark.benchmark
def test_embedding_extraction_throughput():
    """FastReIDExtractor.extract() throughput: crops/second for 1, 10, 20, 50 crops."""
```

### 10.7 Test Coverage Target

| Module | Target Coverage | Critical Paths |
|---|---|---|
| `_reid.py` | >90% | extract(), preprocess, Protocol |
| `_strack.py` (embedding additions) | >95% | update_embedding(), EMA, L2 norm |
| `_matching.py` (appearance additions) | >95% | cosine_distance(), gated_fused_cost() |
| `_tracker.py` (ReID integration) | >85% | 3-stage cascade, backward compat |

---

## 11. CCTV-Specific Considerations

### 11.1 Uniform Crowds (Factory/Warehouse Workers)

**Problem**: Workers in identical uniforms (e.g., factory jumpsuits, warehouse vests) have near-identical appearance. FastReID embeddings for two workers in the same uniform may have cosine distance < 0.1, making them indistinguishable by appearance alone.

**Mitigation**: The dual-gating mechanism (Section 4) prevents appearance from overriding IoU when both objects are visible. The `theta_iou = 0.5` gate ensures that appearance matching only activates when IoU is already ambiguous (objects at similar positions). In practice:
- When both workers are visible and non-overlapping, IoU clearly separates them (d_IoU > 0.5), and appearance is irrelevant
- When workers cross paths (d_IoU < 0.5), appearance cannot distinguish them either, so the cost falls back to IoU (d_hat_cos = 1.0 because d_cos > theta_e for same-uniform pairs)
- Net effect: ReID does not harm and does not help for same-uniform scenarios. Track continuity relies on Kalman prediction + IoU, same as pure ByteTrack

**Recommendation**: For uniform-heavy deployments, consider increasing `min_hits` to 5 (require more frames before confirming a track) and decreasing `max_age` to 15 (remove lost tracks sooner to avoid stale ID re-use).

### 11.2 Night/IR Cameras

**Problem**: FastReID is trained on visible-spectrum RGB images. Infrared (IR) and thermal cameras produce images in a different spectral domain:
- Near-IR (NIR, 700-1000nm): Grayscale, clothing colors are lost, skin tone changes
- Thermal (LWIR, 8-14um): False-color or grayscale, only thermal signatures visible

Feeding IR images to an RGB-trained FastReID model produces embeddings that are essentially random — discriminative power drops to near-chance level.

**Mitigation options** (not implemented in baseline, documented for future work):
1. **Grayscale augmentation during training**: Retraining with aggressive grayscale augmentation (p=0.5) helps the model learn texture/shape features that transfer to NIR
2. **Cross-modality ReID models**: Specialized models (e.g., DDAG, AGW) trained on visible-IR pairs — requires paired training data
3. **Disable ReID for IR streams**: Pass `reid_extractor=None` for IR camera streams, falling back to pure IoU. This is the safest option with no code changes.

**Recommendation**: For night/IR deployments, do NOT use FastReID. Fall back to pure ByteTrack.

### 11.3 Partial Occlusion

**Problem**: Heavily occluded persons produce crops that are majority-occluder (e.g., 70% wall, 30% person). The resulting embedding represents the occluder more than the person, potentially matching against tracks of different people who are occluded by the same object.

**Mitigation**:
1. **Minimum crop area gate** (Section 6.3): Crops below 1,024 pixels (e.g., thin slivers from heavy occlusion) are skipped entirely
2. **Confidence-based skipping**: Heavily occluded detections typically have lower confidence scores, pushing them into Stage 2 (IoU-only). Only high-confidence detections trigger embedding extraction
3. **EMA dampening**: With `eta=0.9`, a single bad (occluded) embedding update shifts the stored embedding by only 10%. Several consecutive clean frames quickly recover the true embedding

### 11.4 24/7 Memory Stability

**Problem**: CCTV systems run continuously. Memory leaks from growing per-track data structures cause eventual OOM crashes.

**Guarantee**: EMA embeddings are constant-size (1,024 bytes per track). Tracks exceeding `max_age` are removed entirely (including their embeddings). The total memory is bounded by `max_concurrent_tracks * 1,024 bytes`. With `max_age=30` at 30 FPS, a track is removed 1 second after being lost. The theoretical maximum concurrent tracks in a single camera view is typically <500 (even in very dense crowds), giving a hard upper bound of ~500 KB for embeddings.

The ORT inference session holds a fixed memory allocation after the first inference call. No per-frame allocations leak.

### 11.5 Multi-Camera Handoff

**Problem**: A person leaving Camera A's field of view and entering Camera B's should ideally maintain the same track ID. This requires cross-camera embedding matching.

**Current scope**: yowo's ByteTracker operates per-stream (each stream has its own tracker instance with independent track IDs). Cross-camera re-identification is NOT in scope for this baseline integration. However, the embedding infrastructure enables it as a future extension:
- When a track is removed in Camera A, its final embedding can be stored in a shared gallery
- When a new track is born in Camera B, its embedding can be compared against the gallery
- Gallery matching would be a separate module, not part of ByteTracker

---

## 12. Benchmark Results from Literature

### 12.1 MOT17 (Private Detection)

| Tracker | ReID Model | HOTA | IDF1 | MOTA | IDS | FPS |
|---|---|---|---|---|---|---|
| **BoT-SORT** | FastReID SBS-S50 | **65.0** | **80.2** | **80.5** | **1212** | ~15 |
| Deep OC-SORT | FastReID (custom) | 64.9 | 79.4 | 79.4 | 1257 | ~14 |
| StrongSORT | OSNet | 64.4 | 79.6 | 79.6 | 1194 | ~10 |
| ByteTrack | None (IoU only) | 63.1 | 77.3 | 80.3 | 2196 | ~30 |
| OC-SORT | None (IoU only) | 63.2 | 77.5 | 78.0 | 1950 | ~28 |
| FairMOT | Joint embed | 59.3 | 72.3 | 73.7 | 3303 | ~25 |

**Key observations**:
- BoT-SORT reduces IDS by 44.8% vs. ByteTrack (2196 -> 1212) while improving HOTA by 3.0%
- MOTA remains similar (80.5 vs 80.3) because MOTA primarily measures detection quality, not association
- IDF1 improves from 77.3 to 80.2 (+3.7%), directly reflecting better identity preservation
- FPS drops from ~30 to ~15 due to FastReID overhead (measured on server GPU with full-resolution crops)

### 12.2 MOT20 (Private Detection)

| Tracker | ReID Model | HOTA | IDF1 | MOTA | IDS |
|---|---|---|---|---|---|
| **Deep OC-SORT** | FastReID | **63.9** | **79.2** | **75.7** | **938** |
| BoT-SORT | FastReID SBS-S50 | 63.3 | 77.8 | 77.8 | 1257 |
| StrongSORT | OSNet | 62.6 | 77.0 | 73.8 | 1066 |
| ByteTrack | None | 61.3 | 75.2 | 77.8 | 1223 |

MOT20 contains denser scenes (>100 pedestrians per frame) than MOT17. Deep OC-SORT's adaptive weighting slightly outperforms BoT-SORT's gated fusion in this extreme density regime, but both significantly reduce IDS compared to pure ByteTrack.

### 12.3 DanceTrack (Complex Motion)

| Tracker | ReID Model | HOTA | IDF1 | MOTA | AssA |
|---|---|---|---|---|---|
| Deep OC-SORT | FastReID | 61.3 | 61.5 | 89.4 | 45.8 |
| BoT-SORT | FastReID SBS-S50 | 53.8 | 56.1 | 89.4 | 37.7 |
| ByteTrack | None | 47.3 | 52.5 | 89.5 | 31.4 |
| OC-SORT | None | 55.1 | 54.6 | 89.4 | 38.3 |

**Key observation**: DanceTrack contains non-linear, fast motion (dancers) where Kalman filter predictions fail. Both ReID-enhanced trackers significantly outperform IoU-only trackers here (HOTA +6.5 to +14.0). This validates the appearance rescue (Stage 3) for scenarios where spatial prediction is unreliable.

Note: DanceTrack results for BoT-SORT are lower than Deep OC-SORT because DanceTrack requires stronger appearance modeling — the adaptive weighting in Deep OC-SORT handles the non-rigid motion patterns better. For typical CCTV (linear walking motion), this gap closes.

### 12.4 Person ReID Benchmarks (FastReID SBS-S50)

| Dataset | mAP | Rank-1 | Rank-5 | Notes |
|---|---|---|---|---|
| Market-1501 | 91.3% | 96.3% | 98.7% | 6 cameras, outdoor |
| DukeMTMC-reID | 83.1% | 92.1% | 96.4% | 8 cameras, campus |
| MSMT17 | 64.5% | 84.2% | 92.1% | 15 cameras, complex |

MSMT17's lower accuracy reflects its larger scale and harder cross-camera scenarios, which is more representative of real-world CCTV deployments.

---

## 13. References

### Papers

1. **ByteTrack**: Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV 2022*.
   - https://arxiv.org/abs/2110.06864

2. **BoT-SORT**: Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. *arXiv preprint*.
   - https://arxiv.org/abs/2206.14651

3. **Deep OC-SORT**: Maggiolino, G., Ahmad, A., Cao, J., & Kitani, K. (2023). Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification. *ICASSP 2023*.
   - https://arxiv.org/abs/2302.11813

4. **FastReID**: He, L., Liao, X., Liu, W., Liu, X., Cheng, P., & Mei, T. (2020). FastReID: A Pytorch Toolbox for General Instance Re-identification. *arXiv preprint*.
   - https://arxiv.org/abs/2006.02631

5. **Bag of Tricks (BN Neck)**: Luo, H., Gu, Y., Liao, X., Lai, S., & Jiang, W. (2019). Bag of Tricks and A Strong Baseline for Deep Person Re-identification. *CVPR Workshops 2019*.
   - https://arxiv.org/abs/1903.07071

6. **Non-Local Neural Networks**: Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local Neural Networks. *CVPR 2018*.
   - https://arxiv.org/abs/1711.07971

7. **GeM Pooling**: Radenovic, F., Tolias, G., & Chum, O. (2019). Fine-tuning CNN Image Retrieval with No Human Annotation. *IEEE TPAMI*.
   - https://arxiv.org/abs/1711.02512

8. **StrongSORT**: Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., & Meng, H. (2023). StrongSORT: Make DeepSORT Great Again. *IEEE TMM*.
   - https://arxiv.org/abs/2202.13514

9. **DeepSORT**: Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP 2017*.
   - https://arxiv.org/abs/1703.07402

10. **CLIP-ReID**: Li, S., Sun, L., & Li, Q. (2023). CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels. *AAAI 2023*.
    - https://arxiv.org/abs/2211.13977

11. **SMILEtrack**: Wang, Y., et al. (2024). SMILEtrack: SiMIlarity LEarning for Occlusion-Aware Multi-Object Tracking. *AAAI 2024*.
    - https://arxiv.org/abs/2211.08824

### GitHub Repositories

12. **FastReID**: JDAI-CV/fast-reid — Official FastReID implementation
    - https://github.com/JDAI-CV/fast-reid

13. **BoT-SORT**: NirAharon/BoT-SORT — Official BoT-SORT implementation
    - https://github.com/NirAharon/BoT-SORT

14. **Deep OC-SORT**: GerardMaggworthy/Deep-OC-SORT — Official Deep OC-SORT implementation
    - https://github.com/GerardMaggworthy/Deep-OC-SORT

15. **ByteTrack**: ifzhang/ByteTrack — Official ByteTrack implementation
    - https://github.com/ifzhang/ByteTrack

16. **ONNX Runtime**: microsoft/onnxruntime — ONNX Runtime inference engine
    - https://github.com/microsoft/onnxruntime

### Benchmarks and Leaderboards

17. **MOTChallenge**: Multi-Object Tracking benchmark
    - https://motchallenge.net/

18. **DanceTrack**: DanceTrack multi-object tracking benchmark
    - https://dancetrack.github.io/

### Pre-trained Models

19. **FastReID Model Zoo**: Pre-trained SBS models
    - https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md

20. **BoT-SORT FastReID Weights**: Market-1501 SBS-S50 checkpoint used by BoT-SORT
    - https://github.com/NirAharon/BoT-SORT#model-zoo

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **HOTA** | Higher Order Tracking Accuracy — geometric mean of detection accuracy (DetA) and association accuracy (AssA). Balances both detection and tracking quality. Range [0, 100]. |
| **IDF1** | ID F1 Score — harmonic mean of ID precision and ID recall. Measures how long correct identity assignments are maintained. Range [0, 100]. |
| **MOTA** | Multi-Object Tracking Accuracy — 1 - (FP + FN + IDS) / GT. Dominated by detection quality (FP, FN). Range (-inf, 100]. |
| **IDS** | Identity Switches — number of times a tracked object changes its assigned ID. Lower is better. |
| **AssA** | Association Accuracy — component of HOTA measuring long-term tracking consistency. |
| **mAP** | Mean Average Precision — averaged over identities, measures ReID retrieval quality. |
| **Rank-1** | Probability that the top-1 retrieval result is the correct identity. |
| **ReID** | Re-Identification — matching the same person/object across different camera views or time gaps. |
| **EMA** | Exponential Moving Average — temporal smoothing filter: `x_t = η·x_{t-1} + (1-η)·z_t`. |
| **BN Neck** | Batch Normalization Neck — domain-adaptive feature normalization layer before classification. |
| **GeM** | Generalized Mean Pooling — learnable spatial pooling that emphasizes discriminative regions. |
| **SBS** | Short Baseline Strong — FastReID's high-accuracy model configuration. |
| **d_cos** | Cosine distance: `1 - dot(a, b)` for unit-norm vectors. Range [0, 2]. |
| **d_IoU** | IoU distance: `1 - IoU(box_a, box_b)`. Range [0, 1]. |

---

## Appendix B: Configuration Quick Reference

```python
# Recommended defaults — BoT-SORT configuration
ByteTracker(
    # Existing ByteTrack params (unchanged)
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    match_thresh=0.8,
    max_age=30,
    min_hits=3,
    # New ReID params
    reid_extractor=FastReIDExtractor("fastreid_sbs_s50.onnx"),
    embedding_momentum=0.9,
    appearance_thresh=0.25,     # theta_e: cosine distance gate
    appearance_iou_gate=0.5,    # theta_IoU: IoU distance gate for appearance
)

# Aggressive appearance matching (for DanceTrack-like scenarios)
ByteTracker(
    reid_extractor=FastReIDExtractor("fastreid_sbs_s50.onnx"),
    embedding_momentum=0.8,     # Lower: adapt faster to appearance changes
    appearance_thresh=0.35,     # Higher: accept weaker appearance matches
    appearance_iou_gate=0.7,    # Higher: allow appearance even when IoU is high
)

# Conservative (CCTV with uniforms, minimize false associations)
ByteTracker(
    reid_extractor=FastReIDExtractor("fastreid_sbs_s50.onnx"),
    embedding_momentum=0.95,    # Higher: very stable embedding, slow to adapt
    appearance_thresh=0.15,     # Lower: only accept very strong appearance matches
    appearance_iou_gate=0.3,    # Lower: only activate when IoU is very ambiguous
)

# Pure IoU mode (backward compatible, no ReID)
ByteTracker()  # Identical to v2.1.0
```
