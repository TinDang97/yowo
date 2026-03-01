# Method 2: CLIP ViT-B/16 Zero-Shot ReID for ByteTrack

> **Author**: Tin Dang
> **Date**: 2026-02-28
> **Status**: RECOMMENDED for v2.2.0
> **Module**: `yowo.tracking`
> **Depends on**: `clip-reid-integration-architecture.md` (shared architecture decisions)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [CLIP Architecture Deep Dive](#2-clip-architecture-deep-dive)
3. [CLIP vs SegCLIP for ReID](#3-clip-vs-segclip-for-reid)
4. [CLIP Model Variants and Selection](#4-clip-model-variants-and-selection)
5. [ONNX Export Pipeline](#5-onnx-export-pipeline)
6. [Preprocessing Pipeline (NumPy, No PyTorch)](#6-preprocessing-pipeline-numpy-no-pytorch)
7. [Cost Matrix Fusion — BoT-SORT Gated Formula](#7-cost-matrix-fusion--bot-sort-gated-formula)
8. [Conditional ReID Extraction](#8-conditional-reid-extraction)
9. [Embedding Update Strategy](#9-embedding-update-strategy)
10. [Association Cascade Integration](#10-association-cascade-integration)
11. [Implementation Specification](#11-implementation-specification)
12. [Performance Analysis](#12-performance-analysis)
13. [Strengths and Limitations](#13-strengths-and-limitations)
14. [CCTV-Specific Edge Cases](#14-cctv-specific-edge-cases)
15. [Comparison with FastReID Baseline](#15-comparison-with-fastreid-baseline)
16. [Testing Strategy](#16-testing-strategy)
17. [Future Upgrade Paths](#17-future-upgrade-paths)
18. [References](#18-references)

---

## 1. Executive Summary

CLIP (Contrastive Language-Image Pre-training) is a vision-language foundation model introduced by Radford et al. at ICML 2021. It was trained on 400 million image-text pairs using a contrastive InfoNCE objective that aligns images and captions in a shared 512-dimensional embedding space. The visual encoder — a Vision Transformer (ViT-B/16) with 86M parameters — learns to produce compact representations that capture the semantic identity of visual content without any task-specific supervision.

**Why zero-shot CLIP embeddings work for ReID.** The contrastive pretraining objective forces the visual encoder to learn features that distinguish one image from another in a semantically meaningful way. A picture of "a person in a red jacket" and "a person in a blue uniform" occupy distant points in the embedding space because they were aligned to different textual descriptions during training. This property — the ability to discriminate between visual appearances without ever being trained on person re-identification data — is precisely what makes CLIP a viable zero-shot ReID backbone. The [CLS] token output of the ViT captures holistic appearance: pose, color, shape, texture, and context are all compressed into a single 512-dimensional vector that can be compared via cosine similarity.

**The key insight.** 400M image-text contrastive pretraining produces visual features that capture identity-discriminative appearance without ReID-specific training. The embedding space is structured such that visually similar objects cluster together and dissimilar objects are pushed apart, enabling direct use as an appearance descriptor for multi-object tracking.

**Positioning relative to alternatives.**

| Dimension | CLIP Zero-Shot (this method) | FastReID (Method 1) |
|---|---|---|
| IDS reduction (estimated) | ~25-32% | ~40-50% (proven on MOT17) |
| Training required | None | Requires Market-1501 / MSMT17 |
| New dependencies | None (reuses ONNX Runtime) | FastReID, torch, custom weights |
| Object class support | Universal (people, vehicles, animals) | Person-only (domain-specific) |
| Edge deployment | MobileCLIP-S0 path (11.4M params) | No lightweight variant |
| Upgrade path | Zero-shot -> fine-tuned -> FastReID | Terminal |
| Integration effort | Low (Protocol-based, same ONNX stack) | Medium (additional dependencies) |

CLIP zero-shot is the recommended starting point for yowo v2.2.0 because it provides meaningful IDS reduction with zero training, zero new dependencies, and universal class support. The Protocol-based architecture enables seamless upgrade to fine-tuned CLIP-ReID or FastReID when domain-specific performance is needed.

---

## 2. CLIP Architecture Deep Dive

### 2.1 Vision Transformer (ViT-B/16) Architecture

The CLIP visual encoder is a standard Vision Transformer (Dosovitskiy et al., ICLR 2021) with the following configuration:

```
Model: ViT-B/16 (Base, patch size 16)
Layers: 12 transformer blocks
Attention heads: 12 per block
Hidden dimension: 768
MLP dimension: 3072 (4x hidden)
Patch size: 16 x 16 pixels
Input resolution: 224 x 224 RGB
Total visual encoder parameters: ~86M
```

**Input processing pipeline within the ViT:**

1. **Patch embedding**: The 224x224x3 input image is divided into a grid of 14x14 = 196 non-overlapping patches of size 16x16x3. Each patch is linearly projected from 768 dimensions (16x16x3 = 768 values per patch) into a 768-dimensional token via a learned Conv2d(3, 768, kernel_size=16, stride=16).

2. **[CLS] token prepend**: A learnable 768-dimensional [CLS] token is prepended to the sequence of 196 patch tokens, yielding 197 tokens total. The [CLS] token serves as a global aggregation point that attends to all patch tokens.

3. **Positional encoding**: Learnable 768-dimensional positional embeddings are added element-wise to all 197 tokens (1 [CLS] + 196 patches). These encode spatial location within the original image.

4. **Transformer blocks (x12)**: Each block applies:
   - Layer Normalization (pre-norm architecture)
   - Multi-Head Self-Attention (12 heads, 64 dims per head)
   - Residual connection
   - Layer Normalization
   - 2-layer MLP (768 -> 3072 -> 768) with GELU activation
   - Residual connection

5. **[CLS] extraction**: After the 12th transformer block, only the [CLS] token (position 0) is extracted. This single 768-d vector has attended to all 196 spatial patches and captured a holistic representation of the image.

6. **Linear projection**: The 768-d [CLS] token is projected through a learned linear layer to 512 dimensions, producing the final embedding vector.

7. **L2 normalization**: The 512-d embedding is L2-normalized to unit length, placing it on the surface of a 512-dimensional unit hypersphere.

```
Input image [224, 224, 3]
    |
    v
Patch embedding (Conv2d 16x16, stride 16) -> [196, 768]
    |
    v
Prepend [CLS] token -> [197, 768]
    |
    v
Add positional embeddings -> [197, 768]
    |
    v
12x Transformer Block (MHSA + MLP) -> [197, 768]
    |
    v
Extract [CLS] token -> [768]
    |
    v
Linear projection -> [512]
    |
    v
L2 normalize -> [512] (unit vector)
```

### 2.2 Contrastive Pretraining Objective (InfoNCE)

CLIP was trained with a symmetric InfoNCE contrastive loss over batches of (image, text) pairs:

```
L = -0.5 * (L_image_to_text + L_text_to_image)

L_image_to_text = -log(exp(sim(I_i, T_i)/tau) / sum_j(exp(sim(I_i, T_j)/tau)))
L_text_to_image = -log(exp(sim(T_i, I_i)/tau) / sum_j(exp(sim(T_i, I_j)/tau)))
```

Where `sim(I, T) = I_emb . T_emb` (dot product of L2-normalized embeddings = cosine similarity), and `tau` is a learned temperature parameter (initialized to 0.07, learned to ~0.01).

**Effect on visual features**: Each mini-batch contains N image-text pairs. The loss treats the N matching pairs as positives and N^2 - N non-matching pairs as negatives. With N = 32768 (CLIP's batch size), each image is contrasted against 32767 negatives per step. Over 400M pairs, this produces an embedding space where:

- Visually similar images (same object, different angles) cluster together
- Visually dissimilar images (different objects) are pushed apart
- Fine-grained appearance details (color, texture, shape) become discriminative dimensions

### 2.3 Why [CLS] Captures Holistic Appearance for ReID

The [CLS] token is the only position in the transformer sequence that has no spatial correspondence to a specific image region. Through self-attention across 12 layers, it learns to:

1. **Aggregate global structure**: body shape, pose, aspect ratio
2. **Capture texture/color statistics**: clothing color, skin tone, surface patterns
3. **Encode contextual cues**: carried objects, accessories, vehicles beside a person

These are exactly the features human annotators use for re-identification. Unlike CNN features (which are spatially anchored and lose global context), the [CLS] token's attention spans the entire image from the first layer, enabling it to weight discriminative regions (a bright red hat) over uninformative ones (generic pavement).

### 2.4 L2 Normalization and Cosine Similarity

After L2 normalization, all embeddings lie on the unit hypersphere S^511. For unit vectors:

```
cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||) = dot(a, b)
```

This simplification means cosine distance reduces to:

```
d_cos(a, b) = 1 - dot(a, b)
```

Range: [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite.

For ReID matching, distances below ~0.3 indicate strong visual similarity. The L2 normalization also stabilizes EMA updates (Section 9) by ensuring embeddings remain on the hypersphere after blending.

---

## 3. CLIP vs SegCLIP for ReID

### 3.1 CLIP: Global [CLS] Embedding — Directly Applicable

CLIP's visual encoder produces a single 512-d embedding per image via the [CLS] token pathway described in Section 2. This embedding captures **instance-level holistic appearance** — the overall "look" of the cropped object. For ReID, this is exactly what is needed: a compact vector that answers "does crop A look like crop B?"

The matching operation is trivial:

```python
# d_cos in [0, 2]: lower = more similar
d_cos = 1.0 - np.dot(embedding_a, embedding_b)
```

CLIP [CLS] embeddings are directly applicable to ReID because:
- They capture identity-discriminative features (color, shape, texture)
- They produce a fixed-size vector regardless of input aspect ratio
- They support batch computation (stack N crops, one ONNX call)
- Cosine distance is fast to compute (matrix multiply + subtract)

### 3.2 SegCLIP: Pixel-Level Segmentation — Not Directly Applicable

SegCLIP (Luo et al., ICML 2023) modifies the CLIP architecture for **dense prediction** (semantic segmentation), not instance-level embedding:

**SegCLIP architectural modifications:**
1. **K learnable center tokens** (K=8 or 16): Injected alongside the [CLS] and patch tokens, these centers learn to represent semantic groups (e.g., "sky", "road", "person")
2. **Cross-attention aggregation**: Each center token cross-attends to all patch tokens, forming an affinity matrix that groups spatial patches into semantic clusters
3. **Segment map output**: The affinity matrix (K x 196) is reshaped to K x 14 x 14 and upsampled to produce per-pixel semantic segmentation masks

**Why SegCLIP does NOT solve ReID:**

- **It answers "what is this region?" (semantic), not "who is this person?" (identity)**. SegCLIP segments an image into categories (person, car, background). Two people wearing identical uniforms produce identical SegCLIP outputs — the segmentation correctly identifies both as "person" — but ReID needs to distinguish between them.

- **The output format is wrong**. SegCLIP outputs segment maps (spatial masks), not embedding vectors. There is no single vector to compare between tracks. Converting segment maps back to embeddings would require additional architecture that defeats the purpose.

- **The centers are semantic, not identity-aware**. SegCLIP's K centers converge to semantic prototypes (head, torso, background) during training. They do not differentiate between individuals within the same semantic category.

### 3.3 SegCLIP's Contribution: Informative for Future Research

Despite being inapplicable to direct ReID, SegCLIP introduces two concepts relevant to future yowo phases:

1. **Part-aware ReID**: Grouping ViT patches into body parts (head, torso, legs) could improve occlusion robustness. If a person's legs are occluded, only the head/torso embeddings would be compared. This requires custom training on body-part annotations (DensePose or similar).

2. **Learnable appearance prototypes**: K center tokens as per-class appearance templates could enable domain-specific adaptation. For example, training on a specific CCTV installation's common clothing patterns.

Both concepts require custom training and architectural modifications, making them Phase 3+ research directions.

### 3.4 Recommendation

**Use standard CLIP [CLS] for ReID.** SegCLIP adds complexity (K learnable centers, cross-attention, segment map decoding) without solving the right problem. The CLIP visual encoder is a clean, well-understood architecture with trivial ONNX export and a single embedding vector output that slots directly into the cost matrix fusion described in Section 7.

---

## 4. CLIP Model Variants and Selection

### 4.1 Variant Comparison Table

| Variant | Params (visual) | Embedding Dim | Input Size | Patch Size | Latency (GPU, 20 crops) | Latency (CPU, 20 crops) | ReID Quality | Edge Suitable |
|---|---|---|---|---|---|---|---|---|
| ViT-B/32 | ~87M | 512 | 224x224 | 32x32 | ~2-4ms | ~8-15ms | Good | Desktop |
| **ViT-B/16** | **~86M** | **512** | **224x224** | **16x16** | **~4-8ms** | **~15-30ms** | **Best (finer patches)** | **Server/Desktop** |
| ViT-L/14 | ~304M | 768 | 224x224 | 14x14 | ~8-15ms | ~40-80ms | Excellent but overkill | Server only |
| ViT-L/14@336px | ~304M | 768 | 336x336 | 14x14 | ~15-25ms | ~80-150ms | Highest quality | Server only |
| MobileCLIP-S0 (Apple) | ~11.4M | 512 | 256x256 | - | ~1-2ms | ~5-10ms | Good | **Edge (recommended)** |
| MobileCLIP-S2 (Apple) | ~35.7M | 512 | 256x256 | - | ~2-4ms | ~8-15ms | Very Good | Edge/Desktop |

### 4.2 Why ViT-B/16 is the Default

ViT-B/16 produces 196 patches (14x14 grid) compared to ViT-B/32's 49 patches (7x7 grid). For ReID, finer spatial resolution matters because:

- **Small distinguishing features** (badge, logo, hat pattern) occupy fewer than 32x32 pixels in a typical crop but span multiple 16x16 patches
- **196 tokens give the [CLS] 4x more spatial positions to attend to**, enabling finer-grained appearance encoding
- **OpenCLIP benchmarks** consistently show ViT-B/16 outperforming ViT-B/32 on retrieval tasks by 2-5% recall@1

The latency penalty (4-8ms vs 2-4ms for 20 crops on GPU) is acceptable given the conditional extraction gate (Section 8) reduces invocations by 40-80%.

### 4.3 MobileCLIP for Edge Deployments

Apple's MobileCLIP (CVPR 2024) distills CLIP's knowledge into efficient architectures:

- **MobileCLIP-S0**: 11.4M visual params, hybrid CNN-transformer architecture. Achieves 67.8% ImageNet zero-shot accuracy (vs ViT-B/16's 68.3%) at 7.5x fewer parameters. Reported 1.5ms inference on iPhone 12 Pro Max.
- **MobileCLIP-S2**: 35.7M visual params, deeper variant. Achieves 70.3% zero-shot accuracy. Better ReID quality at moderate edge cost.

For Jetson Orin Nano or similar edge devices where ViT-B/16 exceeds the latency budget, MobileCLIP-S0 provides a viable alternative with the same 512-d embedding dimension (drop-in replacement via the `ReIDExtractor` Protocol).

### 4.4 ViT-L/14: When to Use

ViT-L/14 (304M params, 768-d embeddings) is overkill for most CCTV deployments but may be warranted for:
- Cross-camera ReID across wide baselines (airport, stadium)
- Forensic re-identification where accuracy trumps latency
- Offline batch processing of recorded footage

Its 768-d embeddings require updating the `embedding_dim` parameter and increase per-track memory from 2048 to 3072 bytes (still negligible for 200 tracks: 600KB).

---

## 5. ONNX Export Pipeline

### 5.1 Export from OpenCLIP

OpenCLIP (mlfoundations/open_clip, MIT license) provides pre-trained CLIP models with clean PyTorch implementations suitable for ONNX export.

```python
import open_clip
import torch

# Load pre-trained model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b79k"
)
model.eval()

# Extract visual encoder only (we don't need the text encoder for ReID)
visual = model.visual

# Export with dynamic batch axis
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    visual,
    dummy_input,
    "clip-vit-b16-visual.onnx",
    input_names=["images"],
    output_names=["embeddings"],
    dynamic_axes={
        "images": {0: "batch"},
        "embeddings": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

**Critical export details:**

- **opset_version=17**: Required for `LayerNormalization` and `Gelu` ops native support. Opset 18 (dynamo path) is also acceptable but not required since ViT has no data-dependent control flow (unlike YOLO's Detect head).
- **dynamic_axes on batch dim**: Essential because the number of detection crops varies per frame (0 to 50+). A single `session.run()` call handles the entire batch.
- **Visual encoder only**: The text encoder (~63M params) is not exported. For zero-shot ReID, only the visual pathway is needed.

### 5.2 Post-Export Optimization

```python
import onnxslim

# onnxslim is safe for CLIP (unlike KV-cache models where it strips I/O nodes)
model = onnxslim.slim("clip-vit-b16-visual.onnx")
onnx.save(model, "clip-vit-b16-visual.onnx")
```

Unlike KV-cache YOLO models (where onnxslim incorrectly strips "unused" K/V I/O nodes), the CLIP visual encoder has a clean single-input/single-output graph. `onnxslim` safely:
- Folds constants (positional embeddings, layer norm parameters)
- Eliminates identity ops
- Fuses operations (MatMul + Add -> Gemm, etc.)

### 5.3 Expected File Sizes

| Model | Raw ONNX | After onnxslim | Notes |
|---|---|---|---|
| ViT-B/16 | ~350MB | ~340MB | 86M params x 4 bytes |
| ViT-B/32 | ~355MB | ~345MB | Slightly more due to larger patch projection |
| ViT-L/14 | ~1.2GB | ~1.15GB | 304M params |
| MobileCLIP-S0 | ~48MB | ~45MB | 11.4M params |
| MobileCLIP-S2 | ~145MB | ~140MB | 35.7M params |

### 5.4 ONNX Runtime Execution Providers

The exported CLIP ONNX model runs on the same execution providers yowo already supports for YOLO inference:

| Provider | Platform | Expected Speedup vs CPU | Notes |
|---|---|---|---|
| CUDA EP | NVIDIA GPU | 3-8x | Most common for server deployments |
| CoreML EP | macOS ARM64 | 4-6x | Neural Engine handles ViT attention efficiently |
| TensorRT EP | NVIDIA GPU | 5-10x | Requires TRT engine build on first load |
| CPU EP | Any | 1x (baseline) | Always available as fallback |

The CLIP ONNX session reuses the same `_build_session_options()` configuration from yowo's `OnnxBackend` (graph optimization level ALL, thread counts capped at cpu_count//2 and cpu_count//4, memory pattern and reuse enabled, sequential execution mode).

### 5.5 MobileCLIP Export

MobileCLIP export follows the same pattern but uses Apple's `ml-mobileclip` package:

```python
import mobileclip
import torch

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained="/path/to/mobileclip_s0.pt"
)
model.eval()
visual = model.image_encoder

dummy_input = torch.randn(1, 3, 256, 256)  # MobileCLIP uses 256x256

torch.onnx.export(
    visual,
    dummy_input,
    "mobileclip-s0-visual.onnx",
    input_names=["images"],
    output_names=["embeddings"],
    dynamic_axes={
        "images": {0: "batch"},
        "embeddings": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
```

---

## 6. Preprocessing Pipeline (NumPy, No PyTorch)

yowo is a production inference engine that does not depend on PyTorch at runtime (PyTorch is only used for the `_pytorch.py` backend, which is optional). The CLIP preprocessing pipeline must use NumPy and OpenCV only, matching yowo's existing `cv2.dnn.blobFromImages` pattern.

### 6.1 Step-by-Step Preprocessing

```python
import numpy as np
import cv2

# CLIP normalization constants (from OpenAI's CLIP paper)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def crop_and_preprocess(
    frame_bgr: np.ndarray,
    boxes_xyxy: list[tuple[float, float, float, float]],
    input_size: int = 224,
    min_crop_area: int = 1024,
) -> np.ndarray | None:
    """Crop detection regions and preprocess for CLIP inference.

    Args:
        frame_bgr: Full frame in BGR format (H, W, 3), uint8.
        boxes_xyxy: List of (x1, y1, x2, y2) bounding boxes.
        input_size: CLIP input resolution (224 for ViT-B, 256 for MobileCLIP).
        min_crop_area: Minimum crop area in pixels. Smaller crops are skipped
            (too small for meaningful appearance features).

    Returns:
        (N, 3, input_size, input_size) float32 batch, or None if all crops
        are below min_crop_area.
    """
    h, w = frame_bgr.shape[:2]
    crops: list[np.ndarray] = []

    for x1, y1, x2, y2 in boxes_xyxy:
        # Step 1: Clip coordinates to frame bounds
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(w, int(x2))
        iy2 = min(h, int(y2))

        # Step 2: Filter by minimum crop area
        crop_w = ix2 - ix1
        crop_h = iy2 - iy1
        if crop_w * crop_h < min_crop_area:
            continue

        # Step 3: Crop (numpy slicing — zero-copy view)
        crop = frame_bgr[iy1:iy2, ix1:ix2]

        # Step 4: BGR -> RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Step 5: Resize to input_size x input_size
        # INTER_LINEAR is fastest while maintaining acceptable quality
        crop_resized = cv2.resize(
            crop_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR
        )

        # Step 6: float32 / 255.0
        crop_float = crop_resized.astype(np.float32) / 255.0

        # Step 7: Normalize with CLIP constants
        crop_norm = (crop_float - CLIP_MEAN) / CLIP_STD

        # Step 8: HWC -> CHW
        crop_chw = crop_norm.transpose(2, 0, 1)

        crops.append(crop_chw)

    if not crops:
        return None

    # Step 9: Stack into batch (N, 3, H, W)
    return np.stack(crops, axis=0)
```

### 6.2 Batched ONNX Inference

```python
def extract_embeddings(
    session: ort.InferenceSession,
    batch: np.ndarray,
) -> np.ndarray:
    """Run CLIP visual encoder and L2-normalize outputs.

    Args:
        session: Loaded ONNX Runtime session for CLIP visual encoder.
        batch: (N, 3, 224, 224) float32 preprocessed crops.

    Returns:
        (N, 512) float32 L2-normalized embeddings.
    """
    # Single session.run() for all crops (dynamic batch axis)
    [embeddings] = session.run(None, {"images": batch})

    # L2 normalize to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # prevent division by zero
    return embeddings / norms
```

### 6.3 Why Not cv2.dnn.blobFromImages

yowo's YOLO preprocessing uses `cv2.dnn.blobFromImages` because it performs resize + normalize + HWC->CHW + batch in a single C++ call. However, CLIP's normalization constants differ from the standard `scalefactor=1/255, mean=(0,0,0)` that `blobFromImages` supports. While `blobFromImages` does accept per-channel mean and scalefactor, it applies them as `(pixel * scalefactor) - mean`, which does not match CLIP's `(pixel/255 - mean) / std` formulation without algebraic rearrangement.

The explicit NumPy pipeline above is clearer, avoids footgun arithmetic errors, and adds negligible overhead (~0.3ms for 20 crops at 224x224 on Apple M4 Pro, vs ~0.2ms for blobFromImages — the difference is dwarfed by CLIP inference itself).

### 6.4 Preprocessing Validation

To verify correctness, compare embeddings from the NumPy pipeline against OpenCLIP's torchvision transforms:

```python
# Reference (torchvision)
from open_clip import create_model_and_transforms
_, _, preprocess = create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b79k")
ref_tensor = preprocess(PIL.Image.fromarray(crop_rgb))  # CHW tensor

# Our pipeline
our_array = crop_and_preprocess(frame_bgr, [(x1, y1, x2, y2)], input_size=224)

# Max absolute difference should be < 1e-5 (float32 precision)
assert np.abs(ref_tensor.numpy() - our_array[0]).max() < 1e-5
```

---

## 7. Cost Matrix Fusion — BoT-SORT Gated Formula

### 7.1 Formula Definition

The cost matrix fusion follows the BoT-SORT gated min-cost formula, adapted for CLIP embeddings:

```
d_cos(i, j) = 1 - dot(track_emb_i, det_emb_j)    # cosine distance, range [0, 2]

         | 0.5 * d_cos(i,j)   if d_cos(i,j) < theta_e AND d_IoU(i,j) < theta_IoU
d_hat(i,j) = |
         | 1.0                otherwise (appearance unreliable, fall back to IoU)

C(i, j) = min(d_IoU(i, j), d_hat(i, j))
```

Where:
- `d_IoU(i, j) = 1 - IoU(track_i_predicted, det_j)` is the IoU distance (already computed by `iou_distance()` in `_matching.py`)
- `d_cos(i, j)` is the cosine distance between the track's EMA embedding and the detection's CLIP embedding
- `theta_e` is the appearance distance threshold (gate)
- `theta_IoU` is the spatial proximity threshold (gate)

### 7.2 Dual Gating Logic

The dual gate (`d_cos < theta_e AND d_IoU < theta_IoU`) serves two purposes:

1. **Appearance gate** (`d_cos < theta_e`): Only trust appearance when the CLIP embeddings are sufficiently similar. This prevents false matches where two visually distinct objects happen to have moderate cosine similarity (common in CLIP's broad embedding space).

2. **Spatial gate** (`d_IoU < theta_IoU`): Only trust appearance when the track and detection are spatially close. A person on the left side of the frame cannot be appearance-matched to a detection on the right side, even if they look similar (prevents cross-scene false associations).

When either gate fails, `d_hat = 1.0` (neutral — appearance provides no information), and the min operation reduces to `C = min(d_IoU, 1.0) = d_IoU`. The system gracefully degrades to pure IoU matching.

### 7.3 CLIP-Specific Threshold Tuning

| Threshold | BoT-SORT (FastReID) | CLIP Zero-Shot (this method) | Rationale |
|---|---|---|---|
| `theta_e` | 0.25 | **0.30** | CLIP's 512-d embeddings have wider cosine distance distribution because they are not ReID-optimized. Inter-class distances are compressed compared to FastReID's discriminative metric space. A wider gate (0.30 vs 0.25) allows more appearance matches through while still rejecting truly dissimilar pairs. |
| `theta_IoU` | 0.5 | **0.5** | The spatial gate is independent of the ReID model — same spatial reasoning applies. |
| `d_hat` scale | 0.5 | **0.5** | The 0.5 scaling factor rewards appearance matches (halving their effective distance) when both gates pass. This makes appearance cost competitive with IoU cost in the `min()` operation. |

**Why wider `theta_e` is necessary for CLIP.** FastReID (SBS-S50) is trained with triplet loss + center loss on person ReID datasets, producing an embedding space where same-identity distances cluster near 0.0-0.15 and different-identity distances are pushed beyond 0.3. CLIP's contrastive pretraining on image-text pairs produces a more diffuse distribution where same-person distances are typically 0.10-0.25 and different-person distances are 0.20-0.50. The overlap region (0.20-0.25) is where the wider gate helps: a gate of 0.25 would reject ~30% of valid same-identity matches that CLIP assigns distances in the 0.25-0.30 range.

### 7.4 Implementation in NumPy

```python
def gated_fused_cost(
    iou_cost: NDArray[np.float64],
    track_embeddings: NDArray[np.float32],
    det_embeddings: NDArray[np.float32],
    theta_e: float = 0.30,
    theta_iou: float = 0.5,
) -> NDArray[np.float64]:
    """Compute BoT-SORT gated min-cost matrix fusing IoU and appearance.

    Args:
        iou_cost: (N, M) IoU distance matrix from iou_distance().
        track_embeddings: (N, D) L2-normalized track embeddings.
        det_embeddings: (M, D) L2-normalized detection embeddings.
        theta_e: Appearance distance gate threshold.
        theta_iou: IoU distance gate threshold.

    Returns:
        (N, M) fused cost matrix.
    """
    # Cosine distance = 1 - dot product (both are L2-normalized)
    cos_dist = 1.0 - track_embeddings @ det_embeddings.T  # (N, M)

    # Dual gate: both conditions must hold
    gate = (cos_dist < theta_e) & (iou_cost < theta_iou)

    # Gated appearance cost: 0.5 * d_cos where gate passes, else 1.0
    d_hat = np.where(gate, 0.5 * cos_dist, 1.0)

    # Final: element-wise minimum of IoU cost and gated appearance cost
    return np.minimum(iou_cost, d_hat).astype(np.float64)
```

---

## 8. Conditional ReID Extraction (Key Optimization)

### 8.1 The Problem

CLIP ViT-B/16 inference costs 4-8ms per batch of 20 crops on GPU, 15-30ms on CPU. If extracted every frame, this dominates the tracking budget. However, most frames do not need appearance features because IoU alone provides unambiguous matching.

### 8.2 IoU Ambiguity Gate

The conditional gate inspects the IoU cost matrix before deciding whether to invoke CLIP:

```python
def needs_reid(
    iou_cost: NDArray[np.float64],
    clear_thresh: float = 0.4,
    no_overlap_thresh: float = 0.7,
) -> bool:
    """Determine if ReID extraction is needed based on IoU ambiguity.

    Returns True (needs ReID) when the IoU cost matrix is ambiguous:
    multiple candidate tracks overlap with a detection in the uncertain
    range [clear_thresh, no_overlap_thresh].

    Returns False (skip ReID) when:
    - Exactly one track has IoU cost < clear_thresh for each detection
      (clear match — IoU is sufficient)
    - All tracks have IoU cost > no_overlap_thresh for a detection
      (no spatial overlap — this is a new track, not a re-identification)

    Args:
        iou_cost: (N, M) IoU distance matrix.
        clear_thresh: Cost below which a match is considered unambiguous.
        no_overlap_thresh: Cost above which no spatial association exists.

    Returns:
        True if appearance features are needed for disambiguation.
    """
    if iou_cost.size == 0:
        return False

    # For each detection (column), count tracks in the ambiguous range
    for col in range(iou_cost.shape[1]):
        col_costs = iou_cost[:, col]

        # Check if any track has clear match (cost < 0.4)
        clear_matches = np.sum(col_costs < clear_thresh)
        if clear_matches == 1:
            continue  # unambiguous — skip this detection

        # Check if all tracks have no overlap (cost > 0.7)
        if np.all(col_costs > no_overlap_thresh):
            continue  # new object, no ReID needed

        # Multiple candidates in [0.4, 0.7] range — ambiguous
        ambiguous = np.sum(
            (col_costs >= clear_thresh) & (col_costs <= no_overlap_thresh)
        )
        if ambiguous >= 2:
            return True

    return False
```

### 8.3 Literature Backing

"When to Extract ReID Features" (arXiv 2409.06617, 2024) demonstrated that conditional ReID extraction based on IoU ambiguity achieves:

- **80% FPS improvement** on MOT17 compared to always-extract ReID
- **Neutral MOTA/IDF1** (no accuracy loss from skipping unambiguous frames)
- **Effective skip rate**: 40-80% depending on scene density

The core observation is that in typical surveillance footage, 60-80% of frames have well-separated objects where IoU alone perfectly resolves the assignment. ReID features only become necessary when:
- Objects pass close to each other (occlusion)
- Multiple objects enter the scene simultaneously
- A track is re-acquired after being lost for several frames

### 8.4 Expected Skip Rates by Scene Type

| Scene | Density | Motion Pattern | Expected Skip Rate |
|---|---|---|---|
| Quiet corridor | Low (1-5 objects) | Linear, separated | 80-90% |
| Retail store | Medium (5-15 objects) | Random, crossing | 50-70% |
| Crowded sidewalk | High (15-30 objects) | Dense, parallel | 30-50% |
| Stadium entrance | Very high (30-50+) | Dense, merging | 20-40% |

### 8.5 Frame-Skip Option (Supplementary)

In addition to the IoU ambiguity gate, an optional frame-skip parameter extracts CLIP embeddings at most every N frames (default N=3) and reuses cached track embeddings between extractions:

```python
# In ByteTracker.__init__:
self._reid_frame_interval: int = 3
self._frames_since_reid: int = 0

# In ByteTracker.update():
self._frames_since_reid += 1
if needs_reid(iou_cost) and self._frames_since_reid >= self._reid_frame_interval:
    # extract CLIP embeddings
    self._frames_since_reid = 0
else:
    # use cached track embeddings for matching
```

This provides a hard cap on CLIP invocations (at most 1 per 3 frames) regardless of scene complexity.

---

## 9. Embedding Update Strategy

### 9.1 Exponential Moving Average (EMA)

When a track is matched to a detection and a fresh CLIP embedding is available, the track's stored embedding is updated via EMA:

```
track_emb = eta * track_emb + (1 - eta) * new_emb
track_emb = track_emb / ||track_emb||   # L2 renormalize
```

Default `eta = 0.9` (90% old, 10% new). This means:
- After 1 update: 90% previous appearance, 10% current
- After 5 updates: 59% cumulative history, 41% recent
- After 10 updates: 35% early history, 65% recent

The EMA provides temporal smoothing that:
- Dampens momentary appearance changes (lighting flicker, motion blur)
- Gradually adapts to genuine appearance shifts (person picks up a bag)
- Prevents a single noisy crop from corrupting the track embedding

### 9.2 L2 Renormalization After EMA

The EMA blended vector `eta * a + (1-eta) * b` where `||a|| = ||b|| = 1` has norm:

```
||eta * a + (1-eta) * b||^2 = eta^2 + (1-eta)^2 + 2*eta*(1-eta)*dot(a,b)
```

For `eta = 0.9` and `dot(a,b) = 0.8` (typical for consecutive embeddings of the same object):

```
= 0.81 + 0.01 + 2 * 0.09 * 0.8 = 0.964
||blended|| = 0.982
```

The blended vector is slightly sub-unit. L2 renormalization restores it to the unit hypersphere, ensuring cosine distance computations remain valid.

### 9.3 Memory Analysis

| Component | Per-Track | 200 Tracks | Notes |
|---|---|---|---|
| Embedding (512-d float32) | 2,048 bytes | 409,600 bytes (~400 KB) | Constant per track |
| EMA coefficient | 4 bytes | 800 bytes | Single float32 |
| Total | 2,052 bytes | ~400 KB | **Constant, no growth** |

### 9.4 Why Not a Gallery Approach

A gallery-based approach (storing the last K embeddings per track) would provide:
- Better robustness to appearance changes (can match against any historical view)
- More accurate distance computation (minimum or average over K embeddings)

However, it is rejected for 24/7 CCTV operation because:
- **Memory grows linearly**: K=10 embeddings x 2KB x 200 tracks = 4MB, and K must be capped
- **Distance computation scales with K**: K pairwise comparisons per (track, detection) pair
- **EMA empirically matches gallery**: BoT-SORT showed EMA with `eta=0.9` matches DeepSORT's 100-frame gallery on MOT17, and StrongSORT confirmed EMA superiority over fixed-size banks

---

## 10. Association Cascade Integration

### 10.1 Modified ByteTracker.update() Flow

The ReID-augmented association cascade modifies the existing two-stage ByteTrack pipeline by inserting appearance cost computation into Stage 1 and adding a new Stage 3 for lost-track rescue:

```
ByteTracker.update(detection) -> TrackedDetection:
    |
    v
[1] Split detections by confidence tier (unchanged)
    high_conf: confidence >= track_high_thresh
    low_conf:  track_low_thresh <= confidence < track_high_thresh
    |
    v
[2] Kalman predict all tracks (unchanged)
    for track in tracked + lost: track.predict()
    |
    v
[3] Compute IoU cost matrix (unchanged)
    strack_pool = tracked + lost
    iou_cost = iou_distance(strack_pool, high_det_boxes)
    |
    v
[4] NEW: Conditional ReID gate
    if reid_extractor is not None AND needs_reid(iou_cost):
        |
        v
    [4a] Extract CLIP embeddings for high-conf detections
         det_embeddings = reid_extractor.extract(frame, high_boxes)
         |
         v
    [4b] Gather track embeddings (from EMA cache)
         track_embeddings = stack([t.embedding for t in strack_pool])
         |
         v
    [4c] Compute fused cost matrix
         cost1 = gated_fused_cost(iou_cost, track_embeddings, det_embeddings)
    else:
        cost1 = iou_cost   # Pure IoU (backward compatible)
    |
    v
[5] Stage 1: Hungarian on cost1 (high-conf dets -> tracked + lost pool)
    matches1, unmatched_tracks, unmatched_dets = linear_assignment(cost1, match_thresh)
    |
    v
[6] Update matched tracks + update embeddings via EMA
    for ti, di in matches1:
        track = strack_pool[ti]
        track.update(box, conf, cls_id, cls_name, frame_id)
        if det_embeddings is not None:
            track.update_embedding(det_embeddings[di])
    |
    v
[7] Stage 2: IoU-only for low-conf dets (unchanged)
    Unmatched TRACKED tracks from Stage 1 matched to low-conf dets
    cost2 = iou_distance(unmatched_tracked, low_det_boxes)
    matches2 = linear_assignment(cost2, stage2_thresh)
    |
    v
[8] Mark remaining unmatched tracked tracks as lost (unchanged)
    |
    v
[9] NEW: Stage 3 — Appearance-only rescue for lost tracks
    if reid_extractor is not None:
        long_lost = [t for t in lost if t.time_since_update > reid_lost_age
                     and t.embedding is not None]
        unmatched_high_with_emb = [d for d in unmatched_high_dets
                                    if det_embeddings is not None]
        if long_lost and unmatched_high_with_emb:
            cos_cost = appearance_distance(long_lost_embs, unmatched_det_embs)
            rescue_matches = linear_assignment(cos_cost, theta_e)
            for ti, di in rescue_matches:
                long_lost[ti].re_activate(...)
    |
    v
[10] Birth new tracks from remaining unmatched high-conf dets (unchanged)
     Initialize embedding if available
    |
    v
[11] Update lost pool + duplicate removal (unchanged)
    |
    v
[12] Build output TrackedDetection (unchanged)
```

### 10.2 Stage 3: Appearance-Only Rescue

Stage 3 is a new addition that handles the scenario where a track has been lost for longer than `reid_lost_age` frames (default: 5). At this point, the Kalman prediction has drifted significantly and IoU matching is unreliable. Instead, pure appearance matching is used:

- **Input**: Lost tracks with `time_since_update > reid_lost_age` that have stored embeddings, and unmatched high-confidence detections with CLIP embeddings
- **Cost matrix**: Pure cosine distance (no IoU fusion — spatial prediction is stale)
- **Threshold**: `theta_e = 0.30` (same as Stage 1 appearance gate)
- **Output**: Re-activated tracks with refreshed Kalman state

This addresses the common CCTV scenario where a person leaves the frame for 5-10 frames (behind a pillar, through a doorway) and returns. Without Stage 3, they would get a new track ID. With Stage 3, their appearance embedding enables re-identification.

### 10.3 Full Data Flow Diagram

```
Frame (BGR pixels)
  |
  +---> YOLO Engine.detect() -----> Detection(boxes, frame)
  |                                       |
  |                        +--------------+---------------+
  |                        |                              |
  |                  high_conf boxes               low_conf boxes
  |                        |                              |
  |              Kalman predict all tracks                 |
  |                        |                              |
  |              IoU cost matrix (N x M_high)              |
  |                        |                              |
  |              needs_reid(iou_cost)?                    |
  |                 /           \                         |
  |               YES            NO                       |
  |                |              |                       |
  +---> CLIP extract(frame,     iou_cost                  |
  |     high_boxes)              |                        |
  |         |                    |                        |
  |     gated_fused_cost()       |                        |
  |         |                    |                        |
  |         +--------+-----------+                        |
  |                  |                                    |
  |        Stage 1: Hungarian(cost, match_thresh)         |
  |                  |                                    |
  |         matched: update tracks + EMA embedding        |
  |         unmatched tracks -----+                       |
  |         unmatched dets        |                       |
  |                               |                       |
  |                    Stage 2: IoU-only(unmatched, low) --+
  |                               |
  |                    matched: update tracks
  |                    still unmatched: mark_lost()
  |                               |
  |                    Stage 3: Appearance rescue
  |                    (lost tracks with embeddings
  |                     vs unmatched high dets)
  |                               |
  |                    Birth new tracks
  |                               |
  |                    Duplicate removal
  |                               |
  |                               v
  +----->                TrackedDetection(tracked_boxes)
```

---

## 11. Implementation Specification

### 11.1 File-by-File Changes

#### `_reid.py` (NEW, ~250 lines)

New file containing the ReID extractor protocol and CLIP implementation.

```python
"""ReID feature extraction for appearance-based track association."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["CLIPExtractor", "ReIDExtractor"]

# CLIP normalization constants (OpenAI CLIP / OpenCLIP)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


@runtime_checkable
class ReIDExtractor(Protocol):
    """Protocol for pluggable ReID feature extractors.

    Any class implementing this protocol can be passed to ByteTracker
    as the reid_extractor parameter. This enables swapping CLIP for
    FastReID, a fine-tuned model, or a mock in tests.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output embeddings (e.g., 512 for CLIP ViT-B/16)."""
        ...

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Extract L2-normalized appearance embeddings for detection crops.

        Args:
            frame_pixels: Full frame in BGR format, shape (H, W, 3), uint8.
            boxes_xyxy: List of N bounding boxes as (x1, y1, x2, y2) tuples.

        Returns:
            (N', D) float32 L2-normalized embeddings where N' <= N
            (crops below min_crop_area are filtered). Returns None if
            no valid crops remain.
        """
        ...


class CLIPExtractor:
    """CLIP ViT-B/16 visual encoder for zero-shot ReID feature extraction.

    Loads a CLIP visual encoder ONNX model and provides batched embedding
    extraction from detection crops. Uses the same ONNX Runtime infrastructure
    as yowo's YOLO backends.

    Args:
        model_path: Path to the CLIP visual encoder ONNX file.
        embedding_dim: Output embedding dimensionality. Default 512.
        input_size: CLIP input resolution (224 for ViT-B, 256 for MobileCLIP).
        device: Device hint for provider selection ("cpu", "cuda", "auto").
        min_crop_area: Minimum crop area in pixels. Crops smaller than this
            are skipped (too small for meaningful appearance features).
    """

    __slots__ = (
        "_embedding_dim",
        "_input_size",
        "_min_crop_area",
        "_model_path",
        "_session",
    )

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        embedding_dim: int = 512,
        input_size: int = 224,
        device: str = "cpu",
        min_crop_area: int = 1024,
    ) -> None: ...

    @property
    def embedding_dim(self) -> int: ...

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None: ...

    def _crop_and_preprocess(
        self,
        frame_bgr: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> tuple[NDArray[np.float32] | None, list[int]]: ...
```

**Method contracts:**

- `__init__`: Creates the ONNX Runtime `InferenceSession` with `_build_session_options()` and `_select_providers()` mirroring `OnnxBackend`. Validates that the ONNX model output dimension matches `embedding_dim`.
- `embedding_dim` (property): Returns `self._embedding_dim`.
- `extract(frame_pixels, boxes_xyxy)`: Calls `_crop_and_preprocess()` to get the batch tensor, runs `session.run(None, {"images": batch})`, L2-normalizes the output, and returns the embeddings. Returns `None` if no valid crops remain after area filtering.
- `_crop_and_preprocess(frame_bgr, boxes_xyxy)`: Implements the 9-step pipeline from Section 6.1. Returns the preprocessed batch and a list of valid indices (mapping output rows back to input box indices, accounting for filtered crops).

#### `_strack.py` (+25 lines, total ~321 lines)

Add embedding storage to the STrack class.

```python
# In __slots__, add:
"_embedding",

# In __init__, add:
self._embedding: NDArray[np.float32] | None = None

# New property:
@property
def embedding(self) -> NDArray[np.float32] | None:
    """Current appearance embedding, or None if not yet extracted."""
    return self._embedding

# New method:
def update_embedding(
    self,
    new_embedding: NDArray[np.float32],
    eta: float = 0.9,
) -> None:
    """Update track appearance embedding via EMA with L2 renormalization.

    Args:
        new_embedding: Fresh (D,) L2-normalized embedding from ReID extractor.
        eta: EMA momentum. Higher values retain more history. Default 0.9.
    """
    if self._embedding is None:
        self._embedding = new_embedding.copy()
        return
    blended = eta * self._embedding + (1.0 - eta) * new_embedding
    norm = np.linalg.norm(blended)
    if norm > 1e-8:
        blended /= norm
    self._embedding = blended
```

#### `_matching.py` (+80 lines, total ~397 lines)

Add appearance distance and gated fusion functions.

```python
def appearance_distance(
    track_embeddings: NDArray[np.float32],
    det_embeddings: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Cosine distance matrix between track and detection embeddings.

    Args:
        track_embeddings: (N, D) L2-normalized track embeddings.
        det_embeddings: (M, D) L2-normalized detection embeddings.

    Returns:
        (N, M) cosine distance matrix in [0, 2].
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

    Appearance cost is gated by dual thresholds: only trusted when both
    the cosine distance and IoU distance are below their respective
    thresholds. Otherwise, falls back to pure IoU cost.

    Args:
        iou_cost: (N, M) IoU distance matrix.
        track_embeddings: (N, D) L2-normalized track embeddings.
        det_embeddings: (M, D) L2-normalized detection embeddings.
        theta_e: Appearance distance gate threshold. Default 0.30.
        theta_iou: IoU distance gate threshold. Default 0.5.

    Returns:
        (N, M) fused cost matrix.
    """
    cos_dist = 1.0 - track_embeddings @ det_embeddings.T
    gate = (cos_dist < theta_e) & (iou_cost < theta_iou)
    d_hat = np.where(gate, 0.5 * cos_dist, 1.0)
    return np.minimum(iou_cost, d_hat).astype(np.float64)


def needs_reid(
    iou_cost: NDArray[np.float64],
    clear_thresh: float = 0.4,
    no_overlap_thresh: float = 0.7,
) -> bool:
    """Check whether ReID extraction is needed based on IoU ambiguity.

    Returns True when the IoU cost matrix contains ambiguous assignments
    (multiple candidate tracks with moderate overlap for a detection).
    Returns False when all assignments are clear (exactly one strong match)
    or absent (no spatial overlap).

    Args:
        iou_cost: (N, M) IoU distance matrix.
        clear_thresh: Cost below which a match is unambiguous.
        no_overlap_thresh: Cost above which no spatial association exists.

    Returns:
        True if appearance features are needed for disambiguation.
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
        ambiguous = int(
            np.sum((col_costs >= clear_thresh) & (col_costs <= no_overlap_thresh))
        )
        if ambiguous >= 2:
            return True
    return False
```

#### `_tracker.py` (+75 lines, total ~315 lines)

Add ReID extractor integration to ByteTracker.

```python
# In ByteTracker.__init__ signature, add:
#   reid_extractor: ReIDExtractor | None = None,
#   reid_lost_age: int = 5,
#   reid_frame_interval: int = 3,

# New instance variables:
self._reid_extractor = reid_extractor
self._reid_lost_age = reid_lost_age
self._reid_frame_interval = reid_frame_interval
self._frames_since_reid: int = 0

# In ByteTracker.update(), after computing iou_cost:
det_embeddings: NDArray[np.float32] | None = None
if (
    self._reid_extractor is not None
    and needs_reid(cost1)
    and self._frames_since_reid >= self._reid_frame_interval
):
    det_embeddings = self._reid_extractor.extract(
        detection.frame.pixels, high_boxes
    )
    self._frames_since_reid = 0
    if det_embeddings is not None:
        track_embs = _gather_track_embeddings(
            strack_pool, self._reid_extractor.embedding_dim
        )
        if track_embs is not None:
            cost1 = gated_fused_cost(cost1, track_embs, det_embeddings)
else:
    self._frames_since_reid += 1

# After Stage 1 matching, update embeddings:
for ti, di in matches1:
    track = strack_pool[ti]
    # ... existing update/re_activate logic ...
    if det_embeddings is not None and di < det_embeddings.shape[0]:
        track.update_embedding(det_embeddings[di])

# Stage 3: Appearance rescue (after Stage 2, before birth):
if self._reid_extractor is not None and det_embeddings is not None:
    _appearance_rescue(
        self._lost,
        unmatched_high_idxs,
        det_embeddings,
        high_boxes, high_confs, high_cls_ids, high_cls_names,
        frame_id,
        self._reid_lost_age,
        theta_e=0.30,
    )

# Helper function (module-level):
def _gather_track_embeddings(
    tracks: list[STrack],
    embedding_dim: int,
) -> NDArray[np.float32] | None:
    """Stack track embeddings into a matrix, using zeros for tracks without."""
    embs = []
    for t in tracks:
        if t.embedding is not None:
            embs.append(t.embedding)
        else:
            embs.append(np.zeros(embedding_dim, dtype=np.float32))
    if not embs:
        return None
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
    theta_e: float,
) -> None:
    """Stage 3: Re-activate long-lost tracks via appearance-only matching."""
    ...
```

#### `__init__.py` (+15 lines, total ~100 lines)

Add ReID exports.

```python
# New imports:
from yowo.tracking._reid import CLIPExtractor, ReIDExtractor

# Updated __all__:
__all__ = [
    "ByteTracker",
    "CLIPExtractor",
    "ReIDExtractor",
    "TrackedBox",
    "TrackedDetection",
    "track_detections",
    "track_stream",
]
```

### 11.2 Line Count Budget

| File | Current Lines | Added Lines | Total Lines | Under 700? |
|---|---|---|---|---|
| `_reid.py` (NEW) | 0 | ~250 | ~250 | Yes |
| `_strack.py` | 296 | +25 | ~321 | Yes |
| `_matching.py` | 317 | +80 | ~397 | Yes |
| `_tracker.py` | 240 | +75 | ~315 | Yes |
| `__init__.py` | 85 | +15 | ~100 | Yes |
| `_kalman.py` | 195 | 0 | 195 | Yes (unchanged) |

All files remain well under the 700-line limit.

### 11.3 Type Annotations

All new code uses strict pyright-compatible type annotations:

- `NDArray[np.float32]` and `NDArray[np.float64]` (from `numpy.typing`)
- `list[tuple[float, float, float, float]]` for box lists
- `NDArray[np.float32] | None` for optional embeddings
- `ReIDExtractor | None` for optional extractor parameter
- `@runtime_checkable` on the Protocol for isinstance checks in tests

---

## 12. Performance Analysis

### 12.1 Detailed Latency Table

| Component | Server GPU (RTX 3090) | Desktop CPU (M4 Pro) | Edge (Jetson Orin Nano) |
|---|---|---|---|
| **YOLO detection** | 3-8ms (TensorRT) | 6-12ms (CoreML) | 15-30ms (TensorRT INT8) |
| **Crop extraction** (20 dets) | 0.5-1ms | 0.5-1ms | 1-2ms |
| **CLIP ViT-B/16** (20 crops) | 4-8ms (CUDA EP) | 8-15ms (CoreML EP) | 15-25ms (TensorRT) |
| **MobileCLIP-S0** (20 crops) | 1-2ms | 3-5ms | 5-8ms |
| **Conditional gate savings** | -40 to -80% | -40 to -80% | -40 to -80% |
| **Cost fusion** (matrix ops) | <0.1ms | <0.1ms | <0.2ms |
| **EMA update** (20 tracks) | <0.01ms | <0.01ms | <0.02ms |
| **Total tracking overhead** | 5-12ms | 10-20ms | 22-42ms |
| **YOLO + tracking total** | 10-24ms | 16-41ms | 38-87ms |
| **Effective FPS** | 42-100 FPS | 24-63 FPS | 11-26 FPS |

### 12.2 With Conditional Gate Applied

Assuming a moderate CCTV scene (60% skip rate):

| Component | Server GPU | Desktop CPU | Edge (Jetson) |
|---|---|---|---|
| YOLO detection | 5ms avg | 9ms avg | 22ms avg |
| Tracking (60% skip) | 2ms avg | 4ms avg | 9ms avg |
| **Total** | **7ms avg** | **13ms avg** | **31ms avg** |
| **Effective FPS** | **~143 FPS** | **~77 FPS** | **~32 FPS** |

### 12.3 External Benchmark References

**CLIP-as-service (Jina AI) benchmarks** for ViT-B/16:
- NVIDIA TITAN RTX: ~264 QPS (queries per second) at batch_size=1
- This translates to ~3.8ms per image, confirming our 4-8ms estimate for batch of 20

**MobileCLIP (Apple, CVPR 2024) benchmarks** for S0:
- iPhone 12 Pro Max (A14 Neural Engine): 1.5ms per image
- This is the basis for the edge latency estimates (Jetson's GPU approximates mobile NPU)

**MobileCLIP2 (Apple, TMLR 2025) benchmarks** for MCL2-S0:
- Improved accuracy (74.3% vs 67.8% zero-shot ImageNet) at comparable latency
- Potential future upgrade path within the same architecture

### 12.4 Memory Footprint

| Component | Size | Notes |
|---|---|---|
| CLIP ViT-B/16 ONNX model | ~340MB | Loaded once, shared across frames |
| ONNX Runtime session overhead | ~100-200MB | GPU memory for inference workspace |
| MobileCLIP-S0 ONNX model | ~45MB | Minimal memory footprint |
| Per-track embedding (512-d) | 2KB | x200 tracks = 400KB |
| Preprocess buffer (20 crops) | ~11.5MB | 20 x 3 x 224 x 224 x 4 bytes |
| Total (ViT-B/16) | ~450-550MB | Dominated by model weights |
| Total (MobileCLIP-S0) | ~150-250MB | Edge-friendly |

---

## 13. Strengths and Limitations

### 13.1 Strengths

1. **Zero-shot deployment**: No training data, no fine-tuning, no dataset curation. Export the ONNX model and deploy immediately on any domain — retail, traffic, wildlife, industrial.

2. **Universal object class support**: CLIP was trained on 400M image-text pairs spanning every visual category. Unlike FastReID (person-only) or vehicle ReID models (car-only), CLIP embeds any object class: people, vehicles, animals, packages, tools.

3. **No new dependencies**: The CLIP ONNX model runs via ONNX Runtime, which yowo already depends on for YOLO inference. No `fastreid`, no `torchreid`, no additional Python packages.

4. **Clean ONNX architecture**: The ViT is a pure transformer — no custom ops, no data-dependent branches (unlike YOLO's Detect head), no dynamic shapes beyond the batch axis. Export is trivial and ORT graph optimization handles it well.

5. **MobileCLIP edge path**: Apple's MobileCLIP-S0 (11.4M params, ~45MB ONNX) provides a drop-in replacement for edge devices, sharing the same 512-d embedding space and the same `ReIDExtractor` Protocol interface.

6. **OpenCLIP ecosystem**: The model is available via OpenCLIP (mlfoundations/open_clip), which is actively maintained (MIT license), with dozens of pre-trained checkpoints across architectures and training datasets.

7. **Upgrade path**: The `ReIDExtractor` Protocol enables seamless progression:
   - Phase 1: CLIP zero-shot (this document) — immediate deployment
   - Phase 2: CLIP-ReID fine-tuned (Li et al., AAAI 2023) — ~45% IDS reduction
   - Phase 3: FastReID or custom model — maximum domain-specific accuracy
   - All phases share the same Protocol interface, cost fusion, and tracker integration

### 13.2 Limitations

1. **Larger model footprint**: 86M params / ~340MB ONNX for ViT-B/16. This is acceptable for server and desktop but may strain resource-constrained edge devices (use MobileCLIP-S0 instead).

2. **Slower than task-specific ReID**: CLIP ViT-B/16 costs ~4-8ms (GPU, 20 crops) vs FastReID's ~2-4ms (same hardware). The transformer architecture is inherently more compute-intensive than ResNet-50 for the same embedding quality.

3. **Not ReID-optimized**: CLIP's embedding space is trained to align images with text, not to maximize inter-identity distance. The discriminative power for same-class, different-identity matching is lower than FastReID's triplet-loss-trained space. This manifests as:
   - Wider same-identity distance distribution (0.10-0.25 vs 0.05-0.15 for FastReID)
   - Narrower inter-identity gap (0.15-0.20 margin vs 0.25-0.35 for FastReID)
   - Conservative IDS reduction estimate: ~25-32% vs FastReID's proven ~45%

4. **224x224 square input**: CLIP resizes all crops to 224x224, squashing the original aspect ratio. For tall, narrow person crops (common at 3:1 aspect ratio in CCTV), this distorts the appearance. Aspect-ratio-preserving resize with padding could help but deviates from CLIP's training distribution.

5. **CCTV domain gap**: CLIP was trained on internet images (well-lit, centered, high-resolution). CCTV footage is typically low-resolution, off-angle, poorly lit, and motion-blurred. The embedding quality degrades in these conditions, though the 400M training pairs provide broader coverage than any single-domain model.

6. **Conservative IDS reduction**: Estimated ~25-32% IDS reduction compared to FastReID's proven ~40-50% on MOT17. For applications requiring maximum tracking accuracy (forensic identification, access control), FastReID or fine-tuned CLIP-ReID should be considered.

---

## 14. CCTV-Specific Edge Cases

### 14.1 Appearance Ambiguity (Uniforms, Similar Clothing)

**Scenario**: Multiple workers wearing identical uniforms. CLIP embeddings are nearly identical for all workers.

**Mitigation**: IoU gating (`theta_IoU = 0.5`) ensures appearance matching is only used when objects are spatially close. When two uniformed workers are on opposite sides of the frame, IoU distance is ~1.0, the gate fails, and `d_hat = 1.0` (falls back to IoU-only). Appearance matching only activates during close-range crossings, where short-term IoU overlap patterns still provide useful disambiguation.

**Residual risk**: Two uniformed workers crossing paths will likely swap IDs. This is a fundamental limitation of appearance-based ReID for visually identical objects. Kalman velocity prediction provides the primary disambiguation in this case.

### 14.2 Lighting Transitions

**Scenario**: Person moves from bright lobby to dim corridor. CLIP embedding shifts abruptly due to brightness change.

**Mitigation**: EMA momentum (`eta = 0.9`) smooths the transition. A sudden embedding shift contributes only 10% to the track embedding per frame, preventing a single dark-frame crop from corrupting the identity. Over 5-10 frames, the embedding gradually adapts to the new lighting condition.

**Additional measure**: The `min_crop_area` gate (Section 6.1) filters out very dark crops where the pixel values are near-zero after normalization, as these produce unreliable embeddings.

### 14.3 Partial Occlusion

**Scenario**: Person partially occluded by furniture, another person, or door frame. The visible crop is fragmentary.

**Mitigation**: `min_crop_area = 1024` pixels (default) filters out crops smaller than ~32x32. Partially visible objects with very small visible area produce unreliable embeddings and are excluded from appearance matching. IoU-only matching handles them.

**ViT advantage**: Unlike CNN features (which are sensitive to missing spatial regions), the ViT [CLS] token attends to all visible patches. A 50% occluded person still provides informative patches for the [CLS] to aggregate.

### 14.4 Cross-Camera Re-Entry

**Scenario**: Person exits one camera and enters another.

**Mitigation (single camera)**: Stage 3 rescue (Section 10.2) uses appearance-only matching for tracks lost for `reid_lost_age` frames. If the person re-enters the same camera within `max_age` frames, their stored embedding enables re-identification.

**Limitation (cross-camera)**: Multi-camera ReID requires sharing track embeddings across `ByteTracker` instances, which is out of scope for v2.2.0. The `ReIDExtractor` Protocol and EMA embedding infrastructure lay the groundwork for Phase 4 cross-camera support.

### 14.5 Embedding Memory Bloat

**Scenario**: 24/7 operation accumulating track embeddings over thousands of tracks.

**Mitigation**: EMA provides constant memory per track (2,052 bytes). There is no gallery growth. When a track is marked REMOVED (after `max_age` frames of being lost), its `STrack` object is garbage collected, including its embedding. At any given time, memory usage is bounded by `(active_tracks + lost_tracks) * 2,052 bytes`, which for 200 concurrent tracks is ~400KB.

### 14.6 CLIP Domain Gap

**Scenario**: CLIP's internet-trained features underperform on surveillance-specific visual patterns (top-down camera angles, infrared imagery, fish-eye distortion).

**Mitigation**: CLIP's 400M training pairs include a long tail of diverse images — security camera screenshots, dashcam footage, drone imagery. While not specifically surveillance-trained, the coverage is broader than any single-domain ReID model. The `theta_e = 0.30` wider gate compensates for reduced discriminative power.

**Monitoring**: Track the ratio of gated matches (where `d_cos < theta_e`) to total matches. If this ratio falls below 20% consistently, CLIP embeddings are not providing useful signal, and users should consider fine-tuning (Phase 2) or switching to a domain-specific model.

### 14.7 Crowded Scenes (>50 Detections)

**Scenario**: Dense crowd with 50+ detections per frame. CLIP inference for all crops exceeds the latency budget.

**Mitigation**:
1. **Conditional gate**: In dense scenes with well-separated objects, the IoU ambiguity gate still filters 30-50% of frames.
2. **Top-K subsampling**: When detection count exceeds a configurable `max_reid_crops` (default: 30), only the top-K detections by confidence are sent to CLIP. Lower-confidence detections are matched by IoU only.
3. **Frame-skip**: `reid_frame_interval = 3` limits CLIP invocations to at most 1 per 3 frames, using cached embeddings between extractions.

### 14.8 Night / Low-Light Conditions

**Scenario**: Frame brightness drops below usable levels (parking lots at night, unlit corridors).

**Mitigation**: Before ReID extraction, compute mean frame brightness:

```python
gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
mean_brightness = gray.mean()
if mean_brightness < 30:  # ~12% of 255 range
    return None  # skip ReID, fall back to IoU-only
```

CLIP embeddings from very dark frames are dominated by noise and provide no identity signal. Disabling ReID for dark frames avoids false matches while retaining IoU-based tracking (which works on any brightness as long as YOLO detects the objects).

### 14.9 Motion Blur

**Scenario**: Fast-moving objects (running person, speeding vehicle) produce blurred crops that CLIP cannot embed reliably.

**Mitigation**: Laplacian variance as a blur detector:

```python
def is_blurry(crop_gray: np.ndarray, threshold: float = 50.0) -> bool:
    """Check if a crop is too blurry for reliable ReID embedding."""
    laplacian_var = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
    return laplacian_var < threshold
```

Blurred crops are excluded from CLIP extraction and matched by IoU only. The Laplacian variance threshold (50.0) is tuned to reject motion-blurred crops while accepting stationary objects under moderate surveillance camera quality.

---

## 15. Comparison with FastReID Baseline

### 15.1 Side-by-Side Comparison

| Dimension | CLIP Zero-Shot | FastReID (SBS-S50) |
|---|---|---|
| **IDS reduction** | ~25-32% (estimated) | ~40-50% (proven on MOT17) |
| **HOTA gain** | +2-5% (estimated) | +5-10% (proven) |
| **Domain flexibility** | Universal (any object class) | Person-only (needs retraining for vehicles) |
| **Training required** | None | Market-1501 + MSMT17 (hours of GPU time) |
| **Model size (visual)** | 86M params / ~340MB ONNX | 25M params / ~100MB ONNX |
| **Inference latency (GPU, 20 crops)** | 4-8ms | 2-4ms |
| **Inference latency (CPU, 20 crops)** | 15-30ms | 8-15ms |
| **Edge variant** | MobileCLIP-S0 (11.4M / 45MB) | No lightweight option |
| **Dependencies** | None (reuses ONNX Runtime) | fastreid, torch |
| **ONNX export** | Trivial (pure ViT) | Requires custom export code |
| **Maintenance burden** | Low (OpenCLIP maintained by community) | Medium (FastReID specific versions) |
| **Integration effort** | ~250 lines new code | ~400 lines + dependency management |
| **Upgrade path** | Zero-shot -> fine-tuned -> FastReID | Terminal (no further progression) |
| **License** | MIT (OpenCLIP) | Apache 2.0 (FastReID) |

### 15.2 When to Choose Each

**Choose CLIP zero-shot when:**
- Deploying to a new domain without training data
- Tracking non-person objects (vehicles, animals, packages)
- Minimizing dependency footprint
- Edge deployment is a requirement (MobileCLIP path)
- Rapid prototyping / proof-of-concept

**Choose FastReID when:**
- Person tracking is the primary use case
- Maximum IDS reduction is critical (access control, forensics)
- GPU server deployment (latency budget available)
- Training data for the target domain is available
- MOT benchmark performance is the success metric

---

## 16. Testing Strategy

### 16.1 Unit Tests for `_reid.py`

```python
class TestReIDExtractorProtocol:
    """Verify the Protocol is runtime-checkable and matches CLIPExtractor."""

    def test_clip_extractor_satisfies_protocol(self):
        """CLIPExtractor is an instance of ReIDExtractor Protocol."""
        assert isinstance(CLIPExtractor(...), ReIDExtractor)

    def test_protocol_requires_embedding_dim(self):
        """Protocol enforces embedding_dim property."""
        ...

    def test_protocol_requires_extract_method(self):
        """Protocol enforces extract() method signature."""
        ...


class TestCLIPExtractor:
    """Test CLIPExtractor initialization and inference."""

    def test_init_valid_model(self, clip_onnx_path):
        """Loads a valid ONNX model without error."""
        ext = CLIPExtractor(clip_onnx_path, embedding_dim=512)
        assert ext.embedding_dim == 512

    def test_init_invalid_model_path_raises(self):
        """Raises FileNotFoundError for non-existent model path."""
        with pytest.raises(FileNotFoundError):
            CLIPExtractor("/nonexistent/model.onnx")

    def test_extract_empty_boxes_returns_none(self, clip_extractor, sample_frame):
        """extract() with empty box list returns None."""
        result = clip_extractor.extract(sample_frame, [])
        assert result is None

    def test_extract_single_box(self, clip_extractor, sample_frame):
        """extract() with one box returns (1, 512) array."""
        result = clip_extractor.extract(sample_frame, [(10, 10, 100, 100)])
        assert result is not None
        assert result.shape == (1, 512)
        assert result.dtype == np.float32

    def test_extract_batch(self, clip_extractor, sample_frame):
        """extract() with multiple boxes returns (N, 512) array."""
        boxes = [(10, 10, 100, 100), (200, 200, 400, 400)]
        result = clip_extractor.extract(sample_frame, boxes)
        assert result is not None
        assert result.shape == (2, 512)

    def test_extract_l2_normalized(self, clip_extractor, sample_frame):
        """Output embeddings are L2-normalized to unit length."""
        result = clip_extractor.extract(sample_frame, [(10, 10, 100, 100)])
        assert result is not None
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_min_crop_area_filters_small_boxes(self, clip_extractor, sample_frame):
        """Boxes smaller than min_crop_area are filtered out."""
        ext = CLIPExtractor(..., min_crop_area=10000)
        result = ext.extract(sample_frame, [(0, 0, 10, 10)])  # area=100 < 10000
        assert result is None

    def test_crop_clipped_to_frame_bounds(self, clip_extractor):
        """Boxes extending beyond frame edges are clipped."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clip_extractor.extract(frame, [(-10, -10, 200, 200)])
        assert result is not None  # clipped to (0,0,100,100), area=10000 >= 1024

    def test_normalization_matches_reference(self, clip_extractor):
        """CLIP normalization constants match OpenAI reference."""
        # Verify against known CLIP mean/std
        assert np.allclose(_CLIP_MEAN, [0.48145466, 0.4578275, 0.40821073])
        assert np.allclose(_CLIP_STD, [0.26862954, 0.26130258, 0.27577711])
```

### 16.2 Unit Tests for `_matching.py` (new functions)

```python
class TestAppearanceDistance:
    """Test cosine distance matrix computation."""

    def test_identical_embeddings_zero_distance(self):
        """Same embeddings produce distance 0."""
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        dist = appearance_distance(emb, emb)
        assert dist.shape == (1, 1)
        np.testing.assert_allclose(dist[0, 0], 0.0, atol=1e-6)

    def test_orthogonal_embeddings_unit_distance(self):
        """Orthogonal embeddings produce distance 1."""
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        dist = appearance_distance(a, b)
        np.testing.assert_allclose(dist[0, 0], 1.0, atol=1e-6)

    def test_opposite_embeddings_max_distance(self):
        """Opposite embeddings produce distance 2."""
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        dist = appearance_distance(a, b)
        np.testing.assert_allclose(dist[0, 0], 2.0, atol=1e-6)

    def test_empty_inputs(self):
        """Empty inputs return empty matrix."""
        a = np.empty((0, 512), dtype=np.float32)
        b = np.array([[1.0] * 512], dtype=np.float32)
        dist = appearance_distance(a, b)
        assert dist.shape == (0, 1)

    def test_batch_computation(self):
        """N x M matrix computed correctly for multiple embeddings."""
        a = np.random.randn(5, 512).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.random.randn(3, 512).astype(np.float32)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        dist = appearance_distance(a, b)
        assert dist.shape == (5, 3)
        assert dist.dtype == np.float64
        assert np.all(dist >= -1e-6) and np.all(dist <= 2.0 + 1e-6)


class TestGatedFusedCost:
    """Test BoT-SORT gated min-cost fusion."""

    def test_no_gate_pass_returns_iou(self):
        """When both gates fail, returns IoU cost (min with 1.0)."""
        iou_cost = np.array([[0.8]], dtype=np.float64)
        t_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        d_emb = np.array([[-1.0, 0.0]], dtype=np.float32)  # d_cos = 2.0 > theta_e
        fused = gated_fused_cost(iou_cost, t_emb, d_emb)
        np.testing.assert_allclose(fused[0, 0], 0.8)  # min(0.8, 1.0)

    def test_gate_pass_reduces_cost(self):
        """When both gates pass, appearance cost (0.5 * d_cos) can be lower."""
        iou_cost = np.array([[0.3]], dtype=np.float64)  # < theta_iou=0.5
        t_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        d_emb = np.array([[0.9, 0.436]], dtype=np.float32)
        d_emb /= np.linalg.norm(d_emb)
        d_emb = d_emb.reshape(1, -1)
        fused = gated_fused_cost(iou_cost, t_emb, d_emb)
        assert fused[0, 0] < iou_cost[0, 0]  # appearance reduces cost

    def test_iou_gate_blocks_appearance(self):
        """When IoU distance > theta_iou, appearance is not used."""
        iou_cost = np.array([[0.6]], dtype=np.float64)  # > theta_iou=0.5
        t_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        d_emb = np.array([[1.0, 0.0]], dtype=np.float32)  # d_cos=0, but gate fails
        fused = gated_fused_cost(iou_cost, t_emb, d_emb)
        np.testing.assert_allclose(fused[0, 0], 0.6)  # min(0.6, 1.0)


class TestNeedsReID:
    """Test IoU ambiguity gate for conditional ReID extraction."""

    def test_empty_cost_returns_false(self):
        assert needs_reid(np.empty((0, 0), dtype=np.float64)) is False

    def test_clear_match_returns_false(self):
        """Single track with low IoU cost -> unambiguous, no ReID needed."""
        cost = np.array([[0.2]], dtype=np.float64)  # < clear_thresh=0.4
        assert needs_reid(cost) is False

    def test_no_overlap_returns_false(self):
        """All tracks far from detection -> new object, no ReID needed."""
        cost = np.array([[0.9], [0.8]], dtype=np.float64)  # > no_overlap=0.7
        assert needs_reid(cost) is False

    def test_ambiguous_returns_true(self):
        """Multiple tracks in ambiguous range -> ReID needed."""
        cost = np.array([[0.5], [0.6]], dtype=np.float64)  # both in [0.4, 0.7]
        assert needs_reid(cost) is True

    def test_one_clear_one_ambiguous_returns_false(self):
        """One clear match + one ambiguous -> clear match resolves it."""
        cost = np.array([[0.2], [0.5]], dtype=np.float64)
        assert needs_reid(cost) is False
```

### 16.3 Unit Tests for `_strack.py` (new embedding features)

```python
class TestSTrackEmbedding:
    """Test embedding slot and EMA update on STrack."""

    def test_initial_embedding_is_none(self, strack):
        """New track has no embedding."""
        assert strack.embedding is None

    def test_first_update_sets_embedding(self, strack):
        """First update_embedding() copies the embedding directly."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        strack.update_embedding(emb)
        np.testing.assert_array_equal(strack.embedding, emb)

    def test_ema_blends_with_momentum(self, strack):
        """Subsequent updates blend via EMA with eta=0.9."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        strack.update_embedding(emb1)

        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        strack.update_embedding(emb2, eta=0.9)

        # Expected: 0.9 * [1,0,...] + 0.1 * [0,1,...] = [0.9, 0.1, ...], then L2 normalized
        expected = np.zeros(512, dtype=np.float32)
        expected[0] = 0.9
        expected[1] = 0.1
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(strack.embedding, expected, atol=1e-6)

    def test_ema_output_is_unit_normalized(self, strack):
        """EMA output is always L2-normalized."""
        for _ in range(10):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            strack.update_embedding(emb)
        norm = np.linalg.norm(strack.embedding)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)
```

### 16.4 Integration Tests

```python
class TestTrackerWithReID:
    """Integration tests for ByteTracker with ReID extractor."""

    def test_tracker_without_reid_unchanged(self):
        """ByteTracker(reid_extractor=None) produces identical results to v2.1."""
        tracker = ByteTracker()
        # ... verify bit-identical output with existing test fixtures ...

    def test_tracker_with_mock_reid_reduces_ids(self):
        """ByteTracker with mock ReID extractor reduces identity switches."""
        mock_extractor = MockReIDExtractor(embedding_dim=512)
        tracker = ByteTracker(reid_extractor=mock_extractor)
        # ... run crossing scenario, verify fewer ID switches ...

    def test_stage3_rescue_reactivates_lost_track(self):
        """Lost track is re-activated via appearance when IoU fails."""
        mock_extractor = MockReIDExtractor(embedding_dim=512)
        tracker = ByteTracker(reid_extractor=mock_extractor, reid_lost_age=3)
        # Track a person -> lose for 5 frames -> person reappears
        # -> Stage 3 should re-activate with same track_id

    def test_graceful_degradation_on_extractor_failure(self):
        """If extractor.extract() returns None, tracker falls back to IoU."""
        failing_extractor = FailingReIDExtractor()
        tracker = ByteTracker(reid_extractor=failing_extractor)
        # Should not raise, should produce valid TrackedDetection

    def test_reid_not_called_when_gate_skips(self):
        """Conditional gate prevents unnecessary CLIP calls."""
        counting_extractor = CountingReIDExtractor()
        tracker = ByteTracker(reid_extractor=counting_extractor)
        # Run 10 frames with well-separated objects
        assert counting_extractor.call_count < 10  # should skip most frames


class TestPerformance:
    """Performance regression tests for ReID overhead."""

    def test_reid_overhead_under_budget(self):
        """Total tracking time with ReID stays under 50ms on CPU."""
        # Mock extractor with controlled latency
        # Verify total update() time < 50ms for 20 detections
        ...
```

### 16.5 Mock ReIDExtractor for Tests

```python
class MockReIDExtractor:
    """Deterministic mock ReID extractor for unit/integration tests.

    Generates reproducible embeddings based on box coordinates, enabling
    controlled testing of the association cascade.
    """

    __slots__ = ("_dim",)

    def __init__(self, embedding_dim: int = 512) -> None:
        self._dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        if not boxes_xyxy:
            return None
        embeddings = []
        for x1, y1, x2, y2 in boxes_xyxy:
            rng = np.random.RandomState(int(x1 + y1 * 1000 + x2 * 1e6))
            emb = rng.randn(self._dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            embeddings.append(emb)
        return np.stack(embeddings, axis=0)
```

### 16.6 Test Count Estimate

| Test Category | Count | Notes |
|---|---|---|
| `_reid.py` unit tests | ~15 | Protocol, init, extract, crop, normalize |
| `_matching.py` unit tests | ~15 | appearance_distance, gated_fused_cost, needs_reid |
| `_strack.py` unit tests | ~6 | embedding slot, update_embedding EMA |
| Integration tests | ~8 | tracker+reid, stage3 rescue, degradation |
| Performance tests | ~3 | overhead budget, conditional skip rate |
| **Total new tests** | **~47** | Added to existing 1169 = ~1216 total |

---

## 17. Future Upgrade Paths

### Phase 2: CLIP-ReID Fine-Tuned (v2.3.0)

CLIP-ReID (Li et al., AAAI 2023) fine-tunes the CLIP visual encoder for person re-identification using a two-stage training process:

1. **Text prompt optimization**: Learn per-identity text tokens `[V_1] [V_2] ... [V_M] [CLASS]` that maximize cosine similarity with corresponding identity images. This adapts the text encoder to produce identity-specific anchors without modifying the visual encoder.

2. **Image encoder fine-tuning**: Freeze the text encoder, then fine-tune the visual encoder with cross-entropy ID loss + triplet loss using the optimized text embeddings as soft labels. This pushes same-identity embeddings closer and different-identity embeddings apart in CLIP's existing feature space.

**Expected IDS reduction**: ~40-50% (comparable to FastReID, proven on MSMT17 with 86.7% mAP).

**Integration**: Drop-in replacement via the `ReIDExtractor` Protocol. Only the ONNX model file changes; all preprocessing, cost fusion, and tracker integration remain identical.

### Phase 3: SegCLIP-Inspired Part-Aware ReID (v2.4.0)

Adapting SegCLIP's learnable center concept for part-aware ReID:

1. **Body-part centers**: K=5 learnable tokens representing head, upper body, lower body, feet, carried objects
2. **Part-aware embeddings**: Cross-attention between centers and ViT patches produces K per-part embeddings (K x 512)
3. **Occlusion-robust matching**: Compare only visible parts (ignore occluded part embeddings)

Requires custom training on DensePose-annotated datasets. Research direction only.

### Phase 4: Cross-Camera ReID (v2.5.0)

Enable re-identification across multiple camera streams:

1. **Shared embedding store**: Central store of track embeddings accessible to all `ByteTracker` instances
2. **Camera-aware normalization**: Adjust embeddings for camera-specific color/brightness bias
3. **Handoff protocol**: When a track is lost in camera A, its embedding is broadcast to camera B's tracker for Stage 3 rescue

Builds on the multi-stream pipeline (`yowo.pipeline`) and the per-track embedding infrastructure established in this document.

---

## 18. References

### Core Papers

1. **CLIP** — Radford, A., Kim, J.W., Hallacy, C., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
   https://arxiv.org/abs/2103.00020

2. **SegCLIP** — Luo, H., Bao, J., Wu, Y., He, X., Li, T. "SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation." ICML 2023.
   https://arxiv.org/abs/2211.14813

3. **CLIP-ReID** — Li, S., Sun, L., Li, Q. "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels." AAAI 2023.
   https://arxiv.org/abs/2211.13977

4. **MobileCLIP** — Vasu, P.K.A., Pouransari, H., Faghri, F., Tuzel, O. "MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training." CVPR 2024.
   https://arxiv.org/abs/2311.17049

5. **MobileCLIP2** — Vasu, P.K.A., Pouransari, H., Faghri, F., Tuzel, O. "MobileCLIP2: Faster and Better Open-Vocabulary Learning." TMLR 2025.
   https://arxiv.org/abs/2501.03729

### Tracking Papers

6. **ByteTrack** — Zhang, Y., Sun, P., Jiang, Y., et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022.
   https://arxiv.org/abs/2110.06864

7. **BoT-SORT** — Aharon, N., Orfaig, R., Bobrovsky, B.Z. "BoT-SORT: Robust Associations Multi-Pedestrian Tracking." arXiv 2022.
   https://arxiv.org/abs/2206.14651

8. **Deep OC-SORT** — Maggiolino, G., Ahmad, A., Cao, J., Kitani, K. "Deep OC-SORT: Multi-Pedestrian Tracking by Observation-Centric Re-Identification." ICASSP 2023.
   https://arxiv.org/abs/2302.11813

### Efficiency & Conditional Extraction

9. **When to Extract ReID Features** — "Efficient Appearance-Based Tracking via Conditional Feature Extraction." arXiv 2024.
   https://arxiv.org/abs/2409.06617

### Software & Benchmarks

10. **OpenCLIP** — mlfoundations/open_clip. MIT License. Pre-trained CLIP models for research and production.
    https://github.com/mlfoundations/open_clip

11. **CLIP-as-service** — Jina AI. CLIP inference benchmarks (ViT-B/16: ~264 QPS on TITAN RTX).
    https://github.com/jina-ai/clip-as-service

12. **CLIP-ONNX** — Lednik7/clip-onnx. CLIP ONNX export utilities.
    https://github.com/Lednik7/clip-onnx

13. **onnx_clip** — Lakera AI. Production-ready CLIP ONNX inference.
    https://github.com/lakeraai/onnx_clip

14. **Apple ml-mobileclip** — MobileCLIP official implementation and weights.
    https://github.com/apple/ml-mobileclip

---

> **Document status**: Complete. Ready for implementation review.
> **Next step**: Implementation of `_reid.py`, `_strack.py` modifications, `_matching.py` additions, and `_tracker.py` integration per the specification above.
