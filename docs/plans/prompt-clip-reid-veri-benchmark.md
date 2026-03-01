# Prompt: Integrate CLIP-ReID Pretrained Weights for VeRi-776 Cross-Camera Vehicle ReID

## Expert Persona

You are a Principal Computer Vision Engineer with 15 years specializing in vehicle re-identification systems, CLIP-based transfer learning, and ONNX model deployment. You have deployed CLIP-ReID models achieving mAP >80% on VeRi-776 in production traffic surveillance systems. You excel at:
- Extracting inference-only pipelines from PyTorch research codebases
- ONNX export of ViT-based CLIP models with proper preprocessing
- Benchmarking ReID models on standard datasets (VeRi-776 protocol)
- Integrating pretrained checkpoints into existing production systems

## Stakes

This integration is the single biggest accuracy improvement for yowo's cross-camera vehicle tracking — going from mAP=9.32% (CLIP zero-shot) to mAP=83.3% (CLIP-ReID fine-tuned). Getting the preprocessing and model loading right is critical — wrong normalization or input size produces garbage embeddings. I'll tip you $200 for a working end-to-end pipeline with correct VeRi-776 metrics.

## Context — What Already Exists

### yowo's current ReID infrastructure (`src/yowo/tracking/_reid.py`)

```python
# Protocol — any extractor must implement this
@runtime_checkable
class ReIDExtractor(Protocol):
    @property
    def embedding_dim(self) -> int: ...
    def extract(self, frame_pixels: NDArray[np.uint8], boxes_xyxy: list[tuple[float, float, float, float]]) -> NDArray[np.float32] | None: ...

# Existing implementations
class CLIPExtractor:       # 512-dim, CLIP ViT-B/16 visual encoder ONNX, zero-shot
class FastReIDExtractor:   # 256-dim, SBS-S50 ONNX, person-trained
```

### VeRi benchmark experiment (`tmp/experiment_veri_reid.py`)
- Already parses VeRi-776 dataset (query/gallery/gt_index/jk_index)
- Computes Rank-1, Rank-5, Rank-10, mAP
- Generates grouped visual results (query + top-K matches with green/red borders)
- **Current results**: CLIP zero-shot mAP=9.32%, Rank-1=31.82%

### VeRi-776 dataset location
```
/Users/tindang/Downloads/VeRi/
├── image_train/    (37,778 images, 576 vehicles)
├── image_query/    (1,678 images, 200 vehicles)
├── image_test/     (11,579 images, 200 vehicles)
├── gt_index.txt    (1-based gallery indices per query line)
├── jk_index.txt    (1-based junk indices per query line)
├── name_query.txt  (ordered query filenames)
├── name_test.txt   (ordered gallery filenames)
└── test_label.xml  (vehicleID, cameraID, colorID, typeID per image)

Filename format: {vehicleID}_c{cameraID}_{timestamp}_{flag}.jpg
Images are pre-cropped vehicle patches, variable size (~100-400px).
```

### yowo Quality Gates
```bash
uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q
```
- 1269 tests pass, 0 lint errors, 0 pyright errors
- Files MUST be < 700 lines. Use `__slots__` on all classes.

## CLIP-ReID Reference Implementation

**Repository:** `github.com/Syliz517/CLIP-ReID`
**Paper:** "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels" (arXiv:2211.13977)

### Architecture
- Based on CLIP ViT-B/16 visual encoder (768-dim transformer output)
- **Stage 1**: Freeze text+image encoders, learn text prompt tokens per vehicle ID
- **Stage 2**: Freeze text encoder, fine-tune image encoder with learned prompts as supervision
- Output: 768-dim features from ViT [CLS] token, then BN + classifier head
- **Inference uses only the image encoder** — no text encoder needed

### Pretrained VeRi Weights (from README table)

| Variant | Model Download | Test Log | mAP | Rank-1 |
|---------|---------------|----------|-----|--------|
| **ViT-CLIP-ReID** | [Google Drive `1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e`](https://drive.google.com/file/d/1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e/view?usp=share_link) | mAP=83.3%, R1=97.4% | 83.3% | 97.4% |
| **ViT-CLIP-ReID-SIE-OLP** | [Google Drive `1vb-mMGp7q_aqAB1U_uAGsHZ1U9HViOgE`](https://drive.google.com/file/d/1vb-mMGp7q_aqAB1U_uAGsHZ1U9HViOgE/view?usp=share_link) | mAP=84.5%, R1=97.3% | 84.5% | 97.3% |

### Config (from `configs/veri/vit_prom.yml`)
```yaml
MODEL:
  NAME: 'ViT-B-16'
  STRIDE_SIZE: [16, 16]
INPUT:
  SIZE_TRAIN: [256, 256]
  SIZE_TEST: [256, 256]
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]
```

**Critical preprocessing (different from standard CLIP!):**
- Input size: **256×256** (not 224×224)
- Normalization: **mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]** (not ImageNet mean/std)
- Channel order: **RGB** (standard for ViT)

### Checkpoint Structure
The `.pth` file contains a `state_dict` with keys like:
```
base.image_encoder.conv1.weight
base.image_encoder.transformer.resblocks.0.attn.in_proj_weight
...
base.classifier.weight        # ID classification head (not needed for inference)
base.bottleneck.0.weight      # BN layer after features
```

### Inference Pipeline (from `processor_clipreid_stage2.py`)
```python
# Forward pass extracts features before classifier
def forward(self, x, ...):
    image_features = self.image_encoder(x)   # ViT forward
    # Use features from BN (bottleneck) layer for retrieval
    feat = self.bottleneck(image_features)    # BN normalized
    # During eval: return BN features for distance computation
    return feat  # shape: (batch, 768)
```

## Task Decomposition

Take a deep breath and work through this step by step:

### Step 1: Download and Load CLIP-ReID VeRi Checkpoint

**Goal:** Download the ViT-CLIP-ReID pretrained checkpoint and extract the image encoder weights.

1. Download the VeRi ViT-CLIP-ReID model from Google Drive
   - File ID: `1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e`
   - Save to: `tmp/weights/clip_reid_veri_vit_b16.pth`
2. Load checkpoint via `torch.load(path, map_location="cpu")`
3. Inspect state_dict keys to understand the model structure
4. Identify the image encoder + bottleneck weights (discard classifier head)

### Step 2: Build CLIPReIDExtractor (`src/yowo/tracking/_reid.py`)

**Goal:** Add `CLIPReIDExtractor` class implementing `ReIDExtractor` protocol.

**Architecture:**
```python
class CLIPReIDExtractor:
    """CLIP-ReID ViT-B/16 fine-tuned on VeRi-776 for vehicle ReID.

    Uses the image encoder from CLIP-ReID (ViT-B/16 + BN bottleneck)
    fine-tuned on VeRi-776 training set. Produces 768-dim L2-normalized
    embeddings optimized for vehicle re-identification.

    Preprocessing: resize to 256×256, RGB, normalize with mean=0.5, std=0.5.
    """
    __slots__ = (
        "_embedding_dim", "_model", "_device",
        "_input_size", "_min_crop_area",
    )
```

**Two deployment paths (implement both, user chooses):**

**Path A: PyTorch inference (simpler, needs torch)**
- Load CLIP-ReID checkpoint into a minimal ViT model
- `model.eval()` + `torch.no_grad()` for inference
- Output: BN-normalized 768-dim features

**Path B: ONNX export + ORT inference (production, no torch at runtime)**
- Load checkpoint → build model → trace → export to ONNX
- Use `CLIPReIDExtractor(onnx_path)` just like `CLIPExtractor`
- Export script: `tmp/export_clip_reid_onnx.py`

**Preprocessing (CRITICAL — different from standard CLIP!):**
```python
# CLIP-ReID VeRi config:
_CLIPREID_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_CLIPREID_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_CLIPREID_INPUT_SIZE = 256  # NOT 224

def _crop_and_preprocess(self, frame, boxes):
    # crop → BGR→RGB → resize(256, 256) → /255 → (x - 0.5) / 0.5 → CHW
```

**Extract method:**
```python
def extract(self, frame_pixels, boxes_xyxy):
    # Same pattern as CLIPExtractor:
    # 1. Crop + preprocess → batch tensor
    # 2. Forward pass → raw features
    # 3. L2 normalize
    # 4. Map back to (N, D) with zeros for filtered crops
    # Returns: (N, 768) float32 L2-normalized
```

### Step 3: ONNX Export Script (`tmp/export_clip_reid_onnx.py`)

**Goal:** Export CLIP-ReID VeRi checkpoint to ONNX for production deployment.

```python
"""Export CLIP-ReID VeRi checkpoint to ONNX.

Usage:
    uv run python tmp/export_clip_reid_onnx.py \
        --checkpoint tmp/weights/clip_reid_veri_vit_b16.pth \
        --output tmp/exports/clip-reid-veri-vit-b16.onnx

Requires: torch, CLIP-ReID model code (cloned or vendored).
"""
```

**Steps:**
1. Clone/vendor minimal CLIP-ReID model code (just `model/make_model_clipreid.py` + `model/clip/`)
2. Build model with VeRi config (num_classes=576, camera_num=20)
3. Load pretrained weights
4. Set model to eval mode
5. Create dummy input `(1, 3, 256, 256)`
6. `torch.onnx.export()` — export only image encoder + BN (not classifier)
7. Optimize with `onnxslim.slim()`
8. Verify: compare PyTorch vs ONNX outputs (max_diff < 1e-4)

**Output ONNX model:**
- Input: `images` (batch, 3, 256, 256) float32
- Output: `embeddings` (batch, 768) float32

### Step 4: Benchmark on VeRi-776

**Goal:** Run the VeRi benchmark experiment with CLIP-ReID and compare against CLIP zero-shot and FastReID.

**Modify `tmp/experiment_veri_reid.py`:**
1. Add `CLIPReIDExtractor` to the extractor list
2. Run all three extractors: CLIP, FastReID, CLIP-ReID
3. Generate comparison table + visual grids

**Expected results:**

| Extractor | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|
| CLIP zero-shot | 9.32% | 31.82% | 52.98% | 63.41% |
| FastReID SBS-S50 | 8.43% | 30.21% | 48.15% | 57.21% |
| **CLIP-ReID VeRi** | **~83%** | **~97%** | **~98%** | **~99%** |

If results don't match within ±2% of reference (83.3% mAP, 97.4% R1), debug:
- Check preprocessing: mean/std must be [0.5, 0.5, 0.5], input 256×256
- Check feature extraction point: must use BN-normalized features, not raw ViT output
- Check L2 normalization: embeddings must be unit-norm
- Check VeRi evaluation protocol: exclude junk (same-camera) images

### Step 5: Integration into yowo CrossCameraTracker

**Goal:** Make CLIP-ReID the recommended extractor for vehicle cross-camera tracking.

1. Add to `src/yowo/tracking/__init__.py` exports:
   ```python
   from ._reid import CLIPReIDExtractor
   ```
2. Update `EmbeddingGallery` default threshold:
   - CLIP zero-shot: threshold 0.10 (embeddings too close)
   - **CLIP-ReID: threshold 0.35-0.40** (much wider separation between same/different vehicles)
3. Document recommended usage:
   ```python
   from yowo.tracking import CLIPReIDExtractor, CrossCameraTracker

   reid = CLIPReIDExtractor("tmp/exports/clip-reid-veri-vit-b16.onnx")
   tracker = CrossCameraTracker(reid_extractor=reid, match_threshold=0.35)
   ```

### Step 6: Unit Tests

**Required tests:**

**TestCLIPReIDExtractor:**
- Protocol compliance: `isinstance(extractor, ReIDExtractor)`
- `embedding_dim` returns 768
- `extract()` returns correct shape `(N, 768)`
- L2-normalized output (norm ≈ 1.0)
- Filtered crops → zero rows
- Empty boxes → None
- Preprocessing uses 256×256 and mean/std=[0.5, 0.5, 0.5]

**TestCLIPReIDVeRiBenchmark (integration, optional):**
- mAP > 80% on VeRi-776 test set (skip if weights not available)

### Step 7: Quality Gates

```bash
uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q
```

Verify `_reid.py` stays < 700 lines. If it exceeds, extract `CLIPReIDExtractor` to `_clip_reid.py`.

## Chain-of-Thought Guidance

For each step:
- **Preprocessing is the #1 failure mode.** CLIP-ReID uses `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]` and `256×256`. NOT ImageNet mean/std. NOT 224×224. Getting this wrong produces ~50% mAP drop.
- **Feature extraction point matters.** Use BN-normalized features (after bottleneck), not raw ViT [CLS] token. The BN layer whitens the features for cosine distance.
- **The classifier head is NOT used at inference.** Discard `classifier.weight` from the checkpoint — only `image_encoder.*` and `bottleneck.*` are needed.
- **ONNX export must trace the eval path.** In training mode, the model returns different outputs (logits + features). Set `model.eval()` before trace.
- **VeRi evaluation protocol:** For each query, compute distance to ALL gallery images, exclude same-camera (junk) images, then compute CMC and mAP. This is exactly what `tmp/experiment_veri_reid.py` already does.

## CLIP-ReID Key Files Reference

```
github.com/Syliz517/CLIP-ReID/
├── configs/veri/vit_prom.yml           # VeRi config (input 256×256, mean/std=0.5)
├── model/make_model_clipreid.py        # Model builder (build_transformer class)
├── model/clip/model.py                 # CLIP ViT implementation
├── model/clip/clip.py                  # CLIP loading utils
├── datasets/veri.py                    # VeRi dataset loader
├── datasets/make_dataloader_clipreid.py # DataLoader factory
├── processor/processor_clipreid_stage2.py # Inference processor (do_inference)
├── test_clipreid.py                    # Test entry point
└── train_clipreid.py                   # Training entry point
```

## Self-Evaluation Framework

After implementation, rate confidence (0-1) on:

1. **Completeness**: CLIP-ReID extractor + ONNX export + VeRi benchmark + integration
2. **Clarity**: Code follows yowo conventions (__slots__, Protocol, L2 normalize)
3. **Practicality**: Works with existing pipeline, no breaking changes
4. **Optimization**: ONNX path for production, PyTorch path for development
5. **Edge Cases**: Missing weights file, wrong preprocessing, empty crops
6. **Self-Evaluation**: VeRi mAP within ±2% of reference (83.3%)

Provide a score for each (0-1).
If any score < 0.9, refine your answer before presenting.
