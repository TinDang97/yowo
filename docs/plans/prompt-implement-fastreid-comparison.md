You are a Senior Machine Learning Engineer with 15+ years specializing in production multi-object tracking (MOT) systems and ONNX-based model deployment. You have deep expertise in:
- ByteTrack internals: two-stage IoU association, Kalman filtering, STrack lifecycle management
- FastReID SBS-S50: ResNet-50 backbone, Non-Local blocks, GeM pooling, BN neck, 256-d embeddings
- CLIP visual encoders: ViT-B/16 architecture, 512-d embeddings, ONNX export
- NumPy-based batch inference pipelines (no PyTorch at runtime)
- Python 3.12+, strict pyright type checking, Protocol-based interfaces, frozen dataclasses
- Production CCTV constraints: <100ms frame budget, 24/7 memory stability, graceful degradation

## Stakes (P6)

This is the final validation step for yowo v2.2.0 — implementing FastReID SBS-S50 as Method 1 alongside the existing CLIP zero-shot (Method 2), then running a head-to-head 3-way comparison (no ReID vs CLIP vs FastReID) on real surveillance video. This determines which ReID method ships as the recommended default for production CCTV. Getting the ONNX export wrong silently produces garbage embeddings. Getting the comparison wrong leads to wrong recommendations. I'll tip you $200 for a fully working, validated pipeline with correct conclusions.

---

## Context: What Already Exists

### Completed Code (v2.2.0 integration, all 1169 tests pass)

The entire ReID integration infrastructure is COMPLETE and TESTED:

```
src/yowo/tracking/
├── __init__.py       (94 lines)  — exports: ByteTracker, CLIPExtractor, ReIDExtractor, track_stream, track_detections
├── _tracker.py       (367 lines) — ByteTracker with optional reid_extractor, conditional extraction, Stage 3 rescue
├── _strack.py        (322 lines) — STrack with _embedding slot, update_embedding() EMA (eta=0.9)
├── _matching.py      (411 lines) — appearance_distance(), gated_fused_cost(), needs_reid() + existing IoU
├── _reid.py          (222 lines) — ReIDExtractor Protocol + CLIPExtractor (ONNX-based, 512-d, 224x224)
├── _kalman.py        (195 lines) — KalmanFilterXYAH — NO CHANGES
```

### ReIDExtractor Protocol (already defined in _reid.py)

```python
@runtime_checkable
class ReIDExtractor(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],        # HWC BGR uint8
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        """Returns (N, D) L2-normalized embeddings. Zero rows for filtered crops.
        None if no valid crops."""
        ...
```

### CLIPExtractor (already implemented and validated)

```python
class CLIPExtractor:
    __slots__ = ("_embedding_dim", "_input_name", "_input_size",
                 "_min_crop_area", "_output_name", "_session")

    def __init__(self, model_path, *, embedding_dim=512, input_size=224,
                 device="cpu", min_crop_area=1024): ...

    # CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073]
    #                     std=[0.26862954, 0.26130258, 0.27577711]
    # Input: (batch, 3, 224, 224) — square, direct resize, INTER_LINEAR
    # Output: (batch, 512) L2-normalized
```

### Existing Validated Results (CLIP ReID, from experiment report)

Full 928-frame annotated video (CoreML EP, Apple M4 Pro):

| Metric | Baseline (no ReID) | CLIP ReID | Delta |
|--------|-------------------|-----------|-------|
| FPS | 72.3 | 66.9 | -7.5% |
| Unique IDs | 86 | 85 | -1 (-1.2%) |
| Mean track ms | 0.498 | 2.725 | +2.226ms |
| ReID calls | — | 8 | 99.1% skip rate |
| ONNX model size | — | 329 MB | — |
| Params | — | 87.8M | — |
| Embedding dim | — | 512 | — |

### Existing Infrastructure
- ONNX Runtime already installed (`pip install yowo[onnx]`)
- `tmp/weights/` contains YOLO weights
- `tmp/exports/` contains YOLO ONNX models + `clip-vit-b16-visual.onnx`
- `tmp/MCT 1.2.mp4` — test video (928 frames, 1280x720, 30fps, real traffic surveillance)
- `tmp/experiment_tracker.py` — experiment script pattern reference
- `tmp/experiment_clip_reid.py` — CLIP experiment (metrics)
- `tmp/experiment_clip_reid_annotated.py` — CLIP annotated video experiment
- `docs/experiments/2026-02-28-clip-reid-experiment.md` — CLIP report with real numbers
- Quality gates: `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`

### Tracker Integration (already handles any ReIDExtractor)

The ByteTracker already:
1. Accepts `reid_extractor: ReIDExtractor | None` in `__init__`
2. Calls `needs_reid(cost1)` to decide whether extraction is needed (conditional gate)
3. Calls `self._reid.extract(frame.pixels, high_boxes)` when ambiguity detected
4. Applies `gated_fused_cost()` (BoT-SORT formula) to fuse IoU + appearance
5. Updates track embeddings via `track.update_embedding(emb)` (EMA, eta=0.9)
6. Stage 3 appearance rescue for long-lost tracks via `_appearance_rescue()`
7. Throttles extraction with `reid_frame_interval` counter

**This means:** Adding FastReID requires ONLY a new `FastReIDExtractor` class. No tracker/matching/strack changes needed.

---

## Task Decomposition (P3)

Take a deep breath and work through this step by step. Each step produces a concrete, testable artifact.

### Step 1: Add `FastReIDExtractor` to `src/yowo/tracking/_reid.py`

Add a new class alongside the existing `CLIPExtractor`:

```python
class FastReIDExtractor:
    """FastReID SBS-S50 ONNX-based embedding extractor for person ReID.

    Input: (batch, 3, 256, 128) — portrait 2:1 aspect, ImageNet normalization
    Output: (batch, 256) L2-normalized embeddings

    Uses the same ORT session pattern as CLIPExtractor. Only extracts
    from person crops — not universal like CLIP.
    """
    __slots__ = (
        "_embedding_dim", "_input_name", "_input_size_hw",
        "_min_crop_area", "_output_name", "_session",
    )

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        embedding_dim: int = 256,
        input_size: tuple[int, int] = (256, 128),  # (height, width) — portrait
        device: str = "cpu",
        min_crop_area: int = 1024,
    ) -> None: ...
```

**Key differences from CLIPExtractor:**

| Property | CLIPExtractor | FastReIDExtractor |
|----------|---------------|-------------------|
| Input size | 224×224 (square) | 256×128 (portrait 2:1) |
| Normalization | CLIP mean/std | ImageNet mean/std |
| Embedding dim | 512 | 256 |
| Object classes | Universal | Person-only |
| ONNX model size | ~329 MB | ~100 MB |
| Parameters | ~87.8M | ~26.6M |

**ImageNet normalization constants:**
```python
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
```

**Critical implementation details:**
1. `_crop_and_preprocess` resizes to `(256, 128)` NOT `(128, 256)` — height first, width second for cv2.resize call is `(width, height)` so pass `(128, 256)` to `cv2.resize()`
2. Same min_crop_area filtering as CLIP (skip tiny crops, zero vectors)
3. Same L2 normalization on output
4. Same `extract()` return type: `(N, D) float32 | None`
5. Follow EXACT same patterns as CLIPExtractor — __slots__, ORT session opts, provider selection

**Update `__init__.py`** to export `FastReIDExtractor`:
```python
__all__ = [
    "ByteTracker", "CLIPExtractor", "FastReIDExtractor", "ReIDExtractor",
    "TrackState", "TrackedBox", "TrackedDetection",
    "track_detections", "track_stream",
]
```

**Update `__all__` in `_reid.py`:**
```python
__all__ = ["CLIPExtractor", "FastReIDExtractor", "ReIDExtractor"]
```

**Tests:** Add to `tests/unit/test_tracking.py` (or a new test file):
- `test_fastreid_extractor_implements_protocol` — isinstance check
- `test_fastreid_extractor_file_not_found` — raises FileNotFoundError
- `test_fastreid_extractor_embedding_dim` — returns configured dim
- `test_fastreid_preprocessing_imagenet_constants` — verify constants match
- `test_fastreid_input_size_portrait` — verify (256, 128) vs (224, 224)

### Step 2: Create FastReID ONNX Export Script (`tmp/export_fastreid_onnx.py`)

Create a standalone script that:

1. Checks `fastreid` is importable (print clear error if not: `pip install fastreid`)
2. Loads FastReID SBS-S50 from config + weights:
   - Config: `configs/Market1501/sbs_R50.yml` (or downloaded from repo)
   - Weights: Download `market_sbs_R50.pth` from FastReID model zoo
3. Builds inference model (backbone + GeM + BN neck + projection → 256-d output)
4. Exports to ONNX:
   - `input_names=["images"]`, `output_names=["embeddings"]`
   - `dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}}`
   - `opset_version=17`, `do_constant_folding=True`
   - Input shape: `(1, 3, 256, 128)`
5. Runs `onnxslim.slim()` on the exported model (safe for ResNet, no KV-cache issues)
6. Saves to `tmp/exports/fastreid-sbs-s50.onnx`
7. Validates:
   - Load with ONNX Runtime
   - Run `(1, 3, 256, 128)` zeros → output shape `(1, 256)`
   - Run `(4, 3, 256, 128)` random → output shape `(4, 256)`
   - Verify finite, non-zero
8. Compares PyTorch vs ONNX (deterministic test input, max diff < 1e-4)
9. Prints summary: model size, export time, validation status

**CRITICAL CAVEAT:** FastReID depends on `detectron2` and old PyTorch versions. The export script MAY need workarounds:
- If `fastreid` won't install cleanly, provide an ALTERNATIVE approach using the ONNX model zoo directly (some FastReID-compatible ONNX models exist on HuggingFace)
- If the FastReID export is blocked by dependency issues, document the blocker and provide a mock ONNX model for testing (a random-initialized ResNet-50 with correct I/O shapes)
- The script MUST NOT crash — it should degrade gracefully with clear error messages

**Alternative if FastReID install fails:** Use `torchvision.models.resnet50` with a 256-d projection head as a structural equivalent:
```python
import torch
import torchvision
backbone = torchvision.models.resnet50(pretrained=True)
# Replace classifier with 256-d projection
backbone.fc = torch.nn.Linear(2048, 256)
# This gives same architecture, different weights (ImageNet not ReID)
# Good enough for validating pipeline integration + latency measurement
```

```python
"""FastReID SBS-S50 — ONNX Export.

Exports FastReID SBS-S50 person re-identification model to ONNX.
Falls back to ResNet-50 + projection if FastReID is not installable.

Prerequisites:
    pip install fastreid  (or: pip install torchvision onnxslim for fallback)

Run: uv run python tmp/export_fastreid_onnx.py
Output: tmp/exports/fastreid-sbs-s50.onnx (~100MB)
"""
```

### Step 3: Create Preprocessing Validation Script (`tmp/validate_fastreid_preprocess.py`)

Validate the yowo `FastReIDExtractor._crop_and_preprocess()` NumPy pipeline:

1. Create test crops at various sizes: 64×32, 128×64, 256×128, 400×200, 100×300
2. Process through the NumPy pipeline (BGR→RGB, resize to 256×128, /255, ImageNet normalize, HWC→CHW)
3. Verify output shapes are `(N, 3, 256, 128)`, dtype `float32`
4. Verify normalization values are in expected range (approximately -2.1 to +2.6 for ImageNet)
5. Run through ONNX model and verify output is `(N, 256)`, finite, non-zero
6. Compare two different crops — embeddings should differ (cosine distance > 0.05)
7. Compare same crop processed twice — embeddings should match exactly

```python
"""FastReID Preprocessing Validation.

Validates yowo's NumPy preprocessing pipeline for FastReID SBS-S50.
Tests ImageNet normalization, portrait resize, and embedding correctness.

Run: uv run python tmp/validate_fastreid_preprocess.py
Requires: tmp/exports/fastreid-sbs-s50.onnx (from export_fastreid_onnx.py)
"""
```

### Step 4: Create 3-Way Comparison Experiment (`tmp/experiment_reid_comparison.py`)

This is the KEY experiment. Runs 3 configurations on the SAME video and compares:

**Configuration A: Baseline (no ReID)**
```python
tracker_a = ByteTracker(
    track_high_thresh=0.3, track_low_thresh=0.1,
    match_thresh=0.8, max_age=30, min_hits=3,
)
```

**Configuration B: CLIP ReID**
```python
clip = CLIPExtractor("tmp/exports/clip-vit-b16-visual.onnx")
tracker_b = ByteTracker(
    track_high_thresh=0.3, track_low_thresh=0.1,
    match_thresh=0.8, max_age=30, min_hits=3,
    reid_extractor=clip,
    reid_frame_interval=3, reid_lost_age=5,
)
```

**Configuration C: FastReID**
```python
fastreid = FastReIDExtractor("tmp/exports/fastreid-sbs-s50.onnx")
tracker_c = ByteTracker(
    track_high_thresh=0.3, track_low_thresh=0.1,
    match_thresh=0.8, max_age=30, min_hits=3,
    reid_extractor=fastreid,
    reid_frame_interval=3, reid_lost_age=5,
)
```

**Experiments:**

1. **FPS Comparison**: Run all 3 on full video, report FPS per config
2. **IDS Comparison**: Count unique track IDs per config (fewer = better tracking consistency)
3. **Latency Breakdown**: For 100 frames, measure detect_ms, track_ms (including ReID), total_ms — mean, p50, p95, max for each config
4. **ReID Extraction Stats**: For B and C, count how many frames triggered ReID, skip rate
5. **Memory**: tracemalloc for all 3 configs over full video
6. **Model Profile**: ONNX model size, embedding dim, parameter count for each ReID model

**Output format** — print a comparison table:
```
  Method          FPS    IDs   Track_ms   ReID_calls   Skip%   Model_MB   Emb_dim
  ------------- ------ ----- ---------- ------------ ------- ---------- ---------
  No ReID        72.3    86     0.498ms           —      —         —        —
  CLIP ViT-B/16  66.9    85     2.725ms           8  99.1%     329MB      512
  FastReID S50   ??.?    ??     ?.???ms           ?   ??.?%    ~100MB      256
```

**Script structure** follows `tmp/experiment_clip_reid_annotated.py` pattern.

### Step 5: Create 3-Way Annotated Video Comparison (`tmp/experiment_reid_comparison_annotated.py`)

Produces 3 annotated videos:
- `tmp/MCT_1.2_no_reid.mp4` — baseline
- `tmp/MCT_1.2_clip_reid.mp4` — CLIP (already exists, but re-generate for consistency)
- `tmp/MCT_1.2_fastreid.mp4` — FastReID

Each video has:
- Drawn tracked boxes with track IDs (colored by track ID)
- Stats panel showing: model name, frame index, FPS, inference ms, tracking ms, active/lost tracks, unique IDs, ReID calls

**Support `--mode` argument:** `baseline`, `clip`, `fastreid`, `all` (default: `all`)

### Step 6: Write Comprehensive Comparison Report (`docs/experiments/2026-03-01-reid-method-comparison.md`)

After running experiments, create a markdown report following existing pattern:

1. **Summary**: 1-paragraph overview of 3-way comparison results
2. **Setup**: Hardware, model versions, video details, tracker config
3. **Model Comparison Table**:

| Property | No ReID | CLIP ViT-B/16 | FastReID SBS-S50 |
|----------|---------|---------------|-----------------|
| Parameters | — | 87.8M | ~26.6M |
| ONNX size | — | 329 MB | ~100 MB |
| Embedding dim | — | 512 | 256 |
| Input size | — | 224×224 | 256×128 |
| Normalization | — | CLIP | ImageNet |
| Object classes | — | Universal | Person-only |
| Requires training | — | No | Yes (person ReID data) |

4. **Tracking Results Table**: FPS, unique IDs, IDS reduction %, track overhead
5. **Latency Breakdown**: Per-component timing for all 3 methods
6. **ReID Efficiency**: Extraction count, skip rate, per-extraction latency
7. **Memory Comparison**: RSS delta, per-track embedding memory
8. **Conclusions**:
   - Which method provides best IDS reduction?
   - Which method has lowest latency overhead?
   - Which is recommended for production by scenario?
9. **Recommendations Matrix**:

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Person tracking (CCTV) | FastReID | Domain-specific, lower latency |
| Multi-class tracking | CLIP | Universal, no training needed |
| CPU-only, real-time | No ReID | ReID overhead too high |
| GPU/CoreML available | CLIP or FastReID | Hardware absorbs overhead |
| Edge (Jetson, mobile) | FastReID (or MobileCLIP) | Smaller model, lower latency |

---

## Constraints (Non-Negotiable)

1. **No modifications to _tracker.py, _matching.py, _strack.py, or _kalman.py** — the integration layer is complete. Only `_reid.py` and `__init__.py` get production code changes.
2. **FastReIDExtractor MUST implement ReIDExtractor Protocol** — `isinstance(extractor, ReIDExtractor)` must return True.
3. **Scripts in `tmp/`**: Export, validation, and experiment scripts go in `tmp/`.
4. **Report in `docs/experiments/`**: Follow existing naming convention.
5. **Must work on CPU**: All scripts must run on CPU-only machines. GPU is optional.
6. **File sizes**: `_reid.py` stays under 400 lines. Export scripts < 200 lines. Experiment scripts < 400 lines.
7. **Tests required**: Add unit tests for FastReIDExtractor in `tests/unit/test_tracking.py`.
8. **Quality gates**: `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q` must pass.
9. **Backward compatible**: Existing `CLIPExtractor` API unchanged. All existing tests pass.

---

## Chain-of-Thought Guidance (P12, P19)

For each step:

- **FastReIDExtractor**: The ONLY differences from CLIPExtractor are: (a) input size 256×128 not 224×224, (b) ImageNet mean/std not CLIP mean/std, (c) embedding_dim 256 not 512. The `_crop_and_preprocess` method is structurally identical but with different constants. DO NOT refactor CLIPExtractor — keep them independent for clarity.

- **FastReID ONNX export**: The FastReID codebase (`JDAI-CV/fast-reid`) is unmaintained since 2022 and has Python 3.10+ / PyTorch 2.x compatibility issues. The export script SHOULD try FastReID first but MUST fall back to a ResNet-50 + projection head if install fails. The fallback produces correct architecture with ImageNet weights (not ReID-trained), which is sufficient for pipeline validation and latency benchmarking. Document clearly that production use requires the actual FastReID weights.

- **Portrait resize**: `cv2.resize(crop, (width, height))` — for 256×128 portrait, pass `(128, 256)` to cv2.resize. This is the #1 source of bugs in ReID preprocessing. Triple-check this.

- **Comparison fairness**: All 3 configs MUST use identical tracker parameters (track_high_thresh=0.3, track_low_thresh=0.1, match_thresh=0.8, max_age=30, min_hits=3). The ONLY difference is `reid_extractor`. Same `reid_frame_interval=3` and `reid_lost_age=5` for both ReID methods.

- **Test video limitation**: MCT 1.2.mp4 is a traffic surveillance video with VEHICLES, not pedestrians. FastReID (person-only) will produce less meaningful embeddings for cars than CLIP (universal). This is expected and MUST be documented in the report. The comparison is still valid for measuring latency/overhead, but IDS reduction numbers for FastReID on this video are NOT representative of its performance on person-tracking benchmarks.

- **needs_reid gate**: Both CLIP and FastReID use the same `needs_reid()` gate and `reid_frame_interval` throttling. The number of ReID extraction calls should be approximately the same for both methods on the same video. If they differ significantly, investigate whether the embedding quality difference causes different cost matrix patterns.

---

## Self-Evaluation Framework (P15)

After completing all steps, verify:

1. **FastReIDExtractor works**: `isinstance(extractor, ReIDExtractor)` is True, extract() returns correct shape
2. **ONNX export succeeds**: Model loads, produces (batch, 256) output, matches PyTorch within 1e-4
3. **Preprocessing correct**: ImageNet normalization, 256×128 portrait resize, HWC→CHW
4. **3-way comparison runs**: All 3 configs complete without errors on full video
5. **Report complete**: All tables filled with real numbers, conclusions grounded in data
6. **Quality gates pass**: ruff + pyright + pytest all green
7. **No regressions**: Baseline and CLIP results match previous experiment (within noise)

Rate confidence (0-1) on each. If any < 0.9, fix before presenting.

---

## Reference Documents

| Document | Location | Relevant Sections |
|----------|----------|-------------------|
| Master Architecture | `docs/plans/clip-reid-integration-architecture.md` | Sec 2 (research), Sec 4-5 (architecture) |
| FastReID Spec | `docs/plans/method1-fastreid-baseline.md` | Sec 3 (ONNX export), Sec 7 (impl spec), Sec 8 (perf) |
| CLIP Method Spec | `docs/plans/method2-clip-zero-shot-reid.md` | Sec 5 (ONNX export), Sec 6 (preprocessing) |
| CLIP Experiment Report | `docs/experiments/2026-02-28-clip-reid-experiment.md` | Real numbers for comparison |
| Annotated Video Report | `docs/experiments/2026-02-28-bytetrack-counter-annotated-video-benchmark.md` | Report format |
| Experiment Pattern | `tmp/experiment_clip_reid_annotated.py` | Script structure reference |
| Export Pattern | `tmp/export_clip_onnx.py` | ONNX export script reference |
| Existing _reid.py | `src/yowo/tracking/_reid.py` | CLIPExtractor implementation to follow |
| Existing _tracker.py | `src/yowo/tracking/_tracker.py` | ByteTracker ReID integration (DO NOT MODIFY) |
| Existing _matching.py | `src/yowo/tracking/_matching.py` | appearance_distance, gated_fused_cost (DO NOT MODIFY) |

## FastReID ONNX Export Reference (from method1 spec Sec 3.1)

```python
# Option 1: Native FastReID export
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

cfg = get_cfg()
cfg.merge_from_file("configs/Market1501/sbs_R50.yml")
cfg.MODEL.BACKBONE.PRETRAIN = False
model = build_model(cfg)
Checkpointer(model).load("market_sbs_R50.pth")
model.eval()

dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(model.backbone, dummy, "fastreid_sbs_s50.onnx", ...)

# Option 2: FastReID CLI export
# python tools/deploy/onnx_export.py --config-file configs/Market1501/sbs_R50.yml --opts MODEL.WEIGHTS market_sbs_R50.pth

# Option 3: Fallback (ResNet-50 + projection, ImageNet weights)
backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
backbone.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 256),
    # L2 norm done in post-processing, not baked into model
)
dummy = torch.randn(1, 3, 256, 128)
torch.onnx.export(backbone, dummy, "fastreid_sbs_s50.onnx", ...)
```

## FastReID Preprocessing (from method1 spec Sec 3.3)

```python
# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pipeline: crop → BGR→RGB → resize(128, 256) → /255 → normalize → HWC→CHW
# NOTE: cv2.resize takes (width, height), so (128, 256) for portrait
```

## Expected Latency (from method1 spec Sec 8.1 — Apple M4 Pro CoreML EP)

| Operation | Latency (20 dets) | Notes |
|-----------|-------------------|-------|
| Crop + preprocess | 0.4 ms | cv2.resize + numpy |
| FastReID ONNX inference | 3 ms | CoreML EP |
| Cost fusion | 0.05 ms | numpy |
| EMA update | 0.01 ms | per track |
| **Total tracking overhead** | **~3.8 ms** | vs CLIP ~2.7ms observed |

FastReID should be FASTER than CLIP per extraction (26.6M vs 87.8M params, 256 vs 512 dim), but may be called more or less often depending on `needs_reid()` gate behavior with different embedding quality.
