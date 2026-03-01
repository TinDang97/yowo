# Prompt: Complete CLIP ReID Integration — Model Download, Export, Inference & Experiments

> **Type**: Implementation prompt (sub-agent / direct execution)
> **Target Model**: Claude Opus 4.6
> **Depends on**: Completed code integration (Steps 1-5 in `clip-reid-integration-architecture.md`)
> **Output**: Working end-to-end pipeline + benchmark results

---

## Expert Persona (P16)

You are a Senior Machine Learning Engineer with 15+ years specializing in production multi-object tracking (MOT) systems and ONNX-based model deployment. You have deep expertise in:
- ByteTrack internals: two-stage IoU association, Kalman filtering, STrack lifecycle management
- CLIP visual encoders: ViT-B/16 architecture, ONNX export, preprocessing, embedding normalization
- NumPy-based batch inference pipelines (no PyTorch at runtime)
- Python 3.12+, strict pyright type checking, Protocol-based interfaces, frozen dataclasses
- Production CCTV constraints: <100ms frame budget, 24/7 memory stability, graceful degradation

## Stakes (P6)

This is the final integration step for yowo v2.2.0 — making the CLIP ReID actually work end-to-end with a real model. The code integration is done (STrack, _reid.py, _matching.py, _tracker.py) but there is no downloaded model, no ONNX export script, and no experiment validating that IDS reduction is real. Without this, the feature is dead code. Getting the ONNX export wrong will silently produce garbage embeddings. I'll tip you $200 for a fully working, validated pipeline.

---

## Context: What Already Exists

### Completed Code (v2.2.0 integration, all tests pass)

```
src/yowo/tracking/
├── __init__.py       (94 lines)  — exports: ByteTracker, CLIPExtractor, ReIDExtractor, track_stream, track_detections
├── _tracker.py       (367 lines) — ByteTracker with optional reid_extractor, conditional extraction, Stage 3 rescue
├── _strack.py        (322 lines) — STrack with _embedding slot, update_embedding() EMA
├── _matching.py      (411 lines) — appearance_distance(), gated_fused_cost(), needs_reid() + existing IoU
├── _reid.py          (222 lines) — ReIDExtractor Protocol + CLIPExtractor (ONNX-based)
├── _kalman.py        (195 lines) — KalmanFilterXYAH — NO CHANGES
```

### CLIPExtractor API (already implemented in _reid.py)

```python
from yowo.tracking import CLIPExtractor, ByteTracker, track_stream

# Initialize with ONNX model
clip = CLIPExtractor(
    model_path="path/to/clip-vit-b16-visual.onnx",
    embedding_dim=512,
    input_size=224,
    device="cpu",  # or "cuda", "coreml"
    min_crop_area=1024,
)

# Create tracker with ReID
tracker = ByteTracker(
    reid_extractor=clip,
    reid_frame_interval=3,
    reid_lost_age=5,
)

# Or via convenience function
with InferenceEngine() as engine:
    for tracked in track_stream(engine, source, reid_extractor=clip):
        for box in tracked.boxes:
            print(f"Track {box.track_id}: {box.class_name}")
```

### Existing Infrastructure
- ONNX Runtime already installed (`pip install yowo[onnx]`)
- `tmp/weights/` contains YOLO weights
- `tmp/exports/` contains YOLO ONNX models (yolo26n.onnx, etc.)
- `tmp/bunny_clip.mp4` — test video (Big Buck Bunny, ~8s at 24fps)
- `tmp/experiment_tracker.py` — reference for experiment script pattern
- Quality gates: `uv run ruff check && uv run pyright && uv run pytest tests/unit/ -x -q`

### CLIP ONNX Export Details (from method2-clip-zero-shot-reid.md)

- Model: OpenCLIP ViT-B/16, pretrained on LAION-2B (`laion2b_s34b_b79k`)
- Export visual encoder ONLY (not text encoder)
- Dynamic batch axis for variable crop counts per frame
- opset_version=17, do_constant_folding=True
- onnxslim post-export optimization (safe for CLIP, no KV-cache issues)
- Input: `images` shape (batch, 3, 224, 224) float32
- Output: `embeddings` shape (batch, 512) float32
- Expected ONNX file size: ~340MB after onnxslim

### CLIP Normalization Constants (already in _reid.py)

```python
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
```

---

## Task Decomposition (P3)

Take a deep breath and work through this step by step. Each step produces a concrete, testable artifact.

### Step 1: Create CLIP ONNX Export Script (`tmp/export_clip_onnx.py`)

Create a standalone script that:

1. Installs/verifies `open_clip_torch` and `onnxslim` are available (print clear error if not)
2. Downloads the OpenCLIP ViT-B/16 model (pretrained=`laion2b_s34b_b79k`)
3. Extracts the visual encoder (`model.visual`)
4. Exports to ONNX with:
   - `input_names=["images"]`, `output_names=["embeddings"]`
   - `dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}}`
   - `opset_version=17`, `do_constant_folding=True`
5. Runs `onnxslim.slim()` on the exported model
6. Saves to `tmp/exports/clip-vit-b16-visual.onnx`
7. Validates the exported model:
   - Load with ONNX Runtime
   - Run dummy input `(1, 3, 224, 224)` zeros
   - Verify output shape `(1, 512)`
   - Run batch input `(4, 3, 224, 224)` random
   - Verify output shape `(4, 512)`
   - Verify outputs are finite and non-zero
8. Compares a single-image embedding between PyTorch and ONNX:
   - Create a deterministic test image (e.g., `np.random.RandomState(42).rand(224, 224, 3)`)
   - Run through PyTorch visual encoder
   - Run through ONNX session
   - Assert max absolute difference < 1e-4
9. Print summary: model size, export time, validation status

**Script pattern** (follow `tmp/experiment_tracker.py` style):
```python
"""CLIP ViT-B/16 Visual Encoder — ONNX Export.

Downloads OpenCLIP ViT-B/16, exports visual encoder to ONNX,
optimizes with onnxslim, and validates output correctness.

Prerequisites:
    pip install open-clip-torch onnxslim

Run: uv run python tmp/export_clip_onnx.py
Output: tmp/exports/clip-vit-b16-visual.onnx (~340MB)
"""
```

**Do NOT**:
- Export the text encoder (not needed for ReID)
- Use torch.compile or torch.jit.trace (use torch.onnx.export directly)
- Hardcode GPU — script must work on CPU

### Step 2: Create Preprocessing Validation Script (`tmp/validate_clip_preprocess.py`)

Create a script that validates yowo's CLIPExtractor preprocessing matches OpenCLIP's torchvision reference:

1. Load a real image (from `tmp/bunny_clip.mp4` first frame, or a generated test image)
2. Process through OpenCLIP's official `preprocess` transforms (torchvision)
3. Process through `CLIPExtractor._crop_and_preprocess()` (our NumPy pipeline)
4. Compare the preprocessed tensors: assert max abs diff < 1e-5
5. Run both through the ONNX model and compare embeddings: assert max abs diff < 1e-4
6. Test with multiple crop sizes (small, medium, large) to verify resize consistency

```python
"""CLIP Preprocessing Validation.

Compares yowo's NumPy preprocessing pipeline against OpenCLIP's
torchvision reference to ensure embedding correctness.

Run: uv run python tmp/validate_clip_preprocess.py
Requires: tmp/exports/clip-vit-b16-visual.onnx (from export_clip_onnx.py)
"""
```

### Step 3: Create End-to-End Experiment Script (`tmp/experiment_clip_reid.py`)

This is the main experiment script. It runs 6 experiments comparing ByteTrack with and without CLIP ReID on real video:

**Experiment 1: Baseline ByteTrack (no ReID)**
- Run `track_stream()` on `tmp/bunny_clip.mp4` with default ByteTracker
- Record: total frames, unique track IDs, FPS, tracking_time_ms statistics

**Experiment 2: ByteTrack + CLIP ReID**
- Run `track_stream()` with `reid_extractor=CLIPExtractor("tmp/exports/clip-vit-b16-visual.onnx")`
- Record: same metrics as Exp 1

**Experiment 3: IDS Comparison**
- Compare unique track IDs between Exp 1 and Exp 2
- Lower unique IDs = fewer identity switches (if same objects tracked)
- Report: `baseline_ids`, `reid_ids`, `reduction_%`

**Experiment 4: Latency Breakdown**
- For 50 frames, measure:
  - `detect_ms`: engine.detect() time
  - `reid_extract_ms`: time spent in ReID extraction (if any)
  - `track_update_ms`: tracker.update() time (includes ReID)
  - `total_ms`: end-to-end per-frame
- Report: mean, p50, p95, max for each

**Experiment 5: Conditional ReID Skip Rate**
- Track how many frames actually triggered ReID extraction vs skipped
- Report: `total_frames`, `reid_extracted_frames`, `skip_rate_%`
- This validates the `needs_reid()` and `reid_frame_interval` throttling

**Experiment 6: Memory Stability**
- Run 200 frames and measure:
  - Peak RSS at start vs end
  - Number of active tracks over time
  - Embedding memory: `num_tracks * embedding_dim * 4 bytes`
- Verify no memory growth (EMA = constant per track)

**Script structure**:
```python
"""CLIP ReID Integration — End-to-End Experiment.

Compares ByteTrack with and without CLIP ReID on real video.
6 experiments: baseline, reid, IDS comparison, latency, skip rate, memory.

Run: uv run python tmp/experiment_clip_reid.py
Requires:
  - tmp/exports/yolo26n.onnx (YOLO detection model)
  - tmp/exports/clip-vit-b16-visual.onnx (CLIP visual encoder)
  - tmp/bunny_clip.mp4 (test video)
"""
```

**Config at top of script**:
```python
VIDEO = "tmp/bunny_clip.mp4"
YOLO_WEIGHTS = Path("tmp/exports/yolo26n.onnx")
CLIP_WEIGHTS = Path("tmp/exports/clip-vit-b16-visual.onnx")
FAMILY = ModelFamily.YOLO26
SIZE = ModelSize.NANO
BACKEND = BackendType.ONNX
MAX_FRAMES = 200
```

**Important**: Use `time.perf_counter()` for latency measurement. Use `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` for memory on macOS (divide by 1024 for MB).

### Step 4: Write Experiment Report (`docs/experiments/2026-02-28-clip-reid-experiment.md`)

After running the experiments, create a markdown report following the existing pattern in `docs/experiments/`. Include:

1. **Summary**: 1-paragraph overview of results
2. **Setup**: Hardware, model versions, video details
3. **Results Table**: All 6 experiments with metrics
4. **Latency Breakdown Table**: Per-component timing
5. **IDS Analysis**: Track ID comparison with explanation
6. **Conditional Skip Rate Analysis**: How effective the needs_reid gate is
7. **Memory Analysis**: RSS stability over 200 frames
8. **Conclusions**: Is CLIP ReID viable for production? What are the trade-offs?
9. **Recommendations**: Default config, when to enable/disable, edge deployment notes

---

## Constraints (Non-Negotiable)

1. **No new production dependencies**: `open_clip_torch` and `onnxslim` are ONLY for the export script (tmp/). The runtime code in `src/yowo/` uses only `onnxruntime` (already a dependency).
2. **Scripts in `tmp/`**: All scripts go in `tmp/` directory — they are tooling, not production code.
3. **Report in `docs/experiments/`**: Follow existing report naming convention.
4. **No modifications to src/yowo/**: The code integration is complete and validated. Do NOT modify any files in `src/`.
5. **Must work on CPU**: All scripts must run on CPU-only machines. GPU is optional acceleration.
6. **File size**: Export script < 200 lines. Experiment script < 400 lines. Report < 300 lines.

---

## Chain-of-Thought Guidance (P12, P19)

For each step:
- **Export script**: The critical detail is extracting `model.visual` not the full CLIP model. The text encoder is ~63M params and NOT needed. Exporting the full model would waste 150MB+ and produce a different output shape.
- **Preprocessing validation**: OpenCLIP's `preprocess` uses `torchvision.transforms.Compose([Resize(224, bicubic), CenterCrop(224), ToTensor(), Normalize(mean, std)])`. Our pipeline uses `cv2.resize(INTER_LINEAR)` which differs from bicubic. The max diff should still be < 1e-4 for embeddings because CLIP is robust to resize interpolation method. If it's > 1e-4 but < 1e-3, that's acceptable — document the cause.
- **IDS comparison**: The test video (Big Buck Bunny) has animated characters, not real CCTV. IDS reduction may be less dramatic than the 25-32% from the research. Focus on verifying the pipeline works correctly rather than hitting specific IDS numbers.
- **Memory**: On macOS, `ru_maxrss` returns bytes (not KB). On Linux, it returns KB. Handle both.
- **Skip rate**: If `bunny_clip.mp4` has few overlapping tracks, `needs_reid` may return False most frames, resulting in high skip rate (70-90%). This is correct behavior — it means the conditional gate is working and saving compute.

---

## Self-Evaluation Framework (P15)

After completing all 4 steps, verify:

1. **Export Correctness**: ONNX model loads, produces (batch, 512) output, matches PyTorch within 1e-4
2. **Preprocessing Match**: NumPy pipeline matches torchvision reference within 1e-5 (tensor) / 1e-4 (embedding)
3. **End-to-End Works**: `track_stream()` with `reid_extractor=CLIPExtractor(...)` produces TrackedDetection without errors
4. **No Regressions**: Baseline (no ReID) FPS unchanged from v2.1.0
5. **Report Complete**: All 6 experiments documented with real numbers

Rate confidence (0-1) on each. If any < 0.9, fix before presenting.

---

## Reference Documents

| Document | Location | Relevant Sections |
|----------|----------|-------------------|
| Master Architecture | `docs/plans/clip-reid-integration-architecture.md` | Sec 2 (research), Sec 4-5 (architecture) |
| CLIP Method Spec | `docs/plans/method2-clip-zero-shot-reid.md` | Sec 5 (ONNX export), Sec 6 (preprocessing), Sec 12 (perf) |
| FastReID Baseline | `docs/plans/method1-fastreid-baseline.md` | Sec 7 (latency tables) — for comparison |
| Existing Experiment | `tmp/experiment_tracker.py` | Script pattern reference |
| Existing Report | `docs/experiments/2026-02-28-bytetrack-counter-annotated-video-benchmark.md` | Report format reference |

## ONNX Export Reference (from method2-clip-zero-shot-reid.md Sec 5.1)

```python
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b79k"
)
model.eval()
visual = model.visual  # ONLY the visual encoder

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    visual, dummy_input, "clip-vit-b16-visual.onnx",
    input_names=["images"], output_names=["embeddings"],
    dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
    opset_version=17, do_constant_folding=True,
)

# Post-export optimization
import onnxslim
optimized = onnxslim.slim("clip-vit-b16-visual.onnx")
import onnx
onnx.save(optimized, "clip-vit-b16-visual.onnx")
```
