# Prompt: Cross-Camera Vehicle ReID & Traffic-Domain Optimization for yowo

## Expert Persona

You are a Principal Computer Vision Engineer with 15 years specializing in multi-camera multi-target (MCMT) vehicle tracking systems deployed on traffic CCTV infrastructure. You have shipped city-scale systems processing 40+ camera feeds with IDF1 > 80% on CityFlow benchmarks. You excel at:
- Vehicle re-identification across cameras with viewpoint/lighting invariance
- Camera link models for spatial-temporal pruning
- Online cross-camera association without full-trajectory clustering
- ONNX-based ReID model integration optimized for edge/CPU inference
- Embedding gallery management with bounded memory

## Stakes

This is the core differentiator for yowo v3.0 — evolving from single-camera ByteTrack to a city-scale multi-camera vehicle tracking system. Getting the architecture right avoids $50,000+ in rework when scaling from 2 cameras to 40. A production-ready design that composes cleanly with the existing single-camera tracker is critical. I'll tip you $200 for a solution that works end-to-end with unit tests.

## Context — What Already Exists in yowo

### Single-Camera Tracking (`src/yowo/tracking/`)
```
ByteTracker         — 4-stage association: high-conf IoU → low-conf IoU → ReID rescue → unconfirmed
STrack              — per-track state: Kalman, embedding (EMA α=0.9), hits, age, confidence
KalmanFilterXYAH    — 8-state (x, y, a, h, vx, vy, va, vh), canonical ByteTrack noise
ReIDExtractor       — Protocol: embedding_dim property + extract(frame, boxes) → (N, D) embeddings
CLIPExtractor       — CLIP ViT-B/16 ONNX, 512-dim, zero-shot (no training needed)
FastReIDExtractor   — SBS-S50 ONNX, 256-dim, person-optimized (256×128 portrait)
track_stream()      — generator yielding TrackedFrame per frame
_matching.py        — iou_distance, appearance_distance, gated_fused_cost, needs_reid, fuse_score
```

Key design patterns:
- `ReIDExtractor` is a `@runtime_checkable Protocol` — any class with `embedding_dim` + `extract()` is valid
- Embeddings are L2-normalized, compared via cosine distance (`1 - dot`)
- `needs_reid()` gates ReID calls: only when tracks are lost for `reid_lost_age` frames AND `reid_frame_interval` elapsed
- EMA embedding update: `new = 0.9 * old + 0.1 * fresh`, then re-normalize

### Multi-Stream Pipeline (`src/yowo/pipeline/`)
```
FrameCollector      — thread-per-stream reader, yields TaggedFrame(source_id, frame_index, pixels)
BatchScheduler      — collects frames into batches with timeout, yields List[TaggedFrame]
DetectionRouter     — routes batch detections[i] back to source_id via positional index
run_pipeline()      — orchestrates collector → scheduler → engine.detect() → router
```

### Engine (`src/yowo/engine.py`)
```
InferenceEngine     — detect(frames) → List[Detection], backend-agnostic
                    — supports ONNX, TensorRT, CoreML, PyTorch backends
                    — batch inference, observable (EventBus, MetricsCollector)
```

### Types (`src/yowo/types.py`)
```
TrackedBox          — frozen dataclass: x1,y1,x2,y2, confidence, class_id, class_name, track_id, is_confirmed
TrackedFrame        — frame + boxes + inference_time_ms + tracking_time_ms
TaggedFrame         — source_id + frame_index + pixels (from pipeline)
Frame               — pixels + source_id + frame_index
```

### Quality Gates
```bash
uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q
```
- 1245 tests pass, 0 lint errors, 0 pyright errors
- Files MUST be < 700 lines
- All code: Python 3.11+, strict type hints, `__slots__`, no mocks/stubs

## Task Decomposition

Take a deep breath and implement this step by step:

### Step 1: Vehicle-Optimized ReID Extractor (`src/yowo/tracking/_reid.py`)

**Goal:** Add `VehicleReIDExtractor` that implements `ReIDExtractor` protocol, optimized for traffic CCTV vehicles.

**Requirements:**
- ONNX-based, same pattern as `CLIPExtractor` and `FastReIDExtractor`
- Input: vehicle crop resized to (224, 224) — square aspect for vehicles (not portrait like person ReID)
- Preprocessing: BGR→RGB, /255, ImageNet normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Output: 512-dim L2-normalized embeddings
- `min_crop_area` filter (default 2048 — vehicles are larger than people)
- Must handle partial vehicles (edge entry) gracefully — zero vector for filtered crops
- Add to `__all__` and `src/yowo/tracking/__init__.py` exports

**Vehicle-specific preprocessing considerations:**
- Vehicles have ~1:1 to ~2:1 aspect ratio (not 2:1 portrait like persons)
- Larger crops = more detail, but need consistent input size
- Color is a strong discriminator for vehicles (unlike persons where clothing changes)

**Model:** The extractor should accept any ONNX vehicle ReID model. Document recommended models:
- VeRi-trained ResNet50 (VehicleID/VeRi-776 benchmarks)
- CLIP ViT-B/16 (zero-shot fallback — already exists as `CLIPExtractor`)

### Step 2: Embedding Gallery (`src/yowo/tracking/_gallery.py` — NEW FILE)

**Goal:** A bounded, queryable embedding gallery that stores per-camera track embeddings for cross-camera matching.

**Requirements:**
```python
class EmbeddingGallery:
    """Bounded gallery of track embeddings across cameras for cross-camera ReID.

    Stores finalized track embeddings (from tracks that left the scene or
    expired) indexed by (camera_id, local_track_id). Provides efficient
    nearest-neighbor lookup for cross-camera identity matching.
    """
    __slots__ = ("_entries", "_max_entries", "_embedding_dim")

    def __init__(self, embedding_dim: int, max_entries: int = 10_000) -> None: ...

    def add(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: NDArray[np.float32],
        metadata: GalleryEntry | None = None,
    ) -> int:
        """Add finalized track embedding. Returns global_id."""
        ...

    def query(
        self,
        embedding: NDArray[np.float32],
        *,
        exclude_camera: str | None = None,
        top_k: int = 5,
        threshold: float = 0.4,
    ) -> list[GalleryMatch]:
        """Find nearest gallery entries. Exclude same-camera matches."""
        ...

    def evict_oldest(self, keep: int) -> int:
        """Remove oldest entries when gallery exceeds max_entries."""
        ...

    @property
    def size(self) -> int: ...
```

**Key design decisions:**
- `GalleryEntry` dataclass: `global_id, camera_id, local_track_id, embedding, class_id, timestamp, bbox_history`
- `GalleryMatch` dataclass: `global_id, distance, camera_id, local_track_id`
- Cosine distance: `1 - dot(query, gallery)` — embeddings are pre-L2-normalized
- Same-camera exclusion: never match within same camera (that's what single-camera tracker does)
- FIFO eviction when `len > max_entries`
- No external dependencies (no FAISS) — numpy-only for portability. Can add FAISS adapter later.
- Thread-safe: multiple cameras add/query concurrently → use `threading.Lock`

### Step 3: Camera Link Model (`src/yowo/tracking/_camera_link.py` — NEW FILE)

**Goal:** Spatial-temporal constraints that prune impossible cross-camera matches, reducing false positives by 90%+.

**Requirements:**
```python
@dataclass(frozen=True, slots=True)
class CameraLink:
    """Defines spatial-temporal transition between two cameras."""
    src_camera: str
    dst_camera: str
    min_transit_sec: float  # minimum travel time between cameras
    max_transit_sec: float  # maximum travel time between cameras
    direction: str = "any"  # "north", "south", "east", "west", "any"

class CameraLinkModel:
    """Prunes cross-camera matches using spatial-temporal constraints.

    If vehicle exits Camera A heading north at time T, it can only appear
    in Camera B between T + min_transit and T + max_transit.
    """
    __slots__ = ("_links", "_default_window")

    def __init__(
        self,
        links: list[CameraLink] | None = None,
        default_window: tuple[float, float] = (5.0, 120.0),
    ) -> None: ...

    def is_feasible(
        self,
        src_camera: str,
        dst_camera: str,
        src_exit_time: float,
        dst_entry_time: float,
    ) -> bool:
        """Check if transition is temporally feasible."""
        ...

    def filter_matches(
        self,
        query_camera: str,
        query_time: float,
        matches: list[GalleryMatch],
        gallery: EmbeddingGallery,
    ) -> list[GalleryMatch]:
        """Remove infeasible matches based on camera link constraints."""
        ...
```

**Key design decisions:**
- Links are directional: `(A→B, 10-30s)` doesn't imply `(B→A, 10-30s)`
- Default window `(5, 120)` seconds when no link is configured — permissive fallback
- Links can be loaded from YAML/JSON config
- No link between cameras = default window applies (graceful degradation)

### Step 4: Cross-Camera Tracker (`src/yowo/tracking/_cross_camera.py` — NEW FILE)

**Goal:** Orchestrate per-camera `ByteTracker` instances + gallery + link model to assign global IDs.

**Requirements:**
```python
@dataclass(frozen=True, slots=True)
class GlobalTrackedBox:
    """Extends TrackedBox with global cross-camera identity."""
    box: TrackedBox          # original single-camera tracked box
    camera_id: str
    global_id: int | None    # None if not yet matched cross-camera
    local_track_id: int      # per-camera track ID

class CrossCameraTracker:
    """Manages per-camera ByteTrackers and cross-camera identity assignment.

    Architecture:
        Camera 1 frames → ByteTracker[1] → local tracks → ReID embeddings ─┐
        Camera 2 frames → ByteTracker[2] → local tracks → ReID embeddings ─┤
        Camera N frames → ByteTracker[N] → local tracks → ReID embeddings ─┤
                                                                            ↓
                                                            EmbeddingGallery
                                                            CameraLinkModel
                                                                            ↓
                                                            Global ID Assignment
    """
    __slots__ = (
        "_trackers", "_gallery", "_link_model", "_reid",
        "_next_global_id", "_local_to_global", "_lock",
    )

    def __init__(
        self,
        reid_extractor: ReIDExtractor,
        camera_link_model: CameraLinkModel | None = None,
        gallery_max_entries: int = 10_000,
        match_threshold: float = 0.4,
        **tracker_kwargs,
    ) -> None:
        """
        Args:
            reid_extractor: Shared ReID model (VehicleReIDExtractor or CLIPExtractor).
            camera_link_model: Optional spatial-temporal constraints.
            gallery_max_entries: Max embeddings in gallery.
            match_threshold: Cosine distance threshold for cross-camera match.
            **tracker_kwargs: Passed to each per-camera ByteTracker.
        """
        ...

    def update(
        self,
        camera_id: str,
        frame: NDArray[np.uint8],
        detections: list[Detection],
        timestamp: float,
    ) -> list[GlobalTrackedBox]:
        """Process one frame from one camera.

        1. Run ByteTracker.update() for this camera
        2. Extract ReID embeddings for confirmed tracks
        3. For new/unmatched tracks: query gallery for cross-camera match
        4. For expired tracks: add to gallery for future matching
        5. Return boxes with global IDs
        """
        ...

    def register_camera(self, camera_id: str) -> None:
        """Create a new ByteTracker for this camera."""
        ...

    def finalize_track(self, camera_id: str, track: STrack) -> None:
        """Track expired — add its embedding to the gallery."""
        ...

    @property
    def gallery_size(self) -> int: ...

    @property
    def camera_count(self) -> int: ...
```

**Global ID assignment logic:**
1. When a track first becomes confirmed (`is_confirmed=True`), extract its embedding
2. Query gallery with `exclude_camera=this_camera`
3. If best match distance < `match_threshold` AND `link_model.is_feasible()`:
   - Assign same `global_id` as matched gallery entry
   - This is the cross-camera re-identification!
4. If no match: assign new `global_id`
5. When track expires (`mark_removed()` or exceeds `max_age`): add embedding to gallery

**Thread safety:**
- `CrossCameraTracker.update()` is called from different threads (one per camera in FrameCollector)
- `_local_to_global` dict and gallery must be thread-safe
- Each `ByteTracker` is camera-local, no cross-thread access

### Step 5: Integration with Pipeline (`src/yowo/pipeline/`)

**Goal:** Wire `CrossCameraTracker` into the existing multi-stream pipeline.

**Requirements:**
- Add `track_pipeline()` function or extend `run_pipeline()` with optional `CrossCameraTracker`
- `DetectionRouter.route()` returns per-camera detections → feed to `CrossCameraTracker.update()`
- Output: `GlobalTrackedFrame` with `camera_id` + `global_id` on each box
- Compose with existing `FrameCollector` + `BatchScheduler` + `DetectionRouter`

**Architecture:**
```
FrameCollector (N cameras)
    → BatchScheduler (batched frames)
    → InferenceEngine.detect() (batched detections)
    → DetectionRouter (per-camera detections)
    → CrossCameraTracker.update(camera_id, frame, dets, timestamp)
    → GlobalTrackedFrame (boxes with global IDs)
```

### Step 6: Unit Tests

**Required tests (minimum):**

**TestVehicleReIDExtractor:**
- Initialization with ONNX model path
- `embedding_dim` returns 512
- `extract()` returns correct shape (N, 512)
- Filtered crops → zero rows
- Empty boxes → None
- Protocol compliance: `isinstance(extractor, ReIDExtractor)`

**TestEmbeddingGallery:**
- Add and query — nearest neighbor returns correct match
- `exclude_camera` prevents same-camera matches
- `threshold` filters distant matches
- `max_entries` eviction works (FIFO)
- Thread safety: concurrent add/query
- Empty gallery → empty results
- Multiple cameras with same global_id

**TestCameraLinkModel:**
- Feasible transition within window
- Infeasible transition outside window
- Default window for unconfigured camera pairs
- `filter_matches` removes infeasible entries
- Bidirectional vs unidirectional links

**TestCrossCameraTracker:**
- Single camera: behaves like ByteTracker + global IDs
- Two cameras: vehicle exits camera A → appears in camera B → same global_id
- Same camera: no cross-camera matching (local track IDs only)
- Gallery grows as tracks expire
- Thread safety: concurrent updates from multiple cameras
- Graceful degradation: no link model → uses default window

### Step 7: Quality Gates

```bash
uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q
```

All must pass: 0 lint errors, 0 pyright errors, all tests pass.

Verify file sizes:
- `_reid.py` — check < 700 lines after adding VehicleReIDExtractor
- `_gallery.py` — new file, should be ~150-200 lines
- `_camera_link.py` — new file, should be ~100-150 lines
- `_cross_camera.py` — new file, should be ~200-300 lines

## Chain-of-Thought Guidance

For each step:
- Read the existing code before writing — understand patterns, imports, naming conventions
- Use `__slots__` on all classes (yowo convention)
- Use `ReIDExtractor` Protocol — don't couple to specific extractor
- Embeddings are always L2-normalized, cosine distance = `1 - dot(a, b)`
- Thread safety matters: FrameCollector spawns one thread per camera
- No FAISS dependency — numpy matmul for cosine similarity is fast enough for 10K gallery
- Keep `ByteTracker` unchanged — compose from outside, don't modify internal tracking logic

## Reference: State of the Art

### What top MTMC vehicle trackers do (CityFlow benchmark, 2024-2025):

| Method | Key Innovation | IDF1 |
|--------|---------------|------|
| DGT (Dynamic Global Tracker) | Online cross-camera association (no post-hoc clustering) | — |
| Score-based Matching | Halved runtime vs AI City 2022 winner | 83.4% |
| MCBLT | 3D BEV tracking + hierarchical GNN | SOTA (AICity'24) |
| SAE-MCVT | First scalable real-time edge MCVT | — |

### yowo's approach vs these:
- yowo adopts DGT's online approach (no full-trajectory clustering)
- Gallery-based matching is simpler than GNN but production-deployable
- Camera link model provides the spatial-temporal pruning that Score-based Matching uses
- No 3D/BEV — stays 2D for simplicity and camera-agnostic deployment

## Self-Evaluation Framework

After implementation, rate confidence (0-1) on:

1. **Completeness**: All 7 steps done, all required classes/functions implemented
2. **Clarity**: Code is self-documenting, follows yowo conventions
3. **Practicality**: Works with existing pipeline, no breaking changes
4. **Optimization**: Gallery query is O(N) cosine similarity — acceptable for N ≤ 10K
5. **Edge Cases**: Empty gallery, single camera, no link model, concurrent access
6. **Self-Evaluation**: Tests cover happy path + failure modes + thread safety

Provide a score for each (0-1).
If any score < 0.9, refine your answer before presenting.
