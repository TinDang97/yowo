# Codebase Structure

**Analysis Date:** 2026-03-07

## Directory Layout

```
yowo/
├── src/yowo/                # Main package source
│   ├── __init__.py          # Public API re-exports (~185 lines)
│   ├── _async.py            # Async streaming bridge
│   ├── _convenience.py      # One-call detect()/classify() functions
│   ├── _streaming.py        # StreamingMixin with 4 streaming strategies
│   ├── classify_engine.py   # ClassificationEngine(BaseEngine)
│   ├── config.py            # InferenceConfig, ClassificationConfig, ExportConfig, presets
│   ├── engine.py            # BaseEngine + DetectionEngine + InferenceEngine alias
│   ├── errors.py            # YowoError hierarchy (14 exception types)
│   ├── types.py             # Core enums, dataclasses, constants
│   ├── py.typed             # PEP 561 marker
│   ├── README.md            # Module-level docs
│   ├── arch/                # Native YOLO model architecture (no ultralytics)
│   ├── backends/            # Inference backend Protocol + 5 implementations
│   ├── cache/               # Feature map caching (mmap-backed)
│   ├── cli/                 # Click-based CLI (detect, classify, track, count, export, info, models)
│   ├── counter/             # Line-crossing + zone object counting
│   ├── events/              # Publish-subscribe EventBus
│   ├── export/              # Model export (ONNX, TensorRT, OpenVINO, CoreML)
│   ├── hardware/            # CPU/GPU detection, capability probing
│   ├── io/                  # Frame sources, preprocessing, output sinks
│   ├── metrics/             # Engine telemetry (latency, counts, errors)
│   ├── models/              # Model registry + weight resolution/download
│   ├── pipeline/            # Multi-stream pipeline (collector, scheduler, router)
│   ├── postprocess/         # NMS (detection) + softmax/top-k (classification)
│   ├── tracking/            # ByteTrack, ReID, cross-camera, embedding galleries
│   └── utils/               # Drawing helpers, factory utilities
├── tests/
│   ├── unit/                # ~60 unit test files, 1615+ tests
│   └── integration/         # Integration tests (requires GPU/models)
├── bench/                   # Benchmark scripts
├── docs/                    # Documentation
├── examples/                # Usage examples
├── tmp/                     # Tooling scripts, weights, experiments (not committed)
├── pyproject.toml           # Project config (uv, ruff, pyright, pytest)
├── uv.lock                  # Lock file
├── CHANGELOG.md             # Release history
├── CONTRIBUTING.md          # Contribution guidelines
├── CONTEXT.md               # Project context
├── CLAUDE.md                # AI assistant instructions
├── LICENSE                  # License file
└── README.md                # Project readme
```

## Directory Purposes

**`src/yowo/arch/`:**
- Purpose: Native YOLO11 + YOLO26 model construction without ultralytics
- Contains: PyTorch `nn.Module` building blocks, model configs, weight loading
- Key files: `_yolo.py` (YOLOModel, ClassifyModel), `_blocks.py` (Conv, C2f, C3k2, SPPF), `_attention.py` (C2PSA, PSABlock, Attention), `_neck.py` (neck construction), `_heads.py` (Detect, Classify), `_config.py` (model scaling), `_weights.py` (checkpoint loading)

**`src/yowo/backends/`:**
- Purpose: Inference runtime abstraction with lazy SDK imports
- Contains: Protocol definitions, backend selector, 5 backend implementations
- Key files: `__init__.py` (InferenceBackend/ModelBuilder Protocols, create_backend factory), `_selector.py` (select_backend, get_fallback_backends), `_pytorch.py`, `_onnx.py`, `_tensorrt.py`, `_openvino.py`, `_coreml.py`

**`src/yowo/io/`:**
- Purpose: Frame input/output — source dispatch, preprocessing, writing results
- Contains: Source protocol + factory, letterbox preprocessing with buffer pooling, threaded reader, JSON/image output
- Key files: `_source.py` (FrameSource protocol, open_source factory), `_decode.py` (preprocess, preprocess_into, PreprocessBuffer, PreprocessBufferPool), `_reader.py` (ThreadedFrameReader, PreparedItem), `_sink.py` (write_json, write_annotated_frames)

**`src/yowo/tracking/`:**
- Purpose: Multi-object tracking and cross-camera re-identification
- Contains: ByteTrack tracker, Kalman filter, matching, ReID extractors, embedding galleries
- Key files: `_tracker.py` (ByteTracker), `_strack.py` (TrackedBox, TrackedDetection, TrackState), `_kalman.py`, `_matching.py`, `_reid.py` (ReIDExtractor Protocol + CLIPExtractor, FastReIDExtractor, VehicleReIDExtractor), `_clip_reid.py` (CLIPReIDExtractor for VeRi), `_gallery.py` (EmbeddingGallery), `_chroma_gallery.py` (ChromaEmbeddingGallery), `_cross_camera.py` (CrossCameraTracker), `_camera_link.py` (CameraLinkModel)

**`src/yowo/pipeline/`:**
- Purpose: Multi-stream inference through a single engine
- Contains: Stream collection, batch scheduling, result routing, pipeline orchestration
- Key files: `_collector.py` (FrameCollector), `_scheduler.py` (BatchScheduler), `_router.py` (DetectionRouter), `__init__.py` (run_pipeline)

**`src/yowo/export/`:**
- Purpose: Model format conversion (PyTorch -> ONNX/TensorRT/OpenVINO/CoreML)
- Contains: Export orchestrator, INT8 quantization, calibration data generation, metadata sidecars
- Key files: `_exporter.py` (export_model), `_int8.py` (quantize_onnx_static), `_calibration.py` (TensorRTCalibrator, calibration_batches), `_kv_wrapper.py`, `_metadata.py` (ExportMetadata)

**`src/yowo/hardware/`:**
- Purpose: System hardware detection, cached as process-level singleton
- Contains: GPU enumeration, CPU detection, installed SDK probing
- Key files: `__init__.py` (HardwareProfile, get_hardware_profile), `_detect.py` (detect_gpus, detect_cpu_device), `_capabilities.py` (InstalledLibraries, detect_libraries), `_device.py` (Device dataclass)

**`src/yowo/counter/`:**
- Purpose: Object counting via line-crossing and zone containment
- Key files: `_counter.py` (ObjectCounter), `_geometry.py` (line/zone geometry), `_types.py` (CountLine, CountZone, CountResult, LineCrossEvent, CrossDirection)

**`src/yowo/cli/`:**
- Purpose: Command-line interface
- Key files: `_main.py` (Click group: detect, classify, track, count, info, models, export)

**`src/yowo/models/`:**
- Purpose: Model metadata registry and weight file resolution
- Key files: `_registry.py` (model registry: family+size -> ModelMeta), `_weights.py` (resolve_weights, download)

**`src/yowo/postprocess/`:**
- Purpose: Raw tensor output -> structured results
- Key files: `_nms.py` (NMS + box decoding for detection), `_classify.py` (softmax + top-k for classification)

**`src/yowo/metrics/`:**
- Purpose: Engine performance telemetry
- Key files: `_collector.py` (MetricsCollector), `__init__.py` (EngineMetrics snapshot dataclass)

**`src/yowo/events/`:**
- Purpose: Lightweight pub-sub event system
- Key files: `__init__.py` (EventBus with sync + async callback support)

**`src/yowo/cache/`:**
- Purpose: Feature map caching for repeated inference on similar frames
- Key files: `_store.py` (FeatureCache), `_similarity.py` (frame similarity computation)

**`src/yowo/utils/`:**
- Purpose: Shared helpers
- Key files: `_draw.py` (bounding box / label drawing on images), `_factory.py` (factory helpers)

## Key File Locations

**Entry Points:**
- `src/yowo/__init__.py`: Public API surface — all user-facing imports
- `src/yowo/cli/_main.py`: CLI entry point (`yowo` command)
- `src/yowo/_convenience.py`: One-call `detect()` and `classify()` functions

**Configuration:**
- `src/yowo/config.py`: All config dataclasses + YAML loader + env var overrides + presets
- `pyproject.toml`: Project metadata, dependencies, tool config (ruff, pyright, pytest)

**Core Logic:**
- `src/yowo/engine.py`: BaseEngine + DetectionEngine (main inference orchestrator)
- `src/yowo/classify_engine.py`: ClassificationEngine
- `src/yowo/_streaming.py`: StreamingMixin (4 streaming strategies)
- `src/yowo/_async.py`: Async streaming bridge

**Types & Errors:**
- `src/yowo/types.py`: All core data types, enums, constants
- `src/yowo/errors.py`: Exception hierarchy (14 types)

**Testing:**
- `tests/unit/`: ~60 test files, co-located by module
- `tests/integration/`: Integration tests requiring hardware/models

## Naming Conventions

**Files:**
- Private modules: `_name.py` (underscore prefix) — e.g., `_streaming.py`, `_nms.py`, `_pytorch.py`
- Public modules: No underscore — e.g., `engine.py`, `config.py`, `types.py`, `errors.py`
- Test files: `test_{module}.py` — e.g., `test_engine.py`, `test_nms.py`
- Package init: `__init__.py` re-exports the public API for each subpackage

**Directories:**
- Lowercase, singular: `arch/`, `cache/`, `counter/`, `export/`
- Plural for collections: `backends/`, `models/`, `metrics/`, `events/`, `utils/`

**Classes:**
- PascalCase: `DetectionEngine`, `ByteTracker`, `FrameCollector`, `PreprocessBuffer`
- Protocols: PascalCase same as classes: `InferenceBackend`, `FrameSource`, `ReIDExtractor`, `GalleryProtocol`

**Functions:**
- snake_case: `detect()`, `preprocess_into()`, `run_pipeline()`, `track_stream()`
- Private: underscore prefix: `_stream_dispatch()`, `_run_gpu()`, `_resolve_model_meta()`

**Constants:**
- UPPER_SNAKE_CASE: `IMAGE_EXTS`, `VIDEO_EXTS`, `RTSP_SCHEMES`, `EVENT_CLASSIFICATION`

## Where to Add New Code

**New Inference Backend:**
- Create `src/yowo/backends/_newbackend.py` implementing `InferenceBackend` Protocol
- Add `BackendType.NEWBACKEND` enum value in `src/yowo/types.py`
- Add case to `create_backend()` match in `src/yowo/backends/__init__.py`
- Add fallback entry in `src/yowo/backends/_selector.py`
- Tests: `tests/unit/test_newbackend.py`

**New Model Family:**
- Add `ModelFamily.NEW` enum in `src/yowo/types.py`
- Add config in `src/yowo/arch/_config.py`
- Add model construction in `src/yowo/arch/_yolo.py`
- Register in `src/yowo/models/_registry.py`
- Add to `_FAMILY_MAP` in `src/yowo/_convenience.py`
- Tests: `tests/unit/test_arch_model.py`, `tests/unit/test_registry.py`

**New Engine Task (e.g., segmentation):**
- Create `src/yowo/segment_engine.py` extending `BaseEngine`
- Implement `_process_batch()` for task-specific postprocessing
- Add result type in `src/yowo/types.py`
- Add postprocessor in `src/yowo/postprocess/_segment.py`
- Add config dataclass in `src/yowo/config.py`
- Add convenience function in `src/yowo/_convenience.py`
- Add CLI command in `src/yowo/cli/_main.py`
- Re-export in `src/yowo/__init__.py`
- Tests: `tests/unit/test_segment_engine.py`

**New Tracking Feature:**
- Add implementation in `src/yowo/tracking/_newfeature.py`
- Re-export in `src/yowo/tracking/__init__.py`
- Tests: `tests/unit/test_tracking_newfeature.py`

**New CLI Command:**
- Add `@cli.command("name")` in `src/yowo/cli/_main.py`
- Tests: `tests/unit/test_cli.py`

**Utilities:**
- Shared helpers: `src/yowo/utils/_newutil.py`
- Re-export in `src/yowo/utils/__init__.py`

## Special Directories

**`tmp/`:**
- Purpose: Weights, benchmark scripts, experiment scripts, comparison tools
- Generated: Partially (weights downloaded on demand)
- Committed: No (gitignored)

**`bench/`:**
- Purpose: Benchmark scripts for performance testing
- Generated: No
- Committed: Yes

**`docs/`:**
- Purpose: Project documentation
- Generated: No
- Committed: Yes

**`examples/`:**
- Purpose: Usage examples
- Generated: No
- Committed: Yes

**`dist/`:**
- Purpose: Built wheel/sdist artifacts
- Generated: Yes
- Committed: No (gitignored)

**`.planning/`:**
- Purpose: GSD planning artifacts (codebase analysis, phase plans)
- Generated: Yes (by tooling)
- Committed: Yes

---

*Structure analysis: 2026-03-07*
