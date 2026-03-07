# Architecture

**Analysis Date:** 2026-03-07

## Pattern Overview

**Overall:** Layered pipeline architecture with Protocol-driven abstractions

**Key Characteristics:**
- Strict layer separation: types -> arch -> backends -> io -> engine -> CLI/convenience
- Protocol-based polymorphism (structural typing) for backends and ReID extractors
- Mixin composition for streaming strategies (`StreamingMixin` mixed into `BaseEngine`)
- Lazy imports throughout to avoid pulling heavy SDKs (PyTorch, ONNX, TensorRT) at module load
- Process-level singleton for hardware detection with thread-safe caching
- Event-driven observability via `EventBus` (sync + async callbacks)

## Layers

**Types Layer:**
- Purpose: Core data primitives shared across all module boundaries
- Location: `src/yowo/types.py`
- Contains: Enums (`BackendType`, `ModelFamily`, `ModelSize`, `Precision`, `FrameDropPolicy`), frozen dataclasses (`BoundingBox`, `Detection`, `Frame`, `PreprocessedTensor`, `ModelSpec`, `ClassificationResult`), media constants
- Depends on: numpy only
- Used by: Every other module

**Architecture Layer:**
- Purpose: Native YOLO11/YOLO26 model construction (no ultralytics dependency)
- Location: `src/yowo/arch/`
- Contains: Model building blocks (`_blocks.py`), attention (`_attention.py`), neck (`_neck.py`), detection/classification heads (`_heads.py`), model assembly (`_yolo.py`), config scaling (`_config.py`), weight loading (`_weights.py`)
- Depends on: PyTorch, types
- Used by: `backends/_pytorch.py`

**Hardware Layer:**
- Purpose: Detect CPU/GPU capabilities, installed SDK versions; cached singleton
- Location: `src/yowo/hardware/`
- Contains: GPU detection (`_detect.py`), device descriptors (`_device.py`), library version probing (`_capabilities.py`)
- Depends on: types
- Used by: backends (selector), engine

**Backend Layer:**
- Purpose: Abstract inference runtime behind a Protocol; factory with lazy imports
- Location: `src/yowo/backends/`
- Contains: `InferenceBackend` Protocol + `ModelBuilder` Protocol (`__init__.py`), backend selector (`_selector.py`), implementations: PyTorch (`_pytorch.py`), ONNX (`_onnx.py`), TensorRT (`_tensorrt.py`), OpenVINO (`_openvino.py`), CoreML (`_coreml.py`), ORTValue helpers (`_ortvalue.py`)
- Depends on: hardware, arch (PyTorch backend only), types
- Used by: engine

**IO Layer:**
- Purpose: Frame source abstraction, preprocessing (letterbox + normalize), output sinks
- Location: `src/yowo/io/`
- Contains: `FrameSource` Protocol + `open_source()` factory (`_source.py`), preprocessing with zero-copy buffers (`_decode.py`), threaded frame reader (`_reader.py`), JSON/image output (`_sink.py`)
- Depends on: types, numpy, OpenCV
- Used by: engine, pipeline, CLI

**Postprocessing Layer:**
- Purpose: Transform raw model output into structured results
- Location: `src/yowo/postprocess/`
- Contains: NMS for detection (`_nms.py`), softmax + top-k for classification (`_classify.py`)
- Depends on: types, numpy
- Used by: engine (DetectionEngine, ClassificationEngine)

**Engine Layer:**
- Purpose: Orchestrate the full inference lifecycle: hardware detect -> backend select -> load -> preprocess -> infer -> postprocess -> emit events
- Location: `src/yowo/engine.py`, `src/yowo/classify_engine.py`, `src/yowo/_streaming.py`, `src/yowo/_async.py`
- Contains: `BaseEngine(StreamingMixin)` abstract base, `DetectionEngine(BaseEngine)`, `ClassificationEngine(BaseEngine)`, streaming strategies (single/live/pipeline/sync), async bridge
- Depends on: backends, io, postprocess, models, metrics, events, hardware, cache, types
- Used by: CLI, convenience API, pipeline, tracking

**Model Registry:**
- Purpose: Map (family, size) -> metadata (input shape, num_classes, weight URLs)
- Location: `src/yowo/models/`
- Contains: Registry lookup (`_registry.py`), weight resolution + download (`_weights.py`)
- Depends on: types
- Used by: engine

**Config Layer:**
- Purpose: Dataclass configs with YAML + env var override chain
- Location: `src/yowo/config.py`
- Contains: `InferenceConfig`, `ClassificationConfig`, `ExportConfig`, `load_config()`, `preset_config()`, device/source classifier helpers
- Depends on: types, hardware (for presets)
- Used by: engine, CLI

**Tracking Layer:**
- Purpose: Multi-object tracking (ByteTrack), cross-camera ReID, embedding galleries
- Location: `src/yowo/tracking/`
- Contains: ByteTracker (`_tracker.py`), Kalman filter (`_kalman.py`), Hungarian matching (`_matching.py`), track state (`_strack.py`), ReID extractors Protocol + implementations (`_reid.py`, `_clip_reid.py`), embedding gallery (`_gallery.py`, `_chroma_gallery.py`), cross-camera tracker (`_cross_camera.py`), camera link model (`_camera_link.py`)
- Depends on: types, numpy, scipy
- Used by: convenience functions `track_stream()`, `track_detections()`

**Pipeline Layer:**
- Purpose: Multi-stream inference through a single engine
- Location: `src/yowo/pipeline/`
- Contains: `FrameCollector` (`_collector.py`), `BatchScheduler` (`_scheduler.py`), `DetectionRouter` (`_router.py`), `run_pipeline()` wiring function
- Depends on: engine, types
- Used by: user code

**Counter Layer:**
- Purpose: Object counting with line-crossing and zone counting
- Location: `src/yowo/counter/`
- Contains: `ObjectCounter` (`_counter.py`), geometry helpers (`_geometry.py`), types (`_types.py`)
- Depends on: tracking types, types
- Used by: CLI `count` command, user code

**Export Layer:**
- Purpose: Convert PyTorch models to ONNX/TensorRT/OpenVINO/CoreML
- Location: `src/yowo/export/`
- Contains: Export orchestrator (`_exporter.py`), INT8 quantization (`_int8.py`), calibration (`_calibration.py`), KV wrapper for export (`_kv_wrapper.py`), metadata sidecar (`_metadata.py`)
- Depends on: arch, models, types, torch
- Used by: CLI `export` command

**Metrics Layer:**
- Purpose: Engine telemetry (inference latency, frame counts, error counts)
- Location: `src/yowo/metrics/`
- Contains: `MetricsCollector` (`_collector.py`), `EngineMetrics` snapshot dataclass
- Depends on: types
- Used by: engine

**Events Layer:**
- Purpose: Publish-subscribe event bus for engine observability
- Location: `src/yowo/events/`
- Contains: `EventBus` with sync/async callback support
- Depends on: nothing (standalone)
- Used by: engine

**Cache Layer:**
- Purpose: Feature map caching for repeated inference on similar frames
- Location: `src/yowo/cache/`
- Contains: `FeatureCache` (`_store.py`), similarity computation (`_similarity.py`)
- Depends on: numpy
- Used by: engine (via backends)

**Utils Layer:**
- Purpose: Drawing utilities, factory helpers
- Location: `src/yowo/utils/`
- Contains: Bounding box drawing (`_draw.py`), factory helpers (`_factory.py`)
- Depends on: types, OpenCV
- Used by: CLI, user code

**CLI Layer:**
- Purpose: Click-based CLI with commands: detect, classify, track, count, info, models, export
- Location: `src/yowo/cli/`
- Contains: Click group + commands (`_main.py`)
- Depends on: engine, config, io, tracking, counter, export, types
- Used by: end users via `yowo` command

**Convenience Layer:**
- Purpose: One-call inference functions (`detect()`, `classify()`, `parse_model_name()`)
- Location: `src/yowo/_convenience.py`
- Contains: `detect()`, `classify()`, `parse_model_name()`
- Depends on: engine, classify_engine, io, types (all via late imports)
- Used by: end users via `from yowo import detect`

## Data Flow

**Single-Frame Detection:**

1. User calls `detect("image.jpg")` or `engine.detect([frame])`
2. `open_source()` dispatches to appropriate `FrameSource` (image/video/RTSP/webcam)
3. `FrameSource` yields `Frame(pixels=ndarray, source_id, frame_index)`
4. `preprocess()` or `preprocess_into()` converts `list[Frame]` -> `PreprocessedTensor` (BCHW float32, letterboxed, normalized)
5. `backend.infer(tensor)` -> raw `NDArray[np.float32]` output
6. `postprocess()` applies NMS -> `list[Detection]` with `BoundingBox` objects
7. `EventBus.emit("detection", results)` notifies subscribers
8. Results returned to caller

**Streaming Detection:**

1. `engine.stream(source)` calls `_stream_dispatch(source)`
2. Dispatch selects strategy: `_stream_single` (1 frame), `_stream_live` (RTSP/webcam), `_stream_pipeline` (video file), `_stream_sync` (no prefetch)
3. Live path: `ThreadedFrameReader` decodes + preprocesses on background thread, engine consumes from bounded queue
4. Pipeline path: `ThreadPoolExecutor` overlaps preprocessing with inference; `infer_lock` serializes GPU access while NMS runs in parallel
5. Each batch yields `Detection` objects to caller's iterator

**Multi-Stream Pipeline:**

1. `FrameCollector` manages N `ThreadedFrameReader` instances
2. `BatchScheduler` pulls `TaggedFrame` objects, accumulates into batches by timeout or size
3. `run_pipeline()` feeds batches to `engine.detect()`, optionally overlapping assembly with inference
4. `DetectionRouter` routes `Detection[i]` to the callback for `batch[i].source_id` (positional index routing)

**State Management:**
- Engine state: `_loaded`, `_health_state`, `_shutting_down` (threading.Event), `_active_streams`
- Metrics: `MetricsCollector` tracks inference latency, frame counts, errors
- Tracking state: `ByteTracker` maintains per-track Kalman filters, ReID embeddings
- Hardware profile: Process-level singleton (`_cached` in `hardware/__init__.py`)

## Key Abstractions

**InferenceBackend Protocol:**
- Purpose: Decouple engine from specific inference runtime
- Examples: `src/yowo/backends/_pytorch.py`, `src/yowo/backends/_onnx.py`, `src/yowo/backends/_tensorrt.py`, `src/yowo/backends/_openvino.py`, `src/yowo/backends/_coreml.py`
- Pattern: Structural typing via `@runtime_checkable Protocol`. No base class required. Methods: `load()`, `infer()`, `unload()`, `warmup()`, `clear_kv_cache()`, `set_source_id()`

**ModelBuilder Protocol:**
- Purpose: Allow custom (non-YOLO) architectures to use PyTorchBackend's device management
- Examples: `src/yowo/backends/__init__.py` (definition)
- Pattern: Protocol with `build(num_classes, device) -> Module` and `input_shape` property

**FrameSource Protocol:**
- Purpose: Unified interface for all input sources (image, video, RTSP, webcam, directory)
- Examples: `src/yowo/io/_source.py`
- Pattern: Protocol with `__iter__`, `close()`, `is_live`, `total_frames`, `resolution`, `fps`

**ReIDExtractor Protocol:**
- Purpose: Pluggable appearance feature extraction for tracking
- Examples: `src/yowo/tracking/_reid.py` (CLIPExtractor, FastReIDExtractor, VehicleReIDExtractor), `src/yowo/tracking/_clip_reid.py` (CLIPReIDExtractor)
- Pattern: Protocol with `extract(image, boxes) -> ndarray`

**GalleryProtocol:**
- Purpose: Pluggable embedding storage for cross-camera ReID
- Examples: `src/yowo/tracking/_gallery.py` (EmbeddingGallery), `src/yowo/tracking/_chroma_gallery.py` (ChromaEmbeddingGallery)
- Pattern: Structural typing for `add()`, `query()`, `remove()`, `clear()`

**BaseEngine ABC:**
- Purpose: Shared lifecycle, streaming, metrics for detection and classification engines
- Examples: `src/yowo/engine.py` (DetectionEngine), `src/yowo/classify_engine.py` (ClassificationEngine)
- Pattern: Abstract base with `_process_batch()` hook; `StreamingMixin` provides streaming strategies

## Entry Points

**CLI:**
- Location: `src/yowo/cli/_main.py`
- Triggers: `yowo detect|classify|track|count|info|models|export` commands
- Responsibilities: Parse args, build config, create engine, run inference, write output

**Convenience API:**
- Location: `src/yowo/_convenience.py`
- Triggers: `from yowo import detect, classify`
- Responsibilities: One-call inference with sensible defaults; creates + tears down engine internally

**Engine API:**
- Location: `src/yowo/engine.py` (`DetectionEngine`), `src/yowo/classify_engine.py` (`ClassificationEngine`)
- Triggers: User constructs engine with config, uses as context manager
- Responsibilities: Full lifecycle control: load/detect/stream/close

**Pipeline API:**
- Location: `src/yowo/pipeline/__init__.py` (`run_pipeline()`)
- Triggers: Multi-stream use case; user composes collector + scheduler + router
- Responsibilities: Wire N streams through single engine with batch scheduling

**Export API:**
- Location: `src/yowo/export/__init__.py` (`export_model()`)
- Triggers: `yowo export` CLI or direct call
- Responsibilities: Convert PyTorch model to ONNX/TensorRT/OpenVINO/CoreML

## Error Handling

**Strategy:** Typed exception hierarchy rooted at `YowoError`; each module raises only its own subtype

**Hierarchy (defined in `src/yowo/errors.py`):**
```
YowoError
├── DependencyError         # Missing optional package
├── BackendError            # Backend init/inference failure
│   ├── BackendLoadError    # Failed to load model into backend
│   ├── InferenceError      # Runtime inference failure
│   └── ShutdownError       # Engine is shutting down
├── DeviceError             # Device not found / OOM
├── ModelError
│   ├── ModelNotFoundError  # .pt or exported file not found
│   └── ModelLoadError      # File found but corrupt/incompatible
├── ExportError             # Export operation failed
│   └── ExportUnsupportedError
├── SourceError             # Cannot open input source
│   └── SourceTimeoutError
├── ConfigError             # Invalid configuration
└── TrackingError           # Object tracking failure
```

**Patterns:**
- Backend fallback chain in `engine.py` `load()`: tries primary backend, falls back to alternatives on `BackendLoadError`/`BackendError`
- Graceful shutdown via `threading.Event` (`_shutting_down`); operations raise `ShutdownError` when engine is closing
- Error threshold triggers `HealthStatus.DEGRADED` (configurable via `error_threshold`)
- `_metrics.record_error()` called in `_run_batch()` catch block for telemetry

## Cross-Cutting Concerns

**Logging:** Standard `logging` module; each module creates `logger = logging.getLogger(__name__)`

**Validation:** Dataclass `__post_init__` validation in `config.py` (range checks on confidence, iou, batch_size, etc.)

**Authentication:** Not applicable (local inference library, no auth)

**Concurrency:** Thread-safe shutdown protocol (lock + Event), `infer_lock` for GPU serialization in pipeline mode, `ThreadedFrameReader` with bounded deque for frame prefetch, `ThreadPoolExecutor` for pipeline overlap

---

*Architecture analysis: 2026-03-07*
