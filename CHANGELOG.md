# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Releases after `v0.1.0` are generated automatically by
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
from [Conventional Commits](https://www.conventionalcommits.org/).

---

## [Unreleased]

### Security

- **weights**: No checkpoint is deserialized before its pinned SHA-256 is
  compared. `ModelMeta` carries a `sha256` for all 10 detection variants;
  `resolve_weights` verifies on first download and on every cache hit, and the
  verified digest is threaded into the loader so a file swapped between resolve
  and load is caught before conversion. Both the inference path
  (`PyTorchBackend`) and the export path (`export_model`) supply the pin.

- **weights**: The checkpoint unpickler admits by name, not by shape. The
  `torch.nn.modules.*` prefix rule and the `*Storage` suffix rule are both gone,
  replaced by an explicit set of 11 measured `(module, class)` pairs.
  `scripts/checkpoint_globals_manifest.json` records the 53 globals observed
  across the pinned checkpoints that justify the set.

- **weights**: All four entry points of the unpickler shim are restricted.
  `torch.load`'s legacy non-zip reader calls `pickle_module.load` on the file
  header three times *before* constructing an `Unpickler`, so a stock `load`
  there was an unrestricted read of attacker-controlled bytes — a checkpoint
  could execute code without `find_class` ever being consulted.

- **io**: RTSP credentials are redacted at the boundary where the URL is
  stored, so `Frame.source_id`, result payloads and feature-cache keys carry
  `redact_url`'s output rather than the raw URL. Userinfo is also scrubbed out
  of third-party exception text — `requests` keeps `user:pass@` in
  `PreparedRequest.url` even when it also sets an `Authorization` header, so
  redacting the URL alone was not enough. `yowo models` no longer echoes a
  registered credentialed weights URL.

  Not yet covered: `open_source()` still interpolates a credentialed non-RTSP
  camera URL verbatim into three `SourceError` messages
  (`io/_source.py:545`, `:550`, `:558`), because redaction happens inside
  `RTSPStreamSource` and that factory raises before any source object exists.

### Added

- **release**: Publish to PyPI over OIDC trusted publishing. No API token on the
  release path.
- **ci**: `ci.yml` runs on `pull_request` with four jobs — Quality Gate, Source
  Distribution, Reproducible Build, Weight Fixture — all four required on
  `main`. CI obtains a real weight, digest-verified before use, without
  committing it or fetching per job.
- **docs**: `NOTICE` and `SECURITY.md`. Model weights are AGPL-3.0 ultralytics
  artifacts and are documented as such; the package itself stays Apache-2.0 and
  ships no weights.

### Fixed

- **packaging**: The sdist is restricted to an explicit `only-include`
  allowlist. `2.4.0` and `2.4.1` shipped a 5.6 MB `yolo11n.pt` to PyPI inside an
  Apache-2.0 declaration; both are now yanked.
- **packaging**: Builds are reproducible — CI builds twice under a pinned
  `SOURCE_DATE_EPOCH` and compares digests.
- **version**: `yowo.__version__`, `yowo --version`, distribution metadata and
  export sidecars are single-sourced, with a check that fails the build on
  drift.
- **hardware**: A Jetson no longer reads as a machine with no GPU, and an old
  driver's rejected compute capability no longer hides the GPU.
- **engine**: `DetectionEngine` labels detections with COCO names for any model.
- **obb**: Importing `yowo` no longer drags `torch` in.

### Tests

- 2278 unit tests (up from 2084 in v2.4.1).

---

## [2.5.0] — 2026-03-25

No code changes.

`v2.5.0` differs from `v2.4.1` by a single line — the version string in
`pyproject.toml`. Every commit listed in the GitHub release note for this tag
is an ancestor of `v2.4.1` and shipped in that release; semantic-release
generated the note from a range that had already been published.

The tag was never published to PyPI: the release workflow had no publish step
at the time. `2.5.0` therefore exists only as a git tag and a GitHub release.

Recorded rather than omitted so the version sequence has no silent gap.

---

## [2.4.1] — 2026-03-17

### Added

- **compat**: Python 3.8 support for Jetson Nano — lowered `requires-python`
  from `>=3.9` to `>=3.8`. The codebase already used `from __future__ import
  annotations` in all source files, making annotation syntax Python 3.8
  compatible. Configuration changes: `ruff target-version = "py38"`, Python 3.8
  classifier, `tracking` extra gated to `python_version >= "3.9"` (scipy has no
  3.8 wheels).

### Fixed

- **compat**: Resolved 4 runtime Python 3.8 incompatibilities found during
  manual testing on Python 3.8.20:
  - `counter/_geometry.py`: `Point = tuple[float, float]` → `Tuple[float, float]`
    (runtime type alias, not protected by `from __future__`)
  - `events/__init__.py`: `_ListenerEntry = tuple[Callable[...], ...]` →
    `Tuple[...]` and `collections.abc.Callable` → `typing.Callable` (not
    subscriptable on 3.8)
  - `cache/_store.py`: `_Features = tuple[NDArray[...], ...]` → `Tuple[...]`
  - `batch/_runner.py`: parenthesized context manager `with (...):`  →
    backslash continuation (parenthesized form requires Python 3.10+)

### Tests

- 2084 unit tests (up from 1910 in v2.4.0). 174 new guard tests ensuring
  Python 3.8 compatibility: `from __future__ import annotations` presence,
  no runtime union syntax in `__init__.py`, no runtime lowercase generic type
  aliases (`tuple[...]`, `list[...]` etc.), StrEnum backport validation, and
  public API import smoke tests.

---

## [2.4.0] — 2026-03-12

### Added

- **compat**: Python 3.9 and 3.10 support — backported for ONNX GPU, TensorRT,
  and Jetson deployment. `StrEnumBase` shim for `enum.StrEnum` (3.11+),
  `match/case` replaced with `if/elif/else`, `slots=True` removed from
  dataclasses, `typing.Self` guarded under `TYPE_CHECKING`, `datetime.UTC`
  replaced with `datetime.timezone.utc`, `zip(strict=)` removed,
  `isinstance(X | Y)` replaced with tuple form. Dev deps gated with
  `python_version >= '3.11'` markers. `onnx-gpu` extra pinned `<1.20` for
  Python 3.9 (last version with cp39 wheels).

- **engine**: Custom `num_classes` support — override the registry default (80
  COCO / 1000 ImageNet) for fine-tuned models. Thread from user API
  (`InferenceEngine(num_classes=10)`, `ClassificationEngine(num_classes=10)`,
  `detect(..., num_classes=10)`, `classify(..., num_classes=10)`) down through
  `InferenceConfig` / `ClassificationConfig` → `ModelSpec` → `PyTorchBackend` →
  `build_model()` / `build_classify_model()`. Validation: `num_classes < 1`
  raises `ConfigError`. Env var: `YOWO_NUM_CLASSES`.

- **engine**: `ModelBuilder` protocol — plug custom (non-YOLO) architectures
  into `PyTorchBackend`'s optimization pipeline (fuse, eval, channels_last, KV
  cache, feature cache hooks). Runtime-checkable structural protocol with
  `build(num_classes, device) -> Module` and `input_shape -> (H, W)`. Builder
  owns architecture construction, weight loading, and device placement. When no
  matching registry entry exists, `_resolve_model_meta()` synthesizes a
  `ModelMeta` from the builder's `input_shape`. Non-PyTorch backends ignore the
  builder (they load serialized models).

- **cli**: `--num-classes` option on `yowo detect` and `yowo classify` commands.

### Refactored

- **engine**: Streaming strategy methods (`_stream_dispatch`, `_stream_single`,
  `_stream_live`, `_stream_pipeline`, `_stream_sync`) extracted from `engine.py`
  into `_streaming.py` as `StreamingMixin`. `BaseEngine` inherits from the mixin.
  Reduces `engine.py` from 790 to 620 lines (under the 700-line limit).

### Tests

- 1583 unit tests (up from 1543 in v2.2.2). 40 new tests covering
  `num_classes` threading (ModelSpec, InferenceConfig, ClassificationConfig,
  DetectionEngine, ClassificationEngine, PyTorchBackend, convenience API) and
  `ModelBuilder` protocol (structural check, create_backend, PyTorchBackend
  delegation, engine threading, `_resolve_model_meta` fallback).

### Performance

- **engine**: `auto_letterbox` — stride-aligned non-square input tensors instead
  of always padding to square. For 16:9 input (1280×720), produces a 384×640
  tensor (~40% fewer pixels), yielding **1.4–1.6× inference speedup** on
  PyTorch CPU/GPU. Benchmark: yolo26n 25.6 ms → 17.3 ms, yolo11n 23.8 ms →
  17.1 ms. E2E streaming now matches or exceeds ultralytics by 1–7%.
  Enabled via `InferenceConfig(auto_letterbox=True)`,
  `ClassificationConfig(auto_letterbox=True)`, env var `YOWO_AUTO_LETTERBOX`,
  and CLI `--auto-letterbox` on `detect` / `classify` / `track` / `count`.
  `ConfigError` raised on `load()` for non-PyTorch backends (fixed input shapes
  are incompatible with dynamic spatial dimensions).

- **engine**: `infer_lock` in `_stream_pipeline` now covers GPU inference only
  (`backend.infer`). NMS postprocessing runs outside the lock, allowing pipeline
  workers to postprocess concurrently while the next worker uses the GPU.

- **async**: Fire-and-forget async enqueue — replaced per-frame
  `asyncio.run_coroutine_threadsafe` with `loop.call_soon_threadsafe` +
  `_enqueue_or_drop`, eliminating one event-loop round-trip per frame.
  `QueueFull` is handled explicitly (no unhandled callback exception); silently
  dropped frames are recorded in `EngineMetrics`.

- **pipeline**: `FrameCollector` O(N) round-robin polling (50 ms per-stream
  timeout) replaced with a shared `queue.Queue` and per-stream bridge daemon
  threads. O(1) per-frame dispatch regardless of stream count.

### Added

- **metrics**: `EngineMetrics.frames_dropped` — counter for frames silently
  dropped when the async queue is full. `MetricsCollector.record_frame_dropped()`
  increments it; `reset()` zeroes it. `astream()` passes the callback into
  `_enqueue_or_drop`.

- **engine**: `BaseEngine._postprocess_and_emit()` — extracted shared helper
  that runs `_process_batch` then emits the result event. Eliminates duplication
  between `_infer_from_tensor` and `_stream_pipeline._infer_batch`.

### Fixed

- **pipeline**: `_run_bridge` exhaustion sentinel now uses a plain
  `queue.put(timeout=5.0)` instead of `_put_or_stop`. Previously,
  `_stop_entry` set `stop_event` before `bridge.join()`, causing
  `_put_or_stop` to return `False` immediately and drop the sentinel,
  stalling `__iter__` indefinitely.

- **pipeline**: `_stop_entry` now joins the bridge thread (`timeout=5.0`)
  before returning, ensuring clean shutdown ordering.

- **pipeline**: Bounded `shared_q` (`maxsize = max_queue_size × 8`) prevents
  unbounded memory growth under sustained consumer lag.

- **pipeline**: `_put_or_stop` helper — per-frame puts on the shared queue
  respect `stop_event`, exiting the bridge early on shutdown without blocking
  indefinitely on a full queue (0.5 s retry timeout).

- **pipeline**: Dead-bridge detection in `__iter__` — if a bridge thread exits
  without delivering its exhaustion sentinel (dropped after 5 s queue-full),
  the stream is retired automatically so iteration terminates naturally.

- **tracking**: `_munkres` infinite loop on partial-inf cost matrices — `inf −
  inf = NaN` during row reduction caused the algorithm to loop forever. Inf
  values are clamped to `1e9` before Munkres; the threshold gate rejects these
  matches.

- **pipeline**: `_check_stream_errors()` raises `RuntimeError` when all streams
  fail; logs a warning for partial failures (previously errors were silently
  swallowed).

### Refactored

- **pipeline**: `_StreamEntry.stop_flag: bool` replaced with
  `stop_event: threading.Event` for correct cross-thread visibility in bridge
  threads.

- **streaming**: `functools.partial(preprocess, auto_letterbox=auto_lb)` used
  unconditionally (removed asymmetric conditional branch).

### Tests

- 1910 unit tests (up from 1615). New tests: `TestAutoLetterbox` (12 cases
  covering stride alignment, non-square shapes, pixel range, batch consistency,
  mixed-aspect metadata, inverse transform), `TestAlignToStride` (3 cases),
  `record_frame_dropped` metrics (3 cases), `TestAutoLetterboxValidation`
  (ConfigError enforcement + double-unload guard), streaming concurrency (5
  cases), async drop handling (3 cases), collector shared-queue (4 cases +
  4 modified), pipeline error surfacing (4 cases).

---

## [2.2.3] — 2026-03-07

### Performance

- **engine**: `auto_letterbox` — stride-aligned non-square input tensors instead
  of always padding to square. For 16:9 input (1280×720), produces a 384×640
  tensor (~40% fewer pixels), yielding **1.4–1.6× inference speedup** on
  PyTorch CPU/GPU. Benchmark: yolo26n 25.6 ms → 17.3 ms, yolo11n 23.8 ms →
  17.1 ms. E2E streaming now matches or exceeds ultralytics by 1–7%.
  Enabled via `InferenceConfig(auto_letterbox=True)`,
  `ClassificationConfig(auto_letterbox=True)`, env var `YOWO_AUTO_LETTERBOX`,
  and CLI `--auto-letterbox` on `detect` / `classify` / `track` / `count`.
  `ConfigError` raised on `load()` for non-PyTorch backends (fixed input shapes
  are incompatible with dynamic spatial dimensions).

- **engine**: `infer_lock` in `_stream_pipeline` now covers GPU inference only
  (`backend.infer`). NMS postprocessing runs outside the lock, allowing pipeline
  workers to postprocess concurrently while the next worker uses the GPU.

- **async**: Fire-and-forget async enqueue — replaced per-frame
  `asyncio.run_coroutine_threadsafe` with `loop.call_soon_threadsafe` +
  `_enqueue_or_drop`, eliminating one event-loop round-trip per frame.
  `QueueFull` is handled explicitly (no unhandled callback exception); silently
  dropped frames are recorded in `EngineMetrics`.

- **pipeline**: `FrameCollector` O(N) round-robin polling (50 ms per-stream
  timeout) replaced with a shared `queue.Queue` and per-stream bridge daemon
  threads. O(1) per-frame dispatch regardless of stream count.

### Added

- **metrics**: `EngineMetrics.frames_dropped` — counter for frames silently
  dropped when the async queue is full. `MetricsCollector.record_frame_dropped()`
  increments it; `reset()` zeroes it. `astream()` passes the callback into
  `_enqueue_or_drop`.

- **engine**: `BaseEngine._postprocess_and_emit()` — extracted shared helper
  that runs `_process_batch` then emits the result event. Eliminates duplication
  between `_infer_from_tensor` and `_stream_pipeline._infer_batch`.

### Fixed

- **pipeline**: `_run_bridge` exhaustion sentinel now uses a plain
  `queue.put(timeout=5.0)` instead of `_put_or_stop`. Previously,
  `_stop_entry` set `stop_event` before `bridge.join()`, causing
  `_put_or_stop` to return `False` immediately and drop the sentinel,
  stalling `__iter__` indefinitely.

- **pipeline**: `_stop_entry` now joins the bridge thread (`timeout=5.0`)
  before returning, ensuring clean shutdown ordering.

- **pipeline**: Bounded `shared_q` (`maxsize = max_queue_size × 8`) prevents
  unbounded memory growth under sustained consumer lag.

- **pipeline**: Dead-bridge detection in `__iter__` — if a bridge thread exits
  without delivering its exhaustion sentinel (dropped after 5 s queue-full),
  the stream is retired automatically so iteration terminates naturally.

- **tracking**: `_munkres` infinite loop on partial-inf cost matrices — `inf −
  inf = NaN` during row reduction caused the algorithm to loop forever. Inf
  values are clamped to `1e9` before Munkres; the threshold gate rejects these
  matches.

- **pipeline**: `_check_stream_errors()` raises `RuntimeError` when all streams
  fail; logs a warning for partial failures (previously errors were silently
  swallowed).

### Refactored

- **pipeline**: `_StreamEntry.stop_flag: bool` replaced with
  `stop_event: threading.Event` for correct cross-thread visibility in bridge
  threads.

### Tests

- 1615 unit tests (up from 1543 in v2.2.2).

---

## [2.2.2] — 2026-03-04

### Added

- **classify**: `ClassificationEngine` — YOLO image classification inference
  engine. Mirrors `DetectionEngine` API but outputs `ClassificationResult`
  instead of `Detection`. No NMS, no bounding boxes — just top-k class
  probabilities. Supports all streaming modes (image, video, RTSP, webcam),
  async inference via `aclassify()`, and the same backend fallback chain as
  detection.

- **classify**: `ClassifyModel` — backbone → Classify head architecture (no
  neck). `Classify` head: `Conv(c1, 1280) → AvgPool → Dropout → Linear`.
  c\_=1280 hardcoded to match ultralytics. Classification backbone has **no
  SPPF** — layers 0–8 match detection, then C2PSA at layer 9 (not 10).

- **classify**: `postprocess_classify()` — ndim guard (1-D → 2-D auto,
  3-D+ raises `ValueError`), dtype coercion, `_is_softmaxed` heuristic,
  in-place softmax (single allocation), `tuple(row.tolist())` for all_probs.

- **classify**: `ClassificationResult` dataclass with `top1_class_id`,
  `top1_score`, `top_k` list of `(class_id, score)` pairs, and `all_probs`
  tuple for full probability distribution access.

- **classify**: `ClassificationConfig` — configuration dataclass with `top_k`
  parameter (default 5) and all `BaseEngine` fields (batch_size, device,
  precision, streaming, metrics).

- **classify**: `_CLS_LAYER_MAP` weight mapping — maps ultralytics cls
  checkpoint layers (0–10) to yowo backbone/head. SPPF absent; detection
  layer 9 shifted out.

- **cli**: `yowo classify SOURCE` command — classify images/video with
  `--model` (e.g. `yolo11n-cls`), `--weights/-w`, `--backend`, `--device`,
  `--top-k` options. `-cls` suffix required; bare detection names raise
  `BadParameter`.

- **convenience**: `classify()` one-liner — `from yowo import classify;
  results = classify("photo.jpg", model="yolo11n-cls")`.

- **events**: `EVENT_CLASSIFICATION` constant — `ClassificationEngine` emits
  `"classification"` events (not `"detection"`), enabling clean event
  separation when both engines share an event bus.

- **types**: `DeviceType.MPS` enum value for Apple Metal GPU device selection.

### Refactored

- **engine**: `BaseEngine` extracted from `engine.py` — shared by
  `DetectionEngine` and `ClassificationEngine`. `InferenceEngine` is now an
  alias for `DetectionEngine` (backward-compatible).

- **engine**: `astream()` promoted from `DetectionEngine` to `BaseEngine` —
  single implementation inherited by both engines (eliminates 65-line
  duplicate).

- **engine**: `__enter__`/`__aenter__` use `Self` return type — context
  managers return correct subclass type; per-class overrides removed.

- **engine**: `_result_event_name` property — subclasses override to emit
  task-specific events (detection vs classification).

- **classify**: `ClassificationEngine.__init__` uses config-object coalesce
  pattern — `cfg = config or ClassificationConfig(...)` (eliminates 30-line
  dual-branch).

- **backends**: `cudnn.benchmark = True` hoisted to shared path in
  `_pytorch.py` — runs for both classification and detection on CUDA.

- **backends**: MPS device type resolution fixed in `_selector.py` — explicit
  `device_override.startswith("mps")` branch.

- **weights**: `_weights.py` defers `torch` import inside
  `_extract_state_dict()`. Shape mismatch raises `RuntimeError`; missing keys
  emit `WARNING` (not silent skip).

### Tests

- 1543 unit tests (up from 1288 in v2.2.1). 255 new tests covering
  ClassificationEngine lifecycle, classify postprocess, classify weights,
  classify head, classify model, ClassificationResult, ClassificationConfig,
  BaseEngine, DetectionEngine alias, CLI classify command, MPS device type,
  model name parsing, model registry, and convenience API.

---

## [2.2.0] — 2026-03-01

### Added

- **tracking**: Cross-camera Re-Identification (ReID) system. Pluggable
  `ReIDExtractor` Protocol enabling any appearance model (CLIP, FastReID,
  CLIP-ReID, custom) to be injected into `ByteTracker` and
  `CrossCameraTracker`. Built-in extractors:
  - `CLIPExtractor` — CLIP ViT-B/16 zero-shot (512-dim, ONNX)
  - `FastReIDExtractor` — ResNet-50 SBS person ReID (256-dim, ONNX)
  - `VehicleReIDExtractor` — general vehicle ReID (256-dim, ONNX)
  - `CLIPReIDExtractor` — CLIP-ReID fine-tuned on VeRi-776 (1280-dim, ONNX),
    mAP=82.28%, Rank-1=96.66%

- **tracking**: `CrossCameraTracker` — manages per-camera ByteTrackers with
  shared `EmbeddingGallery` for cross-camera identity matching. Auto-registers
  cameras on first update. Thread-safe for concurrent multi-camera pipelines.
  `GlobalTrackedBox` dataclass with `global_id` for cross-camera identity.

- **tracking**: `EmbeddingGallery` — bounded gallery of L2-normalized track
  embeddings with cosine-distance nearest-neighbor query. Same-camera exclusion,
  top-k filtering, FIFO eviction. Thread-safe via `threading.Lock`.

- **tracking**: `CameraLinkModel` with `CameraLink` constraints — spatial-temporal
  transit window filtering that prunes infeasible cross-camera matches (e.g. a
  vehicle exiting Camera A can only appear in Camera B within a configured time
  window). Graceful degradation with configurable default window.

- **tracking**: Appearance-gated cost fusion in `ByteTracker`. `gated_fused_cost()`
  implements BoT-SORT min-cost fusion: cosine distance scaled by 0.5 when both
  appearance gate (theta_e=0.30) and IoU gate (theta_iou=0.5) pass, otherwise
  falls back to IoU-only. `needs_reid()` conditional gate achieves 99.8% skip
  rate on typical surveillance footage — ReID extraction only fires on ambiguous
  IoU assignments.

- **tracking**: `fuse_score()` — penalizes low-confidence detections by scaling
  IoU similarity by detection score (`cost = 1 - (1 - iou_cost) * score`).

- **tracking**: `_appearance_rescue()` — stage-3 re-activation of long-lost
  tracks via appearance-only matching (cosine distance on embeddings, no IoU
  requirement). Configurable `reid_lost_age` threshold.

- **tracking**: `STrack.update_embedding()` — EMA-based appearance embedding
  update with L2 renormalization (eta=0.9 default). `STrack.embedding` property
  for read access.

- **tracking**: `remove_duplicate_tracks()` and `remove_intra_duplicates()` —
  IoU-based track deduplication to eliminate ID fragmentation when overlapping
  tracks compete for the same detection.

### Experiments

- VeRi-776 Cross-Camera Vehicle ReID benchmark: CLIP zero-shot mAP=9.32%,
  FastReID SBS-S50 mAP=8.43%, CLIP-ReID VeRi mAP=82.28% (Rank-1=96.66%).
  See [`docs/experiments/2026-03-01-veri-776-cross-camera-reid-benchmark.md`](docs/experiments/2026-03-01-veri-776-cross-camera-reid-benchmark.md).

- ReID method comparison on 928-frame traffic video: `needs_reid()` gate
  achieves 99.8% skip rate (2 extraction calls per 928 frames). All three
  configurations (no ReID, CLIP, FastReID) produce identical FPS (~107).
  See [`docs/experiments/2026-03-01-reid-method-comparison.md`](docs/experiments/2026-03-01-reid-method-comparison.md).

### Tests

- 1238 unit tests (up from 1064 in v2.1.0). 180 new tests covering ReID
  extractors, embedding gallery, cross-camera tracker, camera link model,
  appearance fusion, fuse_score, needs_reid, duplicate removal, and
  appearance rescue.

---

## [2.1.0] — 2026-02-28

### Added

- **tracking**: ByteTrack multi-object tracker (`yowo.tracking`). Two-stage IoU
  association with Kalman filter (XYAH state), lost-track re-association, and
  velocity zeroing on re-activation. `ByteTracker` class with configurable
  thresholds (`track_high_thresh`, `track_low_thresh`, `match_thresh`, `max_age`,
  `min_hits`). `track_stream()` generator wires detection→tracking in a single
  call. `TrackedDetection` / `TrackedBox` dataclasses with persistent `track_id`
  and `is_confirmed` flag. Optional scipy acceleration for the Hungarian algorithm
  (`pip install yowo[tracking]`). Validated against
  [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) reference implementation.

- **counter**: `ObjectCounter` (`yowo.counter`) — per-class zone occupancy
  (ray-casting point-in-polygon) and line-crossing counting (cross-product sign
  test). Thread-safe via `threading.Lock`. Supports multiple zones and lines.
  `zone_counts`, `line_totals`, `cumulative_counts` properties. `CrossDirection`
  enum (`IN` / `OUT`). `CountZone` and `CountLine` dataclasses for geometry
  definition. `_geometry.py` with `point_in_polygon`, `segments_intersect`,
  `cross_sign`, `box_center` pure functions.

- **utils**: Reusable drawing/annotation utilities (`yowo.utils`). Extracted and
  consolidated from `examples/`, `tmp/`, and `io/_sink.py`:
  - `TRACK_PALETTE` (10 colors), `CLASS_PALETTE` (20 colors) — BGR palettes
  - `color_for_track()`, `color_for_class()` — deterministic color selectors
  - `draw_bounding_boxes()` — class-colored detection boxes with labels
  - `draw_tracked_boxes()` — track-colored boxes with `"ID:N class conf"` labels
  - `draw_zones()` — semi-transparent zone polygon overlays
  - `draw_count_lines()` — counting line overlays with labels
  - `draw_text_panel()` — translucent stats/info panel
  - `make_half_zones()` — top/bottom zone factory
  - `make_center_line()` — horizontal/vertical line factory

- **cli**: `yowo track SOURCE` — track objects with persistent IDs. `yowo count
  SOURCE` — count objects with `--zone`, `--line`, `--track`, `--json` flags.

- **engine**: Observable engine — `EventBus` for pub/sub event system,
  `MetricsCollector` for real-time inference statistics (latency, throughput,
  histograms), `HealthStatus` for backend health monitoring. Async drain API
  via `engine.astream()`.

- **backends**: True batch inference for ONNX and CoreML backends. Dynamic batch
  export via `--dynamic-batch` flag.

### Refactored

- **io**: `write_annotated_frames()` and `write_annotated_frame()` in `_sink.py`
  now delegate to `yowo.utils.draw_bounding_boxes()` instead of inline cv2 drawing.
  Removes duplicated 20-color palette and ~60 lines of drawing code.

- **examples**: `annotated_video.py` refactored to use `yowo.utils` imports instead
  of defining its own 7 drawing/factory functions (~150 lines removed).

### Performance

- **backends**: ORT session opts — `enable_mem_pattern=True`,
  `enable_mem_reuse=True`, `ORT_SEQUENTIAL`, CoreML `MLComputeUnits: ALL`.
  Drives 3–14% latency reduction on CoreML EP.

- **engine**: Pipeline overlap via `ThreadPoolExecutor(1)` + deque to overlap
  `detect()` with batch assembly. Reader-thread preprocess via
  `ThreadedFrameReader(preprocess_fn=...)`.

### Experiments

- ByteTrack + ObjectCounter annotated video benchmark on 928-frame traffic video.
  ONNX+CoreML: YOLO26n 82 FPS, YOLO26x 21 FPS. PyTorch+MPS: YOLO26s 63 FPS.
  CoreML wins 4/5 variants (up to 1.32×). ByteTrack overhead 0.3–0.7ms (1.3–5.2%).
  See [`docs/experiments/2026-02-28-bytetrack-counter-annotated-video-benchmark.md`](docs/experiments/2026-02-28-bytetrack-counter-annotated-video-benchmark.md).

### Tests

- 1064 unit tests (up from 724 in v2.0.0). 0 pyright errors. 0 ruff errors.

---

## [2.0.0] — 2026-02-26

### Added

- **export**: INT8 export completion — TensorRT calibrator (`IInt8EntropyCalibrator2`)
  with device buffer reuse and cache persistence, ONNX static quantization via
  `quantize_onnx_static()`, and `calibration_batches()` iterator using
  `cv2.dnn.blobFromImages`. Wire calibrator into `_convert_tensorrt()`.

- **engine**: Open backend factory — `backend_instance` parameter on
  `InferenceEngine.__init__()` allows injecting custom `InferenceBackend`
  implementations without modifying the library. Skips auto-selection when provided.

- **backends**: Native CoreML backend and export for Apple Silicon — `CoreMLBackend`
  class implementing `InferenceBackend` Protocol, direct PyTorch-to-CoreML export
  via `coremltools.convert()` (no ONNX intermediate), `COREML` enum values in
  `BackendType` and `ExportFormat`, auto-selection priority 3 on macOS ARM64.

- **io**: Source metadata — lazy `_probe()` on `VideoFileSource` consolidating
  `total_frames`, `resolution`, and `fps` into single `VideoCapture` open.
  `resolution` and `fps` properties on all concrete source classes.

---

## [1.3.1] — 2026-02-26

### Fixed

- **config, io, export**: Consolidate duplicated media format constants
  (`IMAGE_EXTS`, `VIDEO_EXTS`, `RTSP_SCHEMES`) into `types.py`. Previously
  defined independently in `config.py`, `io/_source.py`, and
  `export/_calibration.py` — any divergence would cause `classify_source()`
  and `open_source()` to disagree silently. Now a single source of truth.

- **cli**: Replace fragile default-value comparison with
  `ctx.get_parameter_source()` for CLI `--preset` override detection.
  Previously, `--confidence 0.25` (identical to Click default) was
  indistinguishable from "not specified" and silently dropped. Now uses
  Click's `ParameterSource.COMMANDLINE` to correctly detect all explicit
  user input regardless of value.

- **tests**: Fix Jetson test fixtures to include GPU device, matching real
  Jetson hardware topology (CPU + Tegra iGPU). Previously `test_jetson`,
  `test_jetson_video`, and `test_jetson_live` constructed profiles without
  a GPU, which did not exercise the `is_jetson` vs `has_nvidia_gpu` priority
  in `classify_device()`.

### Added

- **types**: `IMAGE_EXTS`, `VIDEO_EXTS`, `RTSP_SCHEMES` public constants —
  canonical set of supported media file extensions and RTSP scheme prefixes.

---

## [1.3.0] — 2026-02-26

### Added

- **pipeline**: Multi-stream inference pipeline — `FrameCollector`, `BatchScheduler`,
  `DetectionRouter`, and `run_pipeline()`. Orchestrates N concurrent video/RTSP streams
  through a single `InferenceEngine` with batched inference, per-stream result routing,
  and health monitoring. Four single-responsibility modules following black box
  architecture principles (validated via profiling that >93% of hot path is native C++).

  - `FrameCollector` — manages N concurrent stream readers via `ThreadedFrameReader`
    composition. Round-robin dispatch, thread-safe add/remove, per-stream health states
    (`StreamState`), and programmatic error inspection (`stream_errors`).
  - `BatchScheduler` — accumulates `TaggedFrame` objects into batches by capacity or
    timeout flush (whichever comes first). Supports `stop_event` for responsive
    cancellation. Accepts `Iterable` for reuse across pipeline runs.
  - `DetectionRouter` — positional-index routing of `Detection` results to per-stream
    callbacks. Deterministic: `detections[i]` maps to `batch[i]`.
  - `run_pipeline()` — thin wiring function connecting all modules through
    `InferenceEngine.detect()`. Auto-disables feature cache for mixed-source batches.

- **types**: `TaggedFrame(dataclass, slots=True)` — frame annotated with `stream_id`.
  `StreamState(StrEnum)` — `RUNNING | RECONNECTING | STOPPED | ERROR`.

### Documentation

- Multi-stream pipeline user guide (section 7) with setup, graceful shutdown, health
  monitoring, and multi-camera warehouse use case.
- Pipeline module README with design principles, data flow, and API reference.
- Self-contained benchmark suite with setup tooling.
- Comprehensive user guide covering CLI, Python API, backends, streaming, export,
  configuration, error reference, and 7 use cases.

---

## [1.2.0] — 2026-02-25

### Added

- **io, engine**: Source-aware pipeline dispatch — Phase 3.
  `InferenceEngine.stream()` now inspects `source.is_live` to select the
  optimal execution path automatically:
  - `_stream_live` — live sources (RTSP, webcam): `ThreadedFrameReader` runs
    in a background daemon thread, decoupling network I/O from inference.
    `FrameDropPolicy` controls queue behaviour under backpressure.
  - `_stream_pipeline` — offline video with `prefetch=True`: background
    prefetch overlaps decode and inference to hide I/O latency.
  - `_stream_sync` — offline video with `prefetch=False`: legacy sequential
    path, zero regression guarantee.
  - `_stream_single` — image sources: single-frame fast path.
  No user configuration required — `open_source("rtsp://...")` sets
  `is_live=True` and the engine dispatches accordingly.

- **io**: `ThreadedFrameReader` — bounded background frame reader.
  `io/_reader.py`: daemon thread fills a `deque(maxlen=max_queue_size)` from
  any `FrameSource`. Exposes `frames_read`, `frames_dropped`, `drop_rate`
  counters. Idle timeout (5 s) triggers clean shutdown without deadlock.
  Thread-safe via `threading.Lock` on all shared state.

- **types**: `FrameDropPolicy(StrEnum)` — three drop strategies for live
  sources under backpressure:
  - `NONE` — block until space is available (no drops, bounded latency risk)
  - `LATEST` — evict oldest frame, insert newest (always-current view)
  - `SKIP_OLDEST` — pop back of queue to make room (FIFO order preserved)

- **io, postprocess**: Pre-allocated I/O buffers eliminate per-frame heap
  allocation on the hot path.
  - `PreprocessBuffer(max_batch, target_size)` — pre-allocated
    `(max_batch, H, W, 3)` uint8 staging array. `preprocess_into()` writes
    directly into the buffer, replacing `cv2.copyMakeBorder` allocation.
  - `PostprocessBuffer(max_detections)` — pre-allocated `(max_detections, 4)`
    float32 scratch for inverse letterbox. `get_inverse_out(n)` returns a view
    with a fresh-allocation fallback for oversized batches.

- **types, engine**: `is_free_threaded()` — runtime GIL detection.
  `not sys._is_gil_enabled()` with `AttributeError` guard for Python < 3.13.
  `InferenceEngine` uses this to auto-set `pipeline_workers=2` on free-threaded
  Python (GIL=OFF) and `pipeline_workers=1` otherwise.

- **engine**: New `InferenceEngine` kwargs for Phase 3 pipeline control.
  `prefetch: bool` (default `True`), `pipeline_workers: int | None` (default
  `None`, auto), `frame_drop_policy: FrameDropPolicy` (default `NONE`),
  `max_queue_size: int` (default `4`). All reflected in `InferenceConfig`.

### Refactored

- **engine, backends**: Black box boundary fixes from architecture audit.
  - All cross-package imports now use public `__init__.py` surfaces instead of
    private `_module` paths.
  - `InferenceBackend` Protocol gains `clear_kv_cache()` and `set_source_id()`.
    Removes all `hasattr`/`type: ignore[attr-defined]` from engine.py.
  - `InferenceEngine` constructor flattened: accepts `InferenceConfig` or
    individual kwargs (all optional, defaults to YOLO26 Nano). `ModelSpec` is
    no longer accepted as a positional argument.

### Performance

- **io**: `PreprocessBuffer` eliminates `cv2.copyMakeBorder` allocation on
  every frame — saves one `(H, W, 3)` uint8 copy (~2.8 MB for 1280×720 input).
- **postprocess**: `PostprocessBuffer` reuses inverse-letterbox scratch array
  across frames — saves one `(max_detections, 4)` float32 allocation per frame.
- **engine**: `pipeline_workers=2` on free-threaded Python (GIL=OFF) delivers
  ~1.5× throughput on YOLO26n CPU inference via true thread parallelism
  (1.47× measured on Apple M4 Pro, Python 3.13.3+freethreaded).

### Fixes

- **engine, io**: Thread safety hardening (P0/P1 review).
  `ThreadedFrameReader` uses `threading.Lock` on `frames_dropped` counter
  increment to prevent lost updates under concurrent reads. `_stop_event` is
  checked before each `deque.append` to avoid post-stop writes.

### Experiments

- Phase 3 source-aware pipeline: offline video 1.03× vs legacy; RTSP
  stream-capped at ~11 FPS on localhost (expected — gains visible on
  high-latency remote cameras); CoreML 3.5–3.75× faster than PyTorch on
  offline video; free-threaded Python 3.13t `pipeline_workers=2` → 58.4 FPS
  vs 39.3 FPS (1.49×).
  See [`docs/experiments/2026-02-25-phase3-source-aware-pipeline-benchmark.md`](docs/experiments/2026-02-25-phase3-source-aware-pipeline-benchmark.md).

### Breaking Changes

- `InferenceEngine(spec, ...)` no longer accepts `ModelSpec` as the first
  positional argument. Use `InferenceEngine(model_family=..., model_size=...)`
  or `InferenceEngine(InferenceConfig(...))` instead.
- `confidence` kwarg renamed to `confidence_threshold` (matches `InferenceConfig`).
- `InferenceConfig` removes 4 dead fields: `max_memory_mb`,
  `reconnect_timeout_s`, `frame_skip`, `max_frames`. Adds 3 fields: `cache`,
  `cache_dir`, `kv_cache`.

---

## [1.1.0] — 2026-02-24

### Features

- **arch, cache**: Feature map caching for sequential inference.
  New `src/yowo/cache/` module with `FeatureCache` coordinator, bounded `FeatureStore`
  (in-memory default, mmap opt-in), and L1 frame-similarity check. On cache hits,
  backbone + neck are skipped entirely — only the detection head runs (~60–85% compute
  savings for slow-moving scenes). Integrated into `PyTorchBackend` via a forward hook
  and exposed via `InferenceEngine(cache=True, cache_dir=Path(...))`.

- **arch, backends, export**: KV cache as explicit ONNX I/O for all runtimes.
  PyTorch backend: fused QKV conv with K,V cached as Python state across frames;
  C2PSA/C3k2PSA blocks skip on spatial-mean fingerprint similarity (< 0.01).
  ONNX export: K,V tensors exposed as explicit model inputs/outputs via `YOLOKVWrapper`
  with a `use_cache` scalar flag — enables stateful streaming on ONNX Runtime, TensorRT,
  and OpenVINO. `export_model(kv_cache=True)` produces KV-capable ONNX models.
  `InferenceEngine(kv_cache=True)` wires the full pipeline.

- **backends, hardware**: CoreML EP auto-detection for Apple Neural Engine.
  `OnnxBackend._select_providers()` adds CoreML EP when `onnxruntime_has_coreml=True`
  (macOS). Provider priority: TensorRT → CUDA → CoreML → CPU → PyTorch.
  Hardware probe adds `onnxruntime_has_coreml: bool` to `InstalledLibraries`.
  No user configuration required — 4–5× speedup vs PyTorch on Apple Silicon.

- **backends, cli**: MPS (Metal GPU) support and CLI image output.
  `PyTorchBackend` accepts `device="mps"`. Manual SDPA fallback in `_attention.py`
  when `q.is_mps` (workaround for PyTorch MPS SDPA shape bug with `key_dim != head_dim`).
  CLI `-o` flag now dispatches on extension: `.jpg`/`.png` → annotated frame via
  `io/_sink.py:write_annotated_frame()`, otherwise → JSON detections file.

### Performance

- **arch**: Phase 2 architecture micro-optimizations (cumulative ~3–6% CPU gain).
  DFL `register_buffer` pre-shaped to `(1,1,c1,1)` — eliminates per-frame reshape.
  `Detect._decode()` caches pre-transposed anchors/strides (`_cached_anchors_t`,
  `_cached_strides_t`) — avoids per-frame `.T` allocation.
  `Conv.forward_fuse_no_act()` bypasses `nn.Identity` dispatch after `fuse()`.
  `Bottleneck` forward specialization post-`fuse()` eliminates per-call branch.
  Results: YOLO26 10–17% faster, YOLO11 1–4% faster vs ultralytics (CPU, FP32, batch=1).

- **backends**: ONNX thread tuning for Apple Silicon.
  `intra_op=cpu_count//2`, `inter_op=cpu_count//4` — avoids E-core scheduling
  overhead that caused thread migration slowdowns with all-core defaults.

### Fixes

- **arch**: `Detect.stride` registered as `persistent=False` buffer with `copy_()`
  to prevent Dynamo recompilation guard failures on stride mutation under
  `torch.compile`.

- **arch**: `Attention` reshape uses explicit `.contiguous().view()` for
  `fullgraph=False` compile path compatibility.

- **backends**: Silence `set_num_interop_threads` error on repeated `load()` calls
  (PyTorch raises if called more than once per process).

- **export**: Internalize external data for KV ONNX exports.
  `torch.onnx.export` dynamo path creates `.onnx.data` sidecar files;
  `_export_onnx_kv()` now loads with `onnx.load(load_external_data=True)` and
  re-saves to internalize all tensors. Fixes CoreML EP load failure:
  `"model_path must not be empty"`.

### Experiments

- CPU arch optimizations: 9/10 variants faster vs ultralytics. Avg 1.07×.
  YOLO26 family 1.10–1.17×; YOLO11 family 1.00–1.04×. Box IoU 0.967–0.995.
  See [`docs/experiments/2026-02-24-arch-inference-optimization-benchmark.md`](docs/experiments/2026-02-24-arch-inference-optimization-benchmark.md).
- CoreML EP: avg **4.36× faster** than PyTorch across all 10 variants.
  Nano models 147–188 FPS; XL models 27–29 FPS on Apple M4 Pro Neural Engine.
  MPS (Metal GPU): avg 1.32× faster than ultralytics (yowo-eager), 1.36× with KV cache.
  See [`docs/experiments/2026-02-24-onnx-coreml-optimization-benchmark.md`](docs/experiments/2026-02-24-onnx-coreml-optimization-benchmark.md).

---

## [0.1.0] — 2026-02-24

### Features

- **arch**: Remove ultralytics dependency — native YOLO11 and YOLO26 architectures
  (`Backbone + FPNPANNeck + Detect`) implemented from published specifications.
  `build_model()` + `load_weights()` public API. Supports all 10 variants:
  `yolo11{n,s,m,l,x}` and `yolo26{n,s,m,l,x}`.

### Fixes

- **arch**: Correct YOLO11/26 weight mapping for all 10 model variants.
  `C3k` now correctly inherits from `C3` (not `C2f`). Added `C3k2PSA`
  for YOLO26 neck layer 22. `backbone_c3k` and `neck_c3k` are now
  size-based (True for m/l/x) instead of family-based — matching
  actual ultralytics checkpoint structure.

- **arch, backend**: Eliminate non-leaf parameter warning in PyTorch backend.
  `fuse_conv_and_bn` wrapped in `torch.no_grad()` to prevent `CopyBackwards`
  grad_fn on fused Conv2d parameters. EMA checkpoint tensors detached before
  `load_state_dict`. CPU thread pool capped at `cpu_count // 2` to prevent
  memory-bandwidth saturation on many-core machines.

### Performance

- **postprocess, io**: Optimize inference hot path with C++ NMS and zero-alloc ops.
  `preprocess()` uses `cv2.dnn.blobFromImages` (single C++ call).
  NMS replaced with `cv2.dnn.NMSBoxes` + class-offset trick.
  Anchor caching in `Detect._decode()`, DFL `.contiguous()` before softmax,
  `non_blocking=True` H2D transfer.

---

## [0.0.1] — 2026-02-23

Initial beta release.

[2.2.2]: https://github.com/TinDang97/yowo/compare/v2.2.1...v2.2.2
[2.2.0]: https://github.com/TinDang97/yowo/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/TinDang97/yowo/compare/v2.0.0...v2.1.0
[1.3.1]: https://github.com/TinDang97/yowo/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/TinDang97/yowo/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/TinDang97/yowo/compare/v1.1.1...v1.2.0
[1.1.0]: https://github.com/TinDang97/yowo/compare/v1.0.2...v1.1.0
[0.1.0]: https://github.com/TinDang97/yowo/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/TinDang97/yowo/releases/tag/v0.0.1
