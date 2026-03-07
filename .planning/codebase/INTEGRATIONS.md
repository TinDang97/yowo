# External Integrations

**Analysis Date:** 2026-03-07

## APIs & External Services

**Weight Downloads (GitHub Releases):**
- YOLO11 weights: `github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{n,s,m,l,x}.pt`
- YOLO26 weights: `github.com/ultralytics/assets/releases/download/v8.4.0/yolo26{n,s,m,l,x}.pt`
  - Client: `requests>=2.31` with retry + exponential backoff (2, 4, 8s)
  - Implementation: `src/yowo/models/_weights.py`
  - Cache: `~/.cache/yowo/weights` (override with `YOWO_CACHE_DIR` env var)
  - Auth: None required (public GitHub releases)
  - Timeout: 60s connection, 300s total download wall-clock

## Inference Backends (Hardware SDKs)

All backends are optional, lazy-imported, and follow the `InferenceBackend` Protocol defined in `src/yowo/backends/__init__.py`.

**PyTorch** (priority 7 - universal fallback):
- Package: `torch>=2.0` (optional dep `yowo[pytorch]`)
- Backend: `src/yowo/backends/_pytorch.py`
- Devices: CPU, CUDA, MPS
- Features: Native YOLO arch, KV cache, feature map cache, model fusion

**ONNX Runtime** (priority 2/4/6 depending on EPs):
- Package: `onnxruntime>=1.17` or `onnxruntime-gpu>=1.17`
- Backend: `src/yowo/backends/_onnx.py`
- Execution providers: CUDAExecutionProvider, CoreMLExecutionProvider, CPUExecutionProvider
- OrtValue optimization: `src/yowo/backends/_ortvalue.py`

**TensorRT** (priority 1 - highest on NVIDIA):
- Package: `tensorrt>=10.0` (manual install from NVIDIA PyPI)
- Backend: `src/yowo/backends/_tensorrt.py`
- Requires: NVIDIA GPU

**OpenVINO** (priority 5):
- Package: `openvino>=2024.0`
- Backend: `src/yowo/backends/_openvino.py`
- Targets: Intel CPUs, Intel iGPUs

**CoreML** (priority 3 on Apple Silicon):
- Package: `coremltools>=7.0`
- Backend: `src/yowo/backends/_coreml.py`
- Targets: Apple Neural Engine, Apple GPU
- Note: 4-5x faster than PyTorch on Apple Silicon

**Backend auto-selection** is implemented in `src/yowo/backends/_selector.py` with the priority chain above. Falls back through `_FALLBACK_CHAIN` if primary backend fails at load time.

## ReID Feature Extractors (ONNX Models)

All ReID extractors use ONNX Runtime sessions internally and follow the `ReIDExtractor` Protocol in `src/yowo/tracking/_reid.py`.

**CLIP ViT-B/16** (zero-shot):
- Class: `CLIPExtractor` in `src/yowo/tracking/_reid.py`
- Input: (batch, 3, 224, 224), CLIP normalization
- Output: 512-dim L2-normalized embeddings
- Model: User-provided ONNX file

**CLIP-ReID VeRi** (fine-tuned vehicle ReID):
- Class: `CLIPReIDExtractor` in `src/yowo/tracking/_clip_reid.py`
- Input: (batch, 3, 256, 256), mean=0.5/std=0.5 normalization
- Output: 1280-dim embeddings (768 + 512 concat)
- Performance: mAP=82.28%, Rank-1=96.66% on VeRi-776

**FastReID SBS-S50** (person ReID):
- Class: `FastReIDExtractor` in `src/yowo/tracking/_reid.py`
- Input: (batch, 3, 256, 128), ImageNet normalization
- Output: 256-dim L2-normalized embeddings

**Vehicle ReID** (generic):
- Class: `VehicleReIDExtractor` in `src/yowo/tracking/_reid.py`
- Input: (batch, 3, 224, 224), ImageNet normalization
- Output: configurable-dim L2-normalized embeddings

## Data Storage

**Databases:**
- ChromaDB (optional) - Persistent vector storage for cross-camera ReID
  - Package: `chromadb>=0.5.0` (optional dep `yowo[chromadb]`)
  - Client: `chromadb.PersistentClient` with cosine distance
  - Implementation: `src/yowo/tracking/_chroma_gallery.py`
  - Configured via constructor path parameter

**File Storage:**
- Weight cache: `~/.cache/yowo/weights/{family}/{size}/{name}.pt`
- Exported models: `~/.yowo/models/` (default, configurable via `ExportConfig.output_dir`)
- Feature map mmap cache: configurable via `InferenceConfig.cache_dir`

**Caching:**
- In-memory feature map cache (PyTorch backend, `InferenceConfig.cache=True`)
- Mmap-backed feature cache (PyTorch backend, `InferenceConfig.cache_dir`)
- KV attention cache (PyTorch backend, `InferenceConfig.kv_cache=True`)
- In-memory embedding gallery (`src/yowo/tracking/_gallery.py`)

## Hardware Detection

**GPU Detection:**
- NVIDIA GPUs via `pynvml` (lazy import) in `src/yowo/hardware/_detect.py`
- Reports: device name, memory total/available, compute capability
- Jetson detection via CPU device metadata

**Library Probing:**
- Implementation: `src/yowo/hardware/_capabilities.py`
- Probes: torch, tensorrt, onnxruntime, openvino, coremltools
- Each probe uses `importlib.util.find_spec` + lazy import
- Result: `InstalledLibraries` frozen dataclass
- Cached: process-level singleton via `get_hardware_profile()` in `src/yowo/hardware/__init__.py`

## Authentication & Identity

**Auth Provider:**
- None - This is an inference library, not a web service
- No user authentication or identity management

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)
- Internal: `EngineMetrics` tracks error counts, health transitions
- Events: `EVENT_ERROR` and `EVENT_HEALTH_CHANGE` via `EventBus`

**Logs:**
- Standard `logging` module throughout
- Logger names follow module hierarchy (e.g., `yowo.hardware._capabilities`)
- No structured logging framework

**Metrics:**
- Built-in `MetricsCollector` (`src/yowo/metrics/`)
- Tracks: latency, throughput, error counts per engine
- Exposed via `EngineMetrics` dataclass
- Disable with `InferenceConfig.metrics_enabled=False`

**Event System:**
- `EventBus` in `src/yowo/events/__init__.py`
- Non-blocking async-capable pub/sub with background daemon thread
- Well-known events: `EVENT_DETECTION`, `EVENT_CLASSIFICATION`, `EVENT_ERROR`, `EVENT_HEALTH_CHANGE`
- Queue capacity: 1000 events, drops silently on overflow

## CI/CD & Deployment

**Hosting:**
- GitHub (source code and releases)

**CI Pipeline:**
- GitHub Actions: `.github/workflows/release.yml`
- Trigger: push to `main`
- Quality gate job: lint (ruff) -> format check (ruff) -> type check (pyright) -> unit tests (pytest)
- Release job: `python-semantic-release` for automated versioning
- Python setup: `actions/setup-python@v5` with `python-version: "3.11"`
- UV setup: `astral-sh/setup-uv@v3`

**Release Process:**
- Conventional commits parsed by `python-semantic-release`
- Auto-version bump: feat -> minor, fix/perf/refactor -> patch
- Artifacts uploaded to GitHub Releases (`upload_to_vcs_release = true`)
- Tag format: `v{version}`
- Build command: `uv build`

## Environment Configuration

**Required env vars:**
- None strictly required - all have sensible defaults

**Optional env vars (runtime):**
- `YOWO_MODEL_FAMILY` - Model family (yolo11/yolo26)
- `YOWO_MODEL_SIZE` - Size variant (n/s/m/l/x)
- `YOWO_BACKEND` - Force backend (pytorch/onnx/tensorrt/openvino/coreml)
- `YOWO_DEVICE` - Force device (auto/cpu/cuda/cuda:0/mps)
- `YOWO_PRECISION` - Force precision (fp32/fp16/int8)
- `YOWO_CONFIDENCE` - Detection confidence threshold [0.0, 1.0]
- `YOWO_IOU` - NMS IoU threshold [0.0, 1.0]
- `YOWO_BATCH_SIZE` - Frames per inference batch
- `YOWO_NUM_CLASSES` - Override output class count
- `YOWO_CACHE_DIR` - Weight cache directory override
- `YOWO_KV_CACHE` - Enable KV attention cache (true/false)
- `CI` - Suppress tqdm progress bars in download (true/1/yes)

**Secrets location:**
- `GITHUB_TOKEN` - CI only, via GitHub Actions secrets for semantic-release

## Webhooks & Callbacks

**Incoming:**
- None - Library, not a service

**Outgoing:**
- None - No external webhook calls
- Internal: `EventBus` provides pub/sub for in-process event callbacks

---

*Integration audit: 2026-03-07*
