# Performance / Model Lifecycle / Edge-Deployment review — yowo production readiness

## Verdict

Not production-ready as a deployment library. The runtime *plumbing* is genuinely good — bounded
queues, drop policies, RTSP reconnect, warmup NaN/Inf validation, a rolling p50/p95/p99 histogram,
a real reproducible `bench/` suite for the arch/ONNX numbers. What is missing is everything an
operator needs to **predict accuracy, latency and memory before they ship**: `precision` is a
decorative field that no backend consumes; exported artifacts carry no runtime provenance and, with
the default install, ship as a broken two-file pair the metadata cannot describe; INT8 is calibrated
on a preprocessing distribution that never occurs at inference and no accuracy delta is ever
measured; the feature cache silently trades recall for speed and is on by default in two presets;
and the benchmark/tune harnesses attribute results to backends that never ran.

The single biggest gap: **there is no artifact contract.** An `.onnx`/`.engine` produced by yowo does
not record its opset, torch/ORT/TRT/driver version, real input shape, or class count, and nothing
validates any of that at load. Every silent-corruption class bug in this report descends from that.

Everything below was read in the repo at `3340bb3`; findings marked **VERIFIED** were reproduced by
running the commands shown. No source file was modified (`git status --porcelain` unchanged).

---

## P0 — blocks launch

### D1. ONNX export ships an artifact the sidecar cannot describe — broken on a default install

- **Symptom:** Without the optional `onnxslim` package, `export_model(..., ExportFormat.ONNX)`
  writes a 612 KB graph stub plus a separate `model.onnx.data` weights file, while
  `ExportMetadata.file_path` names only the `.onnx` and `file_size_bytes` records only its 612 KB.
  Copying "the model file" the metadata names to a target device fails at session creation. With
  `onnxslim` present the `.onnx` is self-contained but an orphaned 11 MB `.onnx.data` is still left
  behind. Separately, `opset_version=17` is requested but opset **18** is produced — torch's
  downgrade throws `No Adapter To Version 17 for Resize` and the export proceeds anyway.
- **Evidence:**
  - `src/yowo/export/_exporter.py:223-260` — `_export_onnx()` never internalizes external data and
    never unlinks `.onnx.data`. Contrast `src/yowo/export/_exporter.py:312-328`, where the KV path
    explicitly loads with `load_external_data=True`, re-saves, and `data_path.unlink(missing_ok=True)`.
    The standard path is the one every user hits.
  - `src/yowo/export/_exporter.py:249-258` — onnxslim is optional (`except ImportError: … skipping`),
    and it is only in the `export` extra (`pyproject.toml:33`). A user on `yowo[pytorch,onnx]`
    has no onnxslim.
  - `src/yowo/export/_exporter.py:180-206` — `file_size_bytes` is `exported_path.stat().st_size`;
    `file_path` is the single `.onnx`. Neither mentions the sidecar weights file.
  - `src/yowo/export/_exporter.py:240` — `opset_version=17` requested.
  - VERIFIED (torch 2.10.0, macOS arm64):

    ```
    # with onnxslim installed
    uv run python -c "from yowo.export import export_model; ..."   # full script in session log
    → producer: pytorch 2.10.0   opset: [('', 18)]      # requested 17
    → dir: yolo11n.onnx (11M), yolo11n.onnx.data (11M, orphan), yolo11n.yowo.json

    # with onnxslim import blocked (simulating the default install)
    → sidecar file_size_bytes: 612255   actual .onnx: 612255
    → external initializers: 130 / 201
    → ships-as-one-file: NO -> [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION :
      filesystem error: in file_size: No such file or directory ["…/yolo11n.onnx.data"]
    ```
- **Why it blocks:** The primary deliverable of an export/deploy library is a file you can copy to a
  device. Under the documented install this produces a file that does not load, and the sidecar
  actively tells you it is complete. The opset mismatch means the artifact may be rejected by the
  older ORT/TRT/OpenVINO parsers on the very edge devices this targets — and nothing records which
  opset it actually is.
- **Smallest fix:** In `_export_onnx`, apply the same internalize-and-unlink block already written at
  `_exporter.py:312-328`; assert the saved `.onnx` has zero `EXTERNAL` initializers before returning.
  Read the real opset back from the saved model and store it in `ExportMetadata` rather than
  assuming the requested value took effect.

### D2. `precision` is inert on every backend — the engine reports a precision it is not running

- **Symptom:** `InferenceConfig(precision=FP16)`, `--precision int8`, and the auto-selected precision
  change nothing about execution. `health_report().precision_current` and `engine.selection.precision`
  report a value (typically `"fp16"` on any NVIDIA GPU) that is not what the model runs.
- **Evidence:**
  - `grep -rn "precision" src/yowo/backends/*.py | grep -v _selector.py` → **no matches.** No backend
    `load()` signature accepts precision (`_tensorrt.py:100`, `_onnx.py:94`, `_openvino.py`,
    `_coreml.py`).
  - `src/yowo/backends/__init__.py:174-182` — `create_backend()` constructs `PyTorchBackend` without
    the `fp16` argument, so `_pytorch.py:43` `fp16: bool = False` is always False and the autocast
    branch at `_pytorch.py:301-302` is dead.
  - `src/yowo/backends/_tensorrt.py:141-154` — the TRT EP provider options set no `trt_fp16_enable`,
    `trt_int8_enable`, `trt_max_workspace_size`, or shape profile.
  - `src/yowo/engine.py:402` — `precision_current=self._selection.precision.value` is surfaced in the
    health report and `export_metrics()`.
  - `src/yowo/engine.py:667-678` — `_try_precision_fallback()` calls `set_precision("fp16")`, a method
    no backend implements, so it always takes the `logger.debug("not supported")` branch.
- **Why it blocks:** Precision is the primary lever an edge deployment pulls. Shipping an API that
  accepts it, records it, reports it in health and metrics, and does nothing with it is worse than
  not offering it — an operator will size a Jetson on a precision they are not running. In practice
  precision is an *artifact* property here (you get INT8 only by pointing `weights_path` at an INT8
  file), but the API presents it as a runtime knob.
- **Smallest fix:** Either (a) thread precision into each backend (`trt_fp16_enable`/`trt_int8_enable`
  for the TRT EP, `fp16=` for PyTorch, `compute_precision` for CoreML) or (b) if precision is
  artifact-derived, delete the runtime knob and derive `selection.precision` from the loaded model's
  actual input/weight dtype. Do not report a value that was never applied.

### D3. Artifacts and caches are keyed on nothing — the stale-load silent-corruption class

- **Symptom:** No exported artifact records the toolchain that built it, nothing validates an
  artifact at load, and two caches are keyed by filename alone.
- **Evidence:**
  - `src/yowo/export/_metadata.py:19-37` — `ExportMetadata` records python/platform/gpu_name but **no**
    opset, torch version, onnx version, onnxruntime version, TensorRT version, CUDA version, NVIDIA
    driver version, num_classes, or class names. Nothing in `src/yowo/hardware/_capabilities.py`
    even probes the driver version (`nvidia-smi --query-gpu=driver_version` is never asked for;
    see `_detect.py:294` `_SMI_BASE_FIELDS`).
  - `src/yowo/export/_exporter.py:191-193` — `batch_size=1` and `input_shape=[1,3,imgsz,imgsz]` are
    hardcoded into the sidecar even when `dynamic=True`. VERIFIED in the sidecar produced above.
  - Nothing reads the sidecar back: `TensorRTBackend.load()` (`_tensorrt.py:100-221`) and
    `OnnxBackend.load()` (`_onnx.py:94-…`) only open a session. `ExportMetadata.load()`
    (`_metadata.py:50-56`) has no caller in `src/`.
  - `src/yowo/backends/_tensorrt.py:141-148` — the TRT EP engine cache is enabled with
    `trt_engine_cache_path = str(Path(model_path).parent)`, i.e. written into the model directory
    (often a shared/NFS/read-only model mount) with no `trt_engine_cache_prefix`, no timing cache,
    and no version stamp of yowo's own.
  - `src/yowo/export/_exporter.py:394` + `src/yowo/export/_int8.py:101-105` — the INT8 calibration
    cache is `engine_path.with_suffix(".calib")` and `read_calibration_cache()` returns it whenever
    the file exists. Re-exporting the same model name with **different `--calibration-data`, a
    different `imgsz`, or different weights silently reuses the previous activation scales.**
- **Why it blocks:** This is the exact failure the persona has shipped: a cache keyed on nothing, so
  a driver/TRT upgrade or a changed calibration set produces an artifact that is either refused at
  load (best case) or silently wrong (the `.calib` case). An operator has no way to answer "was this
  engine built for this machine?" because the artifact does not say.
- **Smallest fix:** Add `opset`, `torch_version`, `onnx_version`, `onnxruntime_version`,
  `tensorrt_version`, `cuda_version`, `driver_version`, `num_classes` to `ExportMetadata`; on backend
  load, read the sidecar if present and log/raise on mismatch. Key the `.calib` filename on
  `sha256(sorted calibration file list + imgsz + source_weights)`. Move the TRT engine cache to a
  writable per-host cache dir under the fingerprint.

### D4. INT8 is calibrated on a preprocessing distribution that never occurs at inference, and no accuracy delta is ever measured

- **Symptom:** Calibration images are **stretch-resized** to a square; inference **letterboxes** with
  114-gray padding. The activation ranges INT8 is fitted to therefore come from a distribution the
  model never sees in production. The docstring claims the opposite.
- **Evidence:**
  - `src/yowo/export/_calibration.py:104` — `cv2.resize(img, (input_size, input_size))` — plain
    stretch, no aspect preservation, no padding.
  - `src/yowo/export/_calibration.py:83` — docstring: *"matching the inference preprocessing
    pipeline"*. It does not.
  - Inference path for comparison: `src/yowo/io/_decode.py:202-229` — uniform
    `scale = min(h/H, w/W)` then `cv2.copyMakeBorder(..., value=(114,114,114))`. On 16:9 input
    roughly 25% of every real input tensor is constant gray; **zero** of the calibration tensors
    contain it.
  - Same reader feeds both quantizers: `src/yowo/export/_int8.py:88-90` (TensorRT) and
    `_int8.py:162-164` (ONNX `quantize_static`).
  - No accuracy gate exists anywhere. `src/yowo/benchmark/_runner.py:259-270` returns per-backend
    `map_50_95` but nothing compares FP32 vs INT8 and nothing has a threshold.
  - `src/yowo/export/_calibration.py:22` — `_MIN_CALIBRATION_IMAGES = 10`; a 10-image INT8
    calibration is allowed with a warning.
  - The quantized artifact cannot even be measured with the shipped harness: `run_benchmark()`
    (`src/yowo/benchmark/__init__.py:86-115`) builds `ModelSpec(family, size, task)` with **no**
    `weights_path` and no precision, and the CLI (`src/yowo/cli/_main.py:48-64`) exposes neither
    `--weights` nor `--precision`. `bench/` contains no mAP measurement at all
    (`grep -rln "mAP\|pycocotools" bench/` → no matches).
- **Why it blocks:** "INT8 export" is an advertised v2.0 feature. A quantization path with a known
  preprocessing mismatch, a 10-image floor, and no reproducible way to measure the resulting mAP is
  an accuracy regression waiting to be discovered by a customer.
- **Smallest fix:** Make `calibration_batches()` call the same `preprocess()` used at inference
  (`yowo/io/_decode.py`) instead of `cv2.resize`. Add `--weights` and `--precision` to
  `yowo benchmark` so an exported artifact can be evaluated, and add a `yowo export --verify-map
  --data <coco> --max-map-drop 0.02` gate that fails the export when the delta exceeds the budget.

### D5. The feature cache silently drops detections, and two presets turn it on by default

- **Symptom:** `cache=True` reuses the previous frame's neck features when the current frame is
  "similar". Similarity is measured **only as the per-channel spatial mean** of the input tensor. Two
  frames with the same mean but different content (a person entering a static scene, a small object
  appearing) score as identical, the backbone and neck are skipped, and the detection head runs on
  stale features — the new object is never detected. No accuracy impact is measured, and the docs
  advertise the speedup with no caveat.
- **Evidence:**
  - `src/yowo/cache/__init__.py:99` — `current_fp = current_tensor.mean(axis=(2, 3))` reduces a whole
    frame to 3 numbers per image.
  - `src/yowo/cache/__init__.py:106-108` — `diff = |current_fp - last_fp|.mean()`, compared against
    `similarity_threshold` default `0.01` (`__init__.py:64`). A 10×10-pixel object in a 640×640 frame
    moves that mean by ~1e-4, three orders of magnitude below the threshold — a guaranteed hit.
  - `src/yowo/cache/_similarity.py:9-28` — a correct full-pixel L1 `frame_similarity()` exists and is
    **not used** by `check_and_load`.
  - Enabled by preset for video on the two most common accelerated configs:
    `src/yowo/config.py:740-744` (`CUDA_HIGH`/`VIDEO`, `cache=True`) and
    `src/yowo/config.py:770-774` (`APPLE_SILICON`/`VIDEO`, `cache=True`).
  - `README.md:342-344` — *"Skip backbone + neck on similar consecutive frames — 60–85% compute
    savings"* with no accuracy warning. `docs/user-guide.md:1068-1084` likewise.
- **Why it blocks:** This is a recall-destroying optimization, on by default in a preset, sold as
  free. For any safety, security, or counting application a missed detection is the failure mode
  that matters, and nothing in the repo measures it.
- **Smallest fix:** Replace the channel-mean fingerprint with a spatially-blocked difference (e.g.
  the existing `frame_similarity` on a 16×16 downsample, max over blocks rather than mean) so a
  localized change cannot hide; drop `cache=True` from `_PRESET_TABLE` until a measured mAP delta on
  a moving-object dataset exists; document the trade-off at the call site.

### D6. The benchmark and tune sweep attribute results to backends that never ran

- **Symptom:** `yowo benchmark --format onnx,tensorrt` and `yowo tune` label rows with the *requested*
  backend, while the engine's silent fallback chain may have run PyTorch. For OBB the requested
  backend is not even passed to the engine.
- **Evidence:**
  - `src/yowo/benchmark/_runner.py:152-159` — the OBB branch constructs `OBBEngine(...)` **without**
    `backend=backend_type`, although `OBBEngine.__init__` accepts it (`src/yowo/obb_engine.py:101`).
    Every OBB row is the auto-selected backend, labeled otherwise.
  - `src/yowo/benchmark/_runner.py:259-262` — `format=backend_type.value` is the *requested* backend.
    The actual one (`engine.selection.backend`) is available on the same object but only
    `selection.device_type` is read (`_runner.py:170`).
  - The fallback that makes this wrong: `src/yowo/engine.py:455-495` — on `BackendLoadError` the
    engine walks `_FALLBACK_CHAIN` (`src/yowo/backends/_selector.py:52-58`) and rewrites
    `self._selection`, logging a WARNING. `run_benchmark` never supplies a `.onnx`/`.engine` path
    (`src/yowo/benchmark/__init__.py:86,106-115`), so `resolve_weights()` hands the ONNX/TRT backend
    a `.pt` — which `OnnxBackend.load` cannot open — and the row silently becomes PyTorch.
  - Same defect in tuning: `src/yowo/tune/_sweep.py:187-221` builds the engine with `backend=backend`
    and the same fallback applies; the measured FPS is then persisted as
    `TuneProfile(backend="tensorrt", …)` and **automatically applied to production configs** at
    `src/yowo/engine.py:169-187`.
  - The sweep also measures FPS on `np.zeros((640,640,3))` black frames
    (`src/yowo/tune/_sweep.py:223-225`) — zero detections, so NMS cost is excluded from the number
    that picks the "optimal" batch size — and it sweeps `precision`, which per **D2** changes
    nothing, so the recorded `precision` is measurement noise.
- **Why it blocks:** These are the only two tools yowo gives an operator to answer "which backend and
  batch size should I ship?", and both can return an answer about a configuration that was never
  executed. A wrong answer is worse than no answer, and the tune answer is then applied silently.
- **Smallest fix:** Report `engine.selection.backend` (not the request) in `BenchmarkResult.format`
  and `SweepResult.backend`; pass `backend=backend_type` in the OBB branch; add a `strict=True` that
  makes an explicitly-requested backend a hard failure instead of a fallback (see D14); measure the
  sweep on real frames.

---

## P1 — before GA

### D7. The OOM ladder measures the wrong allocator and is dead on the GPU backends that matter

- **Symptom:** The three-tier OOM recovery never fires under TensorRT or ONNX-CUDA, and misfires
  under PyTorch.
- **Evidence:** `src/yowo/engine.py:598-605` polls `torch.cuda.memory_reserved(idx)`, the *PyTorch
  caching allocator* pool. `_start_oom_monitor` runs whenever `_is_cuda`
  (`engine.py:565-568`, keyed on `selection.device_type`), which is true for TRT and ONNX-CUDA — but
  those backends allocate through ORT/TRT, so `memory_reserved()` returns ~0 and `pct` never crosses
  `_OOM_TIER1` (`engine.py:132`). Under PyTorch the caching allocator legitimately holds a large
  reserved pool at steady state, so the monitor halves the batch size on a healthy process. It also
  polls every 5 s (`engine.py:596`), which cannot catch a per-batch OOM. Tier 2 is a no-op (D2).
- **Smallest fix:** Read `torch.cuda.mem_get_info(idx)` (device-wide free/total, allocator-agnostic)
  instead of `memory_reserved`, and gate tier-1 on an actual caught `OutOfMemoryError` rather than a
  5-second poll.

### D8. A persistent backend failure is converted into "no objects", and that fallback then crashes YOLO11 postprocess

- **Symptom:** After 3 retries `_infer_with_retry` returns a zero-sized array so "the engine does not
  crash". For YOLO26 this surfaces as a frame with zero detections — a detector reporting an empty
  scene during an outage. For YOLO11 it raises `ValueError` inside postprocess instead.
- **Evidence:** `src/yowo/engine.py:815` returns `np.zeros((1, 0, 6))`. Postprocess orients that to
  `(6, 0)` at `src/yowo/postprocess/_nms.py:485-487` (`cols > rows`), then
  `_nms.py:275-276` does `raw[:, 4:].max(axis=1)` on a zero-length axis. VERIFIED:

  ```
  uv run python -c "
  import numpy as np
  item = np.zeros((1,0,6), np.float32)[0]; rows,cols = item.shape
  if cols>rows: item = item.T
  item[:,4:].max(axis=1)"
  → ValueError: zero-size array to reduction operation maximum which has no identity
  ```

  The retry loop also sleeps 100/200/400 ms while holding `_infer_lock` (`engine.py:830-836`),
  stalling every other stream, and retries CUDA OOM without `empty_cache()`.
- **Smallest fix:** Do not synthesize a fake output. Raise a typed `InferenceError` after exhaustion
  and let the caller decide; if a degraded mode is wanted, emit an explicit `Detection` with a
  `degraded=True` marker so downstream can distinguish "nothing there" from "we could not look".

### D9. Thread and memory limits read the host, not the container

- **Symptom:** In a Kubernetes pod with `cpu: 2` on a 64-core node, ORT is configured with 32
  intra-op threads and PyTorch with 32; the model is CFS-throttled and latency collapses. Memory
  presets are sized from host RAM.
- **Evidence:** `os.cpu_count()` at `src/yowo/backends/_tensorrt.py:354-356`,
  `src/yowo/backends/_onnx.py:335-337`, `src/yowo/backends/_pytorch.py:138-144`. No
  `os.sched_getaffinity`, no `/sys/fs/cgroup/cpu.max`, no `OMP_NUM_THREADS` handling anywhere
  (`grep -rn "sched_getaffinity\|cgroup\|OMP_NUM_THREADS" src/yowo/` → no matches).
  `src/yowo/hardware/_detect.py:72-84` reads `/proc/meminfo`, which reports the host in a container.
  The tune fingerprint inherits the same bug at `src/yowo/tune/_profile.py:93` — a profile calibrated
  on the 64-core host is a fingerprint match inside the 2-core container and is applied silently.
- **Smallest fix:** A single `yowo.utils.available_cpus()` that prefers
  `len(os.sched_getaffinity(0))` and the cgroup v2 `cpu.max` quota, used by all three backends and
  the fingerprint; likewise a cgroup-aware `memory.max` read in `detect_system_memory_mb`.

### D10. Postprocess latency is unbounded in scene content — no pre-NMS top-k, no `max_det`

- **Symptom:** p99 frame latency is a function of how crowded the scene is, with no ceiling. There is
  no candidate cap before NMS and no detection cap after it.
- **Evidence:** `src/yowo/postprocess/_nms.py:274-321` filters by confidence only; every surviving
  anchor goes into `cv2.dnn.NMSBoxes` (`_nms.py:177-182`) and then into a per-box Python
  `BoundingBox` construction loop (`_nms.py:307-321`, and `_nms.py:369-383` for YOLO26).
  Reference implementations cap candidates (~30k) and `max_det` (~300); yowo does neither, so a
  miscalibrated `num_classes`, a low `--confidence`, or a dense scene walks all 8400 anchors through.
- **MEASURED** (macOS arm64, YOLO11n, PyTorch CPU; scripts in session log):

  | stage | cost |
  |---|---|
  | preprocess (1080p → 640) | 1.23 ms (3.1%) |
  | `backend.infer()` | 38.12 ms (96.5%) |
  | postprocess, 0 boxes | 0.15 ms (0.4%) |
  | postprocess, 859 kept of 3000 candidates | 4.91 ms |
  | `cv2.dnn.NMSBoxes` alone, 3000 overlapping candidates | 15.42 ms |
  | Python `BoundingBox` loop, 2956 boxes | 3.05 ms (~1 µs/box) |

  **On the repo's prior claim:** `backend.infer()` = 86-97% of frame time holds *only* for a slow
  backend on an empty scene — the 96.5% above is PyTorch CPU with zero detections. The same 4.9 ms
  of postprocess is ~48% of a 5 ms CoreML inference (README claims 4-5× over PyTorch) and exceeds a
  3 ms TensorRT inference outright. The remaining overhead sits in exactly two places: `cv2` NMS,
  which grows superlinearly with overlapping candidates, and the per-detection Python object loop.
- **Smallest fix:** Add `max_nms` (top-k by score before NMS) and `max_det` (truncate after) to
  `postprocess()`, plumbed from `InferenceConfig`. Optionally build `BoundingBox` objects lazily.

### D11. `BatchScheduler`'s flush timeout is only evaluated when a frame arrives

- **Symptom:** A partial batch can sit indefinitely. With a 1 fps camera and `max_batch_size=4`, every
  batch waits ~4 s regardless of `timeout_ms=50`; with all streams momentarily stalled the wait is
  unbounded. The documented contract is not met.
- **Evidence:** `src/yowo/pipeline/_scheduler.py:120-141` — `next(self._iter)` blocks first
  (line 121); the elapsed check at 139-141 is only reachable *after* a frame arrives. Upstream,
  `FrameCollector.__iter__` blocks on `self._shared_q.get(timeout=5.0)` and `continue`s on timeout
  (`src/yowo/pipeline/_collector.py:314-334`), so it never yields a "nothing yet" signal. The
  docstring at `_scheduler.py:30-31` promises the opposite.
- **Smallest fix:** Give the collector a non-blocking/timed `poll(timeout)` and have the scheduler
  compute its wait as `remaining = timeout_s - elapsed`, flushing on expiry.

### D12. Metrics measure `backend.infer()` only — there is no end-to-end latency

- **Symptom:** The p99 an operator alerts on excludes preprocess, NMS, batch-assembly wait, and queue
  time. Under D11 the dashboard stays green while frames age in the scheduler.
- **Evidence:** `src/yowo/engine.py:832-835` — `record_inference(elapsed_ms, …)` is called inside
  `_run_gpu`, timing only `_infer_with_retry`. Preprocess (`engine.py:869-875`) and postprocess
  (`engine.py:841-847`, deliberately outside the lock) are untimed. `EngineMetrics`
  (`src/yowo/metrics/_collector.py:32-43`) exposes no queue depth, no end-to-end histogram, and no
  GPU-memory gauge.
- **Smallest fix:** Stamp a monotonic timestamp on `Frame` at read time and record
  `now - frame.t_captured` into a second histogram exported as `yowo_frame_latency_ms`.

### D13. Tune profiles silently rewrite backend, batch size and precision at DEBUG level

- **Symptom:** The same code and the same `InferenceConfig` produce different backends and batch
  sizes depending on a YAML file in `~/.cache/yowo/profiles`. The only record is a `logger.debug`.
- **Evidence:** `src/yowo/engine.py:161-187` — `_load_tune_profile` applies `backend`, `batch_size`
  and `precision` whenever the config fields are at defaults, logging at DEBUG (`engine.py:180`).
  The fingerprint (`src/yowo/tune/_profile.py:87-95`) is `sha256("{gpu.name}|{vram}|{cuda_version}")`
  — no driver version, no TensorRT/ORT/torch version, no yowo version, no profile schema version. A
  TensorRT uninstall or a driver upgrade leaves the profile a valid match.
- **Smallest fix:** Log the applied profile at INFO with the fingerprint and `tuned_at`; add
  `schema_version` plus TRT/ORT/torch/driver versions to the fingerprint input; provide
  `YOWO_DISABLE_TUNE_PROFILE=1`.

### D14. Precision selection depends on transient free VRAM, and an explicit backend can silently degrade

- **Symptom:** Two identical processes started minutes apart choose different precisions. A user who
  explicitly demands TensorRT for an SLA can end up on PyTorch CPU.
- **Evidence:** `src/yowo/backends/_selector.py:185` uses `gpu.memory_available_mb` (free, not total)
  against the thresholds at `_selector.py:37-43`. `select_precision` never consults `GPUArch` at all,
  so FP16 is chosen on hardware whose capability was never checked. On the fallback side,
  `src/yowo/engine.py:455-495` applies `_FALLBACK_CHAIN` even when the user passed an explicit
  `backend=`; `_user_provided_backend` (`engine.py:449`) only covers a supplied backend *instance*.
- **Smallest fix:** Key precision on `memory_total_mb` (deterministic), and add
  `InferenceConfig(strict_backend=True)` that raises instead of falling back.

### D15. `GPUArch` is a six-entry whitelist — new and old NVIDIA hardware reads as UNKNOWN

- **Symptom:** Blackwell (sm_100/sm_120), Volta (7.0), Pascal (6.1) and Jetson Xavier (7.2) all map to
  `GPUArch.UNKNOWN`. The detection design is still a table of special cases, not a rule.
- **Evidence:** `src/yowo/hardware/_detect.py:153-168` — `_CC_TO_ARCH` covers exactly
  `{(7,5),(8,0),(8,6),(8,7),(8,9),(9,0)}`. Also `src/yowo/hardware/_detect.py:130-146`: Jetson
  detection relies on `/etc/nv_tegra_release` and `/proc/device-tree/model`, neither of which is
  necessarily visible inside a container on a Jetson — so `classify_device`
  (`src/yowo/config.py:689-690`) picks the wrong preset there. Credit where due: the
  `[N/A]`-memory and old-driver `compute_cap` fixes (`_detect.py:255-271`, `312-325`) are careful and
  well-commented — the *probing* is now robust; the *interpretation* is not.
- **Smallest fix:** Express capability as a rule over `(major, minor)` (FP16 tensor cores ≥ 7.0, INT8
  DP4A ≥ 6.1) and keep the name table for logging only, so unknown silicon degrades to a correct
  capability instead of UNKNOWN.

### D16. INT8 TensorRT calibrator sizes its device buffer from the first batch

- **Symptom:** If the first calibration chunk is short (unreadable images are skipped) and a later one
  is full, `memcpy_htod` copies more bytes than were allocated.
- **Evidence:** `src/yowo/export/_int8.py:96-98` allocates `cuda.mem_alloc(batch.nbytes)` once, on the
  first batch; `src/yowo/export/_calibration.py:100-107` silently drops unreadable images, so batch
  sizes are ragged. TensorRT also expects exactly `get_batch_size()` images per call
  (`_int8.py:83-84`), so a short batch feeds stale device memory into calibration.
- **Smallest fix:** Allocate `batch_size * 3 * input_size**2 * 4` bytes up front and pad or drop
  short batches rather than yielding them.

### D17. CoreML and OpenVINO silently ignore the requested precision at export time

- **Evidence:** `src/yowo/export/_exporter.py:470` — `ct.precision.FLOAT16 if precision == FP16 else
  FLOAT32`, so `--format coreml --precision int8` produces FP32 with no warning.
  `src/yowo/export/_exporter.py:409-422` — `_convert_openvino` takes no `precision` argument at all.
- **Smallest fix:** Raise `ConfigError` for unsupported format/precision combinations instead of
  silently downgrading.

### D18. `yowo batch` ignores `--workers` and runs batch size 1; video results are buffered entirely in RAM

- **Evidence:** `src/yowo/batch/_runner.py:180` and `:222` call `engine.detect([frame])` — one frame
  at a time. `BatchConfig.workers` (`_runner.py:80`) is never referenced in `run_batch`
  (`grep -n "workers" src/yowo/batch/_runner.py` → only the dataclass field and its docstring); the
  CLI does set `pipeline_workers` on the engine (`src/yowo/cli/_main.py:1334`) but `detect()` never
  takes the pipeline path. `_runner.py:216-236` accumulates every frame's detections in
  `frame_detections` and writes one JSONL row per video — a 1-hour 30 fps file holds 108k dicts.
- **Smallest fix:** Route videos through `engine.stream(VideoFileSource(...))` (which honours
  `batch_size` and `pipeline_workers`) and write one JSONL row per frame as it is produced.

### D19. The Python API default for `dynamic_batch` makes TensorRT export unbuildable

- **Symptom:** `export_model(spec, ExportFormat.TENSORRT, out)` exports a dynamic-batch ONNX and then
  builds a TRT engine with **no optimization profile**. TensorRT requires at least one profile when
  the network has a dynamic input dimension; the build returns `None` and yowo raises
  `ExportError("TensorRT engine build returned None")`. The CLI happens to default the other way, so
  the two entry points disagree.
- **Evidence:** `src/yowo/export/_exporter.py:27` (`dynamic_batch: bool = True`) and `:233`
  (`dynamic_axes = {"images": {0: "batch"}, …}`) vs `src/yowo/export/_exporter.py:383-401`, which
  creates a `BuilderConfig`, sets only a 1 GB workspace, and never calls
  `builder.create_optimization_profile()` / `config.add_optimization_profile()`. The CLI default is
  `--no-dynamic-batch` (`src/yowo/cli/_main.py:469`), contradicting the docstring at
  `_exporter.py:44-45` which claims dynamic is the default. *(Code-level finding — no NVIDIA GPU on
  this host, so not executed.)*
- **Related runtime hazard:** even via the CLI path, `TensorRTBackend` sets no shape profile on the
  TRT EP (`_tensorrt.py:141-154`), so a batch size the engine has not seen triggers an in-line engine
  rebuild — a multi-second stall mid-stream. `_halve_batch_size` (`engine.py:657-666`) can trigger
  exactly that during a memory event.
- **Smallest fix:** Add min/opt/max optimization profiles in `_convert_tensorrt` derived from a new
  `batch_sizes` argument, and set `trt_profile_min_shapes`/`opt`/`max` in the EP options.

### D20. `_halve_batch_size` races the inference thread's two-step buffer read

- **Evidence:** `src/yowo/engine.py:869-873` reads `self._preprocess_buf.capacity`, then separately
  passes `self._preprocess_buf` to `preprocess_into`. The OOM daemon rebinds that attribute to a
  smaller buffer at `engine.py:662-666`. Between the two reads the swap yields
  `ValueError: batch size N exceeds buffer capacity` from `_decode.py:274-275` — an extra failure
  precisely during a memory event. Under free-threaded Python (`is_free_threaded()`,
  `engine.py:519`) the GIL-atomicity assumption weakens further. Cosmetic but telling: the recovery
  log at `engine.py:667-671` hardcodes `0.0` so it always reads `"GPU 0.0%"`.
- **Smallest fix:** Snapshot `buf = self._preprocess_buf` once into a local and use it for both the
  capacity check and the call.

---

## P2 — later

- **D21.** The live/RTSP path — the flagship edge case — never uses the pre-allocated preprocess
  buffer: `src/yowo/_streaming.py:74` binds `functools.partial(preprocess, …)`, so every frame pays a
  fresh `copyMakeBorder`. Only the offline path gets `preprocess_into`
  (`_streaming.py:159`, `engine.py:871`). The optimization is applied where it matters least.
- **D22.** The output-orientation heuristic `if cols > rows: item = item.T`
  (`src/yowo/postprocess/_nms.py:485-487`) breaks for a custom model where `num_anchors <
  4 + num_classes` (e.g. 160×160 input with 1000 classes) — relevant now that `num_classes` is
  user-settable. Prefer deciding from `4 + num_classes` explicitly.
- **D23.** Boxes are inverse-letterboxed **and clipped to the frame** *before* NMS
  (`_nms.py:300-305`), whereas reference implementations run NMS in model space and clip afterwards.
  Two boxes extending past the same edge get clipped to identical coordinates and over-suppress. A
  small, silent divergence from the mAP numbers the model was published with.
- **D24.** `FeatureStore` is bounded by *entry count* (32), not bytes
  (`src/yowo/cache/_store.py:48-64`). For `yolo11x` the P3/P4/P5 triple is ~15 MB, so the cache
  ceiling is ~480 MB with no way to express a byte budget.
- **D25.** On Jetson, when nvidia-smi reports `[N/A]` memory the code substitutes **total system RAM**
  as the GPU budget (`src/yowo/hardware/_detect.py:342-352`). Correct in spirit for unified memory,
  but that number then feeds `_VRAM_THRESHOLDS` and `classify_device`'s 8 GB split
  (`src/yowo/config.py:680,697-700`) as if it were dedicated VRAM.
- **D26.** `_PRESET_TABLE` (`src/yowo/config.py:737-795`) is keyed only on device × source category —
  `yolo11x` gets the same `batch_size=4` as `yolo11n` on CUDA_HIGH/VIDEO.
- **D27.** `yowo batch` writes files named `*_annotated.*` that contain the **unannotated** frame:
  `src/yowo/batch/_runner.py:190` writes `frame_arr`, `:231` writes `frame.pixels`; no drawing call
  anywhere in the module.
- **D28.** No deployment/ops documentation: nothing in `README.md` or `docs/user-guide.md` mentions
  ONNX external data, TensorRT engine portability across driver/TRT versions, or the need to
  regenerate engines (`grep -rn "onnx.data\|portab\|driver" README.md docs/user-guide.md` → no
  matches). The multi-stream feature-cache disable *is* documented
  (`docs/user-guide.md:1242`, `src/yowo/pipeline/__init__.py:79-82`) — that one is fine.
- **D29.** `yowo.__version__` reports `2.4.1` while `pyproject.toml:3` says `2.5.0`, and that stale
  value is written into every export sidecar as provenance
  (`src/yowo/export/_exporter.py:199`). VERIFIED in the sidecar produced above.
- **D30.** `torch>=2.0` is unbounded (`pyproject.toml:31-33`) while `torch.onnx.export`'s `dynamo`
  parameter now defaults to `True` (VERIFIED on the installed torch 2.10.0), so the exported graph
  changes shape with the installed torch and nothing records which one was used. Pin an upper bound
  or pass `dynamo=` explicitly, and record `torch_version` in the sidecar (see D3).

---

## What an operator still cannot answer

> *"What latency will I get on an Orin Nano at 1080p with yolo11s INT8, and what mAP do I give up?"*

Nothing in the repo can answer either half.

- **Latency:** `yowo benchmark` has no `--precision` and no `--weights`, so it cannot point at an
  exported INT8 engine (`src/yowo/cli/_main.py:48-64`); it always measures batch 1
  (`_runner.py:191-210`); its `fps_avg` is `1/mean_latency` (`_runner.py:227-228`), not achievable
  pipeline throughput; and per D6 the row may name a backend that never ran. `yowo tune` measures
  black frames. Every published number in `README.md:791-797` is Apple Silicon; there is not one
  CUDA or Jetson figure in the repo, and the pipeline/tracking numbers cite `tmp/bench_*.py` scripts
  that are gitignored (`.gitignore:368`) and therefore unreproducible. The `bench/` suite *is*
  reproducible (`bench/README.md`) but covers only arch/ONNX latency — no mAP, no INT8, no TRT.
- **Accuracy:** no FP32↔INT8 comparison exists, no gate exists, and per D4 the calibration is
  mis-preprocessed to begin with.

**The harness that would close this** is small and mostly reuses parts that already exist:

1. `yowo benchmark --weights <artifact> --precision <p>` — let `run_single_backend` take a concrete
   artifact path and report `engine.selection.backend`, so an exported `.engine`/`.onnx` can be
   measured on the target device.
2. `yowo verify <artifact> --data coco/val2017 --baseline <fp32-artifact> --max-map-drop 0.02` —
   runs both through the existing `evaluate_coco_map` (`src/yowo/benchmark/_evaluator.py`) and exits
   non-zero on regression. This is the accuracy gate D4 asks for and it is ~50 lines on top of what
   is already there.
3. A batch-size × resolution latency sweep that reports p50/p95/p99 **end-to-end** (D12) on real
   frames, emitting a per-device card: `orin-nano / yolo11s / int8 / 1080p → p99 X ms, Y FPS,
   mAP50-95 Z (-0.4 vs fp32)`. That card, stored next to the artifact as part of `ExportMetadata`
   (D3), is the deliverable that makes yowo predictable.

---

## What is already good

Do not re-do this work:

- **Preprocessing** is genuinely tight: one `cv2.resize`, one `copyMakeBorder`, and a single
  `cv2.dnn.blobFromImages` for BGR→RGB + HWC→BCHW + normalize (`src/yowo/io/_decode.py:235-248`);
  `PreprocessBuffer` skips the memset when consecutive frames share a resolution
  (`_decode.py:83-89`, `300-303`). Measured at 1.23 ms for 1080p→640.
- **Postprocess allocation discipline** is careful: in-place `out=` ufuncs in `_inverse_letterbox`
  (`_nms.py:239-241`), scratch buffer with a correct over-capacity fallback rather than truncation
  (`_nms.py:129-137`), the class-offset NMS trick instead of per-class Python loops
  (`_nms.py:165-182`), and deterministic result ordering (`_nms.py:191-198`).
- **Warmup validation** catches a corrupt or mismatched model before the engine goes READY, including
  NaN/Inf checks (`src/yowo/engine.py:531-564`, `1013-1018`). This is the right instinct and most
  libraries skip it.
- **Streaming I/O** is solid: bounded deque with three explicit drop policies
  (`src/yowo/io/_reader.py:150-172`), periodic RTSP reconnect to work around the OpenCV capture leak
  (`_reader.py:185-206`), exponential-backoff reconnect in `RTSPStreamSource`
  (`src/yowo/io/_source.py:274-286`).
- **Multi-stream backpressure** is real and bounded: `_put_or_stop` with a stop-aware retry
  (`src/yowo/pipeline/_collector.py:62-77`), dead-bridge detection so `__iter__` terminates
  (`_collector.py:317-334`), at-most-one pending future in both overlap paths
  (`src/yowo/pipeline/__init__.py:148-161`, `src/yowo/_streaming.py:189-191`), and positional-index
  routing (`src/yowo/pipeline/_router.py:105-108`).
- **The concurrency hazard around the shared postprocess scratch buffer was correctly avoided** —
  `_stream_pipeline` passes `scratch=None` on the concurrent path (`src/yowo/_streaming.py:169`)
  while the sequential path reuses the buffer. That is the right call and easy to get wrong.
- **Metrics plumbing** is well built even if it measures too little: lock-free rolling histogram with
  p50/p95/p99 and a Prometheus exporter (`src/yowo/metrics/_collector.py:56-83`, `216-280`).
- **Hardware probing** has clearly been hardened by real bugs: `[N/A]` memory columns, old drivers
  rejecting `compute_cap`, and NVML-blind Tegra builds are all handled with good comments
  (`src/yowo/hardware/_detect.py:255-271`, `290-325`, `342-352`).
- **`bench/`** is a real, documented, reproducible suite with pinned flags and a "Reproducibility"
  section (`bench/README.md`) — extend it rather than replacing it.
