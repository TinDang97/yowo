# Requirements: YOWO Production-Grade Inference Platform

**Defined:** 2026-03-07
**Core Value:** Inference that is production-ready out of the box -- deploy to any supported device and it works correctly, fast, and reliably under sustained real-world load without manual tuning.

## v1 Requirements

Requirements for production-hardening milestone. Each maps to roadmap phases.

### Correctness

- [x] **CORR-01**: Inference mAP matches ultralytics within 0.5% on COCO val2017 for all 10 detection variants (yolo11/26 n/s/m/l/x)
- [x] **CORR-02**: Classification top-1 accuracy matches ultralytics within 0.5% on ImageNet val for all classification variants
- [x] **CORR-03**: Per-frame inference latency is within 10% of ultralytics on same hardware for all backends
- [ ] **CORR-04**: ONNX exported models produce correct results on CUDA server (onnxruntime-gpu)
- [ ] **CORR-05**: TensorRT exported engines produce correct results on NVIDIA Jetson Orin/Xavier
- [ ] **CORR-06**: OpenVINO exported models produce correct results on Intel NUC with integrated GPU
- [x] **CORR-07**: Export accuracy delta vs PyTorch baseline is < 1% mAP for each format
- [x] **CORR-08**: Model warmup during load() validates output shape and value range before accepting inference requests

### Benchmark

- [x] **BENCH-01**: User can run `yowo benchmark --model MODEL` to get mAP + FPS + model size per export format on current hardware
- [x] **BENCH-02**: Benchmark results include comparison table: format, mAP, FPS, model size, device name
- [x] **BENCH-03**: Benchmark mode supports all backends (PyTorch, ONNX, TensorRT, OpenVINO)
- [x] **BENCH-04**: Benchmark results are persistable to JSON for cross-run comparison

### Streaming

- [x] **STRM-01**: Multi-stream pipeline handles 100+ concurrent RTSP/video streams without crashing on CUDA server
- [x] **STRM-02**: Per-stream memory overhead is bounded and documented per device
- [x] **STRM-03**: No memory leaks over 24-hour sustained operation with 50+ streams
- [x] **STRM-04**: Frame drop rate stays below 5% at target stream count per device
- [x] **STRM-05**: Pipeline gracefully handles individual stream disconnection without affecting other streams

### Reliability

- [x] **RELY-01**: Engine detects approaching OOM condition and reduces batch size or drops precision before crash
- [x] **RELY-02**: Engine recovers from transient GPU errors without full restart
- [x] **RELY-03**: Health check API reports engine status (ready, degraded, unhealthy) queryable via CLI or programmatic API
- [x] **RELY-04**: Structured JSON logging with configurable log levels for all engine operations
- [x] **RELY-05**: Metrics export (latency, throughput, memory, errors) in Prometheus-compatible format or JSON

### Auto-Tuning

- [ ] **TUNE-01**: User can run `yowo tune --model MODEL` to auto-detect optimal backend, batch size, and precision for current hardware
- [x] **TUNE-02**: Auto-tune runs calibration sweep across available backends and precision levels
- [x] **TUNE-03**: Auto-tune results persist to device-specific profile file for instant startup on subsequent runs
- [x] **TUNE-04**: Auto-tune respects device memory constraints (does not OOM during calibration)

### OBB Detection

- [ ] **OBB-01**: OBB detection head produces oriented bounding boxes with rotation angle
- [ ] **OBB-02**: OBB postprocessing includes rotation-aware NMS
- [ ] **OBB-03**: OBB model variants match ultralytics OBB architecture for yolo11 family
- [ ] **OBB-04**: OBB weights load correctly from ultralytics-trained checkpoints
- [ ] **OBB-05**: CLI supports `yowo detect-obb SOURCE --model yolo11n-obb`
- [ ] **OBB-06**: OBB models export to ONNX and TensorRT correctly

### Batch Processing

- [ ] **BATC-01**: User can run `yowo batch SOURCE_DIR --model MODEL` for offline high-throughput processing
- [ ] **BATC-02**: Batch mode maximizes GPU utilization with larger batch sizes than streaming mode
- [ ] **BATC-03**: Batch processing supports resume from checkpoint on interruption
- [ ] **BATC-04**: Batch mode reports progress (processed/total, ETA, throughput)

### Proactive Fixes (from ultralytics known issues)

- [x] **PFIX-01**: RTSP stream memory leak prevention -- OpenCV VideoCapture leaks ~1MB/2-3hrs per stream; implement bounded memory or alternative decode path
- [x] **PFIX-02**: Thread safety for concurrent engine access -- ultralytics model.predict() is not thread-safe; ensure YOWO engines are safe for multi-threaded use
- [x] **PFIX-03**: Deterministic NMS output ordering -- ultralytics NMS output order varies between runs; ensure consistent ordering for reproducible results
- [x] **PFIX-04**: Export format compatibility matrix -- document and test exact version compatibility (TensorRT version vs CUDA vs GPU arch) to prevent silent failures
- [x] **PFIX-05**: Graceful fallback when optional backend unavailable -- clear error messages with install instructions instead of cryptic import errors

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Edge Adaptation

- **EDGE-01**: Thermal-aware throttling detects device temperature and reduces throughput before thermal limit
- **EDGE-02**: Runtime load adaptation dynamically adjusts quality/speed tradeoff based on real-time metrics
- **EDGE-03**: Zero-config edge deployment: `yowo detect SOURCE --device auto` works optimally without config files

### Intelligence

- **INTL-01**: System learns from deployment metrics to improve auto-tuning decisions over time
- **INTL-02**: Cross-device benchmark comparison for fleet deployment decisions

### Advanced Pipeline

- **PIPE-01**: Multi-model ensemble inference (detector + classifier in single pipeline pass)
- **PIPE-02**: Export-and-validate pipeline (`yowo export --validate`)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training / fine-tuning | Inference-only platform; use ultralytics for training |
| Segmentation | Defer to dedicated future milestone |
| Pose estimation | Defer to dedicated future milestone |
| Web UI / dashboard | Library, not web app; expose metrics for Grafana |
| Kubernetes orchestrator | Separate product; provide health endpoints instead |
| gRPC/REST inference server | Use Triton for serving; YOWO is the inference engine |
| Dynamic model hot-swap | Fast cold restart via close() + load() is sufficient |
| Plugin/extension system | Protocol-based architecture already allows custom backends |
| Apple Silicon/CoreML testing | No Mac hardware available this milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORR-01 | Phase 1 | Complete |
| CORR-02 | Phase 1 | Complete |
| CORR-03 | Phase 1 | Complete |
| CORR-04 | Phase 1 | Pending |
| CORR-05 | Phase 1 | Pending |
| CORR-06 | Phase 1 | Pending |
| CORR-07 | Phase 1 | Complete |
| CORR-08 | Phase 1 | Complete |
| BENCH-01 | Phase 1 | Complete |
| BENCH-02 | Phase 1 | Complete |
| BENCH-03 | Phase 1 | Complete |
| BENCH-04 | Phase 1 | Complete |
| STRM-01 | Phase 2 | Complete |
| STRM-02 | Phase 2 | Complete |
| STRM-03 | Phase 2 | Complete |
| STRM-04 | Phase 2 | Complete |
| STRM-05 | Phase 2 | Complete |
| RELY-01 | Phase 2 | Complete |
| RELY-02 | Phase 2 | Complete |
| RELY-03 | Phase 2 | Complete |
| RELY-04 | Phase 2 | Complete |
| RELY-05 | Phase 2 | Complete |
| TUNE-01 | Phase 3 | Pending |
| TUNE-02 | Phase 3 | Complete |
| TUNE-03 | Phase 3 | Complete |
| TUNE-04 | Phase 3 | Complete |
| OBB-01 | Phase 4 | Pending |
| OBB-02 | Phase 4 | Pending |
| OBB-03 | Phase 4 | Pending |
| OBB-04 | Phase 4 | Pending |
| OBB-05 | Phase 4 | Pending |
| OBB-06 | Phase 4 | Pending |
| BATC-01 | Phase 3 | Pending |
| BATC-02 | Phase 3 | Pending |
| BATC-03 | Phase 3 | Pending |
| BATC-04 | Phase 3 | Pending |
| PFIX-01 | Phase 1 | Complete |
| PFIX-02 | Phase 1 | Complete |
| PFIX-03 | Phase 1 | Complete |
| PFIX-04 | Phase 1 | Complete |
| PFIX-05 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 41 total
- Mapped to phases: 41
- Unmapped: 0

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-07 after roadmap creation*
