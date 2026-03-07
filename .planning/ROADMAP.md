# Roadmap: YOWO Production Hardening

## Overview

YOWO v2.3.0 is a mature inference library with native YOLO architectures, 5 backends, streaming pipelines, tracking, and cross-camera ReID. This milestone bridges the gap between "works in unit tests" and "production-ready on real devices under sustained load." The roadmap follows a strict dependency chain: prove correctness first, then scale with reliability guarantees, then add adaptive optimization. OBB detection is an independent vertical that can proceed in parallel.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Correctness, Benchmarking, and Proactive Fixes** - Validate inference accuracy vs ultralytics, build benchmark tooling, and fix known production issues before scaling
- [x] **Phase 2: Reliability and Multi-Stream Scaling** - Harden engine for sustained production load with 100+ streams, memory guards, health checks, and observability (completed 2026-03-07)
- [ ] **Phase 3: Adaptive Optimization and Batch Processing** - Auto-tune per device, add runtime adaptation, and enable high-throughput offline batch processing
- [ ] **Phase 4: OBB Detection** - Add oriented bounding box detection as a new inference task with export support

## Phase Details

### Phase 1: Correctness, Benchmarking, and Proactive Fixes
**Goal**: Users can trust that YOWO produces correct results matching ultralytics, measure performance across all backends, and avoid known production pitfalls
**Depends on**: Nothing (first phase)
**Requirements**: CORR-01, CORR-02, CORR-03, CORR-04, CORR-05, CORR-06, CORR-07, CORR-08, BENCH-01, BENCH-02, BENCH-03, BENCH-04, PFIX-01, PFIX-02, PFIX-03, PFIX-04, PFIX-05
**Success Criteria** (what must be TRUE):
  1. Running `yowo benchmark --model yolo11n` produces a comparison table showing mAP, FPS, and model size per export format on current hardware, and mAP matches ultralytics within 0.5% for all detection and classification variants
  2. ONNX, TensorRT, and OpenVINO exported models produce correct inference results when loaded and run on their respective target devices (CUDA server, Jetson, Intel NUC), with export accuracy delta < 1% mAP vs PyTorch baseline
  3. Engine warmup during load() validates output shape and value range, rejecting bad models before accepting inference requests
  4. RTSP streaming does not leak memory over sustained operation, concurrent engine access from multiple threads is safe, and NMS output ordering is deterministic across runs
  5. Backend unavailability produces clear error messages with install instructions instead of cryptic import errors, and export compatibility constraints are documented and tested
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Proactive fixes: warmup validation, thread safety, deterministic NMS, RTSP reconnect, graceful errors, export compat CLI
- [ ] 01-02-PLAN.md — Benchmark evaluation module: mAP via pycocotools, FPS measurement, rich table rendering, JSON output
- [ ] 01-03-PLAN.md — Benchmark CLI subcommand: wire module into `yowo benchmark` with all flags + human verification

### Phase 2: Reliability and Multi-Stream Scaling
**Goal**: Users can deploy YOWO in sustained production environments with 100+ concurrent streams, automatic failure recovery, and full operational visibility
**Depends on**: Phase 1
**Requirements**: STRM-01, STRM-02, STRM-03, STRM-04, STRM-05, RELY-01, RELY-02, RELY-03, RELY-04, RELY-05
**Success Criteria** (what must be TRUE):
  1. Multi-stream pipeline handles 100+ concurrent RTSP/video streams on CUDA server without crashing, with frame drop rate below 5% and per-stream memory overhead bounded and documented
  2. Engine detects approaching OOM and reduces batch size or precision before crash, and recovers from transient GPU errors without requiring full restart
  3. Running `yowo health` or calling the health check API returns engine status (ready, degraded, unhealthy) with structured JSON logging available at configurable log levels
  4. No memory leaks over 24-hour sustained operation with 50+ streams, and individual stream disconnection does not affect other streams
  5. Metrics (latency, throughput, memory, errors) are exportable in Prometheus-compatible or JSON format for production monitoring
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Multi-stream isolation: _StreamEntry per-stream stats, auto-remove on 3 consecutive errors, StreamConfig, memory-leak tests
- [ ] 02-02-PLAN.md — OOM monitor daemon + GPU error retry: three-tier recovery ladder, _infer_with_retry, InferenceConfig log_level/structured_logging fields
- [ ] 02-03-PLAN.md — Observability: HealthReport + health_report(), JsonFormatter, Prometheus/JSON metrics export, yowo health + yowo metrics CLI

### Phase 3: Adaptive Optimization and Batch Processing
**Goal**: Users can auto-tune YOWO for their specific hardware without manual configuration and process large offline datasets at maximum throughput
**Depends on**: Phase 2
**Requirements**: TUNE-01, TUNE-02, TUNE-03, TUNE-04, BATC-01, BATC-02, BATC-03, BATC-04
**Success Criteria** (what must be TRUE):
  1. Running `yowo tune --model yolo11n` auto-detects optimal backend, batch size, and precision for current hardware via calibration sweep, without OOMing during calibration
  2. Auto-tune results persist to a device-specific profile file and subsequent runs start instantly using the cached profile
  3. Running `yowo batch SOURCE_DIR --model yolo11n` processes a directory of images/videos at maximum GPU utilization with larger batch sizes than streaming mode
  4. Batch processing supports resume from checkpoint on interruption and reports progress (processed/total, ETA, throughput)
**Plans**: 5 plans

Plans:
- [ ] 03-01-PLAN.md — TuneProfile dataclass, YAML persistence, device fingerprint, test scaffolds for all plans
- [ ] 03-02-PLAN.md — Calibration sweep: SweepResult, run_sweep(), OOM guard, backend × batch × precision enumeration
- [ ] 03-03-PLAN.md — Batch runner: BatchConfig, run_batch(), atomic checkpoint, rich progress, JSONL output
- [ ] 03-04-PLAN.md — Engine profile auto-load integration + yowo tune CLI subcommand
- [ ] 03-05-PLAN.md — yowo batch CLI subcommand + human verification checkpoint

### Phase 4: OBB Detection
**Goal**: Users can perform oriented bounding box detection with YOWO using the same workflow as standard detection
**Depends on**: Phase 1
**Requirements**: OBB-01, OBB-02, OBB-03, OBB-04, OBB-05, OBB-06
**Success Criteria** (what must be TRUE):
  1. OBB detection produces oriented bounding boxes with rotation angle, matching ultralytics OBB architecture for yolo11 family, and loads weights from ultralytics-trained checkpoints
  2. OBB postprocessing uses rotation-aware NMS that correctly handles overlapping rotated boxes
  3. Running `yowo detect-obb SOURCE --model yolo11n-obb` works end-to-end from CLI, and OBB models export to ONNX and TensorRT correctly
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order. Phase 4 (OBB) depends only on Phase 1 and can proceed in parallel with Phases 2-3.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Correctness, Benchmarking, and Proactive Fixes | 2/3 | Executing | - |
| 2. Reliability and Multi-Stream Scaling | 3/3 | Complete   | 2026-03-07 |
| 3. Adaptive Optimization and Batch Processing | 2/5 | In Progress|  |
| 4. OBB Detection | 0/2 | Not started | - |
