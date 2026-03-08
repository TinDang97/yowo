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
- [x] **Phase 3: Adaptive Optimization and Batch Processing** - Auto-tune per device, add runtime adaptation, and enable high-throughput offline batch processing (completed 2026-03-07)
- [x] **Phase 4: OBB Detection** - Add oriented bounding box detection as a new inference task with export support (completed 2026-03-08)
- [x] **Phase 5: Integration Bug Fixes** - Close 3 integration gaps found by audit: P0 runtime crash (OBB+kv_cache), P1 OBBEngine tune profile auto-load, P2 missing public API exports (completed 2026-03-08)
- [x] **Phase 6: OBB Integration Fixes** - Close 2 functional integration gaps found by final audit: tune profile key collision (OBBEngine loads wrong profile), OBB models rejected by benchmark module (completed 2026-03-08)
- [x] **Phase 7: OBB Tune Sweep Dispatch Fix** - Close INT-C1: tune/_sweep.py._measure_config dispatches DetectionEngine for all tasks; OBB sweep fails at warmup and saves no profile (completed 2026-03-08)

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
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md — OBBBox/OBBDetection types + OBBHead + dist2rbox + probiou NMS + OBBModel + OBB registry (OBB-01, OBB-02, OBB-03, OBB-04)
- [ ] 04-02-PLAN.md — OBBEngine(BaseEngine) + OBBConfig + load_obb_weights engine task branch (OBB-03, OBB-04)
- [ ] 04-03-PLAN.md — parse_model_name -obb extension + detect-obb CLI + export_model OBB branch + human verification (OBB-05, OBB-06)

### Phase 5: Integration Bug Fixes
**Goal**: Close all integration gaps identified by the v2.3 milestone audit — runtime crash, missing tune profile propagation to OBBEngine, and incomplete public API surface
**Depends on**: Phase 4
**Requirements**: OBB-01, OBB-02, OBB-06, CORR-07, CORR-08, RELY-05, STRM-01, TUNE-01
**Gap Closure:** Closes INT-P0, INT-P1, INT-P2 from v2.3-MILESTONE-AUDIT.md
**Success Criteria** (what must be TRUE):
  1. `yowo export --model yolo11n-obb --kv-cache` does not crash (guard excludes task=obb)
  2. OBBEngine auto-loads a saved tune profile on construction, same as DetectionEngine
  3. Users can write `from yowo import OBBBox, OBBDetection, WarmupValidationError, HealthReport, StreamConfig` without submodule imports
  4. Test coverage exists for OBB+kv_cache export path (regression test for INT-P0)
**Plans**: 1 plan

Plans:
- [ ] 05-01-PLAN.md — Fix OBB kv_cache guard, add OBBEngine tune profile load, add 5 missing `__init__` exports, add regression test

### Phase 6: OBB Integration Fixes
**Goal**: Close the two functional integration gaps discovered by the final v2.3 audit — tune profile key collision (OBBEngine silently loads a detection-calibrated profile) and OBB model variants being rejected by the benchmark module
**Depends on**: Phase 4, Phase 5
**Requirements**: TUNE-01, OBB-03, BENCH-01, OBB-01
**Gap Closure:** Closes INT-A1, INT-A2, FLOW-A1 from v2.3-MILESTONE-AUDIT.md
**Success Criteria** (what must be TRUE):
  1. `yowo tune --model yolo11n-obb` saves a profile under key `"yolo11n-obb"` and OBBEngine auto-loads it on next construction (not the detection-calibrated `"yolo11n"` profile)
  2. `yowo benchmark --model yolo11n-obb` runs without `ValueError` and produces mAP + FPS results for the OBB variant
  3. Regression tests confirm the profile key includes the task suffix and the benchmark pattern accepts `-obb` suffix
**Plans**: 1 plan

Plans:
- [ ] 06-01-PLAN.md — Fix tune profile key collision in engine.py:161 + extend _MODEL_PATTERN for OBB + add OBB benchmark evaluation path

### Phase 7: OBB Tune Sweep Dispatch Fix
**Goal**: OBB models can be auto-tuned via `yowo tune --model yolo11n-obb` — sweep produces results, profile is saved, and OBBEngine auto-loads it on subsequent runs
**Depends on**: Phase 6
**Requirements**: TUNE-01, TUNE-02, TUNE-03 (OBB path)
**Gap Closure:** Closes INT-C1, FLOW-C1 from v2.3-MILESTONE-AUDIT.md
**Success Criteria** (what must be TRUE):
  1. `yowo tune --model yolo11n-obb` completes without warmup validation error and saves an OBB-keyed tune profile
  2. OBBEngine auto-loads the saved OBB tune profile on next construction
  3. Regression test confirms `_measure_config` dispatches OBBEngine for task=obb, DetectionEngine for task=detect
**Plans**: 1 plan

Plans:
- [ ] 07-01-PLAN.md — Add OBB dispatch branch in tune/_sweep.py._measure_config + regression tests

## Progress

**Execution Order:**
Phases execute in numeric order. Phase 4 (OBB) depends only on Phase 1 and can proceed in parallel with Phases 2-3.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Correctness, Benchmarking, and Proactive Fixes | 2/3 | Executing | - |
| 2. Reliability and Multi-Stream Scaling | 3/3 | Complete   | 2026-03-07 |
| 3. Adaptive Optimization and Batch Processing | 5/5 | Complete   | 2026-03-07 |
| 4. OBB Detection | 3/3 | Complete   | 2026-03-08 |
| 5. Integration Bug Fixes | 1/1 | Complete   | 2026-03-08 |
| 6. OBB Integration Fixes | 1/1 | Complete   | 2026-03-08 |
| 7. OBB Tune Sweep Dispatch Fix | 1/1 | Complete   | 2026-03-08 |
