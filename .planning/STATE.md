---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: milestone
status: completed
stopped_at: Completed 07-01-PLAN.md (INT-C1 OBB tune sweep dispatch fix)
last_updated: "2026-03-08T10:43:40.538Z"
last_activity: 2026-03-08 -- Completed 04-03 OBB CLI and export pipeline; human verify checkpoint approved
progress:
  total_phases: 7
  completed_phases: 7
  total_plans: 17
  completed_plans: 17
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-07)

**Core value:** Inference that is production-ready out of the box -- deploy to any supported device and it works correctly, fast, and reliably under sustained real-world load without manual tuning.
**Current focus:** Phase 1 COMPLETE. Next: Phase 2 (OBB / Scale)

## Current Position

Phase: 4 of 4 (OBB Detection)
Plan: 3 of 3 in phase (04-01, 04-02, 04-03 complete — Phase 4 DONE)
Status: Complete
Last activity: 2026-03-08 -- Completed 04-03 OBB CLI and export pipeline; human verify checkpoint approved

Progress: [██████████] 100% (Phase 4 OBB Detection COMPLETE)

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: ~27min
- Total execution time: ~1.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 3     | ~80min | ~27min  |

**Recent Trend:**
- Last 5 plans: 01-01 (25min), 01-02 (20min), 01-03 (35min), 02-01 (45min), 02-02 (24min)
- Trend: steady

*Updated after each plan completion*

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 3     | ~80min | ~27min  |
| 02    | 2     | ~69min | ~35min  |
| Phase 02-reliability-and-multi-stream-scaling P03 | 8 | 2 tasks | 7 files |
| Phase 03 P01 | 12 | 2 tasks | 7 files |
| Phase 03-adaptive-optimization-and-batch-processing P02 | 11 | 2 tasks | 4 files |
| Phase 03 P03 | 7 | 2 tasks | 3 files |
| Phase 03-adaptive-optimization-and-batch-processing P04 | 30 | 2 tasks | 4 files |
| Phase 03 P05 | 15 | 1 tasks | 2 files |
| Phase 04-obb-detection P02 | 8 | 1 tasks | 8 files |
| Phase 04-obb-detection P03 | 7 | 1 tasks | 5 files |
| Phase 05-integration-bug-fixes P01 | 11 | 3 tasks | 6 files |
| Phase 06-obb-integration-fixes P01 | 9 | 3 tasks | 6 files |
| Phase 07-obb-tune-sweep-dispatch-fix P01 | 4 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Correctness before scale -- validate mAP parity and export correctness before any scaling work
- Roadmap: OBB is parallel -- depends only on Phase 1, can proceed alongside Phases 2-3
- Roadmap: PFIX bundled with Phase 1 -- proactive fixes (RTSP leak, thread safety, deterministic NMS) belong with correctness since they affect production reliability baseline
- 01-01: threading.Lock (not RLock) for _infer_lock -- linear inference path, no reentrant risk
- 01-01: Warmup validation with dummy zeros tensor, subclass _validate_output_values() pattern
- 01-01: RTSP reconnect via duck-typing (getattr/callable) for protocol compatibility
- 01-02: Module-level imports in _runner.py for test mockability over local imports
- 01-02: YOLO_TO_COCO as immutable tuple, pycocotools stdout suppressed during eval
- 01-02: pycocotools + rich as optional [benchmark] dependency group in pyproject.toml
- 01-03: CLI validates --data path existence; shows dataset-type-specific download instructions based on -cls suffix
- 01-03: Missing optional deps raise click.UsageError with uv add yowo[benchmark] instruction
- 01-03: CORR-04/05/06 deferred to real-device sessions; CLI tooling ships, hardware validation deferred
- 01-03: pycocotools imgIds must be restricted to predicted images (not full val set) for correct mAP
- [Phase 02-01]: Auto-remove sentinel pattern: bridge sets auto_remove flag, iterator calls remove_stream() to avoid self-join deadlock
- [Phase 02-01]: _auto_removed_errors dict preserves error visibility after stream removal; stream_errors merges both dicts
- [Phase 02-01]: StreamConfig reconnect fields stubbed for API stability; only max_consecutive_errors is active in this plan
- [Phase 02-02]: OOM monitor uses Event.wait(5.0) not time.sleep(5.0) for fast shutdown
- [Phase 02-02]: _halve_batch_size reallocates PreprocessBuffer to prevent oversized batch corruption
- [Phase 02-02]: _infer_with_retry returns zeros on exhaustion (no raise) — engine stays alive per RELY-02
- [Phase 02-02]: log_level validated with frozenset class variable in InferenceConfig/ClassificationConfig
- [Phase 02-reliability-and-multi-stream-scaling]: HealthReport defined in engine.py (not types.py) to avoid circular imports with HealthStatus
- [Phase 02-reliability-and-multi-stream-scaling]: configure_logging uses logger.handlers.clear() for guaranteed idempotency
- [Phase 02-reliability-and-multi-stream-scaling]: yowo health exits 2 (unhealthy/closed) when invoked standalone; yowo metrics always exits 0
- [Phase 03-01]: TuneProfile uses plain frozen dataclass (not slots=True) to allow dataclasses.asdict() YAML serialization
- [Phase 03-01]: Device fingerprint uses GPU VRAM bytes (mb*1024*1024) for bit-level reproducibility; catches TypeError in load_profile for empty-file edge case
- [Phase 03-01]: Atomic profile write: write to .tmp then os.replace — pattern established for all profile writes
- [Phase Phase 03-02]: check_backend_available promoted from private to public API in _selector.py and backends/__init__.py — sweep is a legitimate cross-module caller
- [Phase Phase 03-02]: except Exception + _is_oom() dispatch avoids B030 ruff error from tuple unpacking in except clauses
- [Phase Phase 03-02]: _measure_config uses DetectionEngine(config) via InferenceConfig(model_family=spec.family,...) — plan interface comment was incorrect about DetectionEngine(spec, config)
- [Phase 03]: cv2 imported at module level for test patchability; VideoFileSource used for video iteration instead of ThreadedFrameReader (simpler offline API); atomic checkpoint write pattern reused from tune profile
- [Phase 03]: _hw_cache parameter on BaseEngine.__init__ avoids double get_hardware_profile() call when DetectionEngine pre-computes hw for tune profile lookup
- [Phase 03]: Module-level imports in cli/_main.py for get_hardware_profile/run_sweep/load_profile/save_profile/compute_fingerprint for test patchability at yowo.cli._main.*
- [Phase 03]: count_sweep_dimensions() added as public function to tune/_sweep.py to avoid pyright reportPrivateUsage errors when calling from _main.py
- [Phase 03-05]: Module-level imports in cli/_main.py for run_batch/BatchConfig/DetectionEngine for test patchability
- [Phase 03-05]: sys.exit(result) propagates integer exit codes 0/1/2 from run_batch through Click
- [Phase 04-01]: OBBHead inherits Detect; reuses stride/anchor cache init, DFL, cv2/cv3 — only cv4 angle branch is added
- [Phase 04-01]: dist2rbox placed in _heads.py alongside dist2bbox for co-location; angle encoding (sigmoid-0.25)*pi applied once in OBBHead.forward()
- [Phase 04-01]: OBB registry is YOLO11-only (nc=15 DOTA v1); YOLO26 has no OBB weights at v8.4.0
- [Phase 04-01]: probiou_matrix matches ultralytics batch_probiou exactly via Bhattacharyya distance between Gaussians
- [Phase 04-02]: load_obb_weights reuses _LAYER_MAP — model.23.* -> head.* prefix covers cv4 angle branches automatically
- [Phase 04-02]: OBBEngine skips feature cache and kv_cache (offline DOTA inference, same as ClassificationEngine pattern)
- [Phase 04-02]: _resolve_model_meta now uses if/elif/else for classify/obb/detection task branches
- [Phase 04-03]: OBBEngine/OBBConfig imported at module level in cli/_main.py for test patchability (consistent with Phase 03-05 batch CLI pattern)
- [Phase 04-03]: model_stem uses task_suffix variable: -obb for OBB task to prevent ONNX filename collision with detection exports
- [Phase 04-03]: yolo26*-obb guard fires at parse_model_name time with ConfigError, not silently at registry lookup
- [Phase 05-integration-bug-fixes]: INT-P0: tuple membership guard spec.task not in ('classify', 'obb') instead of chained != conditions
- [Phase 05-integration-bug-fixes]: INT-P1: cast(Any, cfg) bridges OBBConfig/InferenceConfig type mismatch at _load_tune_profile; # type: ignore[reportPrivateUsage] for cross-module private import
- [Phase 05-integration-bug-fixes]: INT-P2: HealthReport imported from yowo.engine (not types.py) to avoid circular import; 5 types added to __all__ in sorted order
- [Phase 06-obb-integration-fixes]: module-level import of load_dota_dataset in benchmark/__init__.py for patch() patchability; lazy OBBEngine import in _runner.py for test isolation
- [Phase 06-obb-integration-fixes]: model_key derived from spec in tune_command (task suffix appended) used for profile storage; raw model string kept only for display
- [Phase 07-01]: Patch target for lazy function-scope imports is the source module (yowo.obb_engine.OBBEngine), not importing module — plan spec had incorrect reasoning about module __dict__ binding

### Pending Todos

- CORR-04/05/06: Run `yowo benchmark --format onnx/trt/openvino` on CUDA server / Jetson / Intel NUC when device access available
- Establish COCO val-set ultralytics baseline mAP before Phase 2 defines pass/fail thresholds

### Blockers/Concerns

- Real-device access (Jetson, NUC, GPU server) needed for CORR-04/05/06 validation (deferred, not blocking)

## Session Continuity

Last session: 2026-03-08T10:43:40.536Z
Stopped at: Completed 07-01-PLAN.md (INT-C1 OBB tune sweep dispatch fix)
Resume file: None
