# Codebase Concerns

**Analysis Date:** 2026-03-07

## Tech Debt

**StreamingMixin relies on 59 `type: ignore[attr-defined]` comments:**
- Issue: `StreamingMixin` in `src/yowo/_streaming.py` accesses parent class attributes (`_backend`, `_batch_size`, `_model_meta`, `_shutting_down`, `_shutdown_lock`, `_active_streams`, `_streams_drained`, `_prefetch`, `_metrics`, `_postprocess_buf`, `_auto_letterbox`, `_max_queue_size`, `_frame_drop_policy`, `_pipeline_workers`) without any type-safe protocol or abstract base. Every access is suppressed with `# type: ignore[attr-defined]`.
- Files: `src/yowo/_streaming.py` (59 suppressions)
- Impact: Zero type-checking coverage on the mixin's attribute access. Renaming any attribute in `BaseEngine` silently breaks `StreamingMixin` at runtime with no static analysis warning.
- Fix approach: Define a `Protocol` (e.g., `_EngineProtocol`) that declares the required attributes, and type `self: _EngineProtocol` in each mixin method. Alternatively, convert `StreamingMixin` to use explicit constructor injection of the shared state.

**Counter module uses `type: ignore[attr-defined]` for structural typing:**
- Issue: `ObjectCounter` and `box_center()` access `.x1`, `.y1`, `.x2`, `.y2`, `.class_name`, `.track_id`, `.frame`, `.boxes` via `# type: ignore[attr-defined]` on plain `object` parameters.
- Files: `src/yowo/counter/_counter.py` (6 suppressions), `src/yowo/counter/_geometry.py` (4 suppressions)
- Impact: No compile-time checking that passed objects have the expected attributes. A refactor of `BoundingBox` or `Detection` fields would not produce type errors.
- Fix approach: Define a `BoundingBoxLike` protocol with `x1`, `y1`, `x2`, `y2` and a `DetectionLike` protocol with `frame`, `boxes`. Use these as parameter types instead of `object`.

**Pyright strict mode partially undermined by blanket suppressions:**
- Issue: `pyproject.toml` sets `typeCheckingMode = "strict"` but then disables 5 `reportUnknown*` categories: `reportUnknownMemberType`, `reportUnknownVariableType`, `reportUnknownArgumentType`, `reportUnknownParameterType`, `reportMissingTypeArgument`.
- Files: `pyproject.toml` (lines 90-94)
- Impact: "Strict mode" gives a false sense of type safety. Unknown types from untyped libraries (onnxruntime, openvino, pynvml, tqdm) propagate unchecked through the codebase.
- Fix approach: Instead of global suppressions, add per-file `# pyright: reportUnknown*=false` only in files that directly import untyped libraries (backends, hardware probes). Re-enable the global checks.

**Files exceeding or approaching 700-line limit:**
- Issue: Three files exceed the project's 700-line soft limit.
- Files: `src/yowo/cli/_main.py` (743 lines), `src/yowo/config.py` (725 lines), `src/yowo/engine.py` (675 lines)
- Impact: `_main.py` and `config.py` are already over 700 lines; `engine.py` is close. Large files are harder to navigate and review.
- Fix approach: Split `_main.py` by extracting helper functions (`_write_json`, `_load_zones`, `_load_lines`, `_parse_model_spec`, `_parse_cls_model_spec`) into `src/yowo/cli/_helpers.py`. Split `config.py` by extracting preset logic (`_PresetOverrides`, `_classify_device`, `preset_config`) into `src/yowo/config_presets.py`.

## Security Considerations

**`torch.load` with `weights_only=False` allows arbitrary code execution:**
- Risk: Loading untrusted `.pt` files can execute arbitrary Python code via pickle deserialization. An attacker who supplies a crafted `.pt` file can achieve RCE.
- Files: `src/yowo/arch/_weights.py:85`
- Current mitigation: Weights are downloaded from known GitHub release URLs. Users can supply custom paths via `--weights` CLI flag.
- Recommendations: Add a warning in documentation about untrusted weight files. Consider `weights_only=True` with a fallback for ultralytics checkpoint format, or use `torch.load(..., weights_only=True)` with `torch.serialization.add_safe_globals()` for known types. At minimum, validate file hash against known checksums for official weights.

**RTSP URL passed directly to OpenCV without validation:**
- Risk: `RTSPStreamSource` passes user-supplied URLs directly to `cv2.VideoCapture()`. While OpenCV's RTSP handler is relatively safe, malformed or malicious URLs could trigger unexpected behavior in ffmpeg backends.
- Files: `src/yowo/io/_source.py:309`
- Current mitigation: URL scheme checking exists upstream in config validation (RTSP_SCHEMES).
- Recommendations: Validate URL format before passing to OpenCV. Consider restricting to `rtsp://` and `rtsps://` schemes only at the source level.

## Performance Bottlenecks

**Event callback failures silently swallowed:**
- Problem: The `EventBus._dispatch_loop` catches all exceptions from event callbacks and silently discards them with a bare `pass`.
- Files: `src/yowo/events/__init__.py:263-264`
- Cause: Designed to prevent a misbehaving callback from crashing the dispatch thread.
- Improvement path: Log the exception at `WARNING` or `DEBUG` level instead of swallowing silently. This preserves fault isolation while making debugging possible. Example: `logger.debug("Event callback %s raised", cb.__name__, exc_info=True)`.

**RTSP reconnect uses blocking `time.sleep()` in iterator:**
- Problem: `RTSPStreamSource.__iter__` calls `time.sleep(wait)` during reconnect, blocking the calling thread entirely.
- Files: `src/yowo/io/_source.py:350`
- Cause: Simple exponential backoff implementation.
- Improvement path: Use a `threading.Event.wait(timeout=wait)` pattern that can be interrupted by the engine shutdown mechanism, or push reconnect logic into the `ThreadedFrameReader`.

## Fragile Areas

**StreamingMixin + BaseEngine coupling:**
- Files: `src/yowo/_streaming.py`, `src/yowo/engine.py`
- Why fragile: `StreamingMixin` is tightly coupled to `BaseEngine` internals via 15+ private attributes accessed without any interface contract. The mixin was extracted purely for file-size reasons (documented in the module docstring), not for architectural separation.
- Safe modification: When modifying `BaseEngine` attributes, grep `_streaming.py` for every attribute name. Run the full test suite — there are no compile-time guards.
- Test coverage: Covered via `tests/unit/test_streaming.py` and `tests/unit/test_engine.py`, but only at the integration level through the public API.

**Engine close() swallows backend unload errors:**
- Files: `src/yowo/engine.py:370-374`
- Why fragile: `close()` catches and silently ignores all exceptions from `self._backend.unload()`. If a backend leaks resources (GPU memory, file handles), the error is invisible.
- Safe modification: Add `logger.debug("Backend unload failed", exc_info=True)` inside the except block.
- Test coverage: `tests/unit/test_shutdown.py` tests the happy path. No tests verify behavior when `unload()` raises.

**Batch inference error path in `_run_batch`:**
- Files: `src/yowo/engine.py:500-504`
- Why fragile: The except block at line 502 calls `self._metrics.record_error()` then re-raises, but does not clean up the preprocess buffer or tensor. If `_infer_from_tensor` fails partway through, the `_postprocess_buf` scratch buffer may be in an inconsistent state for the next call.
- Safe modification: Wrap buffer cleanup in a `finally` block.
- Test coverage: Error paths in batch inference have limited coverage.

## Scaling Limits

**EventBus bounded queue (256 slots):**
- Current capacity: `EventBus` uses a `threading.Semaphore(256)` to bound the dispatch queue.
- Limit: High-frequency detection streams (e.g., 30 FPS with many events per frame) can saturate the queue, causing `emit()` to block the inference thread.
- Scaling path: Make the queue size configurable via `EventBus(max_pending=N)`. Consider dropping events with a warning instead of blocking.

**Feature map cache is per-source, memory-unbounded:**
- Current capacity: `FeatureMapCache` in `src/yowo/cache/__init__.py` stores mmap'd numpy arrays per source_id.
- Limit: With many concurrent sources (multi-stream pipeline), cache entries grow linearly with source count and are only evicted on fingerprint mismatch.
- Scaling path: Add a maximum cache entry count or total memory budget.

## Dependencies at Risk

**`opencv-python-headless` version floor is very low (>=4.8):**
- Risk: Wide version range means different users may hit different OpenCV bugs. Some cv2 type stubs are missing or incorrect (reflected by `type: ignore[assignment]` on `cv2.dnn.blobFromImages` calls).
- Impact: `src/yowo/io/_decode.py`, `src/yowo/postprocess/_nms.py` both have type-ignore comments specifically for cv2 stub issues.
- Migration plan: Pin a narrower range (e.g., `>=4.9,<5.0`) to reduce surface area. Consider using numpy-native operations for blob construction to reduce cv2 dependency.

**`pyyaml` is a hard dependency but only used in config loading:**
- Risk: `pyyaml` is a C-extension package that can cause installation issues on some platforms (Alpine, musl-based containers).
- Impact: `src/yowo/config.py` is the sole consumer (for YAML config file loading).
- Migration plan: Make YAML config loading optional (fall back to env-var-only config) or use `tomllib` (stdlib in 3.11+) for config files.

## Test Coverage Gaps

**No integration tests for RTSP/webcam sources:**
- What's not tested: `RTSPStreamSource` and `WebcamSource` in `src/yowo/io/_source.py` require live hardware/network. The reconnect logic, idle timeout, and frame skip behavior are untested in CI.
- Files: `src/yowo/io/_source.py` (lines 274-438)
- Risk: Reconnect timeout logic has a subtle bug potential — the deadline is reset on each disconnect but the sleep could exceed the deadline between check and sleep.
- Priority: Medium — these are edge/production paths that are hard to test without mocks.

**Backend error paths during `close()`:**
- What's not tested: No test verifies behavior when `backend.unload()` raises an exception during engine shutdown.
- Files: `src/yowo/engine.py:370-374`
- Risk: Resource leaks (GPU memory, file handles) would go unnoticed.
- Priority: Low — the except-and-pass pattern is intentional, but logging should be added.

**EventBus silent exception swallowing:**
- What's not tested: No test verifies that a failing callback does not crash the dispatch thread or affect other callbacks.
- Files: `src/yowo/events/__init__.py:257-264`
- Risk: A callback that raises could theoretically affect dispatch ordering, but current implementation is safe. The real risk is that debugging is impossible without logging.
- Priority: Low.

**Pipeline worker thread error propagation:**
- What's not tested: In `_stream_pipeline`, if `_infer_batch` raises inside a `ThreadPoolExecutor` future, the exception propagates via `future.result()`. But if the future is cancelled (line 198-199), the exception is lost.
- Files: `src/yowo/_streaming.py:153-176`, `src/yowo/_streaming.py:198-199`
- Risk: Silent data loss during pipeline shutdown — cancelled futures may have partially processed frames.
- Priority: Medium.

## Missing Critical Features

**No rate limiting on weight downloads:**
- Problem: `_download_weights` in `src/yowo/models/_weights.py` retries up to 3 times with backoff but has no rate limiting across multiple concurrent engine instances. Multiple engines starting simultaneously could hammer the GitHub CDN.
- Blocks: Safe multi-process deployment without pre-downloaded weights.

**No health check endpoint for production deployments:**
- Problem: `HealthStatus` enum exists in types, and the engine tracks health state internally, but there is no HTTP health endpoint or readiness probe for container orchestrators (Kubernetes, ECS).
- Blocks: Production deployment behind a load balancer without custom wrapper code.

---

*Concerns audit: 2026-03-07*
