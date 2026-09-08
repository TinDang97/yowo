# Runtime robustness, failure handling & observability review — yowo production readiness

## Verdict

**Not production-ready for unattended long-lived deployment.** The architecture is thoughtful —
bounded queues, explicit locks, an OOM ladder, a health enum, a retry wrapper, Prometheus export —
but the pieces that matter at 3am are either absent or non-functional. The single biggest gap is the
**live-source I/O boundary**: there is not one connect or read timeout anywhere in the codebase
(`grep CAP_PROP_OPEN_TIMEOUT|CAP_PROP_READ_TIMEOUT|OPENCV_FFMPEG` over `src/` and `tests/` returns
nothing), the advertised RTSP exponential-backoff reconnect is arithmetically dead code that can
never fire, and the periodic "anti-leak" reconnect leaks a `VideoCapture` every time it runs. A
24/7 camera deployment will wedge, leak, or silently stop — and the metrics will report it as
healthy, because the reader's frame-drop counter is never wired into `engine.metrics` and
`yowo_memory_utilization` is a hardcoded `0.0`.

Second-order: the project's own CLAUDE.md mandates "timeouts, retries, circuit breakers, rollback
strategy in IO request". Of those four, only *retries* exist (in two places), and one of them
manufactures a wrong-shaped result that crashes the classification engine.

---

## P0 — blocks launch

### R1. No connect or read timeout on any OpenCV capture; a half-open socket wedges a thread forever

- **Symptom:** A camera that stops sending but keeps the TCP connection half-open blocks
  `cap.read()` indefinitely. `_stream_live` notices 30 s of silence and breaks the stream
  (`_streaming.py:107-111`), but the reader thread is still parked inside `cap.read()`.
  `ThreadedFrameReader.stop()` joins for 2 s, then **drops the thread handle without checking
  `is_alive()`** — the thread, its ffmpeg context, and its socket are leaked permanently. Restart
  the stream and you leak another. Day nine, the box OOMs.
- **Evidence:**
  - `src/yowo/io/_source.py:310` — `cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)` with no
    `CAP_PROP_OPEN_TIMEOUT_MSEC` / `CAP_PROP_READ_TIMEOUT_MSEC` and no
    `OPENCV_FFMPEG_CAPTURE_OPTIONS` (`stimeout`).
  - `src/yowo/io/_source.py:217` (video file), `:409` and `:434` (webcam) — same, no timeouts.
  - `src/yowo/io/_reader.py:274-276` — `self._thread.join(timeout=2.0); self._thread = None`
    with no `is_alive()` check, no log, no metric.
  - `src/yowo/io/_source.py:376-381` — `RTSPStreamSource.close()` calls `cap.release()` on a
    capture another thread is blocked reading; OpenCV does not make this safe.
- **Why it blocks:** This is the canonical unattended-edge failure. It is unrecoverable in-process
  (no way to interrupt a blocked `cap.read()` from Python) and unobservable (`stop()` reports
  success). Thread + fd + ffmpeg-buffer leak accumulates until the process dies.
- **Smallest fix:** In `RTSPStreamSource._open_cap()` and `WebcamSource.__iter__`, set
  `cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ...)` and `cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, ...)`
  from a new `open_timeout_ms` / `read_timeout_ms` argument (defaults ~5 s / ~10 s), and for FFMPEG
  also set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|stimeout;5000000` before construction.
  In `ThreadedFrameReader.stop()`, after the join, `if self._thread.is_alive(): logger.error(...)`
  and keep the handle so the leak is at least visible.

### R2. RTSP "exponential-backoff reconnect" is dead code — the timeout can never fire, and one failed re-open kills the stream

- **Symptom:** The class docstring says "Reads an RTSP stream with exponential-backoff reconnect".
  In practice there is effectively **one** reconnect attempt, and it raises `SourceError` — not
  `SourceTimeoutError` — if the camera has not come back within ~1 s. A five-second camera reboot
  permanently terminates the stream.
- **Evidence:** `src/yowo/io/_source.py:342-354`

  ```python
  cap.release()
  deadline = time.monotonic() + self._reconnect_timeout_s   # recomputed EVERY disconnect
  wait = min(2**retry_count, 10)
  if time.monotonic() + wait > deadline:                    # ⇔  wait > reconnect_timeout_s
      raise SourceTimeoutError(...)
  time.sleep(wait)
  retry_count += 1
  cap = self._open_cap()                                    # raises SourceError, no retry
  ```

  Because `deadline` is recomputed inside the loop, the guard reduces to `wait > reconnect_timeout_s`.
  With the default `reconnect_timeout_s=30.0` and `wait = min(2**retry_count, 10) <= 10`, that
  condition is **never true** — `SourceTimeoutError` is unreachable (`src/yowo/errors.py:190-195`
  documents an exception that cannot be raised). Meanwhile `_open_cap()` at `:309-313` raises
  `SourceError` unconditionally when the re-open fails, so the loop never gets a second iteration.
- **Why it blocks:** The one resilience feature the streaming layer advertises does not work.
  Consumers of `engine.stream(rtsp_source)` get a hard exception on the first camera blip, and
  cannot distinguish "camera briefly rebooted" from "camera permanently gone" — `SourceError` is
  raised for both.
- **Smallest fix:** Hoist `deadline` out of the loop (compute once, on the first disconnect; reset
  on the first successful read). Wrap `self._open_cap()` in try/except so a failed re-open increments
  `retry_count` and loops instead of propagating, and raise `SourceTimeoutError` only when
  `time.monotonic() >= deadline`. Add a unit test that drives a fake capture through
  fail→fail→succeed and asserts the stream survives.

### R3. The periodic anti-leak reconnect leaks a `VideoCapture` every interval and leaves the iterator reading a released handle

- **Symptom:** Every 300 s the reader thread calls `RTSPStreamSource.reconnect()`, which releases
  the capture and stores a **new** one in `self._active_cap`. But the generator in `__iter__` holds
  the old capture in a *local* variable `cap` and keeps calling `cap.read()` on it. The freshly
  opened `new_cap` is never read from and never released — it is orphaned. The mechanism written to
  *prevent* an OpenCV memory leak *is* a memory leak: one leaked ffmpeg context + socket per stream
  per 5 minutes.
- **Evidence:**
  - `src/yowo/io/_source.py:316-317` — `cap = self._open_cap(); self._active_cap = cap` (local +
    attribute alias).
  - `src/yowo/io/_source.py:366-374` — `reconnect()` releases `self._active_cap` and rebinds only
    the attribute; the generator's local `cap` is now a released capture.
  - `src/yowo/io/_reader.py:174-175, 185-207` — `_maybe_reconnect()` invokes it from the reader loop
    every `reconnect_interval_sec` (default 300 s).
  - `src/yowo/io/_source.py:370-373` — if the re-open fails, the identical call is retried once and
    the result is stored **without checking `isOpened()`**, so a closed capture is installed silently.
- **Why it blocks:** ~12 leaked captures/hour/stream on a 24/7 deployment, plus the stream degrades
  into the broken reconnect path of R2 on the next `cap.read()` (which returns `False` on a released
  capture, then double-releases it at `:343`).
- **Smallest fix:** Make `__iter__` read through the attribute (`self._active_cap`) instead of a
  local, or have `reconnect()` return the new capture and have the reader loop hand it back. Add
  `if not new_cap.isOpened(): new_cap.release(); raise SourceError(...)` and let the caller's
  backoff handle it. A regression test asserting `_active_cap is cap` after `reconnect()` would have
  caught this.

### R4. RTSP credentials are leaked into exceptions, logs, results, and on-disk cache keys

- **Symptom:** `rtsp://admin:hunter2@10.0.0.5/stream` is embedded verbatim in error messages, in
  every `Detection.source_id`, in `Detection.to_dict()` JSON output, and in the feature-cache key.
  There is **no redaction helper anywhere in the package** (`grep -i "redact\|mask\|credential"` over
  `src/yowo` returns nothing).
- **Evidence:**
  - `src/yowo/io/_source.py:312` — `SourceError(f"Cannot open RTSP stream: {self._url}")`
  - `src/yowo/io/_source.py:349-351` — `SourceTimeoutError(f"RTSP stream {self._url} timed out ...")`
  - `src/yowo/io/_source.py:336` — `source_id=self._url` on every `Frame`, which flows into
    `Detection.to_dict()` (`src/yowo/types.py:314+`) and out through `write_json`
    (`src/yowo/io/_sink.py:18-38`).
  - `src/yowo/io/_reader.py:179` — `logger.error("ThreadedFrameReader error: %s", exc)` prints the
    message containing the URL.
  - `src/yowo/pipeline/_collector.py:192` and `src/yowo/pipeline/__init__.py:192-196` — stream
    exceptions are stringified into log lines and into a `RuntimeError` message.
  - `src/yowo/engine.py:832-834` — `source_id` is passed to `backend.set_source_id()`, and
    `src/yowo/cache/_store.py:178-180` hashes it into an on-disk directory name.
- **Why it blocks:** Camera passwords end up in log aggregators, JSON result files committed to
  object storage, and exception traces shipped to error trackers. That is a credential-disclosure
  incident, and it is unavoidable for any user of the RTSP source.
- **Smallest fix:** Add a `_redact(url: str) -> str` helper that rewrites the userinfo component to
  `***:***@` (`urllib.parse`), use it in every message, and set
  `source_id=_redact(self._url)` at `:336` so the credential never enters the data path at all.

### R5. Downloaded weights are never integrity-checked; a truncated download is cached permanently and then unpickled

- **Symptom:** `resolve_weights()` downloads a `.pt` (a Python pickle) over the network and
  `os.replace()`s it into the cache with **no checksum, no signature, and no content-length
  verification**. If the server closes the connection early, `iter_content` ends normally, the
  partial file is promoted to the cache, and every subsequent run short-circuits on
  `if dest.exists(): return dest` — the poisoned cache entry is permanent and there is no way to
  invalidate it short of manual `rm`.
- **Evidence:**
  - `src/yowo/models/_weights.py:113-146` — `_attempt_download` reads `content-length` at `:116`
    only to size the tqdm bar; the value is never compared against bytes written.
  - `src/yowo/models/_weights.py:86` — `os.replace(tmp_path, dest)` immediately after the loop.
  - `src/yowo/models/_weights.py:58-59` — `if dest.exists(): return dest` (no validation on the
    cache-hit path).
  - `src/yowo/models/_registry.py:37, 121, 139, 160` — `ModelMeta.default_weights_url` has **no
    companion `sha256` field**; the registry points at a third-party GitHub release
    (`_ASSETS_BASE = "https://github.com/ultralytics/assets/..."`).
- **Why it blocks:** Two production failures in one. (a) Availability: a flaky first download on an
  edge box bricks that model until someone SSHes in. (b) Supply chain: a `.pt` is arbitrary-code
  pickle; without a pinned hash, anything that can influence the response body (a compromised
  release asset, a corporate TLS-terminating proxy) gets code execution inside the inference process.
- **Smallest fix:** Add `sha256: str` to `ModelMeta`, populate it for the registered variants, and
  in `_download` verify the digest (and the byte count against `content-length` when present) *before*
  `os.replace`. On mismatch, unlink the tmp file and retry. Re-verify on the cache-hit path behind a
  cheap `YOWO_VERIFY_CACHE` toggle, or at minimum verify size.

### R6. The CLI accumulates every `Detection` — including full frame pixels — for unbounded live sources

- **Symptom:** `yowo detect rtsp://cam/stream` appends every result to a list that is never bounded
  and never used until the stream ends (which, for a live source, is never). Each `Detection` holds
  a `Frame`, which holds the full decoded BGR array. At 1080p that is ~6.2 MB per frame; at 25 fps
  the process grows ~155 MB/s and is OOM-killed within a couple of minutes.
- **Evidence:**
  - `src/yowo/cli/_main.py:268-271` — `detections = []` … `for det in engine.stream(src):
    detections.append(det)` inside the `detect` command.
  - `src/yowo/types.py:298` — `Detection.frame: Frame`; `src/yowo/io/_sink.py:57` reads
    `det.frame.pixels`, confirming the pixel buffer is retained.
  - Same pattern in the OBB path around `src/yowo/cli/_main.py:420-443`.
- **Why it blocks:** The primary documented entry point (`yowo detect <rtsp url>`) cannot run for
  more than a few minutes. Anyone evaluating the library against a camera hits this first.
- **Smallest fix:** Only accumulate when an `--output` file was requested *and* the source is not
  live (`src.is_live`); otherwise stream results out and drop the reference. For the file-output case
  on a live source, either stream JSONL incrementally or cap the buffer and warn.

---

## P1 — before GA

### R7. The retry fallback fabricates a detection-shaped result — silent wrong answers for detection, a raw crash for classification

- **Symptom:** When `backend.infer()` fails all retries, `_infer_with_retry` returns
  `np.zeros((1, 0, 6))` so "the engine does not crash". For `DetectionEngine` this is
  indistinguishable from a genuine "no objects in frame" — a broken model reports a clean, empty,
  100 %-healthy stream. For `ClassificationEngine` the shape is wrong and postprocessing raises a
  bare `ValueError`, so the graceful-degradation path becomes an unhandled crash.
- **Evidence:**
  - `src/yowo/engine.py:816` — `return np.zeros((1, 0, 6), dtype=np.float32)`, hardcoded to the
    detection layout, in `BaseEngine` (shared by all three engines).
  - Verified empirically (read-only):
    `uv run python -c "from yowo.postprocess._classify import postprocess_classify; ...
    postprocess_classify(np.zeros((1,0,6),np.float32), [frame], ...)"` →
    `ValueError: Expected 2-D output (batch, nc), got ndim=3`.
  - `src/yowo/obb_engine.py:195-199` indexes `output[:, 4:4+nc, :]` and would likewise mis-handle
    the sentinel.
- **Why it blocks:** "Silently return nothing" is the worst available failure mode for a detector
  — a monitoring system reports zero intrusions rather than an outage. And the escape hatch that
  was supposed to prevent crashes causes one on two of the three engines.
- **Smallest fix:** Make the fallback a `_process_batch`-level hook (`_empty_result(frames)`)
  overridden per engine, and **raise** `InferenceError` by default once a configurable
  consecutive-failure budget is exceeded rather than returning a fake result. At minimum, transition
  `_health_state` to `DEGRADED` and emit a distinct event on every fallback.

### R8. Retry sleeps hold the global inference lock; no jitter; permanent errors are retried

- **Symptom:** `_infer_with_retry` runs *inside* `self._infer_lock`, and it calls `time.sleep()`
  between attempts. A single failing batch stalls every other stream sharing the engine for up to
  300 ms. Retries fire on *any* exception, so a permanent configuration error (shape mismatch,
  missing class count) burns 300 ms per frame forever. There is no jitter, so N streams failing on
  the same camera outage retry in lockstep.
- **Evidence:**
  - `src/yowo/engine.py:830-838` — `with self._infer_lock:` … `self._infer_with_retry(tensor)`.
  - `src/yowo/engine.py:796, 806` — `except Exception as exc:` … `time.sleep(delay)`.
  - `src/yowo/engine.py:777` — `_RETRY_DELAYS = (0.1, 0.2, 0.4)`; the loop only sleeps on attempts
    0 and 1 (`if attempt < len(delays) - 1`), so `0.4` is never used and the documented "100 ms,
    200 ms, 400 ms" backoff is really 100 ms + 200 ms.
- **Smallest fix:** Move the sleep outside the lock (release, sleep, re-acquire), add
  `random.uniform(0, delay/2)` jitter, and restrict the retry to transient error types
  (`InferenceError`, CUDA OOM) — let `ConfigError` / `ValueError` propagate immediately.

### R9. Backend fallback leaks the failed backend; `__enter__` leaks on load failure

- **Symptom:** In the fallback ladder, when a backend fails during `load()` or `warmup()`, the
  engine simply constructs the next one and reassigns `self._backend` — the failed instance is never
  `unload()`ed. A TensorRT or CUDA backend that got as far as allocating a context leaks it. And
  `__enter__` calls `load()` with no `try`, so a `with InferenceEngine(...)` whose load fails never
  runs `__exit__`/`close()`, leaving whatever the backend allocated behind.
- **Evidence:**
  - `src/yowo/engine.py:465-472` — `self._backend = create_backend(bt, ...)` with no
    `self._backend.unload()` first.
  - `src/yowo/engine.py:490-491` — `except (BackendLoadError, BackendError): last_exc = exc` and
    straight to the next iteration.
  - `src/yowo/engine.py:716-718` — `def __enter__(self): self.load(); return self`.
  - Contrast `src/yowo/engine.py:499-501`, where `_finalize_load` *does* remember to
    `contextlib.suppress(Exception): self._backend.unload()` — proving the pattern is known.
- **Smallest fix:** Wrap the per-backend attempt body in try/except that calls
  `contextlib.suppress(Exception): self._backend.unload()` before moving on, and wrap `__enter__`'s
  `load()` in try/except that calls `self.close()` and re-raises.

### R10. `close()` swallows unload errors and can release the backend while a suspended generator still holds it

- **Symptom:** Three problems in one method. (a) `except Exception: pass` around
  `self._backend.unload()` — a failure to free GPU memory is completely invisible. (b) A stream
  generator that the consumer stopped iterating (rather than closing) never runs its `finally`, so
  its `_stop` event stays in `_active_streams`, `_streams_drained` is never set, `close()` burns the
  full 5 s timeout and then **unloads the backend anyway**. (c) If the consumer later resumes that
  generator, it calls `backend.infer()` on an unloaded backend — undefined behaviour, and a likely
  segfault for TensorRT/CUDA rather than a Python exception.
- **Evidence:**
  - `src/yowo/engine.py:703-707` — `try: ... self._backend.unload() except Exception: pass`.
  - `src/yowo/engine.py:699-702` — timeout-bounded `self._streams_drained.wait(...)` with no check
    of the result and no log when it expires.
  - `src/yowo/_streaming.py:49-58, 112-117, 197-204, 225-229` — the `_active_streams.discard` lives
    in generator `finally` blocks, which only run on close/GC.
- **Smallest fix:** Log the unload exception at ERROR (it is the one place that reports a GPU-memory
  leak). Check the return value of `_streams_drained.wait()` and log a warning naming the count of
  undrained streams. Guard the hot path with `if not self._loaded: raise ShutdownError(...)` inside
  `_infer_from_tensor`, so a resumed generator gets a clean library error rather than a segfault.

### R11. Frame drops are counted but never surfaced — an operator cannot answer "why did it drop frames"

- **Symptom:** The default live-stream policy is `FrameDropPolicy.LATEST` with
  `max_queue_size=2`, so dropping is the *normal* operating mode. `ThreadedFrameReader` diligently
  counts every drop — and then the reader is a local variable in `_stream_live`, discarded when the
  stream ends. `engine.metrics.frames_dropped` only ever counts the *async* queue overflow.
  `frames_dropped` is not in the Prometheus export at all.
- **Evidence:**
  - `src/yowo/io/_reader.py:152, 157, 164, 270` — the increments; `:306-316` — `frames_dropped` /
    `drop_rate` properties with no consumer anywhere (`grep` shows the only readers are the
    reader's own tests).
  - `src/yowo/_streaming.py:75-81` — `reader = ThreadedFrameReader(...)` is never exposed.
  - `src/yowo/metrics/_collector.py:189-193` — `record_frame_dropped()` is called from exactly one
    place, `src/yowo/_async.py:42`.
  - `src/yowo/metrics/_collector.py:237-278` — the Prometheus exposition emits seven metrics;
    `frames_dropped` and `events_dropped` are not among them.
  - `src/yowo/pipeline/_collector.py:57-59, 130-134` — `_StreamEntry.frames_dropped`,
    `frames_processed`, and `last_frame_time` are maintained but `FrameCollector` exposes no
    accessor for any of them (only `stream_states`, `stream_errors`, `stream_count`, `active_count`).
- **Smallest fix:** Have `_stream_live` / `_stream_pipeline` push the reader's drop delta into
  `self._metrics.record_frame_dropped()`, add `yowo_frames_dropped_total` and
  `yowo_events_dropped_total` to `export_prometheus`, and add a `stream_stats` property to
  `FrameCollector` returning the per-stream counters it already collects.

### R12. `yowo_memory_utilization` is a hardcoded `0.0`

- **Symptom:** The Prometheus gauge an operator would alert on for GPU pressure is a literal zero.
  A dashboard built on it is a lie, and an alert on it will never fire — including during the exact
  OOM ladder the engine implements.
- **Evidence:** `src/yowo/metrics/_collector.py:273-278` — `_metric("yowo_memory_utilization",
  "gauge", "...", 0.0)`. The real value is computed in `HealthReport` (`src/yowo/engine.py:381-392`)
  but never plumbed into the collector.
- **Smallest fix:** Give `MetricsCollector` an optional `memory_pct_fn: Callable[[], float | None]`
  set by `BaseEngine._finalize_load()`, or move `export_prometheus` onto the engine so it can read
  `health_report().memory_pct`. Omit the metric entirely when it is unknown rather than emitting 0.

### R13. Health status is a one-way latch on a cumulative counter, and false-alarms when idle

- **Symptom:** `errors_total` is monotonic for the process lifetime. Once ten errors have *ever*
  occurred, `engine.health` returns `DEGRADED` forever, even after hours of perfect operation — the
  only escape is `reset_metrics()`, which also wipes the latency histogram. Conversely, a healthy
  but idle engine (event-driven, no traffic for 30 s) is reported `DEGRADED`. Neither direction is
  actionable.
- **Evidence:** `src/yowo/engine.py:355-362`:
  ```python
  if self._metrics.errors_total >= self._error_threshold:   # cumulative, never decays
      return HealthStatus.DEGRADED
  last = self._metrics.last_frame_time
  if self._metrics.frames_total > 0 and last > 0 and time.monotonic() - last > 30.0:
      return HealthStatus.DEGRADED
  ```
- **Smallest fix:** Track errors in a time-windowed counter (e.g. a `deque` of error timestamps,
  same shape as `_RollingHistogram`) and compare the *recent* error rate to the threshold. Gate the
  staleness check on there being at least one active stream, so an idle engine reads `READY`.

### R14. Every event-listener exception is silently swallowed; `close()` abandons the worker after 2 s

- **Symptom:** A user's `on("detection", ...)` callback that raises is discarded with no log, no
  counter, and no event. The callback simply stops working and nothing anywhere records it.
  Separately, `EventBus.close()` joins for 2 s and returns regardless — the daemon worker can still
  be invoking callbacks against engine state while `close()` proceeds to `backend.unload()`.
- **Evidence:**
  - `src/yowo/events/__init__.py:256-263` — `for cb, ... in entries: try: ... except Exception: pass`.
  - `src/yowo/events/__init__.py:186-193` — `self._worker.join(timeout=timeout)` with no
    `is_alive()` check; called from `src/yowo/engine.py:702` immediately before the backend unload
    at `:705`.
- **Smallest fix:** `logger.exception("event listener for %s failed", event)` in the handler, plus a
  `listener_errors` counter next to `events_dropped`. In `close()`, log a warning when the worker
  fails to join, and perform the join *before* `backend.unload()` (it already is — but the abandoned
  case needs to be loud).

### R15. `astream()` converts a fatal stream error into a clean end-of-iteration

- **Symptom:** When the background thread's generator raises (broken RTSP, backend failure), the
  exception is emitted on the event bus and then the sentinel `None` is queued — so the consumer's
  `async for` simply *ends*. Unless the caller separately registered an `EVENT_ERROR` listener, an
  async consumer cannot tell a camera outage from a video reaching its last frame.
- **Evidence:** `src/yowo/_async.py:89-94`:
  ```python
  except Exception as exc:
      emit_fn("error", exc)
  finally:
      gen.close()
      ... q.put(None) ...
  ```
  Also `:109` — `thread.join(timeout=5.0)` with no `is_alive()` check, so a wedged reader thread is
  abandoned silently (compounding R1).
- **Smallest fix:** Store the exception and enqueue it as a sentinel; in the consumer loop, `raise`
  it instead of yielding. Keep the event emission for observers.

### R16. Pipeline streams are permanently removed on error — no reconnect, no circuit breaker

- **Symptom:** A stream whose reader raises three times in a row (which, given R2, is what a brief
  camera outage looks like) is flagged `auto_remove` and deleted from the collector. It is never
  retried and never re-added. In a 32-camera deployment, cameras silently disappear one by one over
  a week until `_check_stream_errors` finally raises "All pipeline streams failed". The
  `StreamState.RECONNECTING` state exists in the enum and is **never assigned anywhere** — the
  reconnect state machine was designed and not built.
- **Evidence:**
  - `src/yowo/pipeline/_collector.py:104-126` — the error/auto-remove branch; note the
    `continue` path has no sleep, so it spins as fast as `reader.get()` can re-raise the *same*
    stored `_error` (`src/yowo/io/_reader.py:257-258` re-raises and never clears `_error`).
  - `src/yowo/pipeline/_collector.py:342-359` — `remove_stream(stream_id)` on the iterator thread.
  - `grep -rn RECONNECTING src/` → only `types.py:128` (definition) and
    `_collector.py:401-406` (a membership test that can never match).
- **Smallest fix:** On auto-remove, instead of dropping the stream, transition it to `RECONNECTING`,
  close the reader, and re-`add_stream` behind an exponential backoff with a cap and a
  `max_reconnect_attempts` in `StreamConfig`. Expose the reconnect count so an operator can see a
  flapping camera.

### R17. A raising consumer callback tears down the entire multi-stream pipeline

- **Symptom:** `DetectionRouter.route()` invokes user callbacks with no exception guard. One
  camera's downstream handler throwing (a full disk, a closed socket) propagates up through
  `_run_overlapped` and kills inference for **all** streams.
- **Evidence:** `src/yowo/pipeline/_router.py:114-123` — the dispatch loop, no try/except.
  `src/yowo/pipeline/__init__.py:156, 161` — `router.route(fut.result(), prev_batch)` unguarded.
- **Smallest fix:** Wrap `cb(stream_id, stream_detections)` in try/except, log at ERROR with the
  stream id, and count consecutive callback failures so a persistently broken consumer can be
  unregistered rather than taking the process with it.

### R18. A single `PostprocessBuffer` is shared across concurrent `detect()` calls, against its own contract

- **Symptom:** `PostprocessBuffer`'s docstring says "Single-writer pattern: must not be shared
  across concurrent detect() calls" — and `BaseEngine._run_batch` passes the one shared instance on
  every call. `_infer_lock` covers only the GPU section; `_process_batch` (NMS + inverse letterbox)
  runs deliberately *outside* it. Two threads calling the public `engine.detect()` concurrently
  therefore write into the same scratch array and silently produce corrupted box coordinates. No
  exception, no metric — just wrong answers.
- **Evidence:**
  - `src/yowo/postprocess/_nms.py:112-117` — the contract.
  - `src/yowo/engine.py:877` — `return self._infer_from_tensor(tensor, frames,
    scratch=self._postprocess_buf)`.
  - `src/yowo/engine.py:824-828` — the docstring explicitly notes postprocessing happens outside
    the lock.
  - `src/yowo/engine.py:1045` — `detect()` is public with no documented threading restriction.
  - Today the in-tree callers are accidentally safe (`pipeline/__init__.py:150` uses
    `max_workers=1`; `_streaming.py:169` passes `scratch=None` on the concurrent path) — nothing
    protects an external caller.
- **Smallest fix:** Make the scratch buffer thread-local (`threading.local()`) or pull it from a
  small pool keyed by thread, and document `detect()`'s concurrency contract either way.

### R19. Structured JSON logs drop tracebacks and the logger name

- **Symptom:** `JsonFormatter` emits only `timestamp`, `level`, `event`, plus six optional context
  fields. It never reads `record.exc_info`, `record.name`, `record.funcName`, or `record.lineno`.
  So with `structured_logging=True` — the mode intended for log aggregation — every
  `logger.exception(...)` in the codebase loses its traceback entirely, and no log line says which
  module produced it.
- **Evidence:**
  - `src/yowo/logging.py:43-60` — the `format()` body; `_OPTIONAL_FIELDS` at `:21-28` has no
    exception or source field.
  - Callers whose value is destroyed: `src/yowo/pipeline/_collector.py:455, 467`
    (`logger.exception("Error stopping reader")`), `src/yowo/io/_reader.py:203-207`
    (`exc_info=True`), `src/yowo/batch/_runner.py:426`.
- **Smallest fix:** Add `"logger": record.name` and, when `record.exc_info`, `"exception":
  self.formatException(record.exc_info)` to the payload.

### R20. Concurrent weight downloads race on a fixed temp filename

- **Symptom:** The temp path is deterministic: `dest.with_suffix(".pt.tmp")`. Two processes on one
  box warming the same shared cache (multi-worker serving, parallel CI, several containers on a
  mounted volume) write interleaved bytes to the same file, and whichever finishes first `os.replace`s
  a corrupt result into the cache — where, per R5, it stays.
- **Evidence:** `src/yowo/models/_weights.py:78` — `tmp_path = dest.with_suffix(".pt.tmp")`;
  `:86` — `os.replace(tmp_path, dest)`; `:93-94` — the loser's cleanup `unlink`s the file the
  winner may still be writing.
- **Smallest fix:** `tempfile.mkstemp(dir=dest.parent, suffix=".pt.tmp")` so each process gets a
  unique path; `os.replace` is already atomic, so the last writer simply wins with a *complete* file.

### R21. `_stream_pipeline` has no idle timeout and can block forever

- **Symptom:** The live path has a 30 s idle guard (`_streaming.py:107-111`). The offline/pipeline
  path has none: it loops on `reader.get(timeout=5.0)` and only exits when
  `reader.is_exhausted`. A source that stalls without erroring (a network mount that hangs, a
  video file on a disconnected NFS share) spins this loop forever with no log and no metric.
- **Evidence:** `src/yowo/_streaming.py:183-194` — `while not _stop.is_set(): raw = reader.get(
  timeout=5.0)` … `if frame is None and reader.is_exhausted: break`.
- **Smallest fix:** Mirror the `idle_since` accounting from `_stream_live` (configurable, generous
  default), logging a warning and terminating with a `SourceTimeoutError`.

### R22. Every weight-fetch failure is reported as `ModelNotFoundError` — callers cannot tell transient from permanent

- **Symptom:** DNS failure, connection reset, HTTP 500, HTTP 404, and a genuinely unregistered
  model all surface as the same exception type. A caller implementing "retry on transient, alert on
  permanent" has to string-match the message.
- **Evidence:** `src/yowo/models/_weights.py:88` — `except Exception as exc:` catches everything;
  `:96-99` — `raise ModelNotFoundError(f"Failed to download weights from {url} ...")`. Contrast
  `src/yowo/errors.py:144-157`, which documents `ModelNotFoundError` as "does not exist" and
  `ModelLoadError` as "found but corrupt" — neither means "the network is down". The hierarchy has
  no transient/network error type at all.
- **Smallest fix:** Add `WeightsDownloadError(ModelError)` (or a `transient: bool` attribute on
  `YowoError`), raise it for network/HTTP-5xx failures, and keep `ModelNotFoundError` for 404 and
  unregistered specs. Also stop retrying 4xx — currently a 404 costs three attempts and 6 s of sleep.

---

## P2 — later

- **OOM log prints a hardcoded `0.0` instead of the utilisation that triggered it.**
  `src/yowo/engine.py:661-665` — `logger.warning("OOM monitor: GPU %.1f%% — batch size halved to
  %d", 0.0, self._batch_size)`. `_halve_batch_size()` never receives `pct`. The one log line that
  explains a batch-size drop tells the operator nothing.
- **`close()` is a no-op on two sources.** `VideoFileSource.close()`
  (`src/yowo/io/_source.py:257-260`) releases `self._cap`, which is assigned `None` at `:174` and
  never set — `__iter__` uses a *local* `cap` (`:223`). `WebcamSource.close()`
  (`:461-462`) is literally `pass`. Both rely on generator finalisation to release the device.
- **No `__del__`, `atexit`, `weakref.finalize`, or signal handling anywhere in the package**
  (`grep -rn "atexit|__del__|weakref.finalize|SIGTERM|signal\." src/yowo` → no matches). An engine
  the caller forgets to `close()` never unloads its backend, and a `SIGTERM` to a serving process
  gets no graceful drain.
- **`_active_streams` is mutated outside the lock.** Added under `_shutdown_lock`
  (`src/yowo/_streaming.py:47, 86, 130, 212`) but `discard`ed without it (`:55, 113, 200, 226`), as
  is the `_streams_drained.set()` decision. `set.discard` is GIL-atomic today, but the codebase
  explicitly targets free-threaded builds (`src/yowo/types.py: is_free_threaded`, used at
  `src/yowo/engine.py:513`), where this is a genuine data race. `HealthReport.stream_count`
  (`engine.py:400`) reads the same set unlocked.
- **Prometheus export has no labels and no health metric.** `src/yowo/metrics/_collector.py:237-278`
  emits bare metric names, so two engines in one process collide, and there is no
  `yowo_health_status`, no `yowo_active_streams`, no queue-depth gauge — an operator cannot answer
  "how far behind is it".
- **Latency and frame counters are polluted by failures.** `src/yowo/engine.py:836-838` records
  `record_inference(elapsed_ms, ...)` unconditionally, *including* the fallback path — so a failed
  batch contributes ~300 ms (mostly `time.sleep`) to the p95/p99 histogram and increments
  `frames_total` as if it succeeded.
- **`PreprocessBufferPool.acquire()` blocks unboundedly** (`src/yowo/io/_decode.py:111`,
  `self._sem.acquire()` with no timeout), and `release()` raises `ValueError` while still holding
  `_lock` and *without* releasing the semaphore (`:120-124`), permanently shrinking the pool.
- **`FeatureStore` wipes a shared cache directory at construction.**
  `src/yowo/cache/_store.py:58-63` — `for d in cache_dir.iterdir(): if _is_hash_dir(d.name):
  shutil.rmtree(...)`. Two engines pointed at the same `cache_dir` destroy each other's live
  entries. `size` (`:101-104`) also reads `_active_entries()` without `_lock`.
- **`write_json` uses a fixed temp name and never fsyncs.** `src/yowo/io/_sink.py:32` —
  `path.with_suffix(".json.tmp")`; two writers to the same destination interleave, and a power cut
  between `write_text` and `os.replace` can leave a truncated file.
- **`configure_logging` runs on every engine load and mutates a process-global logger**, inside
  `try/except Exception: pass`. `src/yowo/engine.py:519-526`. Two engines with different
  `log_level` fight over the `yowo` logger; a library should not attach handlers unasked.
- **`DetectionRouter.route` uses `zip(detections, batch)`**
  (`src/yowo/pipeline/_router.py:107`), which silently truncates on a length mismatch instead of
  raising — a positional-routing invariant violation would go unnoticed.
- **`FrameCollector.close()` joins bridges sequentially at 5 s each**
  (`src/yowo/pipeline/_collector.py:421-422` → `:457`), so a 32-camera shutdown can take 160 s in
  the worst case. Signal all stop events first, then join.
- **Dead-bridge retirement leaves stale state.** `src/yowo/pipeline/_collector.py:326-333` removes a
  dead stream from the local `active` set but leaves the entry in `_streams` with
  `state == RUNNING`, so `active_count` (`:400-407`) over-reports live streams indefinitely.
- **`_finalize_load` swallows every logging-setup error** (`src/yowo/engine.py:525-526`,
  `except Exception: pass`) and `hardware/_detect.py:210-211` / `hardware/_capabilities.py:180-181`
  do the same for device probing — a machine whose GPU probe fails silently degrades to CPU with no
  log line explaining why.

---

## What is already good

Do not re-do these:

- **The exception hierarchy is well-designed and well-documented** (`src/yowo/errors.py`) — a real
  tree with docstrings explaining when each is raised. The gaps are in *mapping onto* it (R22), not
  in its shape. `ShutdownError` subclassing `InferenceError` is a nice touch.
- **No bare `except:` anywhere** in `src/yowo` (verified by grep), and only two
  `except Exception: pass` sites in this lane's files.
- **`ThreadedFrameReader`'s queue discipline is genuinely careful**: bounded deque, explicit
  `Lock` + two `Condition`s, three drop policies with real backpressure on `NONE`, a
  stale-item check via `_enqueue_seq` for the `LATEST` policy, and per-property locking for the
  counters. The counters just need to be *exported* (R11).
- **`FrameCollector`'s shutdown ordering is thought through** — the RESEARCH.md-cited self-join
  deadlock is genuinely avoided by having the iterator thread (not the bridge) call
  `remove_stream`, and the sentinel is always delivered from the bridge's `finally`.
  `_stop_entry` logs when a bridge fails to join (`_collector.py:458-463`) — exactly the check
  `ThreadedFrameReader.stop()` is missing.
- **Atomic writes are already the pattern** where they exist: `write_json`
  (`io/_sink.py:31-38`), the batch checkpoint (`batch/_runner.py:89-103`), and the weight download
  (`models/_weights.py:78-86`) all use tmp + `os.replace`. Only the temp *naming* and the missing
  verification need fixing.
- **The event bus's non-blocking design is correct**: semaphore-budgeted slots, a zero-listener
  fast path, an atomically-replaced snapshot read lock-free by the worker, and a drop counter.
- **`_async.astream` explicitly closes the generator** (`_async.py:92`) so `_stream_live`'s
  `finally` runs immediately rather than waiting for GC — a subtle bug that was clearly hit and
  fixed once already.
- **`_validate_warmup_output` / `_validate_output_values`** (`engine.py:531-559`,
  `classify_engine.py:164-177`, `obb_engine.py:182-199`) catch corrupt models at load time instead
  of at 3am — NaN/Inf for detection, softmax-sum for classification, channel-count for OBB. This is
  better than most inference libraries ship with.
- **The OOM recovery ladder exists at all** (`engine.py:608-685`), with a hysteresis band
  (`_OOM_CLEAR = 0.75` vs `_OOM_TIER1 = 0.80`) and batch-size restoration. It needs the logging
  fixed (P2) and tier-3 implemented, not redesigning.
- **Metrics hot-path cost is taken seriously** — a lock-free rolling histogram with a documented
  5 µs budget and an honest docstring about free-threaded races
  (`metrics/_collector.py:104-108`).

