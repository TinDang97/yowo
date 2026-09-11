---
type: Milestone
title: A live deployment stays correct and bounded, unattended, and says so
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: check, authority: process, via: process, boxes: "EXIT:2" }
  - { by: "Tin Dang", at: 2026-09-11, act: check, authority: process, via: process, boxes: "EXIT:1" }
  - { by: "Tin Dang", at: 2026-09-11, act: check, authority: process, via: process, boxes: "EXIT:3" }
  - { by: "Tin Dang", at: 2026-09-11, act: check, authority: process, via: process, boxes: "EXIT:4" }
advised_by: milestone-planner
---
## CARD
goal: A live deployment stays correct, bounded and observable while nobody is watching it.
why: Every resilience mechanism in the I/O path reads as careful engineering and does not work. There is no capture timeout anywhere in `src/`; the RTSP backoff is arithmetically unreachable; the periodic anti-leak reconnect is itself the leak; the CLI OOMs in about two minutes on the primary documented entry point; and the instruments that would reveal any of it are hardcoded, unwired, or stubbed.
next: add new task <slug>

## SCOPE
In:  connect/read timeouts on every capture · RTSP retry semantics and reconnect correctness · reader/thread shutdown and resource release · bounded memory for live sources · metrics truth (drop counters wired, no hardcoded values, health surface real or removed) · degraded-mode output validity · container-aware CPU/memory sizing and the OOM ladder
Out: cross-backend numeric agreement (→ m3-prove-it) · export and quantization behaviour (→ m4-honest-deployment) · the public API shape of the metrics types (→ m5-declare-ga)

## GROUND
touches: src/yowo/io/_source.py · src/yowo/io/_reader.py · src/yowo/pipeline/ · src/yowo/engine.py · src/yowo/_streaming.py · src/yowo/metrics/_collector.py · src/yowo/cli/_main.py · src/yowo/hardware/_detect.py · src/yowo/backends/
risks:
  - **Stated assumption, accepted at intake:** this milestone was deliberately sequenced BEFORE the evidence milestone (m3-prove-it), so its I/O changes land without a cross-backend parity suite underneath them. Mitigation relied upon: the I/O layer is backend-independent, so parity exposure is concentrated in m4, which still lands after m3. Each task here must still write its own red check first — that discipline is doing the work the parity suite would otherwise do.
  - Fixing frame-drop accounting will make drops visible that were previously silent. Dashboards and expectations calibrated on "zero drops" will change. That is the point, but it must be announced.
  - Changing reconnect semantics changes how long a stream survives a camera reboot — both directions are user-visible.

## EXIT
- [x] Every network capture in the I/O path is CONSTRUCTED with an open and a read timeout, asserted by a structural check that fails on a construction omitting either; no bare `queue.get()` or `thread.join()` exists in the I/O path; AND a wedged-source test raises rather than hangs. AMENDED 2026-09-11 from "a structural check fails on any bare `cap.read()`, `queue.get()` or `thread.join()` without a timeout argument" by human decision: `cv2.VideoCapture.read` is `read([, image]) -> retval, image` and has NO timeout argument — a read timeout is a construction property — so that clause was unsatisfiable by any code, while the other two were already satisfied before the task began. Measured: a bare construction fails an unroutable host after 30.08s (FFmpeg's own default); with the properties set to 2000ms it fails after 2.02s. The four bare `cap.read()` calls that remain are correct — the capture is bounded, not the call   (← capture-timeouts)
- [x] RTSP retry bounds and the terminal error are asserted with a fake clock, and a brief camera outage recovers instead of permanently killing the stream   (← rtsp-reconnect-correctness)
- [x] Thread and capture counts return to baseline after N acquire/release cycles. AMENDED 2026-09-11 from "Thread, fd and capture counts" by human decision: no check counts file descriptors, and the cycle check mocks `cv2.VideoCapture` so none are ever opened — a descriptor assertion against that harness would pass while measuring nothing, which is worse than its absence. RESIDUAL, named rather than hidden: descriptor counts are NOT covered. They become checkable once a check drives a real capture, which `real-backend-smoke` is the node that introduces   (← reader-shutdown)
- [x] An iterator observes a `reconnect()` rebind — or `reconnect()` cannot be called while an iterator holds a capture. A count check cannot catch this: `RTSPStreamSource.reconnect` rebinds `self._active_cap` while `__iter__` read its own local, so counts balance while the iterator reads a released handle. AMENDED 2026-09-11 by human decision: the citation was `_source.py:374`, which the file has since moved past — the rebind is in `RTSPStreamSource.reconnect`, and a symbol survives an edit where a line number rots and sends a reviewer into unrelated code. The FIRST arm was taken: the loop now re-reads the handle the source holds, so the rebind is observed   (← reader-shutdown)
- [ ] RSS slope is below a stated threshold over a stated duration for a live source through the CLI (baseline to beat: OOM in ~2 min at 1080p/25fps, `review-runtime.md` R6), with the measurement recorded   (← cli-bounded-memory)
- [ ] An inventory test walks the metrics snapshot and the health document and asserts every field has a named producer; every drop and degradation increments a counter the operator can reach   (← metrics-truth)
- [x] A real backend — `pytorch`, asked for by name and asserted NOT to have been reached by fallback — is constructed through `create_backend` and executes inference on a real image inside the job CI runs, and the result satisfies invariants a broken backend fails: at least one box, each with `x1` < `x2`, `y1` < `y2`, `confidence` in [0, 1] and a COCO `class_name`. That is what the two tasks below regress against. AMENDED 2026-09-11 from "At least one real backend is constructed and executed by a test — no mock" by human decision: those words were already satisfied by `ci-weight-fixture` before this task began, and satisfied vacuously. Measured: the executed backend was whatever survived a silent `onnx` -> `pytorch` fallback (`OnnxBackend` is handed a `.pt` and raises `INVALID_PROTOBUF`), the input was a black frame yielding 0 boxes against bus.jpg's 5, the assertions were `isinstance(result, list)` and `inference_time_ms` > 0 — all satisfied by a backend returning nothing forever — and `pytest.importorskip` let the leg report green having executed no backend at all. NOT COVERED, named rather than implied: `onnx`, `tensorrt`, `openvino` and `coreml` are constructed by no unmocked check; the claim is the one backend named. RESIDUAL: the selector still picks a backend that cannot load a `.pt` and falls back silently — pinned by a tripwire here, owned by `degraded-mode-correctness`   (← real-backend-smoke, ci-weight-fixture)
- [ ] The backend-failure fallback produces output that postprocess accepts, for detection, classification and OBB   (← degraded-mode-correctness)
- [ ] Thread and memory sizing reads cgroup limits where present, or declares container limits unsupported. Note: this changes what "cpu count" means, and `tune/_profile.py:93` keys the tune cache on `os.cpu_count()` — the invalidation is owned downstream by `artifact-cache-keys`   (← container-aware-sizing)
## CLOSE
evidence: <one row per task>
