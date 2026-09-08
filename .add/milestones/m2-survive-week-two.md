---
type: Milestone
title: A live deployment stays correct and bounded, unattended, and says so
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
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
- [ ] A structural check (lint or grep gate) fails on any bare `cap.read()`, `queue.get()` or `thread.join()` without a timeout argument in the I/O path, AND a wedged-source test raises rather than hangs   (← capture-timeouts)
- [ ] RTSP retry bounds and the terminal error are asserted with a fake clock, and a brief camera outage recovers instead of permanently killing the stream   (← rtsp-reconnect-correctness)
- [ ] Thread, fd and capture counts return to baseline after N acquire/release cycles   (← reader-shutdown)
- [ ] An iterator observes a `reconnect()` rebind — or `reconnect()` cannot be called while an iterator holds a capture. A count check cannot catch this: `_source.py:374` rebinds `self._active_cap` while `__iter__` reads its own local, so counts balance while the iterator reads a released handle   (← reader-shutdown)
- [ ] RSS slope is below a stated threshold over a stated duration for a live source through the CLI (baseline to beat: OOM in ~2 min at 1080p/25fps, `review-runtime.md` R6), with the measurement recorded   (← cli-bounded-memory)
- [ ] An inventory test walks the metrics snapshot and the health document and asserts every field has a named producer; every drop and degradation increments a counter the operator can reach   (← metrics-truth)
- [ ] At least one real backend is constructed and executed by a test — no mock — giving the two non-backend-independent tasks below something to regress against   (← real-backend-smoke)
- [ ] The backend-failure fallback produces output that postprocess accepts, for detection, classification and OBB   (← degraded-mode-correctness)
- [ ] Thread and memory sizing reads cgroup limits where present, or declares container limits unsupported. Note: this changes what "cpu count" means, and `tune/_profile.py:93` keys the tune cache on `os.cpu_count()` — the invalidation is owned downstream by `artifact-cache-keys`   (← container-aware-sizing)
## CLOSE
evidence: <one row per task>
