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
- [ ] Every blocking call in the I/O path — capture open, frame read, queue get, thread join — has a stated timeout, asserted by a check that a wedged source raises rather than hangs   (← capture-timeouts)
- [ ] RTSP retry bounds and the terminal error are asserted with a fake clock, and a brief camera outage recovers instead of permanently killing the stream   (← rtsp-reconnect-correctness)
- [ ] Thread, fd and capture counts return to baseline after N acquire/release cycles   (← reader-shutdown)
- [ ] Memory is flat over a sustained live-source run through the CLI, with the measurement recorded   (← cli-bounded-memory)
- [ ] No metric or health surface reports a value it did not measure; every drop and degradation increments a counter the operator can reach   (← metrics-truth)
- [ ] The backend-failure fallback produces output that postprocess accepts, for detection, classification and OBB   (← degraded-mode-correctness)
- [ ] Thread and memory sizing reads cgroup limits where present, or declares container limits unsupported   (← container-aware-sizing)

## CLOSE
evidence: <one row per task>
