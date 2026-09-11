---
type: Task
title: No reported value that was not measured; drops reach the operator
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - src/yowo/metrics/
  - src/yowo/engine.py
  - src/yowo/_streaming.py
  - src/yowo/io/_reader.py
  - tests/
  - .add/milestones/
gives:
  - S1 the metrics inventory — every field, and the producer that moves it
  - S2 the drop counters — which drops reach `EngineMetrics` and from where
  - S3 the degradation counter — what counts as one and who increments it
  - S4 m2 box 6 — what "every field has a named producer" is worth
depends_on:
  - /tasks/degraded-mode-correctness.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-12, act: freeze, authority: human, direction: "sha256:a9322b04b051e5a8", binding: "sha256:55260fd3d9b3fd67" }
  - { by: "cli", at: 2026-09-12, act: brief, authority: process, brief: "sha256:caa0d22ddc52a3fe" }
  - { by: "builder", at: 2026-09-12, act: replan, authority: process, note: "A mutation survived: deleting on_drop= from _streaming.py left all 16 checks green. The drop check built ThreadedFrameReader directly and passed the callback itself, so it bound the reader's plumbing rather than the engine's wiring — the defect the node exists to fix. Added a check that drives the real stream() path with a slow backend and a queue of 1. No Must, Reject, Edge or gives: wording changed." }
  - { by: "Tin Dang", at: 2026-09-12, act: refreeze, authority: human, direction: "sha256:e664e23412a8767a", binding: "sha256:55260fd3d9b3fd67" }
  - { by: "cli", at: 2026-09-12, act: brief, authority: process, brief: "sha256:a8c3c0fdf8559830" }
  - { by: "process:run", at: 2026-09-12, act: run, authority: process, outcome: PASS, receipt: /tasks/metrics-truth.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-12, act: gate, authority: process, outcome: PASS, receipt: /tasks/metrics-truth.d/runs/1.md, brief: "sha256:063f66e6cce6061e" }
advised_by: edge-reliability-operator
---
## CARD
goal: Every number the engine reports was measured by something a check can point at, and no drop or degradation happens without a counter the operator can read moving.
why: Three of the reported surfaces are wrong or absent today, all measured rather than suspected.
  **Dropped frames are counted where nobody looks.** `MetricsCollector.record_frame_dropped` is wired into exactly one place — `engine.py:768`, the `astream` path. The synchronous `stream()` path builds a `ThreadedFrameReader` (`_streaming.py:75`) with a drop policy and passes it no metrics callback at all; the reader keeps its own `_frames_dropped`, and nothing outside `io/_reader.py` ever reads it. Measured with a full queue and a slow consumer: **reader.frames_dropped = 298, metrics.frames_dropped = 0**. An operator watching the snapshot sees zero while 99% of the stream is discarded.
  **Dropped events are not in the snapshot at all.** `EventBus` counts `events_dropped` and `BaseEngine` exposes it as a property, but `EngineMetrics` has no such field — so the one object an operator is told to read cannot report them.
  **Nothing counts a degradation.** `_halve_batch_size`, `_try_precision_fallback` and the backend fallback in `load()` each flip health to `DEGRADED` and increment nothing. Health is a level, not a count: a stream that degraded and recovered forty times reads exactly like one that never did.
  **And one reported value is fabricated.** `engine.py:662` logs `"OOM monitor: GPU %.1f%% — batch size halved to %d"` with a literal `0.0` as the percentage. It always reports GPU 0.0%, presented as a measurement, in the log line an operator reads while diagnosing an OOM.
beat: done · next: add status

## RULES
<must>
- M1 Every field of `EngineMetrics` and `HealthReport` has a producer a check names and DRIVES — not a citation, an execution that moves the field off its zero value or proves it cannot move.
- M2 A frame dropped by the synchronous streaming path increments `EngineMetrics.frames_dropped`. The field's docstring already promises this; today it is only true on the async path.
- M3 Dropped events are readable from the metrics snapshot, not only from a separate engine property.
- M4 A degradation increments a counter. OOM batch-halving, precision fallback and backend fallback each count, so an operator can tell a stream that degraded forty times from one that never did.
- M5 No reported value is a literal standing in for a measurement. The OOM log reports the percentage it actually read, or does not name one.
- M6 The inventory is EXHAUSTIVE by construction: it enumerates the dataclass fields at runtime, so a field added later without a producer fails the check rather than being silently uncovered.
- M7 Box 6 says what the inventory establishes, records that it was amended and why, and names what it does not cover.
</must>
<reject>
- R:UNPRODUCED A reported field no check drives — including one a check merely names. -> "UNPRODUCED"
- R:UNREACHABLE A drop or degradation that happens without a counter the operator can read moving. -> "UNREACHABLE"
- R:FABRICATED A literal presented as a measured value. -> "FABRICATED"
- R:ROSTER An inventory written as a hand-maintained list of field names, which goes stale the day a field is added. -> "ROSTER"
- R:SILENTREWRITE Amending box 6 without recording that it was amended and why. -> "SILENTREWRITE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose view matters; taking THE OPERATOR reading `engine.metrics` and `engine.health_report()`, because those are the two documented surfaces -> if wrong and only logs matter, the inventory covers the wrong objects.
- A2 [which] covers: S1 · the request does not say which fields are in scope; taking EVERY field of both dataclasses, enumerated at runtime rather than listed, so a new field cannot slip past -> if wrong, the inventory is a roster that rots. · probe: adding a field to either dataclass fails the inventory.
- A3 [when] covers: S1 · the request does not say when a producer must run; taking DURING THE CHECK — each field is driven off its zero value, or the check records why it cannot be -> if wrong, a field is covered by a claim rather than an execution.
- A4 [absent] covers: S1 · the request does not say what to do about a field that is legitimately absent; taking an EXPLICIT allowlist with a stated reason — `memory_pct` is `None` off CUDA, and that is a producer conditioned on hardware, not a missing one -> if wrong, either a real gap hides behind an excuse or a legitimate None fails the build. · probe: the allowlist names a reason per field.
- A5 [order] covers: S1 · n/a · fields are independent; no ordering changes any verdict.
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking whoever added a field — the difficulty is a failure that says only "field uncovered" -> so the message names the field and what a producer would have to do.
- A7 [who] covers: S2 · the request does not say who counts a drop; taking THE COLLECTOR, with the reader reporting into it, because two counters that must be summed is a way to report zero -> if wrong, the reader's own count stays authoritative and nobody reads it.
- A8 [which] covers: S2 · the request does not say which drops count; taking BOTH the queue-policy drops in `ThreadedFrameReader` and the async-path drops already wired -> if wrong, one path stays silent. · probe: a sync stream with a full queue moves the snapshot.
- A9 [when] covers: S2 · the request does not say when the count is visible; taking AS IT HAPPENS, not at stream end, because an operator diagnoses a live stream -> if wrong, the number arrives after it is useful.
- A10 [absent] covers: S2 · the request does not say what a drop policy of NONE means; taking NO DROPS AND THEREFORE NO INCREMENTS — a zero that is true rather than a zero that is unwired -> if wrong, the check cannot tell the two zeroes apart, which is the whole defect.
- A11 [order] covers: S2 · the request does not say whether drop order matters; taking IT DOES NOT — the count is a total, and which frame was discarded is the drop policy's business -> if wrong, an operator wanting to know WHICH frames went is not served, and nothing asks for that.
- A12 [experience] covers: S2 · the request does not say who is harmed; taking an operator sizing a queue — the difficulty is that a reported 0 is indistinguishable from a healthy stream, so they conclude the queue is fine while it discards 99% -> so folding the reader into the existing field is the fix, not a new field beside it.
- A13 [who] covers: S3 · the request does not say who increments a degradation; taking THE ENGINE, at each of the three sites that already set `HealthStatus.DEGRADED` -> if wrong, a fourth site added later is uncounted; M6's enumeration does not cover call sites, only fields, and that is a named limit.
- A14 [which] covers: S3 · the request does not say which events are degradations; taking the three that exist — OOM batch-halving, precision fallback, backend fallback — as ONE counter rather than three, because box 6 asks that a degradation be countable, not that it be classified -> if wrong, an operator cannot tell which kind fired and must read the log. · probe: each of the three sites moves the counter.
- A15 [when] covers: S3 · the request does not say whether recovery decrements; taking NO — it is a cumulative count, like `errors_total`, because a counter that goes back down cannot tell forty degradations from none -> if wrong, a gauge is wanted and the level already exists as `health`.
- A16 [absent] covers: S3 · the request does not say what a degradation that fails means — `_try_precision_fallback` catches its own exception; taking IT STILL COUNTS, because the engine still entered a degraded state -> if wrong, a failed recovery is invisible.
- A17 [order] covers: S3 · n/a · a cumulative counter has no order.
- A18 [experience] covers: S3 · the request does not say who reads the counter; taking someone deciding whether a deployment is stable — the difficulty is that `health` is a level that reads READY again after recovery, so a flapping stream looks healthy every time they check -> so the count must be cumulative and must be in the snapshot.
- A19 [who] covers: S4 · the request does not say who may amend box 6; taking the human maintainer, as with every amended box before it -> if wrong, one interview round.
- A20 [which] covers: S4 · the request does not say which part changes; taking the addition of what the inventory COVERS and what it does not, because "every field has a named producer" is satisfied by a list of names -> if wrong, the box permits exactly the roster R:ROSTER forbids. · probe: the box names the enumeration, not a count of fields.
- A21 [when] covers: S4 · the request does not say whether the amendment is dated; taking YES, in the established format -> if wrong, the history is gone.
- A22 [absent] covers: S4 · the request does not say what the box claims about producers added later; taking NOT CLAIMED for call sites — a new degradation site that forgets to count is not caught by a field enumeration, and the box says so -> if wrong, a reader takes the box as a guarantee about code not yet written.
- A23 [order] covers: S4 · n/a · a box is one criterion.
- A24 [experience] covers: S4 · the request does not say who reads box 6; taking someone deciding whether the numbers can be trusted for capacity planning — the difficulty is that "every field has a named producer" sounds total while the measured defect was a field whose producer existed and was never called -> so the box must say the producers are DRIVEN, not named.

## PLAN
contract: `EngineMetrics` gains `events_dropped: int = 0` and `degradations_total: int = 0`. `MetricsCollector` gains `record_degradation()` and accepts the event bus's drop count into its snapshot. `ThreadedFrameReader` takes an optional `on_drop` callback, which `_streaming.py` wires to `MetricsCollector.record_frame_dropped` at both construction sites. The three degradation sites in `engine.py` call `record_degradation()`. `engine.py:662`'s OOM log stops printing a literal percentage. `tests/unit/test_metrics_truth.py` enumerates both dataclasses with `dataclasses.fields` and drives every field.
strategy: red-first — the inventory is written against the current tree, where it fails on the fields with no producer. The measured 298-versus-0 gap is reproduced as a check before the wiring exists.

## EDGES
- E1 A sync stream with a full queue and `LATEST` — the snapshot's drop count moves, measured at 298 today against a reported 0.
- E2 A drop policy of `NONE` — no drops and no increments, a zero that is true rather than unwired.
- E3 The async path — its drops keep counting, so the fix adds a producer rather than moving one.
- E4 A field added to either dataclass with no producer — the inventory fails rather than silently skipping it.
- E5 `memory_pct` off CUDA — `None` is legitimate, allowlisted with a reason, not counted as a gap.
- E6 A degradation that fails (`_try_precision_fallback` swallowing its exception) — still counts.
- E7 Forty degrade/recover cycles — the counter reads 40 while `health` reads READY.
- E8 Box 6 carries an AMENDED clause with no date, or claims producers are named rather than driven.

## CHECKS
- tests.unit.test_metrics_truth::test_every_metrics_field_has_a_driven_producer · covers: M1, M6, A2, A3, A6, E4, R:UNPRODUCED, R:ROSTER · `dataclasses.fields(EngineMetrics)` enumerated at runtime; every field driven off its zero value or allowlisted with a reason.
- tests.unit.test_metrics_truth::test_every_health_field_has_a_driven_producer · covers: M1, M6, A1, A2, E4, E5, R:UNPRODUCED, R:ROSTER · the same for `HealthReport`, with `memory_pct` allowlisted as hardware-conditioned.
- tests.unit.test_metrics_truth::test_the_allowlist_states_a_reason_for_every_entry · covers: A4, E5 · an allowlist without reasons is a way to exempt a real gap.
- tests.unit.test_metrics_truth::test_a_field_added_without_a_producer_fails_the_inventory · covers: M6, A2, E4, R:ROSTER · the control: a synthetic extra field must fail, or the enumeration proves nothing.
- tests.unit.test_metrics_truth::test_a_sync_stream_drop_reaches_the_snapshot · covers: M2, A7, A8, A9, A12, E1, R:UNREACHABLE · measured 298 dropped against 0 reported before this.
- tests.unit.test_metrics_truth::test_the_engine_wires_the_reader_to_the_counter · covers: M2, A7, A8, E1, R:UNREACHABLE · ADDED DURING BUILD. `test_a_sync_stream_drop_reaches_the_snapshot` constructs the reader itself and hands it the callback, so it proves the reader CAN report and not that the engine DOES wire it. Measured: deleting `on_drop=` from `_streaming.py` left all 16 checks green — the exact defect this node exists to fix, unbound. This drives the real `stream()` path.
- tests.unit.test_metrics_truth::test_a_policy_of_none_reports_a_true_zero · covers: A10, E2 · distinguishes an honest zero from an unwired one.
- tests.unit.test_metrics_truth::test_the_async_path_still_counts_its_drops · covers: A8, E3 · the existing producer is not moved, only joined.
- tests.unit.test_metrics_truth::test_dropped_events_are_in_the_snapshot · covers: M3 · the operator reads one object, not two.
- tests.unit.test_metrics_truth::test_each_degradation_site_increments_the_counter[oom] · covers: M4, A13, A14 · driven at the site, not asserted about it.
- tests.unit.test_metrics_truth::test_each_degradation_site_increments_the_counter[precision] · covers: M4, A13, A14, A16, E6 · a recovery that fails still counts.
- tests.unit.test_metrics_truth::test_each_degradation_site_increments_the_counter[backend] · covers: M4, A13, A14 · driven at the site, not asserted about it.
- tests.unit.test_metrics_truth::test_the_counter_is_cumulative_across_recovery · covers: A15, A18, E7 · forty cycles read 40 while health reads READY.
- tests.unit.test_metrics_truth::test_no_reported_value_is_a_literal · covers: M5, R:FABRICATED · the OOM log does not print a hardcoded percentage.
- tests.unit.test_metrics_truth::test_box_6_records_that_it_was_amended · covers: M7, A21, R:SILENTREWRITE, E8 · an undated rewrite reads as the original.
- tests.unit.test_metrics_truth::test_box_6_claims_driven_producers_not_named_ones · covers: M7, A20, A24, E8 · the measured defect was a field whose producer existed and was never called.
- tests.unit.test_metrics_truth::test_box_6_names_what_the_inventory_does_not_cover · covers: M7, A22 · a new degradation call site is not caught by a field enumeration.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
