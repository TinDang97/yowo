---
type: Task
title: The CLI stops retaining every frame it has ever seen
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - src/yowo/cli/_main.py
  - tests/
  - .add/milestones/
gives:
  - S1 the retention rule — when the CLI keeps a Detection and when it drops it
  - S2 what a user asking for `--output` on a live source is told
  - S3 the RSS measurement — what is measured, how, and the bound it must stay under
  - S4 m2 box 5 — the stated threshold and duration
depends_on:
  - /tasks/real-backend-smoke.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:fb1a364b20990f0d", binding: "sha256:d93b096cff9e6026" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:cc5e528198c147a5" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/cli-bounded-memory.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: process, outcome: PASS, receipt: /tasks/cli-bounded-memory.d/runs/1.md, brief: "sha256:e9db4ac83d976431" }
advised_by: inference-perf-auditor
---
## CARD
goal: `yowo detect` on a stream that never ends holds a bounded amount of memory, measured rather than asserted, and a user who asked for output that cannot be produced is told so.
why: `_main.py:269-274` does `detections = []` then `detections.append(det)` for every frame, and `Detection.frame.pixels` is the full decoded BGR array. Review R6 called this "OOM-killed within a couple of minutes"; measured today it is worse than a claim, it is linear:
  | frames | peak RSS growth |
  |---|---|
  | 20 | 118.8 MB |
  | 40 | 231.6 MB |
  | 80 | 475.0 MB |
  | 160 | 938.2 MB |
  That is ~5.9 MB per frame at 1080p — R6 estimated 6.2 — so ~147 MB/s at 25 fps. A 4 GB container dies about 27 seconds into the stream.
  **The retained list is not merely wasteful on a live source, it is unreachable.** `--output` and `--save-frames` are written AFTER the `for` loop, so on a source that never ends they never run. The CLI accumulates gigabytes to feed a writer it will never reach, and the user who asked for the file gets no file and no message.
  **And R6 framed this as a live-source bug, which it is not.** The same list retains every frame of a finite source too: a one-hour 1080p video is ~90,000 frames, ~558 GB. `yowo detect long-video.mp4` has the same defect and nothing anywhere says so.
beat: done · next: add status

## RULES
<must>
- M1 When no output is requested, the CLI retains no `Detection` beyond the one it is printing — for EVERY source, live or finite. This is the default invocation and the one the review found.
- M2 A user who asks for `--output` or `--save-frames` on a LIVE source is told, before the stream starts, that the file is written at stream end and a live stream has none. The command still runs and still prints.
- M3 The bound is MEASURED, not asserted: a check runs the CLI over a stream and reports peak RSS growth, and fails if growth per frame exceeds a stated bound.
- M4 The measurement is sensitive enough to have caught the defect. A control that retains frames must fail the same check, or the check proves nothing about the code that passes it.
- M5 Nothing changes for a finite source with an output requested: the same file, byte-identical, from the same frames.
- M6 Box 5 states the threshold and the duration, records the measured baseline it beats, and names the residual R6 did not: a finite source WITH output still retains every frame.
- M7 The OBB path (`_main.py:393-398`) carries the identical defect and is fixed with it, or the box says it is not.
</must>
<reject>
- R:UNBOUNDED Retaining a result whose lifetime is the stream's, on any path this node claims to have fixed. -> "UNBOUNDED"
- R:INSENSITIVE A memory check that passes against a deliberately-retaining control. -> "INSENSITIVE"
- R:SILENTLOSS Dropping an output a user asked for without telling them. -> "SILENTLOSS"
- R:PEAKCONTAMINATION Measuring successive frame counts in one process, where `ru_maxrss` is a high-water mark and a later run reads lower than an earlier one. -> "PEAKCONTAMINATION"
- R:SILENTREWRITE Amending box 5 without recording that it was amended and why. -> "SILENTREWRITE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose invocation matters; taking the DEFAULT one — `yowo detect <source>` with no flags — because that is the documented entry point R6 says cannot run for more than a few minutes -> if wrong and only live sources matter, a long video keeps OOM-ing.
- A2 [which] covers: S1 · the request does not say which sources are covered; taking ALL of them when no output is requested, because the list is retained identically for a video file and the review's framing as a live-source bug is simply wrong -> if wrong, the fix is narrower than the defect. · probe: a finite source retains nothing either.
- A3 [when] covers: S1 · the request does not say when a result may be dropped; taking IMMEDIATELY AFTER it is printed, holding at most the one in hand -> if wrong and a small window is wanted, a bounded deque would do; nothing asks for one.
- A4 [absent] covers: S1 · the request does not say what happens with no `--output` AND no `--save-frames`; taking RETAIN NOTHING, which is the whole fix -> if wrong, the default invocation keeps the defect.
- A5 [order] covers: S1 · the request does not say whether print order changes; taking UNCHANGED — results print in frame order exactly as today -> if wrong, a consumer piping stdout sees reordering.
- A6 [experience] covers: S1 · the request does not say who notices; taking someone running the documented RTSP example — the difficulty is that today the process simply dies, with no message tying the death to retention -> so the fix must be silent when it works, and the failure it prevents must be visible in the measurement instead.
- A7 [who] covers: S2 · the request does not say who is warned; taking the person who typed the flag, on stderr, so a `--json` stdout consumer is unaffected -> if wrong, a JSON pipeline gets prose in its stream.
- A8 [which] covers: S2 · the request does not say which flags warn; taking BOTH `--output` and `--save-frames`, because both are written after the loop and both are unreachable on a live source -> if wrong, one of the two fails silently. · probe: both flags produce the warning.
- A9 [when] covers: S2 · the request does not say when the warning appears; taking BEFORE the stream starts, so the user can stop and re-run rather than discover it after an hour -> if wrong, the message arrives when it is useless.
- A10 [absent] covers: S2 · the request does not say what happens to the command; taking IT STILL RUNS AND STILL PRINTS — a warning, not a refusal, because refusing breaks an invocation that works today -> if wrong, a user scripting this gets a new non-zero exit.
- A11 [order] covers: S2 · n/a · one warning, emitted once, before any frame.
- A12 [experience] covers: S2 · the request does not say what the message must convey; taking WHY, not just what — that the file is written at stream end and a live stream has none — because "output ignored" invites a bug report -> if wrong, the wording is merely longer.
- A13 [who] covers: S3 · n/a · a measurement authorises nobody.
- A14 [which] covers: S3 · the request does not say which memory figure; taking PEAK RSS of a CHILD process via `resource.getrusage`, not `psutil` — psutil is only a transitive dependency here, and a check that skips when it is absent is a green skip -> if wrong, current-RSS sampling would give a finer slope; peak is the conservative direction. · probe: the measurement runs with no third-party import.
- A15 [when] covers: S3 · the request does not say over what duration; taking FRAME COUNT rather than wall-clock, because a CI runner's speed varies and the invariant is per-frame retention, not per-second -> if wrong, the box's "duration" is expressed in frames and must say so.
- A16 [absent] covers: S3 · the request does not say what a missing `resource` module means; taking SKIP WITH A REASON on a platform without it, since CI is Linux and the check is not the only thing holding M1 -> if wrong on Windows, the structural check still binds.
- A17 [order] covers: S3 · the request does not say whether measurement order matters; taking ONE SUBPROCESS PER FRAME COUNT, because `ru_maxrss` is a process high-water mark and in-process runs contaminate each other — measured, 120 frames read LOWER than 60 -> if wrong, the slope is noise. · probe: each count is measured in a fresh process.
- A18 [experience] covers: S3 · the request does not say who reads a failure; taking someone who reintroduced retention — the difficulty is that an RSS number alone does not say what leaked -> so the failure message reports MB per frame against the bound.
- A19 [who] covers: S4 · the request does not say who may amend box 5; taking the human maintainer, as with every amended box before it -> if wrong, the cost is one interview round.
- A20 [which] covers: S4 · the request does not say which numbers the box states; taking the BOUND and the frame count, plus the measured baseline being beaten, so the box is checkable rather than aspirational -> if wrong, "a stated threshold" stays unstated. · probe: the box names a number a check enforces.
- A21 [when] covers: S4 · the request does not say whether the amendment is dated; taking YES, in the established AMENDED format -> if wrong, the history is gone.
- A22 [absent] covers: S4 · the request does not say what the box claims about a finite source with output; taking NOT FIXED, named explicitly, because that path still retains every frame -> if wrong, a reader takes the box as covering a case that still OOMs.
- A23 [order] covers: S4 · n/a · a box is one criterion.
- A24 [experience] covers: S4 · the request does not say who reads box 5; taking someone deciding whether yowo survives a week unattended — the difficulty is that "RSS slope below a threshold" says nothing about which invocations were fixed -> so the box must name the covered case and the uncovered one.

## PLAN
contract: `_main.py`'s `detect` and `detect-obb` commands retain a `Detection` only when an output that consumes the list was requested AND the source is not live. When one was requested and the source IS live, a warning goes to stderr before the stream and nothing is retained. `tests/unit/test_cli_bounded_memory.py` measures peak RSS growth per frame by running the CLI in a child process at two frame counts, with a retaining control proving the measurement is sensitive.
strategy: red-first — the measurement check fails on the current tree at ~5.9 MB/frame against a bound well under it. The control is written first, because a memory check that cannot fail is worse than none.

## EDGES
- E1 A live source with no output — nothing retained, results still printed in frame order.
- E2 A live source with `--output` — warning on stderr before the first frame, nothing retained, command still exits 0.
- E3 A live source with `--save-frames` — the same warning; both flags are unreachable on a live source.
- E4 A finite source with `--output` — the file is byte-identical to today's.
- E5 A finite source with no output — nothing retained, which R6 did not ask for and which OOMs today.
- E6 `--json` output with a live source and `--output` — the warning goes to stderr, so stdout stays parseable.
- E7 The measurement run in one process rather than per-count — the slope inverts; the check must not be written that way.
- E8 Box 5 carries an AMENDED clause with no date, or states no number a check enforces.

## CHECKS
- tests.unit.test_cli_bounded_memory::test_the_measurement_catches_a_retaining_control · covers: M4, A17, R:INSENSITIVE, R:PEAKCONTAMINATION, E7 · written FIRST: a control that retains every frame must fail the bound, or nothing the check passes means anything.
- tests.unit.test_cli_bounded_memory::test_peak_rss_growth_per_frame_is_under_the_bound · covers: M1, M3, A14, A15, A18 · the box's measurement, run through the real CLI in a child process.
- tests.unit.test_cli_bounded_memory::test_growth_does_not_scale_with_frame_count · covers: M3, A17, R:UNBOUNDED · doubling the frames must not double the growth; measured today it does.
- tests.unit.test_cli_bounded_memory::test_a_live_source_retains_nothing · covers: M1, A3, A4, E1, R:UNBOUNDED · the retained list never exceeds one result.
- tests.unit.test_cli_bounded_memory::test_a_finite_source_retains_nothing_when_no_output_is_asked_for · covers: M1, A1, A2, E5, R:UNBOUNDED · the case R6 missed.
- tests.unit.test_cli_bounded_memory::test_results_still_print_in_frame_order · covers: A5, E1 · dropping the reference must not reorder or lose output.
- tests.unit.test_cli_bounded_memory::test_output_on_a_live_source_warns_before_streaming[output] · covers: M2, A8, A9, A12, E2, R:SILENTLOSS · the user is told why, before the first frame.
- tests.unit.test_cli_bounded_memory::test_output_on_a_live_source_warns_before_streaming[save-frames] · covers: M2, A8, A9, A12, E3, R:SILENTLOSS · the user is told why, before the first frame.
- tests.unit.test_cli_bounded_memory::test_the_warning_does_not_pollute_json_stdout · covers: A7, E6 · stdout stays parseable for a `--json` consumer.
- tests.unit.test_cli_bounded_memory::test_a_live_source_with_output_still_runs · covers: A10, E2 · a warning, not a refusal; exit code unchanged.
- tests.unit.test_cli_bounded_memory::test_a_finite_source_with_output_writes_the_same_file · covers: M5, E4 · the working path is pinned before the fix touches it.
- tests.unit.test_cli_bounded_memory::test_the_obb_path_is_bounded_too · covers: M7 · the identical defect at `_main.py:393-398`.
- tests.unit.test_cli_bounded_memory::test_box_5_records_that_it_was_amended · covers: M6, A21, R:SILENTREWRITE, E8 · an undated rewrite reads as the original.
- tests.unit.test_cli_bounded_memory::test_box_5_states_a_number_a_check_enforces · covers: M6, A20, E8 · "a stated threshold" must be a number this suite actually holds.
- tests.unit.test_cli_bounded_memory::test_box_5_names_the_residual_it_does_not_cover · covers: M6, A22, A24 · a finite source WITH output still retains every frame.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
