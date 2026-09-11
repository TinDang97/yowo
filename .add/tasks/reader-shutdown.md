---
type: Task
title: No orphaned thread, fd, or capture after a release cycle
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - src/yowo/io/_reader.py
  - src/yowo/io/_source.py
  - src/yowo/pipeline/
  - tests/unit/
gives:
  - S1 `RTSPStreamSource.__iter__`'s capture handle — which capture the loop actually reads
  - S2 `ThreadedFrameReader.stop()` — what it releases, and on what guarantee
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: interview, authority: human, interview: "sha256:72217ad265a92df2", receipt: /tasks/reader-shutdown.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|R:ORPHAN=confirm|R:FAKEOUTAGE=confirm|R:REFCOUNT=confirm|R:COUNTONLY=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:e02cf1b3b363773d", binding: "sha256:7b3cee3fbcffcecc" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:731d428ba13731a3" }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:12efbcf976ba33f4", binding: "sha256:92f01da4bcf59565" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:b7e0b4e688041096" }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:9b9beaee2441f763", binding: "sha256:4365cbe970067f55" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:cd3c439c7017ce93" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/reader-shutdown.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:5b2901e7614fa5ea", binding: "sha256:4365cbe970067f55" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:bd7af2da9f8a56f1" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/reader-shutdown.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: process, outcome: PASS, receipt: /tasks/reader-shutdown.d/runs/2.md, brief: "sha256:bd7af2da9f8a56f1", reason: "12 bound check items green on receipt 2, zero skipped, every Must, Reject and Edge bound. Suite 2477 passed / 11 skipped, ruff clean, ruff-format clean, pyright 0 errors. THE ANTI-LEAK RECONNECT WAS THE LEAK, MEASURED. __iter__ bound the capture as a LOCAL while reconnect() released it, opened a replacement and rebound self._active_cap. Probed: cap #1 released, cap #2 opened and installed, iterator still reading #1, read fails, retry opens #3 -- cap #2 open, unreleased, unreachable. Every periodic reconnect leaked exactly one capture while logging 'released and re-opened capture to prevent memory leak'. Since rtsp-reconnect-correctness landed it also cost a backoff sleep and emitted a false 'recovered after 1.0s' on a healthy stream, training an operator to ignore the line that matters. The loop now re-reads the handle the SOURCE holds on every pass, so the rebind is observed and the replacement is what gets read. THE BOX WAS RIGHT THAT A COUNT CANNOT CATCH IT. The checks assert capture IDENTITY and read counts, never a tally, because the orphan is replaced and the tally balances. I FRAMED A DECISION WITHOUT GROUNDING IT, AND THE HUMAN ANSWERED THE WRONG QUESTION. I asked whether stop() should close the source and described the risk as 'a caller wanting to reuse it gets a closed one'. I had not checked who closes it today. _streaming.py:117 and :204 already call source.close() immediately after stop() on every streaming path, so closing here is a DUPLICATE -- and five engine checks pin 'closed once', correctly. The build surfaced it, the question went back with the evidence, and the answer reversed: the caller owns it. M4, A7 and A8 were corrected on the record and the node refrozen rather than quietly edited. RESIDUAL, named rather than hidden: a ThreadedFrameReader driven directly, outside the engine, still relies on refcounting to free its capture. BOX 1 WAS ALREADY TRUE AND NOTHING HELD IT THERE. Five acquire/release cycles: threads 1 -> 1, every capture released -- because the reader thread returning dropped the last reference and CPython refcounting ran the generator's finally. Incidental to one interpreter, with no check pinning it. There is one now, and it was green from its first run and could not have been red, like this milestone's other preserve-pins. REFUTED BY MUTATION, four executed: restoring the local-snapshot handle reddens 4; making stop() close the source again reddens 1; dropping the join-timeout warning reddens 1; keeping the retry branch's reopen as a local reddens 1 -- but only after a check was added for it. A MUTATION FOUND A RULE BOUND TO NOTHING. Assigning the retry branch's reopened capture to a local left every check green, because no scenario here drove the retry path: after a reconnect the read always succeeded. A5 was written down, cited, and exercised by nothing -- a coverage gap that reads as coverage. test_the_retry_path_publishes_its_capture_to_the_source now breaks the live capture to force that branch and asserts the reopen is published and read from. Third time this session a mutation found what reading could not. TWO OF MY CHECKS ASSERTED PROXIES RATHER THAN SUBJECTS, and both passed against the bug. One asserted the installed capture was 'not released' -- true of an orphan too; the subject is whether it was READ from, so it counts reads now. The other used gc.disable() to prove release did not depend on collection, but gc.disable() disables the CYCLIC collector, not refcounting, so the capture was freed either way and the check could not fail in either direction. Replaced by asserting the call. THE GATE REFUSED ONCE AND WAS RIGHT: E10 was added as an edge and the check I wrote for it cited only A5 and M2, so the edge read as unbound. Bound, refrozen, rebriefed. SCOPE HELD. The outage clock and backoff are untouched and a check pins both from this side; rtsp-reconnect-correctness's and capture-timeouts's pins stay green from theirs. _streaming.py was not edited -- the ownership answer was to leave it alone, not to move four calls out of scope." }
advised_by: edge-reliability-operator
---
## CARD
goal: The capture a reader is reading is the capture the source currently holds, and a release cycle leaves nothing behind on a guarantee stronger than refcounting.
why: Measured 2026-09-11, against the real classes.
  **The anti-leak reconnect is a leak.** `__iter__` binds `cap` as a LOCAL; `reconnect()` releases it, opens a new one and rebinds `self._active_cap`. The iterator never sees it. Probed: cap #1 released, cap #2 opened and set as `_active_cap`, iterator still reading #1, read fails, retry opens #3 — **cap #2 orphaned, open, unreachable**. Every periodic reconnect leaks exactly one capture, and the log line says "released and re-opened capture to prevent memory leak" while it happens.
  **And it now costs a stall.** With `rtsp-reconnect-correctness` landed, the failed read on the released handle enters the outage path: a backoff sleep, a third capture, and a spurious "recovered after 1.0s" — so a mechanism that leaks also fakes an outage every interval. The box is right that a count check cannot catch this: counts balance, because the orphan is replaced.
  **Box 1 is already true, and nothing holds it there.** Five acquire/release cycles: threads 1 -> 1, all five captures released. But that happens because the reader thread returning drops the last reference to the generator and CPython's refcounting runs its `finally`. `stop()` closes nothing. The guarantee is incidental to an implementation detail of one interpreter, and no check pins it.
next: the box names `_source.py:374`; the rebind is at `_source.py:495` now. The line moved, the defect did not.

## RULES
<must>
- M1 The loop reads the capture the SOURCE holds, not a local snapshot of it. A `reconnect()` rebind is observed on the next read.
- M2 No capture is orphaned. After a reconnect, the count of open-and-unreachable captures is zero — the replacement is the one being read.
- M3 A periodic reconnect is not an outage. It must not consume the reconnect budget, emit a "recovered" line, or cost a backoff sleep, because nothing was wrong.
- M4 The source has exactly ONE owner and `stop()` is not it. CORRECTED ON EVIDENCE after the freeze: `_streaming.py:117` and `:204` call `source.close()` immediately after `stop()` on every streaming path, so closing here is a second call, not a fix — and five engine checks pin "closed once", correctly. What `stop()` owes is that it does not RETURN while the thread is still running, so the caller's close cannot land under a live read. RESIDUAL, named rather than hidden: a `ThreadedFrameReader` driven directly, outside the engine, still relies on refcounting to free the capture.
- M5 N acquire/release cycles return threads and captures to baseline, asserted by a check. This passes today; the check is what keeps it passing.
- M6 The existing reconnect contract is otherwise unchanged: still periodic, still live-sources-only, still silent for sources without `reconnect()`.
- M7 Retry, backoff and the outage clock are untouched. `rtsp-reconnect-correctness` owns them and its checks must stay green from the other side.
</must>
<reject>
- R:ORPHAN A capture left open and unreachable after a reconnect. -> "ORPHAN"
- R:FAKEOUTAGE A reconnect that registers as an outage — budget, log line, or sleep. -> "FAKEOUTAGE"
- R:REFCOUNT Relying on garbage collection to release a capture or join a thread. -> "REFCOUNT"
- R:COUNTONLY Proving the rebind is observed with a count check, which the box explicitly says cannot catch it. -> "COUNTONLY"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may swap the capture mid-iteration; taking the reader thread via `reconnect()`, which is the only caller today and the reason the rebind exists -> if wrong and a third party may swap it, the handle needs a lock rather than a re-read.
- A2 [which] covers: S1 · the request does not say which reads observe the rebind; taking the NEXT read after it, not the in-flight one — a read already blocked in FFmpeg cannot be redirected -> if wrong, a rebind is expected to interrupt a read in progress, which no capture API offers. · probe: a rebind between reads is observed on the following read.
- A3 [when] covers: S1 · the request does not say when the loop's handle is refreshed; taking ONCE PER ITERATION at the top, so the window between rebind and observation is one read -> if wrong and it should be per access, the difference is a handle that changes mid-iteration, which is harder to reason about, not easier.
- A4 [absent] covers: S1 · the request does not say what the loop reads if `_active_cap` is None; taking "cannot happen while iterating" — `__iter__` sets it before the loop and every path reassigns it — and asserting rather than silently substituting, since a None here means a bug elsewhere -> if wrong, the loop must tolerate a None and the tolerant version hides the bug. · probe: the attribute is never None during iteration.
- A5 [order] covers: S1 · the request does not say what happens if a reconnect lands between the read and the retry branch; taking the retry branch's own reopen as authoritative, since it assigns `_active_cap` too -> if wrong, a reconnect racing a retry produces two captures and orphans one, which is the very defect.
- A6 [experience] covers: S1 · the request does not say who notices the difference; taking an operator watching logs — the difficulty is that today a healthy stream emits "recovered after 1.0s" every reconnect interval, which trains them to ignore the line that matters -> so a periodic reconnect must be silent about outages.
- A7 [who] covers: S2 · the request does not say who is responsible for releasing the source; taking the CALLER — corrected from "the reader" on evidence found during build: `_streaming.py` already closes it on every path, so the reader closing too is a duplicate, and I had framed the question without that fact -> if wrong and the reader should own it, the four `source.close()` calls in `_streaming.py` must go, which is outside this node's scope. · found: `_streaming.py:116-117` and `:203-204` call `reader.stop()` then `source.close()` in a `finally`.
- A8 [which] covers: S2 · the request does not say which resources `stop()` releases; taking the THREAD only, with the source left to its owner -> the box's "counts return to baseline" is then a property of the reader plus its caller together, which is how it is asserted. · found: baseline holds today across five cycles — threads 1 -> 1, all captures released.
- A9 [when] covers: S2 · the request does not say whether `stop()` is idempotent; taking YES, since `__exit__` calls it and a caller may call it too -> if wrong, a double stop raises where a context manager already called it. · probe: calling `stop()` twice is harmless.
- A10 [absent] covers: S2 · the request does not say what `stop()` does when the thread never started; taking a no-op rather than an error, matching the existing `if self._thread is not None` shape -> if wrong, error handling around a failed start gets harder.
- A11 [order] covers: S2 · the request does not say whether the source closes before or after the join; taking AFTER, because the thread may still be inside a read and closing under it is the use-after-release this node exists to end -> if wrong, a hung thread holds the capture past `stop()`, which the join timeout already bounds.
- A12 [experience] covers: S2 · the request does not say who observes a failed shutdown; taking whoever cycles readers in a long-lived process — the difficulty is that a leaked thread is invisible until the process runs out -> so a join that times out must say so rather than returning silently.

## PLAN
contract: `__iter__` re-reads `self._active_cap` at the top of each pass instead of holding a local, so a `reconnect()` rebind is observed on the next read and the replacement capture is the one used. `reconnect()` is marked so the loop can tell a swap from a failure and not treat it as an outage. `ThreadedFrameReader.stop()` closes the source explicitly after the join, and reports a join that timed out. `RTSPStreamSource.close()` already exists and is what `stop()` calls.
strategy: the orphan check first, since it is the one a count cannot catch — assert the specific capture object identity, not the tally. Then the no-fake-outage check, which only became observable after `rtsp-reconnect-correctness` landed.
regression floor: the full unit suite, including `capture-timeouts`'s and `rtsp-reconnect-correctness`'s pins, both of which assert this node did not touch their surfaces.

## EDGES
- E1 a reconnect between two reads — the next read uses the NEW capture, by identity (M1, R:COUNTONLY).
- E2 after a reconnect, no capture is open and unreachable (M2, R:ORPHAN).
- E3 a periodic reconnect on a healthy stream — no outage, no backoff sleep, no "recovered" line (M3, R:FAKEOUTAGE).
- E4 N acquire/release cycles — threads and captures at baseline (M5).
- E5 `stop()` and the caller's close — exactly one close, from the owner (M4, R:REFCOUNT).
- E6 `stop()` twice, and `stop()` before `start()` — harmless (A9, A10).
- E7 a thread that will not join — reported, not silently ignored (A12).
- E8 a source with no `reconnect()` — still silently skipped (M6).
- E9 the outage clock and backoff — untouched (M7).
- E10 a reopen from the retry branch — published to the source, not kept local (A5, M2).

## CHECKS
The orphan checks assert capture IDENTITY, never a tally: the box says a count check cannot catch
this defect, because the orphan is replaced and the counts balance.

- tests.unit.test_reader_shutdown::test_the_next_read_uses_the_capture_reconnect_installed · covers: M1, A2, E1, R:COUNTONLY · by identity; a count balances here and proves nothing.
- tests.unit.test_reader_shutdown::test_a_reconnect_orphans_no_capture · covers: M2, E2, R:ORPHAN · measured today: cap #2 open, unreleased, unreachable.
- tests.unit.test_reader_shutdown::test_a_periodic_reconnect_is_not_an_outage · covers: M3, E3, R:FAKEOUTAGE · no budget spent, no sleep, no "recovered" line on a healthy stream.
- tests.unit.test_reader_shutdown::test_the_retry_path_publishes_its_capture_to_the_source · covers: A5, M2, E10 · found by mutation: A5 was written down and bound to nothing, because no scenario drove the retry path.
- tests.unit.test_reader_shutdown::test_the_active_capture_is_never_none_while_iterating · covers: A4 · a None here means a bug elsewhere; assert rather than tolerate.
- tests.unit.test_reader_shutdown::test_stop_does_not_close_a_source_the_caller_owns · covers: M4, E5, R:REFCOUNT · one owner, and it is not this class; the engine already closes on every path.
- tests.unit.test_reader_shutdown::test_cycles_return_threads_and_captures_to_baseline · covers: M5, E4, A8 · true today; this is what keeps it true.
- tests.unit.test_reader_shutdown::test_stop_is_idempotent_and_safe_before_start · covers: A9, A10, E6 · `__exit__` calls it and so may the caller.
- tests.unit.test_reader_shutdown::test_a_join_that_times_out_is_reported · covers: A12, E7 · a leaked thread is invisible until the process runs out.
- tests.unit.test_reader_shutdown::test_stop_joins_before_returning · covers: A11 · the caller closes right after `stop()`; returning early would land that close under a live read.
- tests.unit.test_reader_shutdown::test_a_source_without_reconnect_is_still_skipped · covers: M6, E8 · the existing duck-typed contract is unchanged.
- tests.unit.test_reader_shutdown::test_the_outage_clock_and_backoff_are_untouched · covers: M7, E9 · the neighbouring node's surfaces, pinned from this side.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
