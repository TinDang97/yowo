---
type: Task
title: Retry arithmetic that works, asserted with a fake clock
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - src/yowo/io/_source.py
  - tests/unit/
gives:
  - S1 `RTSPStreamSource.__iter__`'s retry loop — what it survives and what ends it
  - S2 `reconnect_timeout_s` — the bound a caller sets, and what it measures
depends_on:
  - /tasks/rtsp-credential-redaction.md
  - /tasks/capture-timeouts.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: interview, authority: human, interview: "sha256:05d02d78a0268a01", receipt: /tasks/rtsp-reconnect-correctness.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A13=confirm|A11=confirm|A12=confirm|R:INERT=confirm|R:DIESONBLIP=confirm|R:SPIN=confirm|R:REALSLEEP=confirm|R:ORPHANFIX=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:57d57c46e0201d56", binding: "sha256:2c2f6cd4c6ae9b1d" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:87650adb9f0b3d6c" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/rtsp-reconnect-correctness.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: process, outcome: PASS, receipt: /tasks/rtsp-reconnect-correctness.d/runs/1.md, brief: "sha256:5f38507f93a072da", reason: "16 bound check items green on receipt 1, zero skipped, every Must and Reject bound. Suite 2465 passed / 11 skipped, ruff clean, ruff-format clean, pyright 0 errors. WHAT WAS WRONG, MEASURED BEFORE AUTHORING. Two opposite failure modes in one loop. The bound was INERT: the guard reduced to 'wait > reconnect_timeout_s' and wait = min(2**n, 10) never exceeds 10, so nothing at or above 10 could fire it -- reachable at 5.0, NOT at 10.0, NOT at the documented default 30.0. reconnect_timeout_s=60 behaved identically to 30. And a BLIP killed the stream anyway: the retry branch called _open_cap(), which raises when the camera is momentarily away, and that propagated straight out of the generator -- two frames, one failed reopen, permanently dead. So it retried forever in the case that did not need it and died instantly in the case it existed for. WHAT IS NOW TRUE. The clock starts at the first failure of a contiguous outage and is cleared by a successful read, so reconnect_timeout_s measures how long the camera has been away. A failed reopen is a retry that feeds the same bound, not an exception that ends the stream. The bound wins over the backoff: if the next sleep would cross the deadline the stream ends then, rather than sleeping through it. 0 makes the first failure terminal, None retries without bound -- the two edges are opposite and explicit, and by human decision None preserves the capability that today's broken arithmetic provided by accident. A recovery is logged, because a silent recovery is indistinguishable from a stall of the same length. REFUTED BY MUTATION, six executed: restoring the old 'wait > timeout' guard reddens 6; resetting the outage clock on every disconnect reddens 6; letting a failed reopen raise again reddens 1; treating None as a number reddens 1; dropping the recovery log reddens 1; never resetting on a successful read reddens 1. THE HANG WAS THE BUG, AND IT TAUGHT ME THE HARNESS. The first run of the check suite did not fail -- it HUNG, for ten minutes, because with reads failing forever the generator never yields and next() never returns. That is precisely the defect under test, but a hung suite is worse than a red one: nobody reads a timeout. The fake clock now carries a sleep BUDGET and raises RetriedForever when it is exhausted, so 'this never terminates' is an assertion rather than a wall-clock. A check whose subject is termination must itself terminate. ONE OF MY CHECKS SURVIVED ITS OWN MUTATION, AND ONLY REFUTATION FOUND IT. test_a_successful_read_resets_the_outage_clock asserted the second outage lasted '> 10s'. With the reset DELETED it still passed, because the second outage inherited roughly 23s of a 30s budget -- comfortably over the threshold. A threshold cannot test a reset. It now measures a CONTROL: a fresh permanent outage from a new stream, and asserts the post-recovery outage matches it within 5%. That is what 'resets' means, and the mutation now reddens. Q4, in my own work, for the second time today. THREE OF MY OTHER CHECKS WERE WRONG TOO, all set up as scenarios other than the ones they described: one made the FIRST open fail and so exercised A10 rather than A2; one capped max_frames below the frame it needed and ended in StopIteration; one asked for a 'recovery' from two consecutive successful reads, recovering from nothing. Each was corrected to stage the situation it names. And test_no_check_here_sleeps failed on its own source, because the literal it searches for appeared in its own assertion -- the needle is now built rather than written. SCOPE HELD. Capture construction and the timeouts capture-timeouts added are untouched, and that node's pin still passes from the other side. The reconnect() orphan -- __iter__ holds a local while reconnect() rebinds the attribute, so every periodic reconnect leaks one capture, measured -- is deliberately NOT fixed here; reader-shutdown's box already names that line, and a check pins that the rebind is unchanged so this node cannot quietly absorb another node's guarantee without its checks." }
advised_by: edge-reliability-operator
---
## CARD
goal: A brief camera outage is survived, a permanent one ends the stream with the terminal error, and `reconnect_timeout_s` is the number that decides which.
why: Two opposite failure modes coexist in one loop, both measured 2026-09-11 against the real code.
  **The bound is inert.** `wait = min(2**retry_count, 10)`, and the guard is `time.monotonic() + wait > deadline` where `deadline = time.monotonic() + self._reconnect_timeout_s` — so it reduces to `wait > reconnect_timeout_s`, and `wait` never exceeds 10. Measured: reachable at `5.0`, NOT reachable at `10.0` or at the documented default `30.0`. `SourceTimeoutError` is dead code for every value a caller is likely to set, and `reconnect_timeout_s=60` behaves identically to `30`. The deadline is also reset on every disconnect, so it can never accumulate across a contiguous outage even if the arithmetic worked.
  **And a brief outage kills the stream anyway.** The retry branch calls `self._open_cap()`, which RAISES `SourceError` when the camera is momentarily unavailable, and that propagates straight out of the generator. Measured: two good frames, camera drops for one reopen, `SourceError` — stream permanently dead, no retry at all.
  So the loop retries forever in the case that does not need it (read fails, reopen succeeds) and dies immediately in the case it exists for (reopen fails once). Every resilience property the parameter advertises is absent in both directions.
next: the loop already has a counter, a backoff and a deadline. None of the three is wired to anything that can end it, and the reopen is outside the part that retries.

## RULES
<must>
- M1 A brief outage is survived. A reopen that fails while the camera reboots is a retry, not the end of the stream. Today it is the end.
- M2 A permanent outage ENDS, with `SourceTimeoutError`, within the bound the caller set. Not "eventually" — the bound is the contract, and today no value at or above 10 can ever reach it.
- M3 `reconnect_timeout_s` measures a contiguous OUTAGE, and resets on a successful read rather than on every disconnect. Resetting per disconnect is what makes it unable to accumulate, so a longer value must mean a longer tolerance — today 30 and 60 are the same number.
- M4 Backoff still bounds the RATE of attempts. Fixing termination must not turn the loop into a spin: the cap on `wait` stays, it just stops being the thing that decides termination.
- M5 The bound and the terminal error are asserted with a FAKE CLOCK. A check that sleeps real seconds to prove a 30s bound is a check nobody runs, and a skipped test is green (Q3).
- M6 The terminal error names the source redacted and the bound that was exceeded, so an operator can tell a dead camera from a misconfigured timeout.
- M8 `reconnect_timeout_s=None` retries without bound, explicitly. Terminating at the bound removes a capability that exists today — infinite retry — and a caller who wants it must be able to ask for it in words rather than by exploiting broken arithmetic. `0` and `None` are the two opposite edges and neither is a synonym for the other.
- M7 Nothing here changes capture construction or the timeouts `capture-timeouts` added, and nothing here fixes the `reconnect()` orphan — `__iter__` holds a local while `reconnect()` rebinds the attribute, leaking one capture per periodic reconnect. Measured, real, and `reader-shutdown`'s box already names that line.
</must>
<reject>
- R:INERT A bound a caller can set that does not change behaviour. -> "INERT"
- R:DIESONBLIP A single failed reopen ending the stream. -> "DIESONBLIP"
- R:SPIN Removing the backoff cap, or retrying without delay. -> "SPIN"
- R:REALSLEEP Proving a bound by sleeping through it. -> "REALSLEEP"
- R:ORPHANFIX Fixing the `reconnect()` capture orphan here. -> "ORPHANFIX"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S2 · the request does not say whose number decides the stream's life; taking the DEPLOYER's, unchanged — `reconnect_timeout_s` already exists and already means this, it simply does not work -> if wrong and it should be global, the per-source parameter stays and is redundant.
- A2 [which] covers: S2 · the request does not say which failures the bound covers; taking BOTH a failed read and a failed reopen as the same outage, since from the camera's side they are one event and a caller cannot distinguish them -> if wrong, a reopen failure needs its own bound and its own parameter. · probe: an outage made of alternating read-failures and reopen-failures terminates at the same bound as either alone.
- A3 [when] covers: S2 · the request does not say when the clock starts; taking FIRST FAILURE of a contiguous outage, reset on the next successful read -> if wrong and it measures stream lifetime, a healthy stream dies after `reconnect_timeout_s` regardless of health, which is absurd; recorded so the reading is visible.
- A4 [absent] covers: S2 · the request does not say what `reconnect_timeout_s=0` means; taking "do not retry at all — the first failure is terminal", which is the reading that makes 0 continuous with small values -> if wrong, 0 means unlimited and a caller asking for no retries gets infinite ones. · probe: 0 terminates on the first failure.
- A5 [order] covers: S2 · the request does not say whether backoff or bound wins when they disagree; taking the BOUND — if the next sleep would cross the deadline the stream ends then, rather than sleeping past it and noticing late -> if wrong the stream outlives its stated bound by up to one backoff interval.
- A6 [experience] covers: S2 · the request does not say who reads the terminal error; taking an operator deciding whether the camera or the config is at fault — the difficulty is that "timed out" alone cannot distinguish a dead camera from a bound set too low -> so the message carries the elapsed time and the bound, not just the bound.
- A7 [who] covers: S1 · n/a · the retry loop is internal to a generator already constructed by its owner; it grants no capability and has no actor whose permission it could misjudge.
- A8 [which] covers: S1 · the request does not say which exceptions from `_open_cap` are retryable; taking `SourceError` — the type it actually raises for an unopenable stream — and letting anything else propagate, since an unexpected exception type is not evidence the camera will return -> if wrong, a retryable condition raising something else ends the stream. · probe: a non-SourceError from the reopen still propagates.
- A9 [when] covers: S1 · the request does not say whether `max_frames` interacts with the bound; taking the existing check first, so a stream that has already yielded its quota ends normally rather than entering retry -> if wrong, a bounded capture can end with a timeout instead of cleanly.
- A10 [absent] covers: S1 · the request does not say what happens when the very FIRST open fails; taking the existing behaviour — it raises before the loop and is not a reconnect — because a stream that never started has nothing to reconnect to -> if wrong, startup gets retry semantics it does not have today, which is a larger behaviour change than this node's subject. · probe: the first open still raises immediately.
- A13 [absent] covers: S2 · the request does not say how a caller asks for unbounded retry once the bound works; taking `None`, distinct from `0` which is zero tolerance, so the two edges are opposite and unambiguous -> if wrong, the capability that exists today is removed with no replacement, which is a breaking change on a 2.5.0 semver promise. · probe: `None` survives an outage longer than any finite bound would.
- A11 [order] covers: S1 · the request does not say whether the retry counter resets on a successful reopen or only on a successful read; taking a successful READ, since a reopen that succeeds and then reads nothing is still the same outage and must not reset the backoff to zero -> if wrong, a camera that accepts connections but never sends frames loops fast forever.
- A12 [experience] covers: S1 · the request does not say who observes a recovery; taking the consumer of the frame iterator — the difficulty is that a silent recovery is indistinguishable from a stall of the same length -> so a recovery is observable, at minimum through the existing logging path, rather than silent.

## PLAN
contract: `__iter__`'s retry branch gains a deadline that starts at the FIRST failure of a contiguous outage and is cleared by a successful read. Both a failed read and a failed reopen feed it: `self._open_cap()` is wrapped so a `SourceError` counts as a retry instead of ending the stream. Termination is decided by elapsed-versus-bound, not by comparing the sleep length to the bound. `wait = min(2**retry_count, 10)` is unchanged — it still bounds the attempt RATE, it simply stops being the termination test. `SourceTimeoutError` carries the redacted source, the elapsed time and the bound.
strategy: a fake clock throughout — the loop reads `time.monotonic` and `time.sleep`, both patchable, so a 30-second bound is asserted in microseconds. Write the outage scenarios first (brief, permanent, flapping, zero-bound), then change the loop.
regression floor: the full unit suite, including `capture-timeouts`'s pin that this node does not touch retry construction, which now has to keep passing from the other side.

## EDGES
- E1 a brief outage — one failed reopen, then the camera returns; the stream continues (M1, R:DIESONBLIP).
- E2 a permanent outage — terminates with `SourceTimeoutError` inside the bound, at the default 30.0 where today it never terminates (M2, R:INERT).
- E3 a flapping camera — read-failure, reopen-failure, read-failure; one contiguous outage, one bound (A2).
- E4 recovery resets the clock — an outage, a good read, then a second outage gets the full bound again (M3, A3).
- E5 `reconnect_timeout_s=0` — the first failure is terminal (A4).
- E6 a longer bound tolerates a longer outage — 60 survives what 30 does not, which is false today (R:INERT).
- E7 the first open failing — still raises immediately, not retried (A10).
- E8 a non-`SourceError` from the reopen — propagates (A8).
- E9 the backoff cap — still `min(2**n, 10)`; no spin (M4, R:SPIN).
- E10 the `reconnect()` orphan — still present and untouched (M7, R:ORPHANFIX).
- E11 `reconnect_timeout_s=None` — outlasts any finite bound; the explicit form of today's accidental behaviour (M8, A13).

## CHECKS
Every timing check drives a fake clock; none sleeps. The node's whole subject is a 30-second
bound, and a check that takes 30 seconds to assert it is a check that gets deselected.

- tests.unit.test_rtsp_reconnect::test_a_brief_outage_is_survived · covers: M1, E1, R:DIESONBLIP · measured today: two frames, one failed reopen, stream permanently dead.
- tests.unit.test_rtsp_reconnect::test_a_permanent_outage_terminates_at_the_default · covers: M2, E2, R:INERT · 30.0 is the documented default and today it can never terminate.
- tests.unit.test_rtsp_reconnect::test_a_longer_bound_tolerates_a_longer_outage · covers: M2, E6, R:INERT · 60 must differ from 30; today they are the same number.
- tests.unit.test_rtsp_reconnect::test_read_failures_and_reopen_failures_share_one_bound · covers: A2, E3 · from the camera's side they are one event.
- tests.unit.test_rtsp_reconnect::test_a_successful_read_resets_the_outage_clock · covers: M3, A3, E4 · a second outage gets the full bound, not the remainder of the first.
- tests.unit.test_rtsp_reconnect::test_a_zero_bound_makes_the_first_failure_terminal · covers: A4, E5 · 0 must be continuous with small values, not a synonym for unlimited.
- tests.unit.test_rtsp_reconnect::test_the_bound_wins_when_the_next_sleep_would_cross_it · covers: A5 · ending late by one backoff interval is still ending late.
- tests.unit.test_rtsp_reconnect::test_the_terminal_error_names_the_source_the_elapsed_and_the_bound · covers: M6, A6 · "timed out" alone cannot separate a dead camera from a bound set too low.
- tests.unit.test_rtsp_reconnect::test_the_retry_counter_resets_on_a_read_not_on_a_reopen · covers: A11 · a camera that accepts connections and sends nothing must not loop fast forever.
- tests.unit.test_rtsp_reconnect::test_a_non_source_error_from_the_reopen_propagates · covers: A8, E8 · an unexpected type is not evidence the camera will return.
- tests.unit.test_rtsp_reconnect::test_the_first_open_still_raises_without_retry · covers: A10, E7 · a stream that never started has nothing to reconnect to.
- tests.unit.test_rtsp_reconnect::test_max_frames_ends_cleanly_rather_than_by_timeout · covers: A9 · a satisfied quota is not an outage.
- tests.unit.test_rtsp_reconnect::test_the_backoff_cap_is_unchanged · covers: M4, E9, R:SPIN · termination moves; the rate limit does not.
- tests.unit.test_rtsp_reconnect::test_no_check_here_sleeps · covers: M5, R:REALSLEEP · the guard against this suite becoming one nobody runs.
- tests.unit.test_rtsp_reconnect::test_the_reconnect_orphan_is_untouched · covers: M7, E10, R:ORPHANFIX · reader-shutdown owns it; this node must not quietly absorb it.
- tests.unit.test_rtsp_reconnect::test_none_retries_without_bound · covers: M8, A13, E11 · the capability that exists today, kept but made explicit rather than accidental.
- tests.unit.test_rtsp_reconnect::test_a_recovery_is_observable · covers: A12 · a silent recovery is indistinguishable from a stall of the same length.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
