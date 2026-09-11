---
type: Task
title: Three m2 boxes say what their tasks actually delivered
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - .add/milestones/
  - tests/unit/
gives:
  - S1 m2 box 1 — what a structural timeout gate can actually assert
  - S2 m2 box 3 — which resources a release cycle returns to baseline
  - S3 m2 box 4 — the rebind requirement and where the code that motivates it lives
generated: { by: add/3.5.0, at: 2026-09-11 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: interview, authority: human, interview: "sha256:d37199de130da544", receipt: /tasks/m2-box-accounting.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A11=confirm|A13=confirm|A14=confirm|A15=confirm|A17=confirm|R:ROUNDING=confirm|R:VACUOUS=confirm|R:SILENTREWRITE=confirm|R:WIDEN=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:792becc79839d3ed", binding: "sha256:d7ed17101f478193" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:4a76002160ae0df2" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/m2-box-accounting.d/runs/1.md }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/m2-box-accounting.d/runs/2.md }
  - { by: "builder", at: 2026-09-11, act: replan, authority: process, note: "CHECKS citations only: the two box-sweeping checks are parametrised, and a bare-name citation binds nothing (lesson M12). Split one line per case with ASCII ids box-1/box-3/box-4, and bound R:VACUOUS on test_box_3_claims_only_threads_and_captures, where the fd clause leaves rather than being bound by a mock that opens no descriptor. No Must, Reject, Edge or gives: wording changed." }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:5330f82eb795b3d5", binding: "sha256:d7ed17101f478193" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/m2-box-accounting.d/runs/3.md }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:4a10bea5c61450a8" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/m2-box-accounting.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: process, outcome: PASS, receipt: /tasks/m2-box-accounting.d/runs/4.md, brief: "sha256:4a10bea5c61450a8" }
---
## CARD
goal: Each of the three boxes states what its task delivered and what a check can establish, so ticking them is a true statement rather than a rounding.
why: Three tasks are done and gated — `capture-timeouts`, `rtsp-reconnect-correctness`, `reader-shutdown` — and of their four boxes only one could be ticked honestly. The other three each fail differently, and every one of them would have been papered over by ticking.
  **Box 1 asks for something no code can do.** "A structural check fails on any bare `cap.read()` ... without a timeout argument". `cv2.VideoCapture.read` is `read([, image]) -> retval, image`; there is no timeout argument, and a read timeout is a CONSTRUCTION property. Its other two clauses were already satisfied before the task began — no bare `.get()` or `.join()` exists in `io/`, `pipeline/` or `_streaming.py`. So the box asked for one impossible thing and two done things, and what `capture-timeouts` actually built — a gate on construction — is not what it says.
  **Box 3 claims a resource I never counted.** "Thread, FD and capture counts return to baseline". I assert threads and captures. I do not assert file descriptors anywhere, and the cycle check mocks `cv2.VideoCapture`, so no real descriptor is ever opened — an fd assertion against that harness would pass while measuring nothing, which is worse than its absence.
  **Box 4 points at the wrong line.** It cites `_source.py:374` for the rebind that motivates it. The rebind is at `:504`. The substance IS satisfied; the citation rots, and a reader who follows it finds unrelated code and mistrusts the box.
next: box 2 is already ticked and needed no amendment — it is the control that says the other three are not a pattern of the milestone being unsatisfiable, only of three specific clauses being wrong.

## RULES
<must>
- M1 Box 1 states the guarantee a check can establish: that every network capture is CONSTRUCTED with open and read timeouts, and that a wedged source raises rather than blocking past the bound. The `.get()`/`.join()` clauses stay, because they are real and they hold.
- M2 Box 1 records that its original wording was unsatisfiable and why, rather than being silently rewritten. A box that quietly changes shape teaches a reader that boxes are negotiable.
- M3 Box 3 claims only what is counted. Either the fd clause is bound by a check that opens real descriptors, or it comes out and says so — it may not stay as a word nobody measured.
- M4 Box 4's line reference points at the rebind that exists. A citation that rots is worse than none, because a reader follows it and concludes the box is confused.
- M5 Every claim each amended box makes is EXECUTED by a check, not restated. This is the box-6 lesson: prose asserting a guarantee drifts silently, and only running it catches that.
- M6 No task node is reopened and no check of a closed node is weakened. The tasks delivered what they delivered; this node changes what the milestone SAYS they delivered.
</must>
<reject>
- R:ROUNDING Ticking a box whose literal words are not satisfied. -> "ROUNDING"
- R:VACUOUS Binding the fd clause with a check that cannot fail — mocked captures open no descriptors. -> "VACUOUS"
- R:SILENTREWRITE Amending a box without recording that it was amended and why. -> "SILENTREWRITE"
- R:WIDEN Amending a box to claim more than its task built. -> "WIDEN"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1, S2, S3 · the request does not say who may amend a milestone box; taking the human maintainer, as with m1 box 6 and m1 box 2 before it -> if wrong and process authority suffices, the cost is one interview round.
- A2 [which] covers: S1 · the request does not say which of box 1's three clauses change; taking ONLY the `cap.read()` clause, since `.get()` and `.join()` are satisfiable and satisfied -> if wrong, two true clauses are removed with the false one. · probe: the amended box still demands the two clauses that hold.
- A3 [when] covers: S1 · the request does not say whether the amendment is dated and attributed; taking YES, in box 2's AMENDED format, because an undated rewrite is indistinguishable from the box always having said that -> if wrong, the milestone reads tidier and its history is gone.
- A4 [absent] covers: S1 · the request does not say what the box claims about non-network captures; taking NOT CLAIMED — the properties are FFMPEG-only and a file or webcam capture does not honour them -> if wrong, the box implies a guarantee for sources that cannot have it.
- A5 [order] covers: S1 · n/a · a box is one criterion; its clauses are a conjunction with no ordering a reader could apply wrongly.
- A6 [experience] covers: S1 · the request does not say who reads box 1; taking someone auditing whether yowo can hang — the difficulty is that "a gate on bare `cap.read()`" sounds satisfied by the four bare `cap.read()` calls still in the file, which are fine because the CAPTURE is bounded -> so the box must say construction, or a reader greps, finds them, and concludes the box is broken.
- A7 [which] covers: S2 · the request does not say whether to bind the fd clause or drop it; taking DROP, with the reason recorded, because the cycle check mocks `cv2.VideoCapture` and a descriptor count against mocks cannot fail -> if wrong and fds must be counted, that needs a real-capture fixture and belongs with `real-backend-smoke`, which is the node that introduces real resources. · probe: the amended box names threads and captures only, and says why fds are absent.
- A8 [when] covers: S2 · the request does not say when the fd clause could return; taking "when a check drives a real capture", and naming that rather than deleting the concern -> if wrong, the concern is lost and nobody counts descriptors ever.
- A9 [absent] covers: S2 · the request does not say what "baseline" means for a process that never opened a descriptor; taking "the same count as before the cycles", which is trivially true under mocks and is exactly why the clause is being dropped rather than kept and satisfied -> if wrong, a vacuous pass is considered evidence.
- A10 [order] covers: S2 · n/a · the resources counted are independent; no order of measurement changes any verdict.
- A11 [experience] covers: S2 · the request does not say who reads box 3; taking an operator deciding whether a long-lived process leaks — the difficulty is that "fd" in the box implies someone checked, and nobody did -> so its absence must be visible in the box, not only in a task receipt.
- A12 [who] covers: S3 · n/a · a line reference authorises nothing.
- A13 [which] covers: S3 · the request does not say whether to cite a line or the symbol; taking the SYMBOL — `RTSPStreamSource.reconnect` — with the line as a hint, because a symbol survives edits and a line number does not, which is the defect being fixed -> if wrong, precision is lost; the symbol is unambiguous in this file. · probe: the amended box names the symbol.
- A14 [when] covers: S3 · the request does not say whether the box's disjunction survives; taking YES, both arms recorded, with the arm taken named — the rebind is observed — so a later reader sees the choice rather than assuming there was none -> if wrong, the box reads as though only one design was ever possible.
- A15 [absent] covers: S3 · the request does not say what happens to the "a count check cannot catch this" clause; taking KEEP, because it is the box's most useful sentence and it is still true -> if wrong, a future node replaces the identity checks with a tally and the defect returns silently.
- A16 [order] covers: S3 · n/a · the box's clauses are a conjunction; no ordering is applied.
- A17 [experience] covers: S3 · the request does not say who follows the line reference; taking a reviewer verifying the box is real — the difficulty is that a stale line lands them in unrelated code and they conclude the box is confused rather than that the file moved -> so the citation must be one that survives an edit.

## PLAN
contract: the three boxes are rewritten in place, each carrying an `AMENDED 2026-09-11 ... by human decision` clause in the format m1 box 2 established, and each then ticked. A new `tests/unit/test_m2_box_accounting.py` executes every claim the amended boxes make — the construction gate, the `.get()`/`.join()` absence, the thread-and-capture baseline, the symbol the rebind box names — so a box that drifts from the code reddens rather than merely reading well.
strategy: amend, then bind, then tick, in that order, so no box is ticked before something executes its claims.
regression floor: the full unit suite, and the three finished nodes' own checks, which must stay green from the other side.

## EDGES
- E1 box 1's `.get()`/`.join()` clauses — still demanded, still true (A2).
- E2 the four bare `cap.read()` calls still in `_source.py` — the amended box must not read as forbidding them (A6).
- E3 the fd clause — absent from box 3, with its absence explained and its return condition named (A7, A8).
- E4 box 4's citation — a symbol that survives an edit, not a line (A13, E7 below).
- E5 a box amended without an AMENDED clause — reddens (R:SILENTREWRITE).
- E6 a box claiming a component no check drives — reddens (R:WIDEN).
- E7 the rebind symbol renamed or removed — box 4's citation reddens rather than rotting silently.

## CHECKS
all in `tests/unit/test_m2_box_accounting.py`. Every box claim is EXECUTED; none is restated.

- tests.unit.test_m2_box_accounting::test_box_1_demands_construction_not_a_read_argument · covers: M1, A6, E2 · `read([, image])` has no timeout argument; the four bare calls are fine because the capture is bounded.
- tests.unit.test_m2_box_accounting::test_box_1_still_demands_the_get_and_join_clauses · covers: A2, E1 · two true clauses must not leave with the false one.
- tests.unit.test_m2_box_accounting::test_no_bare_get_or_join_exists_in_the_io_path · covers: A2, E1 · the clause the box keeps, executed rather than trusted.
- tests.unit.test_m2_box_accounting::test_every_network_capture_is_constructed_with_both_timeouts · covers: M1, M5 · the guarantee box 1 now states, driven through the real constructor.
- tests.unit.test_m2_box_accounting::test_box_3_claims_only_threads_and_captures · covers: M3, A7, E3, R:VACUOUS · the word "fd" may not survive without a check behind it; a mocked capture opens no descriptor, so the clause leaves rather than being bound by a check that cannot fail.
- tests.unit.test_m2_box_accounting::test_box_3_says_why_fds_are_absent_and_when_they_return · covers: A8, A11, E3 · the concern is named, not deleted.
- tests.unit.test_m2_box_accounting::test_threads_and_captures_return_to_baseline · covers: M3, M5 · box 3's remaining claim, executed here as well as in its own node.
- tests.unit.test_m2_box_accounting::test_box_4_cites_a_symbol_that_exists · covers: M4, A13, E4, E7 · a citation that rots is worse than none.
- tests.unit.test_m2_box_accounting::test_box_4_keeps_the_count_check_warning · covers: A15 · the box's most useful sentence, and still true.
- tests.unit.test_m2_box_accounting::test_box_4_names_the_arm_that_was_taken · covers: A14 · a disjunction with the chosen arm recorded.
- tests.unit.test_m2_box_accounting::test_every_amended_box_records_that_it_was_amended[box-1] · covers: M2, A3, E5, R:SILENTREWRITE · an undated rewrite is indistinguishable from the box always saying that.
- tests.unit.test_m2_box_accounting::test_every_amended_box_records_that_it_was_amended[box-3] · covers: M2, A3, E5, R:SILENTREWRITE · an undated rewrite is indistinguishable from the box always saying that.
- tests.unit.test_m2_box_accounting::test_every_amended_box_records_that_it_was_amended[box-4] · covers: M2, A3, E5, R:SILENTREWRITE · an undated rewrite is indistinguishable from the box always saying that.
- tests.unit.test_m2_box_accounting::test_no_amended_box_claims_more_than_its_checks_establish[box-1] · covers: R:WIDEN, E6 · the box-6 guard, applied to three more boxes.
- tests.unit.test_m2_box_accounting::test_no_amended_box_claims_more_than_its_checks_establish[box-3] · covers: R:WIDEN, E6 · the box-6 guard, applied to three more boxes.
- tests.unit.test_m2_box_accounting::test_no_amended_box_claims_more_than_its_checks_establish[box-4] · covers: R:WIDEN, E6 · the box-6 guard, applied to three more boxes.
- tests.unit.test_m2_box_accounting::test_no_amended_box_states_a_claim_a_probe_refutes · covers: M5, R:ROUNDING · every transformation and symbol a box names is executed.
- tests.unit.test_m2_box_accounting::test_the_finished_nodes_checks_are_untouched · covers: M6 · this node changes what the milestone says, not what the tasks did.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
