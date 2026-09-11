---
type: Task
title: The CI smoke executes a real backend on a real image, and says which one
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - tests/
  - .add/milestones/
  - .github/workflows/
gives:
  - S1 the CI-run smoke — what it executes, on what input, and what a broken backend fails
  - S2 how a missing backend, weight or image is reported — in CI versus locally
  - S3 the recorded fact that the default selection falls back on a `.pt` weight, and which backend the fallback lands on
  - S4 m2 box 7 — what "a real backend is executed" is worth as a regression baseline
depends_on:
  - /tasks/ci-weight-fixture.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-11, act: interview, authority: human, interview: "sha256:4bb55537d599fdee", receipt: /tasks/real-backend-smoke.d/interviews/1.md, answers: "A1=confirm|A2=confirm" }
  - { by: "unrecorded", at: 2026-09-11, act: interview, authority: human, interview: "sha256:4bb55537d599fdee", receipt: /tasks/real-backend-smoke.d/interviews/2.md, answers: "A3=confirm|A4=confirm|A5=confirm|A6=confirm|A25=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|A19=confirm|A20=confirm|A21=confirm|A22=confirm|A24=confirm|R:GREENSKIP=confirm|R:VACUOUS=confirm|R:WIDEN=confirm|R:SILENTREWRITE=confirm|R:BEHAVIOUR=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:7c7c76da9bf750c0", binding: "sha256:31467aa0fb417106" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:e0fe13f775860917" }
  - { by: "builder", at: 2026-09-11, act: replan, authority: process, note: "Build found a check that bound M2 but could not fail. test_the_smoke_executes_the_backend_it_asked_for asserted selection.backend is PYTORCH; measured, removing the backend= override kept it green, because engine.load REWRITES self._selection with the backend it fell back to. The check now also asserts the selection was not reached by fallback, and a new check pins the reason prefix engine.py writes so the guard cannot be disarmed by a reword. One CHECKS line added; no Must, Reject, Edge or gives: wording changed." }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:e285982ac0226626", binding: "sha256:31467aa0fb417106" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:106d603d4285087b" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/real-backend-smoke.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: human, outcome: PASS, receipt: /tasks/real-backend-smoke.d/runs/1.md, brief: "sha256:2b877a78a377b00c" }
advised_by: inference-parity-engineer
---
## CARD
goal: The backend CI executes is the one the test asked for, on an input whose result a broken backend cannot produce, in a job that cannot report green without running it.
why: Box 7's literal words — "at least one real backend is constructed and executed by a test — no mock" — are **already satisfied**, by `ci-weight-fixture`, before this task began. `tests/integration/test_engine_integration.py` runs in CI's Weight Fixture job, loads a digest-verified `yolo26n` and calls `detect()`. That is the third m2 box in a row whose words are true and whose value is not, and ticking it would buy the two tasks below it nothing. Three measured facts say why.
  **The backend that runs is not chosen — it is whatever survives a failure.** `select_backend` picks ONNX here (`ONNX Runtime 1.24.2 (CoreML EP — Apple Neural Engine)`); `OnnxBackend.load` is then handed a `.pt` and raises `BackendLoadError: ... INVALID_PROTOBUF : Load model from .../yolo26n.pt failed`; `engine.py` logs `Primary backend onnx failed, trying pytorch` and continues. `test_detection_has_backend_type` asserts `det.backend == BackendType.PYTORCH` and passes **because** of that failure. Nothing anywhere records that a fallback occurred.
  **The execution asserts almost nothing.** The frame is a 640x640 black image — measured, it yields **0 boxes**. The checks are `isinstance(result, list)` and `inference_time_ms > 0`; a backend returning an empty list for every input passes all of them. The real image is already a session fixture (`sample_image_path`, bus.jpg) and measures **5 boxes, classes {bus, person}** — a result a broken backend cannot fake. The assertions that would catch a broken one (`x1 < x2`, `class_name in COCO_CLASSES`) live in `test_cli_e2e.py`, which **CI does not run**: the job names `tests/integration/test_engine_integration.py` alone.
  **The whole thing can vanish into a green skip.** The `engine` fixture calls `pytest.importorskip("torch")`. `tests/integration/conftest.py` went to deliberate trouble to make a missing *input* fail in CI — its own docstring says "a skip is green, so nothing ever reported it" — and the *backend* import walks straight past that guard one layer up.
beat: done · next: add status

## RULES
<must>
- M1 A real backend is constructed through `create_backend` and executes inference on a REAL image inside the job CI runs, and the assertions are ones a broken backend fails: at least one box, and every box satisfying `x1 < x2`, `y1 < y2`, `0.0 <= confidence <= 1.0` and `class_name in COCO_CLASSES`.
- M2 The smoke asserts WHICH backend produced the result, and that it is the backend the test asked for by explicit override — not whatever a fallback left behind.
- M3 The fallback is pinned as a recorded fact: a check drives the DEFAULT selection against a `.pt` weight and asserts what actually happens today — the primary selection is not the backend that ran. If that ever changes, the check reddens and a human looks, instead of the change passing unnoticed.
- M4 No leg of this smoke can report green in CI without executing a backend. A missing backend, weight or image FAILS in CI and skips locally — the asymmetry `tests/integration/conftest.py` already states, applied to the backend import too.
- M5 CI actually invokes the smoke. A test file no workflow names proves nothing, which is the defect `test_cli_e2e.py` demonstrates today.
- M6 Box 7 says what the smoke establishes, records that it was amended and why, and every claim it makes is EXECUTED by a check rather than restated.
- M7 No runtime selection or fallback BEHAVIOUR changes in this node. Whether picking a backend that cannot load the given weight should warn or raise is `degraded-mode-correctness`'s call; this node makes the current answer visible.
</must>
<reject>
- R:GREENSKIP A CI leg that reports green while executing no backend. Scoped to the legs that MUST execute one — the smoke itself, its weight and its image. The fallback tripwire is not such a leg: it records a host-dependent fact, so where the primary backend already loads the weight it skips with a reason rather than failing for a difference that is not a defect (A15). -> "GREENSKIP"
- R:VACUOUS An assertion a backend returning nothing would satisfy — a black frame, an `isinstance(..., list)`, a bare truthiness. -> "VACUOUS"
- R:WIDEN Claiming the smoke covers backends it does not execute. -> "WIDEN"
- R:SILENTREWRITE Amending box 7 without recording that it was amended and why. -> "SILENTREWRITE"
- R:BEHAVIOUR Changing selection or fallback behaviour under cover of a test-scoped node. -> "BEHAVIOUR"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who runs the smoke; taking CI on every PR plus any contributor locally, because a smoke only one machine runs is the defect the integration conftest was written to fix -> if wrong and it is CI-only, a contributor gets a slow local suite.
- A2 [which] covers: S1 · the request does not say which backend the smoke executes; taking PYTORCH by explicit override, because it is the only one that loads a `.pt` and the box asks for "at least one" -> if wrong and every backend is wanted, that is a wider box and a different node. · probe: the smoke names its backend and asserts the result carries it.
- A3 [which] covers: S1 · the request does not say which image; taking the existing `sample_image_path` fixture (bus.jpg), not a synthesised one, because a black frame measures 0 boxes and asserts nothing -> if wrong and a committed fixture is wanted, the licence problem the conftest documents returns. · probe: the smoke's input yields boxes.
- A4 [absent] covers: S1 · the request does not say what a zero-box result means; taking FAILURE — the image contains a bus and people, so zero boxes is a broken backend, not an empty scene -> if wrong and the model legitimately finds nothing, the smoke is flaky. Measured today: 5 boxes, {bus, person}.
- A5 [order] covers: S1 · the request does not say whether box order is asserted; taking NOT ASSERTED — NMS ordering is not this node's contract and pinning it would make the smoke brittle against a postprocess change -> if wrong, an ordering regression passes here and is caught by `m3`'s accuracy work.
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking someone who broke a backend and needs to know from the message which backend, which input and which invariant failed -> so each assertion names them, rather than failing as a bare `assert len(boxes)`.
- A25 [when] covers: S1 · the request does not say when the smoke runs or whether a red one blocks a merge; taking EVERY PR, in the existing Weight Fixture job, blocking exactly as the unit gate does, because a smoke that runs after the merge tells you what you already shipped -> if wrong and it is meant to be nightly, PR feedback slows by the job's runtime (40s today).
- A7 [who] covers: S2 · the request does not say who decides fail-versus-skip; taking the environment — `CI` set means fail, unset means skip — reusing `conftest._unavailable` rather than writing a second policy -> if wrong, two guards drift apart.
- A8 [which] covers: S2 · the request does not say which absences are covered; taking all three the smoke depends on — the torch import, the weight and the image — because any one of them missing makes the run prove nothing -> if wrong, one uncovered absence is the hole the whole rule exists to close. · probe: the backend import is guarded by the same policy as the weight.
- A9 [absent] covers: S2 · the request does not say what a skip should say; taking a message naming what was unavailable and why, because "skipped" alone is what let this rot -> if wrong, the message is noise.
- A10 [when] covers: S2 · the request does not say when the guard is evaluated; taking FIXTURE SETUP, so the failure is attributed to the missing input rather than surfacing as an unrelated error inside a test body -> if wrong, a reader debugs the wrong thing.
- A11 [order] covers: S2 · n/a · the three absences are independent; whichever is reported first is equally actionable.
- A12 [experience] covers: S2 · the request does not say who hits the skip; taking a contributor without network on a first clone — the difficulty is that a skip looks like a pass, so the message must say the run proved nothing -> if wrong, the wording is merely verbose.
- A13 [who] covers: S3 · n/a · a recorded fact authorises nobody; it only reddens when it stops being true.
- A14 [which] covers: S3 · the request does not say which property of the fallback to pin; taking the OUTCOME — that the backend which ran is not the primary selection — rather than the log text, because a log line is not a contract and asserting on it would break on a reword -> if wrong, a silent change of *which* fallback is taken passes. · probe: the check reads the selection, not the log.
- A15 [absent] covers: S3 · the request does not say what happens where ONNX is absent or where it *can* load the weight; taking a check that asserts the fallback only where the primary is genuinely not the backend that ran, and skips with a reason otherwise -> if wrong, the check is red on a machine whose selector picks PyTorch first, which is a CI-host difference and not a defect. This is the one skip R:GREENSKIP permits, and R:GREENSKIP names it.
- A16 [when] covers: S3 · the request does not say when this stops being a fact; taking "when a backend that loads a `.pt` is selected first, or the selector stops offering one that cannot" — either is a real change someone should see -> if wrong, the check is treated as a bug rather than a tripwire, so it says so in its own message.
- A17 [order] covers: S3 · the request does not say whether the fallback CHAIN order is pinned; taking NOT PINNED — only that a fallback occurred and where it landed — because the chain is `degraded-mode-correctness`'s subject -> if wrong, a chain reorder passes here.
- A18 [experience] covers: S3 · the request does not say who reads this check; taking whoever changed the selector and saw it go red — the difficulty is concluding they broke something when they may have FIXED it -> so the message must say the tripwire records today's behaviour and asks for a look, not a revert.
- A19 [who] covers: S4 · the request does not say who may amend box 7; taking the human maintainer, as with m1 box 6 and m2 boxes 1, 3 and 4 -> if wrong and process authority suffices, the cost is one interview round.
- A20 [which] covers: S4 · the request does not say which part of box 7 changes; taking the CLAIM — from "at least one real backend is constructed and executed" to what the smoke establishes — while KEEPING the purpose clause about the two tasks below, because that clause is why the box exists -> if wrong, the box loses the reason it was written. · probe: the amended box still names what it is a baseline for.
- A21 [when] covers: S4 · the request does not say whether the amendment is dated; taking YES, in the AMENDED format m1 box 2 established -> if wrong, the milestone reads tidier and its history is gone.
- A22 [absent] covers: S4 · the request does not say what the box claims about backends the smoke does not execute; taking NOT CLAIMED, named explicitly — ONNX, TensorRT, OpenVINO and CoreML are constructed by no unmocked check -> if wrong, a reader takes the box as a whole-matrix guarantee.
- A23 [order] covers: S4 · n/a · a box is one criterion; its clauses are a conjunction with no ordering a reader could apply wrongly.
- A24 [experience] covers: S4 · the request does not say who reads box 7; taking whoever picks up `degraded-mode-correctness` and asks what they can regress against — the difficulty is that "a real backend is executed" sounds like a working baseline when the execution asserted nothing -> so the box must state what the smoke actually pins.

## PLAN
contract: `tests/integration/test_real_backend_smoke.py` — the CI-run smoke. A module fixture builds an `InferenceEngine` with an explicit `backend=pytorch` override on the digest-verified weight; the checks execute it on `sample_image_path` and assert the box invariants. A second fixture in `tests/integration/conftest.py` guards the backend import through the existing `_unavailable` policy so a missing torch fails in CI. A separate check drives the DEFAULT selection and records that it falls back. `.github/workflows/ci.yml`'s Weight Fixture step is widened to name the new file. `.add/milestones/m2-survive-week-two.md` box 7 is amended and ticked.
strategy: red-first — every check written against the current tree fails for the reason it names before any of it is built. The fallback tripwire is the exception worth calling out: it asserts today's measured behaviour, so it must be shown red by asserting the opposite first, then inverted.

## EDGES
- E1 The image fixture is unavailable (no network, cold cache) — CI fails with a message saying the run proved nothing; local skips with the same message.
- E2 torch is not installed — CI fails rather than skipping; local skips with a reason.
- E3 A host whose selector picks PyTorch first — the fallback tripwire finds no fallback and skips with a reason, rather than failing.
- E4 The detection result is an empty box list — fails, naming the image and that a bus and people are in it.
- E5 A box that violates an invariant — fails naming the box index, the invariant and the value.
- E6 The weight resolves but the backend cannot load it — fails as a load error attributed to the backend, not as an empty result.
- E7 Box 7 carries an AMENDED clause with no date, or claims a backend the smoke does not execute — the box checks redden.

## CHECKS
two files. `tests/integration/test_real_backend_smoke.py` EXECUTES a backend and runs in CI's Weight Fixture job; `tests/unit/test_real_backend_smoke_contract.py` holds the box, the workflow wiring and the guard policy, and runs in the fast gate.

- tests.integration.test_real_backend_smoke::test_the_smoke_executes_the_backend_it_asked_for · covers: M2, A2 · the result carries the backend the override named, and the engine's own selection agrees — not a fallback that happened to land there.
- tests.integration.test_real_backend_smoke::test_a_real_image_yields_boxes · covers: M1, A3, A4, E4, R:VACUOUS · bus.jpg through a real backend; a backend returning nothing fails here.
- tests.integration.test_real_backend_smoke::test_every_box_satisfies_its_invariants · covers: M1, A6, E5 · x1<x2, y1<y2, confidence in [0,1], a COCO class name — each failure naming the box index, the invariant and the value.
- tests.integration.test_real_backend_smoke::test_a_black_frame_does_not_satisfy_this_smoke · covers: R:VACUOUS, A4 · the control. The same assertions against the frame the old smoke used must NOT hold, or the new ones prove nothing either.
- tests.integration.test_real_backend_smoke::test_a_backend_that_cannot_load_the_weight_says_so · covers: E6 · handing a `.pt` to ONNX raises a load error naming the backend, rather than yielding an empty result.
- tests.integration.test_real_backend_smoke::test_the_default_selection_falls_back_on_a_pt_weight · covers: M3, A14, A15, A16, A17, A18, E3 · the tripwire: today the backend that ran is not the primary selection. Read from the selection, never the log text; skips with a reason where the primary already loads the weight.
- tests.unit.test_real_backend_smoke_contract::test_ci_invokes_the_smoke_file · covers: M5 · a test file no workflow names proves nothing — the defect `test_cli_e2e.py` shows today.
- tests.unit.test_real_backend_smoke_contract::test_no_leg_of_the_smoke_can_skip_its_way_green · covers: M4, A8, R:GREENSKIP · no `importorskip` guards the backend, the weight or the image on the CI path; each routes through the one policy.
- tests.unit.test_real_backend_smoke_contract::test_a_missing_input_fails_in_ci · covers: M4, A7, A10, E1, E2, R:GREENSKIP · `_unavailable` executed with CI set — it raises.
- tests.unit.test_real_backend_smoke_contract::test_a_missing_input_skips_locally_with_a_reason · covers: A7, A9, A12, E1, E2 · the same call with CI unset skips, and the message names what was unavailable.
- tests.unit.test_real_backend_smoke_contract::test_the_fallback_reason_prefix_the_smoke_guards_on_still_exists · covers: M2 · ADDED DURING BUILD. Asserting `selection.backend` alone did NOT discriminate — measured: with the override removed the engine rewrites the selection with the backend it fell back to, and the check still passed. The guard now reads `selection.reason`, so the prefix `engine.load` writes is pinned to the source here; a reword reddens rather than silently disarming it.
- tests.unit.test_real_backend_smoke_contract::test_box_7_records_that_it_was_amended[date] · covers: M6, A21, R:SILENTREWRITE, E7 · an undated rewrite is indistinguishable from the box always saying that.
- tests.unit.test_real_backend_smoke_contract::test_box_7_records_that_it_was_amended[reason] · covers: M6, A21, R:SILENTREWRITE, E7 · an undated rewrite is indistinguishable from the box always saying that.
- tests.unit.test_real_backend_smoke_contract::test_box_7_names_the_backends_it_does_not_cover · covers: A22, R:WIDEN, E7 · four backends are constructed by no unmocked check, and the box must say so.
- tests.unit.test_real_backend_smoke_contract::test_box_7_keeps_the_clause_saying_what_it_is_a_baseline_for · covers: A20 · the purpose clause is why the box exists.
- tests.unit.test_real_backend_smoke_contract::test_box_7_claims_nothing_a_probe_refutes · covers: M6 · every symbol and file the box names is resolved, not trusted.
- tests.unit.test_real_backend_smoke_contract::test_no_selection_or_fallback_source_changed · covers: M7, R:BEHAVIOUR · `_selector.py` and `engine.load`'s fallback loop are byte-identical to main; this node makes behaviour visible, it does not move it.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
