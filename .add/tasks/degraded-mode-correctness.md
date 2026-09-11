---
type: Task
title: The failure sentinel is a valid empty result on every engine
status: done
depth: standard
sensitivity: architecture
milestone: m2-survive-week-two
scope:
  - src/yowo/engine.py
  - src/yowo/classify_engine.py
  - src/yowo/obb_engine.py
  - src/yowo/postprocess/
  - src/yowo/types.py
  - tests/
  - .add/milestones/
gives:
  - S1 the failure sentinel — its shape per engine, and who decides it
  - S2 the degraded classification result — how a consumer tells it from a real prediction
  - S3 m2 box 8 — what "output that postprocess accepts" is worth
depends_on:
  - /tasks/real-backend-smoke.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-11, act: interview, authority: human, interview: "sha256:bcd9aa886fc5e458", receipt: /tasks/degraded-mode-correctness.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A19=confirm|A18=confirm|R:FABRICATE=confirm|R:BATCHDROP=confirm|R:BYPASS=confirm|R:SILENT=confirm|R:SILENTREWRITE=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:a4fbf8977d4c8617", binding: "sha256:07584ee4ddac5e57" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:b756a91b5161adf2" }
  - { by: "builder", at: 2026-09-11, act: replan, authority: process, note: "Build found a check that bound A4/E4 but could not fail. test_the_obb_sentinel_is_empty_structurally_not_by_threshold asserted zero boxes at confidence 0.0; measured, the one-zero-anchor sentinel also yields zero boxes there because the class-score filter is a strict >, so the mutation survived. Rewritten to assert zero ANCHORS structurally and to drive both shapes through postprocess_obb at -1.0, with the one-anchor control proving the assertion discriminates. Also: obb_engine.py now imports numpy at runtime rather than only under TYPE_CHECKING, because the sentinel is built on a live path. No Must, Reject, Edge or gives: wording changed." }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:61ff13774014eade", binding: "sha256:07584ee4ddac5e57" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:8e5245f783babbdd" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/degraded-mode-correctness.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: human, outcome: PASS, receipt: /tasks/degraded-mode-correctness.d/runs/1.md, brief: "sha256:1a3620688317a0ce" }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:60c3835ec7fd84e4", binding: "sha256:07584ee4ddac5e57" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:5b8de90051b9cdc9" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/degraded-mode-correctness.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: human, outcome: PASS, receipt: /tasks/degraded-mode-correctness.d/runs/2.md, brief: "sha256:5b8de90051b9cdc9" }
advised_by: inference-parity-engineer
---
## CARD
goal: When a backend fails after load, every engine produces a result its own postprocess accepts, sized to the batch it was given, and a consumer can tell the classification one apart from a real prediction.
why: `BaseEngine._infer_with_retry` (`engine.py:779`) retries three times and, on exhaustion, returns `np.zeros((1, 0, 6))`. Its docstring says this is "an empty result array so the engine does not crash". Measured against a backend that loads and warms up cleanly and then fails every inference, it crashes on all three engines — only the trigger differs.
  **The sentinel is DETECTION-shaped, so the other two engines cannot read it.** `ClassificationEngine` raises `ValueError: Expected 2-D output (batch, nc), got ndim=3`; `OBBEngine` raises `IndexError: max(): Expected reduction dim 1 to have non-zero size`. Both inherit the method from `BaseEngine` and neither overrides it.
  **The sentinel is BATCH-1-shaped, so detection crashes too as soon as a batch is larger.** At `batch_size=3`, `postprocess` raises `IndexError: index 1 is out of bounds for axis 0 with size 1` at `_nms.py:445`. Detection looked healthy only because every probe used one frame.
  **And the obvious fix for classification would fabricate a prediction.** A correctly-shaped `(batch, nc)` zeros sentinel IS accepted by `postprocess_classify` — measured, it returns `top1_class_id=999, top1_score=0.001`: uniform softmax, argmax taking the last index. That output is indistinguishable from a real low-confidence prediction, so an operator reading a degraded stream sees the model confidently reporting nothing is wrong. Satisfying box 8's literal words this way would be worse than the crash, because a crash is at least visible.
beat: done · next: add status

## RULES
<must>
- M1 The failure sentinel is produced per-engine, not by one hardcoded shape in `BaseEngine`. Detection, classification and OBB each get a raw output their own postprocess accepts.
- M2 The sentinel is sized to the batch actually inferred. A failed batch of N frames produces N results, not one.
- M3 A degraded classification result is identifiable as such by a consumer who has only the result object: `top1_class_id` is -1 and `top1_score` is 0.0, and `ClassificationResult` documents that meaning. It may not present a fabricated class as a prediction.
- M4 The degraded path runs through the SAME postprocess the healthy path runs through. A separate bypass would mean the failure path is the one code nobody exercises.
- M5 The failure stays observable: the error is counted and the "error" event is emitted, exactly as today. A valid empty result must not become a silent one.
- M6 Box 8 says what the sentinel establishes — accepted AND identifiable — records that it was amended and why, and every claim it makes is EXECUTED by a check.
- M7 No healthy-path output changes. A real inference produces exactly what it produced before, on all three engines.
</must>
<reject>
- R:FABRICATE Presenting a degraded result as a real prediction — a class id and a score a caller cannot distinguish from the model's own output. -> "FABRICATE"
- R:BATCHDROP A failed batch of N frames yielding fewer than N results. -> "BATCHDROP"
- R:BYPASS Routing the degraded path around postprocess so the failure path and the healthy path share no code. -> "BYPASS"
- R:SILENT Making the failure quieter than it is today — dropping the error count or the error event. -> "SILENT"
- R:SILENTREWRITE Amending box 8 without recording that it was amended and why. -> "SILENTREWRITE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who chooses the sentinel shape; taking EACH ENGINE, via a hook on `BaseEngine` that subclasses override, because the shape is a property of the task and `BaseEngine` cannot know it -> if wrong and one shape should serve all, no such shape exists: `(batch, 0, 6)` and `(batch, nc)` have different ranks. · probe: each engine returns its own shape.
- A2 [which] covers: S1 · the request does not say which engines are covered; taking ALL THREE that exist — detection, classification, OBB — because box 8 names all three -> if wrong, a fourth engine added later inherits a base default; the default stays detection-shaped, which is what the base class is.
- A3 [when] covers: S1 · the request does not say when the sentinel is produced; taking ON RETRY EXHAUSTION ONLY, inside `_infer_with_retry`, leaving the retry count and backoff untouched -> if wrong and it should be produced earlier, failures that a retry would have fixed become degraded results.
- A4 [absent] covers: S1 · the request does not say what "empty" means for OBB; taking ZERO ANCHORS — `(batch, 4+nc+1, 0)` — measured to yield 0 boxes, rather than one zero-filled anchor which yields 0 boxes only because it falls under the confidence threshold -> if wrong, a threshold change would start emitting phantom boxes. · probe: the OBB sentinel produces zero boxes with the confidence threshold at 0.
- A5 [order] covers: S1 · the request does not say whether result order matters; taking FRAME ORDER PRESERVED — result i belongs to frame i, as on the healthy path — because the router keys on position -> if wrong, a degraded batch misattributes results across sources.
- A6 [experience] covers: S1 · the request does not say who meets this; taking an operator whose stream went quiet — the difficulty is that today the traceback names postprocess, so they debug the wrong module -> so the sentinel must not raise at all, and the error the backend actually raised stays in the log.
- A7 [who] covers: S2 · the request does not say who reads the degraded marker; taking any consumer of a `ClassificationResult` — including one deserialising it from a payload, with no access to logs or metrics -> if wrong and only the local caller matters, the marker is redundant but harmless.
- A8 [which] covers: S2 · the request does not say which fields mark it; taking `top1_class_id = -1` and `top1_score = 0.0` together, because -1 is not a valid class index and 0.0 is not reachable through softmax -> if wrong, a caller keying on one field alone misreads it. · probe: the marker is unreachable from a healthy inference.
- A9 [absent] covers: S2 · the request does not say what `topk_class_ids` and `all_probs` hold when degraded; taking EMPTY for topk and the zero vector for all_probs, so nothing iterates into a fabricated ranking -> if wrong, a consumer looping topk silently sees nothing, which is the intended reading.
- A10 [when] covers: S2 · the request does not say when the marker is applied; taking INSIDE `postprocess_classify`, keyed on an all-zero probability row, so the degraded path and the healthy path share the same function (M4) -> if wrong and the engine should mark it, postprocess stays a pure function of its input and the engine grows a special case. · probe: postprocess alone, given the sentinel, returns the marker.
- A11 [order] covers: S2 · n/a · the marker is per-result; no ordering applies.
- A12 [experience] covers: S2 · the request does not say who is harmed by a fabricated prediction; taking someone acting on a classification — the difficulty is that `top1_score=0.001` looks like a real low-confidence call, so a threshold filter hides the outage instead of revealing it -> so the marker must be a value no threshold comparison treats as ordinary.
- A13 [absent] covers: S2 · the request does not say whether a REAL all-zero logit row should be read as degraded; taking YES, and naming it: an all-zero row carries no information, so "no prediction" is a more honest reading of it than uniform-argmax-picks-the-last-index -> if wrong, a model that genuinely emits zeros is reported as degraded, which is the same thing said differently.
- A14 [who] covers: S3 · the request does not say who may amend box 8; taking the human maintainer, as with m1 box 6 and m2 boxes 1, 3, 4 and 7 -> if wrong and process authority suffices, the cost is one interview round.
- A15 [which] covers: S3 · the request does not say which part of box 8 changes; taking the addition of IDENTIFIABLE to ACCEPTED, keeping the three-engine clause, because "accepted" alone is satisfied by the fabricating sentinel this node rejects -> if wrong, the box permits exactly the output R:FABRICATE forbids. · probe: the amended box names both properties.
- A16 [when] covers: S3 · the request does not say whether the amendment is dated; taking YES, in the AMENDED format m1 box 2 established -> if wrong, the milestone reads tidier and its history is gone.
- A19 [absent] covers: S3 · the request does not say what box 8 claims about engines that do not exist yet; taking NOT CLAIMED — the box names the three engines it was written against, and a fourth engine would inherit the base detection-shaped default rather than a guarantee -> if wrong, a reader takes the box as covering whatever engines the repo grows.
- A17 [order] covers: S3 · n/a · a box is one criterion; its clauses are a conjunction.
- A18 [experience] covers: S3 · the request does not say who reads box 8; taking whoever is deciding whether yowo is safe to run unattended — the difficulty is that "postprocess accepts the output" sounds like the failure is handled, when the measured output was a confident wrong answer -> so the box must say the result is identifiable, not merely accepted.

## PLAN
contract: `BaseEngine._empty_raw_output(batch_size: int) -> NDArray[np.float32]` — a new protected hook returning the detection shape `(batch_size, 0, 6)` by default. `ClassificationEngine` overrides it with `(batch_size, num_classes)`; `OBBEngine` with `(batch_size, 4 + num_classes + 1, 0)`. `_infer_with_retry` calls the hook with the batch it was handed instead of returning a literal. `postprocess_classify` maps an all-zero probability row to `top1_class_id=-1`, `top1_score=0.0`, empty `topk_*`, and `ClassificationResult`'s docstring records that meaning.
strategy: red-first throughout — every check here fails on the current tree, because every one of them describes a crash that happens today. The healthy path is pinned first (M7) so the fix cannot be mistaken for a behaviour change.

## EDGES
- E1 A failed batch of N > 1 frames — N results, in frame order, on every engine.
- E2 A failed batch of exactly 1 frame — the case that masked the batch bug; still one result.
- E3 Classification degraded with `top_k` larger than the class count — no fabricated ranking, empty topk.
- E4 OBB degraded with the confidence threshold at 0.0 — still zero boxes, so the emptiness is structural and not threshold-dependent.
- E5 A healthy inference on each engine — byte-identical results to before the change.
- E6 The backend recovers on retry 2 — no sentinel at all, the real output is returned.
- E7 A degraded result serialised and read back with no access to logs — the marker survives.
- E8 Box 8 carries an AMENDED clause with no date, or claims acceptance without identifiability — the box checks redden.

## CHECKS
all in `tests/unit/test_degraded_mode.py` except the box checks. A `FailsAfterLoad` backend loads and warms up cleanly, then fails every inference — the only way to reach the sentinel, since an always-broken backend is caught by warmup validation.

- tests.unit.test_degraded_mode::test_a_failed_batch_yields_one_result_per_frame[detection] · covers: M2, A5, E1, R:BATCHDROP · 3 frames in, 3 results out, in frame order.
- tests.unit.test_degraded_mode::test_a_failed_batch_yields_one_result_per_frame[classification] · covers: M2, A5, E1, R:BATCHDROP · 3 frames in, 3 results out, in frame order.
- tests.unit.test_degraded_mode::test_a_failed_batch_yields_one_result_per_frame[obb] · covers: M2, A5, E1, R:BATCHDROP · 3 frames in, 3 results out, in frame order.
- tests.unit.test_degraded_mode::test_a_failed_single_frame_batch_still_yields_one_result · covers: E2 · the batch size that masked the bug.
- tests.unit.test_degraded_mode::test_each_engine_produces_a_sentinel_its_own_postprocess_accepts[detection] · covers: M1, M4, A1, A2, A6 · the sentinel goes through the real postprocess and returns an empty result rather than raising.
- tests.unit.test_degraded_mode::test_each_engine_produces_a_sentinel_its_own_postprocess_accepts[classification] · covers: M1, M4, A1, A2, A6 · the sentinel goes through the real postprocess and returns an empty result rather than raising.
- tests.unit.test_degraded_mode::test_each_engine_produces_a_sentinel_its_own_postprocess_accepts[obb] · covers: M1, M4, A1, A2, A6 · the sentinel goes through the real postprocess and returns an empty result rather than raising.
- tests.unit.test_degraded_mode::test_the_degraded_path_is_not_a_postprocess_bypass · covers: M4, R:BYPASS · patching the engine's postprocess is observed by the degraded path, so the two paths share code.
- tests.unit.test_degraded_mode::test_a_degraded_classification_is_not_a_prediction · covers: M3, A8, A12, R:FABRICATE · `top1_class_id` is -1 and `top1_score` is 0.0 — not the measured `999 @ 0.001`.
- tests.unit.test_degraded_mode::test_the_degraded_marker_is_unreachable_from_a_healthy_inference · covers: A8, A12, R:FABRICATE · a real inference can produce neither -1 nor 0.0, so the marker is unambiguous.
- tests.unit.test_degraded_mode::test_a_degraded_classification_ranks_nothing · covers: A9, E3 · empty topk and a zero probability vector; nothing iterates into a fabricated ranking.
- tests.unit.test_degraded_mode::test_classification_result_documents_the_degraded_marker · covers: M3 · ADDED DURING BUILD. M3 requires that `ClassificationResult` document the marker's meaning, and no check bound that half of the rule — the gate passed with the docstring untouched. A consumer deserialising a result reads the type, not postprocess.
- tests.unit.test_degraded_mode::test_postprocess_classify_alone_returns_the_marker · covers: A10, E7 · the marker is a property of postprocess, not of engine state, so it survives serialisation and needs no log.
- tests.unit.test_degraded_mode::test_the_obb_sentinel_is_empty_structurally_not_by_threshold · covers: A4, E4 · REWRITTEN DURING BUILD. At threshold 0.0 the check could not fail: the class-score filter is a strict `>`, so a zero-filled anchor is dropped there too, and the mutation swapping zero anchors for one zero anchor passed. It now asserts the shape carries zero anchors AND drives both shapes through `postprocess_obb` at threshold -1.0, where the one-anchor form emits a phantom box and the zero-anchor form cannot — the control that makes the assertion discriminating.
- tests.unit.test_degraded_mode::test_a_recovered_retry_returns_the_real_output · covers: A3, E6 · a backend that fails once and then succeeds produces its real result, not a sentinel.
- tests.unit.test_degraded_mode::test_the_failure_is_still_counted_and_emitted · covers: M5, R:SILENT · `errors_total` increments and the "error" event fires, as today.
- tests.unit.test_degraded_mode::test_a_healthy_inference_is_unchanged[detection] · covers: M7, E5 · the healthy path is pinned before the fix, so the fix cannot hide a behaviour change.
- tests.unit.test_degraded_mode::test_a_healthy_inference_is_unchanged[classification] · covers: M7, E5 · the healthy path is pinned before the fix, so the fix cannot hide a behaviour change.
- tests.unit.test_degraded_mode::test_a_healthy_inference_is_unchanged[obb] · covers: M7, E5 · the healthy path is pinned before the fix, so the fix cannot hide a behaviour change.
- tests.unit.test_degraded_mode::test_box_8_records_that_it_was_amended · covers: M6, A16, R:SILENTREWRITE, E8 · an undated rewrite reads as the original.
- tests.unit.test_degraded_mode::test_box_8_claims_identifiable_not_merely_accepted · covers: M6, A15, A18, E8 · "accepted" alone is satisfied by the fabricating sentinel this node rejects.
- tests.unit.test_degraded_mode::test_box_8_claims_nothing_a_probe_refutes · covers: M6 · every engine and symbol the box names is resolved and driven.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
