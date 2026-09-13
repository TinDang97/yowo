---
type: Task
title: COCO mAP counts missed images as recall misses
status: done
depth: standard
milestone: m3-prove-it
scope:
  - src/yowo/benchmark/
  - tests/unit/
gives:
  - S1 evaluate_coco_map scores a model over the dataset, not over its own output
  - S2 the subset parameter selects the evaluated image set
generated: { by: add/3.5.0, at: 2026-09-13 }
verified:
  - { by: "Tin Dang", at: 2026-09-13, act: freeze, authority: plan, direction: "sha256:90314ea0fa0439d8", binding: "sha256:a50d2a9e14d8036a" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:a1a85e3f8a36656c" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/coco-map-evaluator-repair.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: plan, direction: "sha256:09c0611c1a74961b", binding: "sha256:a50d2a9e14d8036a" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:8b655083c3b0d3fd" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/coco-map-evaluator-repair.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-13, act: gate, authority: process, outcome: PASS, receipt: /tasks/coco-map-evaluator-repair.d/runs/2.md, brief: "sha256:8b655083c3b0d3fd" }
advised_by: security-reviewer
---
## CARD
goal: A model that stops detecting must score worse, not better.
why: Measured 2026-09-13. `_evaluator.py:45` sets `coco_eval.params.imgIds` to only the images that
  HAVE predictions, deleting every missed image from the denominator instead of counting it as a
  recall miss. Measured against real `COCOeval` on synthetic ground truth: a model detecting its
  object in 1 of 10 images scores **0.99999999**, against a perfect model's **1.0**. A regression
  gate on this number REWARDS a model that stops detecting — strictly worse than no gate at all.
  The inversion was deliberate and commented ("so unscored images don't drag down recall"), which is
  why it survived review; no test caught it because `test_benchmark_evaluator.py:219-256` mocks
  `pycocotools` entirely, so `COCOeval` never executes in the suite.
  This blocks `map-regression-gate`, whose first beat is therefore this repair and not the gate.
  It is separated from that node because it needs NO dataset: synthetic ground truth discriminates
  the defect exactly, so the repair can land before the 1.07 GB COCO decision is acted on.
  Second defect in the same function: `subset` is documented "Unused (kept for API compatibility)"
  while the milestone's chosen dataset posture is a pinned 500-image subset, so the parameter the
  plan depends on currently does nothing.
beat: done · next: add status

## RULES
<must>
- M1 An image in the evaluated set with ground truth and NO prediction counts as a recall miss. A model that detects in fewer images scores strictly lower, never higher.
- M2 `subset=N` selects N images from the GROUND TRUTH deterministically, and the score is computed over exactly those N — so a pinned subset is reproducible across runs and machines.
- M3 `subset=None` evaluates every image in the ground truth.
</must>
<reject>
- R:SCOPED_BY_OUTPUT The evaluated image set is derived from the predictions rather than from the ground truth -> "SCOPED_BY_OUTPUT"
- R:MOCKED_COCOEVAL A check proves M1 with `pycocotools` patched. The defect survived precisely because the existing tests mock `COCOeval`, so a mocked check cannot bind this rule -> "MOCKED_COCOEVAL"
</reject>

## ASSUMPTIONS
- A1 [who] n/a · evaluation has no actor distinction; the score depends only on predictions and ground truth.
- A2 [which] covers: S2 · the request does not say WHICH N images a subset takes; taking the first N ground-truth image ids in sorted order, since a pinned baseline must be reproducible and any random or insertion-ordered choice would make two runs incomparable · probe: the same subset value selects the same ids twice -> a non-deterministic subset makes every recorded baseline meaningless.
- A3 [when] n/a · single synchronous call, no time boundary.
- A4 [absent] covers: S1 · the request does not say what an EMPTY prediction list means; taking it as a real score of 0.0 over the evaluated set, which after this repair is what the full evaluation genuinely computes — rather than the pre-repair sentinel, which returned the same 0.0 without evaluating anything · probe: the empty-list result equals what the evaluated path returns for predictions that match nothing -> otherwise 0.0 stays indistinguishable from "did not run".
- A5 [order] covers: S2 · the request does not say what orders the subset; taking sorted image id, the same key the deterministic selection uses.
- A6 [experience] covers: S1 · the request does not say who reads this; taking the operator reading a regression gate, for whom a number that rises when detection degrades is worse than no number.
- A7 [which] covers: S1 · the request does not say WHICH ground-truth annotations count; taking every annotation COCOeval itself counts under its default bbox params, including its `iscrowd` handling — this node repairs the image scope only and changes no annotation filtering · probe: the perfect-model score stays 1.0 after the repair -> silently changing which annotations count would move every historical number for a second, unrelated reason.
- A8 [absent] covers: S2 · the request does not say what `subset=0` means; taking it as an empty evaluated set rather than as `None`, so "evaluate nothing" and "evaluate everything" stay distinguishable · probe: `subset=0` does not silently become a full-dataset run -> a 0 read as None would turn a typo into a full 5000-image CI run.
- A9 [order] covers: S1 · the request does not say what breaks a tie between equally-scoring detections; taking COCOeval's own score-descending match order unchanged, since this node moves the image scope and nothing inside the matcher.
- A10 [experience] covers: S2 · the request does not say who sets `subset`; taking the CI author pinning a 500-image evaluation for runtime, who needs the same 500 images on every run and on every machine or the recorded baseline is not comparable.

## PLAN
contract: `evaluate_coco_map` derives `imgIds` from the ground truth (`coco_gt.getImgIds()`), optionally truncated to the first `subset` ids in sorted order, and never from `predictions`. Missed images then remain in COCOeval's denominator and register as recall misses, which is what COCO mAP means.

## EDGES
- E1 A model that detects in 1 of 10 images scores strictly below a model that detects in all 10 — the exact inversion measured today.
- E2 `subset=N` larger than the ground-truth image count evaluates every image rather than raising.
- E3 A prediction referencing an image id outside the evaluated subset does not inflate the score.
- E4 Empty predictions over a non-empty ground truth score 0.0 by evaluation, not by sentinel.

## CHECKS
- test_a_model_that_misses_images_scores_lower · covers: M1, E1, R:SCOPED_BY_OUTPUT, R:MOCKED_COCOEVAL · drives REAL COCOeval on synthetic ground truth; 1-of-10 must score strictly below 10-of-10.
- test_subset_selects_from_the_ground_truth · covers: M2, A2, A5 · `subset=N` evaluates N ground-truth images regardless of how many the model predicted on.
- test_subset_is_deterministic · covers: M2, A2 · the same subset value selects the same ids on repeated calls.
- test_subset_none_evaluates_every_image · covers: M3 · the full ground truth is scored.
- test_a_subset_larger_than_the_dataset_is_clamped · covers: E2 · no raise.
- test_predictions_outside_the_subset_do_not_inflate_the_score · covers: E3 · out-of-scope predictions are ignored.
- test_empty_predictions_score_zero_by_evaluation · covers: A4, E4 · the empty result equals the evaluated result for non-matching predictions.
- test_a_perfect_model_still_scores_one · covers: A7, A9 · the repair moves image scope only; annotation matching is unchanged.
- test_subset_zero_is_an_empty_set_not_a_full_run · covers: A8, A10 · 0 is not silently read as None.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- A metric that scopes itself to the model's own output measures precision and calls it accuracy. Restricting COCOeval to images that have predictions removed every miss from the denominator, so degrading recall RAISED the score. The comment explaining the choice ("so unscored images don't drag down recall") is the defect stated aloud and shipped anyway. -> add learn tdd
