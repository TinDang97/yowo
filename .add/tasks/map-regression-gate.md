---
type: Task
title: An mAP baseline on a real dataset, enforced
status: direction
depth: deep
milestone: m3-prove-it
scope:
  - tests/
  - src/yowo/benchmark/
  - .github/workflows/
gives:
  - S1 `run_benchmark(model, data, subset=N)` reports an mAP scoped to the N images it evaluated
  - S2 `tests/fixtures/coco_map_baseline.json` — a self-describing recorded baseline
  - S3 an `mAP Gate` CI step that fails a pull request whose measured mAP leaves the recorded band
depends_on:
  - /tasks/accuracy-dataset.md
  - /tasks/ci-weight-fixture.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-15, act: interview, authority: human, interview: "sha256:0e0f009cce2511b2", receipt: /tasks/map-regression-gate.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:UNSCOPED_EVALUATION=confirm|R:SELF_UPDATING_BASELINE=confirm|R:GREEN_BY_SKIP=confirm" }
  - { by: "Tin Dang", at: 2026-09-15, act: freeze, authority: human, direction: "sha256:5ae953a9ae672ab9", binding: "sha256:c45e5a61b60500bc" }
advised_by: inference-parity-engineer
---
## CARD
goal: measure yolo11n's COCO mAP on the pinned 500-image subset, record it as a baseline that describes everything determining it, and fail CI when the number leaves the band.
why: `accuracy-dataset` (PR #51) landed 500 real images and `evaluate_coco_map`'s repair (PR #49) landed a correct scope, but nothing measures a number and nothing defends it. Worse, the PUBLIC path still reports a wrong one — measured 2026-09-15 below.
beat: direction · next: freeze

## RULES
<must>
- M1 the mAP reported for a run is computed against exactly the ground-truth images that run evaluated — never against ground truth it never saw, and never against only the images it happened to predict on
- M2 the recorded baseline names every input that determines the number — model, weights digest, backend, device, confidence and IoU thresholds, image count, and the digest of the subset manifest — so a number that moves can be attributed rather than argued about
- M3 the gate fails when the measured mAP leaves the recorded band in EITHER direction, and the failure message carries the baseline, the measurement and the signed delta
- M4 the gate runs in CI against the pinned 500-image COCO val2017 subset, on the weights the weight fixture pins
</must>
<reject>
- R:UNSCOPED_EVALUATION mAP is never computed against a ground-truth image set derived independently of the predictions' own image list -> "UNSCOPED_EVALUATION"
- R:SELF_UPDATING_BASELINE the gate never writes the baseline it is gating against -> "SELF_UPDATING_BASELINE"
- R:GREEN_BY_SKIP the gate never reports success because the dataset, the weights or pycocotools were absent -> "GREEN_BY_SKIP"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · `run_benchmark` is a library call with no principals; there is no actor whose identity changes the number
- A2 [which] covers: S1 · the request does not say which ground-truth images form the denominator when only a subset was inferred; taking "exactly the images the run evaluated" · probe: `run_benchmark(subset=500)` must report the scoped number, not the full-ground-truth one -> a gate on the wrong scope rewards a model that stops detecting, the exact defect PR #49 fixed one layer down · found 2026-09-15: `run_single_backend:242` calls `evaluate_coco_map(gt_ann_path, coco_results)` with no scope, so the shipped `yowo benchmark --subset 500` reports **0.0419** where the truth is **0.4158** — 4500 of 5000 ground-truth images counted as total misses. The repair landed in the evaluator and stopped at the runner's door.
- A3 [when] covers: S1 · the request does not say where the boundary falls when the requested subset exceeds the dataset; taking "clamp, and report the count actually evaluated" -> a silent clamp that still reports the requested N misattributes the denominator
- A4 [absent] covers: S1 · the request does not say what an omitted scope means; taking "omitted scope evaluates every ground-truth image", preserving today's full-dataset behaviour exactly -> changing the no-argument meaning would silently rescope every existing caller
- A5 [order] covers: S1 · the request does not say what fixes the selection; taking `sorted(getImgIds())[:N]` -> unsorted ids come out of the JSON in file order and the "same" subset differs per machine. NOTE: this rule is now written out in THREE places — `_evaluator.py:199` (evaluate), `_evaluator.py:313` (load), `tests/support/datasets.py:699` (`select_subset_ids`, whose own docstring says it mirrors the evaluator). Passing the evaluated ids instead of re-deriving them collapses two of the three.
- A6 [experience] covers: S1 · the request does not say who reads this; taking "a contributor reading a benchmark table, who has no way to tell 0.0419 from a real regression" -> the defect above shipped precisely because a wrong number looks exactly like a right one
- A7 [who] covers: S2 · the request does not say who may move the baseline; taking "a human, in a reviewed commit, never the gate itself" -> a gate that rewrites its own baseline records history and prevents nothing
- A8 [which] covers: S2 · the request does not say which fields make a baseline; taking M2's list -> a bare float cannot distinguish "the model regressed" from "the weights were re-pinned"
- A9 [when] covers: S2 · the request does not say when a baseline goes stale; taking "when any recorded input changes", which the recorded digests make checkable rather than remembered
- A10 [absent] covers: S2 · the request does not say what a missing baseline file means; taking "the gate fails" -> "no baseline, therefore pass" is R:GREEN_BY_SKIP with extra steps
- A11 [order] covers: S2 · n/a · the baseline is a single record; nothing in it is ordered and nothing ties
- A12 [experience] covers: S2 · the request does not say what the reader needs; taking "the file must be readable without running anything" — JSON with the provenance inline, not a pickle or a bare number
- A13 [who] covers: S3 · n/a · the gate runs as CI; there is no caller identity to distinguish
- A14 [which] covers: S3 · the request does not say which model/backend/device the gate binds; taking "yolo11n · PyTorch · CPU" — the one combination CI can actually run, and the combination the baseline records · probe: the recorded backend and device must match what the gate's own run reports -> a baseline measured on one backend and enforced on another measures the backend, not the model
- A15 [when] covers: S3 · the request does not say the runtime budget; taking "the existing Accuracy Dataset job's 45-minute ceiling" · found 2026-09-15: 500 images inferred in **20.8 s** on local CPU, plus a cached ~1.07 GB dataset restore — the gate is cheap enough to run on every pull request
- A16 [absent] covers: S3 · the request does not say what an absent dataset means in CI; taking "`CI=true` turns a missing dataset from a skip into a failure", the rule `test_coco_dataset.py` already established -> Q3: a skipped test is green
- A17 [order] covers: S3 · n/a · one step, one measurement, nothing sequenced
- A18 [experience] covers: S3 · the request does not say what a failing contributor sees; taking "the message names the baseline, the measurement, the delta and which recorded input to check first" -> a bare `assert 0.41 >= 0.42` sends the reader to the source to learn what the gate even measured

## PLAN
contract:
- `evaluate_coco_map(gt_ann_path, predictions, subset=None, *, image_ids=None)` — `image_ids` scopes the evaluation to exactly those ground-truth images. `subset=N` keeps working and keeps meaning `sorted(getImgIds())[:N]`; passing both and having them disagree is an error, not a silent precedence rule.
- `run_single_backend(..., image_ids=...)` already receives the evaluated ids — it threads them into the evaluation instead of dropping them. `run_all_backends` and `run_benchmark` need no new parameter: the scope rides the ids they already pass.
- `tests/fixtures/coco_map_baseline.json` — measured on CI, recording: model `yolo11n`, weights sha256 `0ebbc80d…4ee1`, backend `pytorch`, device `cpu`, `confidence_threshold` 0.001, `iou_threshold` 0.45, `images` 500, subset manifest sha256 `872ac411…6aad`, `mAP_50_95` / `mAP_50` / `mAP_75`, plus `tolerance`, `measured_on` and `measured_at`.
- band: **two-sided, ±0.005 absolute** on `mAP_50_95` (human decision, 2026-09-15). Repeat runs on one machine are bit-identical (measured: `0.41577614647708055` twice), so the band exists only to absorb cross-machine float drift, which is unmeasured until CI runs. The FIRST CI run records the baseline; this box is not ticked until a second CI run confirms the band holds.
- threshold: **conf 0.001**, the COCO protocol (human decision, 2026-09-15), so the number is comparable to the published yolo11n 39.5. Measured cost of the shipped default: conf 0.25 scores **0.3658**, 5.0 points lower — recorded in the baseline file as `shipped_default_mAP_50_95` so the gap is visible rather than discovered.

## EDGES
- E1 a subset larger than the dataset is clamped, and the count recorded is the count evaluated, not the count requested
- E2 omitting the scope evaluates every ground-truth image — today's behaviour, byte-for-byte unchanged
- E3 `subset` and `image_ids` that disagree raise, rather than one silently winning
- E4 zero predictions score 0.0 and FAIL the gate loudly — a model that detects nothing must not reach the "no predictions, nothing to compare" branch and pass
- E5 a baseline file that is missing, unparseable, or whose recorded weights digest does not match the weights the gate just loaded, fails the gate with that as the named reason
- E6 the gate's own run reports the backend and device it actually used, and a mismatch against the baseline's recorded pair fails rather than being averaged in

## CHECKS
- test_a_subset_benchmark_scores_only_the_images_it_evaluated · covers: M1,A2,S1 · drives `run_benchmark` over a subset and asserts the mAP matches the scoped evaluation, not the full-ground-truth one — red today at 0.0419 against 0.4158
- test_explicit_image_ids_scope_the_evaluation · covers: M1,R:UNSCOPED_EVALUATION,S1 · `evaluate_coco_map(..., image_ids=[...])` evaluates exactly those ids
- test_a_subset_above_the_dataset_size_is_clamped_to_what_exists · covers: E1,A3
- test_an_unscoped_evaluation_still_covers_every_ground_truth_image · covers: E2,A4
- test_a_subset_that_disagrees_with_image_ids_is_an_error · covers: E3
- test_the_selection_order_is_sorted_image_ids · covers: A5
- test_the_baseline_records_every_input_that_determines_the_number · covers: M2,A8,A12,S2
- test_a_missing_baseline_fails_the_gate · covers: A10,E5,R:GREEN_BY_SKIP,S2
- test_a_baseline_whose_weights_digest_does_not_match_fails_the_gate · covers: E5,A9
- test_a_measured_map_below_the_band_fails_with_baseline_measured_and_delta · covers: M3,A18,S3
- test_a_measured_map_above_the_band_also_fails · covers: M3,R:SELF_UPDATING_BASELINE
- test_zero_predictions_fail_the_gate_rather_than_passing_vacuously · covers: E4
- test_a_backend_or_device_mismatch_against_the_baseline_fails · covers: E6,A14
- test_the_gate_fails_rather_than_skips_when_the_dataset_is_absent_under_ci · covers: M4,A16,R:GREEN_BY_SKIP,S3
- test_ci_registers_the_map_gate_step · covers: M4,A13,A15,A17,S3 · the CI contract guard sees the step, in the job, within the declared timeout
- test_the_baseline_is_never_written_by_the_gate · covers: R:SELF_UPDATING_BASELINE,A7
- test_a_who_dimension_free_library_call_needs_no_principal · covers: A1,A6,A11 · the reported table carries the evaluated image count beside the mAP, so a reader can see the denominator without reading the source
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
