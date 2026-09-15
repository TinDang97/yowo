---
type: Task
title: An mAP baseline on a real dataset, enforced
status: done
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
  - { by: "builder", at: 2026-09-15, act: replan, authority: process, note: "CHECK names reconciled to the names pytest actually collects, verified against --collect-only rather than written from memory (M12: a parametrised check cited by its bare name binds nothing, so the seven mismatch cases and the twelve required-field cases are cited by case id). No rule, surface or assumption moved; seventeen declared names became thirty-eight real ones because several declared checks were one name over a parametrised family. Also added: test_the_table_shows_the_denominator_beside_the_map, which binds A6 by putting the evaluated image count in the benchmark table beside the mAP — A6's cost is that 0.0419 looks exactly like a real number, and the table was printing an mAP with no visible denominator." }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/map-regression-gate.d/runs/1.md }
  - { by: "builder", at: 2026-09-15, act: replan, authority: process, note: "Review found eight defects IN THE GATE ITSELF, each with a demonstrated failure scenario; seventeen checks added, all driven red first. The two that matter: (1) NaN passed the band, because abs(nan) > tolerance is False — the guard above it tested isinstance, and isinstance(nan, float) is True, so a non-finite measurement returned normally from the one function that decides the verdict. (2) tolerance was validated for presence but never for range, so editing 0.005 to 1.0 widened the band to unfalsifiable and left all 2798 unit tests, four quality gates and pre-commit green — the only job reading the file being the one the widened band could no longer fail. Also: map_50 and map_75 were required by the loader and read by nothing, so two of the three numbers in a record arguing that a bare float is not a baseline were themselves decorative; images_evaluated counted ids passed rather than images COCOeval scored; an empty image_ids returned pycocotools' -1.0 sentinel labelled as a mAP; the empty-predictions early return skipped scope validation; and the subset/image_ids agreement check was order-sensitive. No rule, surface or assumption moved — every fix strengthens a check that already existed." }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/map-regression-gate.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-15, act: refreeze, authority: human, direction: "sha256:a3ab2926d18fa10e", binding: "sha256:c45e5a61b60500bc" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/map-regression-gate.d/runs/3.md }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:c1810bf35edbee51" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/map-regression-gate.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-15, act: refreeze, authority: human, direction: "sha256:4bd18001c347e0be", binding: "sha256:c45e5a61b60500bc" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:b166eb2c17685591" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/map-regression-gate.d/runs/5.md }
  - { by: "Tin Dang", at: 2026-09-15, act: gate, authority: human, outcome: PASS, receipt: /tasks/map-regression-gate.d/runs/5.md, brief: "sha256:b166eb2c17685591" }
advised_by: inference-parity-engineer
---
## CARD
goal: measure yolo11n's COCO mAP on the pinned 500-image subset, record it as a baseline that describes everything determining it, and fail CI when the number leaves the band.
why: `accuracy-dataset` (PR #51) landed 500 real images and `evaluate_coco_map`'s repair (PR #49) landed a correct scope, but nothing measures a number and nothing defends it. Worse, the PUBLIC path still reports a wrong one — measured 2026-09-15 below.
beat: done · next: add status

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
Every name below is the name pytest collects, taken from `--collect-only` rather
than written from memory, and every parametrised case is spelled out in full.
M12: a parametrised family cited as `[a|b|c]`, or an id truncated with an
ellipsis, binds nothing — which is how A14, E2, E5, E6 and R:GREEN_BY_SKIP came
back unbound from the first gate attempt.

red-first: every check failed first. Recorded in commit a434f8f, which is red on
purpose and carries no implementation; the review-driven additions were driven
red in the same way before f518102.

- test_explicit_image_ids_scope_the_evaluation · covers: M1,S1 · the denominator is the images the run evaluated
- test_image_ids_need_not_be_a_leading_slice · covers: M1,A5 · the set `subset` cannot express at all
- test_image_ids_that_disagree_with_subset_are_an_error · covers: E3
- test_image_ids_that_agree_with_subset_are_accepted · covers: E3
- test_agreement_with_subset_is_by_set_not_by_order · covers: E3,A5 · `[2,1]` is not a disagreement with `subset=2`
- test_an_image_id_absent_from_the_ground_truth_is_an_error · covers: M1,R:UNSCOPED_EVALUATION
- test_duplicate_image_ids_are_refused · covers: M1,E1,A3 · COCOeval applies np.unique; the reported count did not
- test_an_empty_image_ids_is_refused · covers: E4,A4 · zero images score -1.0, which is not a mAP
- test_the_scope_is_validated_even_when_there_are_no_predictions · covers: M1,E4 · the early return skipped validation
- test_the_result_reports_how_many_images_were_evaluated · covers: E1,A3,S1 · the count evaluated, not the count requested
- test_subset_is_deterministic · covers: A5 · pre-existing, PR #49
- test_subset_none_evaluates_every_image · covers: A4,E2 · pre-existing, PR #49
- test_a_subset_larger_than_the_dataset_is_clamped · covers: E1,A3 · pre-existing, PR #49
- test_empty_predictions_score_zero_by_evaluation · covers: E4 · pre-existing, PR #49
- test_a_subset_benchmark_scores_only_the_images_it_evaluated · covers: M1,A2,R:UNSCOPED_EVALUATION,S1 · red at 0.2 before the fix, 1.0 after; 0.0419 against 0.4158 on real COCO
- test_a_full_run_is_unchanged · covers: E2,A4,S1 · today's behaviour, byte-for-byte
- test_a_model_that_misses_images_inside_its_own_subset_still_scores_lower · covers: M1,R:UNSCOPED_EVALUATION · the repair must not make recall free
- test_the_table_shows_the_denominator_beside_the_map · covers: A1,A6,A11,S1 · an mAP with no visible denominator looks exactly like a real one
- test_the_baseline_records_every_input_that_determines_the_number[model] · covers: M2,A8,S2 · dropping model is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[weights_sha256] · covers: M2,A8,S2 · dropping weights_sha256 is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[backend] · covers: M2,A8,S2 · dropping backend is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[device] · covers: M2,A8,S2 · dropping device is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[confidence_threshold] · covers: M2,A8,S2 · dropping confidence_threshold is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[iou_threshold] · covers: M2,A8,S2 · dropping iou_threshold is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[images] · covers: M2,A8,S2 · dropping images is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[subset_manifest_sha256] · covers: M2,A8,S2 · dropping subset_manifest_sha256 is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[map_50_95] · covers: M2,A8,S2 · dropping map_50_95 is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[tolerance] · covers: M2,A8,S2 · dropping tolerance is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[measured_on] · covers: M2,A8,S2 · dropping measured_on is refused by name
- test_the_baseline_records_every_input_that_determines_the_number[measured_at] · covers: M2,A8,S2 · dropping measured_at is refused by name
- test_a_complete_baseline_loads · covers: M2,A12,S2
- test_the_shipped_baseline_loads · covers: M2,S2 · the file the gate actually reads, not a tmp_path copy
- test_the_shipped_band_is_narrow_enough_to_fail_something · covers: M3,R:SELF_UPDATING_BASELINE,S2 · widening the tolerance is re-recording the baseline in everything but name
- test_a_non_positive_tolerance_is_refused · covers: M3,R:SELF_UPDATING_BASELINE
- test_a_missing_baseline_fails_rather_than_passes · covers: A10,R:GREEN_BY_SKIP,S2 · 'no baseline, therefore fine' defends nothing
- test_an_unparseable_baseline_fails · covers: A10,E5,S2
- test_a_measurement_on_the_baseline_passes · covers: M3
- test_a_measurement_inside_the_band_passes · covers: M3
- test_a_measurement_below_the_band_fails · covers: M3,A18,S3 · the message names baseline, measurement and signed delta
- test_a_measurement_above_the_band_also_fails · covers: M3,A18 · a rise is not automatically good news
- test_zero_predictions_fail_rather_than_passing_vacuously · covers: E4
- test_the_gate_never_writes_the_baseline · covers: R:SELF_UPDATING_BASELINE,A7,S2
- test_a_field_of_the_wrong_type_is_refused[tolerance-0.005] · covers: M2,A8 · presence was not validity
- test_a_field_of_the_wrong_type_is_refused[map_50_95-0.4158] · covers: M2,A8 · presence was not validity
- test_a_field_of_the_wrong_type_is_refused[images-500] · covers: M2,A8 · presence was not validity
- test_a_field_of_the_wrong_type_is_refused[confidence_threshold-None] · covers: M2,A8 · presence was not validity
- test_a_non_finite_measurement_fails_rather_than_passes[nan] · covers: M3,E4 · `abs(nan) > tolerance` is False, so NaN returned normally from the verdict
- test_a_non_finite_measurement_fails_rather_than_passes[inf] · covers: M3,E4 · `abs(nan) > tolerance` is False, so NaN returned normally from the verdict
- test_a_non_finite_measurement_fails_rather_than_passes[-inf] · covers: M3,E4 · `abs(nan) > tolerance` is False, so NaN returned normally from the verdict
- test_all_three_recorded_scores_are_defended[map_50_95] · covers: M2,M3,S2 · two of the three recorded numbers were read by nothing
- test_all_three_recorded_scores_are_defended[map_50] · covers: M2,M3,S2 · two of the three recorded numbers were read by nothing
- test_all_three_recorded_scores_are_defended[map_75] · covers: M2,M3,S2 · two of the three recorded numbers were read by nothing
- test_a_mismatched_input_fails_instead_of_being_compared[weights_sha256-0000000000000000000000000000000000000000000000000000000000000000] · covers: E5,A9,A14 · refused rather than compared
- test_a_mismatched_input_fails_instead_of_being_compared[backend-onnx] · covers: E6,A14 · refused rather than compared
- test_a_mismatched_input_fails_instead_of_being_compared[device-cuda] · covers: E6,A14 · refused rather than compared
- test_a_mismatched_input_fails_instead_of_being_compared[confidence_threshold-0.25] · covers: E6,A9 · refused rather than compared
- test_a_mismatched_input_fails_instead_of_being_compared[iou_threshold-0.7] · covers: E6,A9 · refused rather than compared
- test_a_mismatched_input_fails_instead_of_being_compared[images_evaluated-499] · covers: E1,A3,E6 · refused rather than compared
- test_a_mismatched_input_fails_instead_of_being_compared[subset_manifest_sha256-ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff] · covers: A9,E5 · refused rather than compared
- test_ci_job_ids_are_the_frozen_set · covers: M4,A17,S3 · `map-gate` registered in the same commit that adds it
- test_the_map_gate_job_is_registered_with_both_stores_it_needs · covers: M4,A13,A14,S3
- test_the_map_gate_step_runs_under_ci_true_within_its_budget · covers: M4,A15,A16,R:GREEN_BY_SKIP,S3 · without CI=true an absent dataset skips green
- test_the_measured_map_is_inside_the_recorded_band · covers: M3,M4,E5,E6,S3 · the gate itself, over 500 real images
- test_the_gate_evaluated_every_pinned_image · covers: M1,A3,E1,S3
- test_the_model_actually_detected_something · covers: E4,S3

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
