---
type: Task
title: Commit the ultralytics oracle harness and gate on it
status: done
depth: standard
milestone: m3-prove-it
scope:
  - tests/
  - .github/workflows/
gives:
  - S1 a committed equivalence harness comparing yowo's native architecture against ultralytics as an oracle, per variant
  - S2 an `Arch Equivalence` CI job whose result gates a pull request
  - S3 bounds declared as named constants, one per axis, each measured before the run
depends_on:
  - /tasks/ci-weight-fixture.md
  - /tasks/pr-ci-gate.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-16, act: interview, authority: human, interview: "sha256:c12eae8a8093d133", receipt: /tasks/arch-equivalence-in-ci.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:ORACLE_NOT_RUN=confirm|R:POSITION_PAIRED=confirm|R:GREEN_BY_SKIP=confirm|R:BOUND_FITTED_TO_RESULT=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:e3da7e987be964a1", binding: "sha256:5281b03f71b78e38" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/1.md }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/2.md }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/3.md }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:1f6fd3726088444b" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/4.md, brief: "sha256:1f6fd3726088444b" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/5.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/5.md, brief: "sha256:5b1d3fedf1acd13f" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/6.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/6.md, brief: "sha256:5b1d3fedf1acd13f" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/7.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/arch-equivalence-in-ci.d/runs/7.md, brief: "sha256:5b1d3fedf1acd13f" }
advised_by: inference-parity-engineer
---
## CARD
goal: compare every native variant's raw model output against ultralytics on the same weights and the same image, per variant, inside declared bounds, and gate CI on the result.
why: `src/yowo/arch/` reimplements YOLO11 and YOLO26 from scratch with no ultralytics dependency. The equivalence that justified that was checked once, by a harness in `tmp/` that no longer exists — so today nothing in the repo can tell whether the reimplementation still matches the reference.
beat: done · next: add status

## RULES
<must>
- M1 the comparison is against ultralytics executing the SAME weight file, not against a recorded number — an oracle that is not run is a constant with extra steps
- M2 both axes are bounded separately, each by a named constant declared before the run: geometry and class score behave differently and one bound over both hides the tighter axis
- M3 every registered detection variant is compared, and a variant that cannot be compared fails rather than being silently absent from the set
- M4 the result gates a pull request — it is a required-capable CI job, published under a name, and registered in the same commit that adds it
</must>
<reject>
- R:ORACLE_NOT_RUN equivalence is never asserted against a hardcoded expected tensor or a number copied from a previous run -> "ORACLE_NOT_RUN"
- R:POSITION_PAIRED an order-dependent output is never compared row-by-row by position -> "POSITION_PAIRED"
- R:GREEN_BY_SKIP the job never reports success because ultralytics, a weight, or a variant was absent -> "GREEN_BY_SKIP"
- R:BOUND_FITTED_TO_RESULT a bound is never widened to admit a deviation the run just produced -> "BOUND_FITTED_TO_RESULT"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a tensor comparison has no principals
- A2 [which] covers: S1 · the request does not say WHAT is compared; taking "the raw model output, before any postprocessing" -> comparing post-NMS detections would measure yowo's postprocess against ultralytics' postprocess, which is a different claim from architectural equivalence and would let an arch error hide behind an NMS difference
- A3 [when] covers: S1 · the request does not say where the bounds fall; taking two, measured first · probe: every variant lands inside both · found 2026-09-16 on bus.jpg, fp32, same weight file both sides — YOLO11 raw over all 8400 anchors: box 5.6458e-04..**1.4038e-03** px, class 1.3188e-06..**3.5942e-05**; YOLO26 end2end over real detections: box 3.0518e-05..6.1035e-05, conf 1.1921e-07..1.7881e-06. **A single 1e-3 bound would FAIL yolo11l at 1.4038e-03** — the trap for anyone reusing the parity suite's constant.
- A4 [absent] covers: S1 · the request does not say what an absent ultralytics means; taking "fails under CI, skips locally" -> ultralytics is a dev dependency gated on python>=3.11, so a 3.10 runner would otherwise report equivalence it never checked
- A5 [order] covers: S1 · the request does not say how rows are paired; taking "by position for YOLO11's anchor-ordered pre-NMS output, and by confidence rank over real detections for YOLO26's end2end output" · probe: the two layouts must be handled differently -> measured 2026-09-16: position-pairing YOLO26's (1,300,6) top-300 list on a random-noise input reports a max deviation of **6.479e+02**, which is not a defect but junk rows reordering. The same comparison on a real image with real detections gives 3.0518e-05. Pairing is the whole difference between a phantom bug and a measurement.
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking "someone deciding whether the architecture drifted or the oracle moved" -> the message must name the variant, the axis, the value, the bound and the ultralytics version
- A7 [who] covers: S2 · n/a · CI has no caller identity
- A8 [which] covers: S2 · the request does not say which variants run on a pull request; taking "all ten" · probe: the whole set must fit the job budget -> a subset would leave a family or a size unchecked on the branch that ships
- A9 [when] covers: S2 · the request does not say the budget; taking the job's declared timeout · found 2026-09-16: the ten-variant comparison runs locally in the same order as the parity suite's ten, dominated by loading each weight twice
- A10 [absent] covers: S2 · the request does not say what a missing weight means; taking "`CI=true` fails, local skips" — the rule `tests/integration/conftest.py` already applies -> Q3: a skipped test is green
- A11 [order] covers: S2 · n/a · variants are independent
- A12 [experience] covers: S2 · the request does not say what the check is called; taking "`Arch Equivalence`, published and registered in `docs/ci-required-checks.md` in the same commit" -> `test_every_pull_request_check_is_classified_as_gating_or_advisory` now refuses an unclassified job, which is how `Export Parity` shipped blocking nothing
- A13 [who] covers: S3 · the request does not say who may move a bound; taking "a human in a reviewed commit, never to admit a deviation the run just produced"
- A14 [which] covers: S3 · the request does not say which axes; taking "box geometry in pixels, and class score / confidence" -> these differ by two orders of magnitude and one bound over both would be set by the looser
- A15 [when] covers: S3 · the request does not say when a bound may tighten; taking "freely, in the commit that earns it"
- A16 [absent] covers: S3 · the request does not say what an absent bound means; taking "the check fails" -> deleting a constant must not read as satisfying it
- A17 [order] covers: S3 · n/a · two independent scalars
- A18 [experience] covers: S3 · the request does not say where the measurements live; taking "recorded beside the bounds as documentation, never asserted" -> Q11: a number measured on one machine is not a property of the code, and this repo has already had CI reject a pinned per-machine figure

## PLAN
contract:
- `tests/integration/test_arch_equivalence.py`: for each registered variant, load the SAME weight file into ultralytics and into `yowo.arch.build_model` + `load_weights`, run one real image through both, compare.
- two constants: `BOX_TOLERANCE_PX = 5e-3` (worst measured 1.4038e-03, 3.6x headroom) and `CLASS_TOLERANCE = 1e-4` (worst measured 3.5942e-05, 2.8x headroom). Both declared before the run; the measured ranges are recorded as documentation and asserted only against the bound.
- YOLO11 emits `(1, 84, 8400)` pre-NMS and anchor-ordered — position pairing is valid and every anchor is compared. YOLO26 emits `(1, 300, 6)` end2end — only detections above a confidence floor are compared, because the tail of a top-300 list is junk whose ordering is not part of any contract.
- a new `arch-equivalence` job in `ci.yml`, registered in `FROZEN_JOB_IDS` and in `docs/ci-required-checks.md` in the same commit — my own guard now refuses a job that appears in neither.
- ultralytics stays a DEV dependency. It is AGPL-3.0 and this package is Apache-2.0; the oracle runs in CI and is never imported by shipped code.

## EDGES
- E1 ultralytics absent (python 3.10, or not installed) — fails under CI, skips locally, never passes
- E2 a registered variant missing from the comparison set — fails, naming it
- E3 the two models return different output shapes — fails as a structural mismatch rather than being coerced into a comparison
- E4 a bound constant deleted — the check fails rather than reading as satisfied
- E5 YOLO26's end2end output yields zero detections above the floor — fails, because agreement over an empty set proves nothing
- E6 the oracle and yowo are handed different weight files — impossible by construction, and checked

## CHECKS
Names DECLARED here first and used verbatim. Every line carries a trailing
field: without it a check line parses, runs, passes and binds NOTHING (M21,
learned by refusal on `coverage-floor`).

tests/integration/test_arch_equivalence.py:
- test_yolo11_box_geometry_matches_the_oracle[yolo11n] · covers: M1,M2,A2,A3,S1 · every anchor, position-paired
- test_yolo11_box_geometry_matches_the_oracle[yolo11s] · covers: M1,M2,A3,S1 · every anchor, position-paired
- test_yolo11_box_geometry_matches_the_oracle[yolo11m] · covers: M1,M2,A3,S1 · every anchor, position-paired
- test_yolo11_box_geometry_matches_the_oracle[yolo11l] · covers: M1,M2,A3,S1 · the variant a 1e-3 bound would fail
- test_yolo11_box_geometry_matches_the_oracle[yolo11x] · covers: M1,M2,A3,S1 · every anchor, position-paired
- test_yolo11_class_scores_match_the_oracle[yolo11n] · covers: M2,A14,S1 · the tighter axis, bounded separately
- test_yolo11_class_scores_match_the_oracle[yolo11s] · covers: M2,A14,S1 · the tighter axis, bounded separately
- test_yolo11_class_scores_match_the_oracle[yolo11m] · covers: M2,A14,S1 · the tighter axis, bounded separately
- test_yolo11_class_scores_match_the_oracle[yolo11l] · covers: M2,A14,S1 · the tighter axis, bounded separately
- test_yolo11_class_scores_match_the_oracle[yolo11x] · covers: M2,A14,S1 · the tighter axis, bounded separately
- test_yolo26_detections_match_the_oracle[yolo26n] · covers: M1,A5,R:POSITION_PAIRED,S1 · end2end, confidence-ranked
- test_yolo26_detections_match_the_oracle[yolo26s] · covers: M1,A5,R:POSITION_PAIRED,S1 · end2end, confidence-ranked
- test_yolo26_detections_match_the_oracle[yolo26m] · covers: M1,A5,R:POSITION_PAIRED,S1 · end2end, confidence-ranked
- test_yolo26_detections_match_the_oracle[yolo26l] · covers: M1,A5,R:POSITION_PAIRED,S1 · end2end, confidence-ranked
- test_yolo26_detections_match_the_oracle[yolo26x] · covers: M1,A5,R:POSITION_PAIRED,S1 · end2end, confidence-ranked
- test_every_registered_variant_is_compared · covers: M3,E2,A8 · a variant absent from the set fails rather than vanishing
- test_both_sides_are_handed_the_same_weight_file · covers: M1,E6,R:ORACLE_NOT_RUN · the oracle runs on the bytes under test
- test_the_declared_bounds_are_the_measured_ones_with_headroom · covers: M2,A13,A15,A16,E4,R:BOUND_FITTED_TO_RESULT,S3 · deleting or widening a bound is a diff against an assertion
- test_the_recorded_measurements_are_documentation_not_assertions · covers: A18,A6 · Q11, after CI rejected a pinned per-machine figure
- test_a_shape_mismatch_is_a_structural_failure · covers: E3 · never coerced into a comparison
- test_zero_detections_above_the_floor_is_not_agreement · covers: E5 · agreement over an empty set proves nothing
- test_an_absent_oracle_fails_under_ci_rather_than_skipping · covers: A4,E1,R:GREEN_BY_SKIP · ultralytics is python>=3.11 only

tests/unit/test_arch_equivalence_ci.py:
- test_ci_registers_the_arch_equivalence_job · covers: M4,A7,A11,S2 · registered in FROZEN_JOB_IDS in the same commit
- test_the_arch_equivalence_job_publishes_a_requirable_name · covers: M4,A12,S2 · protection binds the published name
- test_the_arch_equivalence_step_runs_under_ci_true_within_its_budget · covers: M4,A9,A10,R:GREEN_BY_SKIP,S2 · an absent weight must not skip green

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
