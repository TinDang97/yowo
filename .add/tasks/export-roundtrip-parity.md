---
type: Task
title: PyTorch to ONNX numeric agreement, asserted per variant
status: direction
depth: standard
milestone: m3-prove-it
scope:
  - tests/
  - .github/workflows/
gives:
  - S1 a parity check binding PyTorch↔ONNX agreement per variant, both bounds declared before the run and the executing provider asserted from the session that produced the numbers
  - S2 the n/s variants run on every pull request
  - S3 all ten variants run on a schedule, in a workflow this repo does not yet have
depends_on:
  - /tasks/ci-weight-fixture.md
  - /tasks/real-backend-smoke.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-15, act: interview, authority: human, interview: "sha256:2d31fa5fd91f58f8", receipt: /tasks/export-roundtrip-parity.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:UNPINNED_PROVIDER=confirm|R:SELF_RATIFYING_BOUND=confirm|R:GREEN_BY_SKIP=confirm|R:COUNT_BLIND=confirm" }
  - { by: "Tin Dang", at: 2026-09-15, act: freeze, authority: human, direction: "sha256:8834c77103e7527c", binding: "sha256:edbb6fcba706ddca" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:f62dde0f6c1205bb" }
advised_by: inference-parity-engineer
---
## CARD
goal: assert PyTorch and ONNX agree to 1e-3 px on coordinates and 1e-4 on confidence, per variant, with the execution provider pinned and asserted — n/s on every pull request, all ten on a schedule.
why: the export path has never been checked per variant. `test_backend_conformance.py` binds coordinates for ONE variant (yolo26n) and binds confidence NOWHERE, so nine of ten exports and the entire confidence axis are unmeasured by CI.
beat: direction · next: freeze

## RULES
<must>
- M1 both bounds are declared as named constants BEFORE the run, and the assertion compares against those constants — never against a number the run itself produced
- M2 the execution provider that produced the compared numbers is asserted, read from the live session rather than from the request that asked for it
- M3 a disagreement names the variant, the axis, the measured value and the bound, so the reader does not have to open the source to learn what was compared
- M4 the n/s variants run on every pull request; all ten run on a schedule, and the two entry points share ONE parity implementation
</must>
<reject>
- R:UNPINNED_PROVIDER parity is never measured through a provider the test did not pin and confirm -> "UNPINNED_PROVIDER"
- R:SELF_RATIFYING_BOUND a bound is never derived, widened or rounded from the deviation the run just measured -> "SELF_RATIFYING_BOUND"
- R:GREEN_BY_SKIP the schedule never reports success because a weight, an export or onnxruntime was absent -> "GREEN_BY_SKIP"
- R:COUNT_BLIND agreement is never asserted over a zip of two lists whose lengths were not compared first -> "COUNT_BLIND"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a numeric comparison between two backends has no principals
- A2 [which] covers: S1 · the request does not say which detections are compared when the two backends return different counts; taking "a count mismatch is a failure in its own right, asserted before any pairing" -> `zip` silently truncates to the shorter list, so a backend that dropped a detection would be compared on the ones it kept and pass · probe: a count mismatch must fail naming both counts · found 2026-09-15: all four PR-CI variants return 5 and 5 on bus.jpg, so the guard is currently inert — which is exactly when it is cheapest to get wrong
- A3 [when] covers: S1 · the request does not say where the bound falls; taking the box's own numbers, **1e-3 px on coordinates and 1e-4 on confidence** · probe: every variant must land inside both · found 2026-09-15, ALL TEN variants, EP pinned to CPUExecutionProvider, bus.jpg, fp32, 5 detections each. Coordinates (px): 11n 1.2207e-04 · 11s 1.2207e-04 · 11m 1.2207e-04 · 11l 9.155e-05 · 11x 1.2207e-04 · 26n 3.0899e-04 · 26s 2.4414e-04 · 26m 2.7466e-04 · 26l 1.8311e-04 · 26x 1.2207e-04 — range 9.155e-05 to 3.0899e-04, worst case 3.2x inside 1e-3, and REPRODUCING the 2026-09-13 range exactly. Confidence: 1.43e-06 · 2.4e-07 · 3.0e-07 · 5.4e-07 · 4.8e-07 · 2.09e-06 · 1.8e-07 · 2.4e-07 · 4.8e-07 · 7.2e-07 — worst 2.09e-06, **48x inside 1e-4**. The box's confidence bound is SATISFIABLE, which was NOT known before this measurement: nothing in the repo had ever compared the confidence axis. YOLO26 deviates ~2x more than YOLO11 on coordinates at every size, which is a real family difference and not noise, but every variant is comfortably inside one bound — so the per-variant tolerance the box's wording implies stays unnecessary.
- A4 [absent] covers: S1 · the request does not say what an absent provider list means; taking "an empty `active_providers` fails the check" -> the property returns `()` when no session is bound, and `() != ("CPUExecutionProvider",)` must read as "could not confirm", not as a pass
- A5 [order] covers: S1 · the request does not say what pairs a PyTorch detection with an ONNX one; taking "sort by descending confidence", the rule `test_backend_conformance.py:147` already established -> output order is not part of the contract, so pairing by position compares unrelated boxes
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking "someone who must decide whether the exporter regressed or the bound was always too tight" -> a bare `assert dev <= 1e-3` tells them neither the variant nor the number
- A7 [who] covers: S2 · n/a · CI has no caller identity
- A8 [which] covers: S2 · the request does not say which variants are "n/s"; taking all four — yolo11n, yolo11s, yolo26n, yolo26s -> "n/s" read as one family would leave a whole architecture unchecked on pull requests
- A9 [when] covers: S2 · the request does not say the budget; taking "inside the existing per-job ceiling" · probe: the four PR-CI variants export and compare inside it · found 2026-09-15: four variants measured end to end locally, export dominating
- A10 [absent] covers: S2 · the request does not say what an absent weight means; taking "`CI=true` fails, local skips", the rule `tests/integration/conftest.py` already applies -> Q3: a skipped test is green
- A11 [order] covers: S2 · n/a · variants are independent; nothing sequences them
- A12 [experience] covers: S2 · the request does not say what the check is called; taking "a published name a contributor can find in the checks list and a maintainer can require" -> see `docs/ci-required-checks.md`, which now demands a verdict for every ci.yml job
- A13 [who] covers: S3 · n/a · a scheduled run has no requester
- A14 [which] covers: S3 · the request does not say which ten; taking the five sizes of both families, the registry's full detection surface -> a schedule that silently covers nine is worse than one that covers four and says so
- A15 [when] covers: S3 · the request does not say the cadence; taking a daily cron -> a weekly schedule means a regression can sit undetected for six days, and the run is cheap enough that daily costs little · probe: the run must complete inside the workflow timeout — the x variants are 114 MB and 118 MB of weights and their exports dominate
- A16 [absent] covers: S3 · the request does not say what a scheduled failure does; taking "it fails the workflow run, visibly" -> a scheduled job nobody watches is the same shape as an advisory check, the state `docs/ci-required-checks.md` was just amended to end
- A17 [order] covers: S3 · n/a · one run, ten independent variants
- A18 [experience] covers: S3 · the request does not say where the schedule lives; taking "a NEW workflow file" -> `ci.yml`'s job ids are frozen by `test_ci_contract.py::test_ci_job_ids_are_the_frozen_set`, and this repo has no `schedule:` or `cron:` anywhere (checked 2026-09-15), so there is nothing to extend

## PLAN
contract:
- ONE parity implementation, parametrised over variants, used by both entry points. The split is which variants each runs, never which assertions.
- bounds as named module constants, `COORD_TOLERANCE_PX = 1e-3` and `CONF_TOLERANCE = 1e-4`, each with a check asserting the constant itself — so widening one is a visible diff against a check, not a quiet edit.
- both backends driven through the real engine, so what is compared is post-NMS detections rather than raw tensors; the provider is read from the engine's live backend, because the EP that matters is the one that produced these numbers, not one confirmed in a separate session.
- a new `.github/workflows/parity.yml` carrying `schedule:` — the first in this repo — plus a `pull_request` trigger for the n/s subset, so the same file shows both halves of the split.
- measured 2026-09-15 and recorded, not asserted: the exporter REQUESTS opset 17 and does not get it. `onnxscript` reports "Failed to convert the model to the target version 17 using the ONNX C API. The model was not modified" and the artifact ships at opset 18. Out of scope here — it changes no number this node binds — but it means `opset_version=17` in `_exporter.py` is a request the artifact does not honour, and nothing reports that.

## EDGES
- E1 PyTorch and ONNX return different detection counts — fails naming both counts, before any pairing
- E2 the ONNX session binds a provider other than the one pinned — fails, and says which ran
- E3 `active_providers` is empty — fails as "could not confirm", never passes
- E4 a variant whose weight cannot be resolved fails under CI and skips locally
- E5 the scheduled workflow and the pull-request workflow disagree about which assertions run — impossible by construction, and checked
- E6 zero detections from both backends is not a vacuous pass: agreement over an empty set proves nothing

## CHECKS
These names are DECLARED here first and used verbatim in the test module, with
`ids=` pinning every parametrised case — the opposite order to the last node,
where names were written from memory, cited as `[a|b|c]` shorthand, and bound
nothing (M12). Each id below is what `--collect-only` must print.

tests/integration/test_export_parity.py:
- test_pytorch_and_onnx_agree_on_coordinates[yolo11n] · covers: M1,A3,S1,S2 · bound 1e-3 px, declared before the run
- test_pytorch_and_onnx_agree_on_coordinates[yolo11s] · covers: M1,A3,S1,S2 · bound 1e-3 px, declared before the run
- test_pytorch_and_onnx_agree_on_coordinates[yolo26n] · covers: M1,A3,S1,S2 · bound 1e-3 px, declared before the run
- test_pytorch_and_onnx_agree_on_coordinates[yolo26s] · covers: M1,A3,S1,S2 · bound 1e-3 px, declared before the run
- test_pytorch_and_onnx_agree_on_confidence[yolo11n] · covers: M1,A3,S1,S2 · bound 1e-4, an axis nothing in CI has ever checked
- test_pytorch_and_onnx_agree_on_confidence[yolo11s] · covers: M1,A3,S1,S2 · bound 1e-4, an axis nothing in CI has ever checked
- test_pytorch_and_onnx_agree_on_confidence[yolo26n] · covers: M1,A3,S1,S2 · bound 1e-4, an axis nothing in CI has ever checked
- test_pytorch_and_onnx_agree_on_confidence[yolo26s] · covers: M1,A3,S1,S2 · bound 1e-4, an axis nothing in CI has ever checked
- test_the_compared_numbers_came_from_the_pinned_provider[yolo11n] · covers: M2,R:UNPINNED_PROVIDER,E2,S1 · read from the live session, not the request
- test_the_compared_numbers_came_from_the_pinned_provider[yolo11s] · covers: M2,R:UNPINNED_PROVIDER,E2,S1 · read from the live session, not the request
- test_the_compared_numbers_came_from_the_pinned_provider[yolo26n] · covers: M2,R:UNPINNED_PROVIDER,E2,S1 · read from the live session, not the request
- test_the_compared_numbers_came_from_the_pinned_provider[yolo26s] · covers: M2,R:UNPINNED_PROVIDER,E2,S1 · read from the live session, not the request
- test_the_declared_bounds_are_the_numbers_the_box_states · covers: M1,A3,R:SELF_RATIFYING_BOUND · widening a bound becomes a visible diff against a check
- test_a_count_mismatch_fails_before_any_pairing · covers: A2,R:COUNT_BLIND,E1 · zip truncates silently, so a dropped detection would pass on the ones that remained
- test_detections_are_paired_by_confidence_not_position · covers: A5 · output order is not part of the contract
- test_an_empty_provider_list_is_not_a_pass · covers: A4,E3 · `()` means could-not-confirm, never confirmed
- test_a_disagreement_names_the_variant_axis_value_and_bound · covers: M3,A6 · the reader must not have to open the source
- test_zero_detections_on_both_sides_is_not_agreement · covers: E6 · agreement over an empty set proves nothing

tests/unit/test_parity_workflow_contract.py:
- test_the_pull_request_trigger_runs_the_four_ns_variants · covers: M4,A8,S2
- test_the_schedule_runs_all_ten_variants · covers: M4,A14,S3
- test_both_entry_points_share_one_parity_implementation · covers: M4,E5 · the split is which variants, never which assertions
- test_the_parity_workflow_declares_a_schedule · covers: A15,A18,S3 · the first `schedule:` in this repo
- test_both_parity_jobs_declare_a_timeout · covers: A9,A15 · the x variants are 114 MB and 118 MB of weights
- test_the_parity_jobs_publish_explicit_names · covers: A12 · a check nobody can name is a check nobody can require
- test_a_missing_weight_fails_under_ci_rather_than_skipping · covers: A10,E4,R:GREEN_BY_SKIP · Q3: a skipped test is green
- test_the_scheduled_run_fails_visibly · covers: A16 · no continue-on-error, no `|| true`

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
