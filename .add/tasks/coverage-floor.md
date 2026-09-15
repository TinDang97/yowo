---
type: Task
title: Branch coverage measured, with a floor that fails the build
status: done
depth: quick
milestone: m3-prove-it
scope:
  - pyproject.toml
  - .github/workflows/
  - tests/
gives:
  - S1 branch coverage configured in pyproject.toml over the whole of src/yowo, with a floor that fails the build
  - S2 a CI step that runs it and cannot report success without it
  - S3 a check binding the floor's value, so lowering it is a visible diff against an assertion
depends_on:
  - /tasks/real-backend-smoke.md
  - /tasks/integration-tier-revival.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-15, act: interview, authority: human, interview: "sha256:2833956615b9398d", receipt: /tasks/coverage-floor.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:FLOOR_LOWERED_TO_PASS=confirm|R:PHANTOM_DENOMINATOR=confirm|R:GREEN_BY_SKIP=confirm" }
  - { by: "Tin Dang", at: 2026-09-15, act: freeze, authority: human, direction: "sha256:6bf1ec80dc901892", binding: "sha256:5e7c6216742b519a" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:01a11cb96ed548fa" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/coverage-floor.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-15, act: refreeze, authority: human, direction: "sha256:e3379409ea39abed", binding: "sha256:5e7c6216742b519a" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:6c9c034a4937b501" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/coverage-floor.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-15, act: gate, authority: human, outcome: PASS, receipt: /tasks/coverage-floor.d/runs/2.md, brief: "sha256:6c9c034a4937b501" }
advised_by: build-craftsman
---
## CARD
goal: measure branch coverage over all of `src/yowo`, fail the build below a floor, and bind the floor so it cannot be quietly lowered.
why: there is no coverage machinery in this repo at all. The number nobody measures is the number nobody defends.
beat: done · next: add status

## RULES
<must>
- M1 coverage is measured with BRANCH coverage on, over the whole of `src/yowo` including modules no test imports — a denominator that omits unreached files flatters the result
- M2 a floor fails the build; a coverage step that reports a number and exits zero gates nothing
- M3 the floor's value is bound by a check, so lowering it is a visible diff against an assertion rather than an edit to a config line
- M4 what the number covers is stated where it is reported — which tests produced it and which source it measures
</must>
<reject>
- R:FLOOR_LOWERED_TO_PASS the floor is never reduced to make a red build green -> "FLOOR_LOWERED_TO_PASS"
- R:PHANTOM_DENOMINATOR coverage is never reported over a source set that silently omits files -> "PHANTOM_DENOMINATOR"
- R:GREEN_BY_SKIP the coverage step never reports success without having measured -> "GREEN_BY_SKIP"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a coverage measurement has no principals
- A2 [which] covers: S1 · the request does not say which source files are in the denominator; taking "every `.py` under `src/yowo`, imported or not" · probe: the report must name as many files as the tree holds · found 2026-09-15: **96 files on disk, 96 in the report, 0 missing**. `--cov=src/yowo` includes unimported modules, so the 2026-09-13 worry about "a floor over a denominator that silently excludes them" does not apply.
- A3 [when] covers: S1 · the request does not say where the floor sits; taking "the CI-measured total, rounded DOWN to a whole percent" · probe: the floor must be at or below what CI measures and above nothing · found 2026-09-15 LOCALLY: combined **80.8737%** (8149/9779 statements, 1830/2560 branches, 308 partial), line-only 83.3316%. The CI figure is NOT this figure and must be measured: ubuntu takes different branches from darwin — no MPS, no CoreML — so the floor is set from the CI run, not from this one.
- A4 [absent] covers: S1 · the request does not say what an unmeasured module means; taking "0% for that file, counted in the total" -> excluding it would let deleting tests raise the percentage
- A5 [order] covers: S1 · n/a · coverage is a set measure; nothing is ordered
- A6 [experience] covers: S1 · the request does not say who reads a breach; taking "someone who must decide whether to write a test or argue with the floor" -> a bare `FAIL Required test coverage of 80% not reached` invites the second
- A7 [who] covers: S2 · n/a · CI has no caller identity
- A8 [which] covers: S2 · the request does not say which tests produce the number; taking "the unit tier only" -> the integration tier is not in this measurement and the box must say so rather than imply whole-project coverage
- A9 [when] covers: S2 · the request does not say the budget; taking "the existing Quality Gate ceiling" · found 2026-09-15: the unit tier under branch coverage runs in **94.33s** locally against ~85s without, so coverage costs roughly ten seconds
- A10 [absent] covers: S2 · the request does not say what a missing coverage tool means; taking "the step fails" -> pytest-cov absent would otherwise make the gate vanish silently
- A11 [order] covers: S2 · n/a · one step, one number
- A12 [experience] covers: S2 · the request does not say where the number is seen; taking "printed in the job log, not only enforced" -> a floor that passes silently teaches nobody where the headroom went
- A13 [who] covers: S3 · the request does not say who may move the floor; taking "a human in a reviewed commit, never to make a red build green"
- A14 [which] covers: S3 · the request does not say what binds the floor; taking "a unit check reading the value out of pyproject.toml" -> the same drift that put three stale copies of the required-context list in this repo
- A15 [when] covers: S3 · the request does not say when the floor may RISE; taking "freely, and only in the same commit that earns it"
- A16 [absent] covers: S3 · the request does not say what an absent floor means; taking "the check fails" -> deleting `fail_under` must not read as satisfying it
- A17 [order] covers: S3 · n/a · one value
- A18 [experience] covers: S3 · the request does not say what the check says on failure; taking "it names the configured value and the value it expected"

## PLAN
contract:
- `[tool.coverage.run]` with `branch = true` and `source = ["src/yowo"]`; `[tool.coverage.report]` with `fail_under` and `show_missing`.
- a `coverage` step on the existing `quality` job — NOT a new job. `ci.yml`'s job ids are frozen by `test_ci_contract.py`, and a fifth quality command belongs beside the other four.
- a unit check reading `fail_under` back out of `pyproject.toml` and asserting its value, so lowering it is a diff against an assertion.
- the floor is set from the CI measurement. The local figure is 80.8737% combined; ubuntu takes different branches (no MPS, no CoreML) and the honest floor is the one the platform that enforces it actually reaches. FIRST CI RUN DECIDES, exactly as the mAP baseline was set.
- the box's own prediction is REFUTED and must be corrected rather than quietly ticked: it says branch coverage "will drop the headline below today's 80% lines". Measured, it drops from 83.3316% line-only to 80.8737% combined — lower, but still above 80.

## EDGES
- E1 `fail_under` absent from pyproject.toml — the check fails, rather than reading as satisfied
- E2 a source file no test imports still appears in the denominator at 0%
- E3 the coverage step measures nothing (collection error, no tests) — fails rather than reporting a vacuous 100%
- E4 the floor in pyproject.toml and the floor the check expects disagree — fails, naming both
- E5 coverage is reported for the unit tier only, and says so where it is reported

## CHECKS
Names DECLARED here first and used verbatim; every id below is what `--collect-only` must print.

FORMAT NOTE, learned by refusal 2026-09-15: a check line binds ONLY when it
carries a trailing ` · <why>` after the covers list. Without it the line parses
but binds nothing, and the gate refuses naming the rules left uncovered — here
A2, E1-E4, M2, M3 and all three Rejects, which is exactly the set whose only
covering lines had no trailing field.

tests/unit/test_coverage_floor.py:
- test_coverage_is_configured_with_branch_coverage_on · covers: M1,S1 · line coverage alone would call a never-taken branch covered
- test_the_denominator_is_every_source_file_not_only_the_imported_ones · covers: M1,A2,A4,E2,R:PHANTOM_DENOMINATOR · measuring only imported modules lets deleting a test raise the percentage
- test_a_floor_is_configured_and_fails_the_build · covers: M2,S1 · a step that reports a number and exits zero gates nothing
- test_the_configured_floor_is_the_value_this_check_expects · covers: M3,A14,A18,E4,R:FLOOR_LOWERED_TO_PASS,S3 · lowering the floor becomes a diff against an assertion
- test_a_missing_floor_is_not_read_as_satisfied · covers: A16,E1,S3 · deleting fail_under must not read as passing it
- test_the_floor_is_not_above_what_ci_measured · covers: A3,A13,A15 · a floor above the measured value is a build that can never go green
- test_ci_runs_coverage_in_the_quality_job · covers: M2,A7,A9,A11,S2 · a fifth quality command, beside the other four
- test_the_coverage_step_cannot_report_success_without_measuring · covers: A10,E3,R:GREEN_BY_SKIP,S2 · no continue-on-error, no swallowed failure, no second copy of the floor
- test_what_the_number_covers_is_stated_where_it_is_reported · covers: M4,A8,A12,A6,E5 · the unit tier only, and it says so

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
