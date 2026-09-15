---
type: Task
title: No hardcoded personal paths; passes from a clean checkout; CI runs it
status: done
depth: standard
milestone: m3-prove-it
scope:
  - tests/
  - .github/workflows/
gives:
  - S1 a check that fails when a test module exists that no pull-request workflow runs
  - S2 CI jobs that actually run the four modules nothing runs today
  - S3 chromadb installed wherever its tests run, so they execute instead of skipping green
depends_on:
  - /tasks/ci-weight-fixture.md
  - /tasks/pr-ci-gate.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-16, act: interview, authority: human, interview: "sha256:86ad6bb6e7dabbe0", receipt: /tasks/integration-tier-revival.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:UNRUN_MODULE=confirm|R:GREEN_BY_SKIP=confirm|R:GUARD_BY_SPELLING=confirm|R:DIVERGENCE=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:46ae7f0a6b8b49d7", binding: "sha256:26d9c8c33b8de944" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:e58e3c5a62f1cfcd" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/integration-tier-revival.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/integration-tier-revival.d/runs/1.md, brief: "sha256:bfb1a93cd8d5a7b1" }
advised_by: build-craftsman
---
## CARD
goal: every test module in the repository is run by a job on a pull request, and a module that no job runs fails a check rather than sitting quietly.
why: measured 2026-09-16 — **42 tests never run in CI and all of them pass locally.** `test_cli_e2e.py` (21) and `test_chroma_gallery_persistence.py` (8) are named by no workflow; `test_onnx_provider_honoured.py` (2) is named by no workflow despite being built in PR #50 *as a gate*; `tests/unit/test_chroma_gallery.py` (11) is inside Quality Gate's path but `importorskip`s chromadb, which CI never installs. A test nobody runs is a claim nobody checked, and all four look like coverage in a file listing.
beat: done · next: add status

## RULES
<must>
- M1 every `test_*.py` under `tests/` is named by at least one job that runs on a pull request, and a module named by none fails a check
- M2 the four unrun modules run in CI, under published job names registered in the same commit
- M3 a module that `importorskip`s a dependency runs in a job that installs that dependency — otherwise the skip is the outcome, and the skip is green
- M4 the pull-request gate and the release gate install the SAME extras, so neither can be the weaker of the two
</must>
<reject>
- R:UNRUN_MODULE a test module is never added without a job that runs it -> "UNRUN_MODULE"
- R:GREEN_BY_SKIP a suite never reports success because its dependency was absent -> "GREEN_BY_SKIP"
- R:GUARD_BY_SPELLING the guard never matches a module by a literal it happens to contain, only by what a job actually runs -> "GUARD_BY_SPELLING"
- R:DIVERGENCE the release path never installs less than the pull-request path -> "DIVERGENCE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a workflow-to-module mapping has no principals
- A2 [which] covers: S1 · the request does not say WHICH modules the guard covers; taking "every `test_*.py` under `tests/`, unit and integration alike" -> the chromadb unit module is inside a job's path and still never executes, so an integration-only guard would call it covered
- A3 [when] covers: S1 · the request does not say when a module counts as run; taking "a job in a workflow with a `pull_request` trigger names it, by path or by directory" · probe: the guard must find today's four -> found 2026-09-16: `test_cli_e2e.py` 21 tests, `test_chroma_gallery_persistence.py` 8, `test_onnx_provider_honoured.py` 2, `tests/unit/test_chroma_gallery.py` 11 skipped at collection. All 31 integration tests pass in 10.52 s; all 11 unit tests pass in 2.73 s.
- A4 [absent] covers: S1 · the request does not say what an absent workflow means; taking "the check fails, naming the module and the fix" -> a guard that reports a count teaches nobody which file to wire
- A5 [order] covers: S1 · n/a · modules are independent
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking "someone who just added a test file"; the message names the module and says to add it to a job -> the failure arrives at the moment the fix is cheapest
- A7 [who] covers: S2 · n/a · CI has no caller identity
- A8 [which] covers: S2 · the request does not say which job runs what; taking "by subject, not by convenience" — `test_onnx_provider_honoured.py` joins `backend-smoke` because both assert a real backend honours a request and that job already restores the weight store; `test_cli_e2e.py` takes its own name because 21 tests failing under someone else's label is a diagnosis nobody can read -> a grab-bag job name makes every failure an investigation
- A9 [when] covers: S2 · the request does not say the budget; taking the declared step timeout · found 2026-09-16: 31 tests in 10.52 s locally, so the budget is dominated by install and cache restore, not by the tests
- A10 [absent] covers: S2 · the request does not say what a missing fixture means; taking "`CI=true` fails, local skips" — the rule `tests/integration/conftest.py` already applies · probe: `test_cli_e2e.py` needs `sample_image_path` and `sample_image_dir`, `test_onnx_provider_honoured.py` needs `verified_weight` -> found 2026-09-16, so both jobs must restore `~/.cache/yowo/test-assets` as well as the weight store
- A11 [order] covers: S2 · n/a · jobs are independent
- A12 [experience] covers: S2 · the request does not say what the checks are called; taking published names registered in `FROZEN_JOB_IDS`, `docs/ci-required-checks.md` and `docs/branch-protection.json` in the same commit -> `Export Parity` shipped blocking nothing because it appeared in one of the three
- A13 [who] covers: S3 · n/a · an install step has no principals
- A14 [which] covers: S3 · the request does not say how chromadb arrives; taking "`--extra chromadb`, the extra `pyproject.toml` already declares at line 48" -> declaring a new dependency would touch a sensitive path to solve a problem an existing extra already solves
- A15 [when] covers: S3 · the request does not say where it is installed; taking "every job that runs a chromadb-dependent module, and the release gate too" · probe: installing is cheap -> found 2026-09-16: chromadb 1.5.2, 78 packages, resolves in 176 ms warm; the 11 unit tests then run in 2.73 s
- A16 [absent] covers: S3 · the request does not say what an absent chromadb means; taking "under `CI=true` the tests must FAIL rather than skip" -> `importorskip` is precisely the construct that converts an absent dependency into a green report
- A17 [order] covers: S3 · n/a · extras are independent
- A18 [experience] covers: S3 · the request does not say who notices divergence; taking "a check binds ci.yml and release.yml to the same extras" -> M4 is otherwise a convention, and the four quality COMMANDS already needed a check to stay identical

## PLAN
contract:
- `tests/unit/test_every_test_module_runs.py`: parse every workflow with a `pull_request` trigger, collect the paths its `run:` steps name, and assert every `test_*.py` under `tests/` is covered by one — by PATH or by an enclosing directory, never by a literal it happens to contain.
- `ci.yml` gains `cli-e2e` ("CLI End-to-End") and `persistent-gallery` ("Persistent Gallery"); `backend-smoke` gains `test_onnx_provider_honoured.py`; `quality` in BOTH `ci.yml` and `release.yml` installs `--extra chromadb`.
- Both new ids registered in `FROZEN_JOB_IDS`, `docs/ci-required-checks.md` and `docs/branch-protection.json` in the same commit. Live protection goes to 12 contexts after merge.
- The guard is the deliverable. The four modules are today's instances; a guard that only fixed them would be the same defect class this repository has now hit five times — a list inside the thing meant to notice drift.

## EDGES
- E1 a new test module added with no job — fails, naming the module
- E2 a job renamed so it no longer names a module — fails, because coverage is computed from what jobs run
- E3 a module covered only by a workflow with no `pull_request` trigger — does not count as covered
- E4 `conftest.py` and `__init__.py` are not test modules — excluded without excusing anything else
- E5 chromadb absent under `CI=true` — the suite fails rather than skipping
- E6 a module matched only because a job's `run:` string contains its name in a comment — not covered; the match is on the paths a command actually runs

## CHECKS
Names DECLARED here first and used verbatim. Every line carries a trailing
`· <why>` — without it a line parses, runs, passes and binds NOTHING (M21).

tests/unit/test_every_test_module_runs.py:
- test_every_test_module_is_run_by_a_pull_request_job · covers: M1,S1,A2,A3,R:UNRUN_MODULE · the guard, over unit and integration alike
- test_the_failure_names_the_module_and_the_fix · covers: A4,A6,E1 · a count teaches nobody which file to wire
- test_coverage_is_computed_from_what_jobs_run_not_from_a_literal · covers: R:GUARD_BY_SPELLING,E6,S1 · a spelling grep already failed this repo once
- test_a_workflow_without_a_pull_request_trigger_does_not_count · covers: E3,A3 · a scheduled job does not gate a pull request
- test_conftest_and_dunder_init_are_not_test_modules · covers: E4 · excluded by kind, not by name
- test_a_module_named_by_a_renamed_job_is_not_covered · covers: E2 · coverage follows the command, not the label

tests/unit/test_chromadb_is_installed_where_its_tests_run.py:
- test_every_job_running_a_chromadb_module_installs_the_extra · covers: M3,S3,A14,A15,R:GREEN_BY_SKIP · importorskip turns an absent dep into green
- test_the_release_gate_installs_what_the_pull_request_gate_installs · covers: M4,A18,R:DIVERGENCE · neither path may be the weaker
- test_an_absent_chromadb_fails_under_ci_rather_than_skipping · covers: E5,A16,R:GREEN_BY_SKIP · Q3

tests/unit/test_revived_jobs.py:
- test_cli_e2e_is_registered_and_published · covers: M2,A12,S2 · three places or it gates nothing
- test_persistent_gallery_is_registered_and_published · covers: M2,A12,S2 · three places or it gates nothing
- test_cli_e2e_restores_the_fixtures_its_tests_require · covers: A10,S2 · needs test-assets, not only weights
- test_the_provider_suite_runs_in_a_job_with_the_weight_store · covers: A8,A10,S2 · joins backend-smoke by subject
- test_every_revived_job_sets_ci_true · covers: M3,A16,R:GREEN_BY_SKIP · a missing input must fail, not skip
- test_every_revived_job_declares_a_budget · covers: A9,A11 · a bare cancellation carries no guidance

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
