---
type: Task
title: CI runs the support surface the package claims, or the claim narrows
status: done
depth: standard
sensitivity: architecture
milestone: m3-prove-it
scope:
  - .github/workflows/
  - pyproject.toml
  - tests/
  - scripts/
gives:
  - S1 a check that fails when the CI matrix and the package's declared Python support disagree, computed from pyproject rather than a hand-written list
  - S2 a job that executes the documented public surface on EVERY claimed Python version
  - S3 the unit tier run on the newest claimed version, with the tiers' real floor and the OS gap recorded rather than implied
depends_on:
  - /tasks/pr-ci-gate.md
  - /tasks/real-backend-smoke.md
  - /tasks/backend-conformance-suite.md
  - /tasks/arch-equivalence-in-ci.md
  - /tasks/export-roundtrip-parity.md
  - /tasks/integration-tier-revival.md
  - /tasks/coverage-floor.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-16, act: interview, authority: human, interview: "sha256:949bb61a9a3c1fb3", receipt: /tasks/ci-matrix.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:HANDLIST=confirm|R:CLAIM_UNRUN=confirm|R:DEVENV_PROVES_SHIP=confirm|R:SILENT_GAP=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:4f1c0fa098729ea7", binding: "sha256:5b6928c90575ac55" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:998e367d24b35c8f" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/ci-matrix.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/ci-matrix.d/runs/1.md, brief: "sha256:8e06c73c653ece99" }
advised_by: build-craftsman
---
## CARD
goal: every Python version the package claims is executed by CI, the matrix is derived from the claim rather than maintained beside it, and what CI still does not cover is written down.
why: `pyproject.toml` declares `requires-python = ">=3.8"` and classifiers for 3.8, 3.9, 3.10, 3.11 and 3.12. **Every job in every workflow runs python 3.11 on ubuntu-latest** — measured 2026-09-16 across all 18 jobs. Four of the five claimed versions have never executed anything, and the package declares no OS classifiers at all, so it implicitly claims every OS while CI runs one.
beat: done · next: add status

## RULES
<must>
- M1 the set of Python versions CI exercises is DERIVED from `pyproject.toml` — the classifiers and the `requires-python` floor — never from a list maintained beside it
- M2 every claimed version executes the documented public surface: import, the public API names, the submodules, the registry and the CLI
- M3 the version job installs the package as a CONSUMER does — no dev group — so it measures the shipped claim rather than the development environment
- M4 what CI does not cover is recorded with its cause, not left to be inferred from the absence of a job
</must>
<reject>
- R:HANDLIST the matrix is never a literal list that can drift from the classifiers -> "HANDLIST"
- R:CLAIM_UNRUN a version is never claimed in metadata while no job executes it -> "CLAIM_UNRUN"
- R:DEVENV_PROVES_SHIP a claim about the shipped package is never evidenced by a run that installed the dev group -> "DEVENV_PROVES_SHIP"
- R:SILENT_GAP an uncovered platform is never left implied by a missing job -> "SILENT_GAP"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a version matrix has no principals
- A2 [which] covers: S1 · the request does not say WHICH versions; taking "exactly the classifiers, cross-checked against the `requires-python` floor" · probe: they must agree -> found 2026-09-16: classifiers name 3.8-3.12 and `requires-python = ">=3.8"`, so the floor and the lowest classifier agree today; nothing enforces that they keep agreeing
- A3 [when] covers: S1 · the request does not say when a version counts as covered; taking "a job executes it on a pull request" -> a classifier is a promise to a user, and an unexecuted promise is the whole subject of this box
- A4 [absent] covers: S1 · the request does not say what a claimed-but-unmatrixed version means; taking "the check fails, naming the version" -> otherwise adding a classifier is free and silently untrue
- A5 [order] covers: S1 · n/a · versions are independent
- A6 [experience] covers: S1 · the request does not say who reads the result; taking "someone deciding whether they can run yowo on their Python"; ONE published check name per version -> a single rolled-up verdict cannot say WHICH version broke, which is the only thing that reader needs
- A7 [who] covers: S2 · n/a · a smoke run has no principals
- A8 [which] covers: S2 · the request does not say what "the support surface" is; taking "what the package documents as public: `import yowo`, the names `__init__` exports, the public submodules, the model registry, and the CLI entry point" · probe: all of it must work on the oldest claim -> found 2026-09-16 on a real CPython 3.8.20: `import yowo` OK at 2.5.0, public API and submodule imports OK, registry returns 10 detection variants, `yowo --help` and `yowo models` both exit 0
- A9 [when] covers: S2 · the request does not say the budget; taking a declared step timeout · found 2026-09-16: the smoke needs core dependencies only (numpy, opencv-python-headless, click, pyyaml, requests, tqdm), no torch and no onnxruntime, so a leg is an install and a few imports
- A10 [absent] covers: S2 · the request does not say what an uninstallable version means; taking "the job fails" -> a version whose install fails is a claim that is false, which is exactly what this job is for
- A11 [order] covers: S2 · n/a · legs are independent; `fail-fast: false` so one bad version does not hide the rest
- A12 [experience] covers: S2 · the request does not say what the checks are called; taking one published name per version, each registered in `FROZEN_JOB_IDS`, `docs/ci-required-checks.md` and `docs/branch-protection.json`
- A13 [who] covers: S3 · the request does not say who may narrow coverage; taking "a human, with the cause recorded beside the gap"
- A14 [which] covers: S3 · the request does not say which tier runs where; taking "the UNIT tier on the newest claimed version in addition to 3.11, and the claim smoke on all five" · probe: the unit tier cannot simply be matrixed -> **found 2026-09-16, and it changes the shape of this box**: under a real CPython 3.8.20, `src/yowo` compiles 96/96 but the TEST SUITE compiles only 141/163 — 22 modules use parenthesized context managers (3.10+) — and on a real 3.10.20 all 163 compile while 6 modules `import tomllib`, which is 3.11+ stdlib. The test suite's floor is therefore **3.11**, four versions above the package's. Matrixing the unit tier over the claim is not possible without rewriting 22 test files, and the claim is about the shipped package rather than the test suite.
- A15 [when] covers: S3 · the request does not say which newest version; taking 3.12, the highest classifier · probe: it must actually pass -> found 2026-09-16 on a real CPython 3.12.13: **2863 passed, 14 skipped** (the extra skips are ad-hoc-env absences, not failures)
- A16 [absent] covers: S3 · the request does not say what a missing OS job means; taking "recorded as a named gap with its cause" — human decision 2026-09-16, asked and answered: record the gap, add no OS job -> `Backend Conformance` already carries a strict xfail scoped to macOS arm64 for an unexplained PyTorch-vs-ONNX divergence owned by m4, so adding macOS to the integration tier would go red on arrival, which is precisely what this box's own wording warns against
- A17 [order] covers: S3 · n/a · tiers are independent
- A18 [experience] covers: S3 · the request does not say where the gap is recorded; taking "a check that fails if the recorded gap disappears" -> a limitation kept only in prose stops being read

## PLAN
contract:
- `scripts/check_public_surface.py`: the documented public surface, exercised as a consumer. Committed and reviewable rather than inline YAML, and runnable locally on any interpreter.
- `python-claim` job: `strategy.matrix.python-version` over the claimed set, `fail-fast: false`, installing the package with NO dev group, then running that script. One published name per version: `Python Claim (3.8)` ... `Python Claim (3.12)`.
- `unit-newest` job: the unit tier on 3.12, published as `Unit Tests (3.12)`. `Quality Gate` keeps 3.11 and its name, so branch protection is untouched by the matrix.
- `tests/unit/test_claimed_python_versions_run_in_ci.py`: derives the claimed set from pyproject and asserts CI exercises exactly it; asserts the floor agrees with the lowest classifier; asserts the gaps are recorded with their causes.
- Six new required contexts, taking `main` from 12 to 18.

## EDGES
- E1 a classifier added with no matrix leg — fails, naming the version
- E2 a matrix leg for a version the package does not claim — fails, because the sets must be equal, not merely overlapping
- E3 `requires-python` floor and lowest classifier disagree — fails
- E4 the claim job given a dev group — fails, because it would then prove the development environment rather than the shipped package
- E5 the recorded unit-tier floor loses its stated cause — fails, so the limitation cannot decay into prose nobody reads
- E6 the recorded OS gap disappears — fails, so dropping it is a decision rather than an omission

## CHECKS
Names DECLARED here first and used verbatim. Every line carries a trailing
`· <why>` — without it a line parses, runs, passes and binds NOTHING (M21).

tests/unit/test_claimed_python_versions_run_in_ci.py:
- test_every_claimed_python_version_is_executed_by_a_job · covers: M1,M2,S1,S2,A2,A3,A4,E1,E2,R:CLAIM_UNRUN · four of five claimed versions had never executed anything
- test_the_matrix_is_derived_from_pyproject_not_a_hand_written_list · covers: M1,R:HANDLIST,S1,A2 · a hand-kept list beside the claim is the drift this repo has found seven times
- test_the_requires_python_floor_agrees_with_the_lowest_classifier · covers: E3,A2 · two claims that can disagree eventually will
- test_every_claim_job_is_registered_and_published · covers: M2,A12,A6,S2 · three places, or it gates nothing
- test_the_claim_job_exercises_the_documented_public_surface · covers: M2,A8,S2 · an import-only smoke would pass with the CLI broken
- test_the_claim_job_installs_the_package_as_a_consumer_does · covers: M3,E4,R:DEVENV_PROVES_SHIP,A9 · the dev group needs 3.9+ and would prove the wrong thing
- test_the_claim_matrix_does_not_stop_at_the_first_failure · covers: A11 · one bad version must not hide the other four
- test_the_unit_tier_runs_on_the_newest_claimed_version · covers: S3,A14,A15 · 3.12 measured at 2863 passed
- test_the_unit_tier_floor_is_recorded_with_its_cause · covers: M4,E5,A14,R:SILENT_GAP · 3.11, because 6 modules import tomllib
- test_the_operating_system_gap_is_recorded_with_its_cause · covers: M4,E6,A16,A13,R:SILENT_GAP · ubuntu only, by human decision, with the arm64 divergence named

scripts/check_public_surface.py driven by tests/unit/test_claimed_python_versions_run_in_ci.py:
- test_the_public_surface_script_fails_when_the_surface_is_broken · covers: M2,A8,A10 · a smoke that cannot fail proves nothing

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
