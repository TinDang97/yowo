---
type: Task
title: One check-run name per workflow, so the required context is unambiguous
status: done
depth: quick
milestone: m1-trust-the-ship
scope:
  - .github/workflows/
  - tests/unit/test_ci_contract.py
  - docs/ci-required-checks.md
gives:
  - S1 the check-run names ci.yml publishes on a pull request
  - S2 the check-run names release.yml publishes on a push to main
  - S3 the required-status-check contexts configured on `main`, and docs/ci-required-checks.md
depends_on:
  - /tasks/pr-ci-gate.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:a1f5958e9fb96c13", receipt: /tasks/check-name-collision.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:AMBIGUOUS=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:8ec63f789aebbec7", binding: "sha256:22249aa61fd2594e" }
  - { by: "cli", at: 2026-09-09, act: brief, authority: process, brief: "sha256:7bf1a3c4aafb1378" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/check-name-collision.d/runs/1.md }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/check-name-collision.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-09, act: gate, authority: human, outcome: PASS, receipt: /tasks/check-name-collision.d/runs/2.md, brief: "sha256:fb46da10089bb6c7" }
advised_by: release-planner
---
## CARD
goal: A required status check names exactly one workflow's job, so what GitHub is gating on is never ambiguous.
why: `release.yml`'s `quality` job and `ci.yml`'s both publish the check-run name `Quality Gate`. Found while closing `pr-ci-gate`'s E3 — a live "has this context been observed?" assertion could be satisfied by the wrong workflow, which is why E3's red demonstration had to be taken against a commit with no check-runs at all rather than against `main`.

UPDATE 2026-09-08: the collision WIDENED before this task was authored. `pypi-trusted-publish` gave
release.yml a `Source Distribution` job to satisfy R:UNGATED on the push-to-main path, publishing the
same check-run name ci.yml already used. Two ambiguous names became four:
  ci.yml      quality -> 'Quality Gate'   sdist -> 'Source Distribution'
  release.yml quality -> 'Quality Gate'   sdist -> 'Source Distribution'
Recorded because it shows the failure mode is live and re-occurring, not historical.

beat: done · next: add status

## RULES
<must>
- G1 No check-run name is published by more than one workflow, so a required status check resolves to
  exactly one job.
- G2 The required contexts configured on `main` name jobs that only the pre-merge workflow publishes.
- G3 docs/ci-required-checks.md states the names as they now are, so the next person configuring
  protection is not reading fiction.
</must>
<reject>
- R:AMBIGUOUS No required status check may be satisfiable by a workflow other than the one intended
  -> "AMBIGUOUS"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may add a job; taking: anyone, which is why the
  uniqueness assertion has to live in a check rather than in review discipline
  -> the next appended job re-creates this, exactly as pypi-trusted-publish just did.
- A2 [which] covers: S1 · the request does not say which names change; taking: ci.yml keeps `Quality
  Gate` and `Source Distribution` unchanged, because those are the contexts branch protection is ALREADY
  configured with and renaming them would silently un-gate `main` · probe: protection still names
  contexts that exist -> renaming the protected context leaves main unguarded with a green tick.
- A3 [when] n/a · names are static workflow configuration; no boundary.
- A4 [absent] covers: S1 · the request does not say what an unnamed job publishes; taking: GitHub falls
  back to the job id, so the check must compare EFFECTIVE names (`name` or id), not declared ones
  -> two jobs with no `name:` and the same id across files collide invisibly.
- A5 [order] n/a · a set of names; ordering is not meaningful.
- A6 [experience] covers: S1 · the request does not say who reads these; taking: a reviewer scanning a
  PR's check list, so ci.yml's names stay the short unqualified ones
  -> the everyday reader pays for a problem that only exists on the release path.
- A7 [who] n/a · release.yml's names are configuration; no actor surface.
- A8 [which] covers: S2 · the request does not say how release.yml's names change; taking: suffix them
  with the workflow they belong to — `Quality Gate (release)`, `Source Distribution (release)` — rather
  than renaming ci.yml -> renaming the pre-merge names would break configured protection (A2).
- A9 [when] covers: S2 · the request does not say whether renaming breaks pr-ci-gate's frozen contract;
  taking: it does not — that contract freezes job IDS and the four quality COMMANDS, not display names,
  and the ids are untouched · probe: test_ci_contract's five assertions stay green
  -> a rename that violates a frozen contract is a change-request, not a build.
- A10 [absent] n/a · every job in both files has an explicit `name:`.
- A11 [order] n/a · declarative.
- A12 [experience] covers: S2 · the request does not say who reads a release-path check; taking: whoever
  is watching a release, for whom `(release)` is the useful disambiguator
  -> two identically-named red checks and no way to tell which pipeline failed.
- A13 [who] covers: S3 · the request does not say who may change protection; taking: an admin, and this
  task does NOT re-run the protection API — it only asserts the configured contexts still resolve
  -> a task that silently rewrites branch protection is a task nobody can review.
- A14 [which] covers: S3 · the request does not say which contexts should be required; taking: unchanged
  — `Quality Gate` only, exactly as pr-ci-gate configured it -> quietly widening the required set here
  would block merges on a job nobody agreed to gate on.
- A15 [when] n/a · protection state is read, not scheduled.
- A16 [absent] covers: S3 · the request does not say what to do if a required context matches NO job;
  taking: FAIL — that is the un-gated-main failure and it must be loud
  -> protection referencing a context nothing publishes never blocks anything, and looks configured.
- A17 [order] n/a · a set of contexts.
- A18 [experience] covers: S3 · the request does not say who reads the doc; taking: whoever configures
  protection next, so it names the exact strings and says which workflow owns each
  -> a doc listing an ambiguous name is how this happened.

## PLAN
contract: release.yml's `quality` and `sdist` jobs get `(release)`-suffixed display names; job ids and
  all steps unchanged. docs/ci-required-checks.md updated to state which workflow owns which name.
strategy: assert uniqueness across every workflow by effective name, so a future append fails the check
  instead of re-creating the ambiguity.
regression floor: test_ci_contract.py's five frozen assertions and test_release_contract.py's eleven
  stay green.

## EDGES
- E1 A job with no `name:` publishes its id — the uniqueness check must compare effective names.

## CHECKS
- test_no_check_run_name_is_published_by_two_workflows · covers: G1, R:AMBIGUOUS, A4, E1 · effective
  names across every workflow file are unique.
- test_ci_keeps_the_names_branch_protection_is_configured_with · covers: G2, A2 · ci.yml still publishes
  exactly `Quality Gate` and `Source Distribution`.
- test_required_contexts_resolve_to_a_ci_job · covers: G2, A16 · each documented required context names
  a job in ci.yml and nothing else.
- test_docs_name_the_owning_workflow · covers: G3, A18 · the doc states both names and their owner.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
