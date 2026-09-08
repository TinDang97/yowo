---
type: Task
title: One check-run name per workflow, so the required context is unambiguous
status: direction
depth: quick
milestone: m1-trust-the-ship
scope:
  - .github/workflows/
  - tests/unit/test_ci_contract.py
  - docs/ci-required-checks.md
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
depends_on:
  - /tasks/pr-ci-gate.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: release-planner
---
## CARD
goal: A required status check names exactly one workflow's job, so what GitHub is gating on is never ambiguous.
why: `release.yml`'s `quality` job and `ci.yml`'s both publish the check-run name `Quality Gate`. Found while closing `pr-ci-gate`'s E3 — a live "has this context been observed?" assertion could be satisfied by the wrong workflow, which is why E3's red demonstration had to be taken against a commit with no check-runs at all rather than against `main`.
beat: scaffold · next: author check-name-collision's RULES, ASSUMPTIONS and CHECKS, then add freeze check-name-collision

## RULES
<must>
- M1 <the rule that must hold>
</must>
<reject>
- R:<NAME> <what must never happen> -> "<NAME>"
</reject>

## ASSUMPTIONS
- A1 [who] covers: <S ids> · the request does not say <who may act / whose data>; taking <reading> -> <cost if wrong>
- A2 [which] covers: <S ids> · the request does not say <which rows/cases are in>; taking <reading> -> <cost if wrong>
- A3 [when] covers: <S ids> · the request does not say <where the boundary falls>; taking <reading> -> <cost if wrong>
- A4 [absent] covers: <S ids> · the request does not say <what a missing value means>; taking <reading> -> <cost if wrong>
- A5 [order] covers: <S ids> · the request does not say <what orders / breaks a tie>; taking <reading> -> <cost if wrong>
- A6 [experience] covers: <S ids> · the request does not say <who receives this and what would make it hard for them>; taking <reading> -> <cost if wrong>
every `gives:` surface is swept on every dimension; `[<dim>] n/a · <why>` retires one. one line, one silence — split, never bundle. `· probe: <what shipped behavior must show>` declares a reading checkable: cite its A id from CHECKS and the gate holds the PASS to it.

## PLAN
contract: <the shape this publishes>

## EDGES
- E1 <a boundary or failure case a check must cover — optional>

## CHECKS
- <test_name> · covers: M1 · <what it proves>
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
