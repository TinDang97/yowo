---
type: Task
title: A published artifact provably matches its source
status: direction
depth: standard
sensitivity: architecture
milestone: m1-trust-the-ship
depends_on:
  - /tasks/sdist-manifest.md
scope:
  - .github/workflows/
  - scripts/
  - tests/unit/
  - pyproject.toml
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
---
## CARD
goal: Two builds of the same commit produce byte-identical artifacts, and the digest is recorded so anyone can diff a local build against what was published.
why: split out of m1 box 1 on 2026-09-08. `sdist-manifest` shipped the box's include-list clause and closed at gate PASS having satisfied its own RULES — which I authored narrower than the box. The reproducibility clauses (build twice under a pinned SOURCE_DATE_EPOCH, assert identical digests, record the digest in EVIDENCE) were never built. Without them there is no way to show that a tarball on PyPI was built from the source it claims.

depends_on sdist-manifest (done): the allowlist has to exist first, or a reproducibility check just proves a whole-repo sweep is reproducibly wrong.

not yet authored — Direction is open. Scope, rules and checks still to be written.
beat: scaffold · next: author reproducible-sdist's RULES, ASSUMPTIONS and CHECKS, then add freeze reproducible-sdist

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
