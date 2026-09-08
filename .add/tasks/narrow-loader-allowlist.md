---
type: Task
title: Enumerate the torch layer classes the allowlist admits, instead of a prefix
status: direction
depth: standard
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/_weights.py
  - tests/unit/test_checkpoint_loader.py
  - scripts/
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
depends_on:
  - /tasks/checkpoint-loader.md
  - /tasks/ci-weight-fixture.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: security-reviewer
---
## CARD
goal: The checkpoint allowlist names the torch layer classes it admits, rather than admitting a namespace.
why: `checkpoint-loader` froze `torch.nn.modules.` as a PREFIX rule, which admits any class in that namespace. Every one is data-bearing so nothing is currently wrong, but it is the broadest entry in the trust boundary and the only one that is not enumerated. Narrowing it needs the real set measured across all ten shipped variants — which is why this depends on `ci-weight-fixture` for reachable checkpoints, rather than being guessed from the single `yolo11n.pt` available when the loader was built.
beat: scaffold · next: author narrow-loader-allowlist's RULES, ASSUMPTIONS and CHECKS, then add freeze narrow-loader-allowlist

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
