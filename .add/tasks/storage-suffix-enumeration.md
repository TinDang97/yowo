---
type: Task
title: The storage rule names its classes, like every other entry in the boundary
status: direction
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/_weights.py
  - tests/unit/test_checkpoint_loader.py
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
generated: { by: add/3.5.0, at: 2026-09-09 }
verified: []
---
## CARD
goal: `torch.<X>Storage` is admitted by an enumerated list of storage classes, not by a name suffix.
why: found by the security residue lens while verifying `narrow-loader-allowlist`, as the last
  un-enumerated entry in the trust boundary that task just narrowed everywhere else.
  - `arch/_weights.py` — `_ALLOWED_STORAGE_SUFFIX = "Storage"` admits any `torch` attribute whose name
    ends in `Storage`, via `getattr(torch, name)`.
  - For a real checkpoint the rule is never even consulted: torch's own `UnpicklerWrapper` intercepts
    storage names before ours sees them. It only fires for a name torch did NOT intercept.
  - The reachable shape is `UntypedStorage`, which a checkpoint can ask to construct at an
    attacker-chosen size. That is memory exhaustion, not code execution — `checkpoint-loader`'s
    unpickler still blocks the latter.

bounded, and the bound is why this is a separate node rather than a HARD-STOP on
`narrow-loader-allowlist`'s gate:
  - no code executes, so the worst case is a process that dies allocating;
  - it requires a checkpoint that already passed digest verification, or an unpinned model;
  - and A3 of `narrow-loader-allowlist` explicitly KEPT the suffix rule as a closed generated family,
    a reading a human confirmed on 2026-09-09. Reversing it inside that node would have been a silent
    widening of a frozen decision.

the measurement that would enumerate it already exists: `scripts/measure_checkpoint_globals.py`
  records every global the corpus names, and reported that no storage class reached our `find_class`
  at all across the 10 pinned variants. So the enumerated list is likely to be small or empty, and
  the honest question this node answers is whether the rule can simply be deleted.
beat: scaffold · next: author storage-suffix-enumeration's RULES, ASSUMPTIONS and CHECKS, then add freeze storage-suffix-enumeration

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
