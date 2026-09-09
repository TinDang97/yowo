---
type: Task
title: The verified digest reaches the loader, closing the resolve-to-load window
status: direction
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/engine.py
  - src/yowo/arch/_weights.py
  - tests/unit/
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
generated: { by: add/3.5.0, at: 2026-09-09 }
verified: []
advised_by: artifact-integrity-steward
---
## CARD
goal: The digest `resolve_weights` verified is passed to the loader, so nothing between them can substitute the file.
why: found by the security residue lens while verifying `weight-integrity`, as a residual limit of that task rather than a defect it introduced.
  `resolve_weights` verifies the cached file against its pin and returns a path. `load_verified_state_dict`
  then re-hashes that path — but only to KEY the converted sidecar, not to re-check it against the pin.
  A file substituted in the window between those two calls yields a different key, so the sidecar misses,
  and the substituted bytes are converted and loaded without ever being compared to the pin.

  `load_verified_state_dict` already TAKES a `raw_digest` argument; nothing passes one. Threading the
  verified value from the engine and re-checking before conversion closes it — but the engine is outside
  `weight-integrity`'s frozen scope, which is why this is a separate node rather than a silent widening.

bounded, and the bound is why this was not a HARD-STOP on weight-integrity's gate:
  - it requires local write access to the weight cache;
  - code execution is already blocked regardless, by `checkpoint-loader`'s restricted unpickler — the
    worst case is substituted model weights, not RCE;
  - before `weight-integrity` there was no verification at any point, so the window is a narrowing of an
    open door, not a new one. Blocking that gate would have kept the door fully open to punish a fix.
beat: scaffold · next: author verified-digest-threading's RULES, ASSUMPTIONS and CHECKS, then add freeze verified-digest-threading

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
