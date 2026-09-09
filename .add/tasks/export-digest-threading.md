---
type: Task
title: The export path verifies its weights the way the inference path does
status: direction
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/export/
  - tests/unit/
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
generated: { by: add/3.5.0, at: 2026-09-09 }
verified: []
---
## CARD
goal: The export path re-compares a weight against its pin before conversion, exactly as the
  inference path now does.
why: found by the security residue lens while VERIFYING `verified-digest-threading`, in code that task
  does not own. That node closed the resolve-to-load window for the inference path only.
  - `export/_exporter.py:69` — `weights_path = resolve_weights(spec)`, then all three loaders are
    called with no digest: `load_classify_weights(model, weights_path)` (:80),
    `load_obb_weights(...)` (:91), `load_weights(...)` (:99).
  - Same window, same conversion, same restricted-unpickler read of the raw `.pt`. A file substituted
    between the resolve and the load is converted without ever being compared to the pin.

  `verified-digest-threading` wired the gate: all three loaders now TAKE `raw_digest` and
  `load_verified_state_dict` re-verifies when one is supplied. This node passes it. Two of the three
  branches (`classify`, `obb`) already hold the `meta` the pin lives on; the detection branch does not
  call `get()` yet and will need to.

not a regression, and not a HARD-STOP on `verified-digest-threading`'s gate: the default is `None`, so
  export behaves exactly as it did before that task. But that node's claim — that the window is shut —
  holds only for the inference path until this one lands, and a claim that is true of one path and
  asserted of the system is the kind of thing that gets believed.

bounded the same way its sibling was: it requires local write access to the weight cache, and
  `checkpoint-loader`'s restricted unpickler still blocks code execution regardless, so the worst case
  is a model exported from substituted weights — not RCE. That is worse here than at inference, in one
  specific way: an exported artifact is a FILE that outlives the process and gets shipped somewhere
  else.
beat: scaffold · next: author export-digest-threading's RULES, ASSUMPTIONS and CHECKS, then add freeze export-digest-threading

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
