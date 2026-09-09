---
type: Task
title: A credentialed weights URL is not echoed or logged
status: direction
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/cli/
  - src/yowo/models/
  - tests/unit/
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
generated: { by: add/3.5.0, at: 2026-09-09 }
verified: []
advised_by: edge-reliability-operator
---
## CARD
goal: A credential in a custom model's weights URL is never echoed, logged or raised.
why: found by the security residue lens while verifying `rtsp-credential-redaction`, in code that task does not own.
  - `cli/_main.py:883` — `yowo models` prints `m.default_weights_url` verbatim for every registered model.
  - `models/README.md:119` — the documented way to register a custom model shows an S3 URL, so users are
    actively guided toward putting a fetchable URL in the registry. `https://KEY:SECRET@bucket.s3...` is
    a completely ordinary shape for one.
  - `models/_weights.py` — download failure paths interpolate the URL into messages.
Built-in models use public ultralytics asset URLs and carry no credential, so this is conditional on a
user registering their own. `redact_url()` already exists and is exported from `yowo.io`, so the fix is
to apply it at the emit points — not to build anything new.

NOT a HARD-STOP on rtsp-credential-redaction's gate, and the reasoning is recorded so it can be
challenged: that task's change introduces no leak and removes four. Blocking its gate over an adjacent
pre-existing issue would leave the RTSP credentials leaking in order to punish a fix.
beat: scaffold · next: author weights-url-redaction's RULES, ASSUMPTIONS and CHECKS, then add freeze weights-url-redaction

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
