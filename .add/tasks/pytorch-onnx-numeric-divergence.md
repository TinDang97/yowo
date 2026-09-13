---
type: Task
title: Close the 0.33 px PyTorch-ONNX coordinate divergence, or declare it
status: direction
depth: standard
milestone: m4-honest-deployment
scope:
  - src/yowo/backends/
  - src/yowo/export/
  - tests/integration/
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
generated: { by: add/3.5.0, at: 2026-09-13 }
verified: []
---
## CARD
goal: Either close the PyTorch-ONNX coordinate divergence, or state plainly why 0.33 px is the right answer.
why: Measured 2026-09-13 by `backend-conformance-suite`, PyTorch vs ONNX on bus.jpg, both pinned cpu/fp32,
  both loading from ONE artifact chain rooted in the same digest-verified weight: detection counts 5 and 5,
  class-id mismatches 0, max box-coordinate deviation **0.33050537 px**, max confidence deviation
  **0.00726026**. The declared tolerance was 1e-3 on coordinates; the measured deviation is 330x it. The
  models are the same model — the arithmetic differs. UNCONFIRMED hypothesis, recorded so the next person
  does not start from zero: the PyTorch backend applies `fuse_conv_and_bn`, folding BN scale and shift into
  the conv weights, while the exported graph does not, and ~0.05% on 640 px coordinates is the magnitude
  that fusion difference produces. That was NOT verified — do not build on it without checking.
  Ruled OUT by measurement: it is not a precision artifact. The exported ONNX is genuinely fp32 (208 FLOAT
  initializers, zero FLOAT16, inputs and outputs FLOAT, sidecar reports `"precision": "fp32"`), and pinning
  both backends to cpu/fp32 produced the identical deviation to letting each choose.
  `tests/integration/test_backend_conformance.py::test_pytorch_and_onnx_agree_within_the_declared_bound`
  holds the bound as `xfail(strict=True)`, so closing this turns that suite RED until the marker is removed
  — that is the signal this task is done, and it is deliberate.
  m3 box 2 stays unchecked until this closes.
beat: scaffold · next: author pytorch-onnx-numeric-divergence's RULES, ASSUMPTIONS and CHECKS, then add freeze pytorch-onnx-numeric-divergence

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
