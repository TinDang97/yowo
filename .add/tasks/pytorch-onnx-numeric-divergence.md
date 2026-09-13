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
goal: Explain — or bound — why PyTorch and ONNX agree on x86_64 and disagree by 0.33 px on Apple Silicon.
why: RE-SCOPED 2026-09-13 after CI refuted the premise this task was first filed on. The original
  filing said PyTorch and ONNX diverge by 0.33050537 px against a 1e-3 bound, measured on the
  author's machine. CI then measured the same comparison, same code, same weight, same artifact
  chain, on ubuntu-latest x86_64: **0.00015450 px — INSIDE the bound**. There is no general
  PyTorch-ONNX divergence. What exists is a **platform split of ~2100x**:
      ubuntu-latest x86_64   0.00015450 px
      macOS 15 arm64         0.33050537 px
  Counts (5 and 5) and every class id agree on both, so it is the same model on both; only arm64's
  coordinate arithmetic drifts. The BN-fusion hypothesis recorded in the original filing is
  STRUCK: fusion is platform-independent and would show on x86_64 too.
  What is NOT yet known, and what this task is for: whether the arm64 path differs in kernel
  selection (PyTorch CPU on ARM via NEON/Accelerate vs onnxruntime's own kernels), in
  accumulation order, or in something that would also affect accuracy rather than only agreement.
  Ruled OUT by measurement: it is not precision. The exported ONNX is genuinely fp32 (208 FLOAT
  initializers, zero FLOAT16) and pinning both backends to cpu/fp32 reproduced the identical
  deviation on each platform.
  `tests/integration/test_backend_conformance.py::test_pytorch_and_onnx_agree_within_the_declared_bound`
  carries `xfail(strict=True)` scoped to macOS arm64, so x86_64 must pass and arm64 must fail —
  closing this turns that suite RED on arm64 until the marker is removed, which is the signal this
  task is done and is deliberate.
  m3 box 2 IS ticked: its words scope it to "the backends CI can run", and on x86_64 they conform.

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
