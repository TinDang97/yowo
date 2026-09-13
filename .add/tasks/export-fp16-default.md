---
type: Task
title: The default export precision produces a loadable artifact
status: done
depth: standard
sensitivity: architecture
milestone: m4-honest-deployment
scope:
  - src/yowo/arch/
  - src/yowo/export/
  - tests/integration/
gives:
  - S1 export_model(spec, fmt, out) at its DEFAULT precision returns a loadable artifact
generated: { by: add/3.5.0, at: 2026-09-13 }
verified:
  - { by: "Tin Dang", at: 2026-09-13, act: interview, authority: human, interview: "sha256:47e993710fea7633", receipt: /tasks/export-fp16-default.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:DEFAULT_NARROWED=confirm|R:MOCKED_EXPORT_PROOF=confirm" }
  - { by: "Tin Dang", at: 2026-09-13, act: freeze, authority: human, direction: "sha256:c5793c1d4f3dde66", binding: "sha256:960275896ecac5fc" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/export-fp16-default.d/runs/1.md }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:bd8547e73926a41e" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/export-fp16-default.d/runs/2.md }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/export-fp16-default.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:1050f78cd04dd34a", binding: "sha256:960275896ecac5fc" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:45969f1820fed85d" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/export-fp16-default.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-13, act: gate, authority: human, outcome: PASS, receipt: /tasks/export-fp16-default.d/runs/4.md, brief: "sha256:45969f1820fed85d" }
advised_by: security-reviewer
---
## CARD
goal: The documented default export invocation writes an artifact that loads, instead of exiting 1 with zero files.
why: Measured 2026-09-13 at HEAD 1853ab0 — `yowo export yolo26n -f onnx`, every flag at its default, exits 1 and writes ZERO files. `export_model` defaults to `Precision.FP16` and the CLI to `-p fp16`; both fail for yolo26n, yolo11n and yolo11n-obb, while `-p fp32` writes 3 files. Only classify survived. This is in no m4 box and sits underneath four of them: "every exported artifact loads from a clean environment" cannot be asserted while the default invocation yields no artifact at all.
beat: done · next: add status

## RULES
<must>
- M1 `export_model` returns an artifact at its DEFAULT precision for detection, OBB and classification models — the value a user gets with no `-p` flag.
- M2 The produced artifact LOADS and infers in a fresh onnxruntime session. Existence and non-zero size are not proof; the FP16 graph must execute.
- M3 FP32 export keeps working unchanged — the fix must not buy the default path at the cost of the path that works today.
</must>
<reject>
- R:DEFAULT_NARROWED The default precision is changed to a value that already worked, making the check pass without repairing FP16 -> "DEFAULT_NARROWED"
- R:MOCKED_EXPORT_PROOF M1 or M2 is proved with `_export_onnx` patched. Every `export_model` call in the unit suite mocks the exporter and passes FP32 or INT8 — never the default — which is exactly why this shipped -> "MOCKED_EXPORT_PROOF"
</reject>

## ASSUMPTIONS
- A1 [who] n/a · export is a local developer action with no actor distinction; no caller identity affects the artifact.
- A2 [which] covers: S1 · the request does not say which model kinds must export at the default; taking all three that ship a CLI verb — detection, OBB, classification — since the CLI offers no kind it cannot export · probe: all three exit 0 and write files -> a kind left broken ships a default that fails for some users.
- A3 [when] n/a · no time or ordering boundary; export is a single synchronous call.
- A4 [absent] covers: S1 · the request does not say what an omitted `precision=` means; taking the signature default `Precision.FP16` as the value under test, not a substitute · probe: the default is still FP16 when the artifact check passes -> if the default silently became FP32 the check would go green over an unfixed FP16 path.
- A5 [order] n/a · single call, no tie to break.
- A6 [experience] covers: S1 · the request does not say who meets this; taking the user running the documented command from the README with no flags -> that user currently gets exit 1 and an empty directory with no indication the default is the problem.

## PLAN
contract: `Detect._init_strides` writes its stride tensor in the dtype of the `stride` buffer it mutates. `torch.export` functionalizes `Tensor.copy_`, so a downstream read of `self.stride` observes the value written rather than the buffer; an fp32 value written into a half buffer makes `make_anchors`' `.to(dtype=half)` raise `Tensor dtype mismatch! Expected: torch.float16, Got: torch.float32` and fails the whole export. Cast before `copy_`. Arithmetic stays in fp32 — strides 8/16/32 and feature sizes 80/40/20 are exact in both dtypes, so the cast loses nothing, and keeping the division in fp32 avoids a half-precision reciprocal.

## EDGES
- E1 FP32 export, which works today, must be unaffected — for an fp32 buffer the added cast is a no-op.
- E2 A classification model, whose export already succeeded at FP16, must keep succeeding.
- E3 An OBB model, which uses the second `_init_strides` call site at `_heads.py:429`, not the detection one.

## CHECKS
- test_the_default_export_invocation_produces_a_loadable_artifact · covers: M1, M2, A2, E2, E3, R:MOCKED_EXPORT_PROOF · runs a REAL `export_model` with no `precision=` argument and loads the result in onnxruntime with a float16 input.
- test_the_default_precision_is_still_fp16 · covers: A4, R:DEFAULT_NARROWED · fails if the default is changed to a value that already worked.
- test_fp32_export_still_works · covers: M3, E1 · the control: the path that worked before the fix is unaffected by it.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- A mocked exporter cannot prove an export works: every `export_model` call in the unit suite patched `_export_onnx` and passed a precision that was never the default, so the default path shipped broken and 50%-covered with lines 132/136 as the only gap in that region. -> add learn tdd
