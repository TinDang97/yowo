---
type: Task
title: Load an ultralytics .pt without ultralytics, and without unpickling foreign code
status: direction
depth: deep
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/_weights.py
  - tests/unit/test_checkpoint_loader.py
  - scripts/verify_checkpoint_equivalence.py
  - tests/fixtures/checkpoint_reference.json
gives:
  - S1 `load_weights(model, checkpoint_path)` — the public entry that populates a native model from a .pt file
  - S2 the restrictive unpickler's accept/refuse decision — which pickled classes may be constructed while reading a checkpoint
  - S3 the error raised when a checkpoint cannot be read safely — the type and message a caller branches on
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-08, act: interview, authority: human, interview: "sha256:612eea024c436da8", receipt: /tasks/checkpoint-loader.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:ARBITRARY_IMPORT=confirm|R:SILENT_PARTIAL=confirm" }
  - { by: "Tin Dang", at: 2026-09-08, act: freeze, authority: human, direction: "sha256:a4ec5ff62314901a", binding: "sha256:5e828bdea5ba3705" }
advised_by: security-reviewer
---
## CARD
goal: A downloaded `.pt` checkpoint loads into a native yowo model without `ultralytics` installed and without constructing any class the checkpoint names.
why: `arch/_weights.py:86` unpickles the checkpoint's `ema`/`model` value, which is an `ultralytics.nn.tasks.DetectionModel`. That makes ultralytics an undeclared runtime dependency the package cannot satisfy — `yowo detect image.jpg` on a clean install raises `ModuleNotFoundError` — and it is why `weights_only=True` cannot simply be switched on.

## RULES
<must>
- M1 `load_weights` populates a native model from an ultralytics-format `.pt` with `ultralytics` NOT importable.
- M2 Reading a checkpoint constructs no class named by the checkpoint. Tensors and plain containers are materialised; every other global resolves to an inert placeholder whose attributes are never executed.
- M3 The state_dict extracted is byte-equivalent to what the current loader extracts, for every shipped variant — detection, `-cls` and `-obb`, across YOLO11 and YOLO26. This is a refactor of HOW the tensors are read, never of WHICH tensors result.
- M4 EMA preference is preserved: `ema` weights win over `model` weights, matching today's behaviour.
- M5 A checkpoint that cannot be read safely fails with a typed yowo error naming the offending global, never with a bare `UnpicklingError`, `ModuleNotFoundError`, or a partially-populated model.
</must>
<reject>
- R:ARBITRARY_IMPORT Resolving a checkpoint-named global by importing the module it names. That is the current behaviour and it is the vulnerability — a checkpoint chooses what gets imported. -> "ARBITRARY_IMPORT"
- R:SILENT_PARTIAL Returning a model with some parameters loaded and some left at their initialised values because a key did not map. A silently half-loaded model produces plausible, wrong detections. -> "SILENT_PARTIAL"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whether a caller may supply a checkpoint from an untrusted source; taking "yes — `--weights /path/to/file.pt` is a documented public flag, so any checkpoint is potentially attacker-chosen" -> if wrong, nothing; this is the conservative reading and costs only strictness.
- A2 [who] covers: S2 · the request does not say who decides the allowlist; taking "this node owns it as a frozen contract, and widening it later is a change-request, not a build detail" -> if wrong, the allowlist grows silently and the trust boundary erodes without review.
- A3 [who] covers: S3 · n/a · an error type has no authorization dimension; who may call `load_weights` is governed by S1.
- A4 [which] covers: S1 · the request does not say which checkpoint formats must load; taking "the ultralytics `.pt` layouts the current loader already handles — raw `state_dict`, and dict-with-`ema`/`model` — and no others" -> if wrong, a format a user has works today and breaks after this lands. Bounded by M3.
- A5 [which] covers: S2 · the request does not say which globals are legitimate; taking "`torch` tensor-rebuild functions, `collections.OrderedDict`, and the primitive containers pickle itself needs; `ultralytics.*` and `models.*` resolve to inert stubs; everything else is refused" -> if wrong, a legitimate checkpoint is rejected, which M5 makes a clear error rather than a mystery. · found: confirmed by the human at freeze, 2026-09-08 — refuse-by-default was chosen over stub-everything, so a checkpoint carrying a non-torch global is reported to the operator rather than silently tolerated. · probe: every shipped variant loads and its state_dict matches the current loader's.
- A6 [which] covers: S3 · the request does not say which yowo error type; taking `ModelLoadError` if it exists, else a new one in `errors.py` — but `review-api.md` records `ModelLoadError` as documented in the backend Protocol and raised by nobody -> if wrong, callers branch on a type that never fires. · found: `ModelLoadError` DOES exist in `src/yowo/errors.py` (class list confirmed 2026-09-08) and is raised by no code path — `review-api.md` records it as documented in the backend Protocol and never raised. So this node gives an existing-but-dead error type its first real caller, rather than adding one. · probe: the raised type is importable from `yowo.errors` and is what the check asserts.
- A7 [when] covers: S1 · the request does not say whether this applies to already-cached checkpoints; taking "yes — the loader is the only reader, so cache age is irrelevant to it" -> low cost.
- A8 [when] covers: S2 · the request does not say when the allowlist is consulted; taking "at `find_class` time, before any object is constructed", which is the only point where refusal still prevents execution -> if wrong, the check happens after the damage.
- A9 [when] covers: S3 · the request does not say whether a refusal aborts the whole load or skips the offending key; taking "aborts", because a skip is R:SILENT_PARTIAL -> if wrong, a hostile checkpoint degrades into a half-loaded model.
- A10 [absent] covers: S1 · the request does not say what happens when neither `ema` nor `model` is present; taking today's behaviour — raise with the available keys listed -> low cost, already the shipped behaviour.
- A11 [absent] covers: S2 · the request does not say what an inert stub must do when the unpickler calls it; taking "accept any constructor args and any state, record nothing, execute nothing" — it exists only so pickle can finish walking the stream to reach the tensors -> if wrong, a stub that raises turns a loadable checkpoint into a failure.
- A12 [absent] covers: S3 · the request does not say what to report when the refused global has no module; taking "name whatever the pickle stream said, verbatim and quoted, without importing it" -> if wrong the operator cannot tell which checkpoint key was hostile.
- A13 [order] covers: S1 · the request does not say the order of EMA preference versus dtype conversion; taking today's order — select `ema` first, then cast to float32 -> if wrong, FP16 EMA weights are cast from the wrong source. Bounded by M3.
- A14 [order] covers: S2 · n/a · `find_class` decisions are independent per global; no ordering exists between them.
- A15 [order] covers: S3 · the request does not say which failure is reported when a checkpoint trips several rules; taking "the first refusal, and abort" -> low cost, and A9 already forbids continuing.
- A16 [experience] covers: S1 · the recipient is a user running `yowo detect image.jpg` on a fresh install. The request does not say what they should experience; taking "nothing — it simply works, with no new flag and no mention of pickling", because the current experience is a `ModuleNotFoundError` naming a package they never installed -> if wrong, the headline bug is still present in a new costume.
- A17 [experience] covers: S2 · the recipient is a future maintainer asked to add a class to the allowlist. The request does not say how they judge; taking "each allowlist entry carries a one-line why, in the source, next to the entry" -> if wrong, the list grows by cargo cult and the boundary stops meaning anything.
- A18 [experience] covers: S3 · the recipient is an operator whose checkpoint was refused. The request does not say what they need; taking "the refused global, the file path, and the sentence that a refusal means the checkpoint contains code, not just weights" -> if wrong they retry the same file, or worse, go looking for a flag to disable the check.

## PLAN
contract: `load_weights` keeps its signature. Internally, `_extract_state_dict` stops calling `torch.load(weights_only=False)` and instead drives `torch.load` with a restricted unpickler whose `find_class` consults a frozen allowlist: torch's tensor-rebuild globals and plain containers are permitted; `ultralytics.*` resolves to an inert stub class; anything else raises. The stub exists purely so the pickle stream can be walked to the tensors it carries.

strategy: (1) write `tests/unit/test_checkpoint_loader.py` — a clean-environment test that blocks the `ultralytics` import and loads the tracked `yolo11n.pt`, plus a refusal test asserting a hostile global is rejected by name, plus an equivalence test comparing the extracted state_dict against the current loader's output. Run RED; (2) implement the restricted unpickler; (3) green; (4) confirm equivalence across every shipped variant per M3.

scope: `src/yowo/arch/_weights.py`, `tests/unit/test_checkpoint_loader.py`, `scripts/verify_checkpoint_equivalence.py`, `tests/fixtures/checkpoint_reference.json`.

hermetic by construction: the unit tests craft ultralytics-format checkpoints with a custom `Pickler` that writes the global name `ultralytics.nn.tasks.DetectionModel` WITHOUT the class existing. So M1, M2, M4 and M5 are provable with no real weights file, no network, and no AGPL artifact in the test tree. Only M3's equivalence needs the real checkpoint.

regression floor: the full unit suite stays green; no public signature changes; `weight-integrity` and `ci-weight-fixture` depend on this node and must not need to re-cross it.

## EDGES
- E1 A checkpoint whose `ema` is present but `None` — `ckpt.get("ema") or ckpt.get("model")` already handles this by falsiness, and the refactor must not lose it.
- E2 A checkpoint carrying a global that is neither torch nor ultralytics — e.g. `posix.system`. This is the attack, and the refusal must name it rather than importing it.
- E3 A raw `state_dict` checkpoint with no wrapper dict at all, which must still load without the stub path being reached.

## CHECKS
- test_loads_without_ultralytics_importable · covers: M1, A5 · blocks the `ultralytics` import, loads the tracked `yolo11n.pt` through `load_weights`, asserts the model populates. Red today with `ModuleNotFoundError: No module named 'ultralytics.nn.tasks'`.
- test_refuses_a_foreign_global_by_name · covers: M2, R:ARBITRARY_IMPORT, E2 · builds a checkpoint naming a non-torch, non-ultralytics global and asserts the load is refused, that the message quotes the global, and that the module was never imported.
- verify_checkpoint_equivalence · covers: M3, E1, E3 · `scripts/verify_checkpoint_equivalence.py <ckpt>` compares the new loader's state_dict against a reference digest captured from the CURRENT loader before the change — same keys, same dtypes, same per-tensor sha256 — and emits JUnit XML. It is a script, not a unit test, because it needs a real multi-megabyte checkpoint that CI cannot obtain until `ci-weight-fixture` lands, and a `skipif`-green unit test would prove nothing.
- test_ema_weights_win_over_model_weights · covers: M4 · asserts EMA preference survives the refactor.
- test_unreadable_checkpoint_raises_a_typed_yowo_error · covers: M5, A6, R:SILENT_PARTIAL · asserts the error is importable from `yowo.errors`, names the offending global, and that no partially-populated model is returned.

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
