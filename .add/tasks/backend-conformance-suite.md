---
type: Task
title: Every backend is executed, or named unverified with the runner it needs
status: done
depth: standard
milestone: m3-prove-it
scope:
  - tests/integration/
  - tests/unit/
  - src/yowo/backends/
  - .github/workflows/ci.yml
  - pyproject.toml
gives:
  - S1 the committed backend roster — every BackendType executed, or unverified with reason and runner
  - S2 the shared parametrized conformance suite across the backends CI can run
  - S3 the declared cross-backend tolerance, and the deviation actually measured against it
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-13, act: interview, authority: human, interview: "sha256:842b8b5f369b5bc4", receipt: /tasks/backend-conformance-suite.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A13=confirm|A14=confirm|R:SKIPGREEN=confirm|R:MOCKED=confirm|R:LOOSENED=confirm|R:FALSEDEP=confirm" }
  - { by: "unrecorded", at: 2026-09-13, act: interview, authority: human, interview: "sha256:ac005bd2de9b262b", receipt: /tasks/backend-conformance-suite.d/interviews/2.md, answers: "A15=confirm" }
  - { by: "unrecorded", at: 2026-09-13, act: interview, authority: human, interview: "sha256:ac005bd2de9b262b", receipt: /tasks/backend-conformance-suite.d/interviews/3.md, answers: "A1=confirm|A15=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A13=confirm|A14=confirm|R:SKIPGREEN=confirm|R:MOCKED=confirm|R:LOOSENED=confirm|R:FALSEDEP=confirm" }
  - { by: "Tin Dang", at: 2026-09-13, act: freeze, authority: human, direction: "sha256:5de5794ae6079a52", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:5fb668e4e6daf672" }
  - { by: "builder", at: 2026-09-13, act: replan, authority: process, note: "CHECKS naming drift, corrected so every covers: key binds a test that exists. The declared test_no_executed_entry_is_satisfied_by_a_mock shipped as test_the_executed_set_is_what_the_conformance_suite_drives, which is strictly stronger: it binds R:MOCKED by asserting the suite parametrises over the roster itself rather than a second hand-maintained list, AND that no mock appears in the suite at all. Six further checks were added during build and are declared so their coverage binds - notably test_the_core_import_is_used_by_load_not_duplicated, the wiring check that lesson Q8 demands, and test_the_backends_ci_cannot_reach_say_which_runner_they_need, which is parametrised and so is declared one line per case per lesson M12." }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:ffc7958c461514d6", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:490616a68132180d" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:ffc7958c461514d6", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:8f1258654c9788be" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:46a9bd150b2b1b7d", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:c1c682f34cc3bd61" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-13, act: gate, authority: human, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/3.md, brief: "sha256:c1c682f34cc3bd61" }
  - { by: loop, at: 2026-09-13, act: reopen, to: direction, reason: "CI refuted the node's central measurement after the gate. The 1e-3 bound was declared correctly, but the deviation recorded against it - 0.33050537 px - was measured on macOS arm64 and pinned in the suite as though it were a property of the code. ubuntu x86_64 measured 0.00015450 for the same code, same weight, same artifact chain: the strict xfail XPASSed and the pinned-deviation check failed, both correctly. The general PyTorch-ONNX divergence this node reported does not exist; a ~2100x PLATFORM split does. The contract moved - CHECKS changed and m3 box 2 went from not-ticked to ticked - so this reopens rather than being edited under a closed gate." }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:da9303b0ef9293bd", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:5eafb0e9f802a1e5" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:7e2028e132122e91", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:1f5f462523e2eb00" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/5.md }
  - { by: "Tin Dang", at: 2026-09-13, act: gate, authority: human, outcome: PASS, receipt: /tasks/backend-conformance-suite.d/runs/5.md, brief: "sha256:1f5f462523e2eb00" }
advised_by: inference-parity-engineer
---
## CARD
goal: Every backend yowo can hand a user is either executed against a real weight, or named as unverified with the runner it would take.
why: Measured 2026-09-13. `create_backend()` returns FIVE backends; exactly one (`pytorch`) is executed
  unmocked anywhere, by `real-backend-smoke`. Coverage: `_openvino.py` **21%**, `_pytorch.py` **58%**,
  `_tensorrt.py` 73%, `_onnx.py` 77% — and there is no `test_pytorch_backend.py` at all. The runtimes
  for `pytorch` and `onnx` are already in the dev group; `openvino>=2024.0` is a declared extra that
  installs cleanly on this platform (openvino==2026.3.1, 2 packages).
  CORRECTED 2026-09-13: this CARD first reported a general PyTorch-ONNX divergence of 0.33 px. CI
  refuted it — x86_64 measures 0.00015450 for the same code and weight. There is no general
  divergence; there is a ~2100x platform split, owned by `pytorch-onnx-numeric-divergence` in m4,
  and the BN-fusion hypothesis is struck because fusion is platform-independent.
  And OpenVINO cannot run at all: `_openvino.py:102` does `from openvino.runtime import Core`, a module
  REMOVED in OpenVINO 2025+, so on 2026.3.1 the import raises and the handler at :105 relabels it
  `DependencyError("openvino", "uv add openvino")` — telling a user to install a package they already
  have. In the same process the modern API works: `ov.Core()` reports `['CPU']`, `read_model` and
  `compile_model` both succeed. The declared extra admits exactly the versions where the backend is dead.
  That is why 21% is 21%.
beat: done · next: add status

## RULES
<must>
- M1 Every `BackendType` `create_backend()` can return is EITHER executed unmocked against a real
  weight by a check CI runs, OR carries a committed roster entry naming why it is not and the runner
  it would need. A check that skipped counts as unverified, never as covered.
- M2 One shared parametrized suite drives every executable backend from a SINGLE artifact chain
  rooted in the digest-verified weight — so the backends are known to hold the same weights rather
  than assumed to — asserting output shape, dtype, behaviour on an empty-detection input, class
  mapping, and the error type raised on bad input.
- M3 The numeric tolerance is declared BEFORE the run — 1e-3 absolute on box coordinates, exact on
  class ids — and the deviation actually measured is reported against it. The bound is not moved to
  fit a result, AND no observed value is asserted as a constant: a figure measured on one machine is
  a property of that machine. CORRECTED 2026-09-13 after CI refuted the first version, which pinned
  0.33050537 as if it were the code's behaviour. PyTorch vs ONNX on bus.jpg, both pinned cpu/fp32,
  same weights, same artifact chain — counts 5 and 5 and **0** class-id mismatches on both, and a
  ~2100x PLATFORM split on coordinates: **ubuntu x86_64 0.00015450 (inside the bound)**, **macOS
  arm64 0.33050537 (330x it)**. The suite asserts the bound on whatever platform runs, records both
  figures as documentation, and scopes a strict xfail to macOS arm64 so x86_64 must PASS and arm64
  must FAIL — either flipping is a red.
- M4 `OpenVinoBackend.load` imports `Core` from the module OpenVINO actually publishes on >=2025,
  falling back to the pre-2025 path, and a load failure that is NOT a missing package stops being
  reported as one.
- M5 CI installs the openvino extra and runs this suite, so the roster's claims are re-proved on
  every PR rather than asserted once.
</must>
<reject>
- R:SKIPGREEN no backend is recorded as covered by a check that skipped -> "SKIPGREEN"
- R:MOCKED no roster entry claims execution that a mock satisfied -> "MOCKED"
- R:LOOSENED the declared tolerance is never widened to make the suite green -> "LOOSENED"
- R:FALSEDEP no backend reports a missing dependency for a failure that is not one -> "FALSEDEP"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the box does not say who owns a backend nobody can run; taking the reading
  that the ROSTER owns it — one committed entry per BackendType, with the runner named — rather than
  a reviewer's memory -> an unrunnable backend silently reads as covered, which is today's state.
  · probe: the shipped roster must enumerate `BackendType` at runtime, not a hand-written list.
- A15 [who] covers: S3 · the box does not say who owns a deviation that exceeds the declared bound;
  taking the reading that THIS node owns stating it and m4 owns closing it, because m3's scope puts
  fixes in m4 and a gap with no named owner is a gap nobody closes -> the 0.33 px divergence is
  recorded once and then belongs to no milestone.
- A2 [who] covers: S2 · the box does not say who supplies the artifact each backend loads; taking the
  reading that ONE chain from the digest-verified weight feeds all three, since two files pinned
  separately cannot be known to hold the same weights -> the suite compares two different models and
  reports their difference as a backend divergence.
- A3 [which] covers: S1 · the box does not say which backends CI "can run"; taking pytorch, onnx and
  openvino — measured installable on ubuntu CPU — with tensorrt (NVIDIA GPU runner) and coreml
  (macOS runner) on the unverified list -> openvino stays at 21% behind a reason that is really a
  choice.
- A4 [which] covers: S2 · the box lists "output shape/dtype, empty detections, class mapping, error
  type on bad input"; taking those five literally as the conformance axes, no more -> the suite grows
  into a parity suite and duplicates `export-roundtrip-parity`.
- A5 [which] covers: S3 · the box does not say which quantity the tolerance bounds; taking box
  COORDINATES in pixels, with class ids exact and detection COUNT exact -> a tolerance on confidence
  alone would pass a backend that boxed the wrong pixels.
- A6 [when] covers: S3 · the box says the tolerance is stated "before the run"; taking that to forbid
  moving it AFTER seeing a result, not to forbid having measured first -> a number chosen to fit the
  deviation, which is the failure the clause exists to stop.
- A7 [when] covers: S1,S2 · the box does not say when the roster is re-proved; taking EVERY PR, via
  CI, rather than once at authoring -> the roster rots exactly like the coverage numbers above did.
- A8 [absent] covers: S1 · the box does not say what an absent runtime means; taking it as
  UNVERIFIED with the runner named, never as absent-therefore-fine -> a `skipif` reports green and
  the backend reads as covered, which is lesson Q3.
- A9 [absent] covers: S2 · the box does not say what an empty-detection input proves; taking the
  reading that every backend must return an empty result of the SAME shape and dtype rather than
  raising or returning None -> the degraded-mode contract m2 box 8 established is unenforced across
  backends.
- A10 [absent] covers: S3 · the box does not say what to do when the measured deviation exceeds the
  declared bound; taking the milestone's own risk line — surface it, do not weaken it — so the check
  keeps the bound and is recorded as an expected failure that alarms the moment it is fixed -> the
  number gets loosened and the suite proves nothing.
- A11 [order] covers: S2 · detections come back in no guaranteed order across backends; taking a sort
  by descending confidence before pairing -> the comparison reports a divergence that is only a
  permutation.
- A12 [order] covers: S1,S3 · n/a · the roster is a set and the tolerance a scalar; neither has an
  order to fix or a tie to break.
- A13 [experience] covers: S1 · the box does not say who reads the roster; taking the reading that it
  is someone choosing a backend for a deployment, who needs the reason and the runner in the same
  line -> "unverified" without a reason is indistinguishable from "untested because nobody looked".
- A14 [experience] covers: S2,S3 · the box does not say what the suite reports on failure; taking the
  reading that it must name the two backends, the axis, and the measured value -> a reader sees
  "assert 0.33 < 0.001" and cannot tell which backends diverged or on what.

## PLAN
contract: `tests/integration/test_backend_conformance.py` holds a session fixture that resolves the
  digest-verified weight and exports it ONCE to ONNX and to OpenVINO IR, so all three backends load
  from one chain. A committed roster enumerates `BackendType` at runtime and maps each member to
  EXECUTED or an unverified entry carrying its reason and required runner; a unit check asserts the
  roster is total over the enum, so a sixth backend cannot be added without a verdict. The
  conformance checks are parametrized over the executable backends for the five axes the box names.
  The numeric comparison declares 1e-3 on coordinates and exact class ids, sorts by confidence before
  pairing, and reports the measured deviation in its failure message. The PyTorch-vs-ONNX numeric
  case is marked `xfail(strict=True)` carrying the measured 0.33050537, so the bound is unchanged,
  the gap is visible, and closing it turns the suite red until the marker is removed. `_openvino.py`
  imports `Core` from `openvino` with a fallback to `openvino.runtime`, and its handler stops
  reporting a non-import failure as a missing dependency. CI gains the openvino extra and a job that
  runs this file with `CI=true`.

## EDGES
- E1 a backend whose runtime is absent -> unverified entry, never a green skip.
- E2 a new `BackendType` added with no roster verdict -> the totality check fails.
- E3 an all-black frame -> every backend returns an empty result of the same shape and dtype.
- E4 a corrupt or wrong-format weight -> every backend raises the same error TYPE.
- E5 detections returned in a different order -> sorted before pairing, no false divergence.
- E6 the measured deviation drops below the declared bound -> the strict xfail turns red, so the fix
  cannot land unnoticed.
- E7 openvino <2025 (the `openvino.runtime` path) -> still loads, the fallback is exercised.
- E8 an OpenVINO load failure that is NOT an ImportError -> reported as a load error, never as a
  missing dependency.

## CHECKS
- test_the_roster_is_total_over_the_backend_enum · covers: M1, A1, E2 · every `BackendType` member
  has a verdict, enumerated at runtime.
- test_every_unverified_entry_names_a_reason_and_a_runner · covers: M1, A13, A8, E1 · no bare
  "unverified".
- test_the_executed_set_is_what_the_conformance_suite_drives · covers: R:MOCKED · the suite
  parametrises over the roster itself, not a second list, and contains no mock at all.
- test_every_backend_has_exactly_one_verdict · covers: M1, A1 · executed and unverified are disjoint.
- test_openvino_is_executed_not_excused · covers: M1, A3 · measured installable, therefore executed.
- test_the_backends_ci_cannot_reach_say_which_runner_they_need[tensorrt] · covers: M1, A3, A13 ·
  the reason names a machine, not a shrug.
- test_the_backends_ci_cannot_reach_say_which_runner_they_need[coreml] · covers: M1, A3, A13 ·
  likewise. One line per case: a parametrised check cited by its bare name binds nothing (M12).
- test_the_core_import_is_used_by_load_not_duplicated · covers: M4 · `load()` goes through the
  helper, so the fallback is live on the real path. The wiring check lesson Q8 demands.
- test_the_conformance_job_is_in_the_frozen_ci_contract · covers: M5 · appended under pr-ci-gate's
  APPEND rule rather than smuggled in.
- test_a_skipped_backend_is_not_recorded_as_covered · covers: R:SKIPGREEN, A8 · a skip maps to
  unverified.
- test_all_backends_load_from_one_artifact_chain · covers: M2, A2 · the ONNX and IR artifacts are
  derived from the digest-verified weight in-fixture.
- test_output_shape_and_dtype_agree[pytorch] · covers: M2, A4 · valid box geometry and dtypes.
- test_output_shape_and_dtype_agree[onnx] · covers: M2, A4 · likewise.
- test_output_shape_and_dtype_agree[openvino] · covers: M2, A4 · likewise.
- test_an_empty_input_yields_an_empty_result_everywhere[pytorch] · covers: M2, A9, E3 · an empty result, not a raise.
- test_an_empty_input_yields_an_empty_result_everywhere[onnx] · covers: M2, A9, E3 · likewise.
- test_an_empty_input_yields_an_empty_result_everywhere[openvino] · covers: M2, A9, E3 · likewise.
- test_class_mapping_agrees · covers: M2, A4 · the same class ids and names.
- test_bad_input_raises_the_same_error_type[pytorch] · covers: M2, E4 · one error type across backends.
- test_bad_input_raises_the_same_error_type[onnx] · covers: M2, E4 · likewise.
- test_bad_input_raises_the_same_error_type[openvino] · covers: M2, E4 · likewise.
- test_detections_are_paired_by_confidence_not_position · covers: A11, E5 · a permutation is not a
  divergence.
- test_the_declared_tolerance_is_one_thousandth_of_a_pixel · covers: M3, R:LOOSENED, A5, A6 · the
  bound is pinned as a constant a reader can find, and pinned again in the milestone box.
- test_the_arm64_expectation_is_strict_and_platform_scoped · covers: E6 · the gap is pinned on both
  sides — strict, so arm64 conforming turns red; platform-scoped, so x86_64 must pass on its own.
  An xfail cannot bind E6 itself, because an expected failure never reports as passing (Q10).
- test_pytorch_and_onnx_agree_within_the_declared_bound · covers: M3, A10 · the declared bound,
  asserted on whatever platform runs. Passes on x86_64 at 0.00015450; carries a strict xfail scoped
  to macOS arm64, which measures 0.33050537. Both sides are pinned: either flipping is a red.
- test_a_disagreement_names_both_backends_the_axis_and_the_value · covers: M3, A14 · what a reader
  sees when two backends disagree, checked without needing a disagreement and without pinning a
  machine's number.
- test_both_platform_measurements_are_documented_not_asserted · covers: M3, A6 · the two platform
  figures are recorded for a reader and never asserted against a live measurement. Added after CI
  refuted the first version, which pinned the arm64 figure as though it were a property of the
  code (lesson Q11).
- test_class_ids_match_exactly_across_backends · covers: M3, A5 · measured 0 mismatches; this half
  of the tolerance is not xfailed.
- test_openvino_imports_core_from_the_published_module · covers: M4, E7 · the modern path first,
  the pre-2025 path as fallback.
- test_an_openvino_load_failure_is_not_reported_as_a_missing_dependency · covers: M4, R:FALSEDEP, E8 · a non-ImportError failure raises a load error, not a missing package.
- test_openvino_actually_executes_on_a_real_weight · covers: M1, M4 · the fix is proved by running
  it, not by reading the import.
- test_ci_installs_the_openvino_extra_and_runs_this_suite · covers: M5, A7 · the workflow is read,
  and the job is named.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
