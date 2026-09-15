---
type: Task
title: A dependency no job installs turns its tests into a green skip
status: done
depth: standard
milestone: m3-prove-it
scope:
  - tests/
  - .github/workflows/
gives:
  - S1 a check that fails when an `importorskip`'d dependency is not provided by the job running that module, resolved from pyproject.toml
  - S2 the one measured gap closed: openvino installed in the job that runs tests/unit/
  - S3 both quality gates providing the same dependencies, bound by a check
generated: { by: add/3.5.0, at: 2026-09-16 }
verified:
  - { by: "unrecorded", at: 2026-09-16, act: interview, authority: human, interview: "sha256:fc288d570d1d2abd", receipt: /tasks/conditional-dep-install.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:HANDMAP=confirm|R:GREEN_BY_SKIP=confirm|R:SILENT_ACCEPT=confirm|R:TEXT_MATCH=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:f3ff9606c636d384", binding: "sha256:22a1d46ededa489a" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:cd6efb6d8cedfc83" }
  - { by: "builder", at: 2026-09-16, act: replan, authority: process, note: "PLAN said the new guard REPLACES test_chromadb_is_installed_where_its_tests_run.py. The replacement is PARTIAL: that file's test_an_absent_chromadb_fails_under_ci_rather_than_skipping asserts CI=true on every step running a conditional module, which this node declared no check for and the general guard does not cover. Keeping both rather than deleting a real check to make a sentence in my own PLAN true. Redundancy between two guards costs nothing; a deleted check costs the thing it checked." }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/conditional-dep-install.d/runs/1.md }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/conditional-dep-install.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/conditional-dep-install.d/runs/2.md, brief: "sha256:0387149bc090ba5f" }
advised_by: security-reviewer
---
## CARD
goal: every `importorskip`'d dependency is provided by the job that runs the module, or the gap is recorded as an accepted skip with a reason — and availability is resolved from `pyproject.toml`, never from a hand-written map.
why: `integration-tier-revival` closed the chromadb instance and shipped a guard that only understands chromadb. Its own CI run then revealed a second instance it could not see: `tests/unit/test_backend_roster.py` `importorskip`s openvino, the `quality` job does not install it, and the test it skips — `test_an_openvino_load_failure_is_not_reported_as_a_missing_dependency` — guards a bug that actually shipped ("exactly what happened on openvino 2026.3.1 for every load"). Measured 2026-09-16: 18 modules use `importorskip` across 8 dependencies; exactly one is unprovided.
beat: done · next: add status

## RULES
<must>
- M1 availability is resolved from `pyproject.toml` — the dev group plus whichever extras a job's `uv sync` names — never from a dependency-to-extra map written by hand
- M2 a module whose `importorskip`'d dependency no running job provides fails the check, naming the module, the dependency and the job
- M3 a gap may be accepted only by an explicit, reasoned entry; an accepted gap is a decision on the record, not an absence
- M4 the pull-request gate and the release gate provide the same dependencies, so neither can be the weaker
</must>
<reject>
- R:HANDMAP the guard never resolves a dependency through a list maintained by hand -> "HANDMAP"
- R:GREEN_BY_SKIP a job never reports success over a module that skipped for want of a dependency -> "GREEN_BY_SKIP"
- R:SILENT_ACCEPT a gap is never accepted by deleting the check or by omission -> "SILENT_ACCEPT"
- R:TEXT_MATCH a call is never detected by matching source text -> "TEXT_MATCH"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a dependency resolution has no principals
- A2 [which] covers: S1 · the request does not say WHICH dependencies count; taking "every argument to a real `importorskip` call anywhere under `tests/`" · probe: the set must be discoverable -> found 2026-09-16: 18 modules, 8 dependencies — chromadb, torch, onnxruntime, onnx, onnxruntime.quantization, yowo.backends._onnx, openvino, pycocotools
- A3 [when] covers: S1 · the request does not say when a dependency counts as provided; taking "the dev group, plus the extras that job's `uv sync` names, resolved from pyproject.toml" · probe: a hand-map gets this wrong -> found 2026-09-16: my own first pass mapped pycocotools to the `benchmark` extra and reported the `quality` job at risk for two modules. `pycocotools>=2.0.4` is in the dev group DIRECTLY as well, so both were fine. CI had logged exactly one skip. The hand-map produced two false positives out of three.
- A4 [absent] covers: S1 · the request does not say how a submodule or an internal module resolves; taking "the top-level distribution name" -> `onnxruntime.quantization` resolves through `onnxruntime`, `yowo.backends._onnx` through the package itself
- A5 [order] covers: S1 · n/a · dependencies are independent
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking "someone deciding install-or-accept"; the message names module, dependency, job and both options -> otherwise the cheap fix is to delete the check
- A7 [who] covers: S2 · the request does not say who may accept a gap; taking "a human, in a reviewed commit, with the reason beside the entry"
- A8 [which] covers: S2 · the request does not say which gaps exist today; taking the measured one · probe: exactly one -> found 2026-09-16: openvino in `ci.yml:quality`, and CI's own log confirms a single skip, `tests/unit/test_backend_roster.py:177`
- A9 [when] covers: S2 · the request does not say install-or-accept for that one; taking INSTALL -> the skipped test guards R:FALSEDEP, a defect that shipped on openvino 2026.3.1 and sent users to install a package they already had; `conformance` already proves the extra installs on ubuntu CPU in 2 packages
- A10 [absent] covers: S2 · the request does not say what an empty accepted-gap list means; taking "valid, and the strongest state" -> an empty list must never read as the check being disabled
- A11 [order] covers: S2 · n/a · gaps are independent
- A12 [experience] covers: S2 · the request does not say where an accepted gap lives; taking "beside the check, with its reason, so the diff that accepts one is reviewable"
- A13 [who] covers: S3 · n/a · an install step has no principals
- A14 [which] covers: S3 · the request does not say which gates must match; taking `ci.yml:quality` and `release.yml:quality` -> they already run four identical commands, and `integration-tier-revival` bound their extras for chromadb alone
- A15 [when] covers: S3 · the request does not say when they may diverge; taking "never, for anything `tests/unit/` conditionally imports"
- A16 [absent] covers: S3 · the request does not say what an absent extra means on the release path; taking "the check fails" -> the release gate silently running fewer tests is the weaker-gate failure
- A17 [order] covers: S3 · n/a · install flags are order-independent
- A18 [experience] covers: S3 · the request does not say who notices divergence; taking "a check, not a convention" -> the four quality COMMANDS already needed one to stay identical

## PLAN
contract:
- `tests/unit/test_conditional_dependencies_are_installed.py` generalises and REPLACES the chromadb-specific guard in `test_chromadb_is_installed_where_its_tests_run.py`.
- `importorskip` arguments come from `ast.walk` over real `Call` nodes — a regex over source matched a docstring quoting the call as an example, which is R:TEXT_MATCH and already happened once.
- Provision is computed by reading `pyproject.toml`: `dependency-groups.dev` plus `project.optional-dependencies[<extra>]` for each `--extra` a job's `uv sync` names, normalised to top-level distribution names.
- `ACCEPTED_GAPS` is an explicit mapping with a reason per entry. It ships EMPTY, because the one measured gap is being closed rather than accepted.
- `quality` in both `ci.yml` and `release.yml` gains `--extra openvino`.

## EDGES
- E1 a new `importorskip` added for a dependency no job provides — fails, naming all three of module, dependency, job
- E2 a dependency provided only through the dev group — counts as provided, with no extra named
- E3 a submodule argument (`onnxruntime.quantization`) — resolves through its top-level distribution
- E4 an extra removed from one gate only — fails as divergence
- E5 `ACCEPTED_GAPS` empty — valid, and the strongest state
- E6 an entry in `ACCEPTED_GAPS` that no longer corresponds to a real gap — fails, so the list cannot outlive its reason

## CHECKS
Names DECLARED here first and used verbatim. Every line carries a trailing
`· <why>` — without it a line parses, runs, passes and binds NOTHING (M21).

tests/unit/test_conditional_dependencies_are_installed.py:
- test_every_conditional_dependency_is_provided_by_the_job_that_runs_it · covers: M2,S1,S2,A2,A8,R:GREEN_BY_SKIP · the guard, over every dependency rather than one
- test_provision_is_read_from_pyproject_not_from_a_hand_written_map · covers: M1,A3,R:HANDMAP,S1 · a hand-map gave 2 false positives out of 3 on its first run
- test_a_dependency_from_the_dev_group_counts_as_provided · covers: E2,A3 · pycocotools ships in the dev group AND an extra
- test_a_submodule_resolves_through_its_top_level_distribution · covers: E3,A4 · onnxruntime.quantization is onnxruntime
- test_importorskip_is_detected_by_parsing_not_by_matching_text · covers: R:TEXT_MATCH,A2 · a regex matched a docstring quoting the call
- test_the_failure_names_the_module_the_dependency_and_the_job · covers: A6,E1 · otherwise the cheap fix is to delete the check
- test_an_accepted_gap_requires_a_reason · covers: M3,A7,A12,R:SILENT_ACCEPT · omission is not acceptance
- test_an_accepted_gap_that_is_no_longer_real_fails · covers: E6,A10 · the list must not outlive its reason
- test_the_accepted_gap_list_may_be_empty · covers: E5,A10 · empty is the strongest state, never a disabled check
- test_both_quality_gates_provide_the_same_dependencies · covers: M4,A14,A15,A16,A18,E4,S3,R:DIVERGENCE · neither path may be the weaker

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
