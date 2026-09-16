---
type: Task
title: Report the backend that executed, never the one requested
status: done
depth: standard
sensitivity: architecture
milestone: m4-honest-deployment
scope:
  - src/yowo/benchmark/
  - src/yowo/tune/
  - tests/
gives:
  - S1 a BenchmarkResult whose every column is measured for the row it appears in — the backend that executed, the artifact that loaded, and the request kept separately so a fallback is visible
  - S2 an OBB benchmark row that executes the configuration it is labelled with
  - S3 a tune sweep whose ranking includes the postprocess cost it is ranking, and a persisted profile that says whether it did
depends_on:
  - /tasks/backend-conformance-suite.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-16, act: interview, authority: human, interview: "sha256:20575c72f70e4c69", receipt: /tasks/benchmark-truth.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A6=confirm|A8=confirm|A9=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:REQUEST_AS_RESULT=confirm|R:PLACEHOLDER_AS_MEASUREMENT=confirm|R:SOURCE_FOR_ARTIFACT=confirm|R:RANK_WITHOUT_COST=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:9828c1023eeae641", binding: "sha256:77742a1fc048c889" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:342542cb7af92982" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/benchmark-truth.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:603202eb1f4c2d2a", binding: "sha256:77742a1fc048c889" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:bb0ed48407832d93" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/benchmark-truth.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: plan, outcome: PASS, receipt: /tasks/benchmark-truth.d/runs/2.md, brief: "sha256:bb0ed48407832d93" }
advised_by: inference-parity-engineer
---
## CARD
goal: every number a benchmark or tune row prints is measured for that row, and any number that cannot be measured is absent rather than invented.
why: re-verified by measurement 2026-09-16 (the box's findings are from 2026-09-13 and all four still hold). Requesting the onnx backend logs `Primary backend onnx failed, trying pytorch` and the row still reports `format=onnx`. The onnx row and the pytorch row therefore BOTH executed pytorch, reporting 16.97 and 20.31 fps — a format comparison in which both sides are the same format, offered to a user choosing a deployment format.
beat: done · next: add status

## RULES
<must>
- M1 every column is measured for the row it appears in, or is absent from it
- M2 the executed configuration is reported, and the requested one is kept beside it so a fallback is legible rather than hidden
- M3 the configuration a row is labelled with is the configuration that was executed — no path may label a row with something it never passed to the engine
- M4 a tune ranking includes the cost it is ranking, and a persisted profile records whether its measurement was representative
</must>
<reject>
- R:REQUEST_AS_RESULT a row never reports a requested value as though it were measured -> "REQUEST_AS_RESULT"
- R:PLACEHOLDER_AS_MEASUREMENT a sentinel is never rendered as a measurement — no literal 0.0 MB, no "unknown" device -> "PLACEHOLDER_AS_MEASUREMENT"
- R:SOURCE_FOR_ARTIFACT the source checkpoint's size is never reported for a row that loaded a different artifact -> "SOURCE_FOR_ARTIFACT"
- R:RANK_WITHOUT_COST a winner is never persisted from a measurement that excluded the cost being ranked, without saying so -> "RANK_WITHOUT_COST"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who reads a row; taking "someone choosing a deployment format from the table" -> that reader cannot tell a fallback from a result, and the fallback is the common case here
- A2 [which] covers: S1 · the request does not say WHICH columns are in scope; taking all of them, then measuring which lie · probe: check each against what executed -> found 2026-09-16: `format` reports the request (`backend_type.value`, lines 215 and 264); `model_size_mb` is 5.354 for every format because `_get_model_size_mb` stats the source `.pt` (which is itself 5.354 MB); `device` is honest, read from `engine.selection.device_type` at line 170; the early return at line 223 hardcodes `device="unknown"` although `device` is bound at that point
- A3 [when] covers: S1 · the request does not say what to do when a number is unmeasurable; taking "report nothing rather than something wrong" · probe: is the loaded artifact reachable? -> found 2026-09-16: NO. `engine.selection` carries only backend, device_index, device_type, precision, reason; `PyTorchBackend` holds `_spec` but no artifact path; no backend exposes one. So the loaded artifact's size is not obtainable from inside this node's scope for any non-pytorch backend.
- A4 [absent] covers: S1 · the request does not say what an absent measurement renders as; taking "an explicit dash in the table and null in JSON" -> `0.0 MB` and `unknown` both read as findings; `_get_model_size_mb` returns a literal 0.0 on two separate failure paths
- A5 [order] covers: S1 · n/a · columns are independent
- A6 [experience] covers: S1 · the request does not say how a fallback is surfaced; taking "both the executed and the requested value in the row" -> dropping the request would hide that a fallback happened at all, which is a second way to mislead the same reader
- A7 [who] covers: S2 · n/a · the OBB path has no principals
- A8 [which] covers: S2 · the request does not say which paths mislabel; taking the one measured · found 2026-09-16: `run_single_backend` passes `backend=backend_type` for `classify` (line 150) and for `detect` (line 166) and NOT for `obb` (lines 155-159), while every row is labelled `format=backend_type.value`
- A9 [when] covers: S2 · the request does not say what to do if OBBEngine cannot take a backend; taking "pass it; if it cannot, the row must not claim it" · probe: check the signature before assuming it can
- A10 [absent] covers: S2 · n/a · the parameter is either passed or not
- A11 [order] covers: S2 · n/a
- A12 [experience] covers: S2 · the request does not say how this is prevented from recurring; taking "a check that every engine construction in the benchmark path receives the backend the row is labelled with" -> fixing the one branch leaves the next branch free to repeat it
- A13 [who] covers: S3 · the request does not say who is harmed by a bad ranking; taking "every user of the tuned model, silently" · probe: is the winner actually applied? -> found 2026-09-16: yes. `save_profile` persists it and `src/yowo/engine.py:205-209` loads and applies it to production inference.
- A14 [which] covers: S3 · the request does not say what input tune should rank on; taking "representative input supplied by the caller, with the synthetic frame remaining the default and marked as such" -> the package cannot ship a photograph: `tests/integration/conftest.py` already records that the sample image is third-party under no stated licence, which is why it is fetched rather than committed
- A15 [when] covers: S3 · the request does not say how much the synthetic frame excludes; taking a measurement · found 2026-09-16, yolo11n / pytorch / cpu, median of 15 runs: `np.zeros((640,640,3))` yields **0 boxes at 50.31 ms**, bus.jpg at native resolution yields **5 boxes at 63.63 ms**. The excluded cost is **13.3 ms, about 21% of a real frame**, with inference time itself unchanged (56.44 vs 54.30 ms). **The box's own figures — "0.15 ms of postprocess, excluding 100% of the 176-281 ms it is ranking" — do NOT reproduce here** and are not repeated; they were measured on another machine or another configuration.
- A16 [absent] covers: S3 · the request does not say how an old profile reads; taking "absent means not representative" · probe: does the loader tolerate a new field? -> found 2026-09-16: `load_profile` builds `TuneProfile` from explicit `raw[...]` keys, so a new field read with `raw.get(..., False)` leaves existing files loadable and reads them as the honest default
- A17 [order] covers: S3 · n/a
- A18 [experience] covers: S3 · the request does not say where the caveat is visible; taking "on the persisted profile, so it travels with the decision it justified" -> a warning printed once at sweep time is gone by the time the profile is applied

## PLAN
contract:
- `BenchmarkResult.format` becomes the backend that EXECUTED, captured from `engine.selection.backend.value` inside the `with engine:` block exactly as `device` already is at line 170. A new `requested_format: str` carries what was asked for.
- `model_size_mb` becomes `float | None`: the size of the artifact that loaded when that is knowable (pytorch, where the resolved `.pt` IS what loaded), and `None` otherwise. Never a literal 0.0. `_report.py` renders `None` as a dash and emits JSON null.
- the early return's `device="unknown"` is replaced by the device actually captured.
- the OBB branch passes `backend=backend_type`, and a check binds every engine construction in the benchmark path to receiving it.
- `_measure_config` and `run_sweep` accept `sample_frames`; the synthetic frame stays the default and sets `postprocess_representative=False` on the resulting `TuneProfile`, read back with `raw.get(..., False)`.
- RESIDUAL by construction: reporting the artifact size for onnx/openvino/tensorrt rows needs a path accessor on the backend Protocol, which is `src/yowo/backends/` — a sensitive path, a public contract surface, and `backend-extension-contract`'s subject. This node reports `None` there rather than reaching outside its scope or inventing a number.

## EDGES
- E1 a backend falls back — the row shows the executed backend AND the request, and the two differing is legible
- E2 the artifact size cannot be determined — the column is absent, never 0.0
- E3 `_get_model_size_mb` cannot resolve or stat the weight — `None`, not 0.0
- E4 no latencies were collected — the row still reports the device that was actually selected, not "unknown"
- E5 a tune sweep run without representative frames — the profile records `postprocess_representative: False`
- E6 a profile written before this field existed — loads, and reads as not representative

## CHECKS
Names DECLARED here first and used verbatim. Every line carries a trailing
`· <why>` — without it a line parses, runs, passes and binds NOTHING (M21).

tests/unit/test_benchmark_truth.py:
- test_the_row_reports_the_backend_that_executed_not_the_one_requested · covers: M2,S1,A1,A2,E1,R:REQUEST_AS_RESULT · onnx and pytorch rows both executed pytorch
- test_the_row_keeps_the_requested_backend_beside_the_executed_one · covers: M2,A6,E1 · dropping the request hides that a fallback happened
- test_the_artifact_size_is_absent_rather_than_the_source_checkpoints · covers: M1,S1,A3,R:SOURCE_FOR_ARTIFACT · 5.354 MB was reported for every format
- test_an_unmeasurable_size_is_none_and_never_zero · covers: A4,E2,E3,R:PLACEHOLDER_AS_MEASUREMENT · a literal 0.0 renders as "0.0 MB"
- test_the_report_renders_an_absent_size_as_a_dash_not_a_number · covers: A4,E2 · the table is where the sentinel would be read as a finding
- test_the_json_report_emits_null_for_an_absent_size · covers: A4,E2 · a consumer must not parse 0.0 as measured
- test_a_row_without_latencies_still_reports_the_selected_device · covers: E4,A2,R:PLACEHOLDER_AS_MEASUREMENT · device is bound; "unknown" was needless
- test_the_obb_path_passes_the_backend_it_is_labelled_with · covers: M3,S2,A8,A9 · classify and detect pass it, obb did not
- test_every_engine_construction_in_the_benchmark_path_receives_the_backend · covers: M3,A12,S2 · fixing one branch leaves the next free to repeat it
- test_a_sweep_without_representative_frames_is_marked_unrepresentative · covers: M4,S3,A14,A18,E5,R:RANK_WITHOUT_COST · the winner auto-applies to production
- test_a_sweep_given_representative_frames_is_marked_representative · covers: M4,S3,A14 · the flag must track the input, not be a constant
- test_a_profile_written_before_the_flag_existed_still_loads · covers: E6,A16 · explicit raw[...] keys would raise on a missing one
- test_the_recorded_postprocess_measurement_is_documentation_not_an_assertion · covers: A15 · Q11, and the box's own 176-281 ms figure did not reproduce
- test_the_profile_the_engine_applies_carries_the_flag · covers: A13,S3,M4 · the winner is loaded and applied to production inference, so the caveat must reach that consumer

red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
