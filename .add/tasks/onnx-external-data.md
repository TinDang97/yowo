---
type: Task
title: A self-contained artifact, and a recorded opset that is the produced opset
status: done
depth: standard
milestone: m4-honest-deployment
scope:
  - src/yowo/export/
  - tests/unit/test_export_sidecar_truth.py
  - tests/integration/test_export_artifact_selfcontained.py
  - .github/workflows/parity.yml
gives:
  - S1 the `.yowo.json` sidecar contract — every field `ExportMetadata` records about an artifact, and what each is read from
  - S2 the file set an ONNX export leaves in `output_dir`, and which of those files the sidecar names
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:491e2862fdcbdd7f", receipt: /tasks/onnx-external-data.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|R:ORPHAN=confirm|R:COLLIDE=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:c823f13450f82593", binding: "sha256:3fe32d0ce13601f2" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:c7d77090157c2883" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/onnx-external-data.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:c85bb5469835342a", receipt: /tasks/onnx-external-data.d/interviews/2.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|R:ORPHAN=confirm|R:REQUEST_AS_FACT=confirm|R:COLLIDE=confirm|R:SILENT_DEGRADE=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:ebe37f1273a44d7e", binding: "sha256:3fe32d0ce13601f2" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:342f5b74169ec1ad" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/onnx-external-data.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/onnx-external-data.d/runs/2.md, brief: "sha256:342f5b74169ec1ad" }
  - { by: loop, at: 2026-09-16, act: reopen, to: direction, reason: "box 1 names opset, input shape AND DYNAMISM read back from the produced artifact. opset and input shape landed in PR #70; dynamic is still dynamic_batch, the request. read_onnx_graph_facts already marks a symbolic dim DYNAMIC_DIM, so the field was simply not used. Caught by reading the box's literal words before ticking it, not by a check." }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:592d6c459594a7aa", binding: "sha256:5cc6b457250c5c3b" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:2dc876a53a8220c9" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/onnx-external-data.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/onnx-external-data.d/runs/3.md, brief: "sha256:2dc876a53a8220c9" }
advised_by: security-reviewer
---
## CARD
goal: An ONNX export leaves exactly the files its sidecar names, and every field describing the graph is read back from the graph rather than copied from the request.
why: Measured 2026-09-16 on yolo11n/fp32. On a `yowo[pytorch]` install — the one `_exporter.py:94`'s own DependencyError names — the export writes a 612,255 B stub plus a 10,616,832 B `.onnx.data`, records `file_size_bytes: 612255`, names only the stub, and the artifact fails to load from a directory holding what the sidecar names. With onnxslim the `.onnx` is self-contained and the 10,616,832 B `.onnx.data` is simply orphaned — 2.00x the recorded size sitting on disk, named by nothing, deleted by nobody. `opset_version=17` is requested and `ai.onnx 18` is produced, with the downconversion raising and the failure swallowed; `ExportMetadata` has no opset field so neither number is recorded. `yolo11n-cls` records `input_shape: [1,3,640,640]` for a `[batch,3,224,224]` graph. `model_stem` has an `obb` branch and no `classify` branch, so a classify export overwrites the detect artifact at `yolo11n.onnx` under the identical `model_name`. And `ExportMetadata.load()` has zero callers in `src/`, so the sidecar is write-only and none of it is caught.
beat: done · next: add status

## RULES
<must>
- M1 Every file an ONNX export leaves in `output_dir` is named by the sidecar, and every file the sidecar names exists.
- M2 The sidecar records the opset of the produced graph, read back from that graph — not the opset the exporter asked for.
- M3 The sidecar's `input_shape` is the produced graph's declared input shape, read back from that graph.
- M4 The size the sidecar records accounts for every file the export produced, not only the entry file.
- M5 An exported artifact loads in a directory containing only the files the sidecar names, on an install with onnxslim and on one without it.
- M6 The sidecar records the task it was exported for, and two exports of the same family and size for different tasks do not write to the same path.
- M7 `export_model` reads its own sidecar back through `ExportMetadata.load()` before returning, and refuses to return a sidecar that does not describe the artifact on disk.
- M8 A sidecar written before these fields existed still loads, and reads as what it was.
- M9 The sidecar's `dynamic` is whether the produced graph actually has a symbolic dimension, read back from that graph -- not whether one was requested.
</must>
<reject>
- R:ORPHAN an export never leaves a file that the sidecar does not name -> "ORPHAN"
- R:REQUEST_AS_FACT no sidecar field describing the produced graph is taken from the request when it can be read from the graph -> "REQUEST_AS_FACT"
- R:COLLIDE two exports differing only in task never write the same artifact path -> "COLLIDE"
- R:SILENT_DEGRADE an export never returns success having swallowed a failure that changed what it produced -> "SILENT_DEGRADE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the box does not say who reads the sidecar; taking it that the reader is a deploying operator on a different machine with only `output_dir` and no yowo installed, which is why "names every file" must mean relative names usable without the exporter -> if wrong and the only reader is yowo itself, the file list is dead weight and a load-time check would have been the whole job
- A2 [who] covers: S2 · the box does not say who installs; taking `yowo[pytorch]` as a REAL user environment for export and not a mis-install, because `_exporter.py:94` names it as the remedy and ONNX export guards on nothing else · probe: an export on an interpreter where `import onnxslim` raises must still produce a loadable artifact -> if wrong and only `yowo[export]` is supported, the fix is a dependency guard instead of internalization, and the guard belongs at line 94
- A3 [which] covers: S1 · the box says "the produced artifact" without saying which formats; taking ONNX as the only format whose graph is read back, because tensorrt/openvino/coreml artifacts are produced by converters this node does not own · probe: the sidecar for a non-ONNX format must record no opset rather than a wrong one -> if wrong, three formats keep recording a request-shaped `input_shape` and the box is half-true by format
- A4 [when] covers: S2 · the box does not say when "self-contained" stops being achievable; taking the 2 GB protobuf ceiling as the boundary — below it internalize, at or above it external data is unavoidable and the sidecar must NAME the companion file instead · probe: the code path that names companion files must be reachable and exercised, not dead -> if wrong and internalization is always possible, one branch is untested dead code; if wrong the other way, every large model silently fails export
- A5 [absent] covers: S1 · the box does not say what a missing `opset`/`task`/file-list means in an OLD sidecar; taking absent as "this export predates the field", loaded as a default rather than an error, because `load()` does `cls(**data)` and the sidecars already written are the ones users have · probe: a sidecar written before this node still loads -> if wrong and absence should be fatal, every artifact exported before today becomes unreadable by its own loader
- A6 [absent] covers: S2 · the box does not say what an UNEXPECTED file in `output_dir` means; taking the sidecar's obligation as covering only what THIS export wrote, not everything present, because `output_dir` is a user directory that may hold anything · probe: an unrelated pre-existing file in `output_dir` must not make the export fail -> if wrong, exporting twice into one directory becomes an error
- A7 [order] covers: S2 · the box does not say what happens when two exports target one directory; taking M6's distinct paths as the fix rather than refusing the second export, because overwriting a DIFFERENT task's artifact is the defect and overwriting the SAME one is a legitimate re-export -> if wrong, a re-export needs a force flag this node does not add
- A8 [experience] covers: S1, S2 · the box does not say who meets a failure; taking the export caller as the audience for a raised error and the operator as the audience for the sidecar, so a swallowed onnxslim failure becomes a warning while a swallowed opset-conversion failure becomes a recorded fact rather than an exception · probe: an export whose opset downconversion fails must still succeed, with the produced opset recorded -> if wrong and an unhonoured opset should be fatal, exports that work today start failing
- A9 [which] covers: S1 · the box's "recorded opset" clause does not say whether the REQUESTED opset is worth recording too; taking both — `opset` as produced and the request kept only if they differ — because a reader debugging a runtime rejection needs to know the ask was not honoured -> if wrong, one extra field is noise
- A10 [when] n/a · S2 · there is no time or scheduling dimension to when an export writes its files: the write is synchronous within `export_model` and no other writer participates
- A11 [which] covers: S2 · the box does not say which of the files in `output_dir` are the export's; taking it that the sidecar names only the files THIS call wrote and, for the ONNX intermediate on a tensorrt/openvino export, that the intermediate `.onnx` IS one of them because it is left on disk and a reader who deletes it cannot re-run the conversion · probe: a tensorrt or openvino export's sidecar must name the `.onnx` it left behind, or the export must remove it -> if wrong and the intermediate is meant to be transient, the sidecar names a file the user is expected to delete
- A12 [order] covers: S1 · the box does not say what orders the recorded fields against the conversion steps; taking the LAST artifact in the chain as the one the sidecar describes — for openvino the converted directory, not the ONNX it came from — and the readback as happening after every conversion, not after the ONNX step · probe: an openvino export's recorded shape and size must come from the openvino artifact -> if wrong and the sidecar should describe the ONNX source, every converted format records the shape of a file that is not `file_path`

## PLAN
contract: `ExportMetadata` gains four fields, all with defaults so `load()` of an existing sidecar is unchanged — `opset: int | None` (read back from the produced graph, `None` for formats whose graph this node does not read), `task: str = "detect"`, `artifact_files: list[str]` (relative names, entry file first), and `total_size_bytes: int` alongside the existing `file_size_bytes` (which keeps meaning the entry file). `_export_onnx` internalizes external tensor data and removes the companion file, mirroring `_export_onnx_kv:353-360` which already does exactly this and is the in-repo precedent; above the 2 GB ceiling it keeps the companion and names it. `input_shape` and `opset` are read from the saved graph after every conversion step, not from `imgsz`. `model_stem` gains a `classify` branch. `export_model` ends by loading its own sidecar through `ExportMetadata.load()` and checking it against `output_dir`, giving that method its first caller in `src/`.
strategy: Read-back first — make the sidecar describe the artifact — because every other clause is then a consequence rather than a separate fix. Internalization second, since M5's no-onnxslim leg is what makes the artifact unloadable rather than merely mis-described. The `classify` suffix last, as it is the one user-visible path change and wants its own commit.

## EDGES
- E1 onnxslim absent — the `yowo[pytorch]` install, simulated by an import hook, must still produce an artifact that loads from a directory holding only the sidecar-named files.
- E2 a classify export — a 224 graph under a 640 request — records 224 and writes a path the detect export does not.
- E3 an obb export — already suffixed today — keeps its path and gains `task: "obb"` without regressing.
- E4 a sidecar written before this node, with no `opset`, `task`, `artifact_files` or `total_size_bytes`, still loads and reads as what it was.
- E5 the opset downconversion fails (it does, today, on both detect and classify) — the export still succeeds and records the opset it actually produced.
- E6 an unrelated file already sitting in `output_dir` is not claimed by the sidecar and does not fail the export.
- E7 `dynamic_batch=True` is requested and the produced graph is STATIC anyway -- the export path that ignores `dynamic_axes` must not leave the sidecar claiming a dynamism the graph does not have.

## CHECKS
- test_the_sidecar_names_every_file_the_export_wrote · covers: M1, R:ORPHAN, E6 · exports and diffs `output_dir` against `artifact_files`, so an orphaned `.onnx.data` fails and a pre-existing unrelated file does not
- test_the_recorded_opset_is_the_graphs_opset · covers: M2, R:REQUEST_AS_FACT, A9 · reads `opset_import` off the produced graph and compares it to the sidecar, so recording the requested 17 for a produced 18 fails
- test_the_recorded_input_shape_is_the_graphs_input_shape · covers: M3, R:REQUEST_AS_FACT, E2 · compares the sidecar against the graph's declared input for a classify export, where request and graph disagree 640 vs 224
- test_the_recorded_size_accounts_for_every_produced_file · covers: M4 · sums the files on disk against `total_size_bytes`, so counting only the entry file fails
- test_an_export_loads_from_only_the_files_the_sidecar_names · covers: M5, E1 · copies the sidecar-named files into an empty directory and loads, with onnxslim importable and with its import blocked
- test_a_classify_export_does_not_overwrite_a_detect_export · covers: M6, R:COLLIDE, E3 · exports detect then classify into one directory and asserts two artifacts, two sidecars, and two recorded tasks
- test_export_reads_its_own_sidecar_back_before_returning · covers: M7 · asserts `ExportMetadata.load()` is called on the written sidecar and that a sidecar disagreeing with the directory raises
- test_a_sidecar_without_the_new_fields_still_loads · covers: M8, E4, A5 · loads a captured pre-change sidecar and asserts the defaults, not an exception
- test_a_swallowed_opset_conversion_failure_is_recorded_not_hidden · covers: R:SILENT_DEGRADE, E5, A8 · asserts the export succeeds and the sidecar shows the produced opset when the 18->17 downconversion raises
- test_an_export_without_onnxslim_produces_a_loadable_artifact · covers: A2, E1 · blocks the `onnxslim` import and asserts the artifact is self-contained
- test_the_large_model_branch_names_its_companion_file · covers: A4 · drives the above-ceiling path and asserts the companion file is named rather than deleted
- test_a_non_onnx_format_records_no_opset_rather_than_a_wrong_one · covers: A3 · asserts `opset is None` for a format whose graph this node does not read
- test_export_sidecar_truth_runs_in_ci · covers: A1 · asserts the new modules are reachable from a CI command, so the no-onnxslim leg is not invisible the way it is today
- test_the_sidecar_does_not_claim_a_file_it_did_not_write · covers: A6, E6 · plants an unrelated file in `output_dir` before exporting and asserts the sidecar neither names nor touches it
- test_the_intermediate_onnx_is_named_not_orphaned · covers: A11 · asserts the file set names every file the export added, not only the entry, which is what makes a tensorrt/openvino run's leftover `.onnx` named rather than orphaned
- test_the_sidecar_describes_the_last_artifact_in_the_chain · covers: A12 · asserts the graph facts are read from the artifact `file_path` points at, not from the ONNX the conversion consumed
- test_the_recorded_dynamism_is_the_graphs_dynamism · covers: M9, R:REQUEST_AS_FACT, E7 · exports with `dynamic_batch` both ways and compares the sidecar against whether the produced graph really carries a symbolic dimension, so copying the request fails when the two disagree

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
