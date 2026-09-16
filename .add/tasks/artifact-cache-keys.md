---
type: Task
title: Every cache key names its full invalidation set
status: done
depth: standard
sensitivity: architecture
milestone: m4-honest-deployment
scope:
  - src/yowo/export/
  - src/yowo/cache/
  - src/yowo/backends/_tensorrt.py
  - src/yowo/tune/_profile.py
  - src/yowo/hardware/
  - tests/unit/test_cache_keys.py
gives:
  - S1 the invalidation set — the named inputs a cached artifact depends on, and the key derived from them
  - S2 where each cache writes: the tune profile path, the INT8 calibration cache file, and the TensorRT engine cache
depends_on:
  - /tasks/container-aware-sizing.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:5503a84a6e91b986", receipt: /tasks/artifact-cache-keys.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=correct|A7=confirm|A8=confirm|A9=confirm|A11=correct|A10=confirm|R:SILENT_OMISSION=confirm|R:HANDLIST=confirm|R:COLLIDE=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:19627f90430b9243", receipt: /tasks/artifact-cache-keys.d/interviews/2.md, answers: "A6=confirm|A11=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:19627f90430b9243", receipt: /tasks/artifact-cache-keys.d/interviews/3.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A11=confirm|A10=confirm|R:SILENT_OMISSION=confirm|R:HANDLIST=confirm|R:COLLIDE=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:dcf20cdff5239bfe", binding: "sha256:9308bc1e57b62481" }
  - { by: "builder", at: 2026-09-16, act: replan, authority: process, note: "Build discovered three things the Direction did not foresee. (1) Adding driver_version to the nvidia-smi query shifted compute_cap's column and broke four existing architecture checks, because the parser read fixed offsets and the fixtures were hand-written -- fixed by parsing columns BY NAME from the query actually sent, and pinned by a new check that the fixture row width equals the field count. (2) A forged-collision leg cannot exercise the field escaping at all: sorted rendering emits exactly one prefixed line per field, so an injected newline only ADDS a line and can never displace one. The mutation survived not because the check was weak but because the STRUCTURE already guards collisions; the property escaping alone carries is one-line-per-field, and the check now pins that. (3) Two checks beyond the frozen ten were needed -- the digest function can be correct while the exporter never calls it, and the TensorRT EP seam must be recorded as unexecuted in the node itself." }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/artifact-cache-keys.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:92a2a44d657091b0", binding: "sha256:9308bc1e57b62481" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:484ae47754def8e9" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/artifact-cache-keys.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:20bbbcab1fd84346", binding: "sha256:9308bc1e57b62481" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:e6adfe2e4e71d6f2" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/artifact-cache-keys.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/artifact-cache-keys.d/runs/3.md, brief: "sha256:e6adfe2e4e71d6f2" }
advised_by: artifact-integrity-steward
---
## CARD
goal: Every cache key names the inputs its artifact depends on, in a structure a test can enumerate, so an element cannot be omitted silently.
why: Measured 2026-09-16. Three caches, three keys, and each omits inputs that change what the artifact IS. (a) The tune profile key is `compute_fingerprint(hw)`, whose signature takes ONLY a `HardwareProfile` — so backend and precision cannot be in it by construction. All six backend x precision combinations resolve to one path on this machine: `~/.cache/yowo/profiles/fa3304a0/yolo11n.yaml`. A second sweep silently overwrites the first, and `engine.py` then applies whichever ran last to production inference. (b) The INT8 calibration cache is `engine_path.with_suffix(".calib")` — the calibration IMAGE SET is not in the key, so exporting against a different set reuses the first set's calibration table. (c) The TensorRT engine cache sets `trt_engine_cache_enable` and points `trt_engine_cache_path` at the model's own directory with no `trt_engine_cache_prefix`, so the engine filename is whatever the EP chooses.
  TWO CORRECTIONS TO THE BOX, both measured. It recorded `trt_version` and `gpu_arch` as "0 hits", which is true of those exact strings and misleading: `InstalledLibraries.tensorrt_version` and `Device.arch` both EXIST and are populated. Those two elements are missing from KEYS, not from the codebase. Only `driver_version` and `shape_profile` are genuinely unprobed — `_detect.py:330` queries `index,name,memory.total,memory.free` and `Device` has no driver field at all.
beat: done · next: add status

## RULES
<must>
- M1 The inputs a cached artifact depends on are named in one structure, not spelled out separately at each cache site.
- M2 A test enumerates that structure MECHANICALLY and asserts the key changes when each element changes — so an element added later is perturbed without anyone remembering to add it to a list.
- M3 A tune profile's batch size is applied only when the backend and the precision it was swept with are the ones that will actually run. (Probed 2026-09-16: `TuneProfile` ALREADY stores `backend` and `precision`; `load_profile` compares only `fingerprint`; `engine.py:217` then applies `profile.batch_size` whenever `config.batch_size == 1`, whatever backend was chosen. The data is on disk and unread — so this is a comparison to add, not a path to move.)
- M4 The INT8 calibration cache key includes the calibration image set, so a different set does not reuse the previous table.
- M5 The TensorRT engine cache is keyed on a prefix this package computes, not on whatever the execution provider chooses.
- M6 An element that cannot be probed on this machine is recorded as absent by name, not silently omitted.
- M7 Reading a cache entry written under a different invalidation set is a miss, not a reuse.
</must>
<reject>
- R:SILENT_OMISSION no key omits an input that changes what the artifact is, without the omission being named -> "SILENT_OMISSION"
- R:HANDLIST no check over the invalidation set reads a hand-maintained list of its elements -> "HANDLIST"
- R:COLLIDE two artifacts built from different inputs never resolve to one cache path -> "COLLIDE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the box does not say whose cache this is; taking the reader to be a single machine reusing its own artifacts across runs, not a shared or network cache, because every path here is under `~/.cache/yowo` -> if wrong and the cache is shared between machines, the key needs a host identity too and collisions become cross-machine
- A2 [which] covers: S1 · the box lists eight elements; taking them as the MINIMUM and the structure as open, because the box's own list omits the calibration set's CONTENT (it names the set, not its digest) and two files with the same directory name are not the same calibration · probe: changing an image inside the calibration directory must change the key · found: `cache_file = engine_path.with_suffix(".calib")` (`_exporter.py:508`) depends on the ENGINE path alone — not on the directory name and not on its contents, so even renaming the directory does not invalidate -> if wrong and the directory name is the intended granularity, re-calibrating in place silently reuses
- A3 [which] covers: S2 · the box does not say which caches are in; taking the three that exist — tune profile, INT8 calibration, TensorRT engine — and NOT the feature cache, whose shape clause the box names separately and which `feature-cache-honesty` closed on 2026-09-16 -> if wrong, the feature cache is re-litigated and the box's last sentence is done twice
- A4 [when] covers: S1 · the box does not say when an element is "unprobeable"; taking it that absence is decided at key-construction time on the live machine and recorded in the key as an explicit absent marker, not as an empty string, because an empty string is indistinguishable from a probe that returned nothing · probe: a key built with a missing driver version must differ from one built with a driver version of "" -> if wrong, two genuinely different machines share a key
- A5 [absent] covers: S1 · the box does not say what a missing element means for reuse; taking "absent" to be a VALUE that participates in the key rather than a wildcard that matches anything, so an artifact built where the driver was unknown is not reused where it is known -> if wrong, the conservative reading costs a cache miss on machines that cannot probe
- A6 [absent] covers: S2 · the box does not say what to do with cache entries already on disk · probe: must an existing profile be treated as unreachable? · found: NO — a profile is SELF-DESCRIBING. It stores its own `fingerprint`, and `load_profile` already compares it and returns None with a re-tune warning on a mismatch (`_profile.py:197`). `backend` and `precision` are stored the same way and simply never compared. So taking the NON-DESTRUCTIVE reading: a stricter comparison against data already on disk, no profile orphaned, nobody re-runs a sweep they did not have to -> if wrong and the stored fields cannot be trusted, the comparison passes on a forged file and the path must carry the key instead
- A7 [order] covers: S1 · the box does not say whether element order affects the key; taking the key to be order-INDEPENDENT of how the structure is declared but stable across runs, by rendering fields in a canonical sorted order, so reordering the dataclass does not invalidate every cache on disk -> if wrong and order should matter, every field reorder is a silent cache flush
- A8 [experience] covers: S1, S2 · the box does not say who meets a stale hit; taking the harmed party to be someone whose benchmark or deployment silently used an artifact built for other inputs, so a key MISS must be cheap and silent while a key that cannot be built at all must be loud -> if wrong, every cache miss logs a warning and the logs become noise
- A9 [when] covers: S2 · the box does not say when the key is computed relative to the export; taking it as computed BEFORE the artifact is produced, so the path it writes to is already the keyed one and no rename is needed -> if wrong, an artifact is written then moved and a crash between the two leaves an unkeyed file
- A11 [who] covers: S2 · the box does not say who writes these three caches · probe: does any public API accept a cache path from the caller? · found: YES — both `save_profile(profile, path=None)` and `load_profile(model, hw, path=None)` take one, and both are exported from `yowo.tune`. So taking it that the DEFAULT scheme is ours to choose while the override stays public, because changing a default takes nothing from a caller who names their own path -> if wrong and callers depend on the default layout itself, moving it breaks them with no deprecation
- A10 [order] covers: S2 · the box does not say what happens when two processes build the same key at once; taking last-writer-wins as acceptable because both wrote the same invalidation set and therefore the same artifact, which is exactly what the key asserts -> if wrong and artifacts are not deterministic, the key is claiming more than it can

## PLAN
decided-by-me: Three scope questions were put to the author on 2026-09-16 and timed out after 300s. I took the recommended reading on each, and these are MY calls, not an approval: (1) build the full `InvalidationSet` and wire all three caches, rather than fixing only the measured tune-profile defect; (2) change the `.calib` filename to carry an image digest, accepting that every existing calibration table on disk stops being found and the next INT8 export re-calibrates once per model — correctness over a one-time cost, because a table calibrated on the wrong images degrades INT8 accuracy silently and nothing today would reveal it; (3) ship the TensorRT leg with its key construction tested as a pure function and the EP-level effect recorded as NEVER EXECUTED, so no one later reads this node's green gate as proof the engine cache actually re-keyed. Any of the three is cheap to revisit; (2) is the only one with a user-visible cost.
contract: `src/yowo/cache/_keys.py` publishes a frozen `InvalidationSet` dataclass naming every input a cached artifact depends on — model, backend, precision, runtime versions, GPU name and arch, driver version, TensorRT version, shape profile, calibration digest, effective CPU count, platform — and `InvalidationSet.key()` rendering it canonically and hashing it. Absent elements are an explicit marker, not an empty string. The three cache sites build one and use its key: the tune profile path gains it, the `.calib` filename gains it, and the TensorRT EP gets `trt_engine_cache_prefix` from it. `_detect.py` gains `driver_version` in the nvidia-smi query — asked for separately and dropped on failure, mirroring how `compute_cap` is already handled there.
strategy: The structure first, because M2's mechanical enumeration is what makes every later element free. The three sites second. The driver probe last, since it is the only part that needs hardware this machine does not have — and the KEY function is pure, so its behaviour with and without a driver version is fully testable here. That is what dissolves the "no GPU runner" blocker: nothing needs a GPU to prove a key changes when the GPU changes.

## EDGES
- E1 an element that cannot be probed on this machine — no GPU, so no arch, no driver, no TensorRT version.
- E2 a calibration directory whose NAME is unchanged but whose IMAGES changed.
- E3 a profile already on disk written before this change — must still load, and must now be REJECTED when its stored backend or precision is not the one about to run.
- E4 a field added to `InvalidationSet` later — the perturbation test must cover it without being edited.
- E5 two elements swapped in value (backend "onnx"/precision "fp16" against backend "fp16"/precision "onnx") — the rendering must not let them collide.
- E6 the same inputs on two runs — the key must be identical, or nothing is ever reused.

## CHECKS
- test_every_element_of_the_invalidation_set_changes_the_key · covers: M1, M2, R:HANDLIST, E4 · enumerates `dataclasses.fields(InvalidationSet)` and perturbs each one, so a field added later is covered without anyone editing a list
- test_the_same_inputs_give_the_same_key · covers: E6 · a key that never repeats would satisfy every other check here and make every cache useless
- test_a_profile_swept_on_one_backend_does_not_set_the_batch_size_for_another · covers: M3, R:COLLIDE, E3 · the measured defect: a profile swept on pytorch currently supplies its batch size to a tensorrt run, because only `fingerprint` is compared
- test_the_calibration_cache_key_follows_the_images_not_the_directory_name · covers: M4, A2, E2 · changing an image inside the directory changes the key, so a re-calibration is not served the previous table
- test_the_tensorrt_engine_cache_uses_a_prefix_this_package_computes · covers: M5 · asserts the provider options carry a prefix derived from the invalidation set rather than leaving the filename to the EP
- test_an_unprobeable_element_is_recorded_as_absent_not_as_empty · covers: M6, A4, A5, E1 · a key built with no driver version must differ from one built with a driver version of the empty string
- test_reordering_the_declaration_does_not_change_the_key · covers: A7 · rendering is canonical, so a field reorder is not a silent flush of every cache on disk
- test_two_elements_cannot_swap_values_into_the_same_key · covers: E5 · the rendering separates fields unambiguously
- test_a_profile_whose_stored_precision_differs_is_a_miss · covers: M7, A6 · the precision leg of the same comparison, so neither field is guarded alone
- test_the_driver_version_is_asked_for_and_survives_a_driver_that_cannot_report_it · covers: M6, E1 · the query gains the field and a failure drops it, mirroring how compute_cap is already handled
- test_the_export_names_its_calibration_table_after_the_images · covers: M4, R:SILENT_OMISSION · ADDED during build: the digest function can be right while the export still never calls it, and no other check opens the exporter
- test_the_tensorrt_engine_cache_effect_is_recorded_as_never_executed · covers: M6, A8 · ADDED during build: the EP-level behaviour cannot run anywhere in this project, so the node must SAY so or a future reader takes this file's green as proof that it did
- test_the_caller_can_still_name_its_own_profile_path · covers: A11 · ADDED during build: A11 is only safe because the override is public, and nothing else in this node would notice it being dropped
- test_the_full_query_row_has_a_column_per_field_asked_for · covers: M6, R:HANDLIST · ADDED during build, in tests/unit/test_hardware.py: adding driver_version silently shifted compute_cap's column and broke four architecture checks, because hand-written fixture rows were parsed at fixed offsets

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
