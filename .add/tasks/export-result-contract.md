---
type: Task
title: The documented return type is the returned type
status: done
depth: standard
sensitivity: architecture
milestone: m4-honest-deployment
scope:
  - src/yowo/export/
  - src/yowo/__init__.py
  - src/yowo/types.py
  - src/yowo/cli/_main.py
  - src/yowo/cli/README.md
  - src/yowo/README.md
  - tests/unit/test_export_documented_contract.py
gives:
  - S1 the export return contract — what `export_model` returns, and what every document says it returns
  - S2 the documented signature of `export_model` and the flags of `yowo export`
  - S3 the status of `ExportResult`, a public name that nothing returns
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:19e19d0d2508c210", receipt: /tasks/export-result-contract.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|R:DOCDRIFT=confirm|R:BREAKING=confirm|R:HANDLIST=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:af7ef6e50bf3439a", binding: "sha256:a16e39be3c19a785" }
  - { by: "builder", at: 2026-09-16, act: replan, authority: process, note: "Build found four things the Direction did not foresee, three of them defects in MY OWN checks that running them exposed. (1) The parameter check anchored on a trailing comma, so was silently skipped -- it saw only the parameters that happened to be uncommented, and reported 1 drift where there were 5. Closing that gap surfaced against a real , a SIXTH drift the milestone box never named. (2) The flag check scanned a fixed 4000-character window after 'yowo export', which bled into the next command's table and falsely reported as missing; it now bounds on the next markdown heading. (3) The copy-a-block check scanned whole files for , matching the DETECT examples and reporting frame_index, source_id and top1_score as export drift; it now reads only fenced blocks that call export_model. (4) The flag I added EXISTED and was useless: torch.onnx writes progress to STDOUT, so the first implementation emitted 1220 bytes of which only the tail parsed. The existing check asserts the flag exists, which a broken flag satisfies -- exactly the shape of dishonesty this milestone is about. stdout is now redirected to stderr across the export call, verified against the on-disk sidecar (identical record, zero differing fields), and a new check pins it. MUTATION SWEEP: 8 mutations, all killed first pass." }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:d2b177329f9fa4b3", binding: "sha256:a16e39be3c19a785" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:0c32b532112d5754" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/export-result-contract.d/runs/1.md }
  - { by: "builder", at: 2026-09-16, act: replan, authority: process, note: "CORRECTION: the previous replan note on this node was TRUNCATED by shell backtick expansion and records only its first clause -- read this one instead. Build found four things the Direction did not foresee, three of them defects in MY OWN checks that running them exposed. (1) The parameter check anchored on a trailing comma, so a line like 'target_format: str, # onnx' was silently skipped: it saw only the parameters that happened to be uncommented, and reported 1 drift where there were 5. Closing that gap surfaced calibration_data documented as 'Path | None' against a real 'str | None' -- a SIXTH drift the milestone box never named. (2) The CLI flag check scanned a fixed 4000-character window after 'yowo export', which bled into the next command's table and falsely reported --family as missing; it now bounds on the next markdown heading. (3) The copy-a-block check scanned whole files for attribute access, matching the DETECT examples and reporting frame_index, source_id and top1_score as export drift; it now reads only fenced blocks that call export_model. (4) The --json flag EXISTED and was useless: torch.onnx writes its progress to STDOUT, so the first implementation emitted 1220 bytes of which only the tail parsed. The flag check asserts the flag exists, which a broken flag satisfies -- exactly the shape of dishonesty this milestone is about. stdout is now redirected to stderr across the export call, verified against the on-disk sidecar (identical record, zero differing fields), and a new check pins it. MUTATION SWEEP: 8 mutations, all killed on the first pass." }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:c6d49f76241ebe51", binding: "sha256:a16e39be3c19a785" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:079a20d8846a986b" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/export-result-contract.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/export-result-contract.d/runs/2.md, brief: "sha256:079a20d8846a986b" }
advised_by: build-craftsman
---
## CARD
goal: Every document that describes the export surface describes the surface that exists — return type, signature and CLI flags — and a check fails when they diverge again without anyone re-reading a document.
why: Re-verified by measurement 2026-09-16 before authoring, per lesson M29. Every claim in the box holds, with one number corrected and one doc site the box does not name.
  THREE INCOMPATIBLE CONTRACTS, confirmed. (1) `export_model` really returns `ExportMetadata` — now **24** fields, not the 20 the box records; `opset`, `requested_opset`, `task`, `artifact_files` and `total_size_bytes` were added by `onnx-external-data` on 2026-09-16. (2) `yowo.types.ExportResult` is a 7-field frozen dataclass — `model_name, format, precision, output_path, file_size_bytes, export_time_s, created_at` — exported in `yowo.__all__`, unit-tested, and returned by NOTHING in `src/`. (3) `export/README.md:56` documents a THIRD shape under the same name: `ExportResult(file_path, format: str, precision, metadata)`. Three things called the same thing, none of them agreeing.
  SIGNATURE DRIFT, confirmed field by field: README says `target_format: str` (real `ExportFormat`), `output_dir` optional defaulting to `~/.yowo/exports/` (real: required, no default), `dynamic_batch=False` (real `True`), and `-> ExportResult` (real `-> ExportMetadata`).
  A FLAG THAT DOES NOT EXIST: `cli/README.md:83` tabulates `--json` for `yowo export`; `yowo export --help` lists no such flag. Note the pattern IS real elsewhere — `detect` and `detect-obb` both have `--json` — so this reads as a promise never kept rather than a typo.
  A FOURTH DOC SITE THE BOX DOES NOT NAME: `src/yowo/README.md:155` writes `from yowo import export_model, ExportMetadata, ExportResult`, teaching the orphan as part of the export surface.
beat: done · next: add status

## RULES
<must>
- M1 Every document that names the export return type names the type `export_model` actually returns.
- M2 Every parameter a document ascribes to `export_model` matches the real signature in name, type and default.
- M3 Every CLI flag a document lists for `yowo export` exists in that command.
- M4 A public name that nothing returns says so where a reader meets it.
- M5 No public name is removed, renamed, or changed in shape by this node.
- M6 A check fails when a document and the code disagree, WITHOUT anyone re-reading the document.
</must>
<reject>
- R:DOCDRIFT no document describes an export surface the code does not have -> "DOCDRIFT"
- R:BREAKING no public name is removed, renamed or reshaped here -> "BREAKING"
- R:HANDLIST no check over the documented surface reads a hand-maintained copy of it -> "HANDLIST"
</reject>

## ASSUMPTIONS
- A1 [which] covers: S1, S3 · the box calls this a return-type mismatch and says removing `ExportResult` is BLOCKED on m5; taking the fix to be DOCUMENTARY plus a note on the orphan, never a removal, because m4 is additive-only (recorded in the milestone 2026-09-16) and 2.5.0 is published to PyPI so the semver promise is real · probe: is `ExportResult` returned anywhere in `src/`? · found: NO — the only hits are `types.py` (its definition), `__init__.py:92,170` (import and `__all__`), and four DOCUMENTS. Nothing constructs it -> if wrong and something does return it, changing the docs makes a second contract true rather than retiring one
- A2 [who] covers: S1, S2 · the box does not say who is harmed; taking the reader to be someone writing an integration against the README who will get an `AttributeError` at runtime — `ExportMetadata` has no `file_path`, the documented field — rather than a maintainer, because a maintainer reads the code -> if wrong and only maintainers read these files, the whole node is cosmetic
- A3 [who] covers: S3 · the box does not say who imports the orphan; taking it that someone may already `from yowo import ExportResult` on the strength of `src/yowo/README.md:155`, so the name must keep importing and keep its shape -> if wrong and nobody imports it, m5 may simply delete it and the note written here is wasted
- A4 [which] covers: S2 · the box names two documents; taking the surface to be EVERY document in the repository that describes `export_model` or `yowo export`, found by search rather than by the box's list, because the box itself missed `src/yowo/README.md:155` · probe: search all markdown for the export surface · found: FOUR doc sites, not two -> if wrong and only the named two matter, the check is broader than the contract
- A5 [when] covers: S2 · the box does not say when drift is caught; taking it as caught at CI time by a check that PARSES the documents and compares them to the live signature, not at review time by a human reading both -> if wrong and parsing is too brittle, the check becomes the thing that breaks and gets deleted
- A6 [absent] covers: S2 · the box says a documented `--json` flag does not exist; taking "exists" to be satisfiable EITHER by implementing it or by not documenting it, and choosing to implement, because `detect` and `detect-obb` already carry `--json` so the pattern is established and adding it is additive · probe: does `--json` exist on sibling commands? · found: YES, on both -> if wrong and export must stay minimal, the doc line is deleted instead and users lose a promised capability
- A7 [absent] covers: S3 · the box does not say what marks a name as orphaned; taking a docstring note plus a check as sufficient, rather than a `DeprecationWarning`, because this repo HAS no deprecation mechanism — zero `DeprecationWarning` in `src/` — and inventing one here would pre-empt `deprecation-policy` · probe: is there any DeprecationWarning in src/? -> if wrong, the orphan reads as supported until m5 removes it without warning
- A8 [order] covers: S1 · the box does not say which of the three contracts is canonical; taking the one the CODE returns as canonical and the other two as drift, because a document can be corrected and a shipped return type cannot without a break -> if wrong and `ExportResult` was the intended contract, this node documents the accident and entrenches it
- A9 [order] covers: S2 · the box does not say whether a doc may describe parameters in a different ORDER than the signature; taking order as free and names, types and defaults as binding, because a README often groups by topic -> if wrong, the check fails on a harmless reordering and gets loosened until it asserts nothing
- A10 [experience] covers: S1, S2 · the box does not say what a reader should meet; taking the README code block to be something a reader COPIES, so a block that would not run is the defect, not merely an inaccurate table -> if wrong and the blocks are illustrative, correcting them is over-precision
- A11 [when] covers: S3 · the box does not say when the orphan is removed; taking removal as OWNED BY m5 and explicitly out of scope here, with this node's job being to stop it being taught as current -> if wrong and it should go now, m4 becomes a breaking release
- A12 [experience] covers: S3 · the box does not say what a reader of `types.py` should conclude; taking it that the docstring must say plainly that nothing returns it and which type does, because a reader who finds a 7-field `ExportResult` beside a 24-field `ExportMetadata` has no way to tell which is live -> if wrong, the note is noise in a type module
- A13 [absent] covers: S1 · the box does not say what happens if a document names a type that does not exist AT ALL; taking that as the same defect and the same check, not a separate case -> if wrong, a typo'd type name passes a check built only for mismatches
- A14 [which] covers: S3 · the box does not say whether the orphan's SHAPE may change; taking its seven fields as frozen for the same reason its name is — it is public and unit-tested -> if wrong, its fields could be aligned with ExportMetadata now and the m5 removal made smaller

- A15 [when] covers: S1 · the box does not say when the return contract is fixed relative to the other three formats; taking the contract as settled NOW for all formats at once, because `export_model` has one return type regardless of `target_format` and a per-format contract would be a worse design than the drift it replaces · probe: does the return type vary by format? · found: NO, one `-> ExportMetadata` for every branch -> if wrong and a format needs its own result shape, this node entrenches a single type that cannot carry it
- A16 [order] covers: S3 · the box does not say where the orphan's note must sit relative to the class; taking it as belonging in the CLASS DOCSTRING rather than in a module comment or a changelog entry, because that is what a reader sees on `help(ExportResult)` and in an IDE hover, which is where the confusion actually happens -> if wrong and the note belongs in release notes, a reader hovering the type still learns nothing

## PLAN
decided-by-the-author: Three decisions were put to the author on 2026-09-16 and ANSWERED, not timed out. (1) `ExportMetadata` is canonical: what the code returns is the contract, and the other two shapes are drift to be corrected. The alternatives were named and rejected -- treating `ExportResult` as the intended design would make this a real return-type change, which m4's additive-only rule forbids, and designing a fresh result type belongs in m5 alongside the deprecation policy. (2) `--json` is IMPLEMENTED rather than undocumented. Both satisfy the box's literal "every documented CLI flag exists"; implementing wins because `detect` and `detect-obb` already carry `--json`, so it is an established pattern rather than an invention, it is purely additive, and deleting the line would take away a capability the docs have promised. The cost is one more public surface for `public-surface-audit` in m5. (3) The orphan gets a DOCSTRING NOTE ONLY, not a `DeprecationWarning` -- inventing this repo's first deprecation mechanism inside an unrelated node would pre-empt the design `deprecation-policy` exists to make.
contract: `export_model` keeps returning `ExportMetadata` — unchanged. The four documents that describe the export surface are corrected to match it: `src/yowo/export/README.md` (return type, the three signature fields, and the `ExportResult` dataclass block), `src/yowo/cli/README.md` (`--json`), `src/yowo/README.md:155` (the import line). `yowo.types.ExportResult` keeps its name, its place in `__all__` and all seven fields; its docstring gains a note that nothing returns it, that `export_model` returns `ExportMetadata`, and that its future is owned by `deprecation-policy`. `yowo export` gains `--json`, printing the sidecar as JSON to stdout, matching how `detect-obb` already spells it.
strategy: The check first, because it is the only part that keeps this fixed — parse each document for what it claims about `export_model` and compare against `inspect.signature`, so a future edit to either side fails CI without anyone re-reading a README. The documents second. `--json` last, since it is the only behaviour change and the rest must be green without it.

## EDGES
- E1 a README block that is illustrative pseudo-code rather than a claim about the real signature.
- E2 a parameter described in prose rather than in a signature block.
- E3 a flag documented in a markdown table rather than in a code block.
- E4 a document naming a type that does not exist at all, rather than one that exists with a different shape.
- E5 the orphan: `from yowo import ExportResult` must keep working, keep its seven fields, and keep its place in `__all__`.
- E6 a doc file added later that no check knows to read — the drift defect one level up.
- E7 a signature parameter that is documented nowhere; absence from a doc is not drift, only a wrong statement is.

## CHECKS
- test_every_document_names_the_return_type_the_function_returns · covers: M1, A8, R:DOCDRIFT · the box's headline, and the one a reader hits first
- test_every_documented_parameter_matches_the_real_signature · covers: M2, A9, E2, E7 · name, type and default, with order free and silence allowed
- test_every_documented_export_cli_flag_exists · covers: M3, A6, E3 · a flag tabulated in a README but absent from the command is a promise the software does not keep
- test_the_orphan_says_it_is_an_orphan_where_a_reader_meets_it · covers: M4, A12 · a 7-field ExportResult beside a 24-field ExportMetadata gives a reader no way to tell which is live
- test_no_public_name_is_removed_or_reshaped · covers: M5, A3, A11, A14, E5, R:BREAKING · m4 is additive-only and 2.5.0 is on PyPI, so this is the rule the whole node is authored under
- test_the_documented_surface_is_discovered_not_listed · covers: M6, A4, E6, R:HANDLIST · the box itself missed a fourth doc site, which is exactly what a hand-maintained list of doc sites would do again
- test_a_document_naming_a_type_that_does_not_exist_also_fails · covers: A13, E4 · a typo'd type name must fail the same check as a mismatched one
- test_nothing_in_the_package_returns_the_orphan · covers: A1 · ADDED during build: the whole node rests on this probe, and if something DID return it the doc fix would make a second contract true rather than retire one
- test_the_return_type_does_not_vary_by_export_format · covers: A15 · ADDED during build: a per-format result type would be worse than the drift it replaces, and the corrected documents describe a single return
- test_this_node_invents_no_deprecation_mechanism · covers: A7 · ADDED during build: inventing this repo's first DeprecationWarning here would pre-empt the design deprecation-policy exists to make
- test_a_documented_json_flag_emits_parseable_json · covers: M3, A6, R:DOCDRIFT · ADDED during build: the flag check asserts only that `--json` EXISTS, and the first implementation emitted 1220 bytes of which only the tail parsed, because torch.onnx writes progress to stdout
- test_a_readme_block_a_reader_would_copy_actually_runs · covers: A10, A2, E1 · the harmed party copies the block, so a block that raises AttributeError is the defect

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
