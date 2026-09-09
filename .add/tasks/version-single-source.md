---
type: Task
title: One version source, asserted across all stamps
status: done
depth: quick
sensitivity: architecture
milestone: m1-trust-the-ship
scope:
  - pyproject.toml
  - src/yowo/__init__.py
  - src/yowo/export/_exporter.py
gives:
  - S1 `yowo.__version__` — the version the running package reports about itself
  - S2 the `yowo_version` field an export sidecar stamps onto every produced artifact
  - S3 the release bump path — the file(s) semantic-release rewrites when it cuts a version
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:4b9a4f09c1646dfa", receipt: /tasks/version-single-source.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:LITERAL=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:c367cc6fd91da96a", binding: "sha256:22249aa61fd2594e" }
  - { by: "cli", at: 2026-09-09, act: brief, authority: process, brief: "sha256:3672762457fb976d" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/version-single-source.d/runs/1.md }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/version-single-source.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-09, act: gate, authority: human, outcome: PASS, receipt: /tasks/version-single-source.d/runs/2.md, brief: "sha256:6f015852ef68e6b7" }
advised_by: artifact-integrity-steward
---
## CARD
goal: One place declares the version; every other surface derives it, so drift is impossible rather than merely fixed.
why: they already disagree, and it is shipping. `pyproject.toml` says 2.5.0, distribution metadata says 2.5.0, `yowo --version` says 2.5.0 — and `src/yowo/__init__.py:123` says 2.4.1. `_exporter.py:199` stamps sidecars from `yowo.__version__`, so every model exported from the current tree is labelled with a version that was yanked from PyPI. `[tool.semantic_release] version_toml = ["pyproject.toml:project.version"]` bumps only the TOML, so nothing has ever updated the constant and nothing ever will.
beat: done · next: add status

## RULES
<must>
- G1 `yowo.__version__` is DERIVED from installed distribution metadata, not declared as a literal.
- G2 all four surfaces — `yowo.__version__`, `yowo --version`, distribution metadata and the export
  sidecar — report the same string, asserted by a check that fails the build on drift.
- G3 a release bump touches exactly one declaration, so a future release cannot re-open this gap.
</must>
<reject>
- R:LITERAL No version literal may be declared anywhere but the single source -> "LITERAL"
</reject>

## ASSUMPTIONS
- A1 [who] n/a · a module constant and a metadata read; no actor or authorization surface.
- A2 [which] covers: S1 · the request does not say which source wins; taking: `pyproject.toml`'s
  `project.version`, read at runtime via `importlib.metadata.version("yowo")` — it is what semantic-release
  already bumps and what PyPI actually serves · probe: `__version__` equals metadata for the installed
  package -> picking the constant as the source would mean teaching semantic-release a second file, and
  two writers is the failure we are removing.
- A3 [when] covers: S1 · the request does not say what an editable or source checkout reports; taking:
  the same metadata read, which works for `uv sync`/`pip install -e` because a dist-info exists
  -> a contributor running from a bare clone sees an exception on import.
- A4 [absent] covers: S1 · the request does not say what happens when the package is NOT installed
  (`PackageNotFoundError` — a bare `sys.path` checkout); taking: fall back to `"0.0.0+unknown"`, a string
  that is obviously not a release, rather than a plausible-looking literal
  · probe: the fallback is not a real version number
  -> a stale literal fallback silently re-creates exactly the drift this task removes.
- A5 [order] n/a · a single read; nothing is ordered.
- A6 [experience] covers: S1 · the request does not say who reads `__version__`; taking: a user filing a
  bug report, so import must not become expensive or failure-prone — `importlib.metadata` is stdlib and
  the existing `test_torch_free_import` guard still applies
  -> a heavier import path punishes every user to fix a maintainer problem.
- A7 [who] n/a · the sidecar field is written by the exporter; no authorization surface.
- A8 [which] covers: S2 · the request does not say which artifacts carry the stamp; taking: every export
  format the sidecar covers, since `_exporter.py:199` is a single shared call site
  -> a per-format stamp would drift again, one format at a time.
- A9 [when] covers: S2 · the request does not say whether already-exported sidecars are corrected;
  taking: forward-only — artifacts on disk are not rewritten -> someone trusts a 2.4.1 stamp on a model
  built from 2.5.0 source and cannot reproduce it. Accepted: the stamp is provenance, not a contract.
- A10 [absent] covers: S2 · the request does not say what the sidecar records when the version is
  unknown; taking: the same `"0.0.0+unknown"`, propagated rather than papered over
  -> a sidecar claiming a real version for an artifact built from an uninstalled tree.
- A11 [order] n/a · one field, written once per export.
- A12 [experience] covers: S2 · the request does not say who reads the sidecar; taking: whoever is
  deciding whether a deployed artifact is affected by a CVE or a bug, so the field must be the version
  they can `pip install` -> provenance that cannot be traced back to a release is not provenance.
- A13 [who] n/a · the release bump path is CI configuration; no actor surface.
- A14 [which] covers: S3 · the request does not say which files a bump may touch; taking: exactly one,
  `pyproject.toml:project.version`, which is already the sole entry in `version_toml`
  · probe: no `version_variables` entry is added -> two bump targets is the bug, restated.
- A15 [when] n/a · bump timing is semantic-release's, unchanged by this task.
- A16 [absent] covers: S3 · the request does not say what a check should do when a surface disagrees;
  taking: FAIL the build, never warn -> a warning nobody reads is how 2.4.1 survived two releases.
- A17 [order] n/a · declarative configuration.
- A18 [experience] covers: S3 · the request does not say who maintains this; taking: the check names the
  disagreeing surfaces and both values, so the fix is obvious without reading the test
  -> "version drift detected" sends the reader hunting across four files.

## PLAN
contract: `__version__ = _resolve_version()` in `src/yowo/__init__.py`, reading
  `importlib.metadata.version("yowo")` and falling back to `"0.0.0+unknown"`. No other change to the
  export path — it already reads `yowo.__version__`, which becomes correct by construction.
strategy: replace the literal, then assert agreement across all four surfaces in one check so a future
  drift fails rather than ships.
regression floor: `tests/unit/test_public_api.py` and `test_torch_free_import.py` stay green — `__version__`
  is in `__all__` and import weight is guarded.

## EDGES
- E1 The package not installed at all (`PackageNotFoundError`) must yield the sentinel, not raise —
  importing yowo from a bare checkout must not explode.

## CHECKS
- test_version_is_derived_not_declared · covers: G1, R:LITERAL, A14 · no version literal in __init__.py.
- test_all_surfaces_report_the_same_version · covers: G2, A2 · __version__, metadata and the CLI agree.
- test_export_sidecar_stamps_the_package_version · covers: G2, A8 · the sidecar field equals __version__.
- test_uninstalled_package_yields_the_sentinel · covers: E1, A4 · PackageNotFoundError -> "0.0.0+unknown".
- test_only_pyproject_declares_a_version · covers: G3, A14 · semantic_release has no version_variables.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
