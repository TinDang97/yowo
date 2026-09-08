---
type: Task
title: Explicit sdist include list
status: direction
depth: standard
sensitivity: architecture
milestone: m1-trust-the-ship
scope:
  - pyproject.toml
  - .github/workflows/
  - scripts/
  - tests/unit/
gives:
  - S1 [tool.hatch.build.targets.sdist] — the allowlist that defines what a source distribution contains
  - S2 the `sdist` job appended to .github/workflows/ci.yml — the pre-merge assertion on the built artifact
  - S3 scripts/verify_sdist_contents.py — builds both distributions, reports every violation, emits JUnit XML
needs: [/tasks/pr-ci-gate.md#gives]
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: artifact-integrity-steward
---
## CARD
goal: The published source distribution contains what we chose to publish, and nothing the working tree happened to be holding.
why: yowo 2.4.0 and 2.4.1 both shipped a 5.6 MB AGPL-licensed yolo11n.pt to PyPI under an Apache-2.0 declaration, because hatchling has no sdist target configured and falls back to a whole-repo sweep filtered only by .gitignore.
beat: direction · next: run the checks red, then add interview sdist-manifest

## GROUND
- `pyproject.toml:83-88` — build backend is hatchling; `[tool.hatch.build.targets.wheel] packages = ["src/yowo"]`
  constrains the **wheel** only. There is **no `[tool.hatch.build.targets.sdist]` section at all**.
- With no sdist target, hatchling selects files by walking the repo and honouring VCS ignore files.
  `.gitignore` carries no `*.pt` rule (`grep -nE '\.pt|weights' .gitignore` → no match), so an untracked,
  unignored weight is neither tracked nor excluded — it is simply swept in.
- `yolo11n.pt` is untracked and unignored in the working tree right now
  (`git status --porcelain yolo11n.pt` → `?? yolo11n.pt`; `git check-ignore` → no match).
- Published evidence — the sdist has never been curated, the weight only made it obvious:
  - 2.4.0 top level: `.github CLAUDE.md CONTEXT.md bench docs examples uv.lock yolo11n.pt src tests …` (5.71 MB)
  - 2.2.1 top level: the same whole-repo dump plus a stray `chroma/` ChromaDB scratch directory (0.64 MB)
- No `MANIFEST.in` exists (and hatchling would ignore it).
- `.github/workflows/ci.yml:1-20` — the frozen append contract from `pr-ci-gate`: APPEND a job id, never
  rename or restructure an existing one, register the new id in `FROZEN_JOB_IDS` in
  `tests/unit/test_ci_contract.py` in the SAME commit, and add no `needs:` unless this task declares it.
- `.github/workflows/release.yml` contains **no PyPI upload step** — `semantic-release publish` uploads to the
  GitHub Release. Both defective versions were therefore uploaded from a local working tree, which is
  precisely the path that swept the untracked file in.

## RULES
<must>
- M1 The sdist contains only an explicit allowlist: the package source, the test suite, and the project's
  legal, readme, changelog and build metadata.
- M2 The wheel and the sdist agree on the package payload — both ship exactly the contents of `src/yowo`.
- M3 The include list is an allowlist, not an ignore list: a file present in the working tree but absent
  from the allowlist cannot enter the artifact, whatever its tracked or ignored state.
- M4 CI builds the real distributions and asserts their contents before merge, as a job appended under the
  frozen `ci.yml` contract with its id registered in `FROZEN_JOB_IDS` in the same commit.
</must>
<reject>
- R:BINARY No model weight, checkpoint or exported artifact (`*.pt`, `*.pth`, `*.onnx`, `*.engine`,
  `*.mlpackage`, `*.tflite`, `*.bin`) may appear in either distribution -> "BINARY"
- R:SWEEP The build must never fall back to whole-repo file selection: removing or emptying the sdist
  include list must fail a check, not silently widen the artifact -> "SWEEP"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may widen the allowlist later; taking: anyone may edit
  pyproject.toml, and S2 is what makes a widening visible in review rather than at upload time
  -> a silent re-widening lands exactly the way this defect landed, and nobody sees it until PyPI does.
- A2 [which] covers: S1 · the request says "explicit include list" but names no entries; taking:
  `src/yowo`, `tests`, `README.md`, `LICENSE`, `CHANGELOG.md`, `pyproject.toml` in — and `docs/`, `bench/`,
  `examples/`, `.github/`, `CLAUDE.md`, `CONTEXT.md`, `uv.lock`, `chroma/` out
  · probe: a built sdist's top-level entries equal that set exactly
  · found: the entry set is right, but `include` is the WRONG KEY — with
    `[tool.hatch.build.targets.sdist] include = [...]` the sdist still carried `.add/`, `bench/`, `docs/`
    and `.gitignore`, because hatchling's `include` ADDS to the default whole-repo sweep rather than
    replacing it. `only-include` is the key that restricts. It also force-ships `.gitignore` whatever the
    selection says, so `.gitignore` joins `PKG-INFO` as admitted-but-not-required.
    With `only-include`: top level is exactly `src tests README.md LICENSE CHANGELOG.md pyproject.toml`
    plus `PKG-INFO` and `.gitignore`; sdist 5.99 MB -> 468K, wheel 292K; `src/yowo/` nests correctly; the
    E1 and R:BINARY checks flip green. (evidence: `uv build --out-dir /tmp/probe-dist` on a scratch
    pyproject, reverted; 5 failing checks -> 3, the 3 remaining being the CI job and the two assertions
    this finding corrected.)
  -> an sdist missing a file the build backend needs cannot be built from source at all; and, as found,
     freezing `include` would have shipped a contract that reads strict and behaves as a whole-repo sweep.
- A3 [when] covers: S1 · the request does not say whether this applies to already-published releases;
  taking: forward-only — 2.4.0 and 2.4.1 are a separate human decision (yank), not something a build
  config can reach -> the AGPL weight stays reachable on PyPI while we believe this task fixed it.
- A4 [absent] covers: S1 · the request does not say what a listed-but-missing path means; hatchling
  silently skips an include pattern that matches nothing; taking: S3 asserts each required entry is
  PRESENT, so a deleted LICENSE fails the check
  · probe: removing LICENSE from the allowlist's expected set must fail the verifier
  -> an sdist ships with no LICENSE and the Apache-2.0 attribution obligation goes unmet.
- A5 [order] n/a · an include list has no ordering semantics; hatchling applies the patterns as a set.
- A6 [experience] covers: S1 · the request does not say who receives the sdist; taking: downstream
  repackagers (Debian, conda-forge, nixpkgs) who build from source and expect to run the suite, which is
  why `tests` stays IN despite adding weight -> distros that cannot validate a build drop the package.
- A7 [who] covers: S2 · the request does not say whether this job gates merge; taking: it runs on every PR
  but this task does NOT add it to the branch-protection required contexts — that is a human action, and
  `pypi-trusted-publish` is the node that should own promoting it
  -> a red sdist job that nobody enforces, which is a check that asserts nothing.
- A8 [which] covers: S2 · the request does not say which events trigger it; taking: the file's existing
  `pull_request: branches: [main]` trigger with no `paths:` filter
  -> a paths filter would let a pyproject.toml edit skip the very check that guards pyproject.toml.
- A9 [when] covers: S2 · the request does not say when it runs relative to `quality`; taking: no `needs:`,
  running in parallel, per the ci.yml contract's stated default (A14)
  -> a serialized job wastes minutes on PRs whose lint already failed.
- A10 [absent] covers: S2 · the request does not say what happens when the build tooling is unavailable;
  taking: the job installs uv via `astral-sh/setup-uv@v3` exactly as `quality` does, and a build failure
  is a job FAILURE, never a skip -> a green job that built nothing reads as proof and is not.
- A11 [order] n/a · a single appended job with no intra-job ordering the request constrains.
- A12 [experience] covers: S2 · the request does not say who reads a failure; taking: the PR author, so
  the job surfaces the offending path AND the allowlist it violated, not a file count
  -> "sdist check failed" with no named path sends the author to read the workflow to learn what broke.
- A13 [who] n/a · S3 is a script invoked by CI and by a developer; it exposes no authorization surface.
- A14 [which] covers: S3 · the request does not say which distributions it inspects; taking: BOTH sdist
  and wheel, because M2 asserts they agree and a wheel-only regression would otherwise pass
  · probe: the verifier's output names both artifacts -> a wheel regression ships unnoticed.
- A15 [when] covers: S3 · the request does not say whether a binary INSIDE `src/yowo` counts; taking: yes,
  R:BINARY applies to the whole artifact including the package source, because yowo bundles no weights
  today -> a legitimate future bundled asset trips the check and someone weakens it under deadline.
- A16 [absent] covers: S3 · the request does not say what a failed build means; taking: exit non-zero
  carrying the backend's stderr — never "no violations found" over an artifact that does not exist
  · probe: a build that produces no sdist must fail, not pass vacuously
  -> a broken build reports as a clean artifact, which is the worst possible reading.
- A17 [order] covers: S3 · the request does not say whether it stops at the first violation; taking:
  report EVERY violation in one run, sorted by path -> a fix-one-rerun loop across a 15-file sweep.
- A18 [experience] covers: S3 · the request does not say who consumes its output; taking: `add run`, which
  parses JUnit XML, so each assertion is its own `<testcase>` element
  -> one aggregate testcase makes a receipt read 1/1 and binds no individual rule.

## PLAN
contract:
  - S1 `[tool.hatch.build.targets.sdist]` with `only-include = [...]` naming exactly the A2 set.
    NOT `include` — probe-falsified above; `include` adds to the sweep instead of replacing it.
  - S2 a job appended to ci.yml (id `sdist`, name `Source Distribution`), registered in `FROZEN_JOB_IDS`.
  - S3 `scripts/verify_sdist_contents.py` — `build() -> (sdist_path, wheel_path)`, `violations(...) -> list[str]`,
    `--junitxml=PATH`, exit non-zero on any violation or on a build that produced no artifact.
strategy: write the verifier first against the CURRENT config so it fails on the real defect (the untracked
  weight is still in the tree — the red run is the live bug, not a simulation); then add the sdist target
  until it goes green; then append the CI job. Static config assertions live in
  `tests/unit/test_sdist_manifest.py` and stay hermetic; artifact assertions call into S3, which builds once
  per session.
regression floor: `tests/unit/test_ci_contract.py` must stay green — the append must not disturb the five
  assertions `pr-ci-gate` froze.

## EDGES
- E1 A file untracked by git AND absent from .gitignore — exactly the `yolo11n.pt` case — must not enter
  the sdist. This is the live defect and the reason an ignore-list fix would not have been enough.
- E2 The verifier must FAIL on a planted forbidden file, not merely pass when none is present. A checker
  that has never been shown to reject anything is indistinguishable from one that asserts nothing.

## CHECKS
- test_sdist_contains_exactly_the_allowlisted_top_level_entries · covers: M1, A2 · builds the sdist and
  compares its top-level entry set against the frozen allowlist, both directions.
- test_sdist_and_wheel_ship_the_same_package_payload · covers: M2 · the `yowo/` member paths in the wheel
  equal the `src/yowo/` member paths in the sdist.
- test_untracked_unignored_file_does_not_enter_the_sdist · covers: M3, E1 · plants an untracked, unignored
  file at the repo root, builds, asserts it is absent.
- test_no_binary_artifact_in_either_distribution · covers: R:BINARY, A14 · scans both artifacts' member
  lists for the forbidden suffixes.
- test_pyproject_declares_an_explicit_sdist_include_list · covers: R:SWEEP · asserts the sdist target exists
  and its include list is non-empty.
- test_verifier_rejects_a_planted_forbidden_file · covers: E2 · plants a `.pt` inside a synthetic artifact
  and asserts `violations()` names it.
- test_verifier_fails_when_no_artifact_was_produced · covers: A16 · a build directory with no sdist must
  raise, not return an empty violation list.
- test_required_entry_missing_from_the_artifact_fails · covers: A4 · asserts a missing LICENSE is reported.
- test_ci_appends_an_sdist_job_and_registers_its_id · covers: M4 · the job exists in ci.yml, its id is in
  FROZEN_JOB_IDS, it declares no `needs:`, and the five frozen ci_contract assertions still pass.
red-first: 5 of 9 failed for the right reason — the sdist target and the CI job are absent, and the
  build reproduced the live defect verbatim (`assert ['yolo11n.pt'] == []`, plus `.add/` and `CLAUDE.md.bak`,
  so the tree is now dirtier than what 2.4.0 shipped).
  DEVIATION, recorded rather than hidden: the four checks binding E2, A16, A4 and M2 were GREEN at
  direction, because S3 (`scripts/verify_sdist_contents.py`) was authored while grounding the defect
  instead of during Build. Their subject is the verifier's own rejection behavior, so with S3 absent they
  would have failed on ImportError — a wrong-reason red, which direction.md says proves nothing. They are
  retained because the gate still needs those referents bound; they are not evidence of red-first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
