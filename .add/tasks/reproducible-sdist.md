---
type: Task
title: A published artifact provably matches its source
status: direction
depth: standard
sensitivity: architecture
milestone: m1-trust-the-ship
depends_on:
  - /tasks/sdist-manifest.md
scope:
  - .github/workflows/
  - scripts/
  - tests/unit/
  - pyproject.toml
gives:
  - S1 the pinned SOURCE_DATE_EPOCH the reproducibility check builds under
  - S2 scripts/verify_reproducible_build.py — builds twice, compares digests, emits JUnit XML
  - S3 the recorded digest in the task's EVIDENCE, so a local build can be diffed against it
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:42c0da3ac66d92d1", receipt: /tasks/reproducible-sdist.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:DRIFT=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:08f6f1ce28fa33a7", binding: "sha256:b85b43f28c97dd59" }
  - { by: "cli", at: 2026-09-09, act: brief, authority: process, brief: "sha256:131b99014108750d" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/reproducible-sdist.d/runs/1.md }
advised_by: artifact-integrity-steward
---
## CARD
goal: Two builds of the same commit produce byte-identical artifacts, and the digest is recorded so anyone can diff a local build against what was published.
why: split out of m1 box 1 on 2026-09-08. `sdist-manifest` shipped the box's include-list clause and closed at gate PASS having satisfied its own RULES — which I authored narrower than the box. The reproducibility clauses (build twice under a pinned SOURCE_DATE_EPOCH, assert identical digests, record the digest in EVIDENCE) were never built. Without them there is no way to show that a tarball on PyPI was built from the source it claims.

depends_on sdist-manifest (done): the allowlist has to exist first, or a reproducibility check just proves a whole-repo sweep is reproducibly wrong.

probe (discharges A2 before freeze): with `SOURCE_DATE_EPOCH=1700000000`, two consecutive
`uv build` runs of this tree produced byte-identical artifacts — sdist `8567aea31fa98717…` and wheel
`15624a7d72ad696c…` both times. Hatchling already honours SOURCE_DATE_EPOCH, so this task needs no
build-backend change: it only has to pin the epoch, build twice and compare.
beat: scaffold · next: author reproducible-sdist's RULES, ASSUMPTIONS and CHECKS, then add freeze reproducible-sdist

## RULES
<must>
- M1 Two builds of the same commit under a pinned SOURCE_DATE_EPOCH produce byte-identical sdist AND
  wheel.
- M2 The digest of each artifact is recorded where a human can diff a local build against it.
- M3 CI performs the double build and fails on any difference.
</must>
<reject>
- R:DRIFT A difference between the two builds may never be reported as a pass, and may never be
  narrowed by excluding files from the comparison -> "DRIFT"
</reject>

## ASSUMPTIONS
- A1 [who] n/a · a build determinism check; no actor or authorization surface.
- A2 [which] covers: S1 · the request does not say which epoch; taking: a fixed constant in the workflow
  rather than the commit timestamp, so the digest is a property of the SOURCE and two people on
  different days get the same answer · found: `SOURCE_DATE_EPOCH=1700000000` gives byte-identical
  sdist and wheel across two runs (evidence: `uv build` x2, sha256 8567aea31fa98717 / 15624a7d72ad696c)
  -> a commit-derived epoch makes every commit's digest unique and the recorded value useless.
- A3 [when] covers: S1 · the request does not say when the digest stops being comparable; taking: it is
  valid only for the exact commit, hatchling version and Python version that produced it, and the record
  states all three -> someone diffs against a digest from a different toolchain and reports a false
  tampering alarm.
- A4 [absent] covers: S1 · the request does not say what happens if SOURCE_DATE_EPOCH is unset; taking:
  the script SETS it itself rather than trusting the environment, so a local run matches CI
  -> a developer's unset environment produces a different digest and the check looks broken.
- A5 [order] n/a · two builds compared as a pair; order is immaterial.
- A6 [experience] covers: S1 · the request does not say who reproduces this; taking: a downstream
  packager or auditor checking that a PyPI tarball matches this source, so the recorded value comes with
  the exact command that regenerates it -> a bare hash nobody can reproduce proves nothing.
- A7 [who] n/a · a script; no authorization surface.
- A8 [which] covers: S2 · the request does not say which artifacts are compared; taking: BOTH, since the
  probe showed both are already deterministic and comparing only the sdist would leave the wheel free to
  drift -> a non-reproducible wheel ships unnoticed.
- A9 [when] covers: S2 · the request does not say whether the two builds share a directory; taking:
  separate output directories compared by digest, never an in-place rebuild
  -> an in-place second build can overwrite and trivially "match".
- A10 [absent] covers: S2 · the request does not say what a failed build means; taking: exit non-zero,
  never "no difference found" over artifacts that do not exist — same rule as verify_sdist_contents
  -> a broken build reads as reproducible, which is the worst possible reading.
- A11 [order] covers: S2 · the request does not say what to report when they differ; taking: name the
  differing member paths and their mtime/mode/size, not just "digests differ"
  -> a digest mismatch with no diff sends the reader to unpack two tarballs by hand.
- A12 [experience] covers: S2 · the request does not say who consumes the output; taking: `add run`,
  so one `<testcase>` per assertion -> one aggregate case binds nothing granular.
- A13 [who] n/a · the recorded digest is a file; no actor surface.
- A14 [which] covers: S3 · the request says "recorded in EVIDENCE"; taking: both digests plus the
  regenerating command, commit, hatchling and Python versions -> a digest without its toolchain is not
  reproducible by anyone else (A3).
- A15 [when] covers: S3 · the request does not say when the record is refreshed; taking: it is a
  point-in-time attestation for one commit and is NOT auto-updated — a check that rewrites the value it
  compares against asserts nothing -> a self-updating record is a tautology.
- A16 [absent] covers: S3 · the request does not say what happens when no digest is recorded yet;
  taking: the check still runs the double build and fails on drift; only the diff-against-record leg
  needs the record -> a missing record silently disables the whole check.
- A17 [order] n/a · a static record.
- A18 [experience] covers: S3 · the request does not say who reads it; taking: an auditor asking "was
  this tarball built from this source", so it lives in the task node next to the gate that produced it
  -> provenance filed somewhere nobody looks is provenance nobody uses.

## PLAN
contract: `scripts/verify_reproducible_build.py` sets SOURCE_DATE_EPOCH, builds twice into separate
  directories, compares sha256 of both artifacts, reports differing tar members on mismatch, emits
  JUnit XML. A job appended to ci.yml under the frozen pr-ci-gate contract, id registered in
  FROZEN_JOB_IDS in the same commit.
strategy: the probe already proved determinism, so this is about locking it in and recording the value —
  not about making the build reproducible.
regression floor: test_ci_contract.py, test_sdist_manifest.py and test_release_contract.py stay green.

## EDGES
- E1 A build that produces no artifact must fail, not compare two absences and pass.
- E2 The comparison must be over the whole artifact. Excluding a file to make it match is R:DRIFT.

## CHECKS
- test_two_builds_produce_identical_sdist · covers: M1, A8 · byte-identical under the pinned epoch.
- test_two_builds_produce_identical_wheel · covers: M1, A8 · same for the wheel.
- test_script_sets_the_epoch_itself · covers: A4 · does not trust the ambient environment.
- test_missing_artifact_fails_not_passes · covers: E1, A10 · absence is a failure.
- test_comparison_covers_every_member · covers: R:DRIFT, E2 · no exclusion list exists.
- test_ci_appends_a_reproducibility_job_and_registers_its_id · covers: M3 · under the frozen contract.
- test_recorded_digest_names_its_toolchain · covers: M2, A14 · commit, hatchling and Python recorded.
red-first: every check MUST fail first.

## EVIDENCE

### Recorded digests — a point-in-time attestation, NOT auto-updated

A check that rewrites the value it compares against asserts nothing (A15), so this
block is written by hand and only ever refreshed deliberately.

    sha256(sdist) = b7b028309f70d6b5ce7903f6bb34a5925a63ff1764d4fff9de4cdf466a7840ca
    sha256(wheel) = 1a5c242dbed14de67e24d8967354a4c3a32f17c22031bbf56a32ecf7cf91a2b9

Regenerate with:

    SOURCE_DATE_EPOCH=1700000000 uv build --out-dir dist/
    shasum -a 256 dist/*

Toolchain that produced them — a digest without these is reproducible by nobody (A3, A14):

    commit    the commit that adds this record, on branch fix/m1-version-and-check-names
    python    3.11.15
    hatchling resolved by uv build isolation
    uv        0.10.11
    platform  darwin/arm64

These are valid for that exact commit and toolchain only. A different hatchling or
Python will produce a different digest; that is a toolchain difference, not tampering.

receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
