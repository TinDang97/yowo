---
type: Task
title: A cached, integrity-verified model weight CI can reach
status: done
depth: standard
sensitivity: data
milestone: m1-trust-the-ship
scope:
  - tests/
  - .github/workflows/
  - src/yowo/models/
gives:
  - S1 the session fixture a test uses to obtain a real, digest-verified weight
  - S2 the CI cache of the weight store — how a job avoids a per-job network fetch
  - S3 the sample-image fixture, today a per-session network download that skips on failure
depends_on:
  - /tasks/checkpoint-loader.md
  - /tasks/weight-integrity.md
  - /tasks/pr-ci-gate.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:363410a04f73fc5e", receipt: /tasks/ci-weight-fixture.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:LOCALPATH=confirm|R:UNVERIFIEDFIXTURE=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:eb69775c0eaddc4d", binding: "sha256:b85b43f28c97dd59" }
  - { by: "cli", at: 2026-09-09, act: brief, authority: process, brief: "sha256:f60ac746ccda1517" }
  - { by: "builder", at: 2026-09-09, act: replan, authority: process, note: "A2 named yolo11n as 'the smallest pinned weight'. That fact was wrong: yolo26n is 5.29 MB against yolo11n's 5.35 MB. The fixture now resolves yolo26n, which follows A2's stated rationale rather than contradicting it, and additionally matches the model these tests were written for — feeding them YOLO11 weights produced 64-vs-16 channel mismatches. No rule, covers: key or frozen gives: changed." }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/ci-weight-fixture.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-09, act: gate, authority: human, outcome: PASS, receipt: /tasks/ci-weight-fixture.d/runs/1.md, brief: "sha256:d328b6ca107628f1" }
advised_by: artifact-integrity-steward
---
## CARD
goal: A test can obtain a real, digest-verified weight on any machine and in CI, without a per-job download and without the weight entering the repository.
why: the integration tier is not merely unrun — it is unrunnable, and it reports green while being so.
  - `tests/integration/conftest.py:16` — `_LOCAL_WEIGHTS = Path("/Users/tindang/Downloads/Ultralytics
    YOLO26.pt")`, an absolute path on one maintainer's personal machine. For everyone else the fixture
    calls `pytest.skip`, so every weight-dependent test in the tier has always skipped.
  - `tests/integration/test_engine_integration.py:25` — `tmp/weights/yolo26n_statedict.pt`, inside a
    gitignored directory that does not exist in a fresh clone. Also always skips.
  - `tests/integration/conftest.py:52` — `bus.jpg` is fetched from the network every session and
    `pytest.skip`s on failure, so a network blip silently disables the tests rather than failing them.
  A skip is green. 33 tests collect, the weight-dependent ones never execute, and nothing has ever said so.

  `weight-integrity` just made this tractable: `resolve_weights()` now downloads AND verifies against a
  pinned digest, so a fixture no longer has to choose between "trust an arbitrary local file" and
  "fetch something unverified".

not this task: turning the whole tier on in CI belongs to `integration-tier-revival` (m3). This task
  supplies the fixture and the cache; that one decides what runs and when.
beat: done · next: add status

## RULES
<must>
- M1 A weight fixture resolves on ANY machine — no path under a particular user's home directory, no
  gitignored directory assumed to exist.
- M2 The weight is digest-verified before a test uses it, via the same path production uses.
- M3 CI performs at most one weight download across a workflow run; subsequent jobs and runs restore
  from cache keyed on the pinned digest.
- M4 The weight never enters the repository, and never enters an sdist or wheel.
- M5 A fixture that cannot obtain its input FAILS in CI rather than skipping — a skip is green, and a
  green skip is how this tier died.
</must>
<reject>
- R:LOCALPATH No fixture may depend on a path specific to one machine or one contributor -> "LOCALPATH"
- R:UNVERIFIEDFIXTURE No test may consume a weight that was not digest-verified -> "UNVERIFIEDFIXTURE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who runs these; taking: CI and any contributor on a
  fresh clone, which is the population the current fixture excludes
  -> a fixture that works only for its author is the defect being fixed.
- A2 [which] covers: S1 · the request does not say which model; taking: `yolo11n` — the smallest pinned
  weight at 5.4 MB, so a cold CI download is seconds rather than minutes
  · probe: the fixture resolves the smallest registered weight -> pinning the tier to a 109 MB weight
  makes a cache miss punitive and invites someone to disable it.
- A3 [when] covers: S1 · the request does not say when the download happens; taking: session scope, on
  first use, not at collection -> downloading during collection penalises every run that selects no
  integration test.
- A4 [absent] covers: S1 · the request does not say what happens with no network AND no cache; taking:
  FAIL in CI, skip locally, keyed on the `CI` environment variable
  · probe: the CI branch raises rather than skips
  -> a contributor on a plane cannot work; a green CI that tested nothing is worse.
- A5 [order] n/a · one fixture, resolved once per session.
- A6 [experience] covers: S1 · the request does not say what a failure says; taking: it names the model,
  the cache path and the pinned digest, so "why did this fail" is answerable without reading the fixture
  -> "weights not found" sends the reader into conftest.py, which is where the rot was.
- A7 [who] n/a · the CI cache has no actor surface.
- A8 [which] covers: S2 · the request does not say what is cached; taking: the whole weight store
  `~/.cache/yowo/weights`, not one file, so adding a second model needs no cache change
  -> a per-file cache key multiplies as models are added and silently misses.
- A9 [when] covers: S2 · the request does not say what invalidates the cache; taking: the key embeds the
  PINNED DIGEST, so a re-pinned weight misses and re-downloads automatically
  · probe: the cache key contains the digest, not just the model name
  -> a name-keyed cache serves the old bytes forever after a re-pin, defeating weight-integrity.
- A10 [absent] covers: S2 · the request does not say what a cache miss does; taking: download once and
  verify — never proceed unverified because the cache was cold
  -> a cold cache becoming the unverified path is R:UNVERIFIEDFIXTURE by the back door.
- A11 [order] covers: S2 · the request does not say what happens with parallel jobs; taking: each job
  restores independently and a miss downloads independently; the cache is a saving, not a lock
  -> inventing cross-job coordination for a 5 MB file.
- A12 [experience] covers: S2 · the request does not say who reads cache behaviour; taking: whoever is
  debugging a slow CI run, so the key is legible in the workflow rather than computed in a script
  -> an opaque key makes "why did it re-download" unanswerable from the logs.
- A13 [who] n/a · the image fixture has no actor surface.
- A14 [which] covers: S3 · the request does not say whether the sample image is in scope; taking: YES —
  it is a per-session network fetch that skips on failure, the same disease as the weight, and the exit
  box says "without a network fetch per job" -> fixing only the weight leaves the tier still
  network-dependent and still silently skippable.
- A15 [when] covers: S3 · the request does not say where the image comes from; my first reading was
  "commit it to `tests/fixtures/` — a small JPEG is not licence-encumbered the way a weight is".
  CORRECTED BEFORE FREEZE, because that reading was wrong on its own terms: `bus.jpg` is served from
  ultralytics.com and states no licence, and the repository commits no images at all today. Committing
  a third-party asset of unknown licence into an Apache-2.0 repository is the exact failure
  `licensing-provenance` exists to document — reproducing it here to save a cache step would be absurd.
  · found: `tests/fixtures/` holds only `checkpoint_reference.json`; `git ls-files` matches no image;
    ultralytics.com states no licence for the asset.
  Taking instead: the image is treated EXACTLY like the weight — fetched once, cached in CI, never
  committed, and a failure to obtain it FAILS in CI rather than skipping.
  -> committing it would put an unlicensed asset in the tarball we just spent two tasks cleaning up.
- A16 [absent] covers: S3 · the request does not say what if the image is missing; taking: FAIL — a
  committed file that vanished is a repository problem, not a reason to skip
  -> the current skip-on-failure is exactly what hid the tier's death.
- A17 [order] n/a · fixtures are independent.
- A18 [experience] covers: S3 · the request does not say which image; taking: one that produces real
  detections at default confidence, so a test asserting "found something" is meaningful
  -> an image with no objects makes every detection assertion vacuously satisfiable.

## PLAN
contract: a `verified_weight` session fixture calling `resolve_weights()`; a committed sample image
  under `tests/fixtures/`; an `actions/cache` step keyed on the pinned digest over
  `~/.cache/yowo/weights`. The two hardcoded paths are deleted.
strategy: reuse the production resolution path rather than inventing a test-only downloader — that is
  what makes M2 true by construction instead of by a second implementation.
regression floor: `tests/unit/` stays green; `tests/integration/` must COLLECT unchanged (33 tests) —
  this task changes how inputs are obtained, not what is tested.

## EDGES
- E1 No network and no cache, in CI: must fail with a message naming the model and digest, never skip.
- E2 A re-pinned digest must miss the cache and re-download, not serve the old bytes.

## CHECKS
- test_no_fixture_depends_on_a_machine_specific_path · covers: M1, R:LOCALPATH · no absolute home paths.
- test_weight_fixture_uses_the_verified_resolution_path · covers: M2, R:UNVERIFIEDFIXTURE · resolve_weights.
- test_weight_fixture_targets_the_smallest_pinned_model · covers: A2 · yolo11n, not yolo11x.
- test_ci_caches_the_weight_store_keyed_on_the_digest · covers: M3, A8, A9, E2 · key contains the pin.
- test_weight_is_not_committed_or_shipped · covers: M4 · absent from git and from the sdist allowlist.
- test_missing_input_fails_in_ci_and_skips_locally · covers: M5, A4, E1, A16 · the CI branch raises.
- test_sample_image_is_cached_and_fails_loudly · covers: A14, A15, A16 · the image fixture is cached
  like the weight and raises in CI rather than skipping; it is NOT committed, because its licence is
  unstated and this repository ships no third-party assets.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
