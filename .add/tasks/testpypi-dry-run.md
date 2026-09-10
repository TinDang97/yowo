---
type: Task
title: The publish path is proved against TestPyPI from a tag, before it is trusted against PyPI
status: direction
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - .github/workflows/
  - tests/unit/
  - docs/
gives:
  - S1 a tag-triggered workflow that publishes to TestPyPI over OIDC
generated: { by: add/3.5.0, at: 2026-09-10 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:5b963717ca2864bd", receipt: /tasks/testpypi-dry-run.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:TOKEN=confirm|R:REALINDEX=confirm|R:SHAPEONLY=confirm" }
  - { by: "cli", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:c0dd5a9402c40165", binding: "sha256:403f71e509d37f74" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:f736f08b43ec41be" }
---
## CARD
goal: A tag publishes this package to TestPyPI over OIDC, so the publish mechanism is proved somewhere harmless before it is trusted with PyPI.
why: m1 box 2 clause (ii) asks for exactly this and it has never existed. Verified 2026-09-10: `test.pypi.org/pypi/yowo/json` returns 404, the string `testpypi` appears nowhere in this repository outside the box text, and `release.yml:3-5` is `on: push: branches: [main]` with no `tags:` trigger at all — nothing here can be triggered by a tag. This is the clause that would have caught the failure the milestone is currently sitting on: the release workflow has failed on EVERY push since 2026-08-26 at `Mint a scoped App token`, `Publish to PyPI` has never executed once, `v2.5.0` is tagged but absent from PyPI — and all eleven checks in `test_release_contract.py` pass throughout, because they assert the shape of a YAML file describing a job that cannot start. A workflow nobody has ever seen succeed is a hypothesis. This node makes the hypothesis testable without risking the real index.
next: this node ships the mechanism; the BOX also needs a recorded successful run, which needs a TestPyPI pending publisher only the maintainer can create — see `docs/release-setup.md`.

## RULES
<must>
- M1 A tag push publishes the built distributions to TestPyPI, in a workflow that can also be run on demand so the path can be exercised without inventing a tag.
- M2 TestPyPI is reached over OIDC trusted publishing. No token, no `password:`, no secret — the same mechanism the real publish uses, or this proves nothing about it.
- M3 What is uploaded is built from the tagged commit, and the same `uv build` the real release path uses. A dry run of a different build proves the wrong thing.
- M4 The dry run cannot publish to PyPI. Its environment, its permissions and its repository URL are TestPyPI's, and no branch of it can reach the real index.
- M5 It does not gate, block or race the real release. A failing dry run is loud but never stands between a merge and a release — this proves a mechanism, it is not a quality gate.
- M6 Its job names are distinct from every name `ci.yml` and `release.yml` publish, so a required status check can never be satisfied by this workflow instead of the one it names.
- M7 A run that could not publish because the maintainer has not configured the TestPyPI publisher fails with that as the message, not with an OIDC error nobody can read.
- M8 Every action this workflow uses is pinned to an exact version. Box 2 clause (iii) requires that of every release-path tool, and this workflow is on the release path.
</must>
<reject>
- R:TOKEN Any credential on this path — a `password:` input, a `TEST_PYPI_*` secret, `TWINE_*`, or `__token__`. -> "TOKEN"
- R:REALINDEX Any branch of this workflow that can upload to pypi.org. -> "REALINDEX"
- R:SHAPEONLY Closing box 2 clause (ii) on the existence of this file. The clause asks that a dry run PUBLISHES; a workflow that has never run satisfies it exactly as well as the one that has failed since August. -> "SHAPEONLY"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who triggers it; taking a tag push plus `workflow_dispatch`, since the maintainer needs to exercise it on demand while setting up the publisher and should not have to cut a throwaway tag to do so -> if wrong and only tags should trigger it, an extra manual entry point exists that nobody uses.
- A2 [which] covers: S1 · the request does not say which tags; taking `v*`, matching `tag_format = "v{version}"` in `[tool.semantic_release]`, so the tags this repo actually produces are the ones exercised -> if wrong, dry runs fire on tags that are not releases, which is noise rather than risk.
- A3 [when] covers: S1 · the request does not say whether the dry run precedes or follows the real publish; taking independent — it neither gates nor is gated, because `release.yml` tags and publishes inside one job chain and inserting a dependency there would make a TestPyPI outage able to block a real release -> if wrong, a broken publish path can still reach PyPI before anyone has seen the mechanism work.
- A4 [absent] covers: S1 · the request does not say what happens before the maintainer configures the TestPyPI publisher; taking a clear failure naming that as the cause and pointing at `docs/release-setup.md`, never a skip -> if wrong, a green skipped job reads as a working publish path, which is precisely the failure this node exists to expose. · probe: with no publisher configured the run must fail, and its message must name the missing configuration.
- A5 [order] covers: S1 · the request does not say what happens when the same version is dry-run twice; taking `skip-existing: true`, matching the real publish step, so a re-run is a no-op rather than a red job — a red job on a successful release is how five failures went unread here -> if wrong, a genuine duplicate-version mistake is hidden.
- A6 [experience] covers: S1 · the request does not say who reads the result; taking the maintainer setting up trusted publishing for the first time, who needs to know WHICH of the four TestPyPI publisher fields is wrong, since all four must match and the OIDC refusal does not say -> if wrong, they toggle settings blindly against a fifteen-minute feedback loop.

## PLAN
contract: a new `.github/workflows/release-dry-run.yml`, triggered by `push: tags: ['v*']` and `workflow_dispatch`. It checks out the tagged commit, builds with the same `uv build` the release path uses, and uploads to TestPyPI with `pypa/gh-action-pypi-publish` pinned to the same exact version `release.yml` pins, with `repository-url: https://test.pypi.org/legacy/` and `skip-existing: true`, under `permissions: id-token: write` and a `testpypi` environment. `release.yml` is not modified. `docs/release-setup.md` gains the TestPyPI publisher fields beside the PyPI ones.

## EDGES
- E1 A `v*` tag push — builds from that tag and uploads. The clause's own case.
- E2 `workflow_dispatch` on a branch — same path, so the mechanism can be exercised while the publisher is being configured.
- E3 The same version dry-run twice — no-op, green.
- E4 No TestPyPI publisher configured — fails, naming that, pointing at the runbook.
- E5 A tag that is not `v*` — does not trigger.
- E6 Every job name here is absent from `ci.yml` and `release.yml`, and absent from `main`'s required contexts.
- E7 A floating `@v4` on any action here — refused, same as everywhere else on the release path.

## CHECKS
- tests.unit.test_testpypi_dry_run::test_a_version_tag_push_triggers_the_dry_run · covers: M1, E1, A2 · red: no workflow file exists.
- tests.unit.test_testpypi_dry_run::test_it_can_also_be_run_on_demand · covers: M1, E2, A1 · red: no `workflow_dispatch` trigger.
- tests.unit.test_testpypi_dry_run::test_it_publishes_over_oidc_with_no_credential_input · covers: M2, R:TOKEN · red: no upload step to inspect.
- tests.unit.test_testpypi_dry_run::test_no_testpypi_credential_exists_anywhere_in_the_repository · covers: R:TOKEN · red: passes vacuously before the file exists, so it is written to scan every workflow AND assert the dry-run file is among what it scanned.
- tests.unit.test_testpypi_dry_run::test_it_builds_from_the_checkout_with_the_release_paths_build_command · covers: M3 · red: nothing checks out and nothing builds.
- tests.unit.test_testpypi_dry_run::test_every_upload_in_this_workflow_targets_testpypi · covers: M4, R:REALINDEX · red: no upload step.
- tests.unit.test_testpypi_dry_run::test_it_neither_gates_nor_is_gated_by_the_release_path · covers: M5, A3 · red: no workflow.
- tests.unit.test_testpypi_dry_run::test_its_job_names_collide_with_nothing_ci_or_release_publishes · covers: M6, E6 · red: no workflow.
- tests.unit.test_testpypi_dry_run::test_a_publish_failure_names_the_missing_testpypi_publisher · covers: M7, E4, A4 · red: no failure-path step.
- tests.unit.test_testpypi_dry_run::test_a_repeat_dry_run_of_the_same_version_is_a_no_op · covers: A5, E3 · red: no `skip-existing`.
- tests.unit.test_testpypi_dry_run::test_a_non_version_tag_does_not_trigger_it · covers: E5 · red: no tag filter to evaluate.
- tests.unit.test_testpypi_dry_run::test_every_action_is_pinned_to_an_exact_version · covers: M8, E7 · red: no actions to inspect.
- tests.unit.test_testpypi_dry_run::test_the_runbook_names_the_four_testpypi_publisher_fields · covers: A6 · red: `docs/release-setup.md` says nothing about TestPyPI.
red-first: every check MUST fail first.

what these prove, and what they do not: every check above is an assertion about a YAML file and a markdown file. Together they prove the dry-run path is CONFIGURED correctly and cannot reach PyPI. NONE of them proves a dry run publishes — R:SHAPEONLY. Box 2 clause (ii) closes on a run, not on this file, and the run needs a TestPyPI pending publisher only the maintainer can create.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
