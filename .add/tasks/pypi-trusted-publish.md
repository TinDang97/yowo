---
type: Task
title: OIDC trusted publishing, every release tool pinned
status: done
depth: standard
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - .github/workflows/
  - tests/unit/
gives:
  - S1 the `pypi` publish step in release.yml — OIDC upload of the built dist/ to PyPI, no credential
  - S2 the release-push identity — a scoped GitHub App token minted in-workflow, used for checkout and for semantic-release's push to protected main
  - S3 the pinned release toolchain — python-semantic-release and every action in the release path at an exact version
depends_on:
  - /tasks/sdist-manifest.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-08, act: interview, authority: human, interview: "sha256:7fefa6b6c5ea7025", receipt: /tasks/pypi-trusted-publish.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:SECRET=confirm|R:UNGATED=confirm" }
  - { by: "Tin Dang", at: 2026-09-08, act: freeze, authority: human, direction: "sha256:be89099f775501a7", binding: "sha256:b85b43f28c97dd59" }
  - { by: "cli", at: 2026-09-08, act: brief, authority: process, brief: "sha256:4a776d4ccfd140dd" }
  - { by: "builder", at: 2026-09-08, act: replan, authority: process, note: "Build discovered that python-semantic-release sets a `released` step output only when used as a GitHub Action; invoked as a plain `run:` command it sets nothing. The publish job is gated on that output, so PyPI publishing would have silently never run while every job stayed green — the exact limit the CHECKS honesty note predicted. Wrote the output by hand from a populated dist/, and ADDED a 10th check (test_release_job_writes_the_released_output). Adding a check strengthens the contract; no existing check was changed and no frozen `gives:` moved." }
  - { by: "process:run", at: 2026-09-08, act: run, authority: process, outcome: PASS, receipt: /tasks/pypi-trusted-publish.d/runs/1.md }
  - { by: "builder", at: 2026-09-08, act: replan, authority: process, note: "Verify's security residue lens found workflow-level `permissions: contents: write` being inherited by the `quality` and `sdist` jobs, which never write. Pre-existing, not introduced by this task, and not exploitable on its own — jobs run trusted code from main — so a hardening opportunity rather than a HARD-STOP finding. Scoped permissions per job (workflow default read-only; `release` gets contents:write, `publish` gets id-token:write) and added an 11th check so it cannot silently regress." }
  - { by: "process:run", at: 2026-09-08, act: run, authority: process, outcome: PASS, receipt: /tasks/pypi-trusted-publish.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-08, act: gate, authority: human, outcome: PASS, receipt: /tasks/pypi-trusted-publish.d/runs/2.md, brief: "sha256:5a1a56a438cac27f" }
advised_by: release-planner
---
## CARD
goal: yowo reaches PyPI from a pinned CI workflow authenticated by OIDC, with no long-lived credential anywhere.
why: there is no PyPI upload step in CI at all — `release.yml:69` runs `semantic-release publish`, which uploads to the GitHub Release, so both 2.4.0 and 2.4.1 were pushed from a local working tree. That is the mechanism that swept an untracked AGPL weight into two published sdists.

human decision (recorded, not derived) — the publishing path is **OIDC trusted publishing, no stored secret**.
  Asked with three options; the human answered "trusted publishing".
  Rejected: storing a freshly-minted API token — works, but keeps a long-lived credential and would have
    required reopening m1-trust-the-ship's exit leg (iv), "PyPI shows zero active project API tokens".
  Rejected: continuing to publish manually — leaves intact the local-working-tree upload path that caused
    the defect twice.
  Consequence: `release.yml:9` already declares `id-token: write` and nothing consumes it, so the workflow
  half is small. The PyPI half is a human action that cannot be automated from here — see PREREQUISITE.

PREREQUISITE (human, on pypi.org — blocks this task's Build, not its Direction):
  1. Revoke the API token that was exposed in plaintext: https://pypi.org/manage/account/token/
  2. Add a trusted publisher at https://pypi.org/manage/project/yowo/settings/publishing/ with exactly:
       PyPI Project Name  yowo
       Owner              TinDang97
       Repository name    yowo
       Workflow name      release.yml
       Environment name   release
     (`release` already exists as a GitHub environment on this repo, and `release.yml`'s publish job
     already declares `environment: release`.)

human decision 2 (recorded, not derived) — the release commit and tag reach protected `main` via a
  **GitHub App token added to the branch-protection bypass list**.
  Asked with four options; the human answered "GitHub App token with bypass".
  Rejected: release-opens-a-PR — nothing bypasses protection, every release reviewable, but each release
    then needs a human merge and a second Quality Gate run.
  Rejected: tag-only releases — nothing pushes to main at all, but pyproject.toml's `project.version`
    stops being the source of truth, which collides with the `version-single-source` task.
  Rejected: a PAT with bypass rights — simplest, but re-introduces a long-lived credential tied to a
    personal account, contradicting the criterion this task exists to satisfy.

  CAVEAT the human should see before Build (flagged, not silently absorbed): a GitHub App is not
  credential-free. `actions/create-github-app-token` needs the App ID and the App's PRIVATE KEY stored as
  repository secrets, and that private key is long-lived. What the App buys is that the credential is
  scoped (contents:write on this repo only), owned by an app rather than a person, independently
  revocable, and never valid for PyPI. The PyPI half stays genuinely secretless via OIDC. m1's exit
  leg (iv) is "PyPI shows zero active project API tokens", which the App key does not breach — it is a
  GitHub credential and can never authenticate to PyPI.
  CORRECTION: an earlier version of this line quoted m1 as requiring "no long-lived token exists
  anywhere". No such wording exists in m1-trust-the-ship; it was my paraphrase, repeated until it read
  as a citation. The milestone has always been PyPI-scoped. Left visible rather than quietly deleted,
  because a fabricated citation that shaped two human decisions is worth a record.

why the release workflow has failed five times running (evidence, replacing an earlier wrong guess of
  "unpinned python-semantic-release"): `semantic-release version` commits the bump and tag and pushes
  them straight to `main`. Runs on 3a9b78b, 13e0b69 and 3340bb3 failed 403 before this session;
  31b7e8b failed `GH006: Protected branch update failed` after `pr-ci-gate` enabled branch protection
  with `enforce_admins: true`. `release.yml:8` already declares `contents: write`, so this was never a
  missing workflow permission.

scope widened during Direction (the node is unfrozen, so this is authoring, not a seal break): this task
  now covers three surfaces, not two — OIDC publishing to PyPI, the release-push identity above, and
  pinning the release tooling. They are one deliverable: a release path that works and holds no
  credential that can reach PyPI. Splitting them would leave neither half able to cut a release.

ordering: `depends_on: sdist-manifest` is deliberate and must hold — automating publication of an artifact
that is still a whole-repo sweep would only make the defect faster.
beat: done · next: add status

## RULES
<must>
- M1 What reaches PyPI is the artifact the gates passed: built once by `semantic-release version`, then
  uploaded from `dist/` without a rebuild.
- M2 PyPI authentication is OIDC only. The workflow presents no PyPI credential because none exists.
- M3 The release commit and tag reach protected `main` under a scoped GitHub App identity — never a
  personal credential, and never by weakening branch protection for everyone.
- M4 Every tool in the release path is pinned to an exact version, so a release is reproducible and a new
  upstream release cannot silently change what gets published.
</must>
<reject>
- R:SECRET No PyPI token, password or `__token__` may appear in any workflow, repository secret,
  environment secret or env var -> "SECRET"
- R:UNGATED Nothing publishes to PyPI unless BOTH the Quality Gate and the sdist verification passed on
  that same commit -> "UNGATED"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say which identity PyPI trusts; taking: the trusted publisher
  is bound to `TinDang97/yowo` + `release.yml` + environment `release`, so a fork or another workflow in
  this repo cannot mint a publishable token -> a workflow added later publishes to yowo without review.
- A2 [which] covers: S1 · the request does not say which files upload; taking: `dist/*` as
  `dist_glob_patterns` already names — the sdist and the wheel, both now allowlist-constrained
  -> a stray file in dist/ from an earlier step gets published and cannot be unpublished, only yanked.
- A3 [when] covers: S1 · the request does not say when publication happens relative to the GitHub Release;
  taking: PyPI upload runs AFTER `semantic-release version` (which builds and tags) and after
  `publish` (which uploads to the GitHub Release), so a PyPI failure leaves a valid tag to retry from
  -> publishing to PyPI first would strand an un-yankable release with no matching git tag.
- A4 [absent] covers: S1 · the request does not say what a re-run on an already-published version does;
  taking: `skip-existing: true`, so a re-run is a no-op rather than a hard failure
  -> a red release job that everyone learns to ignore, which is how the last five failures went unread.
- A5 [order] n/a · a single upload step; ordering relative to the other jobs is A3.
- A6 [experience] covers: S1 · the request does not say who reads a failed publish; taking: whoever
  pushed to main, so the step name says PyPI explicitly and the job does not bury it inside
  "Semantic Release" -> a failure indistinguishable from the tagging failures that preceded it.
- A7 [who] covers: S2 · the request does not say what the App may do; taking: `contents: write` on this
  repository only, installed on `TinDang97/yowo`, added to the branch-protection bypass list — and
  explicitly NOT granted any PyPI-reaching permission -> an over-scoped App becomes the credential the
  OIDC work exists to avoid.
- A8 [which] covers: S2 · the request does not say which operations use the App token; taking: BOTH the
  `actions/checkout` token and semantic-release's `GH_TOKEN`, because a checkout under `GITHUB_TOKEN`
  leaves a remote the App token cannot push to -> the push fails GH006 exactly as it does today.
- A9 [when] covers: S2 · the request does not say when the token is minted; taking: as the first step of
  the release job, so expiry (App tokens last one hour) cannot elapse mid-release
  -> a long quality run burns the token and the release fails after tagging.
- A10 [absent] covers: S2 · the request does not say what happens when the App token cannot be minted —
  App uninstalled, key rotated, secret missing; taking: the job FAILS at that step and never falls back
  to `GITHUB_TOKEN` -> a silent fallback re-creates today's GH006 failure while looking like a new bug.
- A11 [order] covers: S2 · the request does not say what happens if two pushes to main race; taking: the
  existing `concurrency: release` group already serializes them and is kept
  -> two releases interleave and one overwrites the other's tag.
- A12 [experience] covers: S2 · the request does not say whose name appears on the release commit;
  taking: the App's bot identity, which makes automated commits distinguishable from human ones in
  `git log` -> release commits appear to come from a person who did not write them.
- A13 [who] n/a · S3 is a set of version pins; it exposes no actor or authorization surface.
- A14 [which] covers: S3 · the request says "every release tool" without listing them; taking:
  `python-semantic-release`, `actions/checkout`, `actions/setup-python`, `astral-sh/setup-uv`,
  `pypa/gh-action-pypi-publish` and the App-token action — every action in release.yml, not only the
  Python one · probe: no action in release.yml resolves to a floating major tag
  -> one unpinned action is the whole supply chain, since each runs with the release job's permissions.
- A15 [when] covers: S3 · the request does not say how tight a pin; taking: an exact version for
  `python-semantic-release` and an exact tag for each action, NOT a commit SHA — SHA pinning is stricter
  but this repo has no automation to refresh them, and a stale SHA nobody updates is its own risk
  -> a compromised upstream tag republishes under the same name; accepted knowingly, revisit if a
  supply-chain scanner lands.
- A16 [absent] covers: S3 · the request does not say what an unpinned tool should do; taking: a check
  FAILS the build rather than warning -> pins rot silently and the guarantee decays to nothing.
- A17 [order] n/a · pins are declarative; no ordering is constrained.
- A18 [experience] covers: S3 · the request does not say who maintains the pins; taking: whoever bumps
  one reads a named check telling them the pin is asserted, so a bump is deliberate
  -> a pin gets loosened to make a red build green, which is how the guarantee dies.

## PLAN
contract:
  - S1 a `Publish to PyPI` step using `pypa/gh-action-pypi-publish` with `skip-existing: true`, no
    `password:` input, running in environment `release` under the existing `id-token: write`.
  - S2 an `actions/create-github-app-token` step minted first, feeding `actions/checkout`'s `token:` and
    semantic-release's `GH_TOKEN`. `secrets.GITHUB_TOKEN` disappears from the release job entirely.
  - S3 exact pins on python-semantic-release and every action in release.yml.
  - a `sdist` verification job in release.yml so R:UNGATED holds on the push-to-main path too — ci.yml's
    job only runs on pull_request. It does NOT touch the `quality` job, whose steps `pr-ci-gate` froze
    to mirror ci.yml.
strategy: the checks are static assertions over release.yml, because a real release cannot be rehearsed
  in a unit test. That bounds what they can prove — see the honesty note in CHECKS.
regression floor: `tests/unit/test_ci_contract.py` stays green; release.yml's `quality` job stays
  byte-identical in its four commands to ci.yml's.

## EDGES
- E1 A workflow re-run on an already-published version must be a no-op, not a failure — PyPI rejects
  duplicate filenames, and a job that goes red on a successful release trains everyone to ignore it.
- E2 When the App token cannot be minted, the release fails at that step and never falls back to
  `GITHUB_TOKEN`.

## CHECKS
- test_publish_uses_oidc_with_no_password_input · covers: M2 · the publish step has no `password:`.
- test_no_pypi_credential_in_any_workflow · covers: R:SECRET · greps every workflow file for PyPI
  token/password patterns. NARROWED DELIBERATELY: a hermetic unit test cannot enumerate live repository
  secrets, and making it shell out to `gh` would put the network in the unit suite. The "no PyPI secret
  is configured" half is a human attestation, recorded below and re-checkable in one command.
- test_publish_runs_only_after_quality_and_sdist · covers: R:UNGATED · the publishing job's `needs:`
  includes both gate jobs.
- test_release_push_uses_app_token_not_github_token · covers: M3, A8 · `secrets.GITHUB_TOKEN` appears
  nowhere in the release job; checkout and GH_TOKEN both read the minted App token.
- test_app_token_is_minted_before_checkout · covers: A9 · step order.
- test_no_fallback_to_github_token_on_mint_failure · covers: E2 · no `||` fallback, no `continue-on-error`.
- test_every_action_and_tool_in_release_is_pinned · covers: M4, A14, A16 · no floating major tags.
- test_duplicate_upload_is_skipped_not_failed · covers: E1, A4 · `skip-existing: true`.
- test_pypi_upload_does_not_rebuild_the_artifact · covers: M1 · no build command between build and upload.
red-first: 8 of 9 fail because the publish job, the App-token step and the pins do not exist.
  The 9th, `test_no_pypi_credential_in_any_workflow`, PASSES already and is meant to: R:SECRET is a
  guard against introducing a credential, not a behavior to build. A Reject of the form "never do X"
  is green from the start; its value is that it turns red the day someone adds a PyPI token. Recorded
  so a reader does not mistake it for a check that was never run.

attestation (R:SECRET, live half) — `gh secret list` and `gh secret list --env release` both returned
EMPTY on 2026-09-08, before any of this work. No PyPI credential has ever been configured on this
repository; the exposed token was never stored here. Re-check with those two commands.

HONESTY NOTE, recorded because the gate cannot infer it: these are assertions about a YAML file. They
prove the release path is CONFIGURED correctly. They cannot prove it WORKS — only an actual release can,
and the first one is the test. A green gate here means "the contract we wrote is present", not "yowo
published successfully". The first real release must be watched, not assumed.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
