---
type: Task
title: OIDC trusted publishing, every release tool pinned
status: direction
depth: standard
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - .github/workflows/
gives:
  - S1 <the surface this publishes — an endpoint, function, or section>
depends_on:
  - /tasks/sdist-manifest.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: release-planner
---
## CARD
goal: yowo reaches PyPI from a pinned CI workflow authenticated by OIDC, with no long-lived credential anywhere.
why: there is no PyPI upload step in CI at all — `release.yml:69` runs `semantic-release publish`, which uploads to the GitHub Release, so both 2.4.0 and 2.4.1 were pushed from a local working tree. That is the mechanism that swept an untracked AGPL weight into two published sdists.

human decision (recorded, not derived) — the publishing path is **OIDC trusted publishing, no stored secret**.
  Asked with three options; the human answered "trusted publishing".
  Rejected: storing a freshly-minted API token — works, but keeps a long-lived credential and would have
    required reopening m1-trust-the-ship's frozen exit criterion "no long-lived token exists anywhere".
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
  criterion "no long-lived token exists anywhere" is verified by leg (iv) — "PyPI shows zero active
  project API tokens" — so the App key does not breach it as written, but the wording oversells the
  result and should be read as scoped to PyPI.

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
beat: scaffold · next: author pypi-trusted-publish's RULES, ASSUMPTIONS and CHECKS, then add freeze pypi-trusted-publish

## RULES
<must>
- M1 <the rule that must hold>
</must>
<reject>
- R:<NAME> <what must never happen> -> "<NAME>"
</reject>

## ASSUMPTIONS
- A1 [who] covers: <S ids> · the request does not say <who may act / whose data>; taking <reading> -> <cost if wrong>
- A2 [which] covers: <S ids> · the request does not say <which rows/cases are in>; taking <reading> -> <cost if wrong>
- A3 [when] covers: <S ids> · the request does not say <where the boundary falls>; taking <reading> -> <cost if wrong>
- A4 [absent] covers: <S ids> · the request does not say <what a missing value means>; taking <reading> -> <cost if wrong>
- A5 [order] covers: <S ids> · the request does not say <what orders / breaks a tie>; taking <reading> -> <cost if wrong>
- A6 [experience] covers: <S ids> · the request does not say <who receives this and what would make it hard for them>; taking <reading> -> <cost if wrong>
every `gives:` surface is swept on every dimension; `[<dim>] n/a · <why>` retires one. one line, one silence — split, never bundle. `· probe: <what shipped behavior must show>` declares a reading checkable: cite its A id from CHECKS and the gate holds the PASS to it.

## PLAN
contract: <the shape this publishes>

## EDGES
- E1 <a boundary or failure case a check must cover — optional>

## CHECKS
- <test_name> · covers: M1 · <what it proves>
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
