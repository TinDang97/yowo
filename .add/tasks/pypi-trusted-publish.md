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
