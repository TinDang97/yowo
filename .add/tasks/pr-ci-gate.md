---
type: Task
title: The quality gate runs before merge, not after
status: direction
depth: quick
milestone: m1-trust-the-ship
scope:
  - .github/workflows/ci.yml
  - tests/unit/test_ci_contract.py
  - scripts/verify_branch_protection.py
  - docs/ci-required-checks.md
gives:
  - S1 the `on:` trigger of `.github/workflows/ci.yml` — which events run the gate
  - S2 the job-id namespace of `.github/workflows/ci.yml` — the frozen ids later tasks append beside
  - S3 the documented required-status-check name — what a human must configure for the gate to block a merge
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-08, act: interview, authority: human, interview: "sha256:4e01440a957dcae9", receipt: /tasks/pr-ci-gate.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A16=confirm|A17=confirm|A18=confirm|A19=confirm|R:SILENT_PASS=confirm|R:DIVERGENCE=confirm" }
  - { by: "Tin Dang", at: 2026-09-08, act: freeze, authority: human, direction: "sha256:f98bc5dcf2027e9e", binding: "sha256:666ebb922435c321" }
advised_by: release-planner
---
## CARD
goal: A pull request cannot merge without the quality gate having run on it.
why: The gate's content is already good — it runs after the merge it was meant to guard, and six later tasks need a `ci.yml` that does not yet exist to append their jobs to.
beat: direction · next: run the checks red, then add freeze pr-ci-gate

## RULES
<must>
- M1 `.github/workflows/ci.yml` exists and triggers on `pull_request` targeting `main`.
- M2 It runs the same four quality steps `release.yml` runs — ruff check, ruff format --check, pyright, pytest tests/unit — so the pre-merge gate and the post-merge gate cannot disagree.
- M3 Its job ids are a declared, frozen set. Later tasks ADD a job; none renames or restructures an existing one.
- M4 The required-status-check configuration is written down in the repo, naming the exact check names GitHub will expose.
- M5 Branch protection on `main` is actually enabled — required check `Quality Gate`, `enforce_admins: true` — and its live state is asserted by a runnable check, not by a claim. The exact API payload is shown to the human and approved before it is applied.
</must>
<reject>
- R:SILENT_PASS A job that reports success while its command did not run — a `continue-on-error`, an `|| true`, or a step skipped by a condition that is always false. -> "SILENT_PASS"
- R:DIVERGENCE `ci.yml` and `release.yml` running different versions of the same check, so a PR can be green and `main` red on identical code. -> "DIVERGENCE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whether the gate runs on fork PRs from outside collaborators; taking the GitHub default — it runs, with a read-only token and no secrets — because this workflow needs no secrets -> if wrong, an outside contributor's PR shows no checks and looks unreviewed.
- A2 [who] covers: S3 · the request does not say who may bypass a failing required check; taking "nobody, including the repository owner" (`enforce_admins: true`) because a solo maintainer with bypass is the state the repo is in today -> if wrong, the gate is advisory and the finding it fixes returns quietly. · found: decided by the human at freeze, 2026-09-08 — `enforce_admins: true`, with the emergency path being to disable protection in settings, merge, and re-enable, which is visible in the audit log. No longer a priced guess.
- A3 [who] covers: S2 · n/a · a job id is not an authorization surface; who may edit it is governed by repository write access, which this task does not change.
- A4 [which] covers: S1 · the request does not say which paths should skip CI; taking "no path filter — every PR runs the gate", because a `paths-ignore` that skips docs would have skipped this very PR and the repo has no reliable docs/code split -> if wrong, CI minutes are spent on documentation-only PRs. Cheap to reverse.
- A5 [which] covers: S2 · the request does not say whether `release.yml`'s existing `quality` job is renamed or left alone; taking "left untouched, and `ci.yml` is additive", because renaming it would break the `release` job's `needs: [quality]` -> if wrong, the two gates drift apart, which R:DIVERGENCE exists to catch.
- A6 [which] covers: S3 · the request does not say which of the four steps must be required checks versus merely reported; taking "the single `ci.yml` job is the one required check", because per-step required checks do not exist in GitHub -> if wrong, a human configures a check name that never appears and the branch is unprotected while looking protected. · probe: the documented check name matches the `name:` the workflow actually publishes.
- A7 [when] covers: S1 · the request does not say whether the gate reruns on every push to an open PR or only on open; taking `on: pull_request` default activity types (`opened`, `synchronize`, `reopened`) so every new commit is gated -> if wrong, a PR is approved on a commit that was never tested.
- A8 [when] covers: S2 · n/a · job ids have no temporal boundary; they are a naming contract, not an event.
- A9 [when] covers: S3 · the request does not say when protection is enabled relative to this task landing; taking "the workflow lands and runs green on at least one PR FIRST, then this task enables protection via the API", because GitHub will not accept a required-check context it has never observed -> if wrong, the API call succeeds with a context that never reports and every subsequent merge blocks forever. This ordering is the whole reason protection cannot be enabled in the same step that creates the workflow.
- A10 [absent] covers: S1 · the request does not say what happens when `uv sync` cannot resolve; taking "the job fails", which is the default -> low cost, this is the desired behavior.
- A11 [absent] covers: S2 · the request does not say what a later task should do if the job id it wants already exists; taking "it appends a NEW id rather than reusing one", because reuse silently changes what an existing required check means -> if wrong, a required check named `quality` starts asserting something the branch-protection setting was never approved for.
- A12 [absent] covers: S3 · the request does not say what to do while protection is absent; taking "this task enables it via `gh api -X PUT`, with the payload approved by the human before it runs, and `scripts/verify_branch_protection.py` asserting the live result", because `main` is currently unprotected — `gh api repos/TinDang97/yowo/branches/main/protection` returns 404 and `rulesets` returns `[]` -> if wrong, the milestone's exit box reads as satisfied by a file that gates nothing. · found: main has no branch protection and no rulesets (evidence: `gh api repos/TinDang97/yowo/branches/main/protection` → 404 "Branch not protected"; `gh api repos/TinDang97/yowo/rulesets` → `[]`, 2026-09-08).
- A13 [order] covers: S1 · the request does not say whether the four steps run in a fixed order or in parallel jobs; taking "one job, steps in `release.yml`'s existing order (lint, format, types, tests)", because a single job is a single required check and cheap steps failing first is the useful ordering -> if wrong, a slow test run masks a one-second lint failure.
- A14 [order] covers: S2 · the request does not say what orders jobs relative to one another; taking "no `needs:` between appended jobs unless a later task declares one", so the six appenders stay independent -> if wrong, one slow appended job serializes the whole gate.
- A15 [order] covers: S3 · n/a · a required-check set is unordered by GitHub's own model; all listed checks must pass and no sequence is implied.
- A16 [experience] covers: S1 · the recipient is a contributor watching their PR. The request does not say what they see on failure; taking "the failing step's own output, no summary layer", because adding a reporting layer is scope this task does not have -> if wrong, a contributor reads a 15-minute pytest log to find one lint error. Mitigated by A13's ordering.
- A17 [experience] covers: S2 · the recipient is the author of one of the six later tasks, reading this contract to know where their job goes. The request does not say where that contract is written; taking "a comment block at the top of `ci.yml` naming the frozen ids and the append rule", because a convention documented outside the file it governs is not read -> if wrong, six tasks each invent their own structure and the collision this task exists to prevent happens anyway.
- A18 [experience] covers: S3 · the recipient is the maintainer configuring branch protection once, in a web UI, months from now. The request does not say what they need; taking "exact check name, exact setting path, and the `enforce_admins` decision, written in the repo", because a half-remembered setting is the failure mode here -> if wrong, protection is enabled with the wrong check name and merges are blocked on a check that never reports.
- A19 [absent] covers: S1 · the request does not say whether `ci.yml` should assert the lockfile is current; taking "plain `uv sync --group dev`, mirroring `release.yml`", NOT `--locked`, because `uv lock --check` reports the committed lockfile already needs updating, so `--locked` would fail every PR from day one -> if wrong, CI silently re-resolves dependencies the lockfile claims to pin, and a PR is tested against packages `uv.lock` does not name. · found: `uv lock --check` → "The lockfile at `uv.lock` needs to be updated" (2026-09-08); local uv 0.10.11 writes `revision = 3`, committed lock is `revision = 2`. Filed as specs/quality Q1; refreshing the lock is its own node, not this one.

## PLAN
contract: A new `.github/workflows/ci.yml`, triggered on `pull_request` against `main`, containing exactly one job whose id is `quality` and whose `name:` is `Quality Gate` — matching `release.yml`'s job so the two gates are recognisably the same gate. A header comment declares the append rule for the six later tasks. `docs/ci-required-checks.md` records the check name and the branch-protection settings a human must apply, with the `enforce_admins` decision.

strategy: (1) write `tests/unit/test_ci_contract.py` asserting the trigger, the job-id set, the four step commands, and the absence of `continue-on-error` / `|| true` — run it RED against the absent file; (2) add `ci.yml`; (3) add `docs/ci-required-checks.md`; (4) green on the file contract; (5) let `ci.yml` run green on a real PR, so GitHub observes the `Quality Gate` context; (6) show the human the exact `gh api` payload, apply it on approval; (7) `scripts/verify_branch_protection.py` asserts the live state and emits JUnit XML for the receipt. No change to `release.yml` — A5 keeps `needs: [quality]` intact.

why a bespoke runner: the live-state assertion cannot be a pytest test in the unit suite — it needs an authenticated admin token and network, so in CI it could only ever `skipif` green, which `inference-parity-engineer` rules out as proof. `add run` parses JUnit XML regardless of what produced it, so a script comparing the API response against the frozen RULES earns the same bound receipt.

sequencing note: steps 5–7 mean this task cannot close in one sitting — it needs one real PR to have run first. That is inherent to the API, not to the plan.

regression floor: the existing unit suite stays green; `release.yml` is byte-unchanged.

## EDGES
- E1 A workflow file that parses as valid YAML but declares a trigger GitHub ignores (e.g. `on: pull-request`) — the file exists, the test that only checks existence passes, and no check ever runs.
- E2 A later task appends a job that reuses the id `quality`, silently changing what the configured required check asserts.
- E3 Protection is enabled naming a required context GitHub has never observed — the API accepts it, and every merge blocks forever on a check that will never report. A9's ordering exists to prevent this; the verify script must catch it if the ordering is violated.

## CHECKS
- test_ci_workflow_triggers_on_pull_request · covers: G1 · loads `ci.yml` with `yaml.safe_load` and asserts `pull_request` is a key of the parsed `on:` mapping with `main` in its `branches` — catches E1, since `on: pull-request` parses fine and fails this.
- test_ci_job_ids_are_the_frozen_set · covers: G2 · asserts the job-id set is exactly `{"quality"}`, so any later append is a deliberate edit of this assertion rather than a silent addition — catches E2.
- test_ci_runs_the_same_four_quality_commands_as_release · covers: G2 · parses both workflows and asserts `ci.yml`'s step `run:` commands for ruff/format/pyright/pytest are equal to `release.yml`'s — binds R:DIVERGENCE.
- test_ci_has_no_silently_passing_step · covers: G2 · asserts no step in `ci.yml` carries `continue-on-error: true` and no `run:` contains `|| true` — binds R:SILENT_PASS.
- test_required_check_name_matches_published_job_name · covers: G3 · asserts the check name recorded in the required-checks doc equals the `name:` the `quality` job publishes — binds A6's probe, and catches protection configured against a name that never reports.
- verify_branch_protection · covers: G3 · `scripts/verify_branch_protection.py` queries `repos/TinDang97/yowo/branches/main/protection` and asserts `required_status_checks.contexts` contains the published job name AND `enforce_admins.enabled is true`; exits non-zero and emits JUnit XML otherwise — binds M5. Currently red for the strongest possible reason: the endpoint returns 404 `Branch not protected`.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
