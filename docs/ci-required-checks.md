# Required status checks

This records the branch-protection configuration that makes the CI gate actually
block a merge. The workflow file alone gates nothing — GitHub runs it and reports
the result, but a merge is only *blocked* when the check is marked required on the
branch.

Frozen by ADD task `pr-ci-gate`. Change this file and
`.github/workflows/ci.yml` together, or `tests/unit/test_ci_contract.py` will fail.

## The configuration

| Setting | Value | Why |
|---|---|---|
| Branch | `main` | the only protected branch; releases cut from it |
| Required status checks | **all eight `ci.yml` jobs** — see the table below | GitHub exposes the job *name*, not its id. Every job `ci.yml` publishes is required; there is no advisory job, deliberately (see "Why all eight") |
| Strict (require branches up to date) | `false` | a solo maintainer rebasing every PR before merge is friction without a corresponding risk here |
| `enforce_admins` | **`true`** | decided 2026-09-08. Nobody bypasses a failing check, repository owner included — an advisory gate is the state this task exists to change |
| Required approving reviews | none | single maintainer; the gate is automated, not social |

## Check-run names, and which workflow owns each

A required status check is configured by **name**, not by workflow. If two workflows
publish the same check-run name, GitHub cannot tell which one satisfied the requirement —
a gate meant to prove a pull request passed can be satisfied by a push-to-`main` run
instead. Both workflows once published `Quality Gate` *and* `Source Distribution`; the
release path is now suffixed so every name resolves to exactly one job. The
TestPyPI dry run takes a name of its own for the same reason: it publishes on every
`v*` tag, and a required check it could satisfy would be a gate proving the wrong
run passed.

| Check-run name | Workflow | Job id | Required on `main`? |
|---|---|---|---|
| `Quality Gate` | `ci.yml` | `quality` | **yes** |
| `Source Distribution` | `ci.yml` | `sdist` | **yes** |
| `Reproducible Build` | `ci.yml` | `reproducible` | **yes** |
| `Weight Fixture` | `ci.yml` | `weights` | **yes** |
| `Real Backend Smoke` | `ci.yml` | `backend-smoke` | **yes** |
| `Backend Conformance` | `ci.yml` | `conformance` | **yes** |
| `Accuracy Dataset` | `ci.yml` | `accuracy-dataset` | **yes** |
| `mAP Gate` | `ci.yml` | `map-gate` | **yes** |
| `Quality Gate (release)` | `release.yml` | `quality` | no |
| `Source Distribution (release)` | `release.yml` | `sdist` | no |
| `Semantic Release` | `release.yml` | `release` | no |
| `Publish to PyPI` | `release.yml` | `publish` | no |
| `Publish to TestPyPI` | `release-dry-run.yml` | `testpypi` | no |

**Never rename a name in the left column that is marked required.** Branch protection
stores the string; renaming the job leaves protection referencing a context nothing
publishes, which never blocks anything and still looks configured.
`tests/unit/test_check_name_uniqueness.py` asserts both halves.

## Why all eight, not just `Quality Gate`

Amended 2026-09-15 by human decision. `Real Backend Smoke`, `Backend Conformance`
and `Accuracy Dataset` had each been running on every pull request while blocking
nothing, and `mAP Gate` joined them the day it landed. The cost was not
hypothetical: **Backend Conformance caught a real design fault that 2656 green
unit tests missed** — a precision request the backend could honour being refused —
and it could not have stopped that merging. A check that runs, reports, and cannot
fail a merge is worse than no check, because everybody reads it as a gate.

`tests/unit/test_check_name_uniqueness.py::test_every_ci_job_is_classified_as_gating_or_advisory`
now requires every `ci.yml` job to appear in the table above with a verdict, so a
new job cannot land silently advisory — which is how the previous four got there.
That test, and the one asserting this doc agrees with `branch-protection.json`,
read the payload rather than a copy of it: `REQUIRED_CONTEXTS` was a hardcoded
`("Quality Gate",)` while protection had required four since 2026-09-10, so the
guard against silently un-gating the branch was itself checking a stale list.

The cost is real and was accepted: a flake in any of the eight now blocks a merge,
and with `enforce_admins: true` there is no bypass short of the emergency path
below. The two slowest are `Backend Conformance` (~3m) and `mAP Gate` (~1m30s).

## Why all four, not just `Quality Gate`

Amended 2026-09-10 by human decision. Until then only `Quality Gate` was required, so
`Source Distribution`, `Reproducible Build` and `Weight Fixture` could each fail and a
pull request still merged. Those three are exactly the checks m1 built — the sdist
allowlist, the double-build digest comparison, and the digest-verified weight fixture.
Leaving them advisory meant the milestone's guarantees were unenforced on the branch
that ships, which is the same shape as a green gate proving nothing.

The cost is real and was accepted: a flake in any of the four now blocks a merge, and
with `enforce_admins: true` there is no bypass short of the emergency path below.

## Ordering — this matters

Protection **cannot** be enabled before `ci.yml` has run green on at least one real
pull request. GitHub will accept a required-check context it has never observed,
and then every merge blocks forever on a check that will never report. Enable it
only after the first green run.

## Applying it

```bash
gh api -X PUT repos/TinDang97/yowo/branches/main/protection \
  --input docs/branch-protection.json
```

Verify the live result — this is the check bound to the task, not a claim:

```bash
python3 scripts/verify_branch_protection.py
```

## The emergency path

`enforce_admins: true` means an urgent fix cannot be merged past a red check.
The deliberate path is to disable protection in
**Settings → Branches → `main`**, merge, and re-enable it. That sequence is
recorded in the repository audit log, which is the point — a bypass should leave
a trace rather than being a silent everyday capability.
