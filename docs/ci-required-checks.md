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
| Required status check | **`Quality Gate`** | the `name:` of the `quality` job in `ci.yml` — GitHub exposes the job *name*, not its id |

## Check-run names, and which workflow owns each

A required status check is configured by **name**, not by workflow. If two workflows
publish the same check-run name, GitHub cannot tell which one satisfied the requirement —
a gate meant to prove a pull request passed can be satisfied by a push-to-`main` run
instead. Both workflows once published `Quality Gate` *and* `Source Distribution`; the
release path is now suffixed so every name resolves to exactly one job.

| Check-run name | Workflow | Job id | Required on `main`? |
|---|---|---|---|
| `Quality Gate` | `ci.yml` | `quality` | **yes** |
| `Source Distribution` | `ci.yml` | `sdist` | no — see `pypi-trusted-publish` |
| `Quality Gate (release)` | `release.yml` | `quality` | no |
| `Source Distribution (release)` | `release.yml` | `sdist` | no |
| `Semantic Release` | `release.yml` | `release` | no |
| `Publish to PyPI` | `release.yml` | `publish` | no |

**Never rename a name in the left column that is marked required.** Branch protection
stores the string; renaming the job leaves protection referencing a context nothing
publishes, which never blocks anything and still looks configured.
`tests/unit/test_check_name_uniqueness.py` asserts both halves.
| Strict (require branches up to date) | `false` | a solo maintainer rebasing every PR before merge is friction without a corresponding risk here |
| `enforce_admins` | **`true`** | decided 2026-09-08. Nobody bypasses a failing check, repository owner included — an advisory gate is the state this task exists to change |
| Required approving reviews | none | single maintainer; the gate is automated, not social |

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
