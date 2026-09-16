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
| Required status checks | **all thirteen `ci.yml` jobs (one of which is a five-leg matrix), plus `Export Parity`** — see the table below | GitHub exposes the job *name*, not its id. Every job `ci.yml` publishes is required; there is no advisory job, deliberately (see "Why all of them") |
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
| `Arch Equivalence` | `ci.yml` | `arch-equivalence` | **yes** |
| `CLI End-to-End` | `ci.yml` | `cli-e2e` | **yes** |
| `Persistent Gallery` | `ci.yml` | `persistent-gallery` | **yes** |
| `Python Claim (3.8)` | `ci.yml` | `python-claim` | **yes** |
| `Python Claim (3.9)` | `ci.yml` | `python-claim` | **yes** |
| `Python Claim (3.10)` | `ci.yml` | `python-claim` | **yes** |
| `Python Claim (3.11)` | `ci.yml` | `python-claim` | **yes** |
| `Python Claim (3.12)` | `ci.yml` | `python-claim` | **yes** |
| `Unit Tests (3.12)` | `ci.yml` | `unit-newest` | **yes** |
| `Export Parity` | `parity.yml` | `parity-pr` | **yes** |
| `Export Parity (all variants)` | `parity.yml` | `parity-all` | no — see below |
| `Quality Gate (release)` | `release.yml` | `quality` | no |
| `Source Distribution (release)` | `release.yml` | `sdist` | no |
| `Semantic Release` | `release.yml` | `release` | no |
| `Publish to PyPI` | `release.yml` | `publish` | no |
| `Publish to TestPyPI` | `release-dry-run.yml` | `testpypi` | no |

**Never rename a name in the left column that is marked required.** Branch protection
stores the string; renaming the job leaves protection referencing a context nothing
publishes, which never blocks anything and still looks configured.
`tests/unit/test_check_name_uniqueness.py` asserts both halves.

## Why `Export Parity (all variants)` is deliberately NOT required

Amended 2026-09-15. `parity-all` runs on a schedule, not on pull requests, so on
a pull request it reports **`skipping`** — and GitHub treats a skipped required
check as **satisfied**. Requiring it would produce a context that is green on
every pull request without ever executing: a gate that never runs, which is
strictly worse than an advisory job because it also looks enforced.

Its evidence comes from the scheduled run instead. That run must be watched:
nothing blocks a merge on it, by construction.

Recorded because it is the second time this shape has appeared today. The
classification guard below was written to catch a job that runs and blocks
nothing — and was scoped to `ci.yml`, so `parity.yml` escaped it within the
hour, publishing `Export Parity` on every pull request while appearing in no
table. It now covers every workflow with a `pull_request` trigger.

## What CI does NOT cover, and why

Recorded 2026-09-16 by ADD task `ci-matrix`. These are decisions, not
oversights, and `tests/unit/test_claimed_python_versions_run_in_ci.py` fails if
either disappears from this file — a limitation kept only in a comment stops
being read.

**The unit tier runs on 3.11 and 3.12 only, not on the whole claimed range.**
Not laziness: measured on real interpreters, `src/yowo` compiles 96/96 under
CPython 3.8.20, but the TEST SUITE compiles only 141/163 there — 22 modules use
parenthesized context managers, which are 3.10+. Under CPython 3.10.20 all 163
compile, yet 6 modules `import tomllib`, and that is 3.11+ stdlib. The test
suite's floor is therefore **3.11**, four versions above the package's. Raising
the package's floor to match would be a breaking change for users; lowering the
suite's would mean rewriting 22 test files. Instead, `Python Claim (3.8)`
through `(3.12)` install the package the way a consumer does and exercise its
documented public surface, which is what the classifiers actually promise.

**CI runs ubuntu-latest only.** The package declares no OS classifiers, so it
implicitly claims every operating system, and nothing here tests macOS or
Windows. Human decision 2026-09-16: record the gap, add no OS job.
`Backend Conformance` already carries a strict xfail scoped to macOS arm64 for
an unexplained PyTorch-vs-ONNX divergence of 0.33 px against a 1e-3 bound,
owned by `pytorch-onnx-numeric-divergence` in m4-honest-deployment. Adding
macOS to the integration tier today would go red on arrival, which is what
m3's own wording warns against — "do not multiply a red suite by eight".

## Why all of them, not just `Quality Gate`

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

The cost is real and was accepted: a flake in any of the 18 contexts now blocks a merge,
and with `enforce_admins: true` there is no bypass short of the emergency path
below. The two slowest are `Backend Conformance` (~3m) and `mAP Gate` (~1m30s).
`Arch Equivalence` joined them on 2026-09-16: it compares all ten variants
against ultralytics, and the comparison itself takes under 10 s. `CLI
End-to-End` and `Persistent Gallery` joined the same day, reviving 29 tests
that existed and that no workflow ran.

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
python3 scripts/verify_branch_protection.py --ref <a pull-request head sha>
```

**Pass `--ref`.** `ci.yml` runs on `pull_request`, so a commit on `main` carries
only `release.yml`'s checks and observation cannot be confirmed from there. Run
without it and the script says so; before 2026-09-15 it instead reported that
protection "would block every merge on a check that never reports", which was a
false alarm — the checks report fine, on pull requests, which is where they are
required. The script reads its context list from `branch-protection.json` rather
than carrying a copy; the copy it used to carry said four while the payload said
eight, so dropping half the contexts would have verified green.

## The emergency path

`enforce_admins: true` means an urgent fix cannot be merged past a red check.
The deliberate path is to disable protection in
**Settings → Branches → `main`**, merge, and re-enable it. That sequence is
recorded in the repository audit log, which is the point — a bypass should leave
a trace rather than being a silent everyday capability.
