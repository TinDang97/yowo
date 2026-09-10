# Release path: the setup only the maintainer can do

The release workflow is written, pinned and gated. It has **never run past its
second step.** Every push to `main` since 2026-08-26 fails at `Mint a scoped App
token`, so `Publish to PyPI` has never executed and `v2.5.0` is tagged but absent
from PyPI — while all eleven checks in `tests/unit/test_release_contract.py` pass,
because they assert the shape of a YAML file that describes a job which cannot
start.

```
$ gh run view <latest release run> --json jobs
success  Source Distribution (release)
success  Quality Gate (release)
failure  Semantic Release          ← "Mint a scoped App token"
skipped  Publish to PyPI

Error: The 'client-id' (or deprecated 'app-id') input must be set to a
       non-empty string.
```

Root cause, confirmed 2026-09-10: `gh variable list` and `gh secret list` are both
empty. `release.yml:94-95` reads `vars.RELEASE_APP_ID` and
`secrets.RELEASE_APP_PRIVATE_KEY`, and neither exists on the repository or on the
`release` environment.

Everything below needs credentials or a login that only you have. Nothing here can
be automated from a session.

---

## 1. Why there is a GitHub App at all

`semantic-release version` writes a version bump commit and a tag **directly to
`main`**. `main` is protected. The default `GITHUB_TOKEN` cannot push to a
protected branch — that is the `GH006` failure recorded at `release.yml:87-89`,
and the App exists to replace it. There is deliberately no fallback to
`GITHUB_TOKEN`, because a silent fallback would look like a new bug.

## 2. Create the App

**Settings → Developer settings → GitHub Apps → New GitHub App**

| Field | Value |
|---|---|
| Name | anything unique, e.g. `yowo-release` |
| Homepage URL | `https://github.com/TinDang97/yowo` |
| Webhook | **uncheck "Active"** — it needs none |
| Where can this be installed | *Only on this account* |

**Repository permissions** — grant exactly two:

| Permission | Level | Why |
|---|---|---|
| **Contents** | **Read and write** | push the bump commit, push the tag, and create the GitHub Release `semantic-release publish` uploads to |
| Metadata | Read-only | mandatory, granted automatically |

Grant nothing else. The token this App mints runs every step of the release job.

## 3. Install it on the repository

**Install App → your account → Only select repositories → `yowo`**

An App that exists but is not installed mints a token with no access, and the
failure looks nothing like a permissions problem.

## 4. Generate a private key

On the App's settings page: **Private keys → Generate a private key**. A `.pem`
downloads. It is shown once.

Do not put it in the repository, in `tmp/`, or anywhere `git status` can see it.

## 5. Wire both values in

The two are **different kinds** and the workflow reads them from different places.
Getting this wrong produces the exact error above.

```bash
# a repository VARIABLE — vars.RELEASE_APP_ID
gh variable set RELEASE_APP_ID --body "<the App ID from its settings page>"

# a repository SECRET — secrets.RELEASE_APP_PRIVATE_KEY
gh secret set RELEASE_APP_PRIVATE_KEY < /path/to/downloaded-key.pem
```

Paste the `.pem` **whole**, including the `-----BEGIN RSA PRIVATE KEY-----` and
`-----END RSA PRIVATE KEY-----` lines and the trailing newline.

Then delete the local `.pem`.

Confirm:

```bash
gh variable list      # RELEASE_APP_ID
gh secret list        # RELEASE_APP_PRIVATE_KEY
```

## 6. The step that is likely to bite — protected-branch push

**This is unverified and is the most probable next failure.**

`main` requires four status checks with `enforce_admins: true` (see
`docs/ci-required-checks.md`). Classic branch protection has no bypass list for
required *status checks*, so a direct push from the release job may be rejected
even once the App token mints correctly — a different failure, one step later.

Two ways it can go, and you will only know which by running it:

- **It works.** GitHub applies required status checks to pull requests, and the
  contexts were already reported green on the merge commit the release job is
  pushing on top of.
- **It is rejected** with `GH006 Protected branch update failed`. The fix is to
  convert `main`'s protection to a **repository ruleset** and add the App as a
  bypass actor: **Settings → Rules → Rulesets**, with the same four required
  checks and `yowo-release` under *Bypass list*. Classic protection cannot express
  this; a ruleset can.

Do not pre-emptively weaken protection to avoid this. Find out first.

## 7. PyPI trusted publisher

**pypi.org → your projects → `yowo` → Manage → Publishing → Add a new publisher →
GitHub**

| Field | Value |
|---|---|
| Owner | `TinDang97` |
| Repository | `yowo` |
| Workflow name | `release.yml` |
| Environment | `release` |

All four must match or the OIDC exchange is refused. `release.yml:148-155` binds
the `publish` job to the `release` environment and `id-token: write`, and
`pypa/gh-action-pypi-publish@v1.14.2` takes no `password:` — there is no
credential to fall back on if this is misconfigured.

## 7b. TestPyPI trusted publisher — do this one FIRST

This is m1 box 2 clause (ii), and it is the clause that would have caught
everything else on this page. `release.yml` has failed on every push since
2026-08-26, `Publish to PyPI` has never executed once, and `v2.5.0` is tagged but
absent from PyPI — while every check in `tests/unit/test_release_contract.py`
stayed green throughout, because they assert the shape of a job that cannot start.

`.github/workflows/release-dry-run.yml` exercises the identical OIDC mechanism
against TestPyPI, where a mistake costs nothing. Configure this **before** section
7, so the first thing you learn about trusted publishing is learned somewhere
harmless.

**test.pypi.org → Account settings → Publishing → Add a new PENDING publisher →
GitHub**

Pending, not "add to an existing project": `yowo` does not exist on TestPyPI
(verified 2026-09-10, `test.pypi.org/pypi/yowo/json` → 404). A pending publisher
creates the project on first successful upload.

| Field | Value |
|---|---|
| PyPI Project Name | `yowo` |
| Owner | `TinDang97` |
| Repository name | `yowo` |
| Workflow name | `release-dry-run.yml` |
| Environment name | `testpypi` |

All five must match exactly and the OIDC refusal does not say which one is wrong,
which is why the workflow prints them itself on failure. Note the workflow filename
and environment differ from section 7 on purpose: this publisher can mint a token
only for `release-dry-run.yml` running in `testpypi`, and the PyPI one only for
`release.yml` running in `release`. Neither can stand in for the other.

You also need the `testpypi` environment to exist:
**Settings → Environments → New environment → `testpypi`**. No secrets, no
protection rules — it is a name the publisher binds to, nothing more.

Then run it: **Actions → Release Dry Run → Run workflow**. It needs no tag.

**What "done" means here.** The box closes on a run that actually publishes, not
on the workflow file existing — the file is the part that has been proving nothing
for three weeks. Check `test.pypi.org/project/yowo/` shows a version afterwards.

## 8. Revoke every PyPI API token — and record it

This is m1 box 2 clause (iv), and it is currently **unsatisfied**. The milestone
records that a live PyPI token once sat world-readable on disk.

**pypi.org → Account settings → API tokens**, and **project → Manage → Settings**
for project-scoped tokens. Revoke all of them; trusted publishing needs none.

The clause asks for the result to be recorded as a dated human attestation in
`.add/tasks/pypi-trusted-publish.md`'s EVIDENCE section, which is still an
unfilled template. The existing attestation in that node is a different claim —
it records that `gh secret list` was empty on 2026-09-08, which is about GitHub
secrets, not PyPI tokens.

Tell me what the token page shows and I will write the attestation.

## 9. What is still missing in the repo, not in your settings

Clause (ii) — *a TestPyPI dry-run publishes from a tag* — had never existed. As of
2026-09-10 the **mechanism** ships: `.github/workflows/release-dry-run.yml`,
triggered by a `v*` tag or on demand, built and gated by ADD task
`testpypi-dry-run`.

What still does not exist is a **run**. The clause asks that a dry run *publishes*,
and thirteen checks asserting the shape of that workflow prove exactly as much as
the eleven that have been green over a release job which cannot start. Section 7b
is the part only you can do, and until a run appears on
`test.pypi.org/project/yowo/` this box stays open.

## 10. Verifying it actually worked

Do not trust a green workflow badge; that is how this went unnoticed for two
weeks. Check the thing itself:

```bash
gh run list --workflow release.yml --limit 1     # conclusion must be success
gh run view <id> --json jobs                     # "Publish to PyPI" must not be skipped
curl -s https://pypi.org/pypi/yowo/json | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```

The last one is the only one that proves a release happened.

## Related

- `docs/ci-required-checks.md` — branch protection and the check-name contract
- `.add/tasks/pypi-trusted-publish.md` — the frozen contract this implements
- `.github/workflows/release.yml` — the workflow itself, commented in place
