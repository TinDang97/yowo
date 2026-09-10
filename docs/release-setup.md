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

Clause (ii) — *a TestPyPI dry-run publishes from a tag* — has never existed.
TestPyPI returns `404` for `yowo`, the string appears nowhere in this repository,
and `release.yml:3-5` is `on: push: branches: [main]` with no `tags:` trigger at
all, so nothing here can be triggered by a tag.

That is code, not configuration, and it is the one clause that would have caught
the failure at the top of this page. It needs its own task.

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
