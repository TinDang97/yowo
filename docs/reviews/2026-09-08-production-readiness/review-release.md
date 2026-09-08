# Packaging / CI-CD / Release Engineering / Documentation review — yowo production readiness

## Verdict

**Not production-ready.** The library code is in far better shape than the machinery that ships it.
The single biggest gap is that **there is no end-to-end release pipeline at all**: the only workflow
has been failing on every push to `main` for the last four commits, it has never contained a PyPI
publish step, and every PyPI upload to date was hand-built on a developer laptop — which is how
`yowo 2.4.1` came to ship a 5.6 MB **AGPL-3.0 ultralytics weight file inside an sdist that declares
`License-Expression: Apache-2.0`**. That artifact is public on PyPI right now.

Secondary but structural: there is no PR-triggered CI (every merged PR got zero automated
validation), no OS/Python matrix despite a `requires-python >=3.8` claim aimed at Jetson Nano, and
`yowo.__version__` disagrees with the package version in the artifact currently being built.

All P0s below are cheap to fix individually — they cluster because release engineering has never
had an owner, not because the codebase is unsound.

---

## P0 — blocks launch

### R1. The published sdist contains an AGPL-3.0 model weight, under an Apache-2.0 license declaration

- **Symptom:** `yowo-2.4.1.tar.gz` on PyPI (6.0 MB, uploaded 2026-03-26) contains
  `yowo-2.4.1/yolo11n.pt` (5,613,764 bytes). That file's own serialized metadata reads
  `license: AGPL-3.0 License (https://ultralytics.com/license)` and it deserializes
  `ultralytics.nn.tasks` / `ultralytics.nn.modules.*` classes. The wheel's
  `METADATA` declares `License-Expression: Apache-2.0`.
- **Evidence:**
  - Published artifact: `curl https://pypi.org/pypi/yowo/2.4.1/json` -> sdist `size: 5997783`;
    `tar tzf yowo-2.4.1.tar.gz` -> `yowo-2.4.1/yolo11n.pt`.
  - Weight self-declares AGPL: `strings yolo11n.pt` -> `AGPL-3.0 License (https://ultralytics.com/license)`.
  - Same file reproduces in a local build today: `yowo-2.5.0.tar.gz` (7.4 MB) contains `yolo11n.pt`.
  - `pyproject.toml:6` `license = "Apache-2.0"`; `README.md:801-803` "Apache-2.0 — see LICENSE",
    with no mention of weight licensing anywhere in the README.
  - `CONTRIBUTING.md:27` asserts *"yowo is intentionally ultralytics-free (Apache-2.0 clean)"* —
    materially false as shipped.
  - The GitHub release note for v2.5.0 states *"This release is published under the Apache-2.0 License."*
  - `src/yowo/models/_registry.py:14,108,109` — the runtime downloads weights from
    `https://github.com/ultralytics/assets/releases/download/v8.3.0/` and `.../v8.4.0/`, i.e.
    AGPL-3.0 assets, into `~/.cache/yowo/weights/` on first use (`README.md:137`).
- **Why it blocks:** This is a live license violation on a public index, not a theoretical one.
  Any adopter's legal review will find an AGPL artifact inside a package that promises Apache-2.0
  and a CONTRIBUTING file that explicitly denies the dependency. AGPL-3.0 has network-use copyleft;
  a team deploying yowo as a service on ultralytics-derived weights inherits obligations the README
  told them did not exist. It also makes the `pip install yowo` supply chain carry an unvetted
  serialized torch checkpoint (arbitrary-code-execution risk on `torch.load`) that no one in this
  repo authored.
- **Smallest fix:** Three parts, all small.
  1. Yank/replace the affected PyPI sdists (2.4.1 at minimum) and never build releases from a dirty
     working tree again (see R2).
  2. Add an explicit `[tool.hatch.build.targets.sdist]` allowlist so no `*.pt` can ever be swept in.
  3. Add a `## Model weights and licensing` section to `README.md` stating plainly: the *code* is
     Apache-2.0; the *default weights are downloaded at runtime from ultralytics and are AGPL-3.0*;
     no weights are redistributed by this package; adopters who cannot accept AGPL must supply their
     own weights via `--weights` / `weights_path`. Correct `CONTRIBUTING.md:27` to
     "ultralytics-free at *runtime*" and add a `NOTICE` file.

### R2. The sdist has no include/exclude list — any untracked file in the working tree ships

- **Symptom:** `pyproject.toml:87-88` configures only the *wheel* target
  (`[tool.hatch.build.targets.wheel] packages = ["src/yowo"]`). There is no sdist configuration, so
  hatchling's default sweeps the whole project directory minus VCS-ignored paths. Untracked-but-not-
  ignored files therefore become release artifacts.
- **Evidence:** local `uv build --out-dir <scratch>` of the current tree produced
  `yowo-2.5.0.tar.gz` containing, at top level: `yolo11n.pt` (5.6 MB, untracked),
  `CLAUDE.md.bak` (untracked), `.add/` (**313 files** of agent tooling, untracked),
  `uv.lock` (1.15 MB), `tests/` (102 entries), `docs/` (23 entries incl. `docs/plans/*`),
  `CLAUDE.md`, `CONTEXT.md`, `.github/`, `.pre-commit-config.yaml`.
  `git ls-files --others --exclude-standard` currently lists 24 such shippable files.
  Compare artifact sizes for the same code: CI-built GH release asset
  `yowo-2.5.0.tar.gz = 971,002 B` vs. laptop-built `yowo-2.5.0.tar.gz = 7.4 MB`.
- **Why it blocks:** Releases are not reproducible — the artifact depends on what happened to be
  lying in the maintainer's directory. That is exactly the mechanism that produced R1, and the next
  time it fires the stray file could be a customer video or a private plan doc rather than a model
  weight. It also inflates the edge-install footprint 7x for no benefit.
- **Smallest fix:** in `pyproject.toml`, add

  ```toml
  [tool.hatch.build.targets.sdist]
  include = ["src/yowo", "tests", "README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml"]
  ```

  (an allowlist, not a blocklist — a blocklist re-opens the same hole for the next stray file), and
  only ever build release artifacts in CI (R4).

### R3. The release workflow has failed on every push to `main` for the last four commits

- **Symptom:** `gh run list` shows the four most recent `Release` runs — one per commit since
  v2.5.0 — all `failure`. Root cause in the job log:
  `##[error] type object 'Actor' has no attribute 'name_email_regex'`, raised by
  python-semantic-release against an incompatible GitPython. The tool is resolved **unpinned at run
  time**, so an upstream release silently broke the pipeline.
- **Evidence:**
  - `.github/workflows/release.yml:67-69` —
    `uv tool run --from python-semantic-release semantic-release version` / `... publish`
    with no version specifier.
  - `gh run view 32955168960 --log-failed` -> `Actor has no attribute 'name_email_regex'`,
    `Process completed with exit code 1`.
  - Failing runs: `32955168960`, `32955146828`, `32945852699`, `32945802670` (all `push`/`main`,
    2026-08-26). These correspond to `3340bb3`, `58e04fb`, `13e0b69`, `3a9b78b`.
  - `git log v2.5.0..HEAD` -> 4 `fix(...)` commits with no tag; a patch release was owed and never cut.
- **Why it blocks:** `main` is red and has been for every commit since the last release, and nobody
  noticed — meaning there is no signal at all that a release succeeded or failed. Four user-facing
  bug fixes (including "importing yowo no longer drags torch in" and two Jetson GPU-detection fixes,
  i.e. precisely the edge-fleet audience) are sitting unreleased and unreachable by `pip install`.
- **Smallest fix:** pin the tool and fail loudly —
  `uv tool run --from 'python-semantic-release==9.*' semantic-release ...` (or pin exactly in a
  `.github/release-requirements.txt`), then re-run to cut v2.5.1. Add failure notification so a red
  `main` is not discovered by a reviewer three months later.

### R4. There is no PyPI publish step anywhere — every upload has been manual

- **Symptom:** `semantic-release publish` uploads dist files to the **GitHub release**, not to PyPI
  (PSR dropped PyPI upload in v8). The workflow contains no `twine`, no
  `pypa/gh-action-pypi-publish`, no `uv publish`. Consequently the index is chronically behind the
  tags, and uploads happen by hand from an unclean laptop.
- **Evidence:**
  - `.github/workflows/release.yml:49-69` — the entire `release` job: checkout, setup-python,
    setup-uv, `semantic-release version`, `semantic-release publish`. Nothing else.
  - `pyproject.toml:161-163` `[tool.semantic_release.publish] dist_glob_patterns = ["dist/*"]`,
    `upload_to_vcs_release = true` — VCS release only.
  - Tagged but **never on PyPI**: `0.0.1, 1.0.1, 2.2.2, 2.3.0, 2.5.0`. PyPI latest is **2.4.1**
    while `pyproject.toml:3` says `2.5.0`.
  - Timing proves manual: the v2.4.1 release run finished `2026-03-25T09:27Z`; the PyPI files were
    uploaded `2026-03-26T09:22:50Z` — a day later, by a human.
  - `.github/workflows/release.yml:9` declares `id-token: write` but no OIDC/trusted-publishing
    consumer exists, and no `actions/attest-build-provenance`.
- **Why it blocks:** "Is v2.5.0 released?" currently has three different answers (tag: yes; GitHub
  release: yes; PyPI: no). Adopters install from PyPI. A library whose advertised version cannot be
  installed is not launched. Manual uploads also mean no provenance, no attestation, and — as R1/R2
  show — a different artifact from the one CI reviewed.
- **Smallest fix:** add a publish step to the `release` job using PyPI **Trusted Publishing** (the
  `id-token: write` permission is already there):

  ```yaml
  - run: uv build --out-dir dist
  - uses: pypa/gh-action-pypi-publish@release/v1   # OIDC, no API token secret
  ```

  gated on the semantic-release step having actually produced a new tag
  (`steps.release.outputs.released == 'true'`). Add `actions/attest-build-provenance` for SLSA
  provenance. Remove the human from the loop entirely.

### R5. No PR-triggered CI — the quality gate does not gate

- **Symptom:** the only workflow triggers on `on: push: branches: [main]`. Pull requests run
  nothing. The "quality gate" runs *after* code is already merged into `main`.
- **Evidence:**
  - `.github/workflows/release.yml:3-5`:
    ```yaml
    on:
      push:
        branches: [main]
    ```
  - `gh run list --limit 15` — every run's event is `push`, none is `pull_request`.
  - PRs #7, #8, #9, #11, #12, #13, #14, #15 are all merged into `main` (see `git log --oneline -30`)
    and none of them was validated before merge.
  - `.github/workflows/` contains exactly one file (`ls .github/workflows/` -> `release.yml`).
- **Why it blocks:** a merge flow with no pre-merge signal means `main` can go red at any moment —
  which it did (R3), and stayed red. Adopters reading "quality gates" in `CONTRIBUTING.md:48-52`
  reasonably assume PRs are checked; they are not. There is also no branch-protection-usable status
  check to require.
- **Smallest fix:** split the `quality` job into `.github/workflows/ci.yml` with
  `on: [pull_request, push: branches:[main]]`, keep `release.yml` for the release job, and mark the
  CI job as a required status check on `main`.

### R6. `requires-python >=3.8` is a claim no CI, and no lockfile, has ever tested

- **Symptom:** the package advertises Python 3.8–3.12 support and the README points Jetson Nano
  users at it, but the only Python ever exercised is 3.11 on `ubuntu-latest`. The uv lockfile is
  explicitly restricted to `>=3.9`, so 3.8 is not even resolvable in this repo.
- **Evidence:**
  - `pyproject.toml:7` `requires-python = ">=3.8"`; `pyproject.toml:14-18` classifiers 3.8->3.12.
  - `pyproject.toml:78-81` `[tool.uv] environments = ["python_version >= '3.9'"]` — 3.8 is excluded
    from resolution.
  - `.github/workflows/release.yml:20-21` and `:57-58` — `python-version: "3.11"`, no matrix.
  - `.python-version` -> `3.11`; `pyproject.toml:100` `pythonVersion = "3.11"` for pyright;
    `pyproject.toml:91` `target-version = "py38"` for ruff. Four different Python assumptions in one
    file.
  - Single OS: `runs-on: ubuntu-latest` only — no macOS despite `coreml`/MPS being a headline
    feature (`README.md:156`, `README.md:738`), no Windows, no aarch64/Jetson smoke test.
  - The v2.4.1 changelog entry (`CHANGELOG.md:18-50`) records four 3.8 runtime breakages found by
    **manual testing on Python 3.8.20**, not by CI — the exact failure mode this predicts.
  - `pyproject.toml:41` `tracking = ["scipy>=1.11; python_version >= '3.9'"]` — on 3.8 the
    `tracking` extra installs *nothing*, and `src/yowo/tracking/_matching.py:18` imports
    `scipy.optimize.linear_sum_assignment`. A 3.8 user gets an empty extra and a runtime surprise.
  - Python 3.8 reached end-of-life in October 2024; `opencv-python-headless>=4.8` and
    `numpy>=1.24` have no upper bound, so a fresh 3.8 resolve is unconstrained and untested.
- **Why it blocks:** the flagship edge target (Jetson Nano, Python 3.8) is the least-tested
  configuration in the project, and history shows it broke silently last time. Shipping an
  advertised-but-unexercised `requires-python` floor is how a stranger on a Jetson gets a traceback
  instead of an answer.
- **Smallest fix:** add a matrix to the new `ci.yml`:
  `python-version: ["3.8", "3.9", "3.11", "3.13"]` x `os: [ubuntu-latest, macos-latest]`
  (install with `uv pip install -e ".[onnx]"` rather than the >=3.9-scoped lock, so 3.8 is really
  resolved), plus one `runs-on: ubuntu-24.04-arm` import-and-`yowo info` smoke job for aarch64.
  If 3.8 cannot be made green, raise `requires-python` to `>=3.9` and drop the 3.8 classifier —
  either answer is fine; the current "claimed, never run" is not.

---

## P1 — before GA

### R7. `yowo.__version__` is stale — the shipped wheel reports two different versions

- **Symptom:** In a clean venv installed from the current build, `yowo --version` prints `2.5.0`
  while `yowo.__version__` prints `2.4.1`.
- **Evidence:** `src/yowo/__init__.py:123` `__version__ = "2.4.1"` (hardcoded);
  `pyproject.toml:3` `version = "2.5.0"`; `pyproject.toml:128`
  `version_toml = ["pyproject.toml:project.version"]` — semantic-release bumps *only* the TOML.
  Verified: `git show 59a3f35 --stat` -> the v2.5.0 release commit changed `pyproject.toml` alone,
  1 insertion. `src/yowo/cli/_main.py:45` uses `@click.version_option(package_name="yowo")` (reads
  dist metadata -> correct), while `src/yowo/export/_exporter.py:199` stamps
  `yowo_version=getattr(yowo, "__version__", "0.1.0")` into every `.yowo.json` export sidecar -> every
  exported model since v2.4.1 records the wrong producing version.
- **Fix:** delete the literal and derive it —
  `from importlib.metadata import version; __version__ = version("yowo")` (with a
  `PackageNotFoundError` fallback) — or add `src/yowo/__init__.py:__version__` to
  `[tool.semantic_release] version_variables`.

### R8. The CHANGELOG is not actually generated, and 7 tagged versions have no entry

- **Symptom:** `CHANGELOG.md:8-10` claims *"Releases after v0.1.0 are generated automatically by
  python-semantic-release"*. They are not — PSR has never written to this file.
- **Evidence:** the file contains no PSR insertion flag (`grep -n "version list\|<!--" CHANGELOG.md`
  -> no matches), so PSR silently skips it. `git show 59a3f35` touched only `pyproject.toml`.
  Tagged but absent from the changelog: `1.0.0, 1.0.1, 1.0.2, 1.1.1, 2.2.1, 2.3.0, 2.5.0`.
  Present in the changelog but never tagged: `2.2.3` (`CHANGELOG.md:189`). The `[Unreleased]`
  section (`CHANGELOG.md:14`) is empty while four user-facing fixes sit unreleased on `main`.
  `pyproject.toml:144` `template_dir = ".github/templates"` points at a directory that does not
  exist (`ls .github/` -> `workflows` only).
- **Fix:** add `<!-- version list -->` at the insertion point, drop the dead `template_dir`, and
  backfill the 7 missing entries — or delete the automation claim at `CHANGELOG.md:8-10` and own it
  as hand-maintained. Either is defensible; the current half-state means an adopter cannot tell what
  changed between two installable versions.

### R9. PyPI metadata is unusable for evaluation — no URLs, alpha status, missing classifiers

- **Symptom:** the PyPI page has no link to the repository, issues, docs, or changelog, and tells
  visitors the project is alpha.
- **Evidence:** `pyproject.toml` has **no `[project.urls]` table** at all; confirmed live —
  `curl https://pypi.org/pypi/yowo/json` -> `project_urls: None`, `home_page: None`.
  `pyproject.toml:11` `"Development Status :: 3 - Alpha"` contradicts
  `pyproject.toml:4` "Production YOLO inference and export library" and `README.md:3-5`.
  No `Typing :: Typed` classifier despite the wheel shipping `yowo/py.typed`. No `3.13` classifier
  despite `docs/experiments/2026-02-25-phase3-source-aware-pipeline-benchmark.md` reporting
  free-threaded 3.13t results. No `Operating System` classifiers.
- **Fix:** add

  ```toml
  [project.urls]
  Homepage = "https://github.com/TinDang97/yowo"
  Repository = "https://github.com/TinDang97/yowo"
  Issues = "https://github.com/TinDang97/yowo/issues"
  Changelog = "https://github.com/TinDang97/yowo/blob/main/CHANGELOG.md"
  ```

  bump to `Development Status :: 5 - Production/Stable` (or `4 - Beta`) at launch, add
  `Typing :: Typed` and the 3.13 classifier.

### R10. The README's install section is wrong in a way that breaks first run

- **Symptom:** `README.md:11-13` labels `pip install yowo` as *"Core (PyTorch backend, CPU
  inference)"*. Core dependencies (`pyproject.toml:22-29`) do **not** include torch; `torch` is only
  in the `pytorch`/`export`/`all` extras (`pyproject.toml:32-33,46`). A user who follows line 13 and
  then line 44 (`yowo detect image.jpg`) gets no working backend.
- **Evidence:** verified in a clean 3.11 venv with only the core wheel installed:
  `InferenceEngine()` raises `yowo.errors.DependencyError: Missing optional dependency:
  'inference backend'` — the error handling is *good* (see "What is already good"), but the README
  promised the install would work.
  Also `README.md:24-25` calls `yowo[all]` "Everything (ONNX GPU + OpenVINO)" while
  `pyproject.toml:46` resolves it to `pytorch,export,onnx-gpu,openvino,coreml,tracking,chromadb,benchmark`
  — including `chromadb` (a vector database, hundreds of MB with its own onnxruntime and tokenizers),
  which makes `yowo[all]` a non-starter on the edge devices this library targets.
  `README.md:768` still says `git clone https://github.com/your-org/yowo` — an unreplaced placeholder
  in the Development section (the real URL is used correctly in `CONTRIBUTING.md:38`).
- **Fix:** relabel line 12 to `# Core (no inference backend — pick one below)`, make
  `pip install yowo[onnx]` the recommended default, correct the `[all]` description, move `chromadb`
  out of `all` into its own opt-in, and replace the `your-org` placeholder.

### R11. No security policy, no community health files, no dependency scanning

- **Symptom:** an adopter has no channel to report a vulnerability and no signal that dependencies
  are watched.
- **Evidence:** `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `.github/dependabot.yml`, and
  `.github/ISSUE_TEMPLATE/` all absent (`ls` -> "No such file or directory" for each).
  `.github/` contains only `workflows/`. No `pip-audit`, `safety`, `osv-scanner`, CodeQL, or SBOM
  step anywhere in `release.yml`. Actions are pinned to floating tags, not commit SHAs
  (`release.yml:16,19,24,50,55,61` — `actions/checkout@v4`, `actions/setup-python@v5`,
  `astral-sh/setup-uv@v3`), and the runner is already warning that these target deprecated Node 20.
  Relevant given R1: this package's install path fetches and `torch.load`s third-party checkpoints.
- **Fix:** add `SECURITY.md` (supported versions + private reporting via GitHub Security Advisories),
  `CODE_OF_CONDUCT.md`, a bug-report issue template, `.github/dependabot.yml` for `pip` +
  `github-actions`, and a `pip-audit` step in `ci.yml`. Pin actions to SHAs.

### R12. No deployment/ops guide and no API-stability policy — the two documents a production adopter needs most

- **Symptom:** `docs/user-guide.md` is 2,700+ lines and genuinely thorough on *usage*, but contains
  zero guidance on *operating* the library.
- **Evidence:** `grep -i "docker\|kubernetes\|systemd\|prometheus\|deploy\|upgrade\|deprecat\|SLO"`
  over `docs/user-guide.md` and `README.md` returns no operational content — the only "monitoring"
  hits are `### 7.6 Stream Health Monitoring` (`docs/user-guide.md:1259`, an in-process API) and a
  warehouse *example* (`docs/user-guide.md:2380`).
  This despite `src/yowo/cli/_main.py:1085-1098` shipping `yowo metrics --format prometheus` and a
  `yowo health` command — production-grade features that no document tells anyone how to wire up.
  There is no `RELEASING.md`, no documented SemVer/deprecation policy, and no statement of what
  counts as public API (an `api-contract-guardian` agent exists in the agent roster, but its
  contract is nowhere in the repo docs).
- **Fix:** add `docs/deployment.md` (container base images per backend, Jetson/JetPack notes, systemd
  unit, Prometheus scrape config for `yowo metrics`, `yowo health` as a liveness probe, resource
  sizing) and a short `## API stability` section in the README declaring what is public, the SemVer
  contract, and the deprecation window (e.g. one minor release with a `DeprecationWarning`).

### R13. Integration tests and every optional extra are never exercised anywhere

- **Symptom:** CI runs `tests/unit/` only. The 33 integration tests and all seven extras are
  untested in any automated environment.
- **Evidence:** `.github/workflows/release.yml:38-39` — `uv run pytest tests/unit/ -x -q`.
  `uv run pytest tests/integration --collect-only -q` -> **33 tests collected**, never run in CI.
  `uv run pytest tests/unit --collect-only -q` -> **2142 tests collected**, with
  `SKIPPED tests/unit/test_chroma_gallery.py:11: could not import 'chromadb'` — the `chromadb`
  extra (`pyproject.toml:42-44`) is not in the dev group (`pyproject.toml:56-76`), so its tests
  skip silently on every run including CI. Same for `openvino`, `coremltools`, `tensorrt`: none
  appear in `[dependency-groups] dev`. An extras matrix that resolves only on the author's machine
  is the definition of an untested install path.
- **Fix:** add an `extras` matrix job to `ci.yml` that, per extra, does
  `uv pip install ".[<extra>]"` + `python -c "import yowo; ..."` + the extra's tests; run
  `tests/integration/` on a nightly `schedule:` trigger.

### R14. LICENSE has an unfilled copyright placeholder and there is no NOTICE

- **Symptom:** `LICENSE:189` reads `Copyright [yyyy] [name of copyright owner]` — the Apache-2.0
  appendix boilerplate was never filled in, and no copyright line appears anywhere else in the repo.
- **Why it matters:** Apache-2.0 sections 4(c)/(d) presume attribution notices; a licensor that
  names nobody weakens the grant and fails most corporate license-scanner checks. Compounded by R1.
- **Fix:** replace with `Copyright 2026 Tin Dang` and add a `NOTICE` file that also records the
  third-party weight provenance and its AGPL-3.0 terms.

---

## P2 — later

- **R15.** `pyproject.toml:144` `template_dir = ".github/templates"` is dead config — the directory
  does not exist. Delete it. Relatedly, the CI log shows PSR warning that
  `changelog.changelog_file` (`pyproject.toml:130,145`) "is moving to
  `changelog.default_templates.changelog_file` … compatibility will break in v10"; migrate now while
  pinning (R3).
- **R16.** `.gitignore` is two concatenated GitHub Python templates (lines 1-160 and 161-374, with
  `__pycache__/`, `build/`, `.coverage`, `.env` etc. each declared twice). Deduplicate, and add
  `.DS_Store`, `*.bak`, `.add/`, `*.pt` — all four are currently untracked-but-shippable
  (`git ls-files --others --exclude-standard` -> 24 entries incl. `.DS_Store`, `CLAUDE.md.bak`,
  `yolo11n.pt`). Note `.planning/` is ignored at `.gitignore:371` yet two files under it are already
  tracked (`git ls-files .planning` -> `STATE.md`, `260317-hyd-SUMMARY.md`) — ignore rules do not
  apply to already-tracked files.
- **R17.** Even after R2, decide deliberately whether `CLAUDE.md`, `CONTEXT.md`, `docs/plans/*`
  (design prompts for AI agents) and `docs/experiments/*` belong in a published sdist. They
  currently ship. `docs/Evolutionary Object .md` is a stray, half-named, 20 KB tracked file — rename
  or remove.
- **R18.** No lower-bound verification: `numpy>=1.24`, `opencv-python-headless>=4.8`, `torch>=2.0`
  etc. have never been resolved at their floors (`uv.lock` pins numpy 2.0.2/2.2.6/2.3.5 by Python
  version — i.e. only numpy 2.x is ever tested despite a `>=1.24` claim). Add a
  `uv pip install --resolution lowest-direct` CI job before promising those ranges.
- **R19.** `CONTRIBUTING.md:35` says "Requirements: Python 3.11+" while the package claims 3.8+.
  Harmless for contributors, but state the distinction explicitly.
- **R20.** No hosted API reference (the module `README.md` files shipped in the wheel are good raw
  material for mkdocs/pdoc), no `CITATION.cff`, no `RELEASING.md` runbook.
- **R21.** `Metadata-Version: 2.5` with PEP 639 `License-Expression` (from
  `pyproject.toml:6` `license = "Apache-2.0"`) requires a recent pip/packaging. Given the stated
  Jetson/JetPack audience — which frequently carries an old system pip — smoke-test installation
  with the oldest pip you intend to support, or fall back to a classifier-only license declaration.

---

## What is already good

Do not re-do this work:

- **The quality gate's *content* is right.** `.github/workflows/release.yml:29-40` runs
  `ruff check` + `ruff format --check` + `pyright` (strict, `pyproject.toml:102`) + `pytest` with a
  per-test `--timeout=60` and a 15-minute step cap. It is only its *trigger* (R5) and its
  *downstream* (R3/R4) that are broken — the gate itself passed on all four failing runs.
- **`.pre-commit-config.yaml:1-21`** runs the identical three gates locally, so contributors get the
  same signal the (post-merge) CI gives.
- **The wheel is clean and correct:** 295 KB, 107 files, `yowo/py.typed` present, per-module
  `README.md` files shipped for in-IDE docs, `LICENSE` in `dist-info/licenses/`,
  `entry_points.txt` wiring `yowo = yowo.cli._main:cli` (`pyproject.toml:52-53`). Only the *sdist*
  is unbounded.
- **Clean-environment failure behaviour is excellent.** Installed with core deps only, yowo raises
  a typed `yowo.errors.DependencyError` listing every backend and the exact install command for
  each — far better than the bare `ImportError` most libraries give. (Verified in an isolated 3.11
  venv built from the current wheel.)
- **The public API matches the README.** Every symbol the README imports —
  `InferenceEngine, open_source, classify, ModelFamily, ModelSize, InferenceConfig, load_config,
  export_model`, the six documented error classes, `yowo.types.FrameDropPolicy` — imports
  successfully from a core-only install. The documented surface is real.
- **No committed secrets.** `git grep` across all tracked files for AWS keys, `ghp_`/`gho_` tokens,
  `sk-` keys, PEM private keys and `pypi-AgEIcHlwaS5vcmc` macaroons returns nothing. `.env` is
  ignored (`.gitignore:123-124`, `:298`) and no `.env` exists.
- **Commit discipline is genuine.** Conventional Commits are used consistently
  (`git log --oneline -30`), `[tool.semantic_release.commit_parser_options]`
  (`pyproject.toml:137-141`) maps them to SemVer correctly, and `tag_format`,
  `concurrency: release`, `environment: release` and `[skip ci]` on the release commit
  (`pyproject.toml:135`, `release.yml:46-47`) are all configured the right way. The scaffolding is
  sound; it is the pinning and the publish step that are missing.
- **Someone thought hard about the edge matrix.** The environment markers at
  `pyproject.toml:35-38` (`onnxruntime-gpu<1.20` below 3.10), `:41` (scipy gated to 3.9+) and
  `:68-70` are real, correct constraints for Jetson/JetPack — the gap is that nothing tests them.
- **`CONTRIBUTING.md` is strong** (294 lines: dependency graph, layering rules, 700-line file cap,
  test requirements, PR process) and the CHANGELOG entries that *do* exist are high quality,
  Keep-a-Changelog formatted, and specific (e.g. `CHANGELOG.md:18-50`).
- **`docs/user-guide.md` is substantial and accurate** — 16 sections covering CLI, Python API,
  backend selection, streaming, multi-stream, tracking, cross-camera ReID, counting and annotation,
  with measured numbers. The gap is operational documentation, not usage documentation.
