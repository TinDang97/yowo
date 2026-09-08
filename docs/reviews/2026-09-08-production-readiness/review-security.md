# Security & supply-chain review — yowo production readiness

## Threat model used

`yowo` is a **library**, not a service. The realistic adversary does not get a request handler; they
get one or more of the following inputs that an operator hands to yowo:

1. **Model artifacts** — a `.pt` / `.onnx` / `.engine` / `.mlpackage` file, either downloaded by yowo
   itself or supplied by the operator.
2. **Config** — a YAML file and `YOWO_*` environment variables.
3. **Media** — camera streams (RTSP), video files, image directories.
4. **The distribution channel** — the PyPI package other teams `pip install`.

Findings are calibrated to that. Where something is only reachable by an operator who already
controls the process (env vars, explicit paths), I say so and rate it down. Where an attacker who
controls a *network path* or a *file on disk* gets code execution, that is a P0 regardless of how
"unlikely" the deployment is, because the library gives the operator no way to defend against it.

## Verdict

Not production-ready on this lane. Two independent problems each block launch on their own: a **live
PyPI publish token sitting in plaintext, world-readable, on the developer's disk**, and a
**download-then-`torch.load(weights_only=False)` path with no integrity check whatsoever** — a
network attacker or anyone with write access to `~/.cache/yowo` gets arbitrary code execution in the
inference process. Everything else in this report is downstream of one root cause: **the project has
never had a security pass.** There is no `SECURITY.md`, no threat model, no dependency audit, no
secret scanning, no CI on pull requests, and **zero security tests among 95 test files**
(`grep -rn --include='*.py' -iE "traversal|malicious|checksum|redact" tests/ | wc -l` -> `0`).

The good news: the code itself is *clean* in the ways libraries usually are dirty. No `pickle`, no
`eval` / `exec`, no `os.system`, no `shell=True`, no `yaml.load` — every YAML call is `safe_load`.
The gaps are architectural (no integrity chain for artifacts) and operational (credential handling,
CI hygiene), not sloppy coding.

---

## P0 — blocks launch

### P0-1. Live PyPI publish token in plaintext, world-readable, on disk

- **Symptom:** `/Users/tindang/workspaces/tind-repo/yowo/.env` contains a single key,
  `UV_PUBLISH_TOKEN`, whose value is a **PyPI API token** (prefix `pypi-`, 201 characters).
  File mode is `0644` (`stat -f '%Sp %Su' .env` -> `-rw-r--r-- tindang`), i.e. readable by every
  local user, every process running as that user, every editor plugin, every CI runner that mounts
  the directory, and every AI coding agent given repo access. I read the key name and value length
  in the course of this review without any prompt or gate — which is precisely the exposure.
- **Evidence:**
  - `.env:1` — `UV_PUBLISH_TOKEN=<pypi- token, 201 chars>` (value deliberately not reproduced).
  - Mode `0644`, owner `tindang`.
  - **Not in git, and never was** — verified:
    - `git ls-files | grep -iE '(^|/)\.env'` -> no match.
    - `git log --all --oneline -- .env` -> empty.
    - `.gitignore:123` and `.gitignore:298` both list `.env`.
    - Pickaxe sweep over all 281 commits for `pypi-`, `ghp_`, `github_pat_`, `AKIA`,
      `BEGIN RSA`, `BEGIN OPENSSH` -> **0 commits each**. (`hf_` / `sk-` hits are false positives:
      `task-agnostic`, `task-specific`, etc.)
  - No reference to `UV_PUBLISH_TOKEN` or `uv publish` exists anywhere else in the repo
    (`grep -rn "UV_PUBLISH\|uv publish\|twine\|PYPI" . --exclude-dir=.git --exclude-dir=.venv` ->
    only `.env`), and `.github/workflows/release.yml` has **no PyPI publish step** — so publishing
    is an undocumented manual act from this laptop using this long-lived credential.
  - `dist/yowo-2.4.1-py3-none-any.whl` and `dist/yowo-2.4.1.tar.gz` confirm local builds are
    being cut here.
- **Why it blocks:** This single token is the entire trust anchor for every team that will run
  `pip install yowo`. Anyone who reads that file publishes a malicious `yowo` release under the
  maintainer's name — and because yowo's own weight loader executes pickles (P0-2), a poisoned
  release is a fleet-wide RCE, not just a bad install. There is no scoping, no expiry, no audit
  trail, and no second factor between a stray `cat .env` and a supply-chain compromise. The
  mitigating fact is that it is *not* in git history, so no history rewrite is needed.
- **Smallest fix:**
  1. **Rotate now.** Revoke the token at pypi.org -> Account settings -> API tokens. Assume it is
     burned; it has been sitting at mode `0644` and has been read by at least one automated agent.
  2. **Delete `.env`.** Do not replace it with another plaintext token.
  3. **Move publishing into CI with PyPI Trusted Publishing (OIDC)** — the workflow already grants
     `id-token: write` (`.github/workflows/release.yml:9`), so the plumbing is half there. Add a
     `publish` job gated on `environment: release` that runs `uv publish --trusted-publishing always`,
     and configure the pending publisher on PyPI for `owner/yowo` + `release.yml` + `release`
     environment. This eliminates the long-lived credential entirely.
  4. If a manual fallback is still wanted, keep the token in the OS keychain
     (`uv publish --token $(security find-generic-password -w -s yowo-pypi)`), never a file.
  5. Add `.env` to a `gitleaks` / `trufflehog` pre-commit hook so the next one is caught before it
     ever lands (`.pre-commit-config.yaml` currently has ruff/pyright/pytest only, no secret scan).

### P0-2. Remote weights are downloaded without any integrity check and then unpickled with `weights_only=False`

- **Symptom:** yowo fetches a `.pt` file over the network and hands it straight to
  `torch.load(..., weights_only=False)`. There is no SHA-256, no signature, no size check, no
  provenance check, and no cache-integrity check. `weights_only=False` is *explicit opt-in to
  arbitrary code execution during unpickling* — a crafted checkpoint runs attacker code with
  the full privileges of the inference process the moment `resolve_weights()` -> `load_weights()`
  runs, before a single frame is processed.
- **Evidence:**
  - `src/yowo/arch/_weights.py:86` —
    `ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)`
    Reached from all three loaders: `load_weights` (`:130`), `load_classify_weights` (`:262`),
    `load_obb_weights` (`:346`).
  - `src/yowo/models/_weights.py:61` — `_download(meta.default_weights_url, dest)`; `:86` —
    `os.replace(tmp_path, dest)`. Between download and rename there is **no verification step of any
    kind**. `_attempt_download` (`:105-149`) reads the body and writes it; that is all.
  - `src/yowo/models/_weights.py:113` — `requests.get(url, stream=True, timeout=60)`. TLS
    verification is on (good), but `allow_redirects` defaults to `True`, so a redirect off
    `github.com` to any other host is followed silently and the body is accepted as authoritative.
  - `src/yowo/models/_weights.py:58-59` — **cache hit short-circuits everything**:
    `if dest.exists(): return dest`. Any file placed at
    `~/.cache/yowo/weights/{family}/{size}/{stem}.pt` is loaded and unpickled with zero validation.
  - `src/yowo/models/_weights.py:55` — the cache root is `YOWO_CACHE_DIR`-overridable, and
    `dest.parent.mkdir(parents=True, exist_ok=True)` (`:77`) creates it with the **ambient umask** —
    on a container with `umask 000` this is a world-writable directory that any local user can drop
    a payload into.
  - `src/yowo/models/_registry.py:45-50` — `register()` is **public API**
    (`src/yowo/models/__init__.py:15-23` exports it) and overwrites builtins for the same
    `(family, size)` key. It accepts an arbitrary `default_weights_url` string. `_download` never
    validates the scheme, so a plain `http://` URL registered by any in-process code (or a
    dependency) downgrades the fetch to a MITM-able channel that ends in `torch.load`.
  - No test anywhere covers this:
    `grep -rln --include='*.py' -iE "weights_only|checksum|sha256|malicious" tests/` -> no files.
- **Why it blocks:** This is the exact failure mode the brief names, and it is present in its worst
  form — *unverified download* plus *unpickle-with-code-execution* plus *unverified cache reuse*.
  The library gives an operator no mechanism to pin, verify, or air-gap weights. A hostile mirror, a
  compromised GitHub release asset, a DNS hijack, a redirect, a malicious co-tenant on the host, or
  a poisoned container layer all convert directly into RCE inside the inference process, which on an
  edge deployment is typically the most network-exposed component on the device. The first thing a
  security reviewer at any adopting company will ask is "how do you verify the weights", and today
  the answer is "we don't".
- **Smallest fix (in priority order — the first two are the real fix):**
  1. **Pin a SHA-256 per registered model.** Add `sha256: str` to `ModelMeta`
     (`src/yowo/models/_registry.py:31-37`), populate it for all 25 builtin registrations, and in
     `_download` verify the digest of the temp file *before* `os.replace` at
     `src/yowo/models/_weights.py:86`. Also verify on cache hit at `:58` (cheap: hash once, store a
     `.sha256` sidecar, re-verify on load). Fail closed with a distinct error.
  2. **Try `weights_only=True` first.** At `src/yowo/arch/_weights.py:86`, attempt
     `torch.load(..., weights_only=True)`; only fall back to `weights_only=False` when the
     checkpoint's digest matched a pinned hash *or* the caller passed an explicit
     `trust_remote_code=True`-style opt-in. Ultralytics checkpoints store a pickled `nn.Module` so
     `weights_only=True` will not work for stock `.pt` files — which is exactly why the pinned hash
     in step 1 must gate the unsafe path. Document loudly that a third-party `.pt` is executable
     code.
  3. Reject non-`https` URLs in `_download` and re-check the final response URL's host against an
     allowlist (`github.com`, `objects.githubusercontent.com`) after redirects.
  4. `dest.parent.mkdir(..., mode=0o700)` and `os.chmod` the cache root so a co-tenant cannot
     plant a checkpoint.

### P0-3. RTSP credentials propagate into result JSON, exception messages and logs — with a docstring that promises the opposite

- **Symptom:** `RTSPStreamSource` stamps the **full RTSP URL** onto every `Frame` as `source_id`.
  Operators overwhelmingly use `rtsp://user:password@host/stream`. That credentialed string is then
  (a) embedded in `SourceError` / `SourceTimeoutError` messages that land in tracebacks and any
  error reporter, and (b) written verbatim into the detection JSON that `write_json` persists —
  under a docstring that explicitly states the output is *"safe for logging and wire transport"*.
- **Evidence:**
  - `src/yowo/io/_source.py:336` — `source_id=self._url` on every yielded `Frame`.
  - `src/yowo/io/_source.py:312` — `raise SourceError(f"Cannot open RTSP stream: {self._url}")`.
  - `src/yowo/io/_source.py:350` —
    `raise SourceTimeoutError(f"RTSP stream {self._url} timed out after ...")`.
  - `src/yowo/types.py:314-329` — `Detection.to_dict()` returns `"source_id": self.frame.source_id`.
    The docstring at `:315-320` claims: *"Pixel data (`frame.pixels`) and local paths
    (`model_spec.weights_path`) are excluded — only portable metadata is included so the result is
    safe for logging and wire transport."* It excludes the local weights path and **includes the
    camera password**.
  - `src/yowo/io/_sink.py:18-38` — `write_json` serialises exactly that dict to a file on disk.
  - `src/yowo/cli/_main.py:1046-1052` — `_write_json` does the same for CLI output.
  - `src/yowo/logging.py:21-28` — `JsonFormatter` emits a `stream_id` field into structured logs
    destined for aggregation pipelines.
  - **There is no redaction anywhere in the codebase:**
    `grep -rn --include='*.py' -iE "redact|sanitiz|mask_|scrub" src/` -> **no matches**.
- **Why it blocks:** This is a credential-disclosure primitive that fires on the library's *happy
  path*, not on an edge case. In the deployment yowo is built for — an edge box watching N cameras,
  shipping detections to a dashboard and logs to a SIEM — every camera password ends up in a
  third-party log store and in every result artifact. Recovery is expensive (rotate every camera
  credential across a fleet) and detection is unlikely because nothing looks wrong. The false
  "safe for logging and wire transport" guarantee makes it worse: a downstream integrator who reads
  the docstring and forwards `to_dict()` output is doing exactly what the library told them to do.
- **Smallest fix:** Add one helper, `yowo/utils/_redact.py::redact_url(url: str) -> str`, that
  parses with `urllib.parse` and replaces `netloc` userinfo with `***:***@`. Apply it in exactly
  four places:
  - `src/yowo/io/_source.py:336` — set `source_id=redact_url(self._url)` (keep the raw URL private
    on the instance for `cv2.VideoCapture`; nothing downstream needs the credential).
  - `src/yowo/io/_source.py:312` and `:350` — redact in the message strings.
  - Fix the `Detection.to_dict` docstring at `src/yowo/types.py:315-320` to state what is and is not
    scrubbed.
  Add a regression test asserting `rtsp://u:p@h/s` never appears in `to_dict()`, `write_json`
  output, or either exception message.

### P0-4. The published sdist ships an unreviewed 5.6 MB binary pickle scraped from the developer's working tree — and it is an AGPL-3.0 artifact inside an Apache-2.0 package

- **Symptom:** `yowo-2.4.1.tar.gz` contains `yolo11n.pt` — an ultralytics YOLO11-nano checkpoint
  that happens to be sitting untracked in the repo root. `pyproject.toml` declares no sdist
  include/exclude list, so hatchling's default (everything not VCS-ignored) sweeps up whatever the
  maintainer left lying around at build time. Two separate problems ride on that: an unreviewed
  executable-pickle blob in the distribution, and an AGPL-3.0 licensed asset redistributed inside a
  package that presents itself as Apache-2.0.
- **Evidence:**
  - `tar -tzvf dist/yowo-2.4.1.tar.gz | sort -k3 -nr | head -1` ->
    `5613764  yowo-2.4.1/yolo11n.pt` (largest file in the 245-file sdist; the whole sdist is 6.0 MB
    and this one blob is 5.6 MB of it).
  - `git status --porcelain` -> `?? yolo11n.pt` — untracked, and *not* matched by any `.gitignore`
    rule, which is exactly why it was picked up.
  - `pyproject.toml:87-88` — only `[tool.hatch.build.targets.wheel] packages = ["src/yowo"]` is
    declared. There is **no** `[tool.hatch.build.targets.sdist]` stanza.
  - The wheel is clean (`unzip -l dist/yowo-2.4.1-py3-none-any.whl` -> only `.py`, `README.md`,
    `py.typed`, dist-info) — so this is specifically an sdist-hygiene failure.
  - `.env` was correctly excluded (it is gitignored) — confirming the *only* thing standing between
    a secret and the published sdist is whether someone remembered a `.gitignore` line.
  - Licensing: `pyproject.toml:6` `license = "Apache-2.0"`; `LICENSE:1` Apache License 2.0;
    `README.md:801-803` "Apache-2.0 — see LICENSE". Meanwhile
    `src/yowo/models/_registry.py:14,108-109` point every default download at
    `https://github.com/ultralytics/assets/releases/download/v8.{3,4}.0/`, and ultralytics' YOLO11
    model assets are released under **AGPL-3.0** (with a paid Enterprise license as the alternative).
    `README.md` never mentions AGPL, ultralytics licensing, or the weights' provenance.
    `pyproject.toml:73` additionally pulls `ultralytics>=8.4.21` as a dev dependency.
- **Why it blocks:** The sdist half is a supply-chain integrity failure — the published artifact's
  contents are a function of the maintainer's untidy working directory, and the blob in question is
  a `.pt` pickle that yowo itself will execute (P0-2). Anything that lands in that directory ships.
  The licensing half is a hard blocker for the stated goal ("a library other teams deploy"): every
  prospective adopter's legal review will find an Apache-2.0 package that redistributes an AGPL-3.0
  binary and that, at runtime and by default, downloads more AGPL-3.0 binaries — with no notice
  anywhere in the README or LICENSE. AGPL's network-use clause is the specific thing corporate legal
  teams refuse. This kills adoption at the review stage, quietly.
- **Smallest fix:**
  1. Add an explicit allowlist so the sdist can never pick up stray files:
     ```toml
     [tool.hatch.build.targets.sdist]
     include = ["src/yowo", "tests", "README.md", "LICENSE", "CHANGELOG.md", "pyproject.toml"]
     ```
     Add `*.pt`, `*.onnx`, `*.engine`, `*.mlpackage` to `.gitignore`, and delete the root
     `yolo11n.pt`. Yank or re-release `2.4.1` if it reached PyPI with the blob.
  2. Add a **"Model weights and licensing"** section to `README.md` stating plainly: yowo's *code*
     is Apache-2.0; the default weights are ultralytics assets under **AGPL-3.0**; commercial users
     must either obtain an ultralytics Enterprise license or supply their own weights via
     `--weights` / `ModelSpec.weights_path`. Same note in `docs/user-guide.md`.
  3. Consider making the auto-download opt-in (`YOWO_ALLOW_WEIGHT_DOWNLOAD=1` or an explicit
     `download=True`) so a company that has not accepted AGPL never silently pulls an AGPL artifact.
     This composes cleanly with the P0-2 hash-pinning fix.

---

## P1 — before GA

### P1-1. Release workflow grants `contents: write` + `id-token: write` to a job that runs ~50 third-party build hooks

- **Symptom:** `permissions:` is declared at **workflow** scope, so it applies to *both* jobs. The
  `quality` job runs `uv sync --group dev` — which triggers arbitrary build backends and setup hooks
  from every dev dependency — while holding a `GITHUB_TOKEN` that can push to the repo and mint OIDC
  identity tokens.
- **Evidence:** `.github/workflows/release.yml:7-9` (workflow-level `contents: write`,
  `id-token: write`); `:12-14` `quality` job declares no `permissions:` override; `:27`
  `run: uv sync --group dev`; `:39` `uv run pytest`. Dev group at `pyproject.toml:56-76`.
- **Why it blocks GA:** Classic least-privilege violation. One compromised transitive dev dependency
  (the `ultralytics` / `torch` / `onnx` tree is large) gets repository write and, once P0-1's fix
  lands, the ability to mint a PyPI OIDC token. The trigger is `push: branches: [main]` only, so
  forks cannot reach it — that is the mitigating factor and the reason this is P1 not P0.
- **Smallest fix:** Move permissions to job scope. `quality:` gets `permissions: {contents: read}`;
  `release:` gets `permissions: {contents: write, id-token: write}`. Delete the top-level block.

### P1-2. Every GitHub Action and the release tool itself are floating, mutable references

- **Symptom:** Actions are pinned to mutable tags, and `python-semantic-release` — which runs with
  `GH_TOKEN` and `contents: write` — is resolved fresh at every release with no version constraint
  at all.
- **Evidence:** `.github/workflows/release.yml:16` `actions/checkout@v4`; `:19`
  `actions/setup-python@v5`; `:24` `astral-sh/setup-uv@v3` (third-party); `:50`, `:56`, `:61` same
  again in the release job; `:68-69`
  `uv tool run --from python-semantic-release semantic-release version|publish` — no version pin.
- **Why it blocks GA:** A tag is a pointer the action author (or anyone who compromises them) can
  repoint. The `tj-actions/changed-files` incident in March 2025 was exactly this: retagged `@v35`,
  secrets exfiltrated from thousands of repos. `astral-sh/setup-uv` is third-party and sits in the
  release path. `uv tool run` with no pin means the tool that writes your version, tags your repo,
  and uploads your release assets is whatever PyPI serves that minute.
- **Smallest fix:** Pin every `uses:` to a full 40-char commit SHA with a `# v4.2.2`-style trailing
  comment, and pin the tool:
  `uv tool run --from 'python-semantic-release==9.x.y' semantic-release`. Add Dependabot with
  `package-ecosystem: github-actions` to keep the SHAs moving on a schedule instead of silently.

### P1-3. No PR-triggered CI, no dependency audit, no secret scanning, no SBOM

- **Symptom:** `.github/` contains exactly one file. Nothing runs on a pull request. Nothing ever
  checks dependencies for known vulnerabilities. Nothing scans for committed secrets. No SBOM is
  produced for the published artifact.
- **Evidence:** `find .github -type f` -> `.github/workflows/release.yml` only. No
  `dependabot.yml`, no `codeql.yml`, no `SECURITY.md`
  (`find . -maxdepth 2 -iname "SECURITY*" -not -path "./.venv/*"` -> empty).
  `.pre-commit-config.yaml:1-24` — ruff, ruff-format, pyright, pytest; no `gitleaks`, no
  `detect-secrets`.
- **Why it blocks GA:** The quality gate only runs *after* merge to main, so a bad change is already
  on the default branch before anything checks it. With no `pip-audit` / `osv-scanner`, a published
  CVE in `requests`, `opencv-python-headless`, `torch`, or `onnxruntime` will sit unnoticed
  indefinitely — there is no mechanism that would ever surface it. Enterprise adopters increasingly
  require an SBOM as a procurement gate.
- **Smallest fix:** Add `.github/workflows/ci.yml` on `pull_request` running the same four quality
  steps plus `uv run pip-audit` (or `osv-scanner`). Add `.github/dependabot.yml` for `pip` and
  `github-actions`. Add a `gitleaks` hook to `.pre-commit-config.yaml`. Add a `SECURITY.md` with a
  disclosure address and a response-time commitment. Emit a CycloneDX SBOM in the release job and
  attach it to the GitHub release (`upload_to_vcs_release` is already `true`,
  `pyproject.toml:133`).

### P1-4. `requires-python = ">=3.8"` ships support for interpreters that no longer receive security patches

- **Symptom:** The package advertises and accepts Python 3.8 and 3.9, both past end-of-life
  (3.8 EOL Oct 2024; 3.9 EOL Oct 2025 — today is 2026-09-08). CPython no longer issues security
  fixes for either.
- **Evidence:** `pyproject.toml:7` `requires-python = ">=3.8"`; `:14-15` classifiers for
  `Programming Language :: Python :: 3.8` and `3.9`; `pyproject.toml:91`
  `[tool.ruff] target-version = "py38"`. Note the internal inconsistency: `pyproject.toml:79-81`
  `[tool.uv] environments = ["python_version >= '3.9'"]` and `pyproject.toml:100`
  `[tool.pyright] pythonVersion = "3.11"` — the project does not actually verify 3.8 anywhere, and
  the single CI job runs 3.11 only (`.github/workflows/release.yml:21`).
- **Why it blocks GA:** Advertising 3.8 support tells an operator it is safe to deploy yowo on an
  unpatched interpreter — a meaningful claim for edge devices, which are exactly where stale Python
  lives. It is also an untested claim: nothing in CI exercises 3.8 or 3.9.
- **Smallest fix:** Raise to `requires-python = ">=3.10"` (or `>=3.9` if a shipped device forces it,
  with a dated deprecation note), drop the dead classifiers, set `target-version = "py310"`, and add
  a CI matrix over the versions actually claimed.

### P1-5. ONNX / TensorRT artifacts are deserialised with no validation, and the TRT engine cache is a poisoning primitive

- **Symptom:** ONNX models and TensorRT engines are handed straight to the runtime. Worse, the
  TensorRT EP is configured to read and write its compiled-engine cache **in the model file's own
  directory** — so a `.engine` planted next to a model is silently trusted and loaded in preference
  to recompiling.
- **Evidence:**
  - `src/yowo/backends/_tensorrt.py:146-147` —
    `"trt_engine_cache_enable": True, "trt_engine_cache_path": str(Path(model_path).parent)`.
  - `src/yowo/backends/_tensorrt.py:158` and `src/yowo/backends/_onnx.py:135` —
    `ort.InferenceSession(str(model_path), ...)` with no prior validation.
  - `src/yowo/backends/_openvino.py:111-114` — `core.read_model(str(xml_path))` /
    `core.compile_model(...)`; `src/yowo/backends/_coreml.py:86` — `ct.models.MLModel(...)`.
- **Why it blocks GA:** NVIDIA documents engine deserialisation as a trusted-input-only operation;
  a serialised plan is not a safe format. ONNX protobuf parsers have a CVE history. Pointing the
  engine cache at a directory the operator may not control (a shared model volume, an NFS mount, a
  container layer) means whoever can write there controls what the GPU executes. This is P1 rather
  than P0 because it needs filesystem write access to the model directory, which is a higher bar
  than P0-2's network path.
- **Smallest fix:** Default `trt_engine_cache_path` to a private per-user directory
  (`~/.cache/yowo/trt/<sha256 of model>/`, mode `0700`) rather than the model's parent, and make it
  configurable. Document in `src/yowo/backends/README.md` that `.onnx` / `.engine` / `.mlpackage`
  inputs are trusted code, not data.

### P1-6. `register()` lets any in-process code repoint a builtin model at an arbitrary URL

- **Symptom:** `register()` is exported public API, silently overwrites builtin entries, and
  performs no validation on `default_weights_url`. A dependency, a plugin, or a config-driven
  extension point can repoint `yolo11n` at `http://attacker/yolo11n.pt`, and `_download` will fetch
  it over cleartext and `torch.load` it.
- **Evidence:** `src/yowo/models/_registry.py:45-50` (`register` — docstring even says *"Overwrites
  any existing entry"*); `src/yowo/models/__init__.py:15-23` exports it;
  `ModelMeta.default_weights_url` is a bare `str` (`src/yowo/models/_registry.py:37`);
  `src/yowo/models/_weights.py:113` passes it unvalidated to `requests.get`.
- **Why it blocks GA:** It is the intended extension point, so it will be used — and it currently
  offers no safe way to add a model. It is P1 not P0 because reaching it requires in-process code
  execution, which an external attacker does not have by default. Fixing it is nearly free and it
  closes the "hostile mirror" leg of P0-2 permanently.
- **Smallest fix:** In `register()`, reject any `default_weights_url` whose scheme is not `https`,
  and require the `sha256` field added in P0-2's fix. Log a warning when an existing key is
  overwritten.

### P1-7. Optional dependency floors are "what was current when I wrote it", not security floors

- **Symptom:** `torch>=2.0` admits every 2.x release, including versions with published advisories
  affecting `torch.load` itself — the exact call site of P0-2. No upper bounds, no audit, no
  mechanism to notice.
- **Evidence:** `pyproject.toml:32` `pytorch = ["torch>=2.0"]`; `:33` `export = ["torch>=2.0", ...]`;
  `:65` dev `torch>=2.0`. Also `:24` `opencv-python-headless>=4.8` — the decoder that ingests
  untrusted media (`cv2.imread`, `cv2.VideoCapture`) and whose 4.8.0-era builds carry known image
  codec issues. No `pip-audit` anywhere in CI or pre-commit.
- **Why it blocks GA:** I am flagging the *absence of a mechanism*, not asserting specific CVE
  applicability — that is precisely the point: nobody can tell today whether a yowo install is
  exposed, because nothing checks. For `torch` in particular, floors below 2.6 are worth verifying
  against the `torch.load` advisories given `weights_only=False` at
  `src/yowo/arch/_weights.py:86`.
- **Smallest fix:** Run `uv run pip-audit` in CI (P1-3) and let its output set the floors, rather
  than guessing. Raise `torch>=2.6` in the `pytorch` / `export` extras unless an edge target
  provably cannot supply it, and raise the `opencv-python-headless` floor to the current patch line.

---

## P2 — later

### P2-1. Environment variables steer file paths into `torch.load` — theoretical in this threat model

`src/yowo/config.py:427` (`YOWO_WEIGHTS_PATH` -> `cfg.weights_path`) and
`src/yowo/models/_weights.py:55` (`YOWO_CACHE_DIR` -> cache root) let the environment pick which
file gets unpickled. **This is not independently exploitable** in yowo's threat model: whoever sets
the environment already controls the process. It matters only as defence-in-depth against
environment-injection bugs in a *host* application (a container orchestrator that forwards
user-controlled env, a CGI-style deployment). Worth a one-line note in the config docs that
`YOWO_WEIGHTS_PATH` is security-sensitive; not worth code changes.

### P2-2. Cache directories are created with the ambient umask

`src/yowo/models/_weights.py:77`, `src/yowo/tune/_profile.py:124`, `src/yowo/cache/_store.py:59,124`
all use `mkdir(parents=True, exist_ok=True)` with no `mode=`. On a standard `umask 022` host this
yields `0755` — readable but not writable by others, which is fine. Under `umask 000` (seen in some
minimal container images) it becomes world-writable, which upgrades P0-2's cache-poisoning leg from
"needs the same UID" to "needs any UID". Fix: pass `mode=0o700` on the cache roots. Subsumed by the
P0-2 fix if that is done properly.

### P2-3. `FeatureStore.__init__` recursively deletes subdirectories of a caller-supplied path at construction time

`src/yowo/cache/_store.py:59-63` — constructing a `FeatureStore` with `cache_dir=X` immediately
`shutil.rmtree`s every subdirectory of `X`. It is guarded: only names matching
`^[0-9a-f]{16}$` (`:165-170`) are removed, so ordinary directories survive. Still, destructive I/O
in a constructor is surprising, and a caller who points this at a shared directory loses data with
no confirmation. Fix: move the sweep into an explicit `.reset()` / `clear_on_start=True` flag.

### P2-4. Export sidecar metadata leaks absolute local paths into shipped artifacts

`src/yowo/export/_exporter.py:199` — `source_weights=str(weights_path)` and `:194`
`file_path=str(exported_path.resolve())` are written into the `.json` metadata that accompanies an
exported model. If that metadata travels with the model to a customer, it discloses the build
machine's directory layout and username. Minor information disclosure. Fix: store the basename, or
the P0-2 SHA-256, instead of the absolute path.

### P2-5. Subprocess helpers resolve binaries via `PATH`

`src/yowo/hardware/_detect.py:89-95` (`sysctl`), `:101-107` (`vm_stat`), `:298-309` (`nvidia-smi`)
all invoke unqualified binary names. The calls are otherwise **exemplary** — argv lists (never a
string), `shell=False` by default, `timeout=5`, `check=False`. The only residual exposure is `PATH`
hijack on an already-misconfigured host, which implies prior compromise. Noted for completeness;
not worth changing.

### P2-6. Zero security tests

95 test files, none covering any security property.
`grep -rn --include='*.py' -iE "traversal|malicious|checksum|redact" tests/ | wc -l` -> `0`;
`grep -rln --include='*.py' -iE "weights_only|credential|sha256" tests/` -> no files. Every fix
above should land with a regression test — in particular: checksum-mismatch rejection (P0-2),
credential redaction (P0-3), and sdist-contents assertion (P0-4).

---

## What is already good

Do not spend roadmap time re-doing these — they are correct today:

- **No dangerous dynamic execution anywhere.** `grep` across `src/` for `pickle`, `joblib`,
  `np.load(allow_pickle=)`, `eval`, `exec`, `os.system`, `shell=True`, `__import__` -> **zero
  hits**. Every `getattr` call is a safe attribute probe with a literal name and a default; every
  `importlib` use is a `find_spec` / `import_module` on a hard-coded dependency name
  (`src/yowo/hardware/_capabilities.py:71,90,111,132,149`, `src/yowo/benchmark/_runner.py:48-61`).
- **YAML is always `safe_load`.** `src/yowo/config.py:534`, `src/yowo/tune/_profile.py:166`,
  and `safe_dump` at `:127`. No `yaml.load` anywhere.
- **Git history is clean.** 281 commits, pickaxe-swept for six credential prefix classes -> zero
  hits. `.env` was gitignored from the start (`.gitignore:123`) and never tracked. No history
  rewrite is needed for P0-1 — rotation alone closes it.
- **The wheel is clean** — only `.py`, `README.md`, `py.typed`, and dist-info. The sdist problem in
  P0-4 does not affect wheel consumers.
- **TLS verification is on by default** — `requests.get` at `src/yowo/models/_weights.py:113` does
  not disable `verify`, and `verify=False` appears nowhere in the codebase.
- **The download path is defensively written** in every respect *except* integrity: 3 retries with
  exponential backoff (`src/yowo/models/_weights.py:83-94`), a connect/first-byte timeout, and — a
  nice touch most libraries miss — a wall-clock stall deadline across the whole chunk loop
  (`:135-142`), plus atomic `os.replace` (`:86`) and temp-file cleanup on failure (`:93-94`).
- **Atomic writes are used consistently** for every state file: `src/yowo/io/_sink.py:31-38`,
  `src/yowo/tune/_profile.py:126-127`. No partial-file races.
- **Subprocess usage is textbook** — argv lists, no shell, explicit timeouts, `check=False` with
  return-code handling (`src/yowo/hardware/_detect.py`).
- **Cache directory names are hashed and regex-guarded before deletion**
  (`src/yowo/cache/_store.py:165-178`) — the `rmtree` in P2-3 cannot escape the hash-shaped
  namespace.
- **The release workflow triggers only on `push: branches: [main]`**
  (`.github/workflows/release.yml:3-5`), so fork pull requests cannot reach the privileged job —
  this is what keeps P1-1 out of P0. The release job is additionally gated on
  `environment: release` (`:47`), which supports required reviewers.
- **The engine already threads a `source_id` abstraction end-to-end**
  (`src/yowo/engine.py:832-834`) — P0-3's fix is a one-line redaction at the single point where
  that value is minted, not a refactor.
- **No archive extraction at all** — `grep -rn --include='*.py' -E "zipfile|tarfile|shutil\.unpack|extractall" src/`
  -> zero hits. Zip-slip is not applicable. Directory walking
  (`src/yowo/batch/_runner.py:142`, `src/yowo/export/_exporter.py:498`,
  `src/yowo/backends/_openvino.py:241`) operates on an operator-supplied root and never joins an
  untrusted string onto a base path, so there is no traversal primitive either.
