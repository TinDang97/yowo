# yowo — production-readiness deep review

**Date:** 2026-09-08 · **Version reviewed:** 2.5.0 (`main` @ `3340bb3`) · **Method:** six parallel
specialist lanes, each required to cite `path:line` and to verify claims by execution where possible.

## Verdict

The **code** is largely sound. `pyright` runs clean at `strict` over `src/yowo/` (0 errors,
0 warnings), `py.typed` ships, preprocessing and multi-stream backpressure are well built, the weight
download path has retries, backoff and atomic replace, and there is not a single `eval`, `exec`,
`pickle`, `os.system` or `shell=True` in the tree. Every YAML call is `safe_load`. Git history is
clean of secrets.

What is missing is **everything that would prove it works, or stop it shipping wrong**: no test
executes a real backend forward pass, no cross-backend parity suite exists, no accuracy gate exists,
model artifacts have no integrity chain, there is no PR-triggered CI, and the release workflow has
failed on every push since v2.5.0 while containing no PyPI publish step at all.

## Tally

| Lane | Report | P0 | P1 | Biggest gap |
|---|---|---:|---:|---|
| Release / CI / packaging | [review-release.md](review-release.md) | 6 | 8 | Release workflow red since v2.5.0; no publish step ever existed |
| Runtime robustness | [review-runtime.md](review-runtime.md) | 6 | 16 | No capture timeout anywhere; RTSP backoff is unreachable dead code |
| Test strategy | [review-tests.md](review-tests.md) | 7 | 8 | 2175 tests, 80% lines — none runs a real backend |
| Perf / deployment | [review-deploy.md](review-deploy.md) | 6 | 14 | `precision` is inert on every backend, yet reported in health |
| Security / supply chain | [review-security.md](review-security.md) | 4 | 7 | No integrity chain for model artifacts |
| API surface / backends | [review-api.md](review-api.md) | 4 | 11 | Version identity wrong in 3 places, stamped into export sidecars |
| **Total** | | **33** | **64** | ~28 distinct after dedupe |

Three P0s were found independently by two lanes each — weight integrity (runtime + security),
RTSP credential leakage (runtime + security), and the AGPL sdist (release + security).

## The findings that are actively false, not merely absent

These matter more than the absences, because a reader currently has grounds to believe the opposite:

1. **`src/yowo/types.py:314-329`** — the docstring states the result JSON is *"safe for logging and
   wire transport"*. It excludes the local weights path and **includes the camera password**, because
   `io/_source.py:336` stamps the credentialed RTSP URL onto every `Frame` as `source_id`.
2. **`tmp/compare_arch.py`** — cited in `CLAUDE.md` as the source of the "10/10 variants PASS"
   architecture-equivalence claim. It is gitignored (`.gitignore:368`), was never tracked, and does
   not exist. That evidence is unavailable to anyone, including the maintainer.
3. **`CONTRIBUTING.md:27`** — "yowo is intentionally ultralytics-free (Apache-2.0 clean)". The
   published `yowo-2.4.1.tar.gz` on PyPI contains `yolo11n.pt`, whose own metadata reads
   `AGPL-3.0 License (https://ultralytics.com/license)`, under `License-Expression: Apache-2.0`.
4. **`tests/integration/test_cli_e2e.py:76`** — asserts version `"0.1.0"` against an actual `2.5.0`.
   It has been failing since before v2.4.0, in a tier CI never invokes.
5. **`engine.py:402`** — `health_report().precision_current` reports a precision that no backend
   consumes; `grep -rn "precision" src/yowo/backends/*.py` (excluding the selector) returns nothing.

## Measured facts

Numbers here were produced by commands, not estimated. Each report records the command it ran.

- 2175 tests collected (2142 unit + 33 integration) — not the 1910 recorded in `CLAUDE.md`.
- 80% line coverage over `src/yowo` (8883 statements, 1807 missed). No branch coverage, no
  `fail_under`, no `[tool.coverage]` section.
- `uv run pytest tests/integration -q` → **1 failed, 13 passed, 20 skipped**.
- `_openvino.py` at **0%** coverage; `_pytorch.py` (the default backend) at 49% with `infer()`,
  `warmup()`, `unload()` and `_resolve_device()` entirely uncovered, and no test file at all.
- Same commit builds a **971 KB** sdist in CI and **7.4 MB** on a laptop — releases are not reproducible.
- Hot path, YOLO11n / PyTorch CPU / 1080p: preprocess 1.23 ms, infer 38.12 ms, postprocess 0.15 ms
  — but postprocess rises to 4.91 ms at 3000 candidates (`cv2.dnn.NMSBoxes` alone 15.42 ms), and
  there is no pre-NMS top-k or `max_det`, so p99 is unbounded in scene content.

## How to read these reports

Each is standalone and triaged P0 (blocks launch) / P1 (before GA) / P2 (later), with symptom,
`path:line` evidence, why it blocks, and the smallest correct fix. They are the evidence base for
the production-readiness roadmap tracked in `.add/` — run `python3 .add/tooling/cli.py status`.

**Scope note:** this was a diagnostic pass. No source file was modified by any lane.

---

## Addendum — 2026-09-08, found during Direction, not by this review

A P0 that all six lanes missed, surfaced while grounding the `weight-integrity` task:

### yowo cannot load its own default weights on a clean install

```
pip install yowo[pytorch]     # torch only — ultralytics is NOT a dependency
yowo detect image.jpg         # downloads yolo11n.pt from the ultralytics assets URL
  → ModuleNotFoundError: No module named 'ultralytics.nn.tasks'
```

`src/yowo/arch/_weights.py:86` calls `torch.load(..., weights_only=False)`. The checkpoint's
`ema`/`model` value is a pickled **`ultralytics.nn.tasks.DetectionModel`**, and unpickling requires
that class to be importable. No shim exists — `sys.modules`, `Unpickler` and `add_safe_globals`
return no hits across `src/yowo/`.

**Probes that establish it** (run against the tracked `yolo11n.pt`, 2026-09-08):

| Probe | Result |
|---|---|
| `torch.load(p, weights_only=True)` | `UnpicklingError` — the obvious security fix is unavailable |
| `type(ckpt["ema"] or ckpt["model"])` | `ultralytics.nn.tasks.DetectionModel` |
| `yowo.arch._weights.load_weights(model, "yolo11n.pt")` with `ultralytics` import blocked | `ModuleNotFoundError: No module named 'ultralytics.nn.tasks'` |
| `sha256(yolo11n.pt)` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |

### Why the review missed it, which is the more useful finding

`ultralytics` is a dev dependency, so it is importable on the maintainer's machine and on every CI
runner. The release lane verified that a clean-environment install *imports* correctly and that a
missing backend raises a typed `DependencyError` — both true. Neither establishes that the documented
first-run command works, because **no test in the repository executes a real backend forward pass**
(`review-tests.md` T1). The gap between "the package imports" and "the package works" is exactly the
width of the missing evidence this roadmap's `m3-prove-it` exists to close.

### What it changes

1. `CONTRIBUTING.md:27`'s "intentionally ultralytics-free (Apache-2.0 clean)" is false in a second,
   deeper way than the AGPL sdist: an **undeclared runtime dependency** the package cannot satisfy.
2. `weight-integrity`'s planned fix — set `weights_only=True` — is impossible as written.
3. A restrictive `pickle.Unpickler.find_class` that stubs `ultralytics.*` and extracts only the
   `state_dict` resolves both: it removes the hidden dependency and is strictly safer than
   `weights_only=False`. Tracked as `m1/checkpoint-loader`, which `weight-integrity` and
   `ci-weight-fixture` now depend on.
