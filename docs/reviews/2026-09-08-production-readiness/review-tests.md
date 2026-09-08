# Test strategy, coverage & correctness-evidence review — yowo production readiness

## Measured baseline (commands and output)

All commands run from `/Users/tindang/workspaces/tind-repo/yowo` on darwin/py3.11.15.

```
$ uv run pytest tests/ --collect-only -q 2>&1 | tail -5
2175 tests collected in 27.47s
  SKIPPED tests/integration/test_chroma_gallery_persistence.py:11: could not import 'chromadb'
  SKIPPED tests/unit/test_chroma_gallery.py:11: could not import 'chromadb'

$ uv run pytest tests/unit --collect-only -q 2>&1 | tail -3
2142 tests collected in 3.16s

$ uv run pytest tests/integration --collect-only -q 2>&1 | tail -2
33 tests collected in 1.48s
```

The CLAUDE.md figure of "1910 tests as of v2.4.0" is stale but not inflated: **2175 collected
(2142 unit + 33 integration)** at v2.5.0.

```
$ uv run pytest tests/unit -q -p no:randomly --cov=src/yowo \
      --cov-report=term-missing:skip-covered --timeout=300 | tail -80
...
TOTAL                                    8883   1807    80%
2132 passed, 11 skipped, 31 warnings in 75.38s (0:01:15)
```

**80% line coverage** over `src/yowo` (8883 statements, 1807 missed), 31 files at 100%.
Branch coverage is *not* measured — there is no `[tool.coverage]` section in `pyproject.toml`
and no `--cov-branch` anywhere (`grep -rn "fail_under\|cov-fail" .` → no matches).

```
$ uv run pytest tests/integration -q --timeout=120 | tail -25
1 failed, 13 passed, 20 skipped in 10.38s
FAILED tests/integration/test_cli_e2e.py::TestCLIHelp::test_version_flag
```

```
$ uv run python -c "import openvino / coremltools / tensorrt / chromadb ..."
torch: INSTALLED      onnxruntime: INSTALLED   onnx: INSTALLED       scipy: INSTALLED
ultralytics: INSTALLED  pycocotools: INSTALLED  cv2: INSTALLED
openvino: MISSING     coremltools: MISSING     tensorrt: MISSING     chromadb: MISSING
```

Mock density in the unit tier:

```
$ grep -rho "MagicMock\|monkeypatch\|patch(" tests/unit/*.py | sort | uniq -c
 541 MagicMock
 565 patch(
  58 monkeypatch
$ grep -rlc "MagicMock\|Mock(\|mock.patch\|monkeypatch" tests/unit/*.py | wc -l   # 47 of 94 files
```

`src/yowo` is 23,835 LOC; `tests/` is 28,621 LOC.

### Complete inventory of skip guards

Every `skipif` / `importorskip` / `pytest.skip` in the repository
(`grep -rn "skipif\|importorskip\|pytest\.skip\|xfail" tests/`):

| Guard | Location | Effect in the default env | Effect on CI (`ubuntu-latest`, py3.11, `tests/unit/`) |
|---|---|---|---|
| `importorskip("chromadb")` | `tests/unit/test_chroma_gallery.py:11` | SKIPS (chromadb not installed) | SKIPS — `chromadb` is in no dev group |
| `importorskip("chromadb")` | `tests/integration/test_chroma_gallery_persistence.py:11` | SKIPS | never collected |
| `skipif(not onnxscript)` | `tests/unit/test_kv_export.py:245-248` | runs (onnxscript present on py>=3.10) | runs on 3.11; silently skips on 3.9 |
| `skipif(not _has_onnxruntime)` | `tests/unit/test_batch_inference.py:54` | runs | runs |
| `importorskip("torch")` | `tests/unit/test_torch_free_import.py:131` | runs | runs (intentional guard, correct) |
| `pytest.skip` (future-annotations) | `tests/unit/test_py38_compat.py:124` | 10 skips (all `__init__.py`) | 10 skips |
| `pytest.skip` on `/Users/tindang/Downloads/...` | `tests/integration/conftest.py:32-36` | SKIPS 7 CLI e2e tests | never collected |
| `pytest.skip` on network failure | `tests/integration/conftest.py:50-55` | SKIPS silently on any blip | never collected |
| `pytest.skip` on `tmp/weights/...` | `tests/integration/test_engine_integration.py:29-31` | SKIPS 13 engine tests | never collected |
| `importorskip("torch")` ×4 | `tests/integration/test_engine_integration.py:37,67,76,85,169` | moot (weights already gone) | never collected |

**Production paths with no guard at all — because no test references them:** OpenVINO
(`_openvino.py` 0%), TensorRT/CoreML/OpenVINO factory dispatch
(`backends/__init__.py:188-199`), `PyTorchBackend.infer/warmup/unload`
(`_pytorch.py:277-367`), `VideoFileSource.__iter__` (`io/_source.py:222-255`),
`RTSPStreamSource.reconnect` (`io/_source.py:360-374`). These are worse than a skip: a skip is at
least visible in the run summary.

---

## Verdict

Not production-ready as a correctness-evidence lane. The suite is large, fast (75s), well
organised and genuinely good at *contract* testing — error mapping, config validation, protocol
shape, tensor math on synthetic arrays. But it has a hard ceiling: **no test in this repository
ever runs a real model through a real backend.** Every backend test drives a `MagicMock` session;
`onnxruntime.InferenceSession` is never constructed anywhere in `tests/`; the default PyTorch
backend's `infer()` is 0% covered; and the entire integration tier — the only place real weights
would be loaded — is gated behind a hardcoded path on one developer's laptop
(`tests/integration/conftest.py:16`) and is not run by CI at all.

The single biggest gap: **the product's central correctness claim (native arch is numerically
equivalent to ultralytics, 10/10 variants) has zero reproducible evidence in the repository.**
The script that produced it, `tmp/compare_arch.py`, is referenced in CLAUDE.md but `tmp/` is
gitignored (`.gitignore:368`), was never tracked (`git log --all -- 'tmp/*'` → empty), and the
directory does not exist on disk today. A 2175-test green suite would not catch a regression in
the thing this library exists to do.

---

## P0 — blocks launch

### T1. The mock ceiling: no test executes a real backend forward pass

- **Symptom:** the suite proves the *plumbing around* inference, never inference. A backend that
  returns correctly-shaped garbage passes 2132 tests.
- **Evidence:**
  - `src/yowo/backends/_pytorch.py` — **49% covered**; `infer()` body uncovered at lines 277-317,
    `unload()` 321-334, `warmup()` 341-354, `_resolve_device()` 362-367, and most of `load()`
    (169-242). There is **no `tests/unit/test_pytorch_backend.py`**
    (`ls tests/unit | grep -i pytorch` → only `test_torch_free_import.py`); `PyTorchBackend` is
    referenced only incidentally by `tests/unit/test_custom_num_classes.py` and
    `tests/unit/test_model_builder.py`.
  - `tests/unit/test_onnx_backend.py:19-24` and `tests/unit/test_tensorrt_backend.py:19-23`
    both build the hardware profile and the session as `MagicMock()`.
  - `grep -rn "InferenceSession" tests/` → **no test ever constructs an ONNX Runtime session.**
  - The one place real weights would load — `tests/integration/test_engine_integration.py:25`
    (`_WEIGHTS = Path("tmp/weights/yolo26n_statedict.pt")`) — skips, because `tmp/` does not exist.
- **Why it blocks:** the default backend on every install is PyTorch. Its inference, warmup,
  device selection and unload paths ship with no executed test. Any bug in dtype handling, device
  placement, output layout or memory release reaches users unobserved — this is exactly the class
  of bug the library exists to prevent.
- **Smallest fix:** commit one ~6 MB nano checkpoint (or generate synthetic weights from
  `build_model()` and `torch.save` in a session fixture), add `tests/unit/test_pytorch_backend.py`
  asserting a real `load()→warmup()→infer()→unload()` cycle produces a finite tensor of the
  documented shape, and run it in CI. It costs one fixture and ~5s.

### T2. The integration tier never runs — and is already broken

- **Symptom:** 20 of 33 integration tests skip on any machine but one; 1 fails outright and has
  been failing across at least two minor releases without anyone noticing.
- **Evidence:**
  - `tests/integration/conftest.py:16` — `_LOCAL_WEIGHTS = Path("/Users/tindang/Downloads/Ultralytics YOLO26.pt")`.
    A personal absolute path is the gate for every real-inference CLI test.
  - `tests/integration/test_engine_integration.py:25,31` — skips on `tmp/weights/yolo26n_statedict.pt`,
    a gitignored location.
  - `.github/workflows/release.yml:39` — `uv run pytest tests/unit/ -x -q --tb=short --timeout=60`.
    `tests/integration/` is never invoked by any workflow (`ls .github/workflows/` → `release.yml` only).
  - `tests/integration/test_cli_e2e.py:76` — `assert "0.1.0" in result.output` while
    `pyproject.toml:3` says `version = "2.5.0"`. Confirmed failing:
    `1 failed, 13 passed, 20 skipped`.
  - `CONTRIBUTING.md:123` documents this tier as "may require GPU" — it actually requires a
    specific user's `~/Downloads`.
- **Why it blocks:** the classic silent-skip failure mode. The suite is green, the e2e lane is
  dead, and the rot is already measurable (a stale version assertion survived v2.4.0 and v2.5.0).
  Everything the integration tier is supposed to certify — CLI end-to-end, real detections on a
  real image, JSON output schema against real boxes, ONNX export producing a loadable file — is
  certified by nothing.
- **Smallest fix:** replace the hardcoded paths with a `YOWO_TEST_WEIGHTS` env var plus a
  cached download fixture; fix `test_cli_e2e.py:76` to read `importlib.metadata.version("yowo")`;
  add an `integration` job to CI that runs `pytest tests/integration -m "integration and not slow"`
  and **fails** (rather than skips) if the weights fixture cannot be resolved in CI.

### T3. The architecture-equivalence evidence does not exist in the repository

- **Symptom:** the headline correctness claim ("10/10 variants numerically match ultralytics",
  `Conv.bn eps=1e-3/momentum=0.03`, `Attention.qkv act=False`, `SPPF.cv1 act=not shortcut`) is
  enforced by nothing that a CI run or a new contributor can execute.
- **Evidence:**
  - `CLAUDE.md` / project memory: *"Validation: `uv run python tmp/compare_arch.py --all` → 10/10 PASS"*.
    `ls tmp/` → `"tmp/": No such file or directory`.
  - `.gitignore:368` — `tmp/*`. `git log --oneline --all -- 'tmp/*'` → empty; `git ls-files | grep compare_arch` → empty.
    The script was **never committed**.
  - `ultralytics>=8.4.21` *is* a declared dev dependency (`pyproject.toml:73`) and *is* installed,
    yet no test imports it for tensor comparison:
    `grep -rn "ultralytics" tests/` returns only a mocked `run_ultralytics_benchmark`
    (`tests/unit/test_obb_integration.py:66`), string comments about checkpoint key format
    (`tests/unit/test_obb_weights.py:20`, `tests/unit/test_classify_weights.py:64`) and report
    column names (`tests/unit/test_benchmark_report.py:48`).
  - `src/yowo/arch/_weights.py` — the ultralytics-checkpoint remapper — is **40% covered**;
    `load_weights()` (126-196) and `load_obb_weights()` (346-406) are largely unexecuted.
- **Why it blocks:** an architecture rewrite whose whole selling point is drop-in equivalence,
  with the equivalence proof stored in a gitignored scratch file that has since been deleted. A
  one-line change to a `momentum` or an activation flag would silently degrade every user's
  accuracy and the suite would stay green.
- **Smallest fix:** move `compare_arch.py` into `tests/integration/test_arch_equivalence.py`
  guarded by `pytest.importorskip("ultralytics")`, parametrised over the 10 variants, asserting
  `torch.testing.assert_close(..., atol=1e-4)` on a fixed-seed input, and run it in CI (ultralytics
  is already a dev dep on py3.11, so this costs only weight download time).

### T4. No accuracy / mAP regression gate anywhere

- **Symptom:** the library can silently lose detection quality between releases and no test notices.
- **Evidence:**
  - `tests/unit/test_benchmark_evaluator.py:251-253` — the *only* mAP-adjacent test injects fake
    modules: `sys.modules["pycocotools"] = mock_pycocotools`, then asserts
    `result["mAP_50_95"] == pytest.approx(0.35)` (line 269) against the mock's own return value.
    It tests the plumbing of `evaluate_coco_map()`, not any model's mAP.
  - `src/yowo/benchmark/_evaluator.py` 82%, `src/yowo/benchmark/_runner.py` 64%,
    `src/yowo/benchmark/_dota_evaluator.py` **11%**, `src/yowo/benchmark/_comparison.py` **24%**.
  - `README.md:796` publishes `CLIP-ReID VeRi mAP=82.28%, Rank-1=96.66%` while
    `src/yowo/tracking/_reid.py` is **19%** covered and `src/yowo/tracking/_clip_reid.py` **27%**.
    The only backing is a prose doc under `docs/experiments/`.
- **Why it blocks:** for an inference library, "the output is still correct" is the product.
  Shipping published accuracy numbers with no runnable check behind them is the accuracy-claim-
  with-no-regression-test failure mode verbatim.
- **Smallest fix:** add a nightly (not per-PR) job running `yowo benchmark` over a pinned
  COCO val2017 subset (128-512 images) for `yolo11n` and asserting `mAP_50_95 >= <recorded baseline
  - 0.005>`. Pin the subset and the baseline in the repo so the number is auditable.

### T5. Zero export round-trip / numeric-parity coverage

- **Symptom:** exporters are tested only for the errors they raise, never for the numbers they
  produce. Nothing proves an exported ONNX/TensorRT/CoreML model agrees with the PyTorch source.
- **Evidence:**
  - `src/yowo/export/_exporter.py` — **43% covered**; the actual export bodies at 231-260,
    275-353, 365-406, 411-422 are unexecuted.
  - `tests/unit/test_exporter.py` contains only validation tests
    (`test_int8_without_calibration_raises`, `test_negative_batch_size_raises`, …).
  - `tests/unit/test_int8_export.py:332-339` fabricates `types.ModuleType("onnxruntime")` and
    patches it into `sys.modules` — the quantizer is verified against a fake.
  - `tests/unit/test_kv_export.py:245-248` does export a real `.onnx` file, but only asserts it is
    non-empty (`TestOnnxExport`, guarded by `skipif not onnxscript`); it never loads the result and
    compares outputs. Its `allclose` assertions (lines 43, 95, 161, 172) are torch-vs-torch.
  - No `InferenceSession` is created in any test (see T1).
- **Why it blocks:** export is the reason a team picks this library for edge deployment. An
  exporter that emits a structurally valid but numerically wrong graph (wrong output transpose,
  wrong normalisation baked in, INT8 calibration silently degenerate) passes every existing test.
- **Smallest fix:** one test that exports the nano model to ONNX, runs it under `onnxruntime`
  (already a dev dep) on a fixed input, and asserts `np.allclose(pt_out, ort_out, atol=1e-3)`.
  Extend the same fixture to CoreML/TensorRT on the runners where those exist.

### T6. An advertised backend has 0% coverage; 3 of 5 factory branches are unexecuted

- **Symptom:** `OpenVinoBackend` is auto-selectable at priority 5 and no line of it has ever run
  under test.
- **Evidence:**
  - Coverage report: `src/yowo/backends/_openvino.py   127   127   0%   11-263`.
  - `grep -rn "OpenVINOBackend\|_openvino" tests/` → **no test imports the module**. There is no
    `tests/unit/test_openvino_backend.py`.
  - `src/yowo/backends/_selector.py:266-272` auto-selects it whenever `libs.openvino_version` is
    set; `_selector.py:379-380` handles the explicit-request path (also uncovered).
  - `src/yowo/backends/__init__.py:188-199` — the `create_backend()` branches for
    `TENSORRT`, `OPENVINO` and `COREML` are all in the uncovered range (`backends/__init__.py 81%
    … Missing 188-199`). The factory dispatch for three of five backends is never executed.
  - Related: `src/yowo/backends/_tensorrt.py` 74%, `_coreml.py` 86% — both mock-only, and neither
    library is installed in CI (`pyproject.toml:55-76` dev group has no `openvino`, `coremltools`,
    `tensorrt`, `chromadb`).
- **Why it blocks:** README and `pyproject.toml:9` sell OpenVINO/TensorRT/CoreML as first-class
  backends. A user on an Intel box with `yowo[openvino]` installed hits code that has never
  executed in any test run, ever — including a class-name mismatch risk (`OpenVinoBackend`, not
  `OpenVINOBackend`) that only the untested factory branch would surface.
- **Smallest fix:** at minimum, an import-and-construct smoke test per backend with a mocked
  runtime plus a `create_backend()` parametrised over all five `BackendType` members asserting the
  right class comes back. Then add an optional-extras CI job (`uv sync --extra openvino`) that runs
  the OpenVINO tests for real on ubuntu-latest — OpenVINO installs cleanly on x86 Linux runners.

### T7. The release gate certifies exactly one environment and one test tier

- **Symptom:** the *only* automated gate is `ubuntu-latest` + Python 3.11 + `tests/unit/`.
  Everything else ships uncertified, and the same workflow run that skips it then publishes.
- **Evidence:** `.github/workflows/release.yml:12-40` (single `quality` job: ruff, ruff-format,
  pyright, `pytest tests/unit/ -x -q --timeout=60`), `release.yml:42-45`
  (`release` job `needs: [quality]`, cuts the tag and publishes).
- **What ships untested as a direct result** — precisely:
  - **Python 3.9, 3.10, 3.12** — all four claimed in `pyproject.toml:14-19`; only 3.11 is run.
    (3.8 is also claimed at `pyproject.toml:7,14` while `[tool.uv] environments` at
    `pyproject.toml:79-81` pins `python_version >= '3.9'` — the classifier is not just untested,
    it is contradicted by the lockfile.)
  - **macOS and Windows** — zero runs, despite `DeviceType.MPS` and a native CoreML backend
    existing specifically for Apple Silicon.
  - **aarch64 / Jetson** — `src/yowo/hardware/_capabilities.py` 71%, `_detect.py` 81`%`; the last
    three commits (`13e0b69`, `58e04fb`) are Jetson/driver GPU-detection bug fixes, i.e. this is a
    demonstrated live defect area with no runner behind it.
  - **CUDA / TensorRT** — no GPU runner; `_tensorrt.py` mock-only.
  - **OpenVINO** — 0% (T6). **CoreML** — mock-only. **ChromaDB** — every test skips
    (`tests/unit/test_chroma_gallery.py:11`, `tests/integration/test_chroma_gallery_persistence.py:11`);
    `_chroma_gallery.py` 20%.
  - **All of `tests/integration/`** (T2): CLI end-to-end, real detection output, JSON schema
    against real boxes, ONNX export file production.
  - **Real video decode and RTSP** (see P1-T9).
- **Why it blocks:** the badge of confidence covers a single interpreter on a single OS running a
  single mock-heavy tier, and it is the gate that authorises publishing.
- **Smallest fix:** turn `quality` into a matrix over `{ubuntu-latest, macos-latest} ×
  {3.9, 3.11, 3.12}`, drop `-x` (see P1-T10), and add a second required job that runs
  `tests/integration` with a downloadable weights fixture. Until macOS/aarch64 runners exist,
  demote the corresponding classifiers rather than claiming support.

---

## P1 — before GA

### T8. No pre-merge CI: tests run only after code is already on main

- **Symptom:** a pull request receives no automated verification.
- **Evidence:** `.github/workflows/release.yml:3-5` —
  `on: push: branches: [main]`. There is no `pull_request` trigger and no second workflow
  (`ls .github/workflows/` → `release.yml` only). `CONTRIBUTING.md:215-218` asks contributors to
  run ruff/pyright/pytest by hand; `.pre-commit-config.yaml` enforces it locally *if installed*.
- **Why it matters:** the gate is honour-system until merge. When it does fire, main is already
  broken and the fix lands as a second commit that also triggers a release.
- **Smallest fix:** add `pull_request: branches: [main]` to the `quality` job's trigger and mark
  it a required status check.

### T9. Real media I/O is uncovered — video decode and RTSP reconnect

- **Symptom:** the code paths that every edge deployment lives on are never executed.
- **Evidence:** `src/yowo/io/_source.py` **67%**, missing:
  - `VideoFileSource.__iter__` (`_source.py:222-255`) — the whole decode loop, including
    `--loop` rewind and `frame_skip` — uncovered. `yowo detect video.mp4` is untested.
  - `RTSPStreamSource.reconnect()` (`_source.py:360-374`) and the tail of its `__iter__`
    (343-355) — uncovered. `tests/unit/test_reader.py:467-505` tests that
    `ThreadedFrameReader` *schedules* a reconnect against a fake source, but the actual
    `cv2.VideoCapture` re-open is never exercised.
  - `WebcamSource` (`_source.py:436-437,445,462,465,468`) and `open_source()` dispatch
    (`530-547`) partially uncovered.
- **Smallest fix:** generate a 10-frame MP4 with `cv2.VideoWriter` in a `tmp_path` fixture and
  assert frame count, `--loop`, and `frame_skip` behaviour; unit-test `reconnect()` against a
  patched `cv2.VideoCapture` that fails once then succeeds.

### T10. 36 wall-clock sleeps + `-x` in CI = one flake blocks the release

- **Symptom:** roughly a third of the unit suite's wall time is real sleeping with margins as
  tight as 5-50 ms, and the CI gate aborts the whole run on the first failure.
- **Evidence:**
  - `grep -rn "time.sleep\|asyncio.sleep" tests/ | wc -l` → **36**. Real (non-patched) sleeps at
    `tests/unit/test_reader.py:115,130,142,199,274,400`, `test_events.py:23,64,78,143,221`,
    `test_collector.py:59,362`, `test_streaming.py:238,244`, `test_scheduler.py:52`,
    `test_metrics.py:85,203`, `test_shutdown.py:203,239`, `test_async_api.py:170,173,252,412`.
  - Comments name the hazard directly: `test_reader.py:115` — `time.sleep(0.05)  # let reader read
    several frames`; `test_events.py:64` — `# give worker a chance to deliver B (should not happen)`.
  - Timing cost measured: those six files alone are 119 tests / ~25 s of the 75 s suite.
  - `release.yml:39` uses `-x`. Three consecutive local runs were stable
    (`119 passed` × 3, 24.5-27.3 s) — on an idle 12-core Mac, not a 2-core shared runner.
  - `pyproject.toml:117` — `addopts = "-ra --strict-markers"`; no default `--timeout`, so a hung
    test outside CI blocks forever.
- **Smallest fix:** replace "sleep then assert" with `threading.Event`/condition-variable waits or
  a polling helper with a generous deadline; drop `-x` from CI so one flake does not mask the rest;
  add `--timeout=60` to `addopts` so the local default matches CI.

### T11. No coverage floor, and the documented target measures nothing

- **Symptom:** coverage is reported ad hoc and never enforced; the stated goal is branch coverage
  which is not even collected.
- **Evidence:** no `[tool.coverage]` section in `pyproject.toml` (grep of the file shows
  `[tool.pytest.ini_options]` at 114 followed directly by `[tool.semantic_release]` at 127);
  `grep -rn "fail_under\|cov-fail" .` → no matches; `release.yml:39` runs pytest with no `--cov`.
  `CONTRIBUTING.md:129` — *"Coverage target: new code should not reduce overall branch coverage"* —
  while `--cov-branch` appears nowhere and `README.md:775` documents only line coverage.
- **Smallest fix:** add `[tool.coverage.run] branch = true, source = ["src/yowo"]` and
  `[tool.coverage.report] fail_under = 80`, and put `--cov --cov-fail-under` in the CI test step.
  Ratchet the floor upward as T1/T5/T6 land.

### T12. Integration fixtures reach the network and use an undeclared dependency

- **Symptom:** the integration tier downloads a test image at run time and imports a package that
  is not in any dependency group.
- **Evidence:** `tests/integration/conftest.py:17,51` —
  `_BUS_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"`, `requests.get(..., timeout=30)`,
  and on failure `pytest.skip` (line 55) — a network blip silently removes coverage rather than
  failing. `import requests` at `conftest.py:13` while `requests` appears in neither
  `[project.dependencies]` (`pyproject.toml:22-29`) nor `[dependency-groups] dev`
  (`pyproject.toml:55-76`) — it resolves only transitively via `ultralytics`, which itself is
  conditional on `python_version >= '3.11'`.
  (The unit tier is clean: `grep -rn "requests\.\|urlopen\|http" tests/unit/` finds only two
  `https://example.com` string literals in `test_registry.py:93,120`.)
- **Smallest fix:** vendor a small test image into `tests/data/`, or cache the download under
  `tmp_path_factory` with a checksum and **fail** rather than skip when the fixture is required;
  add `requests` to the dev group if it is kept.

### T13. No soak or long-running coverage for streaming and the multi-stream pipeline

- **Symptom:** the streaming/pipeline subsystem — the part that runs for days on an edge box — is
  tested only in short bursts.
- **Evidence:** `tests/unit/test_pipeline.py` (441 lines), `test_collector.py` (699),
  `test_streaming.py` (294), `test_scheduler.py` (215), `test_router.py` (189) contain no
  long-horizon test. The single resource-leak check is
  `tests/unit/test_collector.py:659-673` (`test_remove_stream_no_leak`, `tracemalloc`, "< 500KB
  delta") — one scenario, one subsystem. `grep -rn "soak\|range(10000\|tracemalloc" tests/`
  finds nothing else. `src/yowo/pipeline/_collector.py` 87% (317-334 uncovered),
  `src/yowo/batch/_runner.py` 79%.
- **Smallest fix:** one `@pytest.mark.slow` nightly test that runs `run_pipeline()` over 4
  synthetic sources for ~10k frames and asserts RSS growth and queue depth stay bounded.

### T14. `tests/conftest.py` is empty — no global test isolation

- **Symptom:** zero shared fixtures, zero autouse cleanup, no seeding policy.
- **Evidence:** `wc -c tests/conftest.py` → **0**. Environment handling is good
  (`monkeypatch.setenv`/`patch.dict` used 29 times; `grep -rn "os.environ\[" tests/` → no direct
  writes), but there is no autouse reset for hardware-detection results, logging configuration,
  or torch thread limits (`src/yowo/backends/_pytorch.py` sets process-wide
  `torch.set_num_threads`), and no `pytest-randomly`/`pytest-random-order` in the dev group to
  detect ordering dependence at all.
- **Smallest fix:** add a root `conftest.py` with an autouse fixture that resets global caches and
  seeds `np.random`/`torch`, and add `pytest-randomly` to the dev group so ordering coupling
  surfaces before it bites.

### T15. Declared markers are never used as a CI selector

- **Symptom:** the `slow`/`integration` split exists on paper and is wired to nothing.
- **Evidence:** `pyproject.toml:119-122` declares both markers; they are applied only inside
  `tests/integration/` (`test_cli_e2e.py:63,94,121,159,202,203,398,399`,
  `test_engine_integration.py:63,98,129,160`). `tests/integration/test_cli_e2e.py:14-18`
  documents `pytest tests/integration/ -m "integration and not slow"` — no workflow runs it.
  Nothing in `tests/unit/` is marked, so `-m "not slow"` cannot be used to build a fast lane.
- **Smallest fix:** once T2 lands, split CI into `-m "not slow"` (per-PR) and `-m slow` (nightly).

---

## P2 — later

- **T16. Unseeded randomness in ~9 tests.** Most tests correctly use
  `np.random.default_rng(<seed>)` (20+ sites), but `tests/unit/test_tracking.py:112-113`,
  `test_tracking_reid.py:144,149`, `test_ortvalue.py:20`, `test_cache.py:151`,
  `test_classify_postprocess.py:84,125,136` use global `np.random.rand/randn/randint`. Low risk
  today (assertions are structural), but a latent nondeterminism source. Fix: seed them or route
  through `tests/utils.py`.
- **T17. No NaN/Inf/huge-count edge cases in NMS.** `tests/unit/test_nms.py` (39 tests) is
  otherwise strong — it covers empty input (line 69), all-below-threshold (273), the OpenCV
  empty-tuple return (137), the class-offset trick (151), letterbox clipping (642), class-id
  overflow fallback (657, 681) and determinism (707, 726). But
  `grep -rn "nan\|inf" tests/unit/test_nms.py tests/unit/test_obb_nms.py` → no matches; NaN is
  covered only at `tests/unit/test_warmup_validation.py:148,160`. A NaN box from a bad
  quantised model reaches `cv2.dnn.NMSBoxes` untested. Add three cases: NaN box, Inf confidence,
  100k-detection input.
- **T18. Python 3.8 support is claimed but unresolvable.** `pyproject.toml:7,14` claim 3.8;
  `pyproject.toml:79-81` (`[tool.uv] environments = ["python_version >= '3.9'"]`) excludes it.
  `tests/unit/test_py38_compat.py` is a static AST guard (checks for
  `from __future__ import annotations`, no runtime `X | Y` in `__init__.py`) — useful, but it
  cannot catch a 3.9+ stdlib call. Either drop the 3.8 classifier or add a real 3.8 CI leg.
- **T19. No property-based testing.** For the geometry-heavy code (letterbox inverse,
  `counter/_geometry.py`, `tracking/_kalman.py`, OBB rotation) `hypothesis` would find edge
  cases the current example-based suite cannot. Optional, but high value per line for
  `postprocess/` and `arch/_heads.py`.
- **T20. Long-tail low-coverage modules** worth a targeted pass once the P0s land:
  `benchmark/_dota_evaluator.py` 11%, `tracking/_reid.py` 19%, `tracking/_chroma_gallery.py` 20%,
  `benchmark/_comparison.py` 24%, `tracking/_clip_reid.py` 27%, `export/_metadata.py` 76%,
  `cli/_main.py` 73% (488-523, 569-613, 678-757, 797-826 unexecuted — the `track`, `count` and
  `export` command bodies).

---

## What is already good

Do not re-do these:

- **Volume and speed.** 2142 unit tests in 75 s with 80% line coverage is a genuinely healthy
  ratio; 31 source files are at 100%.
- **Error-path discipline.** Every backend has systematic tests for `DependencyError`,
  `BackendLoadError`, `InferenceError` mapping and lifecycle guards
  (`test_onnx_backend.py`, `test_tensorrt_backend.py`, `test_coreml_backend.py`,
  `test_shutdown.py`, `test_gpu_retry.py`, `test_oom_monitor.py`,
  `test_warmup_validation.py`). `postprocess/_nms.py` 98%, `_obb_nms.py` 99%,
  `arch/_heads.py` 99%, `tracking/_kalman.py` 98%, `pipeline/_scheduler.py` 97%.
- **Environment hygiene.** No direct `os.environ[...]` writes anywhere; 29 uses of
  `monkeypatch.setenv`/`patch.dict`. No network access in the unit tier. No module-level mutable
  test globals.
- **Determinism where it counts.** Seeded `default_rng` in 20+ places; explicit determinism tests
  at `test_nms.py:707,726`; tight numeric assertions (`atol=1e-12`/`1e-15`) in
  `test_tracking_optimizations.py:238,447`.
- **Genuinely valuable guard tests** that most suites lack:
  `test_torch_free_import.py` (import-time torch leakage), `test_public_api.py` (API surface),
  `test_py38_compat.py` (annotation-deferral AST guard), `test_serialization.py:109`
  (numpy-type leakage into JSON), `test_collector.py:659` (tracemalloc leak check).
- **The pyramid's shape is right.** `tests/unit` vs `tests/integration` is the correct split, the
  markers are declared, and `CONTRIBUTING.md:116-156` documents the intent clearly. The problem is
  that the integration half was never wired up — the design is sound, the execution is missing.
