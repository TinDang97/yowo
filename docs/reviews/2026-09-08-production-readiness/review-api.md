# Public API / Backend Protocol / Type Safety review — yowo production readiness

Reviewed at `main` @ `3340bb3`. No source file was modified.

## Verdict

**Not launch-ready as a library other teams pin — but the distance is short and the work is
mostly bookkeeping, not redesign.** The internals are genuinely good: pyright strict passes with
0 errors, `py.typed` ships, `Any` is 0 in `types.py`/`errors.py`, and all five backends implement
the full nine-member `InferenceBackend` Protocol. The blocker is the **identity layer**: at
version 2.5.0 the package reports `__version__ == "2.4.1"`, stamps that wrong version into every
exported model's provenance sidecar, has no CHANGELOG entry for 2.5.0 at all, ships a
`__all__`-exported `ExportResult` type that `export_model()` has never returned, and offers two
ops-shaped CLI commands (`health`, `metrics`) that are hard-coded stubs — `yowo health` exits 2
unconditionally.

The single biggest gap: **there is no stability contract.** 191 public names across 19 modules,
zero `DeprecationWarning` anywhere in `src/`, no documented deprecation or support policy, and a
`__all__` that omits 15 names the module re-exports explicitly. A 2.5.0 semver number promises
three years of compatible evolution that nothing in this repo can currently deliver.

---

## P0 — blocks launch

### A1. The package reports the wrong version, and stamps it into exported artifacts

- **Symptom:** `yowo.__version__` is `"2.4.1"` while the installed distribution, the git tag, and
  `yowo --version` all say `2.5.0`. Worse, `export_model()` writes `yowo.__version__` into the
  metadata sidecar of every exported ONNX/TensorRT/OpenVINO/CoreML artifact — so artifacts built
  by 2.5.0 are permanently labelled as produced by 2.4.1.
- **Evidence:**
  - `src/yowo/__init__.py:123` — `__version__ = "2.4.1"`
  - `pyproject.toml:3` — `version = "2.5.0"`; `git tag --sort=-v:refname | head -1` → `v2.5.0`
  - `src/yowo/export/_exporter.py:199` — `yowo_version=getattr(yowo, "__version__", "0.1.0")`
  - `src/yowo/export/_metadata.py:31` — `yowo_version: str` (persisted to the `.json` sidecar)
  - Reproduced:
    ```
    $ uv run python -c "import yowo,importlib.metadata as m; print(yowo.__version__, m.version('yowo'))"
    2.4.1 2.5.0
    $ uv run yowo --version
    yowo, version 2.5.0
    ```
  - `pyproject.toml:128` — `version_toml = ["pyproject.toml:project.version"]`. semantic-release is
    configured to bump **only** `pyproject.toml`. Nothing updates `src/yowo/__init__.py`, so this
    drift is structural and recurs on every release.
- **Why it blocks:** Version is the one field every downstream consumer trusts unconditionally —
  for pinning, for bug triage, and (via the sidecar) for auditing which library built a model
  running in production. A library whose `__version__` lies cannot be used to reason about a
  deployed artifact.
- **Smallest fix:** Delete the literal and derive it —
  `__version__ = importlib.metadata.version("yowo")` with a `PackageNotFoundError` fallback — or add
  `version_variables = ["src/yowo/__init__.py:__version__"]` to `[tool.semantic_release]`. Prefer
  the former; it cannot drift.

### A2. `ExportResult` is a public, documented, tested return type that nothing ever returns

- **Symptom:** `ExportResult` is in `yowo.__all__` and `yowo.types.__all__`, has a unit test class,
  and is documented in two module READMEs as the return type of `export_model()`. `export_model()`
  actually returns `ExportMetadata`, whose field names and types differ
  (`output_path: Path` vs `file_path: str`; `export_time_s` vs `export_duration_sec`). A user who
  follows the shipped documentation writes code that fails.
- **Evidence:**
  - `src/yowo/types.py:467-487` — `ExportResult` definition; `src/yowo/__init__.py:151` (`__all__`)
  - `src/yowo/export/_exporter.py` — `def export_model(...) -> ExportMetadata`
  - `src/yowo/README.md:155` — `from yowo import export_model, ExportMetadata, ExportResult`
  - `src/yowo/export/README.md:29,31,34,46,56,69` — documents `export_model() -> ExportResult`
  - `src/yowo/cli/README.md:83` — "`--json` … Print `ExportResult` as JSON to stdout"
  - `grep -rn "ExportResult" src/` outside `types.py`/`__init__.py` returns **only** README hits —
    zero production code constructs or consumes it.
- **Why it blocks:** This is a false promise in the exported namespace. Either it is dead and must
  be removed (a breaking change that gets harder every release), or it is the intended type and
  export is returning the wrong thing. Shipping it as-is guarantees a support burden.
- **Smallest fix:** Decide now. If `ExportMetadata` is the contract: drop `ExportResult` from both
  `__all__` lists, delete it, and correct the three READMEs. Do it before 2.5.0 is advertised, so
  the removal is not a breaking change against a real user.

### A3. `yowo health` and `yowo metrics` are stubs shaped exactly like production probes

- **Symptom:** `yowo health` always prints `{"status": "closed"}` and **always exits 2**,
  regardless of anything. `yowo metrics` always emits a zero-valued snapshot from a
  freshly-constructed collector — it never reads a running engine. Neither is documented in
  `README.md`, `docs/user-guide.md`, or `src/yowo/cli/README.md`.
- **Evidence:**
  - `src/yowo/cli/_main.py:1064-1078` — `report = {"status": "closed", ...}` then unconditional
    `sys.exit(2)`
  - `src/yowo/cli/_main.py:1090-1105` — `collector = MetricsCollector(enabled=True)` (fresh, empty),
    then `sys.exit(0)`
  - `grep -n "yowo health\|yowo metrics" docs/user-guide.md README.md src/yowo/cli/README.md` →
    no matches
- **Why it blocks:** These are the two command names an SRE reaches for first. A k8s liveness probe
  or a Prometheus textfile scraper wired to them will fail permanently (`health`) or silently
  report all-zeros forever (`metrics`) — the worst failure mode, because zeros look like a healthy
  idle service. Shipping commands that *look* like an ops contract and are not is worse than not
  shipping them.
- **Smallest fix:** Either remove both commands from the CLI group, or make `health` exit `0` and
  print a `status: no-engine` document that cannot be mistaken for a live reading. The full fix
  (an engine-attached health/metrics endpoint) is a feature, not a launch prerequisite.

### A4. No stability contract, no deprecation mechanism, at version 2.5.0

- **Symptom:** 191 public names across 19 modules, and nothing in the repo says which of them are
  stable, how long a name survives after it is deprecated, or how a user is told. There is not one
  `DeprecationWarning` in `src/`.
- **Evidence:**
  - Measured surface (`uv run python` over `yowo.__all__` + every submodule `__all__`):
    top-level 72, `arch` 16, `tracking` 20, `io` 13, `utils` 11, `benchmark` 8, `models` 7,
    `postprocess` 7, `backends` 6, `counter` 6, `events` 5, `pipeline` 5, `tune` 4, `hardware` 3,
    `export` 2, `metrics` 2, `batch` 2, `cache` 1, `cli` 1 → **191 total**.
  - `grep -rn "DeprecationWarning\|warnings.warn\|FutureWarning" src/` → 2 hits, both in
    `src/yowo/benchmark/_runner.py:97,104`, neither a deprecation.
  - `grep -in "deprecat\|stability\|stable api" README.md CONTRIBUTING.md` → `CONTRIBUTING.md:174-184`
    maps commit types to semver bumps and `CONTRIBUTING.md:286` forbids "Breaking public API"
    without an issue. That is a contributor rule, not a user-facing contract.
  - `src/yowo/__init__.py:126-198` — `__all__` omits 15 names the module re-exports with the
    explicit PEP-484 `as` form (public to type-checkers, invisible to `import *` and doc tooling):
    `CPUArch`, `DeviceCategory`, `DeviceType`, `EngineMetrics`, `EventBus`, `ExportConfig`,
    `GPUArch`, `IMAGE_EXTS`, `MetricsCollector`, `PreprocessedTensor`, `RTSP_SCHEMES`,
    `SourceCategory`, `VIDEO_EXTS`, `classify_device`, `classify_source`.
  - `pyproject.toml:11` still declares `"Development Status :: 3 - Alpha"` at 2.5.0 — the metadata
    and the version number contradict each other.
  - `git log v2.4.1..HEAD` shows four commits since the v2.5.0 tag, including
    `3a9b78b fix(engine): DetectionEngine labels detections with COCO names for any model` — an
    observable change to `BoundingBox.class_name`, released as a patch, with no CHANGELOG entry.
    `CHANGELOG.md:14-16` shows an empty `[Unreleased]` and **no `2.5.0` section at all**.
- **Why it blocks:** `PreprocessedTensor` is the exact type a third-party backend must accept in
  `infer()`, and it is in the ambiguous set. `yowo.arch` publicly exports 16 architecture symbols,
  freezing the model internals. Without a written policy and a warning mechanism, the first time
  this project needs to remove anything, it breaks users with no notice — which is precisely what
  a 2.x version number promises will not happen.
- **Smallest fix:** Three cheap moves. (1) Add a "Stability and deprecation" section to `README.md`:
  what is public (`yowo.__all__` + the listed submodule `__all__`s), what is not (anything
  underscore-prefixed, `yowo.arch`), and the rule "deprecated names warn for one minor and are
  removed no sooner than the next major". (2) Reconcile `__all__` with the re-exports —
  either add the 15 names or stop re-exporting them. (3) Bump the classifier to
  `5 - Production/Stable` (or `4 - Beta`) and backfill the 2.5.0 CHANGELOG section.

---

## P1 — before GA

### B1. Backend × contract-obligation matrix

Obligations are taken verbatim from the Protocol at `src/yowo/backends/__init__.py:34-108`.

| Obligation (source) | PyTorch | ONNX | TensorRT | OpenVINO | CoreML |
|---|---|---|---|---|---|
| 3 properties + 6 methods present | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DependencyError` when SDK missing | ✅ `_pytorch.py:47,104` | ✅ `_onnx.py:41,124` | ✅ `_tensorrt.py:46,133` | ✅ `_openvino.py:43,104` | ✅ `_coreml.py:32,77` |
| Missing file → typed error | ⚠️ `BackendLoadError` w/ raw torch text | ⚠️ raw ORT text | ⚠️ raw ORT text | ⚠️ raw OV text | ✅ pre-checks `path.exists()` `_coreml.py:81` |
| `ModelLoadError` (**documented** at `__init__.py:63`) | ❌ never | ❌ never | ❌ never | ❌ never | ❌ never |
| `DeviceError` on OOM / bad device | ⚠️ load() only `_pytorch.py:235-239` | ❌ → `BackendLoadError` | ❌ | ❌ | ❌ |
| State reset at top of `load()` | ❌ `_neck_hook_handle` not reset | ✅ `:106-120` | ✅ `:114-128` | ✅ `:94-100` | ⚠️ inside `try` |
| Handle nulled on load failure | ✅ `:236,242` | ✅ `:200` | ✅ `:220` | ✅ `:148` | ✅ `:123` |
| **Validate input dtype/ndim/shape in `infer()`** | ❌ | ❌ | ❌ | ❌ | ❌ |
| `infer()` errors → `InferenceError` | ⚠️ swallows OOM too `:317` | ✅ `:229` | ✅ `:249` | ✅ `:187` | ✅ `:155` |
| `unload()` releases **device** memory | ✅ `cuda.empty_cache()` `:334` | ❌ ref-drop only `:285` | ❌ ref-drop only `:305` | ❌ ref-drop `:200` | ❌ ref-drop `:160` |
| `unload()` idempotent / pre-`load()` safe | ✅ | ✅ | ✅ | ✅ | ✅ |
| `warmup()` warns on batch mismatch | ❌ | ✅ `:295-302` | ✅ `:316-323` | ❌ | ✅ `:167-179` |
| `warmup()` swallows all errors at DEBUG | ⚠️ `:353` | ⚠️ `:317` | ⚠️ `:340` | ⚠️ `:224` | ⚠️ `:192` |
| `set_source_id()` honoured | ✅ | no-op | no-op | no-op | no-op |
| Input node addressed by name | n/a | ✅ | ✅ | ❌ index `{0: …}` `_openvino.py:181` | ✅ |
| Primary output selection | model return | `outputs[0]` | `outputs[0]` | ⚠️ `next(iter(results.values()))` `:183` | `["output0"]` else first key `:146-149` |
| Dedicated unit-test file | ❌ none | ✅ `test_onnx_backend.py` | ✅ `test_tensorrt_backend.py` | ❌ none | ✅ `test_coreml_backend.py` |

The four gaps that matter most:

- **No input validation, anywhere.** The Protocol says "BCHW float32 array in [0, 1]"
  (`__init__.py:69-71`) and no backend enforces it. Passing a `float64` or HWC array produces five
  different native error texts wrapped in `InferenceError`, and on TensorRT the failure mode is
  whatever ORT does with a bad binding. **Fix:** one shared
  `_validate_input(tensor)` in `yowo/backends/_validate.py` (ndim == 4, `dtype == np.float32`,
  H/W match `input_shape` unless dynamic) called at the top of every `infer()`.
- **`DeviceError` is PyTorch-only.** A CUDA OOM under ONNX-CUDA or TensorRT surfaces as
  `BackendLoadError`/`InferenceError`, so a caller cannot write one `except DeviceError` and get
  consistent behaviour. **Fix:** lift the `"cuda"|"device"|"out of memory"` substring mapping out of
  `_pytorch.py:236-239` into a shared `_map_native_error()` used by all five.
- **`ModelLoadError` is documented in the Protocol and raised by nobody.** `grep -rn "raise ModelLoadError" src/`
  → zero hits. Same for `ExportUnsupportedError` and `TrackingError` — all three are in
  `yowo.__all__`, documented in `src/yowo/README.md:187,204,209,212`, and unreachable. **Fix:**
  raise them where documented, or delete them from `errors.py` and `__all__` now.
- **`unload()` promises "Release all device memory"** (`__init__.py:83`); four of five backends only
  drop a Python reference. On ONNX-CUDA and TensorRT, GPU memory is freed at GC, not at
  `engine.close()` — which breaks the model-swap pattern (`close()` then load a bigger model).
  **Fix:** `del self._session` before `= None`, and for the CUDA-EP paths document that release is
  GC-timed, or add an explicit `gc.collect()`.

There is also **no cross-backend conformance suite** — `ls tests/unit/ | grep backend` shows tests
for ONNX, TensorRT and CoreML only; PyTorch and OpenVINO have no backend test file, and nothing
parametrizes one set of contract assertions over all five. That is why the divergences above
survived. **Fix:** one `tests/unit/test_backend_conformance.py` parametrized over the five classes
asserting: missing SDK → `DependencyError`, missing file → typed error, `infer()` before `load()`
→ `InferenceError`, `unload()` twice is safe, bad-dtype input → `InferenceError`.

### B2. What `backend_instance=` promises a third-party author, and what it actually does

- **Symptom:** The open backend factory is real and tested, but three of its guarantees are wrong,
  and it is not documented for users.
- **Evidence:**
  - **It lies in `engine.selection`.** `src/yowo/engine.py:268-276` hard-codes
    `device_type=DeviceType.CPU, precision=Precision.FP32` for *any* injected backend. A
    third-party CUDA backend reports as CPU/FP32 forever, and
    `health_report().precision_current` (`engine.py:373`) and `_is_cuda` (`engine.py:566`) both
    read from it — so the OOM monitor never starts for a custom CUDA backend either.
  - **It downloads weights the custom backend does not want.** `src/yowo/engine.py:443-447`: with
    `backend_instance` set and no `weights_path`, `resolve_weights(self._spec)` runs and will fetch
    `yolo26n.pt` from GitHub (`src/yowo/models/_weights.py:26-60`) before handing the path to the
    user's backend. Only `model_builder` short-circuits this. A custom backend loading its own
    artifact still triggers a network download on first `load()`.
  - **Nothing validates the injected object.** `grep -n "isinstance" src/yowo/engine.py` → no hits.
    A backend missing `clear_kv_cache` or `set_source_id` fails later, deep inside
    `_stream_dispatch`, with an `AttributeError` rather than at construction. And the
    `@runtime_checkable` Protocol is only attribute-name deep — verified: an object whose
    `infer()` returns the string `"not an array"` passes `isinstance(x, InferenceBackend)`.
  - **It is undocumented.** `grep -rn "backend_instance" README.md docs/` → only two hits in
    `docs/user-guide.md:464,577`, both inside a dumped `__init__` signature. No prose, no example,
    no statement of what a backend author must guarantee.
- **Why it blocks GA:** This is the extension point that decides whether yowo is a library or a
  framework. Shipping it undocumented, unvalidated, and reporting false device metadata means the
  first third-party backend author debugs all three of these in production.
- **Smallest fix:** (a) accept an optional `selection: BackendSelection` alongside
  `backend_instance`, or derive `device_type`/`precision` from the backend rather than assuming;
  (b) skip `resolve_weights()` when `self._user_provided_backend and self._spec.weights_path is None`,
  exactly as the `model_builder` branch already does; (c) `if not isinstance(backend_instance, InferenceBackend): raise ConfigError(...)`
  at `engine.py:267`; (d) a 30-line "Writing a custom backend" section in `README.md` that states
  the input contract (BCHW float32 [0,1]) and the error-mapping obligation.

### B3. The three engines return three structurally incompatible result types

- **Symptom:** No consumer can write one function that handles a detection, an OBB detection, and a
  classification result, because the shared concepts live in different places or are missing.
- **Evidence:**
  | field | `Detection` (`types.py:286`) | `OBBDetection` (`types.py:381`) | `ClassificationResult` (`types.py:414`) |
  |---|---|---|---|
  | `source_id` | ❌ (via `.frame.source_id`) | ✅ top-level | ✅ top-level |
  | `frame_index` | ❌ (via `.frame.frame_index`) | ✅ top-level | ✅ top-level |
  | `frame` | ✅ required | ⚠️ `Frame \| None = None` | ❌ absent |
  | `backend` | ✅ | ❌ | ✅ |
  | `model_spec` | ✅ | ❌ | ✅ |
  | `to_dict()` includes `"model"` | ✅ `:327` | ❌ `:399-406` | ✅ `:453` |
  | convenience props | `num_boxes`, `has_detections` | none | none |

  `BoundingBox` has `area`/`as_xyxy` (`types.py:260-270`); `OBBBox` has neither.
- **Why it blocks GA:** Every one of these differences is a breaking change to fix later. A user
  writing a generic sink (`log(result.source_id, result.backend)`) discovers per-engine special
  cases at runtime.
- **Smallest fix:** Add the missing fields with defaults now, while it is still additive:
  `source_id`/`frame_index` properties on `Detection` delegating to `self.frame`, and
  `backend`/`model_spec` on `OBBDetection`. Additive today, breaking in six months.

### B4. `astream()` erases the result type for every engine

- **Symptom:** `stream()` is correctly typed per engine (`Iterator[Detection]`,
  `Iterator[ClassificationResult]`, `Iterator[OBBDetection]`). `astream()` is defined once on
  `BaseEngine` as `AsyncIterator[Any]` and **never overridden** — `grep -rn "def astream" src/`
  returns only `engine.py:739` and the helper in `_async.py:45`. Every async consumer loses all
  typing.
- **Evidence:** `src/yowo/engine.py:739`; no override in `classify_engine.py` or `obb_engine.py`.
- **Smallest fix:** Make `BaseEngine` generic in its result type (`Generic[R]`, `stream() -> Iterator[R]`,
  `astream() -> AsyncIterator[R]`), or add three three-line overrides that narrow the annotation.

### B5. `DetectionEngine(config, batch_size=8)` silently discards the keyword arguments

- **Symptom:** The constructor accepts both a `config` object and ~20 duplicate keyword arguments.
  When `config` is given, every keyword is dropped without an error or a warning.
- **Evidence:** `src/yowo/engine.py:965-968` — `if config is not None: cfg = config` and the
  `else:` branch is the only place the keywords are read. Same shape in
  `classify_engine.py:88` and `obb_engine.py:92`.
- **Why it blocks GA:** `DetectionEngine(cfg, batch_size=8)` reads as an override and is a no-op.
  This is a silent-wrong-behaviour trap in the most-used constructor in the library.
- **Smallest fix:** `raise ConfigError("pass either config or keyword overrides, not both")` when
  `config is not None` and any override differs from its default — or apply the overrides via
  `dataclasses.replace(config, ...)`, which is what a reader expects.

### B6. `_VALID_LOG_LEVELS` is a public constructor field that defeats its own validation

- **Symptom:** It is declared as a plain annotated attribute inside three `@dataclass`es, not as a
  `ClassVar`, so it becomes a real field: it appears in the public `__init__` signature, in
  `dataclasses.asdict()`, and it can be overridden to bypass the very check it exists to enforce.
  It also makes the config unserialisable.
- **Evidence:** `src/yowo/config.py:152` (`InferenceConfig`), and the same line in
  `ClassificationConfig` and `OBBConfig`. Reproduced:
  ```
  $ uv run python -c "..."
  InferenceConfig  private fields in ctor: ['_VALID_LOG_LEVELS']
  accepted bogus levels -> BOGUS ['BOGUS']
  json.dumps(asdict(cfg))   -> TypeError: Object of type frozenset is not JSON serializable
  yaml.safe_dump(asdict(cfg)) -> RepresenterError: cannot represent an object, frozenset({...})
  ```
- **Why it blocks GA:** `load_config()` (`config.py:513`) reads YAML, but a user cannot write the
  round-trip — `yaml.safe_dump(asdict(cfg))` raises. And a leading-underscore name in a public
  constructor signature can never be removed compatibly once anyone touches it.
- **Smallest fix:** `_VALID_LOG_LEVELS: ClassVar[frozenset[str]] = frozenset({...})` in all three
  dataclasses. Three-word change; do it before more users serialize configs.

### B7. Retry exhaustion returns a detection-shaped array to all three engines

- **Symptom:** When `backend.infer()` fails all three retries, `BaseEngine._infer_with_retry`
  returns `np.zeros((1, 0, 6), dtype=np.float32)` and the engine continues. That shape is
  meaningful only for detection; the classification and OBB postprocessors receive it too.
- **Evidence:** `src/yowo/engine.py:815` — `return np.zeros((1, 0, 6), dtype=np.float32)`, in
  `BaseEngine`, reached by `ClassificationEngine._process_batch` (`classify_engine.py:179`) and
  `OBBEngine._process_batch` (`obb_engine.py:201`), whose validators expect
  `(B, num_classes)` and `(B, 4+nc+1, A)` respectively.
- **Why it blocks GA:** Two problems. The generic one: a total inference failure is reported as
  "zero detections", which downstream logic cannot distinguish from "nothing in frame" — silent
  data loss under GPU pressure. The specific one: for classify/OBB the fallback shape is simply
  wrong and will produce a second, confusing exception.
- **Smallest fix:** Make the fallback abstract — a `_empty_output()` hook overridden per engine —
  and gate the swallow behind a config flag so a caller can opt into `raise` instead of zeros.

### B8. `model_builder: Any` where an exported Protocol exists

- **Symptom:** `ModelBuilder` is a `@runtime_checkable` Protocol, exported from `yowo.__all__`, and
  documented in `README.md:276-290`. Every parameter that accepts it is typed `Any`, so pyright
  checks nothing at the one place a user could get it wrong.
- **Evidence:** `src/yowo/engine.py:238` and `:920`, `src/yowo/classify_engine.py:92`,
  `src/yowo/obb_engine.py:96`, `src/yowo/backends/_pytorch.py:40` — all `model_builder: Any | None = None`.
  The Protocol itself is at `src/yowo/backends/__init__.py:112-139`.
- **Smallest fix:** Change the annotation to `ModelBuilder | None` (import under `TYPE_CHECKING` to
  keep the import graph clean). Zero runtime cost, full checker coverage.

### B9. CLI contract gaps

- **`classify` has no machine-readable output.** `detect`, `detect-obb`, `track` and `count` all
  take `--json` (`_main.py:166,351,544,655`); `classify` does not (`_main.py:760-826`) and prints
  only `f"[{result.frame_index}] Top-1: cls_0004 (0.912)"` at `:823` — no class names, no JSON, no
  `--output`. `ClassificationConfig` also has no `class_names` field where `InferenceConfig` does
  (`config.py:131`), so the CLI cannot label anything.
- **CoreML is unreachable from the CLI.** All five `--backend` options are
  `click.Choice(["auto", "pytorch", "onnx", "tensorrt", "openvino"])`
  (`_main.py:140,338,533,623,774`) — `BackendType.COREML` exists and `CoreMLBackend` ships, but no
  command can select it.
- **Exit codes carry no information.** Every failure path is `sys.exit(1)`
  (`:97,109,291,437,505,523,613,757,826`). A script cannot distinguish "model not found" from
  "source unreachable" from "backend unavailable" — a distinction `yowo.errors` already encodes.
- **`src/yowo/cli/README.md` documents 4 of 13 commands** (`detect`, `export`, `info`, `models`);
  `classify`, `detect-obb`, `track`, `count`, `benchmark`, `tune`, `batch`, `health`, `metrics` are
  absent, and line 83 documents a `--json` output type (`ExportResult`) that does not exist.
- **Smallest fix:** Add `--json` to `classify` (and `class_names` to `ClassificationConfig`); add
  `"coreml"` to the five `Choice` lists; map the top-level `yowo.errors` classes to distinct exit
  codes (2 = config/usage, 3 = model, 4 = source, 5 = backend) in one shared handler; regenerate
  the CLI README from `--help`.

### B10. `StreamConfig.auto_reconnect` is a public field that does nothing

- **Symptom:** A frozen public dataclass field whose own docstring says the behaviour is not
  implemented. Setting `auto_reconnect=True` on an RTSP stream produces no reconnect and no warning.
- **Evidence:** `src/yowo/types.py:163-178` — "Currently stubbed — field is defined for API
  stability; reconnect loop is not implemented in this release." Same for
  `reconnect_backoff_base_s` and `reconnect_backoff_max_s`.
- **Smallest fix:** Until the loop exists, `warnings.warn("auto_reconnect is not implemented", ...)`
  in `__post_init__` when it is set to `True`. A silent no-op on a resilience knob is exactly the
  field an operator will trust and then not have.

### B11. Type-checking strictness does not cover the versions the package claims to support

- **Symptom:** `requires-python = ">=3.8"` and `ruff target-version = "py38"`, but pyright runs at
  `pythonVersion = "3.11"` and `[tool.uv] environments = ["python_version >= '3.9'"]`. Python 3.8 —
  the version added specifically for Jetson Nano in 2.4.1 — is never type-checked and cannot even
  be resolved by the project's own tooling.
- **Evidence:** `pyproject.toml:7,80,91,101`. The 2.4.1 CHANGELOG documents four runtime 3.8
  incompatibilities found by **manual** testing, which is the direct consequence.
- **Also:** `typeCheckingMode = "strict"` with five rules disabled — `reportUnknownMemberType`,
  `reportUnknownVariableType`, `reportUnknownArgumentType`, `reportUnknownParameterType`,
  `reportMissingTypeArgument` (`pyproject.toml:106-110`). Justified for untyped SDKs, but it is
  applied package-wide, which is how `def run_benchmark(...) -> dict` (`cli/_main.py:27`) — a bare
  generic on a public wrapper — passes strict mode.
- **Smallest fix:** Run pyright a second time at `pythonVersion = "3.8"` in the quality gate, and
  scope the five suppressions to the modules that import untyped SDKs
  (`hardware/`, `backends/_*`) via a pyright `executionEnvironments` block rather than globally.

---

## P2 — later

- **`OBBEngine` ships two identical stream methods.** `stream()` (`obb_engine.py:276`) and
  `stream_obb()` (`obb_engine.py:297`) have byte-identical bodies; `_convenience.detect_obb` calls
  `stream_obb` (`_convenience.py:250`) while `detect` calls `stream` (`_convenience.py:143`). Pick
  one, alias the other with a deprecation.
- **`InferenceEngine` is the only documented name.** `grep -c` gives 33 uses of `InferenceEngine`
  in `README.md` and **0** of `DetectionEngine`, though `engine.py:1069` calls the former a
  "backward-compatibility alias". The alias is not a trap — it is a plain
  `InferenceEngine = DetectionEngine` binding, so `isinstance`, subclassing and pickling all behave
  — but the docs invert the intended direction, which makes the alias unremovable. Either promote
  `DetectionEngine` in the docs or stop calling the alias legacy.
- **`yowo.arch` publicly exports 16 architecture symbols** (`ClassifyModel`, `OBBModel`,
  `YOLOModel`, `Classify`, `OBBHead`, `dist2rbox`, `load_weights`, …). That freezes the internal
  model implementation as a public contract and drags `torch` onto the import path of anyone who
  touches it. Consider demoting to an underscore module or documenting it as explicitly unstable.
- **`OpenVinoBackend` addresses I/O positionally.** Non-KV inference feeds `{0: tensor.data}`
  (`_openvino.py:181`) while the KV path uses `self._input_name` (`:170`), and the output is
  `next(iter(results.values()))` (`:183`) — dict insertion order. Both work today and both are
  fragile to a model with reordered outputs.
- **`close()` swallows the unload failure silently.** `src/yowo/engine.py:705-707` —
  `except Exception: pass` with no log line. A backend that fails to release GPU memory leaves no
  trace. Add `logger.warning(...)`.
- **`types.py:647` — `_field = field`** with the comment "Silence F401 for field". A module-level
  alias existing only to appease a linter; use `# noqa: F401` on the import instead.
- **`yowo.pipeline` re-exports `StreamConfig`** (`pipeline/__init__.py:60`) which also lives in
  `yowo.types` — two import paths for one type, both public.
- **`run_benchmark(...) -> dict`** (`cli/_main.py:27`) — bare generic on a public wrapper; should be
  `dict[str, object]`.

---

## What is already good

Do not re-do this work:

- **Type checking is genuinely clean.** `uv run pyright src/yowo/` → `0 errors, 0 warnings, 0 informations`
  under `typeCheckingMode = "strict"`. `uv run ruff check src/` exits 0.
- **`py.typed` ships** (`src/yowo/py.typed`) and is inside the wheel package root
  (`pyproject.toml:88`), so downstream type checkers see the annotations.
- **`Any` is concentrated at the boundaries, not smeared.** `types.py` = 0 occurrences,
  `errors.py` = 0, `postprocess/__init__.py` = 0, `tune/` = 0, `utils/` = 0. The 16 in `engine.py`
  are mostly the `_process_batch` polymorphic return and the untyped-torch escape hatches.
- **Precise numpy typing throughout the hot path** — `NDArray[np.float32]` and `NDArray[np.uint8]`
  rather than bare `ndarray`, in `PreprocessedTensor.data`, `Frame.pixels`, and every backend
  `infer()` signature.
- **The exception hierarchy is well designed** (`errors.py:1-25`) — single `YowoError` root,
  sensible subclassing, `DependencyError` carries a machine-readable `package` plus an install
  command. The problem is unused branches, not bad design.
- **All five backends implement the complete nine-member Protocol**, all defer their SDK imports to
  `load()` so `import yowo.backends` works on a bare machine, and all null their handle on load
  failure. The lazy `create_backend()` dispatch (`backends/__init__.py:174-199`) is correct.
- **`__all__` is present and sorted in all 19 modules** — the surface is enumerable, which is
  exactly what makes the P0-A4 fix cheap.
- **Config validation is thorough** — `InferenceConfig.__post_init__` (`config.py:153-190`) checks
  every numeric range and cross-validates `num_classes` against `len(class_names)`.
- **Both `Self`-returning context managers are correct** (`engine.py:716-733`), so
  `with DetectionEngine(...) as e:` narrows to the subclass in both sync and async form.

---

Status: DONE_WITH_CONCERNS
Summary: The type system and backend Protocol design are production-grade — pyright strict passes clean, `py.typed` ships, and all five backends implement the full contract — but the identity and stability layer is not: the package reports the wrong version and stamps it into exported artifacts, exports a return type nothing returns, ships two stub ops commands, and has no deprecation policy or mechanism behind a 2.5.0 semver promise. Below that, five backends diverge on error mapping, input validation and resource release with no conformance suite to catch it.
Top 3 P0s: (1) `yowo.__version__` is "2.4.1" against dist 2.5.0 and is written into every export sidecar — semantic-release never updates `__init__.py`, so it recurs. (2) `ExportResult` is in `__all__`, tested, and documented in two READMEs as `export_model()`'s return type, which is actually `ExportMetadata` with different fields. (3) `yowo health` always exits 2 and `yowo metrics` always reports zeros — undocumented stubs shaped exactly like the probe endpoints an SRE will wire up.
