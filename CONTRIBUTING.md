# Contributing to yowo

Thank you for your interest in contributing. This document covers everything you need to know
to get a change merged: environment setup, code rules, testing requirements, and the PR process.

---

## Table of Contents

1. [Before you start](#before-you-start)
2. [Development setup](#development-setup)
3. [Architecture overview](#architecture-overview)
4. [Code rules](#code-rules)
5. [Testing requirements](#testing-requirements)
6. [Commit format](#commit-format)
7. [Pull request process](#pull-request-process)
8. [Specific contribution guides](#specific-contribution-guides)
9. [What not to contribute](#what-not-to-contribute)

---

## Before you start

- **Open an issue first** for any non-trivial change (new feature, breaking fix, new backend).
  Describe the problem and your proposed approach. This avoids wasted effort.
- **Small PRs only.** One logical change per PR. A PR that touches 10 unrelated files will
  be closed without review.
- **No ultralytics dependency in the code.** yowo's code is intentionally ultralytics-free
  (Apache-2.0 clean code — no `ultralytics` import anywhere in `src/`). Do not reintroduce
  it for any reason. This claim is about the code only: the default model weights are a
  separate concern, Ultralytics AGPL-3.0 assets fetched at runtime — see
  [Model weights and licensing](README.md#model-weights-and-licensing).

---

## Development setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/TinDang97/yowo.git
cd yowo

# Install all dev dependencies
uv sync --group dev

# Install pre-commit hooks (runs ruff + pyright + pytest on every commit)
uv run pre-commit install
```

**Verify the setup:**

```bash
uv run ruff check src/ tests/ --quiet     # 0 errors
uv run pyright src/yowo/                  # 0 errors
uv run pytest tests/unit/ -x -q          # all pass
```

---

## Architecture overview

```
types.py  errors.py          ← leaf nodes (no internal imports)
    ↓
hardware/  models/  io/  postprocess/  arch/   ← depend only on types + errors
    ↓                                    ↓
backends/   ← depends on hardware + types + arch (PyTorch backend)
    ↓
export/     ← depends on models + hardware + arch
    ↓
engine.py   ← public orchestrator (InferenceEngine)
    ↓
cli/        ← Click entry point
```

**Critical rules for this graph:**

- **No circular imports.** A module may only import from layers above it in the graph.
- **No new public types outside `types.py`.** All domain primitives live there.
- **No new domain exceptions outside `errors.py`.** All exceptions inherit from `YowoError`.
- **Max 700 lines per file.** Split before adding if close to the limit.

---

## Code rules

### Non-negotiable

| Rule | Detail |
|------|--------|
| No placeholders | Every function must be fully implemented. No `pass`, `TODO`, `raise NotImplementedError` in production paths. |
| No mocks in production | Mocks are for tests only. |
| No blocking I/O in async | Do not use `time.sleep`, synchronous file I/O, or blocking network calls inside `async def`. |
| Type annotations | All public functions and methods must have complete type annotations. `pyright` strict mode must pass. |
| No `Any` leakage | `Any` is allowed only for untyped SDK boundaries (torch, onnxruntime). Add `# type: ignore[...]` with the specific code. |

### Style

- **ruff** enforces formatting and linting (`line-length = 100`).
- **Conventional Commits** for every commit message (see [Commit format](#commit-format)).
- Module-internal files are prefixed with `_` (e.g. `_selector.py`).
- Each module's `__init__.py` re-exports only the public surface.

### What good code looks like

```python
# Good: explicit types, no unnecessary branching
def scale_channels(base: int, config: ModelConfig) -> int:
    return make_divisible(min(base, config.max_channels) * config.width_mult, 8)

# Bad: untyped, opaque
def scale(b, c):
    return make_divisible(min(b, c.max_channels) * c.width_mult, 8)
```

---

## Testing requirements

### Structure

| Directory | Purpose |
|-----------|---------|
| `tests/unit/` | Pure unit tests — no hardware, no real weights, no network |
| `tests/integration/` | End-to-end CLI tests — may require GPU, marked `@pytest.mark.integration` |

### Requirements for every PR

- **All existing tests must pass.** No regressions.
- **New code must have unit tests.** Every public function, every error path.
- **Coverage target:** new code should not reduce overall branch coverage.
- Tests that require large model downloads are marked `@pytest.mark.slow`.

### Running tests

```bash
# Unit tests (required before every PR)
uv run pytest tests/unit/ -x -q --tb=short

# Single file
uv run pytest tests/unit/test_arch_blocks.py -x -q

# Single test by name
uv run pytest tests/unit/ -k "test_fuse_conv_and_bn" -x
```

### Test conventions

```python
# Use descriptive class names grouping related tests
class TestFuseConvAndBn:
    def test_weight_shape_preserved(self) -> None: ...
    def test_output_numerically_equivalent(self) -> None: ...
    def test_raises_on_mismatched_channels(self) -> None: ...

# Parametrize over variants, not copy-paste
@pytest.mark.parametrize("size", list(ModelSize))
def test_scale_channels_all_sizes(size: ModelSize) -> None: ...
```

---

## Commit format

yowo uses [Conventional Commits](https://www.conventionalcommits.org/). Every commit **must**
follow this format — it drives automated semantic versioning:

```
<type>(<scope>): <short summary>

<body — what changed and why>

<footer — breaking changes, issue references>
```

| Type | SemVer impact | When to use |
|------|--------------|-------------|
| `feat` | minor bump | New user-visible functionality |
| `fix` | patch bump | Bug fix |
| `perf` | patch bump | Performance improvement |
| `refactor` | patch bump | Internal restructure, no behaviour change |
| `test` | none | Adding or fixing tests only |
| `docs` | none | Documentation only |
| `chore` | none | Maintenance (deps, config, CI) |
| `ci` | none | CI/CD changes |
| `BREAKING CHANGE` in footer | **major bump** | Breaks public API |

**Examples:**

```
feat(backends): add OpenVINO INT8 backend

Implements InferenceBackend protocol for OpenVINO with INT8 quantization.
Supports NCHW and NHWC layouts. Hardware auto-detection in _selector.py.

Closes #42
```

```
fix(arch): correct C3k inheritance to derive from C3 not C2f

C3k was inheriting C2f (2-branch iterative) instead of C3 (3-conv CSP).
This caused shape mismatches in all backbone layers for n/s model sizes.
```

---

## Pull request process

1. **Fork** the repository and create a branch from `main`.
   Branch names: `feat/<slug>`, `fix/<slug>`, `chore/<slug>`.

2. **Implement** the change following all rules above.

3. **Run the full quality gate locally** before pushing:
   ```bash
   uv run ruff check src/ tests/ --quiet
   uv run ruff format src/ tests/ --check --quiet
   uv run pyright src/yowo/
   uv run pytest tests/unit/ -x -q --tb=short
   ```
   All four commands must produce no errors. The pre-commit hooks enforce this automatically.

4. **Open the PR** against `main` with:
   - A clear title in Conventional Commit format (`fix(scope): description`)
   - A description answering: *What changed?* *Why?* *How was it tested?*
   - Reference to the issue it addresses (`Closes #N`)

5. **CI must pass.** The release workflow runs the same quality gate.
   Fix any failures before requesting review.

6. **One approval** from a maintainer is required to merge.
   Address all review comments before re-requesting.

---

## Specific contribution guides

### Adding a new backend

A backend implements the `InferenceBackend` protocol in `src/yowo/backends/_protocol.py`.

1. Create `src/yowo/backends/_<name>.py`.
   Required methods: `load()`, `warmup()`, `infer()`, `unload()`.
   Required property: `backend_type: BackendType`.

2. Register in `BackendType` enum in `src/yowo/types.py`.

3. Add to the fallback chain in `src/yowo/backends/_selector.py`.

4. Write unit tests in `tests/unit/test_<name>_backend.py` covering:
   - All error paths (`BackendLoadError`, `DeviceError`, `InferenceError`)
   - `load()` before `infer()` guard
   - `unload()` idempotency
   - `warmup()` no-op when not loaded

5. SDK imports must be deferred inside `load()` — importing the module must never
   fail on machines without the backend's SDK installed.

6. Run `backend-compliance-reviewer` agent before opening the PR:
   > "review `_mybackend.py` with backend-compliance-reviewer"

### Modifying the public API (`types.py`, `errors.py`, `engine.py`)

1. Open an issue first — API changes affect all downstream users.
2. Maintain backward compatibility wherever possible.
3. Run `api-contract-guardian` agent on your diff before opening the PR:
   > "run api-contract-guardian on this diff"
4. Document the change in `CHANGELOG.md` under `Unreleased`.

### Adding a new YOLO family

Register in `src/yowo/models/_registry.py` via `register(ModelMeta(...))`.
Add architecture support in `src/yowo/arch/` following the existing YOLO11/26 patterns.
No other files need to change for registration.

---

## What not to contribute

These will be closed without review:

- **Reintroducing ultralytics** as a dependency for any purpose.
- **Training code.** yowo is inference-only.
- **Files > 700 lines.** Split the module first.
- **PRs without tests** for new or fixed functionality.
- **Commits that don't follow Conventional Commits format.**
- **Breaking public API** without a prior issue and maintainer agreement.
- **Refactoring unrelated to the stated purpose of the PR.**

---

## Questions?

Open a [GitHub Discussion](https://github.com/TinDang97/yowo/discussions) for questions,
or a [GitHub Issue](https://github.com/TinDang97/yowo/issues) for bugs and feature requests.
