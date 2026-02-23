# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dev dependencies
uv sync --group dev

# Quality gates (run before every commit)
uv run ruff check src/ tests/       # lint
uv run ruff format src/ tests/      # format
uv run pyright src/yowo/            # type-check (strict mode, Linux platform assumed)
uv run pytest tests/unit/ --cov=yowo --cov-report=term-missing   # unit tests

# Run a single test file
uv run pytest tests/unit/test_engine.py -x -q

# Run a single test by name
uv run pytest tests/unit/test_selector.py -k "test_select_backend_pytorch_fallback" -x

# Run integration tests (requires hardware setup)
uv run pytest tests/integration/ -x -q

# CLI from source
uv run yowo info
uv run yowo detect image.jpg --model yolo26n
uv run yowo models
```

Pre-commit hooks run ruff (fix+format), pyright, and `pytest tests/unit -x -q --tb=short` automatically on staged `src/` or `tests/` files.

## Architecture

**yowo** implements native YOLO11 and YOLO26 architectures for inference and export, with hardware auto-detection, backend fallback chains, stream resilience, and production CLI/Python API. No ultralytics dependency.

### Module dependency graph (no circular deps)

```
types.py  errors.py          ← leaf nodes
    ↓
hardware/  models/  io/  postprocess/  arch/   ← depend only on types + errors
    ↓                                    ↓
backends/   ← depends on hardware + types + arch (PyTorch backend)
    ↓
export/     ← depends on models + hardware + arch
    ↓
engine.py   ← wires all modules (InferenceEngine)
    ↓
cli/        ← Click entry point (yowo.cli._main:cli)
```

### Key files

| File | Role |
|------|------|
| `src/yowo/types.py` | All core primitives: `Frame`, `PreprocessedTensor`, `Detection`, `BoundingBox`, `ModelSpec`, enums. Frozen dataclasses, logically immutable. |
| `src/yowo/errors.py` | Exception hierarchy rooted at `YowoError`. |
| `src/yowo/engine.py` | `InferenceEngine` — the public orchestrator. Lifecycle: `__init__` → `load()` → `detect()`/`stream()` → `close()`. Context manager supported. |
| `src/yowo/backends/_selector.py` | Pure-function backend auto-selection and fallback chain. Priority: TensorRT → ONNX(CUDA) → OpenVINO → ONNX(CPU) → PyTorch. |
| `src/yowo/models/_registry.py` | Model family registry mapping `(ModelFamily, ModelSize)` → `ModelMeta`. Add new YOLO families here only. |
| `src/yowo/io/_source.py` | `FrameSource` protocol + `open_source()` factory. Handles images, video files, RTSP (auto-reconnect), webcam, directories. |
| `src/yowo/io/_decode.py` | `preprocess()` — letterbox + normalize → `PreprocessedTensor` (BCHW float32). |
| `src/yowo/postprocess/_nms.py` | `postprocess()` — decodes raw backend tensors → `Detection` objects; applies NMS for backends with raw proposals. |
| `src/yowo/hardware/_detect.py` | One-time hardware detection, result cached for session lifetime. |
| `src/yowo/arch/_yolo.py` | `YOLOModel(nn.Module)` — assembles backbone + neck + head. `build_model()` factory + `fuse()` for inference. |
| `src/yowo/arch/_weights.py` | `load_weights()` — loads `.pt` checkpoint weights into native `YOLOModel`, maps state_dict keys. |
| `src/yowo/export/_exporter.py` | `export_model()` — `torch.onnx.export` with fused native model, TensorRT/OpenVINO conversion, `.yowo.json` sidecar. |
| `src/yowo/config.py` | `InferenceConfig`, `ExportConfig`, `load_config()` (YAML + env var override). |

### Data flow

```
open_source() → FrameSource → Frame (BGR uint8 HWC)
  → preprocess() → PreprocessedTensor (float32 BCHW)
  → InferenceBackend.infer() → NDArray[float32]
  → postprocess() → Detection (Frame + BoundingBox tuple)
```

### Backend implementations

Each backend lives in `src/yowo/backends/_<name>.py` and implements the `InferenceBackend` protocol: `load()`, `warmup()`, `infer()`, `unload()`. All are optional extras — unused backends never import their SDK.

Weight cache: `~/.cache/yowo/weights/`

### Adding a new YOLO family

Register in `src/yowo/models/_registry.py` via `register(ModelMeta(...))`. No other files need to change.

## Testing conventions

- `tests/unit/` — pure unit tests, no hardware required, no real model weights
- `tests/integration/` — end-to-end CLI tests, may require GPU; marked `@pytest.mark.integration`
- Use `@pytest.mark.slow` for tests requiring large model downloads
- Pyright is configured for Python 3.11, strict mode, Linux platform (`pythonPlatform = "Linux"`)

## Code conventions

- All public types are in `types.py` — no new domain types elsewhere
- Module internal files are prefixed with `_` (e.g. `_selector.py`, `_decode.py`)
- Each module's `__init__.py` re-exports only its public surface
- `field` from `dataclasses` is re-exported from `types.py` for submodule use
- Max file length: 700 lines

## Build Command

`/build <feature-description>` orchestrates the full development lifecycle in four phases:

```
SPECIFY → PLAN → IMPLEMENT → MEMORIZE
```

| Phase | Output | Gate |
|-------|--------|------|
| SPECIFY | Spec artifact: FR/NFR/AC/risks | User approval |
| PLAN | TaskCreate entries, dependency graph | User approval |
| IMPLEMENT | Code + agent reviews + quality gates | ruff + pyright + pytest all pass |
| MEMORIZE | Memory file + CLAUDE.md update + production claim checklist | All prior gates |

The command will not print `PRODUCTION READY` unless lint, type-check, tests, and all relevant agent reviews pass. See `.claude/skills/build/SKILL.md` for full orchestration rules.

## Subagents

Four project-level agents live in `.claude/agents/`. Invoke them by name when the task matches.

### When to invoke each agent

| Trigger | Agent | Invoke as |
|---------|-------|-----------|
| Writing or modifying any file in `backends/` | `backend-compliance-reviewer` | "review `_onnx.py` with backend-compliance-reviewer" |
| Optimizing engine/backend/io/postprocess hot paths | `inference-perf-auditor` | "audit inference-perf-auditor on `engine.py`" |
| After implementing a feature or fixing a bug | `test-coverage-guardian` | "check test coverage with test-coverage-guardian for `_selector.py`" |
| Before merging changes to `types.py`, `errors.py`, `engine.py`, or any `__init__.py` | `api-contract-guardian` | "run api-contract-guardian on this diff" |

### Mandatory agent gates

These are non-negotiable before committing:

1. **New or modified backend** → run `backend-compliance-reviewer` before writing tests
2. **Any change to `types.py` or `errors.py`** → run `api-contract-guardian` before commit
3. **New feature implementation** → run `test-coverage-guardian` after writing code, before declaring done

### Parallel agent pattern

When implementing a new backend end-to-end, run all relevant agents in parallel after the implementation is written:

```
Simultaneously:
  - backend-compliance-reviewer   → protocol + error mapping
  - inference-perf-auditor        → hot-path performance
  - test-coverage-guardian        → coverage gaps
```

Only proceed to commit after all three return APPROVE / no P0 issues / no uncovered error paths.
