# Technology Stack

**Analysis Date:** 2026-03-07

## Languages

**Primary:**
- Python 3.11+ - All source code (`src/yowo/`, `tests/`)

**Secondary:**
- YAML - Configuration files (`pyproject.toml` references, CI workflows)

## Runtime

**Environment:**
- Python 3.11 (pinned in `.python-version`)
- Supports 3.11 and 3.12 (classifiers in `pyproject.toml`)

**Package Manager:**
- `uv` (Astral) - All dependency management and script running
- Lockfile: `uv.lock` present

## Frameworks

**Core:**
- No web framework - This is an inference library, not a web app
- Click 8.1+ - CLI framework (`src/yowo/cli/_main.py`)
- PyYAML 6.0+ - Configuration loading (`src/yowo/config.py`)
- NumPy 1.24+ - Tensor and array operations (core dependency)
- OpenCV (headless) 4.8+ - Image/video I/O and preprocessing

**Testing:**
- pytest 8.0+ - Test runner (config in `pyproject.toml` `[tool.pytest.ini_options]`)
- pytest-asyncio 0.24+ - Async test support (`asyncio_mode = "auto"`)
- pytest-cov 5.0+ - Coverage reporting
- pytest-mock 3.14+ - Mock fixtures
- pytest-timeout 2.3+ - Test timeout enforcement

**Build/Dev:**
- Hatchling - Build backend (`pyproject.toml` `[build-system]`)
- Ruff 0.9.10 - Linter and formatter (pinned version)
- Pyright 1.1+ - Static type checker (strict mode)
- pre-commit 3.8+ - Git hooks (`.pre-commit-config.yaml`)
- python-semantic-release - Automated versioning and changelog

## Key Dependencies

**Critical (always installed):**
- `numpy>=1.24` - Core tensor operations, preprocessing, postprocessing
- `opencv-python-headless>=4.8` - Frame reading, image decoding, resize, color conversion
- `click>=8.1` - CLI interface (`yowo detect`, `yowo classify`, etc.)
- `pyyaml>=6.0` - YAML config file parsing in `src/yowo/config.py`
- `requests>=2.31` - Weight file downloads from GitHub releases (`src/yowo/models/_weights.py`)
- `tqdm>=4.66` - Download progress bars

**Optional (inference backends):**
- `torch>=2.0` - PyTorch backend (`src/yowo/backends/_pytorch.py`), native YOLO arch
- `onnxruntime>=1.17` - ONNX CPU backend (`src/yowo/backends/_onnx.py`)
- `onnxruntime-gpu>=1.17` - ONNX CUDA backend
- `tensorrt>=10.0` - TensorRT backend (`src/yowo/backends/_tensorrt.py`), manual install
- `openvino>=2024.0` - OpenVINO backend (`src/yowo/backends/_openvino.py`)
- `coremltools>=7.0` - CoreML backend (`src/yowo/backends/_coreml.py`)

**Optional (features):**
- `scipy>=1.11` - ByteTrack linear assignment (`src/yowo/tracking/`)
- `chromadb>=0.5.0` - Persistent embedding gallery (`src/yowo/tracking/_chroma_gallery.py`)
- `onnx>=1.12,<2.0` + `onnxslim>=0.1` - Model export (`src/yowo/export/`)

**Dev-only:**
- `ultralytics>=8.4.21` - Architecture comparison validation (`tmp/compare_arch.py`)
- `setuptools>=82.0.0` - Build dependency
- `onnxscript>=0.6.2` - ONNX export tooling

## Configuration

**Environment:**
- All runtime config via `YOWO_*` env vars (see `src/yowo/config.py` lines 6-26 for full mapping)
- Key env vars: `YOWO_MODEL_FAMILY`, `YOWO_MODEL_SIZE`, `YOWO_BACKEND`, `YOWO_DEVICE`, `YOWO_PRECISION`, `YOWO_BATCH_SIZE`, `YOWO_CONFIDENCE`, `YOWO_IOU`, `YOWO_NUM_CLASSES`
- Config load order: dataclass defaults -> YAML file -> `YOWO_*` env vars (last wins)
- `YOWO_CACHE_DIR` overrides default weight cache at `~/.cache/yowo/weights`
- No `.env` files - all env vars are runtime-only

**Build:**
- `pyproject.toml` - Single source of truth for project metadata, dependencies, tool config
- `[tool.ruff]` - Linter/formatter: `target-version = "py311"`, `line-length = 100`
- `[tool.pyright]` - Type checker: `typeCheckingMode = "strict"`, `pythonVersion = "3.11"`
- `[tool.pytest.ini_options]` - Test config: `testpaths = ["tests"]`, `asyncio_mode = "auto"`
- `[tool.semantic_release]` - Release automation: conventional commits, GitHub releases

**Pre-commit hooks** (`.pre-commit-config.yaml`):
1. `ruff --fix` - Auto-fix lint issues
2. `ruff-format` - Code formatting
3. `pyright` - Type checking
4. `pytest tests/unit -x -q` - Unit tests

## Platform Requirements

**Development:**
- Python 3.11+
- `uv` package manager
- No GPU required (CPU backend available)

**Production:**
- Python 3.11+ runtime
- At least one inference backend installed (PyTorch is the universal fallback)
- Hardware auto-detection: NVIDIA GPU, Apple Silicon, Jetson, x86/ARM CPU
- Supported deployment targets: cloud GPU (CUDA), Apple Silicon (CoreML/MPS), edge devices (Jetson/TensorRT), CPU-only servers (ONNX/OpenVINO)

**Quality gates command:**
```bash
uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q
```

---

*Stack analysis: 2026-03-07*
