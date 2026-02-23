# backends — Inference Backend Implementations

Inference backend implementations and automatic selection. Each backend loads a model file and runs inference; callers interact exclusively through the `InferenceBackend` Protocol.

---

## Design Principles

- **Protocol-based interface**: structural typing, not ABC inheritance. Any object satisfying the Protocol is a valid backend — no base class required.
- **Lazy SDK imports**: `import tensorrt`, `import onnxruntime`, etc. happen inside the backend's `load()` method, not at module level. Importing `yowo.backends` with no CUDA hardware installed does not fail.
- **Single fallback chain**: `_selector.py` produces an ordered list of backends to try. The engine attempts each in sequence at load time, stopping at the first success.

---

## InferenceBackend Protocol

```python
from typing import Protocol, runtime_checkable
from numpy.typing import NDArray
import numpy as np

@runtime_checkable
class InferenceBackend(Protocol):
    backend_type: BackendType          # identifies the implementation
    is_loaded:    bool                 # True after successful load()
    input_shape:  tuple[int, int]      # (height, width) — model's expected spatial dims

    def load(self, model_path: Path, device: str = "auto") -> None:
        """
        Load model from model_path into device memory.

        device: "auto" | "cuda" | "cuda:0" | "cpu"
        Raises BackendError on failure.
        Idempotent — calling load() on an already-loaded backend reloads.
        """

    def infer(self, tensor: PreprocessedTensor) -> NDArray[np.float32]:
        """
        Run inference on a preprocessed tensor batch.

        tensor shape: (B, 3, H, W) float32 in [0, 1]
        Returns raw model output — shape varies by model family.
        Raises BackendError if is_loaded is False or inference fails.
        """

    def unload(self) -> None:
        """Release device memory and file handles. Safe to call multiple times."""

    def warmup(self, batch_size: int = 1) -> None:
        """
        Run one silent inference pass to initialize CUDA contexts and JIT caches.
        Call once after load() before production inference.
        No-op if backend does not benefit from warmup.
        """
```

---

## Module Structure

```
backends/
├── __init__.py      — InferenceBackend Protocol, create_backend() factory
├── _selector.py     — auto-selection logic, precision selection, fallback chain
├── _pytorch.py      — PyTorch fallback backend (ultralytics.YOLO)
├── _onnx.py         — ONNX Runtime backend (CUDA EP + CPU EP)
├── _tensorrt.py     — TensorRT backend (.engine file, async execution)
└── _openvino.py     — OpenVINO backend (Core.compile_model, InferRequest)
```

### `__init__.py`

```python
def create_backend(
    backend_type: BackendType,
    hw_profile: HardwareProfile,
) -> InferenceBackend:
    """
    Instantiate (but do not load) a backend of the given type.

    Imports the backend's SDK lazily inside this function.
    Raises DependencyError if the required SDK is not installed.
    """
```

### `_selector.py`

```python
@dataclass(frozen=True, slots=True)
class BackendSelection:
    backend_type: BackendType
    precision:    Precision
    device:       str           # resolved device string, e.g. "cuda:0"
    fallback_chain: tuple[BackendType, ...]  # ordered alternatives if primary fails

def select_backend(
    model_size:        str,
    backend_override:  BackendType | None,
    device_override:   str | None,
    precision_override: Precision | None,
    profile:           HardwareProfile,
) -> BackendSelection:
    """Apply priority chain and precision heuristics to produce a BackendSelection."""

def select_precision(
    requested:   Precision | None,
    gpu:         Device | None,
    model_size:  str,
) -> Precision:
    """
    Choose precision, downgrading if available VRAM is insufficient.

    INT8 requires VRAM ≥ 4 GB.
    FP16 requires VRAM ≥ 2 GB.
    Falls back to FP32 (CPU-friendly) otherwise.
    """

def get_fallback_chain(
    primary: BackendType,
    profile: HardwareProfile,
) -> tuple[BackendType, ...]:
    """Return ordered alternatives to try if primary backend fails to load."""
```

---

## Backend Selection Priority Chain

Evaluated in order; first condition that is fully satisfied wins.

| Priority | Condition | Selected Backend | Precision default |
|----------|-----------|-----------------|-------------------|
| 1 | CUDA GPU present + TensorRT installed | `TENSORRT` | FP16 |
| 2 | CUDA GPU present + ONNX Runtime installed | `ONNX` (CUDA EP) | FP16 |
| 3 | Intel CPU/iGPU + OpenVINO installed | `OPENVINO` | FP32 |
| 4 | ONNX Runtime installed (any hardware) | `ONNX` (CPU EP) | FP32 |
| 5 | PyTorch installed | `PYTORCH` | FP32 |
| 6 | Nothing available | — | raises `DependencyError` with install instructions |

The fallback chain at load time allows degradation without user intervention: if TensorRT `.engine` deserialization fails (e.g., wrong GPU arch), the engine retries with ONNX automatically.

---

## Precision Selection

| Precision | VRAM requirement | Accuracy impact | Notes |
|-----------|-----------------|----------------|-------|
| FP32 | No GPU needed | Baseline | Always available |
| FP16 | ≥ 2 GB GPU VRAM | Negligible | Default for CUDA backends |
| INT8 | ≥ 4 GB GPU VRAM | ~1-3% mAP | Requires calibration data at export time |

Precision is selected at `BackendSelection` time (before any model file is loaded). INT8 requires a pre-quantized `.engine` or `.onnx` — the backend does not quantize at runtime.

---

## Backend Implementations

### `_pytorch.py` — PyTorch Backend

- Loads `.pt` via `ultralytics.YOLO(model_path)`.
- Wraps `ultralytics.YOLO.predict()` output into `NDArray[float32]`.
- Device: passed directly to `ultralytics.YOLO(..., device=device)`.
- No warmup needed (ultralytics handles internally).

### `_onnx.py` — ONNX Runtime Backend

- Creates `onnxruntime.InferenceSession` with explicit `ExecutionProvider` list.
- CUDA EP: `["CUDAExecutionProvider", "CPUExecutionProvider"]`.
- CPU EP: `["CPUExecutionProvider"]`.
- IO binding: uses `InferenceSession.io_binding()` for zero-copy GPU tensor passing.
- Session options: `graph_optimization_level = ORT_ENABLE_ALL`, `intra_op_num_threads` from CPU count.

### `_tensorrt.py` — TensorRT Backend

- Deserializes `.engine` file via `tensorrt.Runtime.deserialize_cuda_engine()`.
- Allocates CUDA device buffers sized to `engine.get_binding_shape()`.
- Async execution via `tensorrt.IExecutionContext.execute_async_v2()` on a dedicated CUDA stream.
- Warmup: runs 3 silent inferences to initialize CUDA context and fill L2 cache.
- Engine files are GPU-arch-specific: an `sm_89` engine will not run on `sm_80`.

### `_openvino.py` — OpenVINO Backend

- `openvino.Core().compile_model(model_path, device_name="AUTO")`.
- Inference via `CompiledModel.create_infer_request()` and `InferRequest.infer()`.
- Supports NEON CPU features on AArch64 automatically via OpenVINO's runtime dispatch.

---

## Adding a New Backend

1. Implement the `InferenceBackend` Protocol in a new `_mybackend.py` file. No inheritance required.
2. Add `MYBACKEND = "mybackend"` to `BackendType` in `types.py`.
3. Add a case to `create_backend()` in `__init__.py` with lazy import.
4. Insert the appropriate priority condition in `_selector.py`'s priority chain.
5. Add to `get_fallback_chain()` if it should be a fallback target.

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `hardware/` | `HardwareProfile` for backend and precision selection |
| Upstream | `types.py` | `BackendType`, `Precision`, `PreprocessedTensor` |
| Upstream | `errors.py` | `BackendError`, `DependencyError` |
| Downstream | `engine.py` | calls `select_backend()`, `create_backend()`, `backend.load()`, `backend.infer()` |
