# hardware — Hardware Detection

Detects available hardware (GPUs, CPU architecture, memory) and installed inference libraries once at startup, caching results for the session lifetime.

---

## Design Principles

- **Single Responsibility**: detection only — no inference logic, no backend selection.
- **Black box**: callers receive a `HardwareProfile`; raw probe internals are private.
- **Fail-safe**: every probe function catches its own exceptions and returns `None` or a safe default instead of propagating. A missing GPU driver never crashes startup.
- **Cached singleton**: `get_hardware_profile()` runs once per process; subsequent calls return the cached instance with no I/O.

---

## Module Structure

```
hardware/
├── __init__.py        — public surface: HardwareProfile, get_hardware_profile(), clear_cache()
├── _device.py         — Device dataclass and related enums
├── _detect.py         — raw probe functions (nvml, subprocess, /proc, cpuid)
└── _capabilities.py   — SDK presence detection, InstalledLibraries
```

### `_device.py`

Core data types for describing a compute device.

```python
from enum import StrEnum
from dataclasses import dataclass

class DeviceType(StrEnum):
    CUDA = "cuda"
    CPU  = "cpu"

class CPUArch(StrEnum):
    X86_64  = "x86_64"
    AARCH64 = "aarch64"

class GPUArch(StrEnum):
    # SM version -> microarchitecture
    SM_75 = "sm_75"   # Turing  (RTX 20xx, T4)
    SM_80 = "sm_80"   # Ampere  (A100, RTX 30xx)
    SM_86 = "sm_86"   # Ampere  (RTX 30xx consumer)
    SM_87 = "sm_87"   # Ampere  (Jetson Orin)
    SM_89 = "sm_89"   # Ada     (RTX 40xx)
    SM_90 = "sm_90"   # Hopper  (H100)

@dataclass(frozen=True, slots=True)
class Device:
    type:                DeviceType
    index:               int                  # CUDA device index; 0 for CPU
    name:                str                  # "NVIDIA GeForce RTX 4090" | "x86_64"
    arch:                GPUArch | CPUArch
    memory_total_mb:     int
    memory_available_mb: int
    is_jetson:           bool = False

    def supports_fp16(self) -> bool:
        """True for all CUDA sm >= 70 and AARCH64 CPUs."""
        ...

    def supports_int8(self) -> bool:
        """True for CUDA sm >= 75 (Turing+)."""
        ...
```

### `_detect.py`

Raw probe functions. Each is independent and never raises — failures return `None` or safe defaults.

| Function | Method | Fallback |
|----------|--------|----------|
| `detect_gpus() -> list[Device]` | `pynvml` | `nvidia-smi` subprocess |
| `detect_cpu_device() -> Device` | `platform.machine()` | always succeeds |
| `detect_cpu_arch() -> CPUArch` | `platform.machine()` | `X86_64` |
| `detect_system_memory_mb() -> int` | `psutil.virtual_memory()` | `0` |
| `detect_is_jetson() -> bool` | reads `/proc/device-tree/model` | `False` if file absent |
| `detect_cpu_features() -> set[str]` | `cpuid` / `/proc/cpuinfo` | empty set |

`detect_cpu_features()` returns a subset of `{"AVX2", "VNNI", "NEON"}`. These strings are the canonical feature names used by backend selection.

### `_capabilities.py`

Probes whether optional inference SDKs are importable. Uses `importlib.util.find_spec` — never imports the package itself at module level.

```python
@dataclass(frozen=True, slots=True)
class InstalledLibraries:
    tensorrt_version:    str | None   # e.g. "8.6.1" or None
    onnxruntime_version: str | None
    openvino_version:    str | None
    torch_version:       str | None
```

| Probe function | Returns |
|----------------|---------|
| `probe_tensorrt() -> str \| None` | version string or `None` |
| `probe_onnxruntime() -> str \| None` | version string or `None` |
| `probe_openvino() -> str \| None` | version string or `None` |
| `probe_torch() -> str \| None` | version string or `None` |
| `detect_libraries() -> InstalledLibraries` | assembles all probes |

All probe functions return `None` on `ImportError` — never raise.

### `__init__.py` — Public Interface

```python
@dataclass(frozen=True, slots=True)
class HardwareProfile:
    gpus:          list[Device]        # empty list if no CUDA GPUs found
    cpu:           Device
    cpu_features:  frozenset[str]      # {"AVX2", "NEON", ...}
    libraries:     InstalledLibraries

    @property
    def primary_gpu(self) -> Device | None:
        """First GPU in the list, or None."""
        ...

    @property
    def has_cuda(self) -> bool:
        """True if at least one CUDA GPU detected."""
        ...
```

```python
def get_hardware_profile() -> HardwareProfile:
    """
    Return cached HardwareProfile. Runs detection once per process.

    Thread-safe via threading.Lock with double-checked locking.
    Detection runs synchronously (no async) — call once at startup.
    """

def clear_cache() -> None:
    """
    Invalidate the cached profile. Intended for test isolation only.
    Not safe to call from production code while other threads may be reading.
    """
```

---

## Caching Implementation

```
First call:
    acquire Lock
    check _cache is None
    run all detectors (sequential, safe)
    build HardwareProfile
    assign to _cache
    release Lock
    return _cache

Subsequent calls:
    check _cache is not None  (no lock needed — immutable after write)
    return _cache
```

The profile is a frozen dataclass, so it is safe to share across threads without additional locking.

---

## Dependencies

- **Stdlib only at module level**: `platform`, `subprocess`, `threading`, `pathlib`, `importlib.util`
- **Optional runtime deps** (imported lazily inside probe functions): `pynvml`, `psutil`
- **No yowo imports** except `types.py` and `errors.py`
- `torch`, `tensorrt`, `onnxruntime`, `openvino` are **never imported** by this module — only their presence is checked via `importlib.util.find_spec`.

---

## Usage Example

```python
from yowo.hardware import get_hardware_profile

hw = get_hardware_profile()

print(hw.primary_gpu)
# Device(type='cuda', index=0, name='NVIDIA GeForce RTX 4090',
#        arch=<GPUArch.SM_89: 'sm_89'>, memory_total_mb=24564,
#        memory_available_mb=22000, is_jetson=False)

print(hw.cpu_features)
# frozenset({'AVX2', 'VNNI'})

print(hw.libraries.tensorrt_version)
# '8.6.1'

if hw.has_cuda and hw.primary_gpu.supports_int8():
    print("INT8 inference available")
```

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `types.py` | `BackendType`, `Precision` enum values |
| Upstream | `errors.py` | `HardwareError` |
| Downstream | `backends/_selector.py` | consumes `HardwareProfile` to pick backend and precision |
| Downstream | `export/_exporter.py` | consumes `HardwareProfile.primary_gpu` for TensorRT export |

---

## Extension Guide

**To add Intel Arc GPU detection:**

1. Add `INTEL_GPU = "intel_gpu"` to `DeviceType` in `_device.py`.
2. Add `ARC_ACM = "arc_acm"` (or appropriate arch string) to a new `IntelGPUArch` enum in `_device.py`.
3. Implement `detect_intel_gpus() -> list[Device]` in `_detect.py` using `pyze` or `intel_extension_for_pytorch`.
4. Call it inside `detect_libraries()` / `detect_gpus()` equivalents and include result in `HardwareProfile.gpus`.
5. No other files need to change — `backends/_selector.py` reads `DeviceType` to dispatch.
