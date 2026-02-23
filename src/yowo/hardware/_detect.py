"""Hardware probing functions.

All functions return plain data. Side effects limited to reading
/proc, sysctl, pynvml, and nvidia-smi subprocess.

All functions handle errors gracefully - missing hardware returns
None or empty results, never raises.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from yowo.hardware._device import Device
from yowo.types import CPUArch, DeviceType, GPUArch

__all__ = [
    "detect_cpu_arch",
    "detect_cpu_device",
    "detect_cpu_features",
    "detect_gpu_arch",
    "detect_gpus",
    "detect_is_jetson",
    "detect_system_memory_mb",
]

_log = logging.getLogger(__name__)

# Jetson detection paths
_JETSON_DEVICE_TREE = Path("/proc/device-tree/model")
_JETSON_TEGRA_RELEASE = Path("/etc/nv_tegra_release")


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------


def detect_cpu_arch() -> CPUArch:
    """Detect the host CPU architecture.

    Maps ``platform.machine()`` output to ``CPUArch``.
    Normalises macOS "arm64" to AARCH64. Defaults to X86_64 for unknowns.
    """
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return CPUArch.AARCH64
    return CPUArch.X86_64


def detect_system_memory_mb() -> tuple[int, int]:
    """Return (total_mb, available_mb) of system RAM.

    Linux: reads /proc/meminfo.
    macOS: uses ``sysctl -n hw.memsize`` for total; ``vm_stat`` for available.
    Returns (0, 0) on any error — never raises.
    """
    system = platform.system()
    try:
        if system == "Linux":
            return _read_proc_meminfo()
        if system == "Darwin":
            return _read_macos_memory()
    except Exception:
        _log.debug("detect_system_memory_mb: probe failed", exc_info=True)
    return (0, 0)


def _read_proc_meminfo() -> tuple[int, int]:
    """Parse /proc/meminfo for MemTotal and MemAvailable (kB -> MB)."""
    total_kb = 0
    available_kb = 0
    with Path("/proc/meminfo").open() as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
            if total_kb and available_kb:
                break
    return (total_kb // 1024, available_kb // 1024)


def _read_macos_memory() -> tuple[int, int]:
    """Use sysctl + vm_stat to read macOS system memory."""
    # Total RAM via sysctl
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    total_bytes = int(result.stdout.strip()) if result.returncode == 0 else 0
    total_mb = total_bytes // (1024 * 1024)

    # Available pages via vm_stat
    available_mb = 0
    vm_result = subprocess.run(
        ["vm_stat"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if vm_result.returncode == 0:
        page_size = 4096  # Default macOS page size
        free_pages = 0
        for line in vm_result.stdout.splitlines():
            if line.startswith("Mach Virtual Memory Statistics"):
                # Extract page size if present: "page size of 4096 bytes"
                if "page size of" in line:
                    parts = line.split("page size of")
                    page_size = int(parts[1].split()[0])
            elif "Pages free:" in line or "Pages inactive:" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        available_mb = (free_pages * page_size) // (1024 * 1024)

    return (total_mb, available_mb)


# ---------------------------------------------------------------------------
# Jetson detection
# ---------------------------------------------------------------------------


def detect_is_jetson() -> bool:
    """Detect whether the system is an NVIDIA Jetson device.

    Checks /proc/device-tree/model for "NVIDIA Jetson" and
    /etc/nv_tegra_release for Tegra platform markers.
    Returns False on any error or absent files.
    """
    try:
        if _JETSON_TEGRA_RELEASE.exists():
            return True
        if _JETSON_DEVICE_TREE.exists():
            model = _JETSON_DEVICE_TREE.read_text(errors="replace")
            if "NVIDIA Jetson" in model:
                return True
    except Exception:
        _log.debug("detect_is_jetson: probe failed", exc_info=True)
    return False


# ---------------------------------------------------------------------------
# GPU architecture mapping
# ---------------------------------------------------------------------------

_CC_TO_ARCH: dict[tuple[int, int], GPUArch] = {
    (7, 5): GPUArch.TURING,
    (8, 0): GPUArch.AMPERE,
    (8, 6): GPUArch.AMPERE_GA10X,
    (8, 7): GPUArch.ORIN,
    (8, 9): GPUArch.ADA,
    (9, 0): GPUArch.HOPPER,
}


def detect_gpu_arch(major: int, minor: int) -> GPUArch:
    """Map CUDA compute capability (major, minor) to a ``GPUArch`` value.

    Returns ``GPUArch.UNKNOWN`` for unrecognised capabilities.
    """
    return _CC_TO_ARCH.get((major, minor), GPUArch.UNKNOWN)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_gpus() -> list[Device]:
    """Detect available NVIDIA CUDA GPUs.

    Primary path: pynvml (if installed).
    Fallback: nvidia-smi subprocess with --query-gpu.
    Compute capability resolved via torch.cuda when available.
    Returns an empty list on any error — never raises.
    """
    try:
        return _detect_gpus_nvml()
    except Exception:
        _log.debug("detect_gpus: pynvml probe failed, trying nvidia-smi", exc_info=True)

    try:
        return _detect_gpus_smi()
    except Exception:
        _log.debug("detect_gpus: nvidia-smi probe also failed", exc_info=True)

    return []


def _get_compute_capability(index: int) -> tuple[int, int] | None:
    """Try to retrieve compute capability via torch.cuda."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(index)
            return (int(major), int(minor))
    except Exception:
        pass
    return None


def _detect_gpus_nvml() -> list[Device]:
    """Probe GPUs via pynvml library."""
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        cpu_arch = detect_cpu_arch()
        is_jetson = detect_is_jetson()
        devices: list[Device] = []

        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = int(mem_info.total) // (1024 * 1024)
            free_mb = int(mem_info.free) // (1024 * 1024)

            cc = _get_compute_capability(idx)
            arch = detect_gpu_arch(*cc) if cc is not None else GPUArch.UNKNOWN

            devices.append(
                Device(
                    type=DeviceType.CUDA,
                    index=idx,
                    name=name,
                    arch=arch,
                    cpu_arch=cpu_arch,
                    memory_total_mb=total_mb,
                    memory_available_mb=free_mb,
                    is_jetson=is_jetson,
                )
            )
        return devices
    finally:
        pynvml.nvmlShutdown()


def _detect_gpus_smi() -> list[Device]:
    """Probe GPUs via nvidia-smi subprocess (fallback)."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        return []

    cpu_arch = detect_cpu_arch()
    is_jetson = detect_is_jetson()
    devices: list[Device] = []

    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        name = parts[1]
        total_mb = int(parts[2])
        free_mb = int(parts[3])

        cc = _get_compute_capability(idx)
        arch = detect_gpu_arch(*cc) if cc is not None else GPUArch.UNKNOWN

        devices.append(
            Device(
                type=DeviceType.CUDA,
                index=idx,
                name=name,
                arch=arch,
                cpu_arch=cpu_arch,
                memory_total_mb=total_mb,
                memory_available_mb=free_mb,
                is_jetson=is_jetson,
            )
        )
    return devices


# ---------------------------------------------------------------------------
# CPU device
# ---------------------------------------------------------------------------


def detect_cpu_device() -> Device:
    """Build a ``Device`` representing the host CPU.

    Combines cpu_arch, system memory, and Jetson detection.
    The ``arch`` field is always ``None`` for CPU devices.
    Never raises.
    """
    cpu_arch = detect_cpu_arch()
    total_mb, available_mb = detect_system_memory_mb()
    is_jetson = detect_is_jetson()

    proc_name = platform.processor()
    name = proc_name.strip() if proc_name else f"{cpu_arch.value} CPU"
    if not name:
        name = f"{cpu_arch.value} CPU"

    return Device(
        type=DeviceType.CPU,
        index=0,
        name=name,
        arch=None,
        cpu_arch=cpu_arch,
        memory_total_mb=total_mb,
        memory_available_mb=available_mb,
        is_jetson=is_jetson,
    )


# ---------------------------------------------------------------------------
# CPU feature flags
# ---------------------------------------------------------------------------


def detect_cpu_features() -> frozenset[str]:
    """Detect CPU instruction set extensions.

    Linux: parses /proc/cpuinfo flags.
    ARM: checks for NEON/asimd.
    Returns a frozenset of canonical feature names: {"AVX2", "AVX512", "VNNI", "NEON"}.
    Returns an empty frozenset on any error — never raises.
    """
    try:
        cpu_arch = detect_cpu_arch()
        if cpu_arch == CPUArch.AARCH64:
            return _detect_arm_features()
        return _detect_x86_features()
    except Exception:
        _log.debug("detect_cpu_features: probe failed", exc_info=True)
    return frozenset()


def _detect_x86_features() -> frozenset[str]:
    """Parse /proc/cpuinfo for x86 SIMD feature flags."""
    features: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return frozenset()

    with cpuinfo.open() as fh:
        for line in fh:
            if not line.startswith("flags"):
                continue
            flags_str = line.split(":", 1)[1].lower()
            if "avx2" in flags_str:
                features.add("AVX2")
            if "avx512f" in flags_str:
                features.add("AVX512")
            if "avx512vnni" in flags_str:
                features.add("VNNI")
            # Only need first flags line
            break

    return frozenset(features)


def _detect_arm_features() -> frozenset[str]:
    """Parse /proc/cpuinfo for ARM SIMD feature flags."""
    features: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return frozenset()

    with cpuinfo.open() as fh:
        for line in fh:
            if not line.startswith("Features"):
                continue
            flags_str = line.split(":", 1)[1].lower()
            if "neon" in flags_str or "asimd" in flags_str:
                features.add("NEON")
            break

    return frozenset(features)
