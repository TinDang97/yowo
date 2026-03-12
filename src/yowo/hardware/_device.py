"""Device abstraction for hardware targets.

Wraps hardware detection results into immutable Device objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from yowo.types import CPUArch, DeviceType, GPUArch

__all__ = ["Device"]

# GPUArch SM values that support FP16 efficiently (Turing and above = sm_75+)
_FP16_ARCHS: frozenset[GPUArch] = frozenset(
    {
        GPUArch.TURING,
        GPUArch.AMPERE,
        GPUArch.AMPERE_GA10X,
        GPUArch.ORIN,
        GPUArch.ADA,
        GPUArch.HOPPER,
    }
)

# INT8 with tensor cores: Turing and above (sm_75+)
_INT8_ARCHS: frozenset[GPUArch] = _FP16_ARCHS


@dataclass(frozen=True)
class Device:
    """Immutable description of a single compute device.

    For CPU devices: ``arch`` is ``None``, ``cpu_arch`` is always populated.
    For GPU devices: ``arch`` carries the GPU compute capability, ``cpu_arch``
    reflects the host CPU (used for fallback decisions).

    Attributes:
        type: CUDA or CPU.
        index: GPU device index (0-based); 0 for the CPU device.
        name: Human-readable label, e.g. "NVIDIA L4" or "Intel Core i7".
        arch: GPU compute capability enum, or ``None`` for CPU devices.
        cpu_arch: Host CPU architecture; always populated.
        memory_total_mb: VRAM (GPU) or system RAM (CPU) in mebibytes.
        memory_available_mb: Free VRAM or available system RAM in mebibytes.
        is_jetson: True when running on an NVIDIA Jetson SoC.
    """

    type: DeviceType
    index: int
    name: str
    arch: GPUArch | None
    cpu_arch: CPUArch
    memory_total_mb: int
    memory_available_mb: int
    is_jetson: bool = False

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_gpu(self) -> bool:
        """True when this device is a CUDA GPU."""
        return self.type == DeviceType.CUDA

    def supports_fp16(self) -> bool:
        """FP16 is efficient on Turing+ (sm_75+) GPUs.

        Returns False for CPU devices and for GPUs with unknown architecture.
        """
        if not self.is_gpu or self.arch is None:
            return False
        return self.arch in _FP16_ARCHS

    def supports_int8(self) -> bool:
        """INT8 requires tensor cores — Turing+ (sm_75+) GPUs only.

        Returns False for CPU devices and for GPUs with unknown architecture.
        """
        if not self.is_gpu or self.arch is None:
            return False
        return self.arch in _INT8_ARCHS

    def __str__(self) -> str:
        """Human-readable summary.

        GPU example:  'NVIDIA L4 (cuda:0, sm_89, 24576MB)'
        CPU example:  'aarch64 CPU (16384MB RAM)'
        """
        if self.is_gpu and self.arch is not None:
            return f"{self.name} (cuda:{self.index}, {self.arch.value}, {self.memory_total_mb}MB)"
        return f"{self.name} ({self.memory_total_mb}MB RAM)"
