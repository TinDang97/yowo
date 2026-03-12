"""Hardware detection and profile caching.

Detects hardware on first call, then caches immutably for the session.

Public API
----------
- ``HardwareProfile``: frozen snapshot of the system hardware state.
- ``get_hardware_profile()``: returns the cached profile; runs detection once.
- ``clear_cache()``: invalidates the cache (for test isolation only).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from yowo.hardware._capabilities import InstalledLibraries, detect_libraries
from yowo.hardware._detect import detect_cpu_device, detect_cpu_features, detect_gpus
from yowo.hardware._device import Device

__all__ = [
    "HardwareProfile",
    "clear_cache",
    "get_hardware_profile",
]


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of available hardware for this process.

    Detection runs once at first access and is cached for the remainder
    of the process lifetime. All fields are read-only.

    Attributes:
        gpus: Tuple of detected CUDA GPU devices (empty when no GPU found).
        cpu: Host CPU device descriptor.
        cpu_features: Canonical CPU feature flags (e.g. ``{"AVX2", "NEON"}``).
        libraries: Snapshot of installed inference SDK versions.
    """

    gpus: tuple[Device, ...]
    cpu: Device
    cpu_features: frozenset[str]
    libraries: InstalledLibraries

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_nvidia_gpu(self) -> bool:
        """True when at least one CUDA GPU was detected."""
        return len(self.gpus) > 0

    @property
    def is_jetson(self) -> bool:
        """True when the host CPU device reports Jetson platform."""
        return self.cpu.is_jetson

    @property
    def primary_gpu(self) -> Device | None:
        """First GPU in the ``gpus`` tuple, or ``None`` when no GPU is present."""
        return self.gpus[0] if self.gpus else None


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cached: HardwareProfile | None = None


def get_hardware_profile(*, force_refresh: bool = False) -> HardwareProfile:
    """Thread-safe cached hardware detection.

    Runs detection exactly once per process (or once per ``force_refresh``
    call). Uses double-checked locking — after the first successful
    detection the fast path requires no lock.

    Args:
        force_refresh: When True, discard the cached profile and re-detect.

    Returns:
        The (possibly cached) ``HardwareProfile`` for this system.
    """
    global _cached

    if _cached is not None and not force_refresh:
        return _cached

    with _lock:
        # Second check inside the lock for concurrent callers.
        if _cached is not None and not force_refresh:
            return _cached

        _cached = HardwareProfile(
            gpus=tuple(detect_gpus()),
            cpu=detect_cpu_device(),
            cpu_features=detect_cpu_features(),
            libraries=detect_libraries(),
        )

    return _cached


def clear_cache() -> None:
    """Invalidate the cached hardware profile.

    Intended for test isolation only. Not safe to call from production
    code while other threads may be reading the cached profile.
    """
    global _cached
    with _lock:
        _cached = None
