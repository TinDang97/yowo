"""TuneProfile: device fingerprint computation and YAML profile persistence.

Public API
----------
- ``TuneProfile``: frozen dataclass capturing calibration results for a model.
- ``compute_fingerprint``: derive an 8-char device identifier from HardwareProfile.
- ``save_profile``: write profile to YAML atomically.
- ``load_profile``: read profile; returns None on missing, corrupt, or stale data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from yowo.hardware import HardwareProfile

__all__ = [
    "TuneProfile",
    "compute_fingerprint",
    "load_profile",
    "save_profile",
]

_log = logging.getLogger(__name__)

_PROFILE_BASE: Path = Path.home() / ".cache" / "yowo" / "profiles"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneProfile:
    """Calibration result for a specific model on a specific device.

    Attributes:
        model: Model identifier (e.g. ``"yolo11n"``).
        backend: Backend used during sweep (e.g. ``"onnx"``, ``"pytorch"``).
        batch_size: Optimal batch size found during sweep.
        precision: Precision string (``"fp32"``, ``"fp16"``, ``"int8"``).
        fps_achieved: Peak FPS measured at the optimal settings.
        tuned_at: ISO-8601 timestamp of when the sweep was run.
        fingerprint: 8-char device fingerprint at tune time.
    """

    model: str
    backend: str
    batch_size: int
    precision: str
    fps_achieved: float
    tuned_at: str
    fingerprint: str


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def compute_fingerprint(hw: HardwareProfile) -> str:  # type: ignore[name-defined]
    """Derive an 8-char device fingerprint from *hw*.

    GPU path:
        ``SHA-256("{name}|{vram_bytes}|{cuda_version}")[:8]``

    CPU-only path:
        ``SHA-256("cpu|{cpu_count}|{platform_string}")[:8]``

    Args:
        hw: Hardware profile returned by ``get_hardware_profile()``.

    Returns:
        Lowercase 8-character hex string.
    """
    gpu = hw.primary_gpu
    if gpu is not None:
        vram_bytes = gpu.memory_total_mb * 1024 * 1024
        cuda_ver = hw.libraries.cuda_version or "none"
        raw = f"{gpu.name}|{vram_bytes}|{cuda_ver}"
    else:
        raw = f"cpu|{os.cpu_count()}|{platform.platform()}"

    return hashlib.sha256(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _default_profile_path(model: str, fingerprint: str) -> Path:
    return _PROFILE_BASE / fingerprint / f"{model}.yaml"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_profile(profile: TuneProfile, path: Path | None = None) -> None:
    """Write *profile* to YAML atomically.

    Writes to a ``.tmp`` file first, then uses ``os.replace`` for an atomic
    rename so concurrent readers never see a partial file.

    Args:
        profile: The calibration result to persist.
        path: Override destination path.  When ``None``, defaults to
            ``~/.cache/yowo/profiles/{fingerprint}/{model}.yaml``.
    """
    dest = path or _default_profile_path(profile.model, profile.fingerprint)
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(dataclasses.asdict(profile)), encoding="utf-8")
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_profile(
    model: str,
    hw: HardwareProfile,  # type: ignore[name-defined]
    path: Path | None = None,
) -> TuneProfile | None:
    """Load a previously saved profile for *model* on the current device.

    Returns ``None`` in any of the following cases:
    - The profile file does not exist.
    - The file is corrupt (invalid YAML or missing required fields).
    - The stored fingerprint does not match the current device.

    A ``WARNING`` is logged when a fingerprint mismatch is detected so the
    user knows they need to re-tune.

    Args:
        model: Model identifier (used for default path resolution).
        hw: Current hardware profile.
        path: Override file path.  When ``None``, uses the default location.

    Returns:
        The loaded :class:`TuneProfile` or ``None``.
    """
    current_fp = compute_fingerprint(hw)
    dest = path or _default_profile_path(model, current_fp)

    if not dest.exists():
        return None

    try:
        raw = yaml.safe_load(dest.read_text(encoding="utf-8"))
        profile = TuneProfile(
            model=raw["model"],
            backend=raw["backend"],
            batch_size=raw["batch_size"],
            precision=raw["precision"],
            fps_achieved=raw["fps_achieved"],
            tuned_at=raw["tuned_at"],
            fingerprint=raw["fingerprint"],
        )
    except (KeyError, ValueError, yaml.YAMLError, TypeError):
        return None

    if profile.fingerprint != current_fp:
        _log.warning(
            "Hardware changed since last tune. Run `yowo tune --model %s` to update profile.",
            model,
        )
        return None

    return profile
