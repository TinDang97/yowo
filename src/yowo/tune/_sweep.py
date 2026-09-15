"""Calibration sweep: measure FPS for backend x batch_size combinations.

Public API
----------
- ``SweepResult``: dataclass capturing one measurement result.
- ``run_sweep()``: iterate available backends x batch sizes,
  measure FPS on synthetic frames, return results sorted by FPS descending.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from yowo.backends import check_backend_available
from yowo.errors import BackendError
from yowo.types import BackendType, Frame

if TYPE_CHECKING:
    from yowo.hardware import HardwareProfile
    from yowo.types import ModelSpec

__all__ = ["SweepResult", "count_sweep_dimensions", "run_sweep"]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SWEEP_BACKENDS: list[BackendType] = [
    BackendType.TENSORRT,
    BackendType.ONNX,
    BackendType.PYTORCH,
    BackendType.OPENVINO,
    BackendType.COREML,
]

_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# SweepResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """One FPS measurement for a specific backend/batch_size combo.

    Attributes:
        backend: Backend name string (e.g. ``"pytorch"``).
        batch_size: Batch size used during measurement.
        precision: The precision the backend ACTUALLY executed, read back
            from the engine after load — or ``"unknown"`` when the backend
            does not determine it (its artifact does) and for a skipped
            row, where nothing ran. Never a requested value: the sweep no
            longer asks for one.
        fps: Measured frames per second. 0.0 for skipped configs.
        skipped: True if this config was skipped (e.g. OOM).
        skip_reason: Human-readable reason for skipping.
    """

    backend: str
    batch_size: int
    precision: str
    fps: float
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Backend/precision enumeration helpers
# ---------------------------------------------------------------------------


def _enumerate_backends(hw: HardwareProfile) -> list[BackendType]:
    """Return available backends in sweep priority order.

    Calls ``check_backend_available`` for each candidate; silently drops
    any that raise :class:`~yowo.errors.BackendError`.

    Args:
        hw: Current hardware profile.

    Returns:
        Ordered list of usable :class:`~yowo.types.BackendType` values.
    """
    available: list[BackendType] = []
    for backend in _SWEEP_BACKENDS:
        try:
            check_backend_available(backend, hw)
            available.append(backend)
        except BackendError:
            _log.debug("Backend %s not available, skipping in sweep.", backend.value)
    return available


# ---------------------------------------------------------------------------
# OOM detection helper
# ---------------------------------------------------------------------------


def _is_oom(exc: Exception) -> bool:
    """Return True if *exc* represents a CUDA out-of-memory error."""
    if isinstance(exc, MemoryError):
        return True
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Per-config measurement
# ---------------------------------------------------------------------------


def _measure_config(
    spec: ModelSpec,
    hw: HardwareProfile,
    backend: BackendType,
    batch_size: int,
    warmup_frames: int,
    measure_frames: int,
) -> tuple[float, str]:
    """Measure FPS for one (backend, batch_size) configuration.

    Creates an engine, runs ``warmup_frames`` detections (untimed), then times
    ``measure_frames`` detections. The engine is always closed in a finally block.

    No precision is REQUESTED. The sweep used to pass one, which made every row
    an explicit user request as far as the engine is concerned — and since
    nothing consumed the value, every precision row measured an identical
    configuration, so the stored winner was FPS noise picked between
    indistinguishable runs. The precision is now read back from the engine
    AFTER load: what executed, not what was asked for.

    Args:
        spec: Model specification.
        hw: Hardware profile (unused by engine directly; passed for API context).
        backend: Backend to use for this measurement.
        batch_size: Batch size to configure.
        warmup_frames: Number of warmup iterations (not timed).
        measure_frames: Number of measured iterations for FPS calculation.

    Returns:
        ``(fps, executing_precision)`` — FPS as ``measure_frames / elapsed``,
        and the precision the backend actually ran, or the engine's documented
        ``"unknown"`` sentinel when the backend does not determine it.
    """
    task = spec.task
    if task == "obb":
        from yowo.config import OBBConfig
        from yowo.obb_engine import OBBEngine  # lazy import — patchable in tests

        config = OBBConfig(
            model_family=spec.family,
            model_size=spec.size,
            num_classes=spec.num_classes,
            backend=backend,
            batch_size=batch_size,
        )
        engine = OBBEngine(config)
    elif task == "classify":
        from yowo.classify_engine import ClassificationEngine
        from yowo.config import ClassificationConfig

        config = ClassificationConfig(
            model_family=spec.family,
            model_size=spec.size,
            num_classes=spec.num_classes,
            backend=backend,
            batch_size=batch_size,
        )
        engine = ClassificationEngine(config)
    else:
        from yowo.config import InferenceConfig
        from yowo.engine import DetectionEngine

        config = InferenceConfig(
            model_family=spec.family,
            model_size=spec.size,
            num_classes=spec.num_classes,
            backend=backend,
            batch_size=batch_size,
        )
        engine = DetectionEngine(config)

    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    frames = [Frame(pixels=pixels)]

    try:
        engine.load()

        # Callable alias avoids repeating the task branch inside tight loops.
        # type: ignore needed because pyright can't narrow union through assignment.
        if task == "obb":
            infer = engine.detect_obb  # type: ignore[union-attr]
        elif task == "classify":
            infer = engine.classify  # type: ignore[union-attr]
        else:
            infer = engine.detect  # type: ignore[union-attr]

        # Warmup (not timed)
        for _ in range(warmup_frames):
            infer(frames)

        # Timed measurement
        t0 = time.monotonic()
        for _ in range(measure_frames):
            infer(frames)
        elapsed = time.monotonic() - t0

        return measure_frames / elapsed, engine.health_report().precision_current
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Main sweep entry point
# ---------------------------------------------------------------------------


def run_sweep(
    spec: ModelSpec,
    hw: HardwareProfile,
    warmup_frames: int = 50,
    measure_frames: int = 200,
    dry_run: bool = False,
) -> list[SweepResult]:
    """Run calibration sweep across backend x precision x batch_size space.

    For each available backend, each precision that backend supports on this
    hardware, and each batch size in ``_BATCH_SIZES``, this function:

    1. Calls :func:`_measure_config` to get FPS.
    2. On OOM (:class:`torch.cuda.OutOfMemoryError` or ``MemoryError``):
       marks current and all larger batch sizes for this combo as skipped,
       calls ``torch.cuda.empty_cache()`` if CUDA is available.
    3. Returns only **non-skipped** results sorted by FPS descending
       (tie-broken by smaller batch_size).

    Args:
        spec: Model specification (family, size, task, num_classes).
        hw: Hardware profile for backend availability and precision selection.
        warmup_frames: Warmup iterations excluded from FPS timing.
        measure_frames: Measured iterations used to compute FPS.
        dry_run: When True, skip all measurements and return empty list.

    Returns:
        Non-skipped :class:`SweepResult` objects sorted best-to-worst.
    """
    if dry_run:
        _log.info("run_sweep: dry_run=True, skipping all measurements.")
        return []

    # The ONE sentinel for "the precision that executed is not knowable",
    # defined and documented in engine.py. Reused rather than respelled: a
    # second spelling of the same meaning is how a sentinel stops meaning one
    # thing. Imported here, not at module scope, because this file keeps the
    # engine layer out of its import graph on purpose.
    from yowo.engine import PRECISION_UNKNOWN

    available_backends = _enumerate_backends(hw)
    results: list[SweepResult] = []

    for backend in available_backends:
        oom_hit = False

        for batch_size in _BATCH_SIZES:
            if oom_hit:
                _log.debug(
                    "OOM already hit for %s; skipping batch_size=%d.",
                    backend.value,
                    batch_size,
                )
                results.append(
                    SweepResult(
                        backend=backend.value,
                        batch_size=batch_size,
                        precision=PRECISION_UNKNOWN,
                        fps=0.0,
                        skipped=True,
                        skip_reason="OOM",
                    )
                )
                continue

            try:
                fps, executed = _measure_config(
                    spec,
                    hw,
                    backend,
                    batch_size,
                    warmup_frames,
                    measure_frames,
                )
                _log.debug(
                    "Sweep %s/bs=%d -> %.1f FPS (ran %s)",
                    backend.value,
                    batch_size,
                    fps,
                    executed,
                )
                results.append(
                    SweepResult(
                        backend=backend.value,
                        batch_size=batch_size,
                        precision=executed,
                        fps=fps,
                    )
                )

            except Exception as exc:  # pylint: disable=broad-except
                if _is_oom(exc):
                    oom_hit = True
                    _log.warning(
                        "OOM for %s/bs=%d: %s. Skipping larger batch sizes.",
                        backend.value,
                        batch_size,
                        exc,
                    )
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results.append(
                        SweepResult(
                            backend=backend.value,
                            batch_size=batch_size,
                            precision=PRECISION_UNKNOWN,
                            fps=0.0,
                            skipped=True,
                            skip_reason="OOM",
                        )
                    )
                else:
                    _log.error(
                        "Error for %s/bs=%d: %s. Skipping.",
                        backend.value,
                        batch_size,
                        exc,
                    )
                    results.append(
                        SweepResult(
                            backend=backend.value,
                            batch_size=batch_size,
                            precision=PRECISION_UNKNOWN,
                            fps=0.0,
                            skipped=True,
                            skip_reason=str(exc),
                        )
                    )

    non_skipped = [r for r in results if not r.skipped]
    return sorted(non_skipped, key=lambda r: (-r.fps, r.batch_size))


def count_sweep_dimensions(hw: HardwareProfile) -> int:  # type: ignore[name-defined]
    """Return total number of (backend, batch_size) combinations for *hw*.

    Used by ``yowo tune --dry-run`` to report the sweep size without running it.

    Args:
        hw: Current hardware profile.

    Returns:
        Total number of configurations that :func:`run_sweep` would attempt.
    """
    return len(_enumerate_backends(hw)) * len(_BATCH_SIZES)
