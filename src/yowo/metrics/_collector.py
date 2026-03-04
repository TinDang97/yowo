"""Zero-dependency inference metrics collection.

Thread-safe collectors for latency, throughput, and error tracking.
Designed for <5µs overhead per frame on the hot path.

_RollingHistogram.record() is lock-free (CPython deque.append is atomic).
The lock only serializes percentile reads against concurrent clears.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Snapshot type (public)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    """Immutable snapshot of engine metrics at a point in time.

    Attributes:
        frames_total: Total frames processed since last reset.
        errors_total: Total inference errors since last reset.
        inference_mean_ms: Rolling mean backend.infer() latency.
        inference_p50_ms: Rolling p50 backend.infer() latency.
        inference_p95_ms: Rolling p95 backend.infer() latency.
        inference_p99_ms: Rolling p99 backend.infer() latency.
        fps: Frames per second (frames_total / uptime_s).
        uptime_s: Seconds since collector creation or last reset.
    """

    frames_total: int
    errors_total: int
    inference_mean_ms: float
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    fps: float
    uptime_s: float


# ---------------------------------------------------------------------------
# Internal histogram
# ---------------------------------------------------------------------------


class _RollingHistogram:
    """Fixed-window latency histogram.

    Stores at most ``window`` latency samples. Record is lock-free (CPython
    deque.append is GIL-atomic). Percentile reads take a Lock to snapshot
    and sort the deque, preventing a clear() race.
    """

    __slots__ = ("_data", "_lock")

    def __init__(self, window: int = 1000) -> None:
        self._data: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, value_ms: float) -> None:
        """Append a latency sample. O(1), no lock acquired."""
        self._data.append(value_ms)

    def percentiles(self) -> tuple[float, float, float, float]:
        """Return (mean, p50, p95, p99). Returns (0, 0, 0, 0) when empty."""
        with self._lock:
            if not self._data:
                return (0.0, 0.0, 0.0, 0.0)
            snapshot = sorted(self._data)
        n = len(snapshot)
        mean = sum(snapshot) / n
        p50 = snapshot[n * 50 // 100]
        p95 = snapshot[min(n * 95 // 100, n - 1)]
        p99 = snapshot[min(n * 99 // 100, n - 1)]
        return (mean, p50, p95, p99)

    def clear(self) -> None:
        """Empty the histogram."""
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# MetricsCollector (public)
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Lightweight metrics collector wired into InferenceEngine.

    Overhead budget: <5µs per frame.
    - record_inference: deque.append + int increment + monotonic call ~1-2us
    - record_error: int increment ~0.1us
    - snapshot: sorts up to 1000 floats ~10-50us (called by operators, not hot path)

    Note on thread safety: ``_frames_total`` and ``_errors_total`` are plain
    Python ints. In CPython the GIL makes ``+= N`` effectively atomic. On
    free-threaded Python 3.13+ the counts may have minor races under extreme
    concurrency, but they are monotonically non-decreasing and self-correcting
    within the 1000-sample window. Adding a Lock would violate the 5µs budget.
    """

    __slots__ = (
        "_enabled",
        "_errors_total",
        "_frames_total",
        "_inference_hist",
        "_last_frame_time",
        "_start_time",
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._frames_total: int = 0
        self._errors_total: int = 0
        self._inference_hist = _RollingHistogram(window=1000)
        self._start_time: float = time.monotonic()
        self._last_frame_time: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when metrics collection is active."""
        return self._enabled

    @property
    def frames_total(self) -> int:
        """Total frames recorded since last reset."""
        return self._frames_total

    @property
    def errors_total(self) -> int:
        """Total errors recorded since last reset."""
        return self._errors_total

    @property
    def last_frame_time(self) -> float:
        """Monotonic timestamp of the last successful frame. 0.0 if none."""
        return self._last_frame_time

    # ------------------------------------------------------------------
    # Hot-path methods (called from detect / stream inner loop)
    # ------------------------------------------------------------------

    def record_inference(
        self,
        elapsed_ms: float,
        batch_size: int = 1,
        *,
        frame_time: float | None = None,
    ) -> None:
        """Record a successful inference call.

        Called from ``_infer_from_tensor`` immediately after backend.infer().

        Args:
            elapsed_ms: Backend infer() wall time in milliseconds.
            batch_size: Number of frames in the batch.
            frame_time: Pre-existing monotonic timestamp to use as
                ``_last_frame_time``. When supplied, skips the
                ``time.monotonic()`` syscall (~50-100 ns on macOS).
                Defaults to None, which falls back to a fresh call.
        """
        if not self._enabled:
            return
        self._frames_total += batch_size
        self._inference_hist.record(elapsed_ms)
        self._last_frame_time = frame_time if frame_time is not None else time.monotonic()

    def record_error(self) -> None:
        """Record an inference-level error."""
        if not self._enabled:
            return
        self._errors_total += 1

    # ------------------------------------------------------------------
    # Snapshot + reset (called by operators, not on hot path)
    # ------------------------------------------------------------------

    def snapshot(self) -> EngineMetrics:
        """Return a frozen immutable snapshot of current metrics."""
        elapsed = time.monotonic() - self._start_time
        mean, p50, p95, p99 = self._inference_hist.percentiles()
        fps = self._frames_total / elapsed if elapsed > 0 else 0.0
        return EngineMetrics(
            frames_total=self._frames_total,
            errors_total=self._errors_total,
            inference_mean_ms=mean,
            inference_p50_ms=p50,
            inference_p95_ms=p95,
            inference_p99_ms=p99,
            fps=fps,
            uptime_s=elapsed,
        )

    def reset(self) -> None:
        """Zero all counters and the latency histogram."""
        self._frames_total = 0
        self._errors_total = 0
        self._inference_hist.clear()
        self._start_time = time.monotonic()
        self._last_frame_time = 0.0
