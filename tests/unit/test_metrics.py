"""Unit tests for src/yowo/metrics/_collector.py.

Tests _RollingHistogram, MetricsCollector, and EngineMetrics.
"""

from __future__ import annotations

import threading
import time

import pytest

from yowo.metrics import EngineMetrics, MetricsCollector
from yowo.metrics._collector import _RollingHistogram

# ---------------------------------------------------------------------------
# _RollingHistogram
# ---------------------------------------------------------------------------


class TestRollingHistogram:
    def test_empty_returns_zeros(self) -> None:
        h = _RollingHistogram()
        assert h.percentiles() == (0.0, 0.0, 0.0, 0.0)

    def test_single_sample(self) -> None:
        h = _RollingHistogram()
        h.record(10.0)
        mean, p50, p95, p99 = h.percentiles()
        assert mean == pytest.approx(10.0)
        assert p50 == pytest.approx(10.0)
        assert p95 == pytest.approx(10.0)
        assert p99 == pytest.approx(10.0)

    def test_multiple_samples_mean(self) -> None:
        h = _RollingHistogram()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            h.record(v)
        mean, _, _, _ = h.percentiles()
        assert mean == pytest.approx(3.0)

    def test_percentile_ordering(self) -> None:
        h = _RollingHistogram(window=100)
        for v in range(1, 101):  # 1..100
            h.record(float(v))
        mean, p50, p95, p99 = h.percentiles()
        assert mean == pytest.approx(50.5)
        assert p50 < p95 < p99

    def test_window_eviction(self) -> None:
        """With window=3, the 4th record evicts the oldest."""
        h = _RollingHistogram(window=3)
        h.record(100.0)
        h.record(100.0)
        h.record(100.0)
        h.record(1.0)  # evicts oldest 100.0, but deque keeps 100, 100, 1
        mean, _, _, _ = h.percentiles()
        # window contains [100, 100, 1] → mean ≈ 67
        assert mean == pytest.approx((100.0 + 100.0 + 1.0) / 3)

    def test_clear_resets_to_zero(self) -> None:
        h = _RollingHistogram()
        h.record(5.0)
        h.clear()
        assert h.percentiles() == (0.0, 0.0, 0.0, 0.0)

    def test_concurrent_record_and_read(self) -> None:
        """record() is lock-free; percentiles() holds lock — no deadlock or crash."""
        h = _RollingHistogram(window=1000)
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(500):
                try:
                    h.record(float(i))
                except Exception as exc:
                    errors.append(exc)

        def reader() -> None:
            for _ in range(50):
                try:
                    h.percentiles()
                except Exception as exc:
                    errors.append(exc)
                time.sleep(0.001)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == [], f"Unexpected errors: {errors}"

    def test_default_window_is_1000(self) -> None:
        h = _RollingHistogram()
        for i in range(1500):
            h.record(float(i))
        # Only 1000 samples kept; min should be 500 (not 0)
        mean, _, _, _ = h.percentiles()
        assert mean > 500.0


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_initial_state(self) -> None:
        c = MetricsCollector()
        assert c.enabled is True
        assert c.frames_total == 0
        assert c.errors_total == 0
        assert c.last_frame_time == 0.0

    def test_record_inference_increments_frames(self) -> None:
        c = MetricsCollector()
        c.record_inference(5.0)
        assert c.frames_total == 1

    def test_record_inference_batch_size(self) -> None:
        c = MetricsCollector()
        c.record_inference(10.0, batch_size=4)
        assert c.frames_total == 4

    def test_record_inference_updates_last_frame_time(self) -> None:
        c = MetricsCollector()
        before = time.monotonic()
        c.record_inference(1.0)
        after = time.monotonic()
        assert before <= c.last_frame_time <= after

    def test_record_error_increments_errors(self) -> None:
        c = MetricsCollector()
        c.record_error()
        c.record_error()
        assert c.errors_total == 2

    def test_snapshot_returns_frozen_dataclass(self) -> None:
        c = MetricsCollector()
        c.record_inference(10.0)
        snap = c.snapshot()
        assert isinstance(snap, EngineMetrics)
        with pytest.raises((AttributeError, TypeError)):
            snap.frames_total = 99  # type: ignore[misc]

    def test_snapshot_fps_positive(self) -> None:
        c = MetricsCollector()
        c.record_inference(5.0, batch_size=10)
        snap = c.snapshot()
        assert snap.fps > 0.0
        assert snap.uptime_s > 0.0

    def test_snapshot_latency_stats(self) -> None:
        c = MetricsCollector()
        for ms in [1.0, 2.0, 3.0, 4.0, 5.0]:
            c.record_inference(ms)
        snap = c.snapshot()
        assert snap.inference_mean_ms == pytest.approx(3.0)
        assert snap.inference_p50_ms > 0.0
        assert snap.inference_p95_ms >= snap.inference_p50_ms

    def test_snapshot_zero_elapsed_fps(self) -> None:
        """When elapsed is 0, fps must not raise ZeroDivisionError."""
        c = MetricsCollector()
        # elapsed can't actually be 0 but snapshot has guard: elapsed > 0
        snap = c.snapshot()
        assert snap.fps >= 0.0

    def test_reset_zeroes_all_counters(self) -> None:
        c = MetricsCollector()
        c.record_inference(5.0, batch_size=3)
        c.record_error()
        c.reset()
        assert c.frames_total == 0
        assert c.errors_total == 0
        assert c.last_frame_time == 0.0
        snap = c.snapshot()
        assert snap.inference_mean_ms == 0.0
        assert snap.inference_p50_ms == 0.0

    def test_reset_restarts_uptime(self) -> None:
        c = MetricsCollector()
        time.sleep(0.02)
        before_reset = c.snapshot().uptime_s
        c.reset()
        after_reset = c.snapshot().uptime_s
        assert after_reset < before_reset

    def test_disabled_noop(self) -> None:
        c = MetricsCollector(enabled=False)
        c.record_inference(5.0, batch_size=10)
        c.record_error()
        assert c.frames_total == 0
        assert c.errors_total == 0
        assert c.last_frame_time == 0.0

    def test_disabled_snapshot_still_works(self) -> None:
        c = MetricsCollector(enabled=False)
        snap = c.snapshot()
        assert snap.frames_total == 0
        assert snap.fps == 0.0

    def test_multiple_snapshots_independent(self) -> None:
        c = MetricsCollector()
        c.record_inference(5.0)
        snap1 = c.snapshot()
        c.record_inference(5.0)
        snap2 = c.snapshot()
        assert snap2.frames_total == snap1.frames_total + 1

    def test_record_overhead_under_5us(self) -> None:
        """Smoke-check: 10k calls must complete in <200ms (budget is <5µs each)."""
        c = MetricsCollector()
        t0 = time.perf_counter()
        for _ in range(10_000):
            c.record_inference(5.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200, f"record_inference too slow: {elapsed_ms:.1f}ms for 10k calls"


# ---------------------------------------------------------------------------
# EngineMetrics
# ---------------------------------------------------------------------------


class TestEngineMetrics:
    def test_is_frozen(self) -> None:
        m = EngineMetrics(
            frames_total=10,
            errors_total=0,
            inference_mean_ms=5.0,
            inference_p50_ms=4.0,
            inference_p95_ms=8.0,
            inference_p99_ms=10.0,
            fps=30.0,
            uptime_s=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            m.frames_total = 99  # type: ignore[misc]

    def test_slots_defined(self) -> None:
        m = EngineMetrics(
            frames_total=1,
            errors_total=0,
            inference_mean_ms=1.0,
            inference_p50_ms=1.0,
            inference_p95_ms=1.0,
            inference_p99_ms=1.0,
            fps=1.0,
            uptime_s=1.0,
        )
        assert not hasattr(m, "__dict__"), "EngineMetrics should use __slots__"

    def test_all_fields_accessible(self) -> None:
        m = EngineMetrics(
            frames_total=100,
            errors_total=5,
            inference_mean_ms=3.5,
            inference_p50_ms=3.0,
            inference_p95_ms=6.0,
            inference_p99_ms=9.0,
            fps=25.0,
            uptime_s=4.0,
        )
        assert m.frames_total == 100
        assert m.errors_total == 5
        assert m.inference_mean_ms == pytest.approx(3.5)
        assert m.inference_p50_ms == pytest.approx(3.0)
        assert m.inference_p95_ms == pytest.approx(6.0)
        assert m.inference_p99_ms == pytest.approx(9.0)
        assert m.fps == pytest.approx(25.0)
        assert m.uptime_s == pytest.approx(4.0)

    def test_public_export(self) -> None:
        import yowo

        assert hasattr(yowo, "EngineMetrics")
        assert hasattr(yowo, "MetricsCollector")
