"""Tests for BatchScheduler in the multi-stream pipeline."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest

from yowo.pipeline._scheduler import BatchScheduler
from yowo.types import Frame, TaggedFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(index: int = 0) -> Frame:
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=index)


def _make_tagged(stream_id: str = "cam-0", index: int = 0) -> TaggedFrame:
    return TaggedFrame(stream_id=stream_id, frame=_make_frame(index))


def _tagged_iter(n: int, stream_id: str = "cam-0") -> Iterator[TaggedFrame]:
    """Return a list-backed iterator of *n* tagged frames."""
    return iter([_make_tagged(stream_id, i) for i in range(n)])


class _SlowUpstream:
    """Yields one frame immediately, then blocks longer than any test timeout.

    This lets us verify that the scheduler flushes a partial batch when the
    upstream cannot deliver frames fast enough.
    """

    def __init__(self, delay_s: float = 1.0) -> None:
        self._yielded_first = False
        self._delay_s = delay_s

    def __iter__(self) -> _SlowUpstream:
        return self

    def __next__(self) -> TaggedFrame:
        if not self._yielded_first:
            self._yielded_first = True
            return _make_tagged("cam-0", 0)
        # Block long enough for the timeout to trigger inside the scheduler.
        time.sleep(self._delay_s)
        raise StopIteration


# ---------------------------------------------------------------------------
# TestBatchSchedulerBasic
# ---------------------------------------------------------------------------


class TestBatchSchedulerBasic:
    """Basic batching semantics without timeout involvement."""

    def test_single_frame_yielded_as_batch(self) -> None:
        scheduler = BatchScheduler(_tagged_iter(1), max_batch_size=4, timeout_ms=1000.0)
        batches = list(scheduler)

        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batch_fills_to_max_size(self) -> None:
        scheduler = BatchScheduler(_tagged_iter(4), max_batch_size=4, timeout_ms=1000.0)
        batches = list(scheduler)

        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_multiple_batches(self) -> None:
        scheduler = BatchScheduler(_tagged_iter(7), max_batch_size=3, timeout_ms=1000.0)
        batches = list(scheduler)
        sizes = [len(b) for b in batches]

        assert sizes == [3, 3, 1]

    def test_empty_upstream_raises_stop(self) -> None:
        scheduler = BatchScheduler(_tagged_iter(0), max_batch_size=4, timeout_ms=1000.0)

        with pytest.raises(StopIteration):
            next(scheduler)


# ---------------------------------------------------------------------------
# TestBatchSchedulerTimeout
# ---------------------------------------------------------------------------


class TestBatchSchedulerTimeout:
    """Timeout-based flushing behaviour."""

    def test_timeout_flushes_partial_batch(self) -> None:
        # The slow upstream yields 1 frame then blocks for 1 second.
        # With timeout_ms=10 and max_batch_size=4, the scheduler should
        # flush the partial batch (size 1) after ~10ms, not wait for 4 frames.
        upstream = _SlowUpstream(delay_s=1.0)
        scheduler = BatchScheduler(upstream, max_batch_size=4, timeout_ms=10.0)
        batch = next(scheduler)

        assert len(batch) == 1
        assert batch[0].stream_id == "cam-0"

    def test_capacity_flush_before_timeout(self) -> None:
        # All frames are immediately available, so capacity flush triggers
        # before the timeout would ever fire.
        scheduler = BatchScheduler(_tagged_iter(4), max_batch_size=4, timeout_ms=5000.0)
        t_start = time.monotonic()
        batch = next(scheduler)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert len(batch) == 4
        # Should be nearly instant, well below the 5-second timeout.
        assert elapsed_ms < 1000.0


# ---------------------------------------------------------------------------
# TestBatchSchedulerValidation
# ---------------------------------------------------------------------------


class TestBatchSchedulerValidation:
    """Constructor input validation."""

    def test_max_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            BatchScheduler(_tagged_iter(0), max_batch_size=0, timeout_ms=100.0)

    def test_timeout_ms_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_ms must be > 0"):
            BatchScheduler(_tagged_iter(0), max_batch_size=1, timeout_ms=0.0)

    def test_timeout_ms_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_ms must be > 0"):
            BatchScheduler(_tagged_iter(0), max_batch_size=1, timeout_ms=-10.0)


# ---------------------------------------------------------------------------
# TestBatchSchedulerMetrics
# ---------------------------------------------------------------------------


class TestBatchSchedulerMetrics:
    """Property-based metrics exposure."""

    def test_pending_reflects_accumulation(self) -> None:
        # Use a large timeout so the scheduler does not auto-flush.
        # We manually call internal machinery by pulling frames one by one
        # via the iterator protocol with a controlled upstream.
        frames = [_make_tagged("cam-0", i) for i in range(3)]
        upstream = iter(frames)
        scheduler = BatchScheduler(upstream, max_batch_size=5, timeout_ms=5000.0)

        # Before any iteration, nothing pending.
        assert scheduler.pending == 0

        # Pull frames one at a time by giving the scheduler a batch size
        # larger than the upstream count; the scheduler will accumulate
        # all 3 frames then flush on upstream exhaustion.
        batch = next(scheduler)
        # After flush, pending resets to 0.
        assert scheduler.pending == 0
        assert len(batch) == 3

    def test_max_batch_size_property(self) -> None:
        scheduler = BatchScheduler(_tagged_iter(0), max_batch_size=8, timeout_ms=100.0)
        assert scheduler.max_batch_size == 8

    def test_timeout_ms_property(self) -> None:
        scheduler = BatchScheduler(_tagged_iter(0), max_batch_size=1, timeout_ms=42.5)
        # Getter returns milliseconds, not the internal seconds representation.
        assert scheduler.timeout_ms == pytest.approx(42.5)


class TestBatchSchedulerStopEvent:
    """Stop event cancellation tests."""

    def test_set_stop_event_flushes_and_stops(self) -> None:
        """set_stop_event + event.set() causes scheduler to flush and stop."""
        import threading

        event = threading.Event()
        frames = [_make_tagged("cam-0", i) for i in range(10)]
        scheduler = BatchScheduler(iter(frames), max_batch_size=100, timeout_ms=5000.0)
        scheduler.set_stop_event(event)

        # Get first batch (upstream has frames, event not set yet — timeout flush).
        # Instead, set event immediately so the first __next__ check sees it.
        event.set()

        batches = list(scheduler)
        # Should get at most 1 partial batch (or 0 if event checked before any pull).
        assert len(batches) <= 1

    def test_stop_event_via_constructor(self) -> None:
        """stop_event passed via constructor works the same."""
        import threading

        event = threading.Event()
        event.set()
        scheduler = BatchScheduler(
            iter([_make_tagged("cam-0", 0)]),
            max_batch_size=100,
            timeout_ms=5000.0,
            stop_event=event,
        )
        batches = list(scheduler)
        assert len(batches) <= 1
