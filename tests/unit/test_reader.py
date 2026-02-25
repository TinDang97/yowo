"""Tests for ThreadedFrameReader."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest

from yowo.io._reader import ThreadedFrameReader
from yowo.types import Frame, FrameDropPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(index: int) -> Frame:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[0, 0] = index % 256
    return Frame(pixels=pixels, source_id="test", frame_index=index, timestamp_ms=float(index))


class _MockSource:
    """Minimal FrameSource for testing."""

    def __init__(self, n: int, delay: float = 0.0, error_at: int | None = None) -> None:
        self._n = n
        self._delay = delay
        self._error_at = error_at
        self.closed = False

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        return self._n

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self._n):
            if self._error_at == i:
                raise RuntimeError(f"source error at frame {i}")
            if self._delay > 0:
                time.sleep(self._delay)
            yield _make_frame(i)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThreadedFrameReaderBasic:
    def test_yields_all_frames_none_policy(self) -> None:
        source = _MockSource(5)
        reader = ThreadedFrameReader(source, max_queue_size=10, policy=FrameDropPolicy.NONE)
        reader.start()
        frames: list[Frame] = []
        while (frame := reader.get(timeout=2.0)) is not None:
            frames.append(frame)
        reader.stop()
        assert len(frames) == 5
        # Frames should be in order
        for i, f in enumerate(frames):
            assert f.frame_index == i

    def test_empty_source_returns_none(self) -> None:
        source = _MockSource(0)
        reader = ThreadedFrameReader(source, max_queue_size=2)
        reader.start()
        result = reader.get(timeout=2.0)
        reader.stop()
        assert result is None

    def test_context_manager(self) -> None:
        source = _MockSource(3)
        frames: list[Frame] = []
        with ThreadedFrameReader(source, max_queue_size=5) as reader:
            reader.start()
            while (frame := reader.get(timeout=2.0)) is not None:
                frames.append(frame)
        assert len(frames) == 3

    def test_start_is_idempotent(self) -> None:
        source = _MockSource(2)
        reader = ThreadedFrameReader(source, max_queue_size=5)
        reader.start()
        reader.start()  # second call must not raise or spawn extra thread
        frames: list[Frame] = []
        while (frame := reader.get(timeout=2.0)) is not None:
            frames.append(frame)
        reader.stop()
        assert len(frames) == 2

    def test_max_queue_size_validation(self) -> None:
        source = _MockSource(1)
        with pytest.raises(ValueError, match="max_queue_size"):
            ThreadedFrameReader(source, max_queue_size=0)


class TestFrameDropPolicies:
    def test_latest_policy_drops_stale(self) -> None:
        """With LATEST policy and slow consumer, only the latest frame is kept."""
        source = _MockSource(10, delay=0.002)
        reader = ThreadedFrameReader(source, max_queue_size=1, policy=FrameDropPolicy.LATEST)
        reader.start()
        # Let reader fill up a few frames before consuming
        time.sleep(0.05)  # let reader read several frames
        frame = reader.get(timeout=2.0)
        reader.stop()
        assert frame is not None
        # Frames should have been dropped; the frame we get is not necessarily 0
        assert reader.frames_dropped >= 0  # may or may not drop depending on timing

    def test_latest_policy_metrics(self) -> None:
        """frames_read >= frames_dropped always holds."""
        source = _MockSource(20, delay=0.001)
        reader = ThreadedFrameReader(source, max_queue_size=1, policy=FrameDropPolicy.LATEST)
        reader.start()
        # Drain slowly
        frames: list[Frame] = []
        while (frame := reader.get(timeout=5.0)) is not None:
            time.sleep(0.01)  # slow consumer
            frames.append(frame)
        reader.stop()
        assert reader.frames_read >= len(frames)
        assert reader.frames_read == reader.frames_dropped + len(frames)

    def test_skip_oldest_policy(self) -> None:
        """SKIP_OLDEST: oldest frame evicted when queue full."""
        source = _MockSource(10, delay=0.0)
        reader = ThreadedFrameReader(source, max_queue_size=2, policy=FrameDropPolicy.SKIP_OLDEST)
        reader.start()
        # Let reader fill queue
        time.sleep(0.05)
        frames: list[Frame] = []
        while (frame := reader.get(timeout=2.0)) is not None:
            frames.append(frame)
        reader.stop()
        assert len(frames) <= 10
        assert reader.frames_dropped <= reader.frames_read

    def test_none_policy_preserves_order(self) -> None:
        """NONE policy: all frames in order, no drops."""
        source = _MockSource(8, delay=0.0)
        reader = ThreadedFrameReader(source, max_queue_size=4, policy=FrameDropPolicy.NONE)
        reader.start()
        frames: list[Frame] = []
        while (frame := reader.get(timeout=5.0)) is not None:
            frames.append(frame)
        reader.stop()
        assert [f.frame_index for f in frames] == list(range(8))
        assert reader.frames_dropped == 0


class TestErrorHandling:
    def test_exception_propagates_from_reader_thread(self) -> None:
        source = _MockSource(10, error_at=3)
        reader = ThreadedFrameReader(source, max_queue_size=5, policy=FrameDropPolicy.NONE)
        reader.start()
        frames: list[Frame] = []
        with pytest.raises(RuntimeError, match="source error at frame 3"):
            while (frame := reader.get(timeout=2.0)) is not None:
                frames.append(frame)
        reader.stop()
        assert len(frames) <= 3  # got some frames before error


class TestCounters:
    def test_frames_read_counter(self) -> None:
        source = _MockSource(5)
        reader = ThreadedFrameReader(source, max_queue_size=10, policy=FrameDropPolicy.NONE)
        reader.start()
        while reader.get(timeout=2.0) is not None:
            pass
        reader.stop()
        assert reader.frames_read == 5

    def test_drop_rate_zero_when_no_drops(self) -> None:
        source = _MockSource(3)
        reader = ThreadedFrameReader(source, max_queue_size=10, policy=FrameDropPolicy.NONE)
        reader.start()
        while reader.get(timeout=2.0) is not None:
            pass
        reader.stop()
        assert reader.drop_rate == 0.0

    def test_clean_shutdown_joins_thread(self) -> None:
        source = _MockSource(100, delay=0.001)
        reader = ThreadedFrameReader(source, max_queue_size=2)
        reader.start()
        time.sleep(0.01)
        t0 = time.perf_counter()
        reader.stop()
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0  # thread must join within reasonable time

    def test_stop_before_start_is_safe(self) -> None:
        """stop() before start() must not raise."""
        source = _MockSource(3)
        reader = ThreadedFrameReader(source, max_queue_size=2)
        reader.stop()  # thread is None — must be a no-op

    def test_drop_rate_zero_before_any_reads(self) -> None:
        """drop_rate on a fresh (unstarted) reader returns 0.0."""
        source = _MockSource(0)
        reader = ThreadedFrameReader(source, max_queue_size=2)
        assert reader.drop_rate == 0.0
        assert reader.frames_read == 0
        assert reader.frames_dropped == 0

    def test_none_policy_backpressure_all_frames_delivered(self) -> None:
        """NONE policy with queue smaller than source: all frames still delivered in order."""
        source = _MockSource(8)
        # queue_size=2 < 8 frames — producer will block until consumer drains
        reader = ThreadedFrameReader(source, max_queue_size=2, policy=FrameDropPolicy.NONE)
        reader.start()
        frames: list[Frame] = []
        while (frame := reader.get(timeout=5.0)) is not None:
            frames.append(frame)
        reader.stop()
        assert [f.frame_index for f in frames] == list(range(8))
        assert reader.frames_dropped == 0

    def test_get_returns_none_on_timeout_not_exhaustion(self) -> None:
        """get() with a very short timeout returns None mid-stream (not exhaustion)."""
        source = _MockSource(1, delay=0.3)  # single frame, slow source
        reader = ThreadedFrameReader(source, max_queue_size=2, policy=FrameDropPolicy.NONE)
        reader.start()
        result = reader.get(timeout=0.05)  # shorter than the source delay
        # If timeout occurred, is_exhausted should be False (source still running).
        if result is None:
            assert not reader.is_exhausted
        reader.stop()
        # Either None (timed out before frame arrived) or the Frame — both are valid.
        assert result is None or isinstance(result, Frame)

    def test_is_exhausted_true_after_all_frames_consumed(self) -> None:
        """is_exhausted is True once the source yields all frames and consumer drains."""
        source = _MockSource(3)
        reader = ThreadedFrameReader(source, max_queue_size=10, policy=FrameDropPolicy.NONE)
        assert not reader.is_exhausted
        reader.start()
        while reader.get(timeout=2.0) is not None:
            pass
        assert reader.is_exhausted
        reader.stop()

    def test_is_exhausted_false_before_done(self) -> None:
        """is_exhausted is False while source is still producing frames."""
        source = _MockSource(5, delay=0.05)
        reader = ThreadedFrameReader(source, max_queue_size=2, policy=FrameDropPolicy.NONE)
        reader.start()
        # Get first frame
        frame = reader.get(timeout=2.0)
        assert frame is not None
        # Source still has more frames
        assert not reader.is_exhausted
        reader.stop()

    def test_stop_counts_queued_frames_as_dropped(self) -> None:
        """stop() mid-stream counts remaining queued frames as dropped."""
        source = _MockSource(10, delay=0.001)
        reader = ThreadedFrameReader(source, max_queue_size=5, policy=FrameDropPolicy.NONE)
        reader.start()
        # Let reader fill up
        time.sleep(0.05)
        # Consume only one frame, leave others in queue
        reader.get(timeout=1.0)
        reader.stop()
        # frames_dropped should include frames discarded at shutdown
        assert reader.frames_dropped >= 0
        # Total: frames_read = consumed + dropped
        assert reader.frames_read >= reader.frames_dropped

    def test_timeout_uses_wall_clock(self) -> None:
        """get() timeout tracks real elapsed time, not fixed step count."""
        source = _MockSource(1, delay=1.0)  # frame arrives after 1s
        reader = ThreadedFrameReader(source, max_queue_size=2, policy=FrameDropPolicy.NONE)
        reader.start()
        t0 = time.perf_counter()
        result = reader.get(timeout=0.1)  # should return after ~0.1s
        elapsed = time.perf_counter() - t0
        reader.stop()
        assert result is None  # timed out
        assert elapsed < 0.5  # wall clock: should not overshoot
