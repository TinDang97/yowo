"""Tests for FrameCollector multi-stream round-robin dispatch."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest

from yowo.pipeline._collector import FrameCollector
from yowo.types import Frame, StreamState, TaggedFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(index: int) -> Frame:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[0, 0] = index % 256
    return Frame(pixels=pixels, source_id="test", frame_index=index, timestamp_ms=float(index))


class _MockSource:
    """Minimal FrameSource for testing."""

    def __init__(
        self,
        n: int,
        delay: float = 0.0,
        error_at: int | None = None,
        *,
        is_live: bool = False,
    ) -> None:
        self._n = n
        self._delay = delay
        self._error_at = error_at
        self._is_live = is_live
        self.closed = False

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def total_frames(self) -> int | None:
        return None if self._is_live else self._n

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
# TestFrameCollectorBasic
# ---------------------------------------------------------------------------


class TestFrameCollectorBasic:
    def test_single_stream_yields_all_frames(self) -> None:
        source = _MockSource(5)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("cam-0", source)

        tagged: list[TaggedFrame] = list(collector)
        collector.close()

        assert len(tagged) == 5
        assert all(t.stream_id == "cam-0" for t in tagged)
        assert [t.frame.frame_index for t in tagged] == list(range(5))

    def test_multiple_streams_yield_all_frames(self) -> None:
        src_a = _MockSource(3)
        src_b = _MockSource(3)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("a", src_a)
        collector.add_stream("b", src_b)

        tagged = list(collector)
        collector.close()

        assert len(tagged) == 6
        ids = {t.stream_id for t in tagged}
        assert ids == {"a", "b"}
        assert sum(1 for t in tagged if t.stream_id == "a") == 3
        assert sum(1 for t in tagged if t.stream_id == "b") == 3

    def test_empty_source_terminates(self) -> None:
        source = _MockSource(0)
        collector = FrameCollector(max_queue_size=2)
        collector.add_stream("empty", source)

        tagged = list(collector)
        collector.close()

        assert len(tagged) == 0

    def test_context_manager_closes(self) -> None:
        source = _MockSource(3)
        with FrameCollector(max_queue_size=10) as collector:
            collector.add_stream("cam-0", source)
            _ = list(collector)

        # After exiting context, source should be closed.
        assert source.closed


# ---------------------------------------------------------------------------
# TestStreamManagement
# ---------------------------------------------------------------------------


class TestStreamManagement:
    def test_add_stream_starts_reader(self) -> None:
        source = _MockSource(3)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)

        tagged = list(collector)
        collector.close()

        assert len(tagged) == 3

    def test_remove_stream_stops_it(self) -> None:
        src_a = _MockSource(5, delay=0.02)
        src_b = _MockSource(5, delay=0.02)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("a", src_a)
        collector.add_stream("b", src_b)

        # Remove stream "a" before consuming everything
        collector.remove_stream("a")

        # Remaining stream "b" should still deliver frames
        tagged = list(collector)
        collector.close()

        assert all(t.stream_id == "b" for t in tagged)
        assert src_a.closed

    def test_duplicate_stream_id_raises(self) -> None:
        source1 = _MockSource(1)
        source2 = _MockSource(1)
        collector = FrameCollector(max_queue_size=2)
        collector.add_stream("dup", source1)

        with pytest.raises(ValueError, match="already registered"):
            collector.add_stream("dup", source2)

        collector.close()

    def test_add_after_close_raises(self) -> None:
        collector = FrameCollector(max_queue_size=2)
        collector.close()

        with pytest.raises(RuntimeError, match="closed"):
            collector.add_stream("late", _MockSource(1))

    def test_remove_nonexistent_is_noop(self) -> None:
        collector = FrameCollector(max_queue_size=2)
        # Must not raise
        collector.remove_stream("nope")
        collector.close()


# ---------------------------------------------------------------------------
# TestStreamStates
# ---------------------------------------------------------------------------


class TestStreamStates:
    def test_exhausted_stream_marked_stopped(self) -> None:
        source = _MockSource(3)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)

        _ = list(collector)

        states = collector.stream_states
        assert states["s1"] == StreamState.STOPPED
        collector.close()

    def test_erroring_stream_marked_error(self) -> None:
        source = _MockSource(5, error_at=2)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)

        # Drain -- the error is caught by FrameCollector._try_get
        _ = list(collector)

        states = collector.stream_states
        assert states["s1"] == StreamState.ERROR
        collector.close()

    def test_stream_count_and_active_count(self) -> None:
        src_a = _MockSource(2)
        src_b = _MockSource(2)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("a", src_a)
        collector.add_stream("b", src_b)

        assert collector.stream_count == 2
        assert collector.active_count == 2

        # Exhaust both streams
        _ = list(collector)

        # Both should now be STOPPED
        assert collector.stream_count == 2
        assert collector.active_count == 0
        collector.close()


# ---------------------------------------------------------------------------
# TestAutoPolicy
# ---------------------------------------------------------------------------


class TestAutoPolicy:
    def test_live_source_gets_latest_policy(self) -> None:
        """Live source auto-selects LATEST policy; should not hang."""
        source = _MockSource(5, is_live=True)
        collector = FrameCollector(max_queue_size=2)
        collector.add_stream("live", source)

        tagged = list(collector)
        collector.close()

        # LATEST may drop frames, but we should get at least 1 and at most 5
        assert 1 <= len(tagged) <= 5
        assert all(t.stream_id == "live" for t in tagged)

    def test_offline_source_gets_none_policy(self) -> None:
        """Offline source auto-selects NONE policy; all frames delivered."""
        source = _MockSource(5, is_live=False)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("offline", source)

        tagged = list(collector)
        collector.close()

        assert len(tagged) == 5
        assert [t.frame.frame_index for t in tagged] == list(range(5))


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_close_is_idempotent(self) -> None:
        source = _MockSource(3)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)
        _ = list(collector)

        collector.close()
        collector.close()  # second close must not raise

        assert source.closed

    def test_max_queue_size_validation(self) -> None:
        with pytest.raises(ValueError, match="max_queue_size"):
            FrameCollector(max_queue_size=0)

    def test_stream_errors_captures_exception(self) -> None:
        """stream_errors returns the exception for ERROR streams."""
        source = _MockSource(5, error_at=1)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("err", source)

        _ = list(collector)

        errors = collector.stream_errors
        assert "err" in errors
        assert "source error at frame 1" in str(errors["err"])
        collector.close()

    def test_stream_errors_empty_when_no_errors(self) -> None:
        source = _MockSource(2)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("ok", source)

        _ = list(collector)

        assert collector.stream_errors == {}
        collector.close()


class TestCollectorTypeGuard:
    def test_non_frame_item_raises_type_error(self) -> None:
        """FrameCollector raises TypeError if reader yields a non-Frame item."""
        from unittest.mock import MagicMock

        from yowo.io._reader import PreparedItem

        fake_tensor = MagicMock()
        fake_frame = _make_frame(0)
        prepared = PreparedItem(tensor=fake_tensor, frame=fake_frame)

        source = _MockSource(1)
        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("s0", source)

        # Replace the reader with a mock that returns a PreparedItem.
        entry = collector._streams["s0"]
        mock_reader = MagicMock()
        mock_reader.get.return_value = prepared
        mock_reader.is_exhausted = False
        entry.reader = mock_reader

        with pytest.raises(TypeError, match="expects Frame"):
            next(iter(collector))

        collector.close()
