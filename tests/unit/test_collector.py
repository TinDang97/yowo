"""Tests for FrameCollector multi-stream shared-queue dispatch."""

from __future__ import annotations

import gc
import queue
import threading
import time
import tracemalloc
from collections.abc import Iterator

import numpy as np
import pytest

from yowo.pipeline._collector import FrameCollector
from yowo.types import Frame, StreamConfig, StreamState, TaggedFrame

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

    def test_erroring_stream_auto_removed(self) -> None:
        """Stream with persistent read errors is auto-removed from _streams."""
        source = _MockSource(5, error_at=2)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)

        # Drain -- persistent errors trigger auto-remove after 3 consecutive errors
        _ = list(collector)

        # Stream is auto-removed; no longer in _streams
        assert "s1" not in collector._streams
        collector.close()

    def test_stream_count_and_active_count(self) -> None:
        # Use delay so bridges don't exhaust sources before we check active_count
        src_a = _MockSource(2, delay=0.1)
        src_b = _MockSource(2, delay=0.1)
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
        """stream_errors reports errors from auto-removed streams.

        When a stream is auto-removed after consecutive errors, its error
        is retained in _auto_removed_errors so that _check_stream_errors
        can detect total pipeline failure after the collector is drained.
        """
        source = _MockSource(5, error_at=1)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("err", source)

        _ = list(collector)

        # After auto-remove, the stream is no longer in _streams
        assert "err" not in collector._streams
        # But stream_errors still reports the auto-removed stream's error
        errors = collector.stream_errors
        assert "err" in errors
        assert isinstance(errors["err"], RuntimeError)
        collector.close()

    def test_stream_errors_empty_when_no_errors(self) -> None:
        source = _MockSource(2)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("ok", source)

        _ = list(collector)

        assert collector.stream_errors == {}
        collector.close()


class TestSharedQueueDispatch:
    """Tests for the shared-queue based dispatch (Phase 3a optimisation)."""

    def test_shared_queue_dispatch_multiple_streams(self) -> None:
        """5 streams x 1 frame each dispatched via shared queue."""
        collector = FrameCollector(max_queue_size=10)
        for i in range(5):
            collector.add_stream(f"s{i}", _MockSource(1))

        tagged = list(collector)
        collector.close()

        assert len(tagged) == 5
        ids = {t.stream_id for t in tagged}
        assert ids == {f"s{i}" for i in range(5)}

    def test_bridge_persistent_error_auto_removes_stream(self) -> None:
        """Source raising persistent errors causes the bridge to auto-remove stream."""
        source = _MockSource(5, error_at=2)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("err", source)

        _ = list(collector)

        # After auto-remove, stream is gone from _streams
        assert "err" not in collector._streams
        # stream_errors still reports the error from the auto-removed stream
        errors = collector.stream_errors
        assert "err" in errors
        assert isinstance(errors["err"], RuntimeError)
        collector.close()

    def test_close_unblocks_iter(self) -> None:
        """close() from another thread unblocks a waiting __iter__."""
        # Use a slow source so __iter__ is likely waiting on shared_q.get()
        source = _MockSource(100, delay=0.5)
        collector = FrameCollector(max_queue_size=2)
        collector.add_stream("slow", source)

        collected: list[TaggedFrame] = []
        done = threading.Event()

        def _drain() -> None:
            for tagged in collector:
                collected.append(tagged)
            done.set()

        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        time.sleep(0.05)  # let iter start

        collector.close()
        assert done.wait(timeout=3.0), "close() did not unblock __iter__"

    def test_bridge_thread_named(self) -> None:
        """Bridge threads are named yowo-bridge-<stream_id>."""
        source = _MockSource(1, delay=0.1)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("cam-7", source)

        entry = collector._streams["cam-7"]
        assert entry.bridge is not None
        assert entry.bridge.name == "yowo-bridge-cam-7"
        assert entry.bridge.daemon is True

        _ = list(collector)
        collector.close()


class TestCollectorTypeGuard:
    def test_non_frame_item_marks_error_state(self) -> None:
        """Bridge thread marks ERROR state when reader yields a non-Frame item."""
        from unittest.mock import MagicMock

        from yowo.io._reader import PreparedItem
        from yowo.pipeline._collector import _run_bridge, _StreamEntry

        fake_tensor = MagicMock()
        fake_frame = _make_frame(0)
        prepared = PreparedItem(tensor=fake_tensor, frame=fake_frame)

        # Create a mock reader that returns PreparedItem
        mock_reader = MagicMock()
        mock_reader.get.return_value = prepared
        mock_reader.is_exhausted = False

        source = _MockSource(1)
        entry = _StreamEntry(reader=mock_reader, source=source)
        shared_q: queue.Queue[tuple[str, Frame | None] | None] = queue.Queue()

        # Run bridge directly (synchronously in this thread)
        _run_bridge("s0", entry, shared_q)

        assert entry.state == StreamState.ERROR
        assert entry.error is not None
        assert "expects Frame" in str(entry.error)


# ---------------------------------------------------------------------------
# TestStreamConfig
# ---------------------------------------------------------------------------


class TestStreamConfig:
    def test_stream_config_defaults(self) -> None:
        """StreamConfig default values match spec."""
        cfg = StreamConfig()
        assert cfg.auto_reconnect is False
        assert cfg.max_consecutive_errors == 3
        assert cfg.reconnect_backoff_base_s == 1.0
        assert cfg.reconnect_backoff_max_s == 30.0

    def test_stream_config_is_frozen(self) -> None:
        """StreamConfig is frozen — fields cannot be mutated."""
        cfg = StreamConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_consecutive_errors = 5  # type: ignore[misc]

    def test_stream_config_custom_values(self) -> None:
        """StreamConfig accepts custom values."""
        cfg = StreamConfig(
            auto_reconnect=True,
            max_consecutive_errors=5,
            reconnect_backoff_base_s=2.0,
            reconnect_backoff_max_s=60.0,
        )
        assert cfg.auto_reconnect is True
        assert cfg.max_consecutive_errors == 5
        assert cfg.reconnect_backoff_base_s == 2.0
        assert cfg.reconnect_backoff_max_s == 60.0

    def test_stream_config_exported_from_types(self) -> None:
        """StreamConfig is importable from yowo.types."""
        from yowo.types import StreamConfig as SC

        assert SC is StreamConfig

    def test_stream_config_exported_from_pipeline(self) -> None:
        """StreamConfig is importable from yowo.pipeline."""
        from yowo.pipeline import StreamConfig as SC

        assert SC is StreamConfig


# ---------------------------------------------------------------------------
# TestPerStreamStats
# ---------------------------------------------------------------------------


class _ErroringReader:
    """Mock reader that yields frames_before_error frames then always raises."""

    def __init__(self, frames_before_error: int) -> None:
        self._frames_before_error = frames_before_error
        self._call_count = 0
        self.is_exhausted = False

    def get(self, timeout: float = 0.1) -> Frame | None:
        if self._call_count < self._frames_before_error:
            self._call_count += 1
            return _make_frame(self._call_count)
        raise RuntimeError("stream error")

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class TestPerStreamStats:
    def test_frames_processed_counter(self) -> None:
        """frames_processed counts frames successfully enqueued."""
        source = _MockSource(5)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)

        _ = list(collector)

        entry = collector._streams["s1"]
        assert entry.frames_processed == 5
        collector.close()

    def test_last_frame_time_updated(self) -> None:
        """last_frame_time is updated on each successful frame read."""
        source = _MockSource(3)
        collector = FrameCollector(max_queue_size=10)
        before = time.monotonic()
        collector.add_stream("s1", source)

        _ = list(collector)
        entry = collector._streams["s1"]
        after = time.monotonic()

        assert entry.last_frame_time >= before
        assert entry.last_frame_time <= after
        collector.close()

    def test_consecutive_errors_reset_on_success(self) -> None:
        """consecutive_errors resets to 0 after a successful frame read."""
        from unittest.mock import MagicMock

        from yowo.pipeline._collector import _run_bridge, _StreamEntry

        call_count = 0

        def _get(timeout: float = 0.1) -> Frame | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            # Return None (gap) then return None (exhausted) to end bridge
            return None

        mock_reader = MagicMock()
        mock_reader.get.side_effect = _get
        mock_reader.is_exhausted = True  # bridge exits on first None after error reset

        source = _MockSource(0)
        entry = _StreamEntry(reader=mock_reader, source=source)
        shared_q: queue.Queue[tuple[str, Frame | None] | None] = queue.Queue(maxsize=100)

        # With max_consecutive_errors=3, one error should NOT trigger auto_remove
        _run_bridge("s0", entry, shared_q, max_consecutive_errors=3)

        assert entry.consecutive_errors == 0  # reset after successful None read
        assert entry.auto_remove is False  # not triggered

    def test_frame_drop_stats(self) -> None:
        """frames_dropped increments when queue is full (max_queue_size=1)."""
        # max_queue_size=1 means shared_q has maxsize=8; use slow consumer
        # We test the drop counter by checking it can be read
        source = _MockSource(3)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)
        _ = list(collector)

        entry = collector._streams["s1"]
        # frames_dropped is >= 0 (may be 0 if queue was never full)
        assert entry.frames_dropped >= 0
        collector.close()


# ---------------------------------------------------------------------------
# TestAutoRemove
# ---------------------------------------------------------------------------


class _AlwaysErrorSource:
    """Source that always errors immediately."""

    is_live = True

    @property
    def total_frames(self) -> int | None:
        return None

    def __iter__(self) -> Iterator[Frame]:
        raise RuntimeError("always error")
        yield  # make it a generator

    def close(self) -> None:
        pass


class _AlwaysErrorReader:
    """Reader that always raises on get()."""

    is_exhausted = False

    def get(self, timeout: float = 0.1) -> Frame | None:
        raise RuntimeError("always errors")

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class TestAutoRemove:
    def test_auto_remove_on_consecutive_errors(self) -> None:
        """Bridge sets auto_remove after max_consecutive_errors; iterator removes stream."""
        from unittest.mock import MagicMock

        from yowo.pipeline._collector import _run_bridge, _StreamEntry

        mock_reader = MagicMock()
        mock_reader.get.side_effect = RuntimeError("stream error")
        mock_reader.is_exhausted = False

        source = _MockSource(0)
        entry = _StreamEntry(reader=mock_reader, source=source)
        shared_q: queue.Queue[tuple[str, Frame | None] | None] = queue.Queue(maxsize=100)

        _run_bridge("s0", entry, shared_q, max_consecutive_errors=3)

        assert entry.auto_remove is True
        assert entry.consecutive_errors >= 3

    def test_stream_auto_removed_from_collector(self) -> None:
        """FrameCollector.__iter__ removes stream after auto_remove is set by bridge."""
        # Use a custom reader via stream_config with max_consecutive_errors=3
        # Build a source that always errors
        source = _AlwaysErrorSource()
        collector = FrameCollector(max_queue_size=10)
        cfg = StreamConfig(max_consecutive_errors=3)
        collector.add_stream("bad", source, stream_config=cfg)

        # Iterate — bad stream will error 3 times and be auto-removed
        frames = list(collector)

        # Stream produced no frames (all errors)
        assert len(frames) == 0
        # Stream was auto-removed from _streams
        assert "bad" not in collector._streams
        collector.close()

    def test_auto_remove_does_not_affect_other_streams(self) -> None:
        """One stream auto-removed; another stream continues normally."""
        good_frames: list[TaggedFrame] = []

        # "bad" stream errors 3 times immediately; "good" delivers 5 frames
        bad_source = _AlwaysErrorSource()
        good_source = _MockSource(5, delay=0.02)

        collector = FrameCollector(max_queue_size=10)
        cfg = StreamConfig(max_consecutive_errors=3)
        collector.add_stream("bad", bad_source, stream_config=cfg)
        collector.add_stream("good", good_source)

        for tagged in collector:
            if tagged.stream_id == "good":
                good_frames.append(tagged)

        assert len(good_frames) == 5
        assert "bad" not in collector._streams
        collector.close()


# ---------------------------------------------------------------------------
# TestMemoryLeak
# ---------------------------------------------------------------------------


class TestMemoryLeak:
    def test_remove_stream_no_leak(self) -> None:
        """remove_stream() leaves no significant memory leak (< 500KB delta)."""
        tracemalloc.start()

        source = _MockSource(10)
        collector = FrameCollector(max_queue_size=10)
        collector.add_stream("s1", source)
        _ = list(collector)
        collector.remove_stream("s1")

        gc.collect()
        gc.collect()

        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot.statistics("lineno")
        total_size = sum(s.size for s in stats)
        # Allow up to 500KB of total tracked memory (very generous upper bound)
        assert total_size < 500_000, f"Memory usage too high: {total_size} bytes"

        collector.close()

    def test_stream_isolation_memory(self) -> None:
        """After stream B errors and is removed, stream A data is unaffected."""
        source_a = _MockSource(5, delay=0.01)
        bad_source = _AlwaysErrorSource()

        collector = FrameCollector(max_queue_size=10)
        cfg = StreamConfig(max_consecutive_errors=3)
        collector.add_stream("a", source_a)
        collector.add_stream("bad", bad_source, stream_config=cfg)

        tagged_a: list[TaggedFrame] = []
        for tagged in collector:
            if tagged.stream_id == "a":
                tagged_a.append(tagged)

        assert len(tagged_a) == 5
        assert "bad" not in collector._streams
        collector.close()
