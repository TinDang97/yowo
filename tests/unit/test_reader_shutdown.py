"""The capture being read is the capture the source holds, and a cycle leaves nothing.

Red-first for ADD task `reader-shutdown`.

Measured 2026-09-11 against the real classes: `__iter__` binds `cap` as a LOCAL while
`reconnect()` releases it, opens a replacement and rebinds `self._active_cap`. The
iterator never sees the replacement.

    cap #1 released · cap #2 opened, set as _active_cap · iterator still reads #1
    read fails -> retry opens #3
    ORPHANED (open, unreleased, unreachable): [2]

Every periodic reconnect leaks exactly one capture -- while logging "released and
re-opened capture to prevent memory leak".

The orphan checks assert capture IDENTITY, never a tally. m2's box says a count check
cannot catch this, and it is right: the orphan is replaced, so the counts balance.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator
from unittest import mock

import numpy as np
import pytest

from yowo.io import _source as source_mod
from yowo.io._reader import ThreadedFrameReader
from yowo.io._source import RTSPStreamSource


class TrackedCap:
    """Every capture ever built, and whether it was released."""

    def __init__(self, url: str, api: object = None, params: object = None) -> None:
        self.released = False
        self.reads = 0
        self.id = len(LIVE) + 1
        LIVE.append(self)

    def isOpened(self) -> bool:
        return _STATE["can_open"]

    def read(self) -> tuple[bool, object]:
        self.reads += 1
        if self.released:
            return False, None
        return True, np.zeros((2, 2, 3), np.uint8)

    def release(self) -> None:
        self.released = True


LIVE: list[TrackedCap] = []
_STATE: dict[str, object] = {"can_open": True}


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    LIVE.clear()
    _STATE["can_open"] = True
    yield


def _patched_cv2():
    return mock.patch.object(source_mod, "cv2", mock.MagicMock(VideoCapture=TrackedCap))


def _orphans(src: RTSPStreamSource) -> list[int]:
    """Captures that are open, unreleased, and not the one the source holds."""
    return [c.id for c in LIVE if not c.released and c is not src._active_cap]


# ---------------------------------------------------------------------------
# M1, M2 — the rebind is observed, and nothing is left behind
# ---------------------------------------------------------------------------


def test_the_next_read_uses_the_capture_reconnect_installed() -> None:
    """covers: M1, A2, E1, R:COUNTONLY — by identity; a tally balances here.

    The defect is not "too many captures". It is "the wrong one", and only object
    identity can tell those apart.
    """
    with _patched_cv2():
        src = RTSPStreamSource("rtsp://h/s")
        it = iter(src)
        next(it)
        src.reconnect()
        installed = src._active_cap
        next(it)

    assert installed is not None
    # "Was it READ from" -- not "was it released". The first draft asserted the latter
    # and passed against the bug: the orphan is never released either, which is the
    # whole problem. Asserting a proxy for the subject proves nothing about the subject.
    assert installed.reads > 0, (
        f"the capture reconnect installed (#{installed.id}) was never read from. The "
        "loop is still holding its own local, so the replacement is dead on arrival "
        "and the reconnect achieved nothing but a leak (M1)"
    )


def test_a_reconnect_orphans_no_capture() -> None:
    """covers: M2, E2, R:ORPHAN — measured today: cap #2 open, unreleased, unreachable."""
    with _patched_cv2():
        src = RTSPStreamSource("rtsp://h/s")
        it = iter(src)
        next(it)
        src.reconnect()
        next(it)
        leaked = _orphans(src)

    assert leaked == [], (
        f"captures {leaked} are open, unreleased and unreachable after one reconnect. "
        "Every periodic reconnect leaks one — while logging that it prevents a leak "
        "(R:ORPHAN)"
    )


def test_the_retry_path_publishes_its_capture_to_the_source() -> None:
    """covers: A5, M2 — the reopen must publish, not keep its own local.

    Found by mutation: assigning the reopened capture to a LOCAL left every check green,
    because no scenario here drove the retry path at all -- after a reconnect the read
    always succeeded. A5 was written down and bound to nothing.

    If the reopen keeps its handle private, the next pass re-reads `_active_cap` and gets
    the OLD one, and the newly opened capture is orphaned -- the same defect this node
    fixes, relocated from `reconnect()` into the retry branch.
    """
    slept: list[float] = []
    with (
        _patched_cv2(),
        mock.patch.object(
            source_mod, "time", mock.Mock(monotonic=time.monotonic, sleep=slept.append)
        ),
    ):
        src = RTSPStreamSource("rtsp://h/s")
        it = iter(src)
        next(it)
        # Break the capture the loop is about to read, forcing the retry branch.
        first = src._active_cap
        assert first is not None
        first.release()
        next(it)
        reopened = src._active_cap

    assert reopened is not first, "the retry branch never opened a replacement"
    assert reopened is not None and reopened.reads > 0, (
        f"the capture the retry branch opened (#{reopened.id if reopened else '?'}) was "
        "never read from — it was kept as a local, so the next pass re-read the old "
        "handle and this one is orphaned (A5, M2)"
    )
    assert _orphans(src) == [], f"the retry branch orphaned {_orphans(src)}"


def test_the_active_capture_is_never_none_while_iterating() -> None:
    """covers: A4 — a None here means a bug elsewhere; assert rather than tolerate."""
    with _patched_cv2():
        src = RTSPStreamSource("rtsp://h/s")
        it = iter(src)
        for _ in range(3):
            next(it)
            assert src._active_cap is not None, "the source lost its capture mid-iteration"


def test_a_periodic_reconnect_is_not_an_outage() -> None:
    """covers: M3, E3, R:FAKEOUTAGE — nothing was wrong, so nothing should be reported.

    Only observable since `rtsp-reconnect-correctness` landed: the failed read on the
    released handle entered the outage path, costing a backoff sleep and emitting
    "recovered after 1.0s" on a perfectly healthy stream. A line that cries wolf every
    reconnect interval trains an operator to ignore the line that matters.
    """
    slept: list[float] = []
    fake_time = mock.Mock(monotonic=time.monotonic, sleep=slept.append)
    with (
        _patched_cv2(),
        mock.patch.object(source_mod, "time", fake_time),
        mock.patch.object(source_mod, "logger") as log,
    ):
        src = RTSPStreamSource("rtsp://h/s")
        it = iter(src)
        next(it)
        src.reconnect()
        next(it)

    assert slept == [], f"a healthy reconnect cost a backoff sleep of {slept}s (R:FAKEOUTAGE)"
    recovered = [c for c in log.method_calls if "recovered" in str(c)]
    assert recovered == [], (
        f"a healthy reconnect reported a recovery: {recovered}. Nothing was wrong, so "
        "nothing recovered (R:FAKEOUTAGE)"
    )


# ---------------------------------------------------------------------------
# M4, M5 — a release cycle leaves nothing, on a guarantee that is stated
# ---------------------------------------------------------------------------


def test_stop_does_not_close_a_source_the_caller_owns() -> None:
    """covers: M4, E5 — one owner, and it is not this class.

    Authored the other way round and corrected on evidence: `_streaming.py` calls
    `source.close()` immediately after `stop()` on every streaming path, so closing here
    is a SECOND call, not a fix. Five engine checks pin "closed once" and they are right.

    RESIDUAL, named rather than hidden: a `ThreadedFrameReader` driven directly, outside
    the engine, still relies on refcounting to free the capture. That is a real gap and it
    belongs to whichever node owns standalone reader use, not to this one.
    """
    with _patched_cv2():
        src = RTSPStreamSource("rtsp://h/s")
        with mock.patch.object(src, "close", wraps=src.close) as closed:
            reader = ThreadedFrameReader(src)
            reader.start()
            time.sleep(0.05)
            reader.stop()

    assert not closed.called, (
        "stop() closed the source. The caller already does that on every path that "
        "exists, so this is a duplicate call -- and closing a source the caller may "
        "still own is not this class's decision to make (M4)"
    )


def test_cycles_return_threads_and_captures_to_baseline() -> None:
    """covers: M5, E4, A8 — true today; this is what keeps it true."""
    with _patched_cv2():
        baseline = threading.active_count()
        for _ in range(5):
            src = RTSPStreamSource("rtsp://h/s")
            reader = ThreadedFrameReader(src)
            reader.start()
            time.sleep(0.02)
            reader.stop()
        time.sleep(0.1)
        after = threading.active_count()
        unreleased = [c.id for c in LIVE if not c.released]

    assert after == baseline, f"threads {baseline} -> {after} after 5 cycles"
    assert unreleased == [], f"captures {unreleased} survived 5 acquire/release cycles"


def test_stop_is_idempotent_and_safe_before_start() -> None:
    """covers: A9, A10, E6 — `__exit__` calls it and so may the caller."""
    with _patched_cv2():
        src = RTSPStreamSource("rtsp://h/s")
        reader = ThreadedFrameReader(src)
        reader.stop()  # never started
        reader.start()
        time.sleep(0.02)
        reader.stop()
        reader.stop()  # twice


def test_stop_joins_before_returning() -> None:
    """covers: A11 — whatever the caller does next happens after the thread is done.

    The caller closes the source immediately after `stop()` returns. If `stop()` returned
    while the thread were still inside a read, that close would land under it -- the
    use-after-release this node exists to end, just relocated by one stack frame.
    """
    order: list[str] = []

    class SlowSource:
        is_live = True

        def __iter__(self):
            try:
                while True:
                    time.sleep(0.01)
                    yield mock.Mock()
            finally:
                order.append("generator-closed")

        def close(self) -> None:
            order.append("close")

    src = SlowSource()
    reader = ThreadedFrameReader(src)
    reader.start()
    time.sleep(0.05)
    reader.stop()
    src.close()  # what the caller does next

    assert order and order[-1] == "close", (
        f"the caller's close did not land last: {order}. stop() must not return while "
        "the thread is still running (A11)"
    )


def test_a_join_that_times_out_is_reported() -> None:
    """covers: A12, E7 — a leaked thread is invisible until the process runs out."""

    class WedgedSource:
        is_live = True

        def __iter__(self):
            while True:
                time.sleep(0.01)
                yield mock.Mock()

        def close(self) -> None:
            return None

    reader = ThreadedFrameReader(WedgedSource())
    reader.start()
    time.sleep(0.03)
    with (
        mock.patch("yowo.io._reader.logger") as log,
        mock.patch.object(reader._thread, "join", return_value=None),
        mock.patch.object(reader._thread, "is_alive", return_value=True),
    ):
        reader.stop()
    warned = [c for c in log.method_calls if "join" in str(c).lower() or "thread" in str(c).lower()]
    assert warned, (
        "stop() returned silently while the thread was still alive. A leaked reader "
        "thread is invisible until the process runs out of them (A12)"
    )


def test_a_source_without_reconnect_is_still_skipped() -> None:
    """covers: M6, E8 — the existing duck-typed contract is unchanged."""

    class Plain:
        is_live = True

        def __iter__(self):
            yield mock.Mock()

    reader = ThreadedFrameReader(Plain())
    reader._last_reconnect = 0.0
    reader._maybe_reconnect()  # must not raise


def test_the_outage_clock_and_backoff_are_untouched() -> None:
    """covers: M7, E9 — the neighbouring node's surfaces, pinned from this side."""
    import inspect

    src = inspect.getsource(RTSPStreamSource.__iter__)
    assert "min(2**retry_count, 10)" in src, "the backoff cap changed (M7)"
    assert "outage_started" in src, "the outage clock was removed (M7)"
