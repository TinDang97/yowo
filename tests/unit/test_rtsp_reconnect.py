"""A brief outage is survived; a permanent one ends at the bound the caller set.

Red-first for ADD task `rtsp-reconnect-correctness`.

Two opposite failure modes coexisted in one loop, both measured 2026-09-11:

    the bound is INERT      `wait = min(2**n, 10)` and the guard reduced to
                            `wait > reconnect_timeout_s`, so nothing at or above 10
                            could ever fire it. 30 and 60 were the same number.

    a blip KILLED the stream  the retry branch called `_open_cap()`, which RAISES when
                            the camera is momentarily away, and that propagated straight
                            out of the generator. Two frames, one failed reopen, dead.

So it retried forever in the case that did not need it and died instantly in the case it
existed for.

Every timing check here drives a FAKE clock. The subject is a 30-second bound, and a
check that takes 30 seconds to assert it is a check that gets deselected — a skipped
test is green (Q3).
"""

from __future__ import annotations

import inspect
from typing import Iterator
from unittest import mock

import numpy as np
import pytest

from yowo.errors import SourceError, SourceTimeoutError
from yowo.io import _source as source_mod
from yowo.io._source import RTSPStreamSource


class RetriedForever(BaseException):
    """Raised by the fake clock when the loop exceeds its sleep budget.

    A check whose subject is "this terminates" must itself terminate. Without a budget
    an infinite retry does not FAIL the test -- it hangs it, and a hung suite is worse
    than a red one because nobody reads a timeout.
    """


class FakeClock:
    """`monotonic` and `sleep` that move only when the code under test sleeps."""

    def __init__(self, budget: int = 200) -> None:
        self.now = 1000.0
        self.slept: list[float] = []
        self.budget = budget
        self.on_sleep = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if len(self.slept) >= self.budget:
            raise RetriedForever(f"still retrying after {len(self.slept)} backoffs")
        self.slept.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(len(self.slept))


class FakeCap:
    """A capture whose reads and opens follow a script."""

    def __init__(self, url: str, api: object = None, params: object = None) -> None:
        self.opened = _STATE["can_open"]
        _STATE["opens"] += 1

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object]:
        if _STATE["good_reads"] > 0:
            _STATE["good_reads"] -= 1
            return True, np.zeros((2, 2, 3), np.uint8)
        return False, None

    def release(self) -> None:
        return None


_STATE: dict[str, object] = {"can_open": True, "good_reads": 0, "opens": 0}


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    _STATE.update({"can_open": True, "good_reads": 0, "opens": 0})
    yield


def _harness(clock: FakeClock):
    """Patch cv2 and the module's clock together -- nothing here sleeps for real."""
    return mock.patch.multiple(
        source_mod,
        cv2=mock.MagicMock(VideoCapture=FakeCap),
        time=mock.Mock(monotonic=clock.monotonic, sleep=clock.sleep),
    )


def _drain(src: RTSPStreamSource, limit: int = 500) -> tuple[int, BaseException | None]:
    """Pull frames until the stream ends; return how many, and why it ended."""
    got = 0
    it = iter(src)
    try:
        for _ in range(limit):
            next(it)
            got += 1
    except StopIteration:
        return got, None
    except BaseException as exc:  # the reason the stream ended IS the subject here
        return got, exc
    return got, None


# ---------------------------------------------------------------------------
# M1, M2, M8 — what is survived, what ends, and what the bound decides
# ---------------------------------------------------------------------------


def test_a_brief_outage_is_survived() -> None:
    """covers: M1, E1, R:DIESONBLIP — measured today: two frames, one blip, dead."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 2
        src = RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0, max_frames=3)

        it = iter(src)
        next(it)
        next(it)
        _STATE["can_open"] = False  # camera away for exactly one reopen
        _STATE["good_reads"] = 1

        clock.on_sleep = lambda _n: _STATE.__setitem__("can_open", True)
        third = next(it)
    assert third is not None, "a brief outage must not end the stream (M1)"


def test_a_permanent_outage_terminates_at_the_default() -> None:
    """covers: M2, E2, R:INERT — 30.0 is the documented default and never fired."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        got, exc = _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0))
    assert isinstance(exc, SourceTimeoutError), (
        f"a permanently dead camera ended with {exc!r} after {got} frames. At the "
        "documented default the terminal error was unreachable: the guard reduced to "
        "`wait > 30` and `wait` never exceeds 10 (R:INERT)"
    )
    assert clock.now - 1000.0 <= 30.0 + 10.0, (
        f"the stream outlived its 30.0s bound by {clock.now - 1000.0 - 30.0:.1f}s"
    )


def test_a_longer_bound_tolerates_a_longer_outage() -> None:
    """covers: M2, E6, R:INERT — today 30 and 60 are literally the same number."""
    elapsed: dict[float, float] = {}
    for bound in (30.0, 60.0):
        clock = FakeClock()
        with _harness(clock):
            _STATE["good_reads"] = 1
            _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=bound))
        elapsed[bound] = clock.now - 1000.0
    assert elapsed[60.0] > elapsed[30.0], (
        f"60.0 tolerated {elapsed[60.0]:.1f}s and 30.0 tolerated {elapsed[30.0]:.1f}s. A "
        "bound a caller can set that does not change behaviour is not a bound (R:INERT)"
    )


def test_none_retries_without_bound() -> None:
    """covers: M8, A13, E11 — today's capability, kept, but asked for in words."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        got, exc = _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=None))
    assert isinstance(exc, RetriedForever), (
        f"an explicitly unbounded stream ended with {exc!r}. Exhausting the fake clock's "
        "budget is what unbounded LOOKS like here; a SourceTimeoutError means `None` was "
        "treated as a number (M8)"
    )
    assert clock.now - 1000.0 > 60.0, (
        "the unbounded stream did not outlast any finite bound; it is not unbounded"
    )


def test_a_zero_bound_makes_the_first_failure_terminal() -> None:
    """covers: A4, E5 — 0 is zero tolerance, continuous with 1 and 5; not "unlimited"."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        got, exc = _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=0.0))
    assert isinstance(exc, SourceTimeoutError), f"0.0 did not terminate; got {exc!r}"
    assert got == 1, f"0.0 retried before terminating ({got} frames yielded)"


# ---------------------------------------------------------------------------
# M3, A2, A5, A11 — what the clock measures
# ---------------------------------------------------------------------------


def test_read_failures_and_reopen_failures_share_one_bound() -> None:
    """covers: A2, E3 — from the camera's side a flap is one event, not two kinds."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        # The first open must SUCCEED -- a stream that never started exercises A10, not
        # this. The reopens are what fail, from the first backoff onward.
        clock.on_sleep = lambda _n: _STATE.__setitem__("can_open", False)
        got, exc = _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0))
    assert isinstance(exc, SourceTimeoutError), (
        f"an outage of failed reads AND failed reopens ended with {exc!r}. A failed "
        "reopen must feed the same bound, not end the stream by raising (A2)"
    )


def test_a_successful_read_resets_the_outage_clock() -> None:
    """covers: M3, A3, E4 — a second outage gets the FULL bound, not the first's remainder.

    Measured against a control rather than a threshold. A "> 10s" assertion passed with
    the reset deleted, because the second outage still inherited ~23s of a 30s budget --
    a check surviving the exact change it exists to catch (Q4). "Resets" means the second
    outage lasts exactly as long as a fresh one, so that is what is compared.
    """

    def _permanent_outage_from_fresh() -> float:
        clock = FakeClock()
        with _harness(clock):
            _STATE["good_reads"] = 1
            _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0))
            return clock.now - 1000.0

    fresh = _permanent_outage_from_fresh()

    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1

        def _recover_once(n: int) -> None:
            if n == 3:
                _STATE["good_reads"] = 1
                clock.on_sleep = None

        clock.on_sleep = _recover_once
        it = iter(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0))
        next(it)
        second = next(it)  # the recovery frame
        after_recovery = clock.now
        with pytest.raises(SourceTimeoutError):
            next(it)  # a fresh outage, which must get the FULL budget
        post_recovery = clock.now - after_recovery

    assert second is not None
    assert post_recovery == pytest.approx(fresh, rel=0.05), (
        f"a fresh outage lasts {fresh:.1f}s but the one after a recovery lasted "
        f"{post_recovery:.1f}s. The clock must reset on a successful read, or a "
        "long-lived stream accumulates unrelated outages until any blip is terminal (M3)"
    )


def test_the_bound_wins_when_the_next_sleep_would_cross_it() -> None:
    """covers: A5 — ending late by one backoff interval is still ending late."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=25.0))
    assert clock.now - 1000.0 <= 25.0, (
        f"the stream ran {clock.now - 1000.0:.1f}s against a 25.0s bound. If the next "
        "sleep would cross the deadline the stream ends then, rather than sleeping "
        "past it and noticing afterwards (A5)"
    )


def test_the_retry_counter_resets_on_a_read_not_on_a_reopen() -> None:
    """covers: A11 — a camera that accepts connections and sends nothing must not spin."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        _STATE["can_open"] = True  # reopen always succeeds; reads never do
        _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0))
    assert clock.slept, "no backoff at all"
    assert max(clock.slept) > 1.0, (
        f"backoff never grew past {max(clock.slept)}s while every reopen succeeded and "
        "no read did. Resetting the counter on a successful REOPEN makes a silent "
        "camera loop at full speed forever (A11)"
    )


# ---------------------------------------------------------------------------
# M4, M6, A8, A9, A10 — the edges
# ---------------------------------------------------------------------------


def test_the_terminal_error_names_the_source_the_elapsed_and_the_bound() -> None:
    """covers: M6, A6 — 'timed out' alone cannot separate a dead camera from a low bound."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        _, exc = _drain(RTSPStreamSource("rtsp://user:pw@cam/s", reconnect_timeout_s=30.0))
    message = str(exc)
    assert "pw" not in message, f"the terminal error leaked a credential: {message!r}"
    assert "cam" in message, f"the terminal error does not name the camera: {message!r}"
    assert "30" in message, f"the terminal error does not name the bound: {message!r}"


def test_a_non_source_error_from_the_reopen_propagates() -> None:
    """covers: A8, E8 — an unexpected type is not evidence the camera will return."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 1
        src = RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0)
        it = iter(src)
        next(it)
        with (
            mock.patch.object(
                RTSPStreamSource, "_open_cap", side_effect=MemoryError("not a camera problem")
            ),
            pytest.raises(MemoryError),
        ):
            next(it)


def test_the_first_open_still_raises_without_retry() -> None:
    """covers: A10, E7 — a stream that never started has nothing to reconnect to."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["can_open"] = False
        got, exc = _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0))
    assert isinstance(exc, SourceError) and not isinstance(exc, SourceTimeoutError), (
        f"the first open produced {exc!r}; it must raise immediately, not enter retry (A10)"
    )
    assert clock.slept == [], f"the first open backed off before raising: {clock.slept}"


def test_max_frames_ends_cleanly_rather_than_by_timeout() -> None:
    """covers: A9 — a satisfied quota is not an outage."""
    clock = FakeClock()
    with _harness(clock):
        _STATE["good_reads"] = 5
        got, exc = _drain(RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0, max_frames=3))
    assert exc is None and got == 3, f"max_frames ended with {exc!r} after {got} frames (A9)"


def test_the_backoff_cap_is_unchanged() -> None:
    """covers: M4, E9, R:SPIN — termination moves; the rate limit does not."""
    src = inspect.getsource(RTSPStreamSource.__iter__)
    assert "min(2**retry_count, 10)" in src, (
        "the backoff cap changed. Fixing termination must not turn the loop into a "
        "spin — the cap stays, it simply stops deciding termination (R:SPIN)"
    )


def test_a_recovery_is_observable() -> None:
    """covers: A12 — a silent recovery is indistinguishable from a stall of the same length."""
    clock = FakeClock()
    with _harness(clock), mock.patch.object(source_mod, "logger") as log:
        _STATE["good_reads"] = 1
        # One good frame, then a real outage, then the camera returns. Two consecutive
        # good reads would recover from nothing and prove nothing.
        clock.on_sleep = lambda n: _STATE.__setitem__("good_reads", 1) if n == 2 else None
        src = RTSPStreamSource("rtsp://h/s", reconnect_timeout_s=30.0, max_frames=2)
        _drain(src)
    assert log.method_calls, (
        "a recovery emitted nothing. An operator cannot tell a stream that recovered "
        "from one that stalled for the same duration (A12)"
    )


# ---------------------------------------------------------------------------
# M5, M7 — the guards on this node itself
# ---------------------------------------------------------------------------


def test_no_check_here_sleeps() -> None:
    """covers: M5, R:REALSLEEP — the guard against this becoming a suite nobody runs.

    The subject is a 30-second bound. Asserting it with real seconds makes the check slow,
    then marked slow, then deselected, and a skipped test is green (Q3).
    """
    text = __import__("pathlib").Path(__file__).read_text()
    # Built rather than written literally: spelling the needle in full would put it in
    # this file and the check would find ITSELF -- which is exactly what happened on the
    # first draft, a check failing on its own source and proving nothing about the rest.
    needle = "time." + "sleep("
    offenders = [
        line.strip()
        for line in text.splitlines()
        if needle in line and "needle" not in line and not line.strip().startswith("#")
    ]
    assert not offenders, f"a check here sleeps for real: {offenders} (R:REALSLEEP)"
    assert "FakeClock" in text, "the fake clock is how the bound is asserted (M5)"


def test_the_reconnect_orphan_is_untouched() -> None:
    """covers: M7, E10, R:ORPHANFIX — `reader-shutdown` owns it; do not quietly absorb it.

    Measured: `__iter__` binds `cap` as a local while `reconnect()` rebinds
    `self._active_cap`, so every periodic reconnect orphans one capture. Real, and not
    this node's box.
    """
    src = inspect.getsource(RTSPStreamSource.reconnect)
    assert "self._active_cap = new_cap" in src, (
        "reconnect()'s rebind changed. That orphan is reader-shutdown's box — fixing it "
        "here would take another node's guarantee without its checks (R:ORPHANFIX)"
    )
