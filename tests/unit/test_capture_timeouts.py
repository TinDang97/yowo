"""How long yowo waits on a wedged camera is a number yowo states.

Red-first for ADD task `capture-timeouts`.

Measured 2026-09-11, twice — against an unroutable host (TEST-NET-1) and against a
server that completes the TCP handshake and then never speaks:

    cv2.VideoCapture(url, cv2.CAP_FFMPEG)                       -> not opened after 30.08s
    cv2.VideoCapture(url, cv2.CAP_FFMPEG, [OPEN/READ = 2000ms]) -> not opened after  2.02s

So the stall is NOT unbounded; FFmpeg supplies a ~30s default. The defect is that 30s
is FFmpeg's choice and no caller of yowo can change it.

The real-socket checks below carry the GUARANTEE: only a live FFmpeg can show the
property is honoured. The mocked checks carry the WIRING. Which is which is stated on
each, so the second kind is never mistaken for the first (R:MOCKONLY).
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import cv2
import pytest

from yowo.errors import SourceError
from yowo.io._source import RTSPStreamSource, open_source

# Short enough that the suite does not pay 30s, long enough that a loaded CI box does
# not trip it before FFmpeg has a chance to act.
BOUND_MS = 2000
# The measured unconfigured figure, transcribed rather than re-measured: asserting it
# would put a 30-second wait in the unit suite for a number that is not ours anyway.
_UNCONFIGURED_SECONDS = 30.08


@contextmanager
def _silent_server() -> Iterator[str]:
    """A listener that completes the handshake and then never speaks.

    This is the case a port check calls healthy and a connect timeout does not catch:
    the TCP connection succeeds, so only a READ bound ends the wait.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held: list[socket.socket] = []

    def accept_forever() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            held.append(conn)

    threading.Thread(target=accept_forever, daemon=True).start()
    try:
        yield f"rtsp://127.0.0.1:{srv.getsockname()[1]}/stream"
    finally:
        for conn in held:
            conn.close()
        srv.close()


def _captured_params(calls: list) -> list:
    """The params list from a patched `cv2.VideoCapture` call, or None if bare."""
    out = []
    for call in calls:
        args = call.args
        out.append(args[2] if len(args) >= 3 else None)
    return out


# ---------------------------------------------------------------------------
# M3 — the guarantee, against a real socket
# ---------------------------------------------------------------------------


def test_a_wedged_host_raises_within_the_stated_bound() -> None:
    """covers: M3, E1 — REAL socket layer, not a mock.

    TEST-NET-1 is reserved and unroutable by definition, so this cannot depend on
    whatever happens to be at some address.
    """
    src = RTSPStreamSource(
        "rtsp://192.0.2.1:554/stream", open_timeout_ms=BOUND_MS, read_timeout_ms=BOUND_MS
    )
    started = time.monotonic()
    with pytest.raises(SourceError):
        next(iter(src))
    elapsed = time.monotonic() - started
    assert elapsed < BOUND_MS / 1000 * 4, (
        f"an unroutable host took {elapsed:.2f}s to fail with a {BOUND_MS}ms bound. "
        f"Unconfigured this measured {_UNCONFIGURED_SECONDS}s — if the bound is not "
        "being honoured, the parameter is decorative (M3)"
    )


def test_a_silent_server_raises_within_the_stated_bound() -> None:
    """covers: M3, E2 — REAL socket; the handshake succeeds and nothing follows.

    A connectivity check calls this camera healthy. Only a read bound ends the wait.
    """
    with _silent_server() as url:
        src = RTSPStreamSource(url, open_timeout_ms=BOUND_MS, read_timeout_ms=BOUND_MS)
        started = time.monotonic()
        with pytest.raises(SourceError):
            next(iter(src))
        elapsed = time.monotonic() - started
    assert elapsed < BOUND_MS / 1000 * 4, (
        f"a server that accepts and stays silent took {elapsed:.2f}s with a {BOUND_MS}ms "
        f"bound (unconfigured: {_UNCONFIGURED_SECONDS}s). The TCP connection succeeds "
        "here, so this is the case the open timeout alone does not cover (M3)"
    )


def test_the_raise_names_the_source_and_the_bound() -> None:
    """covers: A6 — an operator at 3am must be able to act on the message.

    A stall with no output is indistinguishable from a hung process, so they kill it and
    lose the diagnosis. The message must say which camera and what bound was exceeded.
    """
    with _silent_server() as url:
        src = RTSPStreamSource(url, open_timeout_ms=BOUND_MS, read_timeout_ms=BOUND_MS)
        with pytest.raises(SourceError) as excinfo:
            next(iter(src))
    message = str(excinfo.value)
    assert "127.0.0.1" in message, f"the message does not name the source: {message!r}"
    assert str(BOUND_MS) in message or "2" in message, (
        f"the message does not name the bound that was exceeded: {message!r}"
    )


# ---------------------------------------------------------------------------
# M1, M4 — every network capture is built the same way (wiring; mocked)
# ---------------------------------------------------------------------------


def test_every_network_capture_is_built_with_both_timeouts() -> None:
    """covers: M1, M4, A14, A15 — the clause m2's box is reaching for.

    WIRING check, behavioural rather than source-text: it observes what the capture was
    actually constructed with, so a correctly-spelled call on an unreachable line cannot
    satisfy it (method M9).
    """
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap_cls.return_value.isOpened.return_value = True
        cap_cls.return_value.read.return_value = (False, None)
        src = RTSPStreamSource(
            "rtsp://h/s", open_timeout_ms=BOUND_MS, read_timeout_ms=BOUND_MS, max_frames=0
        )
        list(src)

    assert cap_cls.call_args_list, "no capture was constructed at all"
    for params in _captured_params(cap_cls.call_args_list):
        assert params is not None, (
            "a network capture was constructed with no params list. Unconfigured, FFmpeg "
            f"waits {_UNCONFIGURED_SECONDS}s; configured it waits what you asked for. "
            "Build it through the one path that passes both timeouts (M4)"
        )
        assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params, f"no open timeout in {params!r}"
        assert cv2.CAP_PROP_READ_TIMEOUT_MSEC in params, f"no read timeout in {params!r}"


def test_reconnect_builds_its_capture_the_same_way() -> None:
    """covers: M1, E3, R:PARTIAL — the site most likely to be missed.

    `reconnect()` constructs a capture twice and neither construction is the obvious one.
    A bound applied only to `_open_cap` leaves the periodic reconnect unbounded.
    """
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap_cls.return_value.isOpened.return_value = True
        src = RTSPStreamSource("rtsp://h/s", open_timeout_ms=BOUND_MS, read_timeout_ms=BOUND_MS)
        src._active_cap = cap_cls.return_value
        cap_cls.reset_mock()
        src.reconnect()

    assert cap_cls.call_args_list, "reconnect() constructed no capture"
    for params in _captured_params(cap_cls.call_args_list):
        assert params is not None and cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params, (
            f"reconnect() built a bare capture ({params!r}). Every construction site or "
            "none — the one that is missed is the one that wedges (R:PARTIAL)"
        )


# ---------------------------------------------------------------------------
# M2, M5, A4, A8, A9 — whose number it is
# ---------------------------------------------------------------------------


def test_the_timeouts_reach_the_capture_from_open_source() -> None:
    """covers: M2, R:HARDCODE — carried from the factory, not decided inside."""
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap_cls.return_value.isOpened.return_value = True
        cap_cls.return_value.read.return_value = (False, None)
        src = open_source("rtsp://h/s", open_timeout_ms=1234, read_timeout_ms=5678, max_frames=0)
        list(src)

    params = _captured_params(cap_cls.call_args_list)[0]
    assert params is not None and 1234 in params and 5678 in params, (
        f"the caller's numbers did not reach the capture ({params!r}). A timeout the "
        "caller cannot set repeats the defect one layer up (R:HARDCODE)"
    )


def test_an_unconfigured_source_is_unchanged() -> None:
    """covers: M5, A2, E4, R:SILENTSHORTEN — upgrading must not shorten anyone's wait.

    Unconfigured means NO property is set, which leaves FFmpeg's own default in place —
    byte-for-byte today's behaviour rather than merely the same number.
    """
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap_cls.return_value.isOpened.return_value = True
        cap_cls.return_value.read.return_value = (False, None)
        src = RTSPStreamSource("rtsp://h/s", max_frames=0)
        list(src)

    for params in _captured_params(cap_cls.call_args_list):
        assert params is None, (
            f"an unconfigured source set capture properties ({params!r}). Every existing "
            "deployment's tolerance for a camera blip would change on upgrade, and that "
            "is a deployment decision, not a bugfix (R:SILENTSHORTEN)"
        )


def test_none_leaves_the_backend_default_in_place() -> None:
    """covers: A4 — None and 0 mean different things.

    None leaves FFmpeg's default; 0 means no timeout to FFmpeg. Conflating them turns
    "never time out" into "wait 30 seconds", or the reverse.
    """
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap_cls.return_value.isOpened.return_value = True
        cap_cls.return_value.read.return_value = (False, None)
        list(
            RTSPStreamSource("rtsp://h/s", open_timeout_ms=None, read_timeout_ms=None, max_frames=0)
        )
        assert all(p is None for p in _captured_params(cap_cls.call_args_list)), (
            "None set a property; it must leave the backend default untouched"
        )

        cap_cls.reset_mock()
        list(RTSPStreamSource("rtsp://h/s", open_timeout_ms=0, read_timeout_ms=0, max_frames=0))
        params = _captured_params(cap_cls.call_args_list)[0]
        assert params is not None and 0 in params, (
            f"0 was treated as absent ({params!r}); it is a value FFmpeg understands and "
            "is not the same request as None (A4)"
        )


@pytest.mark.parametrize("source", ["/var/media/clip.mp4", "/var/media/", "0", "photo.jpg"])
def test_a_non_network_source_does_not_silently_ignore_them(source: str) -> None:
    """covers: A8, E5 — a parameter that does nothing is worse than one that is absent.

    `cv2.VideoCapture(path)` and `cv2.VideoCapture(index)` use a non-FFMPEG backend where
    these properties are not honoured. Accepting them there tells a caller they are
    protected when they are not.
    """
    with pytest.raises(SourceError) as excinfo:
        open_source(source, open_timeout_ms=BOUND_MS)
    message = str(excinfo.value)
    # The refusal must NAME the parameter. Matching on "rtsp" or "network" looked right
    # and proved nothing: `open_source`'s generic fallback lists 'RTSP URLs (rtsp://)'
    # among the supported forms, so `/var/media/` passed this check with the refusal
    # deleted -- a check passing under the exact mutation it exists to catch (Q4).
    assert "open_timeout_ms" in message or "read_timeout_ms" in message, (
        f"the refusal does not name the parameter ({message!r}); a caller needs to know "
        "it is unsupported here rather than mistyped, and no other error in this factory "
        "mentions it (A8)"
    )


@pytest.mark.parametrize("bad", [-1, -5000])
def test_a_negative_timeout_is_rejected_at_the_factory(bad: int) -> None:
    """covers: A9, E6 — a typo must surface as an error, not as a hang."""
    with pytest.raises((SourceError, ValueError)):
        open_source("rtsp://h/s", open_timeout_ms=bad)


# ---------------------------------------------------------------------------
# M6 — what this node does NOT touch
# ---------------------------------------------------------------------------


def test_retry_and_backoff_are_untouched() -> None:
    """covers: M6, E7, R:SCOPECREEP — the reconnect deadline stays exactly as wrong.

    `now + wait > deadline` cannot fire while `wait = min(2**n, 10)` and the default
    `reconnect_timeout_s` is 30.0, and `deadline` is reset on every disconnect. That is a
    real defect and `rtsp-reconnect-correctness` owns it. Bounding the CALL is this node;
    deciding how many times to make it is not.
    """
    import inspect

    src = inspect.getsource(RTSPStreamSource.__iter__)
    assert "2**retry_count" in src, (
        "the backoff expression changed. This node must not touch retry semantics "
        "(R:SCOPECREEP) — if another node changed it, re-derive this check"
    )
    assert "deadline" in src, "the reconnect deadline was removed by this node's scope"
    assert RTSPStreamSource("rtsp://h/s")._reconnect_timeout_s == 30.0, (
        "the reconnect timeout default moved; that belongs to rtsp-reconnect-correctness"
    )
