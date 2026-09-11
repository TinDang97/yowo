"""`open_source` refuses a credentialed URL without echoing it.

Red-first for ADD task `source-factory-redaction`.

`rtsp-credential-redaction` closed the leak inside `RTSPStreamSource`, which
redacts where the URL is STORED. `open_source` is upstream of that boundary —
it raises before any source object exists — so three of its `SourceError`
messages interpolate the caller's raw string. Reproduced 2026-09-10::

    Video file not found: https:/admin:hunter2@10.0.0.5/stream.mp4
    Image file not found: https:/admin:hunter2@10.0.0.5/frame.jpg
    Cannot determine source type for: 'http://admin:hunter2@10.0.0.5/video'.

Note `https:/` with ONE slash: the value was laundered through `Path()` before
anyone looked at it (R:MANGLE).

Every check in this file calls `open_source` and asserts on what it raises
(M5). The check that shipped for this ground last time,
`test_non_rtsp_scheme_with_userinfo_is_redacted`, calls `redact_url()` and
never reaches the factory — that is R:PROXY, and
`test_no_check_here_asserts_on_redact_url_alone` exists to stop it recurring.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from yowo.errors import SourceError, YowoError
from yowo.io._redact import redact_url
from yowo.io._source import (
    FrameSource,
    ImageFileSource,
    RTSPStreamSource,
    WebcamSource,
    open_source,
)

USER = "admin"
PASSWORD = "hunter2"
HOST = "10.0.0.5"
PORT = "8443"

VIDEO_URL = f"https://{USER}:{PASSWORD}@{HOST}/stream.mp4"
IMAGE_URL = f"https://{USER}:{PASSWORD}@{HOST}/frame.jpg"
UNKNOWN_URL = f"http://{USER}:{PASSWORD}@{HOST}/video"
PLAYLIST_URL = f"https://{USER}:{PASSWORD}@{HOST}/stream.m3u8"
USERNAME_ONLY_URL = f"http://{USER}@{HOST}/frame.jpg"
PORTED_URL = f"https://{USER}:{PASSWORD}@{HOST}:{PORT}/stream.mp4"
UNPARSEABLE_URL = f"https://{USER}:{PASSWORD}@[::1/stream.mp4"
RTSP_URL = f"rtsp://{USER}:{PASSWORD}@{HOST}:554/Streaming/Channels/101"

# Every input below reaches a `raise` inside `open_source`.
LEAKING_SOURCES = (
    VIDEO_URL,
    IMAGE_URL,
    UNKNOWN_URL,
    PLAYLIST_URL,
    USERNAME_ONLY_URL,
    PORTED_URL,
    UNPARSEABLE_URL,
)


def _no_credential(text: str) -> None:
    assert PASSWORD not in text, f"password leaked: {text!r}"
    assert USER not in text, f"username leaked (half a credential): {text!r}"


def _not_mangled(text: str) -> None:
    """A message built from a `Path()`-laundered value reads `https:/` (R:MANGLE)."""
    assert re.search(r"https?:/(?!/)", text) is None, (
        f"message built from a mangled value: {text!r}"
    )


def _raised_by_open_source(source: str) -> SourceError:
    with pytest.raises(SourceError) as excinfo:
        open_source(source)
    return excinfo.value


def test_the_video_branch_hides_userinfo() -> None:
    """covers: M1, A1, E1 — a video extension that does not exist."""
    message = str(_raised_by_open_source(VIDEO_URL))
    _no_credential(message)
    _not_mangled(message)
    assert f"https://{HOST}/stream.mp4" in message


def test_the_image_branch_hides_userinfo() -> None:
    """covers: M1, E3 — the image branch leaks by the same mechanism."""
    message = str(_raised_by_open_source(IMAGE_URL))
    _no_credential(message)
    _not_mangled(message)
    assert f"https://{HOST}/frame.jpg" in message


def test_the_unknown_type_branch_hides_userinfo() -> None:
    """covers: M1, E2 — the `{source!r}` branch; quoting does not redact."""
    for url, redacted in (
        (UNKNOWN_URL, f"http://{HOST}/video"),
        (PLAYLIST_URL, f"https://{HOST}/stream.m3u8"),
    ):
        message = str(_raised_by_open_source(url))
        _no_credential(message)
        _not_mangled(message)
        assert redacted in message


def test_a_username_without_a_password_is_still_redacted() -> None:
    """covers: R:LEAK, E6 — userinfo is userinfo whether or not a password follows."""
    message = str(_raised_by_open_source(USERNAME_ONLY_URL))
    _no_credential(message)
    assert f"http://{HOST}/frame.jpg" in message


def test_every_raise_in_open_source_reads_the_redacted_form() -> None:
    """covers: M2, A2, A5, R:MANGLE — one redaction at entry, inherited by every branch."""
    # Behaviour: every input that reaches a raise comes back redacted and unmangled.
    for url in LEAKING_SOURCES:
        message = str(_raised_by_open_source(url))
        _no_credential(message)
        _not_mangled(message)

    # Structure: the boundary, not three known lines. A branch added later must
    # inherit the redaction rather than have to remember it.
    func = ast.parse(textwrap.dedent(inspect.getsource(open_source))).body[0]
    assert isinstance(func, ast.FunctionDef)

    def _calls(node: ast.AST, name: str) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        ]

    redactions = _calls(func, "redact_url")
    assert len(redactions) == 1, (
        f"redaction must run once, at the boundary; found {len(redactions)}"
    )

    at = [i for i, stmt in enumerate(func.body) if _calls(stmt, "redact_url")]
    assert len(at) == 1
    entry = at[0]

    later = [
        i
        for i, stmt in enumerate(func.body)
        if any(isinstance(n, (ast.Raise, ast.If)) for n in ast.walk(stmt)) or _calls(stmt, "Path")
    ]
    assert later and min(later) > entry, (
        "the redacted form must be computed before any branch or Path() conversion (A5)"
    )

    assignment = func.body[entry]
    assert isinstance(assignment, ast.Assign)
    target = assignment.targets[0]
    assert isinstance(target, ast.Name)
    safe_name = target.id

    raises = [n for n in ast.walk(func) if isinstance(n, ast.Raise) and n.exc is not None]
    assert len(raises) >= 3
    for node in raises:
        assert node.exc is not None
        names = {n.id for n in ast.walk(node.exc) if isinstance(n, ast.Name)}
        assert safe_name in names, f"a raise does not read the redacted form: {ast.dump(node)}"
        leaky = names & {"source", "source_str", "path"}
        assert not leaky, f"a raise interpolates the raw value {leaky}: {ast.dump(node)}"


def test_a_redacted_message_still_names_host_and_path() -> None:
    """covers: M3, A6 — an operator can tell a mistyped host from a bad credential."""
    message = str(_raised_by_open_source(PORTED_URL))
    _no_credential(message)
    _not_mangled(message)
    assert f"https://{HOST}:{PORT}/stream.mp4" in message, (
        "scheme, host, port and path must survive redaction"
    )


def test_a_plain_path_and_a_webcam_index_are_reported_verbatim(tmp_path: Path) -> None:
    """covers: A3, E5 — a value with no userinfo comes back unchanged."""
    # A relative path is the probe: `Path()` rewrites "./clip.mp4" to "clip.mp4",
    # so today's message does not name the caller's value in full.
    for plain in ("./missing-clip.mp4", "./cam@home.mp4"):
        message = str(_raised_by_open_source(plain))
        assert plain in message, f"a plain path must be reported in full: {message!r}"

    absolute = str(tmp_path / "missing.mp4")
    assert absolute in str(_raised_by_open_source(absolute))

    missing_image = str(tmp_path / "missing.jpg")
    assert missing_image in str(_raised_by_open_source(missing_image))

    unknown = str(tmp_path / "missing.xyz")
    assert unknown in str(_raised_by_open_source(unknown))

    # A webcam index carries no userinfo either: it must still dispatch, verbatim.
    # cv2 is patched only so the device probe succeeds off a machine with a
    # camera — the factory itself is the real one.
    with mock.patch("cv2.VideoCapture") as capture:
        capture.return_value.isOpened.return_value = True
        webcam = open_source("0")
    assert isinstance(webcam, WebcamSource)
    assert webcam._device_index == 0

    with mock.patch("cv2.VideoCapture") as capture:
        capture.return_value.isOpened.return_value = False
        with pytest.raises(SourceError) as excinfo:
            open_source("0")
    assert "0" in str(excinfo.value)


def test_an_unparseable_source_uses_the_sentinel() -> None:
    """covers: A4 — a parse failure yields `<unparseable url>`, never the raw string."""
    message = str(_raised_by_open_source(UNPARSEABLE_URL))
    _no_credential(message)
    _not_mangled(message)
    assert "<unparseable url>" in message
    assert HOST not in message


def test_a_credentialed_rtsp_url_still_opens_as_before() -> None:
    """covers: E4 — the path `rtsp-credential-redaction` already covers must not regress."""
    source = open_source(RTSP_URL)
    assert isinstance(source, RTSPStreamSource)
    assert source._url == RTSP_URL, "the connectable URL must reach the source unmodified"
    _no_credential(source.safe_url)
    assert source.safe_url == f"rtsp://{HOST}:554/Streaming/Channels/101"


def test_open_source_signature_and_raised_type_are_unchanged(tmp_path: Path) -> None:
    """covers: M4 — the source object still gets the original; only what is emitted changes."""
    signature = inspect.signature(open_source)
    parameters = list(signature.parameters.values())
    assert [p.name for p in parameters] == [
        "source",
        "loop",
        "frame_skip",
        "max_frames",
        "reconnect_timeout_s",
        # Added by `capture-timeouts`, by human decision: keyword-only with a `None`
        # default, so every existing call is unaffected and the backend's own ~30s
        # default stays in force unless a caller states otherwise.
        "open_timeout_ms",
        "read_timeout_ms",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[0].default is inspect.Parameter.empty
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters[1:])
    assert [p.default for p in parameters[1:]] == [False, 0, None, 30.0, None, None]

    # The raised type is unchanged.
    error = _raised_by_open_source(VIDEO_URL)
    assert type(error) is SourceError
    assert isinstance(error, YowoError)

    # ... and only what is EMITTED is redacted.
    _no_credential(str(error))

    # The source object receives the caller's original value.
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"")
    opened = open_source(str(image))
    assert isinstance(opened, ImageFileSource)
    assert isinstance(opened, FrameSource)
    assert opened._path == Path(str(image))

    rtsp = open_source(RTSP_URL)
    assert isinstance(rtsp, RTSPStreamSource)
    assert rtsp._url == RTSP_URL


def test_no_check_here_asserts_on_redact_url_alone() -> None:
    """covers: M5, R:PROXY — the shape that let this leak through a green suite."""
    # The helper is already clean, and was before this task existed. A check that
    # asserts on `redact_url` alone therefore passes while the factory leaks —
    # which is exactly what shipped for `rtsp-credential-redaction`'s A3.
    _no_credential(redact_url(VIDEO_URL))
    _no_credential(str(_raised_by_open_source(VIDEO_URL)))

    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    checks = [
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    assert len(checks) == 11, f"CHECKS names eleven checks; found {len(checks)}"
    for check in checks:
        names = {n.id for n in ast.walk(check) if isinstance(n, ast.Name)}
        assert "open_source" in names or "_raised_by_open_source" in names, (
            f"{check.name} never reaches the factory — that is R:PROXY"
        )
