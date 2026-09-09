"""A camera password cannot be recovered from anything yowo emits.

Red-first for ADD task `rtsp-credential-redaction`.

Every assertion here uses a real credentialed URL, as the m1 exit box requires.
All four surfaces leak `hunter2` verbatim today:

    _source.py:312          raise SourceError(f"... {self._url}")
    _source.py:350          reconnect timeout message
    _source.py:336          source_id=self._url  -> Frame -> result JSON
    cache/__init__.py:102   source_id IS the feature-cache key
"""

from __future__ import annotations

from unittest import mock

import pytest

from yowo.errors import SourceError
from yowo.io._source import RTSPStreamSource

USER = "admin"
PASSWORD = "hunter2"
HOST = "10.0.0.5"
PORT = "554"
PATH = "/Streaming/Channels/101"
URL = f"rtsp://{USER}:{PASSWORD}@{HOST}:{PORT}{PATH}"


def _no_credential(text: str) -> None:
    assert PASSWORD not in text, f"password leaked: {text!r}"
    assert USER not in text, f"username leaked (half a credential): {text!r}"


@pytest.fixture
def source() -> RTSPStreamSource:
    return RTSPStreamSource(URL)


def test_source_id_carries_no_credential(source: RTSPStreamSource) -> None:
    """covers: M1, A2 — source_id reaches result JSON and every downstream sink."""
    _no_credential(source.safe_url)


def test_open_failure_exception_carries_no_credential(source: RTSPStreamSource) -> None:
    """covers: M1, A8 — the exception travels to every caller and crash reporter."""
    with mock.patch("cv2.VideoCapture") as cap:
        cap.return_value.isOpened.return_value = False
        with pytest.raises(SourceError) as excinfo:
            source._open_cap()
    _no_credential(str(excinfo.value))


def test_reconnect_timeout_message_carries_no_credential(source: RTSPStreamSource) -> None:
    """covers: M1, A8 — a second message added later must not need its own fix."""
    import inspect

    body = inspect.getsource(type(source))
    assert "{self._url}" not in body, (
        "an emitted message interpolates the raw URL; redaction must happen where it is stored (M2)"
    )


def test_cache_key_carries_no_credential(source: RTSPStreamSource) -> None:
    """covers: M1, A14 — a dict key outlives log scrubbing entirely."""
    _no_credential(source.safe_url)


def test_redacted_url_still_identifies_the_stream(source: RTSPStreamSource) -> None:
    """covers: M3, A6 — 'Cannot open RTSP stream' with no id is unactionable."""
    safe = source.safe_url
    assert HOST in safe and PORT in safe and PATH in safe


def test_connectable_url_is_retained_privately(source: RTSPStreamSource) -> None:
    """covers: A9 — redacting the URL used to connect breaks every stream."""
    assert source._url == URL, "the connectable URL must be retained for VideoCapture"


def test_clean_url_is_returned_unchanged() -> None:
    """covers: A4 — no gratuitous rewriting of URLs that carry no credential."""
    clean = f"rtsp://{HOST}:{PORT}{PATH}"
    assert RTSPStreamSource(clean).safe_url == clean


def test_username_without_password_is_still_removed() -> None:
    """covers: E2 — a bare username is identifying and is half a credential."""
    src = RTSPStreamSource(f"rtsp://{USER}@{HOST}:{PORT}{PATH}")
    _no_credential(src.safe_url)


def test_malformed_url_emits_nothing_of_it() -> None:
    """covers: E1, A10 — the parse-failure path is where a leak would hide."""
    from yowo.io import redact_url

    malformed = f"rtsp://{USER}:{PASSWORD}@[not-a-valid-host:::/{PATH}"
    _no_credential(redact_url(malformed))


def test_non_rtsp_scheme_with_userinfo_is_redacted() -> None:
    """covers: A3 — open_source accepts HTTP(S) streams by the same path."""
    from yowo.io import redact_url

    _no_credential(redact_url(f"https://{USER}:{PASSWORD}@{HOST}/stream.m3u8"))
