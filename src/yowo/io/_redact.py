"""Strip credentials from a source URL before anything else sees it.

Redaction happens where the URL is STORED, not at each place it is emitted.
An RTSP password reached four surfaces before this existed — an exception
message, a reconnect-timeout message, ``Frame.source_id``, and the
feature-cache key derived from it. Fixing them one at a time would leave the
next surface to leak again, and the cache key is the one that matters most:
a password living as a dict key for the process lifetime survives log
scrubbing entirely.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

__all__ = ["redact_url"]


def redact_url(url: str) -> str:
    """Return ``url`` with any userinfo component removed.

    The ENTIRE userinfo goes, username included: a bare username is
    identifying and is half a credential.

    Host, port and path are preserved so an operator can still tell which
    camera failed — "Cannot open RTSP stream" with no identifier is
    unactionable across an estate of cameras.

    A URL carrying no credential is returned unchanged, byte for byte, so
    existing ``source_id`` values for files, directories and webcams are
    untouched.

    A URL that cannot be parsed yields ``"<unparseable url>"`` rather than
    falling back to the raw string. The parse-failure path is exactly where
    a leak would hide, so it emits nothing of the URL at all.
    """
    try:
        parts = urlsplit(url)
        # Touching .hostname/.port is what raises on a malformed authority,
        # so it must happen inside the guard.
        host, port = parts.hostname, parts.port
    except ValueError:
        return "<unparseable url>"

    if "@" not in parts.netloc:
        return url

    if host is None:
        # An authority we cannot decompose: emit nothing of it.
        return "<unparseable url>"

    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
