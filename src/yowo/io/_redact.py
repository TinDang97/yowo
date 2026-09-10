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

import hashlib
from urllib.parse import urlsplit, urlunsplit

__all__ = ["redact_url"]

# Eight hex characters of SHA-256. This is an IDENTITY control, not a security one,
# and the distinction matters. It exists so that two cameras differing only in their
# query stay two dict keys -- `DetectionRouter.register` overwrites silently, so a
# collapse delivers one camera's detections to the other camera's callback.
#
# What it does NOT do: a digest over a LOW-entropy query such as `?channel=2` is
# effectively an encoding of it, because the space is small enough to enumerate. That
# is acceptable, because a channel number is not a secret. A high-entropy token is not
# recoverable from eight hex characters, and that is the case this exists for.
_DIGEST_CHARS = 8


def _identity_digest(query: str, fragment: str) -> str:
    """A stable, one-way marker distinguishing two URLs that redact to the same text.

    Computed over the ORIGINAL query and fragment, before redaction. A digest of the
    REDACTED form would be identical for every URL sharing the same query keys and
    would therefore distinguish nothing at all.
    """
    material = f"{query}\x00{fragment}".encode()
    return f"q{hashlib.sha256(material).hexdigest()[:_DIGEST_CHARS]}"


def _redact_query(query: str) -> str:
    """Empty every value, keep every key, keep the original order.

    Every value, whatever the key is called: a denylist of credential-shaped names
    fails open on the key nobody listed, and the key nobody listed is the one that
    leaks. Keys survive so an operator can still see the SHAPE of what the camera
    used -- `?token=&channel=` is readable, a dropped query is not.

    An item with no `=` has no key/value split to exploit: it could be a bare flag or
    it could be a bare signature, and there is no way to tell, so it is emitted empty
    and its identity is carried by the digest instead.

    Order is preserved, because reordering would merge `?a=1&b=2` with `?b=2&a=1`.
    """
    redacted = []
    for item in query.split("&"):
        key, sep, _value = item.partition("=")
        redacted.append(f"{key}{sep}" if sep else "")
    return "&".join(redacted)


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

    Every QUERY VALUE and the whole fragment go too, whatever the key is
    called -- a signed URL carries its secret there, and a denylist of
    credential-shaped key names fails open on the key nobody listed. Query
    keys and their order are kept so a reader can still see the shape of
    what the camera used, and a short one-way digest of the original query
    and fragment is appended so that two cameras differing only in their
    query remain two identifiers.

    A URL that cannot be parsed yields ``"<unparseable url>"`` rather than
    falling back to the raw string. The parse-failure path is exactly where
    a leak would hide, so it emits nothing of the URL at all. Reaching it
    costs identity -- two such URLs collapse -- so it is reached only when
    there is something to redact.
    """
    try:
        # `urlsplit` itself raises on an invalid IPv6 authority -- an unbracketed
        # `[` -- long before `.port` is touched, so the split belongs inside the
        # guard too. Moving it out was a regression against `rtsp://cam:pw@[2001:db8::1/s`.
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"

    has_userinfo = "@" in parts.netloc
    # A query or a fragment can carry a secret just as a userinfo can -- a signed HLS
    # URL puts it there, and so do a large share of IP cameras. Returning early on
    # "no `@`" was how `rtsp://host/s?token=SECRET` reached every sink untouched.
    has_query_or_fragment = bool(parts.query) or bool(parts.fragment)

    if not has_userinfo and not has_query_or_fragment:
        # Nothing to redact. Byte-identical, so every existing identifier for a file,
        # a directory or a webcam is unchanged -- including one whose authority cannot
        # be parsed, such as `rtsp://host:abc/path`. Collapsing THAT would trade a leak
        # that was never there for a silent cross-stream mix-up.
        return url

    try:
        # Touching .hostname/.port is what raises on a malformed authority,
        # so it must happen inside the guard.
        host, port = parts.hostname, parts.port
    except ValueError:
        return "<unparseable url>"

    if has_userinfo:
        if host is None:
            # An authority we cannot decompose: emit nothing of it.
            return "<unparseable url>"
        netloc = f"{host}:{port}" if port is not None else host
    else:
        # Untouched, deliberately: `.hostname` lowercases and re-renders IPv6, which
        # would be a byte change to a component that carries no credential.
        netloc = parts.netloc

    if not has_query_or_fragment:
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    return urlunsplit(
        (
            parts.scheme,
            netloc,
            parts.path,
            _redact_query(parts.query),
            _identity_digest(parts.query, parts.fragment),
        )
    )
