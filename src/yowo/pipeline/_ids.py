"""The one place an operator-supplied stream identifier is made safe to emit.

A stream id is whatever the caller passed to :meth:`FrameCollector.add_stream` or
:meth:`DetectionRouter.register`, and in practice that is very often the camera URL --
credentials and all. From there the same string reaches a log record, an exception
message, a dict key, a callback argument and a thread name, and a credential in any of
those outlives log scrubbing.

Redaction therefore happens HERE, once, on the way in -- not at each sink. Sinks receive
an already-safe value, so a sink added tomorrow is safe by construction and no future
maintainer has to remember to redact. Every public entry point that accepts an
operator-supplied identifier passes it through :func:`safe_stream_id` before anything
else in the package sees it.

Redaction must not cost identity. A stream id is also a KEY -- of ``_streams``, of
``stream_errors``, of the router's callback table -- and two ids that collapse to one
string become one stream: ``add_stream`` raises "already registered", and
:meth:`DetectionRouter.register` does not raise at all, it silently overwrites, so one
camera's detections would be delivered to another camera's callback.

:func:`yowo.io.redact_url` cannot be applied blindly for that reason. Its parse-failure
path returns ``"<unparseable url>"``, and it reaches that path for any URL whose
authority it cannot decompose -- including ``"rtsp://host:abc/path"`` and
``"rtsp://10.0.0.5:99999/s"``, neither of which carries a credential. Collapsing those
would trade a leak that was never there for a silent cross-stream mix-up.

So the boundary checks first whether userinfo is even possible. A credential must be
introduced by an ``@``; an identifier with no ``@`` anywhere in it cannot carry one, and
is returned unchanged, byte for byte. Only when an ``@`` is present does the id go to
``redact_url``, whose conservative parse-failure path is then the right answer: an
authority we cannot decompose but which does contain an ``@`` is exactly where a
credential would hide.
"""

from __future__ import annotations

from yowo.io import redact_url

__all__ = ["safe_stream_id"]


def safe_stream_id(stream_id: str) -> str:
    """Return `stream_id` with any embedded credential removed.

    Applied at the package boundary, so every downstream sink -- logs, exception
    messages, dict keys, thread names, routing callbacks -- receives a value that is
    already safe to emit.

    Args:
        stream_id: The identifier as supplied by the caller. May be a plain name, a
            path, or a URL carrying userinfo.

    Returns:
        The identifier with any userinfo component removed. An identifier containing no
        ``@`` cannot carry a credential and is returned unchanged, byte for byte. An
        identifier that does contain an ``@`` but whose authority cannot be parsed
        collapses to ``"<unparseable url>"`` -- emitting nothing of it is the safe answer,
        at the cost of no longer distinguishing two such ids from each other.
    """
    # An identifier with no "@", no "?" and no "#" has nowhere to hide a credential:
    # no userinfo, no query, no fragment. Short-circuiting on those three keeps
    # `redact_url`'s parse-failure path -- which returns a single fixed string -- from
    # collapsing distinct credential-free ids onto one another.
    #
    # The "@" alone was not enough. `rtsp://host:abc/p?token=SECRET` has no userinfo,
    # so it was returned BYTE-IDENTICAL and the token reached every log site, the
    # RuntimeError, the dict keys, the thread name and the routing callback.
    if not any(c in stream_id for c in "@?#"):
        return stream_id
    return redact_url(stream_id)
