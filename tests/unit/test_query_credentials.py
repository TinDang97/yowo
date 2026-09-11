"""A secret in a URL's query or fragment is unreadable, and two cameras stay two cameras.

Red-first for ADD task `query-string-credentials`.

`redact_url` strips userinfo and passes the query and fragment through untouched. The
damning form is the mixed one:

    rtsp://camop:hunter2@10.0.0.5:554/s?token=SECRET&password=hunter2
 -> rtsp://10.0.0.5:554/s?token=SECRET&password=hunter2

The userinfo IS stripped, so the `@` is gone and the string wears the visual signature
of a redacted URL while carrying the literal password. That defeats eyeball review of a
log, and it defeats a reviewer diffing before against after, because the diff shows
redaction happening.

The sink checks here reuse `_assert_sink_clean` from `test_credential_sinks.py` rather
than defining a parallel one (M7). Those four sinks are already bound; what this node
proves is that they hold against a NEW credential form.
"""

from __future__ import annotations

import re

import pytest

from tests.unit.test_credential_sinks import _assert_sink_clean
from yowo.io import redact_url
from yowo.pipeline._ids import safe_stream_id

# Obviously fake, and deliberately two different shapes: a low-entropy password that
# also appears as userinfo, and a high-entropy opaque token that only ever appears in
# the query. A fix that handles one and not the other is a fix that is not finished.
USER = "camop"
PASSWORD = "hunter2"
TOKEN = "S3CR3T-0pAqUe-9f2b1d"

MIXED = f"rtsp://{USER}:{PASSWORD}@10.0.0.5:554/s?token={TOKEN}&password={PASSWORD}"

# Every function that must not leak, so each check covers both boundaries at once. A
# fix applied to one early return and not the other is the specific failure M5 names.
BOUNDARIES = (redact_url, safe_stream_id)

_DIGEST = re.compile(r"#q[0-9a-f]{8}$")


def _all_forms(value: str) -> list[str]:
    """`value` in every encoding it could hide in, for R:REVERSIBLE."""
    import base64
    import urllib.parse

    raw = value.encode()
    return [
        value,
        value.lower(),
        raw.hex(),
        base64.b64encode(raw).decode().rstrip("="),
        base64.urlsafe_b64encode(raw).decode().rstrip("="),
        urllib.parse.quote(value),
    ]


def _assert_no_secret(where: str, emitted: str, *secrets: str) -> None:
    leaked = [s for s in secrets if s in emitted]
    assert not leaked, (
        f"CREDENTIAL LEAKED at {where}\n"
        f"  emitted : {emitted!r}\n"
        f"  leaked  : {leaked!r}\n"
        "  Fix at the boundary — redact_url / safe_stream_id — not at the sink (M8)."
    )


# ---------------------------------------------------------------------------
# M1, M5 — everything is redacted, through both boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=lambda f: f.__name__)
def test_the_measured_mixed_form_leaks_nothing(boundary) -> None:
    """covers: M1, E1 — the case that caused this node."""
    out = boundary(MIXED)
    _assert_no_secret(f"{boundary.__name__}(MIXED)", out, PASSWORD, TOKEN, USER)
    assert "10.0.0.5" in out, (
        f"the host is gone from {out!r} — an operator who cannot tell WHICH camera "
        "failed cannot act on the message (A6)"
    )


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=lambda f: f.__name__)
def test_a_query_credential_with_no_userinfo_is_redacted(boundary) -> None:
    """covers: M1, M5, E2 — no `@`, so both early returns skip it today."""
    out = boundary(f"rtsp://10.0.0.5/s?token={TOKEN}")
    _assert_no_secret(f"{boundary.__name__} with no userinfo", out, TOKEN)


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=lambda f: f.__name__)
def test_an_unparseable_authority_with_a_query_credential_is_redacted(boundary) -> None:
    """covers: M1, M5, E3 — the worst case: untouched by both boundaries today."""
    out = boundary(f"rtsp://host:abc/p?token={TOKEN}")
    _assert_no_secret(f"{boundary.__name__} unparseable+query", out, TOKEN)


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=lambda f: f.__name__)
def test_the_fragment_is_redacted(boundary) -> None:
    """covers: M1, E6 — the fragment leaks by the same two paths as the query."""
    out = boundary(f"rtsp://10.0.0.5/s#session={TOKEN}")
    _assert_no_secret(f"{boundary.__name__} fragment", out, TOKEN)


@pytest.mark.parametrize(
    "key", ["token", "sessionId", "x", "auth", "sig", "st", "e", "zzz_unknown_key"]
)
def test_every_value_goes_whatever_its_key_is_called(key: str) -> None:
    """covers: M1, R:DENYLIST — the key nobody listed is the one that leaks."""
    out = redact_url(f"rtsp://10.0.0.5/s?{key}={TOKEN}")
    _assert_no_secret(f"?{key}=", out, TOKEN)


# ---------------------------------------------------------------------------
# M2, M4 — identity survives redaction
# ---------------------------------------------------------------------------


def test_query_keys_and_their_order_survive() -> None:
    """covers: M4, A5, E8 — keys tell a reader the shape of what was there."""
    # Keys deliberately NOT in alphabetical order. With `alpha`/`beta` a `sorted()`
    # in the implementation is indistinguishable from preserving order, and the check
    # passes while proving nothing — verified: that mutation survived the first draft.
    out = redact_url("rtsp://10.0.0.5/s?zulu=1&alpha=2")
    assert "alpha" in out and "zulu" in out, (
        f"query keys were dropped from {out!r}; a dropped query tells an operator "
        "nothing about what the camera used (M4)"
    )
    assert out.index("zulu") < out.index("alpha"), (
        f"query keys were reordered in {out!r}. Reordering merges ?a=1&b=2 with "
        "?b=2&a=1 — the digest still separates them, but the id an operator reads no "
        "longer matches the camera they configured (A5)"
    )
    # The QUERY portion only. Scanning the whole string would match the `1` in the
    # host `10.0.0.5` and turn a correct implementation red for the wrong reason.
    query = out.split("#")[0].partition("?")[2]
    assert "1" not in query and "2" not in query, (
        f"a query VALUE survived in {query!r} (from {out!r}) — keys are kept, values are not (M1)"
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("rtsp://h/s?channel=1", "rtsp://h/s?channel=2"),
        ("rtsp://h/s?a=1&b=2", "rtsp://h/s?b=2&a=1"),
        ("rtsp://h/s#one", "rtsp://h/s#two"),
        ("rtsp://h/s?x=1", "rtsp://h/s?x=1&y=2"),
    ],
)
def test_two_cameras_differing_only_in_query_stay_two_identifiers(a: str, b: str) -> None:
    """covers: M2, R:COLLAPSE, E5 — the id is a dict key, and register() overwrites silently.

    Collapsing two cameras onto one identifier delivers one camera's detections to the
    other camera's callback, with nothing raised. That is why the digest exists.
    """
    ra, rb = redact_url(a), redact_url(b)
    assert ra != rb, (
        f"{a!r} and {b!r} both redact to {ra!r}. They are two cameras; as one dict key "
        "they become one stream, and DetectionRouter.register overwrites without "
        "raising (M2, R:COLLAPSE)"
    )


def test_the_digest_is_computed_before_redaction() -> None:
    """covers: M2, A3 — a digest of the redacted form preserves no identity at all.

    Every URL with the same query KEYS redacts to the same string, so a digest taken
    afterwards is identical for all of them and distinguishes nothing.
    """
    same_keys = [f"rtsp://h/s?token={n}" for n in ("aaa", "bbb", "ccc")]
    digests = {redact_url(u) for u in same_keys}
    assert len(digests) == 3, (
        f"three different tokens under the same key produced {len(digests)} "
        f"identifier(s): {digests!r}. The digest must cover the ORIGINAL query (A3)"
    )


def test_the_digest_does_not_carry_the_secret() -> None:
    """covers: M3, R:REVERSIBLE — not raw, not hex, not base64, not url-quoted."""
    out = redact_url(f"rtsp://10.0.0.5/s?token={TOKEN}")
    for form in _all_forms(TOKEN):
        assert form not in out, (
            f"the secret survives in {out!r} as {form!r}. A digest is an identity "
            "control, and it must not become a way back to the value (R:REVERSIBLE)"
        )
    assert _DIGEST.search(out), (
        f"{out!r} carries no digest marker. A6: a redacted id must be visibly "
        "redacted, not indistinguishable from a camera with genuinely empty parameters"
    )


# ---------------------------------------------------------------------------
# M6 — nothing that works today changes shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clean",
    [
        "rtsp://10.0.0.5:554/stream",
        "rtsp://10.0.0.5/stream",
        "http://camera.local/feed",
        "/var/media/clip.mp4",
        "/var/media/",
        "webcam:0",
        "cam-0",
        "",
    ],
)
def test_a_url_with_nothing_to_redact_is_byte_identical(clean: str) -> None:
    """covers: M6, A4, R:BYTEDRIFT, E9 — no digest, no drift, for either boundary."""
    for boundary in BOUNDARIES:
        assert boundary(clean) == clean, (
            f"{boundary.__name__}({clean!r}) returned {boundary(clean)!r}. A value with "
            "nothing to redact must come back byte-identical, or every existing "
            "source_id for a file, a directory or a webcam changes (M6, R:BYTEDRIFT)"
        )


def test_an_unparseable_authority_with_nothing_to_redact_is_byte_identical() -> None:
    """covers: M6, R:BYTEDRIFT, E4 — the regression `credential-sinks-are-bound` fixed.

    `redact_url`'s parse-failure path returns one fixed string. Reaching it for a URL
    that carries no credential collapses distinct ids onto each other — and the id is
    a dict key, so that cross-delivers detections.
    """
    for value in ("rtsp://host:abc/path", "rtsp://10.0.0.5:99999/s"):
        assert safe_stream_id(value) == value, (
            f"safe_stream_id({value!r}) returned {safe_stream_id(value)!r}. It carries "
            "no credential, so collapsing it trades a leak that was never there for a "
            "silent cross-stream mix-up (M6, R:BYTEDRIFT)"
        )


def test_degenerate_query_forms_do_not_crash_and_stay_distinct() -> None:
    """covers: E7 — `?flag`, `?a=&b=`, a bare `?`, an empty fragment."""
    forms = [
        "rtsp://h/s?flag",
        "rtsp://h/s?a=&b=",
        "rtsp://h/s?",
        "rtsp://h/s#",
        "rtsp://h/s?flag&other",
    ]
    out = [redact_url(f) for f in forms]
    assert len(set(out)) == len(forms), (
        f"degenerate query forms collapsed: {dict(zip(forms, out, strict=True))!r}"
    )


# ---------------------------------------------------------------------------
# M7, M8 — the sinks, and only the boundaries
# ---------------------------------------------------------------------------


def test_the_four_sinks_are_clean_against_a_query_credential() -> None:
    """covers: M7 — the sinks are bound; this proves they hold for a new credential form."""
    from yowo.pipeline import DetectionRouter, FrameCollector

    expected = safe_stream_id(MIXED)
    _assert_no_secret("the redacted form itself", expected, PASSWORD, TOKEN, USER)

    collector = FrameCollector()
    try:
        router = DetectionRouter()
        router.register(MIXED, lambda sid, dets: None)
        # The routing callback table — a dict key that outlives log scrubbing.
        for key in router._callbacks:
            _assert_sink_clean("DetectionRouter callback key", key, expected=expected)
            _assert_no_secret("DetectionRouter callback key", key, TOKEN)
    finally:
        collector.close()


def test_redaction_still_happens_only_at_the_two_boundaries() -> None:
    """covers: M8 — no sink learned about query strings."""
    import inspect

    for module in ("yowo.io._source", "yowo.pipeline._collector", "yowo.pipeline._router"):
        src = inspect.getsource(__import__(module, fromlist=["_"]))
        assert "parse_qsl" not in src and "urlencode" not in src, (
            f"{module} parses query strings. Redaction belongs at the two boundaries "
            "so a sink added tomorrow is safe by construction (M8)"
        )


def test_box_6_is_not_ticked_until_this_is_green() -> None:
    """covers: R:TICKBOX — the query form is one of the three box 6 now names.

    The name is historical and kept deliberately: this node is closed and its CHECKS
    cite it by name, so renaming would dangle that citation (method M12).

    When this node ran, box 6 read "No credential reaches a log ..." and the query form
    held it open. That form is closed now, and the box was amended on 2026-09-11 to
    enumerate what redaction actually covers — userinfo, query, fragment — because a
    path-borne secret survives and redaction cannot take it without collapsing two
    cameras onto one stream id.

    So this check no longer asserts the box is unticked. It asserts that the query form
    this node closed is one of the three the box names, and that the box still carries
    the clause naming what it does not.
    """
    from pathlib import Path

    root = Path(__file__).parent.parent.parent
    milestone = (root / ".add/milestones/m1-trust-the-ship.md").read_text()
    box = [
        line
        for line in milestone.splitlines()
        if line.startswith("- [") and "\u2190 rtsp-credential-redaction" in line
    ]
    assert len(box) == 1, f"m1 box 6 is not where it was ({len(box)} matches)"

    assert "QUERY" in box[0], (
        "m1 box 6 no longer names the QUERY form among the components redaction covers. "
        "This node closed that form; if the box has stopped claiming it, either the claim "
        "was dropped by mistake or the guarantee regressed — check which"
    )
    assert "RESIDUAL RISK" in box[0], (
        "m1 box 6 has lost the clause naming what redaction does NOT cover. Without it "
        "the box reads as covering every credential a URL can carry, which is exactly the "
        "wording that could not be satisfied"
    )

    # Re-derived against the running code rather than trusting the box. The query form is
    # closed, and the path form is why the box needs a residual risk at all: redacting the
    # path would collapse two cameras on one host onto one id, and `register` overwrites
    # silently, so detections would cross-deliver.
    assert TOKEN not in safe_stream_id(MIXED), (
        f"the query form has regressed ({safe_stream_id(MIXED)!r}) — this node closed it, "
        "and box 6 claims it stays closed"
    )
    path_borne = "rtsp://h/live/S3CR3T-signed/stream"
    assert safe_stream_id(path_borne) == path_borne, (
        f"a path-borne token is now redacted ({safe_stream_id(path_borne)!r}). Two cameras "
        "differing only in path have collapsed onto one stream id and detections "
        "cross-deliver — revert, and re-derive box 6's residual-risk clause"
    )


def test_the_digest_is_documented_as_an_identity_control() -> None:
    """covers: A7 — a reader must not mistake eight hex characters for a security control.

    Unlike M3, which is behavioural and proved by driving the code, this guarantee IS
    the documentation: the risk is a future maintainer reading the digest as though it
    protected the secret, and then shortening it, or reusing it somewhere it would have
    to. Reading the docstring here is reading the subject, not a proxy for it.
    """
    import yowo.io._redact as redact_module

    text = redact_module.__doc__ or ""
    text += "\n".join(
        (getattr(redact_module, name).__doc__ or "")
        for name in ("redact_url", "_identity_digest", "_redact_query")
    )
    source = __import__("inspect").getsource(redact_module)

    assert "IDENTITY control, not a security one" in source, (
        "the digest is not documented as an identity control. Eight hex characters "
        "distinguish two cameras; they do not protect a secret, and a maintainer who "
        "reads them the other way will make a wrong decision later (A7)"
    )
    assert "LOW-entropy" in source or "low-entropy" in source, (
        "the honest limit is undocumented: a digest over a low-entropy query such as "
        "`?channel=2` is effectively an encoding of it, because the space is small "
        "enough to enumerate. That is acceptable — a channel number is not a secret — "
        "but it must be said rather than implied away (A7)"
    )
