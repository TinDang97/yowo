"""m1 box 6 claims exactly what the code does, and the limit is named where it is read.

Box 6 was worded "No credential reaches a log, an exception message, a result payload,
or a cache key". That is unsatisfiable by any change to this repository:

    rtsp://h/live/S3CR3T-signed/stream   ->   unchanged, at both boundaries

A path-borne token survives redaction, and redaction cannot take it. The path IS the
camera's identity -- `rtsp://h/live/one` and `rtsp://h/live/two` are two cameras --
so emptying it collapses them onto one stream id, and `DetectionRouter.register`
overwrites a duplicate key without complaint. That is the cross-delivery bug
`credential-sinks-are-bound` fixed at the cause; re-introducing it to satisfy a
sentence would be a strictly worse trade.

So the box now enumerates what redaction covers -- userinfo, query, fragment -- and
names the path as a residual risk in its own body. The checks here hold that
enumeration to its evidence in both directions: every component the box names is
driven and proved clean, and the box may not name one this file does not drive.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from yowo.io._redact import redact_url
from yowo.pipeline._ids import safe_stream_id

_REPO_ROOT = Path(__file__).parent.parent.parent
_MILESTONE = _REPO_ROOT / ".add/milestones/m1-trust-the-ship.md"
_SECURITY = _REPO_ROOT / "SECURITY.md"

# The box is located by its task citation, not by its prose. Locating it by a phrase
# from its own wording is what broke both witnesses the moment the wording changed --
# and a locator that silently matches nothing turns "the box is wrong" into "the box
# is fine", which is the failure mode furthest from what these checks are for.
_BOX_MARKER = "← rtsp-credential-redaction"

SECRET = "S3CR3T-0pAqUe-9f2b1d"
HOST = "10.0.0.5:554"

# One form per component the box names. The value is the same secret in every one, so
# a check that passes for the wrong reason -- matching some other part of the URL --
# fails here too.
COVERED_FORMS = {
    "userinfo": f"rtsp://camop:{SECRET}@{HOST}/stream",
    "query": f"rtsp://{HOST}/stream?token={SECRET}",
    "fragment": f"rtsp://{HOST}/stream#{SECRET}",
}

# The component the box explicitly does NOT claim, and why this file exists.
PATH_BORNE = f"rtsp://{HOST}/live/{SECRET}-signed/stream"

# Every component word a reader could take the box to be claiming. A box naming one
# outside COVERED_FORMS is claiming ground no check here establishes (R:SILENTWIDEN).
_COMPONENT_VOCABULARY = (
    "userinfo",
    "query",
    "fragment",
    "path",
    "host",
    "port",
    "header",
    "scheme",
)


def _box_line() -> str:
    """The box 6 line, located by citation so a rewording cannot silently lose it."""
    lines = [
        line
        for line in _MILESTONE.read_text().splitlines()
        if line.startswith("- [") and _BOX_MARKER in line
    ]
    assert len(lines) == 1, (
        f"m1 box 6 is not where it was ({len(lines)} lines carry {_BOX_MARKER!r}). "
        "These checks assert things ABOUT that box; with the box gone they would all "
        "pass vacuously, which is worse than failing"
    )
    return lines[0]


def _security_section() -> str:
    """The SECURITY.md section covering credentials in a source URL, heading included."""
    text = _SECURITY.read_text()
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    matching = [
        s
        for s in sections
        if "url" in s.splitlines()[0].lower() and "credential" in s.splitlines()[0].lower()
    ]
    assert len(matching) == 1, (
        f"SECURITY.md has {len(matching)} `##` headings naming both a URL and credentials. "
        "Box 6 says path-borne secrets are 'documented in SECURITY.md'; a reader arriving "
        "on that promise needs one findable heading, not a sentence buried in a section "
        "about checkpoint deserialization (M4, A12)"
    )
    return "## " + matching[0]


# ---------------------------------------------------------------------------
# M1, A2, E2 — the box enumerates, and every name it uses is earned
# ---------------------------------------------------------------------------


def test_the_box_names_the_three_covered_components() -> None:
    """covers: M1, A2 — enumerated, not generalised.

    A general phrase -- "URL credentials", "credentials in a source URL" -- re-opens
    exactly the unsatisfiability the amendment exists to close, because the path is a
    URL credential too. The box has to say which three.
    """
    box = _box_line().lower()
    missing = [name for name in COVERED_FORMS if name not in box]
    assert not missing, (
        f"m1 box 6 does not name {missing!r}. The box must enumerate the components "
        "redaction covers; a general phrase silently re-claims the path form and puts "
        "the box back where the amendment found it (M1)"
    )


@pytest.mark.parametrize("component", sorted(COVERED_FORMS))
def test_each_named_component_is_actually_clean_at_every_sink(component: str) -> None:
    """covers: M1, A2, E2 — the box's positive claim, executed rather than restated.

    The exhaustive four-sink binding for the userinfo and query forms lives in
    `test_credential_sinks` and `test_query_credentials`; this drives all three named
    components -- the fragment included, which neither closed node covers by itself --
    through both boundaries and through the one sink that outlives log scrubbing, the
    routing callback's dict key.
    """
    from yowo.pipeline import DetectionRouter

    url = COVERED_FORMS[component]

    for boundary, redacted in (
        ("redact_url", redact_url(url)),
        ("safe_stream_id", safe_stream_id(url)),
    ):
        assert SECRET not in redacted, (
            f"the {component} form leaks at {boundary}: {redacted!r} still carries "
            f"{SECRET!r}, and m1 box 6 claims it does not (M1)"
        )
        assert "10.0.0.5" in redacted, (
            f"the {component} form lost its host at {boundary}: {redacted!r}. An operator "
            "who cannot tell WHICH camera failed cannot act on the message"
        )

    router = DetectionRouter()
    router.register(url, lambda sid, dets: None)
    for key in router._callbacks:
        assert SECRET not in key, (
            f"the {component} form reached the routing callback table as a dict key "
            f"({key!r}). That key outlives log scrubbing for the process lifetime (M1)"
        )


def test_the_box_claims_no_component_the_checks_do_not_establish() -> None:
    """covers: R:SILENTWIDEN — the box may not out-run its own evidence.

    The amendment's whole value is that the box says something true. A later edit that
    adds a fourth component to the sentence without adding it to COVERED_FORMS would
    restore the original defect in a new place, and would do it silently.
    """
    box = _box_line().lower()
    # The residual-risk clause names the path in order to DISCLAIM it, so it is read
    # from the claim half only -- everything before the clause that gives it away.
    claim, _, _residual = box.partition("residual risk")
    assert _residual, (
        "m1 box 6 has no `RESIDUAL RISK` clause, so its claim half cannot be separated "
        "from its disclaimer half (M2)"
    )
    # Word boundaries, not substrings: "host" lives inside "hostage" and this check
    # read the box's own justification as a claim about the host component. Same shape
    # as the `1` that matched inside `10.0.0.5` one node ago -- English embeds these
    # words, so a bare `in` scan over prose reports findings that are not there.
    claimed = {word for word in _COMPONENT_VOCABULARY if re.search(rf"\b{word}\b", claim)}
    unearned = claimed - set(COVERED_FORMS)
    assert not unearned, (
        f"m1 box 6 claims {sorted(unearned)!r}, which no check in this file drives. "
        "Either drive it here and add it to COVERED_FORMS, or take it out of the box — "
        "a box wider than its evidence is the defect the amendment closed (R:SILENTWIDEN)"
    )


# ---------------------------------------------------------------------------
# M2, M3, A6 — the limit is where it gets read
# ---------------------------------------------------------------------------


def test_the_residual_risk_is_in_the_box_body() -> None:
    """covers: M2, A6 — a ticked box ends the reading, so a footnote arrives too late.

    The audience is an operator or auditor scanning m1 to decide whether yowo is safe
    to deploy with signed-URL cameras. A tick reads as "handled"; whatever qualifies it
    has to be inside the line the tick is on.
    """
    box = _box_line()
    assert "RESIDUAL RISK" in box, (
        "m1 box 6 does not carry its residual risk in the box body. A reader who stops "
        "at the tick never reaches a qualification stored anywhere else (M2)"
    )
    assert "SECURITY.md" in box, (
        "m1 box 6 does not point at where the operator constraint is documented (M4)"
    )


def test_the_residual_risk_says_why_it_cannot_be_fixed() -> None:
    """covers: M3 — without the reason it reads as an open TODO.

    A maintainer who finds "the path is not redacted" and no reason has every incentive
    to go and redact it. The box has to carry the consequence, not just the fact.
    """
    box = _box_line().lower()
    for phrase, why in (
        ("identity", "the box must say the path IS the camera's identity"),
        ("collapse", "the box must name the collapse that redacting it would cause"),
    ):
        assert phrase in box, (
            f"m1 box 6's residual risk is missing {phrase!r}: {why}. Stated as a bare "
            "fact rather than a reason, it reads as an invitation to 'fix' it (M3)"
        )


# ---------------------------------------------------------------------------
# R:FICTION, E6 — nothing here states a transformation that does not happen
# ---------------------------------------------------------------------------

_ARROW = re.compile(r"(\S+://\S*)\s+->\s+(\S+)")


def test_the_box_claims_nothing_a_probe_refutes() -> None:
    """covers: R:FICTION, E6 — every transformation written down is executed.

    This is the check the review of #32 wanted and nobody had. `test_box_6_is_not_ticked`
    had its assertions re-derived and its docstring left behind, so the prose went on
    stating two query transformations that the same commit disproved. Prose drifts
    silently; only running it catches that.
    """
    import tests.unit.test_credential_sinks as sinks
    import tests.unit.test_query_credentials as query

    sources = {
        "m1 box 6": _box_line(),
        "SECURITY.md": _security_section(),
        "test_credential_sinks::test_box_6_is_not_ticked": sinks.test_box_6_is_not_ticked.__doc__
        or "",
        "test_query_credentials::test_box_6_is_not_ticked_until_this_is_green": (
            query.test_box_6_is_not_ticked_until_this_is_green.__doc__ or ""
        ),
    }

    executed = 0
    for where, text in sources.items():
        for raw_in, raw_out in _ARROW.findall(text):
            claimed_in = raw_in.strip("`'\",")
            claimed_out = raw_out.strip("`'\",")
            if claimed_out in {"unchanged", "..."}:
                claimed_out = claimed_in
            actual = redact_url(claimed_in)
            executed += 1
            assert actual == claimed_out, (
                f"{where} states a transformation that does not happen:\n"
                f"  it claims : {claimed_in!r} -> {claimed_out!r}\n"
                f"  it is     : {claimed_in!r} -> {actual!r}\n"
                "Prose that asserts a refuted guarantee is the defect this node exists "
                "to stop; re-derive it against the running code (R:FICTION)"
            )

    assert executed >= 2, (
        f"only {executed} transformation claim(s) were found and executed. This check is "
        "the guard against prose drifting away from the code, and it guards nothing if "
        "the documents stop showing worked examples"
    )


def test_the_path_form_is_still_the_exception() -> None:
    """covers: E1 — the reason box 6 is qualified, re-derived rather than remembered.

    If this ever stops holding, path redaction was added and the residual-risk clause is
    now itself a fiction -- so it must be re-derived, not deleted.
    """
    for boundary, redacted in (
        ("redact_url", redact_url(PATH_BORNE)),
        ("safe_stream_id", safe_stream_id(PATH_BORNE)),
    ):
        assert redacted == PATH_BORNE, (
            f"{boundary} no longer returns a path-borne token unchanged ({redacted!r}). "
            "That is the exception m1 box 6 names, so either the box is now wrong or two "
            "cameras on one host just collapsed onto one id — re-derive it (E1)"
        )


# ---------------------------------------------------------------------------
# R:PATHREDACT, A16, E4 — what redacting the path would actually cost
# ---------------------------------------------------------------------------


def test_adding_path_redaction_collapses_two_cameras() -> None:
    """covers: R:PATHREDACT, A16, E4 — the reason, executed on the real router.

    Two cameras on one host differ only by path. Empty the path and they are one dict
    key, and `DetectionRouter.register` takes the second without complaint — so camera
    one's detections are delivered to camera two's callback. This is not a worry; it is
    the bug `credential-sinks-are-bound` fixed, reproduced deliberately.
    """
    from urllib.parse import urlsplit, urlunsplit

    from yowo.pipeline import DetectionRouter

    one = f"rtsp://{HOST}/live/camera-one/stream"
    two = f"rtsp://{HOST}/live/camera-two/stream"

    assert safe_stream_id(one) != safe_stream_id(two), (
        "two cameras differing only in path already share one stream id. Path redaction "
        "has been added, and this is the cross-delivery bug (R:PATHREDACT)"
    )

    def _redacting_the_path(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, "/", parts.query, parts.fragment))

    delivered: list[str] = []
    router = DetectionRouter()
    router.register(_redacting_the_path(one), lambda sid, dets: delivered.append("one"))
    router.register(_redacting_the_path(two), lambda sid, dets: delivered.append("two"))

    assert len(router._callbacks) == 1, (
        "the stub that redacts the path did not collapse two cameras onto one key, so "
        "this check no longer demonstrates the cost it exists to demonstrate — re-derive "
        "it rather than trusting it (A16)"
    )
    assert _redacting_the_path(one) == _redacting_the_path(two), (
        "the two URLs did not collapse; pick two that differ only in their path"
    )


# ---------------------------------------------------------------------------
# M4, A8, A9, A10, A12 — the operator constraint, where an operator will find it
# ---------------------------------------------------------------------------


def test_security_md_documents_the_operator_constraint() -> None:
    """covers: M4, E5 — box 6 says "documented in SECURITY.md"; this is that word being true."""
    section = _security_section().lower()
    for phrase, why in (
        ("path", "the section must name the URL path as the uncovered component"),
        ("not redact", "the section must say plainly that yowo does not redact it"),
        ("identity", "the section must say why — the path is the camera's identity"),
    ):
        assert phrase in section, (
            f"SECURITY.md's URL-credential section is missing {phrase!r}: {why}. Box 6 "
            "claims this is documented, and an unfulfilled claim in a milestone box is "
            "the failure the amendment set out to remove (M4)"
        )


def test_security_md_has_its_own_findable_heading() -> None:
    """covers: A12 — arrived from box 6, must not have to read about CVE intake to find it."""
    heading = _security_section().splitlines()[0]
    assert heading.startswith("## "), f"expected a level-2 heading, got {heading!r}"
    # `_security_section` already refuses anything but exactly one match; asserting the
    # shape here is what makes the requirement visible at the point it is required.


def test_security_md_shows_a_leaking_form_and_a_safe_rewrite() -> None:
    """covers: A8 — a constraint stated without a remedy is a complaint.

    The operator reading this has a camera URL in hand and needs to know what to write
    instead, not merely that what they have is wrong.
    """
    section = _security_section()
    assert "/live/" in section or "rtsp://" in section, (
        "SECURITY.md's URL-credential section shows no example URL. An operator cannot "
        "match their own camera against a description (A8)"
    )
    assert "?" in section and "token=" in section.lower(), (
        "SECURITY.md's URL-credential section does not show the safe rewrite — moving "
        "the secret into the query, which yowo does redact. Without it the section tells "
        "an operator they have a problem and not what to do about it (A8)"
    )


def test_security_md_says_what_to_do_when_the_path_token_is_unavoidable() -> None:
    """covers: A10 — the operator with no choice must not read past their own case.

    Some cameras hard-code the token into the path and offer no query form. Silence
    there reads as "not a problem", and that operator is exactly the one at risk.
    """
    section = _security_section().lower()
    for phrase, why in (
        ("cannot", "the section must acknowledge the operator who has no alternative"),
        ("log", "it must say the logs are then secret-bearing"),
        ("cache", "it must say the feature-cache keys are then secret-bearing too"),
    ):
        assert phrase in section, (
            f"SECURITY.md's URL-credential section is missing {phrase!r}: {why}. The "
            "operator who cannot rewrite the URL finds no case for their own situation "
            "and concludes they are safe (A10)"
        )


def test_security_md_does_not_implicate_non_url_sources() -> None:
    """covers: A9 — a file, a directory and a webcam index carry nothing yowo could redact.

    They are returned byte-identical by design. An operator who reads the section as
    covering local media hunts a risk that is not there, and the time goes on the wrong
    thing.
    """
    section = _security_section().lower()
    assert "file" in section and "webcam" in section, (
        "SECURITY.md's URL-credential section does not scope itself to URL sources. A "
        "reader cannot tell whether a local file path or a webcam index is implicated "
        "(A9)"
    )

    for benign in ("/var/media/clip.mp4", "/var/media/", "webcam:0", "cam-0"):
        assert redact_url(benign) == benign, (
            f"{benign!r} is no longer returned byte-identical ({redact_url(benign)!r}), so "
            "the scoping SECURITY.md states is now false"
        )


# ---------------------------------------------------------------------------
# M7 — this node moved a line and a document, and nothing else
# ---------------------------------------------------------------------------

# Pinned from f9784fc, the commit that shipped query and fragment redaction. These are
# not re-derived from the implementation: a pin computed by the code it pins proves
# nothing. They were produced by running that commit and are transcribed here.
_BEHAVIOUR_AT_F9784FC = {
    "rtsp://camop:hunter2@10.0.0.5:554/s?token=SECRET&password=hunter2": (
        "rtsp://10.0.0.5:554/s?token=&password=#q0734d834"
    ),
    "rtsp://h/s?channel=1": "rtsp://h/s?channel=#q06592c6c",
    "rtsp://h/s?channel=2": "rtsp://h/s?channel=#q66b20e1f",
    "rtsp://user:pw@h/live/S3CR3T/stream": "rtsp://h/live/S3CR3T/stream",
    "rtsp://cam:pw@[2001:db8::1/s": "<unparseable url>",
    "rtsp://host:abc/path": "rtsp://host:abc/path",
    "https://cdn/hls/master.m3u8?st=abc&e=123#tok=Z": "https://cdn/hls/master.m3u8?st=&e=#qc89745f4",
}


@pytest.mark.parametrize("url", sorted(_BEHAVIOUR_AT_F9784FC))
def test_no_redaction_behaviour_changed(url: str) -> None:
    """covers: M7 — a contract line and a document moved; the redactor did not.

    Amending a box to match the code is only honest while the code is the code that was
    measured. If this reddens, the box was amended to describe something that has since
    changed underneath it.
    """
    assert redact_url(url) == _BEHAVIOUR_AT_F9784FC[url], (
        f"redaction of {url!r} changed since f9784fc:\n"
        f"  was: {_BEHAVIOUR_AT_F9784FC[url]!r}\n"
        f"  now: {redact_url(url)!r}\n"
        "m1 box 6 was amended to describe the behaviour at f9784fc. Re-measure it "
        "before letting this pin move (M7)"
    )


# ---------------------------------------------------------------------------
# A18 — what a witness failure teaches the maintainer who caused it
# ---------------------------------------------------------------------------


def test_a_witness_failure_names_the_cross_delivery_consequence() -> None:
    """covers: A18 — the reader of the failure has just 'fixed' path redaction.

    The message has to tell them what they broke. "This string changed" sends them to
    update the assertion; "two cameras now share one id and detections cross-deliver"
    sends them to revert.
    """
    import inspect

    import tests.unit.test_credential_sinks as sinks
    import tests.unit.test_query_credentials as query

    for name, fn in (
        ("test_credential_sinks::test_box_6_is_not_ticked", sinks.test_box_6_is_not_ticked),
        (
            "test_query_credentials::test_box_6_is_not_ticked_until_this_is_green",
            query.test_box_6_is_not_ticked_until_this_is_green,
        ),
    ):
        src = inspect.getsource(fn).lower()
        assert "collapse" in src or "cross-deliver" in src, (
            f"{name}'s failure messages do not name the consequence of redacting the "
            "path. A maintainer who just added path redaction reads the failure, sees "
            "only that a value changed, and updates the assertion (A18)"
        )
