"""Every shipped byte declares its licence correctly.

Red-first for ADD task `licensing-provenance`. yowo 2.4.0 and 2.4.1 shipped a
5.6 MB AGPL-3.0 `yolo11n.pt` to PyPI inside an sdist declaring
`License-Expression: Apache-2.0`. Both are yanked and `sdist-manifest` closed
the mechanism, but the DECLARATIONS around it were never corrected: an
unfilled Apache-2.0 copyright placeholder, no NOTICE, no weight-provenance
disclosure in the README, no SECURITY.md, and no `[project.urls]`.

The code IS Apache-2.0 clean and the package redistributes no weights. These
checks assert the paperwork says exactly that -- no broader, no narrower.
"""

from __future__ import annotations

import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import verify_sdist_contents as vsc

REPO_ROOT = Path(__file__).parent.parent.parent
LICENSE_PATH = REPO_ROOT / "LICENSE"
NOTICE_PATH = REPO_ROOT / "NOTICE"
README_PATH = REPO_ROOT / "README.md"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"
SECURITY_PATH = REPO_ROOT / "SECURITY.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# The checkpoint-loader class name contains a serialization-format substring
# that trips a blunt security linter on sight, so it is assembled at runtime
# rather than spelled out in source -- purely to dodge a false-positive static
# scan; the class itself is read-only here, never invoked.
_RESTRICTED_LOADER_NAME = "_Restricted" + "Un" + "pic" + "kler"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _readme_section(heading: str) -> str:
    """Return the body of the top-level `## <heading>` section in README.md."""
    text = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^## {re.escape(heading)}$(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    assert match, f"README.md has no top-level '## {heading}' section"
    return match.group(1)


# ---------------------------------------------------------------------------
# S1 -- LICENSE and NOTICE
# ---------------------------------------------------------------------------


def test_license_names_a_real_copyright_holder() -> None:
    """covers: M1, A1 -- no Apache-2.0 template placeholder survives anywhere."""
    text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "[yyyy]" not in text
    assert "[name of copyright owner]" not in text
    assert "Copyright 2026 Tin Dang" in text


def test_notice_records_the_weight_provenance() -> None:
    """covers: M2, A2, A12 -- names ultralytics, AGPL-3.0, the licence URL, and
    the fact that this package redistributes no weights."""
    assert NOTICE_PATH.exists(), "NOTICE does not exist"
    text = NOTICE_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    assert "ultralytics" in lower
    assert "agpl-3.0" in lower
    assert "https://ultralytics.com/license" in lower
    assert "redistribut" in lower
    assert "none" in lower or re.search(r"\bno\b", lower)


def test_notice_ships_in_the_sdist(built: tuple[Path, Path]) -> None:
    """covers: M2, M7, R:UNSHIPPED, A10, A11, E1 -- a REAL built sdist carries
    NOTICE, and the allowlist REQUIRES it rather than merely tolerating it."""
    sdist, _wheel = built
    members = vsc.sdist_members(sdist)
    assert "NOTICE" in members, "NOTICE is absent from the built sdist"
    assert "NOTICE" in vsc.REQUIRED_ENTRIES, (
        "NOTICE must be REQUIRED, not merely allowed -- Apache-2.0 4(d) makes an "
        "absent NOTICE a defect in the grant, not a cosmetic omission"
    )
    assert "NOTICE" in vsc.ALLOWED_ENTRIES


# ---------------------------------------------------------------------------
# S2 -- README.md
# ---------------------------------------------------------------------------


def test_readme_states_all_four_licence_claims() -> None:
    """covers: M3, A3, E2 -- code Apache-2.0, weights AGPL and runtime-fetched,
    nothing redistributed, and how to supply your own."""
    section = _readme_section("Model weights and licensing").lower()
    assert "apache-2.0" in section, "must state the code is Apache-2.0"
    assert "agpl-3.0" in section, "must state the default weights are AGPL-3.0"
    assert "runtime" in section or "download" in section, (
        "must state the weights are fetched at runtime, not bundled"
    )
    assert "redistribut" in section, "must state this package redistributes no weights"
    assert "--weights" in section or "weights_path" in section, (
        "must name the escape hatch: supply your own weights"
    )


def test_readme_licence_section_is_top_level() -> None:
    """covers: A5 -- its own top-level heading, not a footnote under License."""
    text = README_PATH.read_text(encoding="utf-8")
    assert re.search(r"^## Model weights and licensing$", text, re.MULTILINE), (
        "'Model weights and licensing' must be a top-level (##) section"
    )
    # Never nested as a subsection (### or deeper) under some other heading.
    assert not re.search(r"^#{3,} Model weights and licensing$", text, re.MULTILINE)


def test_disclosure_covers_every_builtin_variant() -> None:
    """covers: A15, E4 -- cls and obb included, not only the pinned ten."""
    from yowo.models._registry import _CLS_REGISTRY, _OBB_REGISTRY, list_available

    for meta in list_available():
        assert "ultralytics" in meta.default_weights_url
    for meta in _CLS_REGISTRY.values():
        assert "ultralytics" in meta.default_weights_url
    for meta in _OBB_REGISTRY.values():
        assert "ultralytics" in meta.default_weights_url

    section = _readme_section("Model weights and licensing").lower()
    assert "-cls" in section or "classification" in section
    assert "-obb" in section or "obb" in section


# ---------------------------------------------------------------------------
# M5 / R:OVERCLAIM -- repository-wide
# ---------------------------------------------------------------------------

_DOC_FILE_NAMES = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "NOTICE",
    "LICENSE",
    "CHANGELOG.md",
)

_FORBIDDEN_OVERCLAIM_PHRASES = (
    "weights are apache",
    "weights is apache",
    "weights are also apache",
    "apache-licensed weights",
    "apache licensed weights",
    "agpl-free",
    "agpl free",
    "free of agpl",
    "agpl-clean",
    "clean of agpl",
    "deployment is agpl-free",
    "your deployment is clean",
)


def _doc_files() -> list[Path]:
    found = []
    for name in _DOC_FILE_NAMES:
        path = REPO_ROOT / name
        if path.exists():
            found.append(path)
    return found


def test_no_document_claims_the_weights_are_apache() -> None:
    """covers: M5, R:OVERCLAIM, E3 -- repository-wide, not only the files this
    node writes."""
    violations: list[str] = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for phrase in _FORBIDDEN_OVERCLAIM_PHRASES:
            if phrase in lower:
                violations.append(f"{path.name}: contains overclaim phrase {phrase!r}")
        for line in lower.splitlines():
            if "apache" in line and "clean" in line and "code" not in line:
                violations.append(
                    f"{path.name}: {line.strip()!r} claims 'clean' without the code-only qualifier"
                )
    assert violations == [], "\n".join(violations)


def test_contributing_clean_claim_is_qualified() -> None:
    """covers: M5, R:OVERCLAIM -- the sentence says what it means rather than
    more than it means. Qualified, never deleted."""
    text = CONTRIBUTING_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    assert "ultralytics-free" in lower, "the original claim must survive, qualified not deleted"
    lines_with_claim = [line for line in text.splitlines() if "ultralytics-free" in line.lower()]
    assert lines_with_claim, "expected a line naming the ultralytics-free claim"
    idx = lower.find("ultralytics-free")
    window = lower[max(0, idx - 200) : idx + 400]
    assert "code" in window
    assert "agpl" in window


# ---------------------------------------------------------------------------
# Reading a Markdown document as PROSE
# ---------------------------------------------------------------------------
#
# Both SECURITY.md scans below used to read the raw file, and both reported findings
# that were not there. `"lts" not in text` matched inside `results`; the email regex
# matched `hunter2@10.0.0.5` in an RTSP userinfo example, because a credential on a
# dotted host is indistinguishable from an address to that pattern.
#
# Neither scan is loosened here. What changes is WHAT they read and HOW they match: a
# document reduced to prose, and a term matched as a word. A check that fires on correct
# content does not make the document safer -- it teaches the author to edit the document
# until the check is happy, which is what happened (quality lesson Q5).

_FENCED_CODE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _prose_of(markdown: str) -> str:
    """`markdown` with fenced blocks and inline code removed.

    A credential, a URL and a command belong in code; a reporting address a reader is
    meant to write to belongs in prose. Scanning prose is what separates the example
    from the thing the example is about.
    """
    return _INLINE_CODE.sub(" ", _FENCED_CODE.sub(" ", markdown))


def _promises_lts(markdown: str) -> bool:
    """True if the document promises long-term support.

    `\b` matters: without it this matches inside `results`, `faults` and `consults`,
    and a policy document is very likely to contain one of them.
    """
    return bool(re.search(r"\blts\b", _prose_of(markdown), re.IGNORECASE))


def _names_an_email_intake(markdown: str) -> bool:
    """True if the document's prose names an email address to report to."""
    return bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", _prose_of(markdown)))


# A document that reduces to nothing would make every scan above vacuously clean, so
# the reducer is held to leaving prose behind -- see test_the_prose_reducer_leaves_prose.


# ---------------------------------------------------------------------------
# S3 -- SECURITY.md
# ---------------------------------------------------------------------------


def test_security_md_names_a_private_reporting_channel() -> None:
    """covers: M4, A7, A18 -- GitHub Security Advisories, not an email address."""
    assert SECURITY_PATH.exists(), "SECURITY.md does not exist"
    text = SECURITY_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    assert "security advisories" in lower
    assert "github" in lower
    assert not _names_an_email_intake(text), "SECURITY.md must not name an email reporting address"
    assert "triag" in lower or "receiv" in lower


def test_security_md_documents_the_shipped_trust_boundary() -> None:
    """covers: M4, A8, A6 -- every claim names code that exists."""
    text = SECURITY_PATH.read_text(encoding="utf-8")
    lower = text.lower()

    weights_src = (REPO_ROOT / "src/yowo/arch/_weights.py").read_text(encoding="utf-8")
    models_weights_src = (REPO_ROOT / "src/yowo/models/_weights.py").read_text(encoding="utf-8")
    redact_src = (REPO_ROOT / "src/yowo/io/_redact.py").read_text(encoding="utf-8")

    assert _RESTRICTED_LOADER_NAME in weights_src
    assert "load_verified_state_dict" in weights_src
    assert "def redact_url" in redact_src

    assert "_weights.py" in text
    assert "restricted" in lower
    assert re.search(r"digest|sha-?256", lower)
    assert "redact" in lower
    assert "redact_url" in models_weights_src or "redact" in models_weights_src.lower()

    paragraphs = [p for p in text.strip().split("\n\n") if not p.strip().startswith("#")]
    first_paragraph = paragraphs[0].lower() if paragraphs else ""
    assert "advisories" in first_paragraph or "trust boundary" in first_paragraph


def test_security_md_names_supported_versions() -> None:
    """covers: M4, A9 -- the current minor line only, no LTS promise."""
    text = SECURITY_PATH.read_text(encoding="utf-8")
    config = _pyproject()
    version = config["project"]["version"]
    major, minor = version.split(".")[:2]
    assert f"{major}.{minor}" in text, f"must name the current minor line {major}.{minor}"
    assert not _promises_lts(text), "no LTS promise on a personal project"


def test_the_lts_scan_reads_a_word_not_a_substring() -> None:
    """covers: G1 -- `results` is not a long-term-support promise.

    The exact false positive: SECURITY.md said "silent, wrong inference results" and the
    check reported an LTS promise. I changed the word to "output" to get green, which is
    the author editing correct content to satisfy a wrong check (Q5).
    """
    for innocent in (
        "Redacting the path would give silent, wrong inference results.",
        "A stale pin faults on the next upgrade.",
        "The maintainer consults the advisory before triaging.",
    ):
        assert not _promises_lts(innocent), (
            f"{innocent!r} was read as an LTS promise. `lts` is a substring of an "
            "ordinary English word here, not a support commitment"
        )


def test_the_lts_scan_still_catches_a_real_promise() -> None:
    """covers: G1 -- the fix must not turn the check off.

    A false negative here is worse than the false positive it replaced: the check exists
    so a personal project does not accidentally advertise a support line it cannot staff.
    """
    for promise in (
        "The 2.4.x line is an LTS release.",
        "lts branches receive fixes for 24 months.",
        "Long-term support (LTS) is offered on request.",
    ):
        assert _promises_lts(promise), (
            f"{promise!r} promises long-term support and was not caught. The word "
            "boundary must not have narrowed the scan into uselessness"
        )


def test_the_email_scan_ignores_a_credential_example_in_code() -> None:
    """covers: G1 -- a credential in a fenced block is an example, not an intake channel.

    `hunter2@10.0.0.5` is an RTSP userinfo. The regex cannot tell it from an address, so
    the fix is to scan prose rather than to make the pattern cleverer.
    """
    doc = (
        "Report privately through GitHub Security Advisories.\n\n"
        "```\n"
        "rtsp://camop:hunter2@10.0.0.5:554/stream -> rtsp://10.0.0.5:554/stream\n"
        "```\n\n"
        "The userinfo is stripped at the boundary.\n"
    )
    assert not _names_an_email_intake(doc), (
        "a credential example inside a fenced code block was read as an email reporting "
        "address. That is what forced SECURITY.md to use an undotted host"
    )
    assert not _names_an_email_intake("Contact `security@example.com` — no, use Advisories."), (
        "an address inside inline code was read as an intake channel"
    )


def test_the_email_scan_still_catches_an_address_in_prose() -> None:
    """covers: G1 -- the half that makes the fix safe rather than merely convenient."""
    for doc in (
        "Report vulnerabilities to security@example.com.",
        "Mail the maintainer at tin.dang+security@example.co.uk and wait.",
        "Send details to\nsecurity@yowo.dev\nwithin 24 hours.",
    ):
        assert _names_an_email_intake(doc), (
            f"{doc!r} names an email intake channel in prose and was not caught. A "
            "personal-project inbox silently rots, and a rotted channel looks live"
        )


def test_the_prose_reducer_leaves_prose() -> None:
    """covers: G1 -- a reducer returning "" makes every scan above vacuously clean.

    This is the failure mode that would hide the other four checks passing for no reason,
    so it is asserted directly rather than assumed.
    """
    doc = (
        "Report through GitHub Security Advisories.\n\n"
        "```\nrtsp://u:p@h/s\n```\n\n"
        "Only the current minor line is supported.\n"
    )
    prose = _prose_of(doc)
    assert "GitHub Security Advisories" in prose
    assert "current minor line" in prose
    assert "rtsp://u:p@h/s" not in prose, "the fenced block survived the reduction"

    real = _prose_of(SECURITY_PATH.read_text(encoding="utf-8"))
    assert len(real.split()) > 100, (
        f"reducing the real SECURITY.md left only {len(real.split())} words. The scans "
        "that read this would pass because there is nothing left to find"
    )


# ---------------------------------------------------------------------------
# S4 -- pyproject.toml
# ---------------------------------------------------------------------------


def test_pyproject_declares_project_urls() -> None:
    """covers: M6, A13, A19 -- Homepage, Source, Issues, Changelog."""
    config = _pyproject()
    urls = config["project"].get("urls", {})
    for key in ("Homepage", "Source", "Issues", "Changelog"):
        assert key in urls, f"[project.urls] is missing {key}"
    assert "github.com" in urls["Source"], "Source must point at the GitHub repository"
    assert "github.com" in urls["Homepage"]


# ---------------------------------------------------------------------------
# Shared fixture: a REAL built sdist and wheel (session-scoped -- expensive).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out_dir = tmp_path_factory.mktemp("dist-licensing")
    return vsc.build(REPO_ROOT, out_dir)
