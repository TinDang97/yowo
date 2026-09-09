"""Nobody acquires an AGPL artifact through yowo without being told.

Red-first for ADD task `licensing-provenance`.

Verified facts this suite holds the docs to:
  pyproject.toml:6      license = "Apache-2.0"
  every weight URL      -> ultralytics/assets, which GitHub reports as AGPL-3.0
  README                mentions AGPL zero times

These checks assert DISCLOSURE, not compliance. Whether a given deployment
incurs AGPL obligations is a legal question this project does not answer, and
`test_disclosure_asserts_no_legal_conclusion` exists to keep it that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

from yowo.models import list_available

REPO_ROOT = Path(__file__).parent.parent.parent
README = REPO_ROOT / "README.md"
SECURITY = REPO_ROOT / "SECURITY.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_readme_discloses_the_weight_licence() -> None:
    """covers: M1, A2 — a reader must be able to check the claim themselves."""
    text = README.read_text()
    assert "AGPL-3.0" in text, "README never mentions the weights' licence"
    assert "ultralytics/assets" in text, "README does not say where the weights come from"
    assert "Apache-2.0" in text, "README does not state yowo's own licence"


def test_disclosure_asserts_no_legal_conclusion() -> None:
    """covers: M2, R:IMPLIEDADVICE, A6 — false confidence beats silence only if true."""
    text = README.read_text().lower()
    section = text[text.index("agpl-3.0") - 2000 : text.index("agpl-3.0") + 2000]
    for phrase in (
        "you are required to",
        "you must open-source",
        "this is permitted",
        "you do not need to",
        "no obligations",
    ):
        assert phrase not in section, (
            f"the disclosure states a conclusion about the reader's obligations: {phrase!r}"
        )


def test_disclosure_scopes_itself_to_builtin_defaults() -> None:
    """covers: A3, A4, E1 — a user's own weights are not acquired from us."""
    text = README.read_text()
    idx = text.index("AGPL-3.0")
    section = text[max(0, idx - 2000) : idx + 2000]
    assert "--weights" in section or "your own weights" in section.lower(), (
        "the disclosure must say it does not cover user-supplied weights"
    )


def test_security_md_exists_with_a_private_report_route() -> None:
    """covers: M3, A7, A12 — 'open an issue' publishes the vulnerability."""
    assert SECURITY.exists(), "no SECURITY.md"
    text = SECURITY.read_text()
    assert "private" in text.lower() or "security advisor" in text.lower(), (
        "no private reporting route"
    )
    head = text[:600].lower()
    assert "report" in head, "the contact route must be near the top, not buried"


def test_security_md_scopes_itself_and_promises_no_unmeetable_sla() -> None:
    """covers: A8, A9, A10 — a missed published SLA is worse than none."""
    text = SECURITY.read_text().lower()
    assert "scope" in text, "SECURITY.md does not state what is in scope"
    assert "ultralytics" in text or "weights" in text, (
        "SECURITY.md must say upstream weights are not in scope"
    )
    assert not re.search(r"within \d+ (hours|business days) .*(fix|patch|resolv)", text), (
        "a remediation SLA is promised that an unfunded project cannot meet"
    )


def test_project_urls_are_declared_and_well_formed() -> None:
    """covers: M4, A14, A16 — asserts shape, not resolution: no network in a unit test."""
    urls = _pyproject()["project"].get("urls", {})
    for key in ("Homepage", "Repository", "Issues", "Changelog"):
        assert key in urls, f"[project.urls] is missing {key}"
        assert urls[key].startswith("https://github.com/TinDang97/yowo"), (
            f"{key} does not point at this repository: {urls[key]}"
        )


def test_disclosure_matches_the_registry_urls() -> None:
    """covers: M5, E2 — bound to the real URLs so it cannot go stale silently."""
    text = README.read_text()
    hosts = {m.default_weights_url.split("/releases/")[0] for m in list_available()}
    assert len(hosts) == 1, f"weights now come from several sources: {hosts}"
    owner_repo = "/".join(hosts.pop().split("/")[-2:])
    assert owner_repo in text, (
        f"the disclosure names a source the registry no longer uses; registry says {owner_repo}"
    )
