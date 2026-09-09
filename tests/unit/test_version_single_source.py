"""One place declares the version; every other surface derives it.

Red-first for ADD task `version-single-source`.

These fail against the current tree: `src/yowo/__init__.py` declares
`__version__ = "2.4.1"` as a literal while pyproject.toml, distribution
metadata and the CLI all say 2.5.0 — and `_exporter.py` stamps export
sidecars from the stale constant.
"""

from __future__ import annotations

import re
from importlib.metadata import version as dist_version
from pathlib import Path
from unittest import mock

import tomllib

import yowo

REPO_ROOT = Path(__file__).parent.parent.parent
INIT_PY = REPO_ROOT / "src/yowo/__init__.py"

_VERSION_LITERAL = re.compile(r'^__version__\s*=\s*["\']\d+\.\d+', re.M)


def test_version_is_derived_not_declared() -> None:
    """covers: G1, R:LITERAL, A14 — a second declaration is a second writer."""
    assert not _VERSION_LITERAL.search(INIT_PY.read_text()), (
        "__version__ is a hardcoded literal; it must be derived from distribution metadata"
    )


def test_all_surfaces_report_the_same_version() -> None:
    """covers: G2, A2 — the drift that shipped."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]
    assert yowo.__version__ == declared, (
        f"yowo.__version__ is {yowo.__version__!r}, pyproject declares {declared!r}"
    )
    assert yowo.__version__ == dist_version("yowo"), (
        f"yowo.__version__ is {yowo.__version__!r}, "
        f"distribution metadata says {dist_version('yowo')!r}"
    )


def test_export_sidecar_stamps_the_package_version() -> None:
    """covers: G2, A8 — every exported artifact carries this string as provenance."""
    source = (REPO_ROOT / "src/yowo/export/_exporter.py").read_text()
    assert 'yowo_version=getattr(yowo, "__version__"' in source, (
        "the sidecar must stamp the package version, not a literal"
    )
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert yowo.__version__ == pyproject["project"]["version"], (
        "the sidecar would stamp a version that does not match the package"
    )


def test_uninstalled_package_yields_the_sentinel() -> None:
    """covers: E1, A4 — a bare checkout must not explode, and must not lie."""
    from importlib.metadata import PackageNotFoundError

    resolve = getattr(yowo, "_resolve_version", None)
    assert resolve is not None, "no _resolve_version() to exercise"
    with mock.patch("yowo.metadata_version", side_effect=PackageNotFoundError):
        assert resolve() == "0.0.0+unknown"


def test_only_pyproject_declares_a_version() -> None:
    """covers: G3, A14 — two bump targets is the bug, restated."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    sr = pyproject["tool"]["semantic_release"]
    assert sr.get("version_toml") == ["pyproject.toml:project.version"]
    assert "version_variables" not in sr, "a second bump target re-opens the drift this task closes"
