"""Root pytest configuration.

`pytest_addoption` is honoured only in the rootdir conftest, which is why the
parity scope lives here rather than beside the suite that reads it.
"""

from __future__ import annotations

import pytest

#: The two entry points m3 box 3 asks for. The scope selects WHICH variants
#: run; it must never select which assertions run, or the scheduled run and the
#: pull-request run become two different checks wearing one name.
PARITY_SCOPES = ("pr", "all")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--parity-scope",
        action="store",
        default="pr",
        choices=PARITY_SCOPES,
        help=(
            "which variants the export-parity suite compares: 'pr' for the four "
            "n/s variants run on every pull request, 'all' for all ten run on a "
            "schedule. Both run the same assertions."
        ),
    )
