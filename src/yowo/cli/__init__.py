"""CLI package for yowo.

Entry point registered in pyproject.toml::

    [project.scripts]
    yowo = "yowo.cli._main:cli"
"""

from yowo.cli._main import cli

__all__ = ["cli"]
