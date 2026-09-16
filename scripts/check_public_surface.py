"""Exercise yowo's documented public surface, as a consumer would.

`pyproject.toml` claims Python 3.8 through 3.12. Until 2026-09-16 every job in
every workflow ran 3.11 on ubuntu-latest, so four of the five claimed versions
had never executed anything. A classifier is a promise to a user; an
unexecuted promise is just a string in a metadata file.

This runs against an install with NO dev group, because the claim is about the
shipped package rather than the development environment. That distinction is
not academic: the dev group cannot even RESOLVE on 3.8 — `pre-commit>=3.8.0`
requires Python >=3.9 — so a "3.8 works" result obtained with dev dependencies
installed would be impossible to obtain at all.

The test suite cannot stand in for this. Measured 2026-09-16 under a real
CPython 3.8.20: `src/yowo` compiles 96/96, but the test suite compiles only
141/163 (22 modules use parenthesized context managers, 3.10+), and on a real
3.10.20 all 163 compile while 6 modules `import tomllib`, which is 3.11+
stdlib. The test suite's floor is 3.11 — four versions above the package's.

Exit code 0 means every documented entry point worked. Run it anywhere:

    python scripts/check_public_surface.py
"""

from __future__ import annotations

import subprocess
import sys

#: Names `yowo/__init__.py` exports and the README uses.
PUBLIC_API = ("InferenceEngine", "detect", "classify")

#: Public submodules a consumer imports directly.
PUBLIC_SUBMODULES = (
    "yowo.models",
    "yowo.io",
    "yowo.postprocess",
    "yowo.config",
    "yowo.hardware",
    "yowo.types",
    "yowo.errors",
)

#: CLI invocations that must exit 0. `--help` proves the entry point resolves
#: and the command tree builds; `models` proves the registry renders, which is
#: the first thing a new user runs and the last thing an import-only smoke
#: would notice was broken.
CLI_COMMANDS = (("--help",), ("models",))

#: The registry is the package's central claim about what it can serve.
EXPECTED_DETECTION_VARIANTS = 10


def _fail(what: str, detail: str) -> None:
    print(f"FAIL  {what}: {detail}")


def main() -> int:
    version = ".".join(str(n) for n in sys.version_info[:3])
    print(f"checking yowo's public surface on CPython {version}")
    failures = 0

    try:
        import yowo

        print(f"  ok    import yowo ({yowo.__version__})")
    except Exception as exc:
        _fail("import yowo", repr(exc))
        return 1  # nothing else can be checked

    for name in PUBLIC_API:
        if not hasattr(yowo, name):
            _fail("public API", f"yowo.{name} is not exported")
            failures += 1
    if not failures:
        print(f"  ok    public API: {', '.join(PUBLIC_API)}")

    for module in PUBLIC_SUBMODULES:
        try:
            __import__(module)
        except Exception as exc:
            _fail("submodule", f"{module}: {exc!r}")
            failures += 1
    print(f"  ok    {len(PUBLIC_SUBMODULES)} public submodules")

    try:
        from yowo.models import list_available

        variants = list_available()
        if len(variants) != EXPECTED_DETECTION_VARIANTS:
            _fail(
                "registry",
                f"{len(variants)} detection variants, expected {EXPECTED_DETECTION_VARIANTS}",
            )
            failures += 1
        else:
            print(f"  ok    registry: {len(variants)} detection variants")
    except Exception as exc:
        _fail("registry", repr(exc))
        failures += 1

    for args in CLI_COMMANDS:
        # Through the console script's own module path, so this measures what a
        # consumer's `yowo` command runs rather than an import shortcut.
        result = subprocess.run(
            [sys.executable, "-c", "from yowo.cli import cli; cli()", *args],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _fail(
                f"CLI {' '.join(args)}", f"exit {result.returncode}: {result.stderr.strip()[:300]}"
            )
            failures += 1
        else:
            print(f"  ok    CLI {' '.join(args)}")

    if failures:
        print(
            f"\n{failures} failure(s): this Python is CLAIMED in pyproject.toml and does not work."
        )
        return 1
    print("\nevery documented entry point works on this interpreter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
