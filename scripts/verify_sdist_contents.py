#!/usr/bin/env python3
"""Assert that the built distributions contain what we chose to publish.

yowo 2.4.0 and 2.4.1 both shipped a 5.6 MB AGPL-licensed ``yolo11n.pt`` to PyPI
under an Apache-2.0 declaration. The cause was structural: ``pyproject.toml``
configured only the wheel target, so hatchling fell back to a whole-repo sweep
filtered solely by ``.gitignore`` — and an untracked, unignored file is neither
tracked nor excluded. This verifier asserts the artifact itself, not the config,
because the config is what was wrong.

Every violation found in one run is reported, sorted by path (A17), so a fix does
not turn into a rerun loop. Each assertion is emitted as its own JUnit
``<testcase>`` (A18) so an ``add run`` receipt binds them individually.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# The allowlist (A2). PKG-INFO is synthesised by the build backend, and hatchling
# force-ships the VCS ignore file whatever the selection says (verified by probe).
# Both are admitted but never required of the source tree.
# `tests` is deliberately absent: the sdist is a build input, not a test bundle (A6,
# reversed by human decision). Adding it back is a one-line change to `only-include`.
# `NOTICE` was added by `licensing-provenance`: Apache-2.0 4(d) requires a
# redistribution to carry it, so it is REQUIRED, not merely allowed -- named
# here rather than the allowlist being relaxed to admit it.
REQUIRED_ENTRIES = frozenset(
    {"src", "README.md", "LICENSE", "NOTICE", "CHANGELOG.md", "pyproject.toml"}
)
GENERATED_ENTRIES = frozenset({"PKG-INFO", ".gitignore"})
ALLOWED_ENTRIES = REQUIRED_ENTRIES | GENERATED_ENTRIES

# R:BINARY — no model weight, checkpoint or exported artifact, anywhere in either
# distribution including the package source (A15).
FORBIDDEN_SUFFIXES = (
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".mlpackage",
    ".tflite",
    ".bin",
)


class ArtifactError(RuntimeError):
    """A distribution that should exist does not, or cannot be read."""


def find_artifacts(out_dir: Path) -> tuple[Path, Path]:
    """Locate the built sdist and wheel.

    Raises rather than returning empties: a build that produced nothing must never
    read as an artifact with no violations (A16).
    """
    sdists = sorted(out_dir.glob("*.tar.gz"))
    wheels = sorted(out_dir.glob("*.whl"))
    if not sdists:
        raise ArtifactError(f"no sdist (*.tar.gz) was produced in {out_dir}")
    if not wheels:
        raise ArtifactError(f"no wheel (*.whl) was produced in {out_dir}")
    return sdists[-1], wheels[-1]


def build(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build both distributions from ``repo_root`` into ``out_dir``."""
    proc = subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ArtifactError(f"uv build failed ({proc.returncode}):\n{proc.stderr}")
    return find_artifacts(out_dir)


def sdist_members(sdist: Path) -> list[str]:
    """Member paths inside an sdist, relative to its single root directory."""
    with tarfile.open(sdist, "r:gz") as tar:
        names = tar.getnames()
    relative: list[str] = []
    for name in names:
        _, _, rest = name.partition("/")
        if rest:
            relative.append(rest)
    return relative


def wheel_members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def top_level_entries(members: list[str]) -> set[str]:
    return {m.split("/", 1)[0] for m in members if m}


def _package_payload_from_sdist(members: list[str]) -> set[str]:
    prefix = "src/yowo/"
    return {m[len("src/") :] for m in members if m.startswith(prefix)}


def _package_payload_from_wheel(members: list[str]) -> set[str]:
    return {m for m in members if m.startswith("yowo/") and not m.endswith("/")}


def violations(sdist: Path, wheel: Path) -> list[str]:
    """Every way these artifacts breach the rules, sorted by path."""
    found: list[str] = []

    s_members = sdist_members(sdist)
    w_members = wheel_members(wheel)
    entries = top_level_entries(s_members)

    # M1 — the allowlist holds in both directions.
    for extra in sorted(entries - ALLOWED_ENTRIES):
        found.append(f"{extra}: not in the sdist allowlist ({sorted(ALLOWED_ENTRIES)})")
    # A4 — a listed-but-missing path is silently skipped by hatchling; catch it here.
    for missing in sorted(REQUIRED_ENTRIES - entries):
        found.append(f"{missing}: required by the allowlist but absent from the sdist")

    # R:BINARY — scan the whole of both artifacts, package source included.
    for label, members in (("sdist", s_members), ("wheel", w_members)):
        for member in sorted(members):
            if member.endswith(FORBIDDEN_SUFFIXES):
                found.append(f"{member}: forbidden binary artifact in the {label}")

    # M2 — the two distributions must agree on what the package is.
    sdist_payload = _package_payload_from_sdist(s_members)
    wheel_payload = _package_payload_from_wheel(w_members)
    for only_sdist in sorted(sdist_payload - wheel_payload):
        found.append(f"{only_sdist}: in the sdist payload but not the wheel")
    for only_wheel in sorted(wheel_payload - sdist_payload):
        found.append(f"{only_wheel}: in the wheel payload but not the sdist")

    return sorted(found)


def _write_junit(path: Path, results: list[tuple[str, str | None]]) -> None:
    failures = sum(1 for _, msg in results if msg)
    suite = ET.Element(
        "testsuite",
        name="verify_sdist_contents",
        tests=str(len(results)),
        failures=str(failures),
        errors="0",
        skipped="0",
    )
    for name, message in results:
        case = ET.SubElement(suite, "testcase", classname="verify_sdist_contents", name=name)
        if message:
            ET.SubElement(case, "failure", message=message).text = message
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--junitxml", type=Path, default=None)
    args = parser.parse_args(argv)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.out_dir or Path(tmp)
        try:
            sdist, wheel = build(args.repo_root, out_dir)
        except ArtifactError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            if args.junitxml:
                _write_junit(args.junitxml, [("build_produces_both_artifacts", str(exc))])
            return 2
        found = violations(sdist, wheel)

    # A12 — name the offending path and the rule it breached, not a count.
    for line in found:
        print(f"FAIL: {line}", file=sys.stderr)
    if args.junitxml:
        _write_junit(
            args.junitxml,
            [
                ("build_produces_both_artifacts", None),
                (
                    "distributions_contain_only_what_we_publish",
                    "\n".join(found) if found else None,
                ),
            ],
        )
    if found:
        print(
            f"\n{len(found)} violation(s). The sdist allowlist lives in "
            f"[tool.hatch.build.targets.sdist] in pyproject.toml.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {sdist.name} and {wheel.name} contain only what we publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
