#!/usr/bin/env python3
"""Two builds of the same commit must produce byte-identical artifacts.

This is how a published tarball is shown to match the source it claims. It does
not MAKE the build reproducible — a probe during Direction found hatchling
already honours SOURCE_DATE_EPOCH — it locks that property in so a future change
to the build cannot quietly take it away.

The comparison covers every member of the artifact. There is deliberately no
exclusion list: the tempting way to make a digest mismatch disappear is to stop
comparing the file that differs, and that is the failure this exists to catch.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

# A fixed constant, not a commit timestamp: the digest must be a property of the
# SOURCE, so two people building on different days get the same answer (A2).
SOURCE_DATE_EPOCH = "1700000000"


class ArtifactError(RuntimeError):
    """A build produced no artifact, or its output cannot be read."""


def _build_once(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot create {out_dir!r}: {exc}") from exc

    # The epoch is set here, not read from the environment: a developer whose
    # shell has it unset must still get CI's answer (A4).
    env = {**os.environ, "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH}
    proc = subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise ArtifactError(f"uv build failed ({proc.returncode}):\n{proc.stderr}")

    sdists = sorted(out_dir.glob("*.tar.gz"))
    wheels = sorted(out_dir.glob("*.whl"))
    # Absence is a failure, never "no difference found" (E1, A10).
    if not sdists or not wheels:
        raise ArtifactError(f"build produced no sdist and/or wheel in {out_dir}")
    return sdists[-1], wheels[-1]


def build_twice(repo_root: Path, out_root: Path) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """Build into two SEPARATE directories and return both (sdist, wheel) pairs.

    Separate directories, never an in-place rebuild: a second build into the same
    directory can overwrite the first and trivially "match" (A9).
    """
    return (
        _build_once(repo_root, out_root / "a"),
        _build_once(repo_root, out_root / "b"),
    )


def digest(path: Path) -> str:
    """sha256 of an artifact, read whole."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differing_members(a: Path, b: Path) -> list[str]:
    """Members whose name, mtime, mode, uid, gid or size differ between two sdists.

    Reported so a mismatch names what moved, rather than leaving the reader to
    unpack two tarballs by hand (A11).
    """

    def facts(p: Path) -> dict[str, tuple]:
        with tarfile.open(p, "r:gz") as tar:
            return {
                m.name: (m.mtime, m.mode, m.uid, m.gid, m.size, m.type) for m in tar.getmembers()
            }

    fa, fb = facts(a), facts(b)
    out = [f"{n}: only in {a.name}" for n in sorted(set(fa) - set(fb))]
    out += [f"{n}: only in {b.name}" for n in sorted(set(fb) - set(fa))]
    out += [f"{n}: {fa[n]} != {fb[n]}" for n in sorted(set(fa) & set(fb)) if fa[n] != fb[n]]
    return out


def _write_junit(path: Path, results: list[tuple[str, str | None]]) -> None:
    failures = sum(1 for _, msg in results if msg)
    suite = ET.Element(
        "testsuite",
        name="verify_reproducible_build",
        tests=str(len(results)),
        failures=str(failures),
        errors="0",
        skipped="0",
    )
    for name, message in results:
        case = ET.SubElement(suite, "testcase", classname="verify_reproducible_build", name=name)
        if message:
            ET.SubElement(case, "failure", message=message).text = message
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument("--junitxml", type=Path, default=None)
    args = parser.parse_args(argv)

    results: list[tuple[str, str | None]] = []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            (sa, wa), (sb, wb) = build_twice(args.repo_root, Path(tmp))
        except ArtifactError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            if args.junitxml:
                _write_junit(args.junitxml, [("both_builds_succeed", str(exc))])
            return 2
        results.append(("both_builds_succeed", None))

        for label, x, y in (("sdist", sa, sb), ("wheel", wa, wb)):
            dx, dy = digest(x), digest(y)
            if dx == dy:
                print(f"OK: {label} reproducible — sha256 {dx}")
                results.append((f"{label}_is_reproducible", None))
            else:
                detail = f"{label} differs: {dx} != {dy}"
                if label == "sdist":
                    detail += "\n" + "\n".join(differing_members(x, y))
                print(f"FAIL: {detail}", file=sys.stderr)
                results.append((f"{label}_is_reproducible", detail))

    if args.junitxml:
        _write_junit(args.junitxml, results)
    return 1 if any(m for _, m in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
