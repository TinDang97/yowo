#!/usr/bin/env python3
"""Assert that `main`'s branch protection actually gates merges.

This cannot be a pytest case in the unit suite: it needs an authenticated admin
token and network access, so in CI it could only ever `skipif` green — which is
not proof. `add run` parses JUnit XML regardless of what produced it, so this
script earns the same bound receipt by comparing live API state against the
frozen RULES of ADD task `pr-ci-gate` (M5).

Usage:
    python3 scripts/verify_branch_protection.py [--junitxml PATH]

Exits 0 only when every assertion holds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from xml.etree import ElementTree as ET

REPO = "TinDang97/yowo"
BRANCH = "main"
REQUIRED_CONTEXT = "Quality Gate"


def _gh(path: str) -> tuple[int, str]:
    proc = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=30)
    return proc.returncode, (proc.stdout or proc.stderr)


def check() -> list[str]:
    """Return a list of failure messages; empty means the protection is correct."""
    code, out = _gh(f"repos/{REPO}/branches/{BRANCH}/protection")
    if code != 0:
        return [f"branch protection is not configured on {BRANCH}: {out.strip().splitlines()[0]}"]

    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        return [f"protection response was not JSON: {exc}"]

    failures: list[str] = []

    contexts = (data.get("required_status_checks") or {}).get("contexts") or []
    if REQUIRED_CONTEXT not in contexts:
        failures.append(
            f"required status checks {contexts!r} do not include {REQUIRED_CONTEXT!r} — "
            "a merge is not gated on the quality gate"
        )

    if not (data.get("enforce_admins") or {}).get("enabled"):
        failures.append(
            "enforce_admins is not enabled — the repository owner can merge past a "
            "failing check, which makes the gate advisory (task pr-ci-gate, A2)"
        )

    return failures


def write_junit(path: str, failures: list[str], elapsed: float) -> None:
    suite = ET.Element(
        "testsuite",
        name="branch_protection",
        tests="1",
        failures=str(1 if failures else 0),
        time=f"{elapsed:.3f}",
    )
    case = ET.SubElement(
        suite,
        "testcase",
        classname="scripts.verify_branch_protection",
        name="verify_branch_protection",
        time=f"{elapsed:.3f}",
    )
    if failures:
        ET.SubElement(case, "failure", message=failures[0]).text = "\n".join(failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junitxml", help="write a JUnit XML report here")
    args = parser.parse_args()

    started = time.monotonic()
    failures = check()
    elapsed = time.monotonic() - started

    if args.junitxml:
        write_junit(args.junitxml, failures, elapsed)

    for line in failures:
        print(f"FAIL: {line}", file=sys.stderr)
    if not failures:
        print(f"OK: {BRANCH} requires {REQUIRED_CONTEXT!r} and enforces it on admins")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
