#!/usr/bin/env python3
"""Assert that `main`'s branch protection actually gates merges.

This cannot be a pytest case in the unit suite: it needs an authenticated admin
token and network access, so in CI it could only ever `skipif` green — which is
not proof. `add run` parses JUnit XML regardless of what produced it, so this
script earns the same bound receipt by comparing live API state against the
frozen RULES of ADD task `pr-ci-gate` (M5).

Usage:
    python3 scripts/verify_branch_protection.py [--ref SHA] [--junitxml PATH]

Exits 0 only when every assertion holds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = "TinDang97/yowo"
BRANCH = "main"
# All four of ci.yml's job names, not just the quality gate. Amended 2026-09-10:
# until then only "Quality Gate" was required, so the three checks m1 built could
# each fail and a pull request still merged.
#
# READ from docs/branch-protection.json, never copied. This was a hardcoded
# four-entry tuple whose comment asked the reader to "diff the two by eye" —
# and it drifted: the payload required eight while the verifier asserted four
# and reported "requires all 4 checks" as a success, so dropping the other
# four would have verified green.
PAYLOAD = Path(__file__).resolve().parents[1] / "docs" / "branch-protection.json"


def required_contexts() -> tuple[str, ...]:
    """The contexts the applied payload binds."""
    data = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    return tuple(data["required_status_checks"]["contexts"])


def missing_contexts(configured: list[str]) -> list[str]:
    """Which required contexts the live configuration does not bind."""
    return [c for c in required_contexts() if c not in configured]


def _gh(path: str) -> tuple[int, str]:
    proc = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=30)
    return proc.returncode, (proc.stdout or proc.stderr)


def _head_sha(ref: str | None) -> str:
    if ref:
        return ref
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    return proc.stdout.strip()


def _context_was_observed(sha: str) -> tuple[bool, str]:
    """Has GitHub actually REPORTED the required context against this commit?

    Protection accepts a required-check context GitHub has never seen. The API call
    succeeds, the branch looks guarded, and every subsequent merge blocks forever on
    a check that will never report (task pr-ci-gate, E3). Comparing the configured
    context string against the workflow's job name cannot catch this — only asking
    what GitHub has actually observed can.
    """
    code, out = _gh(f"repos/{REPO}/commits/{sha}/check-runs")
    if code != 0:
        return False, f"could not read check-runs for {sha[:8]}: {out.strip().splitlines()[0]}"
    try:
        names = [r.get("name") for r in json.loads(out).get("check_runs", [])]
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        return False, f"check-runs response was not JSON: {exc}"
    return explain_observation(names, sha)


def explain_observation(names: list[str], sha: str) -> tuple[bool, str]:
    """Read a commit's check-run names into a verdict, and an accurate reason.

    `ci.yml` runs on `pull_request`, so a commit on `main` never carries its
    checks — only `release.yml`'s. Run against main, which is exactly what the
    doc's copy-pasteable command does, this reported that protection would
    "block every merge on a check that never reports". That diagnosis is
    false: the check reports fine, on pull requests, which is where it is
    required. Say which commit to ask about instead of blaming the config.
    """
    probe = required_contexts()[0]
    if probe in names:
        return True, ""
    if not any(n in names for n in required_contexts()):
        return False, (
            f"{sha[:8]} carries none of the required checks (observed: "
            f"{names or 'none'}). ci.yml runs on `pull request`, so a commit on "
            f"main carries only release.yml's checks and observation cannot be "
            f"confirmed from here. Re-run against a pull-request head: "
            f"`--ref <sha>`."
        )
    return False, (
        f"GitHub has never reported a check named {probe!r} on {sha[:8]} "
        f"(observed: {names or 'none'}) — protection requiring it would block every "
        "merge on a check that never reports"
    )


def check(ref: str | None = None) -> list[str]:
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
    missing = missing_contexts(contexts)
    if missing:
        failures.append(
            f"required status checks {contexts!r} are missing {missing!r} — "
            "a merge is not gated on every check ci.yml publishes"
        )

    if not (data.get("enforce_admins") or {}).get("enabled"):
        failures.append(
            "enforce_admins is not enabled — the repository owner can merge past a "
            "failing check, which makes the gate advisory (task pr-ci-gate, A2)"
        )

    observed, why = _context_was_observed(_head_sha(ref))
    if not observed:
        failures.append(why)

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
    parser.add_argument("--ref", help="commit to check observation against (default: local HEAD)")
    args = parser.parse_args()

    started = time.monotonic()
    failures = check(args.ref)
    elapsed = time.monotonic() - started

    if args.junitxml:
        write_junit(args.junitxml, failures, elapsed)

    for line in failures:
        print(f"FAIL: {line}", file=sys.stderr)
    if not failures:
        print(
            f"OK: {BRANCH} requires all {len(required_contexts())} checks "
            f"({', '.join(required_contexts())}) and enforces them on admins"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
