#!/usr/bin/env python3
"""Assert the checkpoint loader still extracts exactly the same tensors.

This is a REGRESSION FLOOR, not a red-first check: it passes before the build by
construction, and its whole job is to fail if the loader refactor changes which
tensors come out. The reference in tests/fixtures/checkpoint_reference.json was
captured from the PRE-change loader (torch.load weights_only=False).

It is a script rather than a unit test because it needs a real multi-megabyte
checkpoint, which CI cannot obtain until `ci-weight-fixture` lands — and a
skipif-green unit test would prove nothing (task checkpoint-loader, M3).

Usage:
    python3 scripts/verify_checkpoint_equivalence.py yolo11n.pt [--junitxml PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

REFERENCE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "checkpoint_reference.json"


def compare(checkpoint: Path) -> list[str]:
    from yowo.arch._weights import _extract_state_dict

    ref = json.loads(REFERENCE.read_text())

    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_sha != ref["source_sha256"]:
        return [
            f"{checkpoint.name} is not the checkpoint the reference was captured from "
            f"(sha256 {actual_sha[:16]}… != {ref['source_sha256'][:16]}…) — comparing it "
            "would prove nothing about the refactor"
        ]

    sd = _extract_state_dict(checkpoint)
    expected: dict[str, dict[str, object]] = ref["tensors"]
    failures: list[str] = []

    missing = sorted(set(expected) - set(sd))
    added = sorted(set(sd) - set(expected))
    if missing:
        failures.append(f"{len(missing)} tensor(s) no longer extracted, e.g. {missing[:3]}")
    if added:
        failures.append(f"{len(added)} unexpected tensor(s), e.g. {added[:3]}")

    for key in sorted(set(expected) & set(sd)):
        want, got = expected[key], sd[key]
        if str(got.dtype) != want["dtype"]:
            failures.append(f"{key}: dtype {got.dtype} != {want['dtype']}")
            continue
        if list(got.shape) != want["shape"]:
            failures.append(f"{key}: shape {list(got.shape)} != {want['shape']}")
            continue
        digest = hashlib.sha256(got.detach().cpu().numpy().tobytes()).hexdigest()
        if digest != want["sha256"]:
            failures.append(f"{key}: tensor bytes changed")

    return failures[:20]


def write_junit(path: str, failures: list[str], elapsed: float) -> None:
    suite = ET.Element(
        "testsuite",
        name="checkpoint_equivalence",
        tests="1",
        failures=str(1 if failures else 0),
        time=f"{elapsed:.3f}",
    )
    case = ET.SubElement(
        suite,
        "testcase",
        classname="scripts.verify_checkpoint_equivalence",
        name="verify_checkpoint_equivalence",
        time=f"{elapsed:.3f}",
    )
    if failures:
        ET.SubElement(case, "failure", message=failures[0]).text = "\n".join(failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--junitxml")
    args = parser.parse_args()

    started = time.monotonic()
    failures = compare(args.checkpoint)
    elapsed = time.monotonic() - started

    if args.junitxml:
        write_junit(args.junitxml, failures, elapsed)
    for line in failures:
        print(f"FAIL: {line}", file=sys.stderr)
    if not failures:
        n = len(json.loads(REFERENCE.read_text())["tensors"])
        print(f"OK: all {n} tensors identical to the pre-change loader")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
