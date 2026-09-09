#!/usr/bin/env python3
"""Assert every digest-pinned variant still loads through the ENUMERATED allowlist.

This is `test_every_pinned_variant_still_loads` — the integration-tier check of
ADD task `narrow-loader-allowlist` (M5, E4). `_ALLOWED_TORCH` was measured from
these ten checkpoints; this proves the narrowing did not cut what it measured.

It is a script rather than a unit test for the reason
`verify_checkpoint_equivalence.py` is: it needs the real corpus (~440 MB), and a
`skipif`-green unit test would prove nothing. Every checkpoint comes through
`resolve_weights()`, so every byte read was verified against its pin. It fails
closed: a variant that cannot be fetched is a FAIL, never a skip.

Two things are asserted per variant:
  1. `_extract_state_dict` — the restricted reader on the RAW file. This is the
     trust boundary under test. It is called directly because the public
     loaders serve a converted sidecar when one is cached, which would let a
     stale sidecar make a refused checkpoint look loadable.
  2. `load_weights` into a freshly built model — the tensors still fit the
     architecture, so "loads" means loads, not "unpacks into something".

Usage:
    python3 scripts/verify_every_pinned_variant_loads.py [--junitxml PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from xml.etree import ElementTree as ET

from yowo.models._registry import ModelMeta, list_available
from yowo.models._weights import resolve_weights
from yowo.types import ModelSpec


def check_variant(meta: ModelMeta) -> tuple[str, str | None]:
    """Return (stem, failure) — failure is None when the variant loads."""
    from yowo.arch import build_model
    from yowo.arch._weights import _extract_state_dict, load_weights

    if meta.sha256 is None:
        return meta.weight_stem, "unpinned; the corpus must be digest-verified"
    try:
        path = resolve_weights(ModelSpec(family=meta.family, size=meta.size))
    except Exception as exc:
        return meta.weight_stem, f"could not fetch or verify: {type(exc).__name__}: {exc}"
    try:
        state = _extract_state_dict(path)
    except Exception as exc:
        return meta.weight_stem, f"refused by the restricted reader: {type(exc).__name__}: {exc}"
    if not state:
        return meta.weight_stem, "restricted reader returned no tensors"
    try:
        model = build_model(meta.family, meta.size, num_classes=meta.num_classes)
        load_weights(model, path)
    except Exception as exc:
        return meta.weight_stem, f"tensors do not fit the model: {type(exc).__name__}: {exc}"
    return meta.weight_stem, None


def write_junit(path: str, results: list[tuple[str, str | None, float]]) -> None:
    failures = sum(1 for _, failure, _ in results if failure)
    suite = ET.Element(
        "testsuite",
        name="pinned_variants_load",
        tests=str(len(results)),
        failures=str(failures),
        time=f"{sum(t for _, _, t in results):.3f}",
    )
    for stem, failure, elapsed in results:
        case = ET.SubElement(
            suite,
            "testcase",
            classname="scripts.verify_every_pinned_variant_loads",
            name=f"test_every_pinned_variant_still_loads[{stem}]",
            time=f"{elapsed:.3f}",
        )
        if failure:
            ET.SubElement(case, "failure", message=failure).text = failure
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junitxml")
    args = parser.parse_args()

    results: list[tuple[str, str | None, float]] = []
    for meta in list_available():
        started = time.monotonic()
        stem, failure = check_variant(meta)
        elapsed = time.monotonic() - started
        results.append((stem, failure, elapsed))
        status = f"FAIL: {failure}" if failure else "ok"
        print(f"{stem:<9} {status}  ({elapsed:.1f}s)", file=sys.stderr if failure else sys.stdout)

    if args.junitxml:
        write_junit(args.junitxml, results)
    failed = [stem for stem, failure, _ in results if failure]
    if failed:
        print(f"FAIL: {len(failed)} of {len(results)} pinned variants no longer load: {failed}")
        return 1
    print(f"OK: all {len(results)} pinned variants load through the enumerated allowlist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
