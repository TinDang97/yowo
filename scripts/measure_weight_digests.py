#!/usr/bin/env python3
"""Measure the SHA-256 of every registered model weight, from the URL itself.

R:UNMEASURED — a pin is only worth what its measurement is worth. This downloads
each registered `default_weights_url` and hashes the bytes that URL actually
served, recording the URL, the byte count and the date alongside the digest.
Nothing is copied from another variant, read off a local file of unknown origin,
or transcribed from a third party.

Streams to a temporary file and hashes as it goes, so a 118 MB weight never has
to be held in memory or kept after measuring.

    python3 scripts/measure_weight_digests.py                  # every registry
    python3 scripts/measure_weight_digests.py --task classify  # one of them
    python3 scripts/measure_weight_digests.py --out pins.json

Exit 0 only when every requested URL was measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yowo.models._registry import _CLS_REGISTRY, _OBB_REGISTRY, _REGISTRY

REGISTRIES = {"detect": _REGISTRY, "classify": _CLS_REGISTRY, "obb": _OBB_REGISTRY}
CHUNK = 1 << 20
TIMEOUT = 120
RETRIES = 3


def measure(url: str) -> tuple[str, int]:
    """Return ``(sha256, bytes)`` for what *url* serves, streaming to disk.

    Retries a transient failure: a 500 MB sweep that dies on one flaky
    connection and reports nothing is worse than one that takes a minute longer.
    A non-transient failure (404, bad URL) is raised on the first attempt —
    retrying it only wastes time and hides the real answer.
    """
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                urllib.request.urlopen(url, timeout=TIMEOUT) as response,
                tempfile.TemporaryFile() as sink,
            ):
                while chunk := response.read(CHUNK):
                    digest.update(chunk)
                    sink.write(chunk)
                    size += len(chunk)
            return digest.hexdigest(), size
        except urllib.error.HTTPError:
            raise  # a 404 is an answer, not a blip
        except Exception as exc:  # timeouts, resets, partial reads
            last = exc
            print(f"    attempt {attempt}/{RETRIES} failed: {exc}", file=sys.stderr)
    raise RuntimeError(f"{url}: {RETRIES} attempts failed") from last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(REGISTRIES), action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = args.task or sorted(REGISTRIES)
    records: list[dict[str, object]] = []
    failures: list[str] = []

    for task in tasks:
        for meta in REGISTRIES[task].values():
            url = meta.default_weights_url
            print(f"{task:9} {meta.weight_stem:14} ...", flush=True)
            try:
                sha, size = measure(url)
            except Exception as exc:  # reported, never swallowed
                print(f"{task:9} {meta.weight_stem:14} FAILED: {exc}", file=sys.stderr)
                failures.append(meta.weight_stem)
                continue
            records.append(
                {
                    "task": task,
                    "weight_stem": meta.weight_stem,
                    "url": url,
                    "sha256": sha,
                    "bytes": size,
                    "measured": date.today().isoformat(),
                }
            )
            print(f"{task:9} {meta.weight_stem:14} {sha}  {size / 1e6:7.1f} MB")

    if args.out:
        args.out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {len(records)} measurement(s) -> {args.out}")

    if failures:
        print(f"\nFAILED to measure: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nOK: measured {len(records)} weight(s), every requested URL answered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
