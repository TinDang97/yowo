#!/usr/bin/env python3
"""Measure which globals the shipped checkpoints name, so the loader can ENUMERATE them.

The checkpoint loader (`yowo.arch._weights`) reads an ultralytics `.pt` through a
restricted unpickler whose `find_class` admits an allowlist. `checkpoint-loader`
froze the torch layer classes as a PREFIX rule — `torch.nn.modules.` — because
only one checkpoint was reachable when it was built. A prefix admits a namespace
where the boundary should name a class, so `narrow-loader-allowlist` replaces it
with `_ALLOWED_TORCH`, an exact `(module, name)` set. This script is where that
set comes from: every entry is OBSERVED here, never reasoned into existence.

What is measured (A2 of the node):
  * the 10 digest-pinned DETECTION variants, each fetched through
    `resolve_weights()` so every measured byte was verified against its pin —
    and re-hashed here so the digest recorded is the digest read; and
  * an in-process sweep of our own `YOLOModel` / `ClassifyModel` / `OBBModel`
    across every registered size, serialised and read back through the same
    recorder. The `-cls` and `-obb` registry entries carry no pin, so their
    checkpoints are NOT fetched — measuring a trust boundary from unverified
    bytes would let the corpus choose the allowlist. Their layer types come
    from code we own instead.

How it reads: `torch.load` is driven with a `find_class` that RECORDS every
global and then applies the production loader's own rules for everything
outside `torch.nn.*` — the exact set, the storage suffix, an inert stub for
`ultralytics.*` — while constructing `torch.nn.*` so the stream can be walked.
It measures the boundary; it does not widen what runs.

This is a MAINTAINER action, not a CI job (A7, A8): the corpus is ~440 MB, and
a per-run download is the check someone eventually disables. It refuses to
run under `CI`. Re-run it when a pin changes or a variant is added to the
registry (A9) — the two events that can introduce a new class — and commit the
manifest it writes next to the list it feeds.

It fails and writes NOTHING if any checkpoint cannot be fetched or verified
(A10). A set measured over 7 of 10 variants would narrow the allowlist past
what the shipped models need and look identical to a complete one.

Usage:
    python3 scripts/measure_checkpoint_globals.py [--write PATH] [--cache-dir DIR]

Output is sorted by (module, name) so a re-measure diffs cleanly (A11), and
each global is printed with the variants that named it, so a maintainer
deciding whether to widen the list has a basis for the judgement (A12).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from yowo.models._registry import _CLS_REGISTRY, _OBB_REGISTRY, ModelMeta, list_available
from yowo.models._weights import file_digest, resolve_weights
from yowo.types import ModelSpec

if TYPE_CHECKING:
    import torch as _torch

MANIFEST = Path(__file__).resolve().parents[1] / "scripts" / "checkpoint_globals_manifest.json"
CHANGE_REQUEST = ".add/tasks/narrow-loader-allowlist.md"

# What the measurement proposes for `_ALLOWED_TORCH`: every observed global under
# torch's neural-network namespace — layer classes, and `Parameter` if a stream
# ever names it. Everything else observed is reported against the rule that
# already covers it, so the maintainer sees the whole boundary, not one bucket.
_TORCH_NN = "torch.nn."

Observations = dict[tuple[str, str], set[str]]


class MeasurementError(RuntimeError):
    """A checkpoint could not be measured; no manifest may be written."""


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


def _recording_module(label: str, observations: Observations):
    """A `pickle_module` for `torch.load` whose `find_class` records, then rules.

    Recording happens before any decision, so a global is observed even when the
    stream would be refused in production. The decision that follows is the
    production loader's for everything outside `torch.nn.*` — reused, not
    re-stated — plus constructing `torch.nn.*`, which is the set under
    measurement and cannot be consulted while it is being measured.
    """
    import pickle
    import types

    from yowo.arch._weights import (
        _ALLOWED_EXACT,
        _ALLOWED_STORAGE_SUFFIX,
        _inert_module_cls,
    )

    inert = _inert_module_cls()

    class _RecordingUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> object:
            observations.setdefault((module, name), set()).add(label)
            if module.startswith(_TORCH_NN):
                return super().find_class(module, name)
            if (module, name) in _ALLOWED_EXACT:
                return super().find_class(module, name)
            if module == "torch" and name.endswith(_ALLOWED_STORAGE_SUFFIX):
                return super().find_class(module, name)
            # Anything else — `ultralytics.*`, our own `yowo.arch.*` in the sweep,
            # or something new — is stubbed so the stream can still be walked to
            # the end. It is recorded above and reported below; it is never run.
            return inert

    shim = types.ModuleType("yowo_recording_pickle")
    shim.Unpickler = _RecordingUnpickler  # type: ignore[attr-defined]
    shim.load = pickle.load  # type: ignore[attr-defined]
    shim.Pickler = pickle.Pickler  # type: ignore[attr-defined]
    shim.dump = pickle.dump  # type: ignore[attr-defined]
    shim.dumps = pickle.dumps  # type: ignore[attr-defined]
    shim.loads = pickle.loads  # type: ignore[attr-defined]
    return shim


def _walk(source: Path | io.BytesIO, label: str, observations: Observations) -> None:
    import torch

    torch.load(
        source,
        map_location="cpu",
        weights_only=False,
        pickle_module=_recording_module(label, observations),
    )


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def measure_checkpoint(
    meta: ModelMeta, observations: Observations, cache_dir: Path | None
) -> dict[str, str]:
    """Fetch one pinned variant through the production path, verify, record, walk.

    Returns the manifest row: stem, url, and the digest of the bytes actually read.
    """
    if meta.sha256 is None:
        raise MeasurementError(
            f"{meta.weight_stem} has no pinned sha256; an unpinned checkpoint cannot "
            "define a trust boundary (A2). Pin it in yowo.models._registry first."
        )
    spec = ModelSpec(family=meta.family, size=meta.size)
    try:
        path = resolve_weights(spec, cache_dir=cache_dir)
    except Exception as exc:
        raise MeasurementError(
            f"could not fetch or verify {meta.weight_stem}: {type(exc).__name__}: {exc}"
        ) from exc
    # `resolve_weights` verified the file; hash it again so the digest RECORDED is
    # the digest of the bytes this process read, not a value copied from the pin.
    digest = file_digest(path)
    if digest != meta.sha256:
        raise MeasurementError(
            f"{meta.weight_stem} at {path} hashes to {digest}, pin is {meta.sha256}"
        )
    _walk(path, meta.weight_stem, observations)
    return {"stem": meta.weight_stem, "url": meta.default_weights_url, "sha256": digest}


def _sweep_models() -> list[tuple[str, _torch.nn.Module]]:
    """Every model class we ship, at every registered size, built unfused.

    Unfused is the shape an upstream checkpoint has: `BatchNorm2d` present, no
    folded convolutions. The x-sized models are a few hundred MB each; the
    caller walks and drops them one at a time.
    """
    from yowo.arch import build_classify_model, build_model, build_obb_model

    built: list[tuple[str, _torch.nn.Module]] = []
    for meta in list_available():
        built.append(
            (
                f"arch:{meta.weight_stem}",
                build_model(meta.family, meta.size, num_classes=meta.num_classes),
            )
        )
    for meta in sorted(_CLS_REGISTRY.values(), key=lambda m: m.weight_stem):
        built.append(
            (
                f"arch:{meta.weight_stem}",
                build_classify_model(meta.family, meta.size, num_classes=meta.num_classes),
            )
        )
    for meta in sorted(_OBB_REGISTRY.values(), key=lambda m: m.weight_stem):
        built.append(
            (
                f"arch:{meta.weight_stem}",
                build_obb_model(meta.family, meta.size, num_classes=meta.num_classes),
            )
        )
    return built


def measure_arch_sweep(observations: Observations) -> list[str]:
    """Serialise each of our models in memory and read it back through the recorder.

    A round trip through the same recorder — rather than walking `.modules()` —
    means the sweep records what a STREAM of this architecture names, which is
    the only thing `find_class` will ever be asked about.
    """
    import torch

    labels: list[str] = []
    for label, model in _sweep_models():
        buf = io.BytesIO()
        torch.save(model, buf)
        buf.seek(0)
        _walk(buf, label, observations)
        labels.append(label)
        del model, buf
    return labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rule_for(module: str, name: str, observers: set[str]) -> str:
    """Which production rule covers an observed global that is NOT a torch.nn one."""
    from yowo.arch._weights import _ALLOWED_EXACT, _ALLOWED_STORAGE_SUFFIX, _STUBBED_PREFIXES

    if (module, name) in _ALLOWED_EXACT:
        return "exact"
    if module == "torch" and name.endswith(_ALLOWED_STORAGE_SUFFIX):
        return "storage-suffix"
    if module.startswith(_STUBBED_PREFIXES):
        return "stub"
    if all(o.startswith("arch:") for o in observers):
        return "sweep-scaffold (only our own serialised model names it; a checkpoint is refused)"
    return "NOT COVERED — the loader refuses this today"


def build_manifest(
    checkpoints: list[dict[str, str]], arch_sweep: list[str], observations: Observations
) -> dict[str, object]:
    import torch

    allowed_torch = sorted(g for g in observations if g[0].startswith(_TORCH_NN))
    return {
        "schema": 1,
        "measured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "measured_by": "scripts/measure_checkpoint_globals.py",
        "change_request": CHANGE_REQUEST,
        "checkpoints": sorted(checkpoints, key=lambda c: c["stem"]),
        "arch_sweep": sorted(arch_sweep),
        "allowed_torch": [list(g) for g in allowed_torch],
        "observations": {f"{m}.{n}": sorted(observations[(m, n)]) for m, n in sorted(observations)},
    }


def print_report(manifest: dict[str, object], observations: Observations) -> None:
    checkpoints: list[dict[str, str]] = manifest["checkpoints"]  # type: ignore[assignment]
    arch_sweep: list[str] = manifest["arch_sweep"]  # type: ignore[assignment]
    print("Checkpoints measured (every byte digest-verified):")
    for row in checkpoints:
        print(f"  {row['stem']:<9} sha256={row['sha256']}")
    print(f"Arch sweep: {len(arch_sweep)} models serialised in process")
    print()
    print("Proposed _ALLOWED_TORCH — observed torch.nn globals, sorted by (module, name):")
    for module, name in sorted(g for g in observations if g[0].startswith(_TORCH_NN)):
        print(f"  ({module!r}, {name!r})")
        print(f"      named by: {', '.join(sorted(observations[(module, name)]))}")
    print()
    print("Other observed globals, with the production rule that covers each:")
    for module, name in sorted(g for g in observations if not g[0].startswith(_TORCH_NN)):
        observers = observations[(module, name)]
        print(f"  {module}.{name}  [{_rule_for(module, name, observers)}]")
        print(f"      named by: {', '.join(sorted(observers))}")


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    """Atomic: the manifest appears whole or not at all."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--write",
        type=Path,
        default=MANIFEST,
        help=f"where to record the manifest (default: {MANIFEST})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="override the weight cache root (default: resolve_weights' own)",
    )
    args = parser.parse_args(argv)

    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        parser.exit(
            2,
            "measure_checkpoint_globals: refusing to run under CI. This fetches ~440 MB "
            "of pinned checkpoints and is a maintainer action, run locally when a pin "
            f"changes or a variant is added; see {CHANGE_REQUEST}.\n",
        )

    started = time.monotonic()
    observations: Observations = {}
    try:
        checkpoints = [
            measure_checkpoint(meta, observations, args.cache_dir) for meta in list_available()
        ]
        arch_sweep = measure_arch_sweep(observations)
    except MeasurementError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print("No manifest written: a partial measurement is not a measurement.", file=sys.stderr)
        return 1

    manifest = build_manifest(checkpoints, arch_sweep, observations)
    print_report(manifest, observations)
    write_manifest(manifest, args.write)
    print()
    print(
        f"OK: {len(checkpoints)} checkpoints + {len(arch_sweep)} in-process models measured "
        f"in {time.monotonic() - started:.1f}s; manifest written to {args.write}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
