"""No weight is deserialized before its pinned SHA-256 is verified.

Red-first for ADD task `weight-integrity`.

Today `resolve_weights` verifies nothing: a cache hit returns immediately and
`_download` has no `hashlib` at all. `models/README.md:67` nonetheless tells the
reader that a SHA-256 is "checked against a sidecar .sha256 file" and that the
atomic write verifies the hash — a documented control that does not exist.
"""

from __future__ import annotations

import hashlib
import inspect
import warnings
from pathlib import Path

import pytest

from yowo.models import _weights as weights_mod
from yowo.models import list_available
from yowo.models._weights import _warn_unpinned
from yowo.types import ModelFamily, ModelSize, ModelSpec

REPO_ROOT = Path(__file__).parent.parent.parent
README = REPO_ROOT / "src/yowo/models/README.md"

# Captured from the canonical release assets during Direction. yolo11n's value was
# cross-checked against the local cache entry and matched byte for byte.
KNOWN_PINS = {
    "yolo11n": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo26n": "9b09cc8bf347f0fc8a5f7657",  # prefix; full value pinned in the registry
}


def test_every_builtin_carries_a_pinned_digest() -> None:
    """covers: M1, A2 — a URL with no digest is a URL you cannot verify."""
    unpinned = [m.weight_stem for m in list_available() if not getattr(m, "sha256", None)]
    assert unpinned == [], f"registered models with no pinned digest: {unpinned}"


def test_yolo26_urls_use_the_release_that_has_them() -> None:
    """covers: M1 — all five YOLO26 detection weights 404 today.

    `_make_meta` builds every detection URL from the v8.3.0 base, but YOLO26
    ships under v8.4.0. Verified by hand: v8.3.0/yolo26n.pt -> 404,
    v8.4.0/yolo26n.pt -> 200.
    """
    bad = [
        m.weight_stem
        for m in list_available()
        if m.family == ModelFamily.YOLO26 and "v8.4.0" not in m.default_weights_url
    ]
    assert bad == [], f"YOLO26 models pointing at a release without them: {bad}"


def test_pinned_digest_matches_the_real_upstream_file() -> None:
    """covers: A2 — a wrong pin bricks a model more thoroughly than no pin."""
    by_stem = {m.weight_stem: m for m in list_available()}
    meta = by_stem["yolo11n"]
    assert getattr(meta, "sha256", None) == KNOWN_PINS["yolo11n"]


def test_download_verifies_before_moving_into_cache() -> None:
    """covers: M2, R:UNVERIFIED, A8, E1 — a half-file is trusted forever today."""
    src = inspect.getsource(weights_mod)
    assert "hashlib" in src, "the download path computes no digest at all"
    dl = inspect.getsource(weights_mod._download)
    assert "sha256" in dl, "_download does not verify before os.replace"


def test_cache_hit_is_verified_every_load() -> None:
    """covers: M3, A9 — `if dest.exists(): return dest` trusts the file forever."""
    src = inspect.getsource(weights_mod.resolve_weights)
    assert "sha256" in src or "verify" in src, (
        "resolve_weights returns a cache hit without verifying it"
    )


def test_mismatch_never_falls_back_to_the_bad_file() -> None:
    """covers: R:SILENT, A10 — offline with a bad cache entry must raise."""
    assert hasattr(weights_mod, "verify_digest"), "no verify_digest to exercise"
    with pytest.raises(Exception) as excinfo:
        weights_mod.verify_digest(Path(__file__), "0" * 64)
    assert "0000" in str(excinfo.value) or "digest" in str(excinfo.value).lower()


def test_unverifiable_cached_weight_is_refetched_with_a_warning() -> None:
    """covers: M5, E2 — the human chose warn-and-re-fetch over hard-fail."""
    src = inspect.getsource(weights_mod)
    assert "warn" in src.lower(), "no warning path for a pre-verification cache entry"


def test_routine_load_uses_weights_only() -> None:
    """covers: M4, A14 — the steady-state path must not unpickle."""
    from yowo.arch import _weights as arch_weights

    # Strip comments first: arch/_weights.py:86 CONTAINS the string
    # "weights_only=True" inside a comment explaining why it cannot be used on a
    # raw ultralytics checkpoint. Matching that would be a check that passes on
    # prose while the unpickler still runs on every load.
    code = "\n".join(line.split("#", 1)[0] for line in inspect.getsource(arch_weights).splitlines())
    assert "weights_only=True" in code, (
        "no weights_only=True load path in code; every load still runs the unpickler"
    )


def test_conversion_happens_after_verification() -> None:
    """covers: A11 — converting first would execute the payload before checking it."""
    from yowo.arch import _weights as arch_weights

    assert hasattr(arch_weights, "load_verified_state_dict"), "no verified conversion entry point"


def test_readme_matches_the_implementation() -> None:
    """covers: M6 — the README documents a control that does not exist."""
    doc = README.read_text().lower()
    src = inspect.getsource(weights_mod)
    arch_src = inspect.getsource(__import__("yowo.arch._weights", fromlist=["x"]))

    # Keyed on the MECHANISM, not on a sentence — a phrase-match check breaks the
    # moment the prose is reworded, which teaches people to edit the test.
    # Each pair is (the code really does this, the doc really says so).
    for what, implemented, documented in (
        ("digest verification", "hashlib" in src, "sha256" in doc),
        ("cache-hit verification", "verify_digest(dest" in src, "cache hit" in doc),
        ("warn-and-refetch", "re-downloading once" in src, "re-downloaded" in doc),
        ("unpinned models warn", "_warn_unpinned" in src, "sha256=None".lower() in doc),
        ("weights_only load", "weights_only=True" in arch_src, "weights_only=true" in doc),
    ):
        assert implemented == documented, (
            f"README and code disagree about {what}: code={implemented}, doc={documented}"
        )


def test_digest_helper_hashes_a_real_file() -> None:
    """covers: M2 — the primitive itself, so a mismatch test cannot pass vacuously."""
    assert hasattr(weights_mod, "file_digest"), "no file_digest helper"
    expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert weights_mod.file_digest(Path(__file__)) == expected


def test_unpinned_custom_model_warns_but_loads(tmp_path: Path) -> None:
    """covers: A1, A4, E3 — a private bucket has no digest we could know.

    Named in the frozen CHECKS but missing from the first implementation; the
    gate refused the PASS until it existed. Refusing to load an unpinned model
    would break every custom-registration deployment on upgrade, so the
    contract is warn-and-load, and this is what holds it to that.
    """
    from yowo.models._weights import resolve_weights

    weight = tmp_path / "custom.pt"
    weight.write_bytes(b"stand-in bytes")

    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO, weights_path=weight)
    assert resolve_weights(spec) == weight, "an explicit weights_path must still resolve"

    # And the registry path: an unpinned meta warns rather than refusing.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_unpinned("custom_unpinned")
    assert any("no pinned sha256" in str(x.message) for x in w), (
        "an unpinned model must warn, not pass silently"
    )
    # Not `isinstance(msg, Exception)` — Warning subclasses Exception, so that
    # can never be false for a recorded warning. What actually distinguishes
    # "warn" from "refuse" is the category, and that the call returned at all.
    assert all(issubclass(x.category, Warning) for x in w), (
        "an unpinned model must warn, not refuse"
    )
