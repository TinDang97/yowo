"""A `.state_dict.pt` sidecar may not attest to itself with a public value.

`load_verified_state_dict` keyed its sidecar cache on `sidecar_key`, which is the
REGISTRY PIN whenever one is supplied -- and every registry pin is printed in
`src/yowo/models/_registry.py`. The sidecar fast path returned before
`verify_digest(checkpoint_path, pin)` ever ran, so anyone able to write
`~/.cache/yowo/weights/` could drop a `yolo11n.state_dict.pt` carrying
``raw_sha256: <the published pin>`` and arbitrary tensors, and it was served with
no error and no read of the raw checkpoint at all.

Code execution stayed blocked -- the sidecar loads under ``weights_only=True`` --
but weight integrity did not. A silently substituted detector is the harm
`weight-integrity` exists to prevent, reached by routing around its control
rather than by breaking it.

The irony that made it easy to miss: the UNPINNED branch already hashed the raw
file, because `sidecar_key` fell back to `_raw_digest`. Only the pinned branch --
the one meant to be stronger -- skipped it.

The rule these checks bind: the pin authenticates the FILE; the file's own digest
authenticates the SIDECAR. Neither may borrow the other's authority, and one
measurement answers both questions.

Fixtures are built in-process, following `test_verified_digest_threading.py`: a
checkpoint naming `ultralytics.nn.tasks.DetectionModel` is produced by
registering that module only while saving, then removing it. No real weights
file, no network, no AGPL artifact.

Authored red under ADD task `sidecar-not-self-attesting`.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from yowo.errors import ModelNotFoundError
from yowo.models._weights import WeightIntegrityError

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# In-process checkpoint fixtures
# ---------------------------------------------------------------------------


class _Stand_in(torch.nn.Module):
    """Stands in for ultralytics' DetectionModel while a fixture is written."""

    def __init__(self, fill: float) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, bias=False))
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(fill)


def _install_fake_ultralytics() -> None:
    pkg = types.ModuleType("ultralytics")
    nn = types.ModuleType("ultralytics.nn")
    tasks = types.ModuleType("ultralytics.nn.tasks")
    _Stand_in.__module__ = "ultralytics.nn.tasks"
    _Stand_in.__qualname__ = "DetectionModel"
    tasks.DetectionModel = _Stand_in  # type: ignore[attr-defined]
    pkg.nn = nn  # type: ignore[attr-defined]
    nn.tasks = tasks  # type: ignore[attr-defined]
    sys.modules.update({"ultralytics": pkg, "ultralytics.nn": nn, "ultralytics.nn.tasks": tasks})


def _remove_fake_ultralytics() -> None:
    for name in [m for m in sys.modules if m.startswith("ultralytics")]:
        del sys.modules[name]
    _Stand_in.__module__ = __name__
    _Stand_in.__qualname__ = "_Stand_in"


def _write_checkpoint(path: Path, fill: float) -> str:
    """Write an ultralytics-shaped checkpoint and return its SHA-256."""
    _install_fake_ultralytics()
    try:
        torch.save({"ema": _Stand_in(fill), "model": None}, path)
    finally:
        _remove_fake_ultralytics()
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def genuine(tmp_path: Path) -> tuple[Path, str]:
    """A real checkpoint and the digest its bytes hash to."""
    path = tmp_path / "yolo11n.pt"
    return path, _write_checkpoint(path, fill=0.25)


def _sidecar_of(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".state_dict.pt")


_POISON_KEY = "model.0.conv.weight"
_POISON_VALUE = 66.6


def _plant_sidecar(checkpoint: Path, claimed_key: str) -> Path:
    """Write a sidecar claiming *claimed_key*, carrying a tensor nothing else has.

    The value is the tell: if it comes back out of `load_verified_state_dict`,
    the planted sidecar was believed.
    """
    sidecar = _sidecar_of(checkpoint)
    torch.save(
        {"raw_sha256": claimed_key, "tensors": {_POISON_KEY: torch.tensor([_POISON_VALUE])}},
        sidecar,
    )
    return sidecar


def _served_the_plant(state: object) -> bool:
    """True when *state* is the planted sidecar's tensors rather than real ones."""
    if not isinstance(state, dict):
        return False
    tensor = state.get(_POISON_KEY)
    if tensor is None:
        return False
    return bool(torch.allclose(tensor, torch.tensor([_POISON_VALUE])))


def _public_pin() -> str:
    """The yolo11n digest, straight out of the registry any attacker can read."""
    from yowo.models._registry import get
    from yowo.types import ModelFamily, ModelSize

    pin = get(ModelFamily.YOLO11, ModelSize.NANO).sha256
    assert pin, "yolo11n has no registry pin; the reproduction's premise is gone"
    return pin


class _Sha256Counter:
    """Counts every SHA-256 construction, whoever makes it.

    Counting the primitive rather than `_raw_digest` is deliberate: a repair that
    hashes twice by routing the second read through `verify_digest`'s own
    `file_digest` would be invisible to a spy on `_raw_digest` alone, and that is
    precisely the shape R:REHASH forbids.
    """

    def __init__(self) -> None:
        self.count = 0
        self._real = hashlib.sha256

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self._real(*args, **kwargs)


# ---------------------------------------------------------------------------
# M1, M2, A1, E1, R:SELFKEYED -- the reproduction
# ---------------------------------------------------------------------------


def test_a_sidecar_keyed_on_the_public_pin_is_refused(tmp_path: Path) -> None:
    """covers: M1, M2, A1, E1, R:SELFKEYED -- a public value cannot authenticate.

    The orchestrator's reproduction, verbatim. The raw checkpoint is garbage that
    hashes to nothing in particular; the sidecar beside it carries the PUBLISHED
    yolo11n pin as its `raw_sha256`. Nothing here is secret: the pin is printed
    in `_registry.py`, so a sidecar keyed on it is a sidecar that authenticates
    itself with a value anyone can read.

    Refusal is the only acceptable outcome. Returning the planted tensors is the
    defect; returning real tensors is impossible, since the raw bytes are not a
    checkpoint at all.
    """
    from yowo.arch._weights import load_verified_state_dict

    raw = tmp_path / "yolo11n.pt"
    raw.write_bytes(b"GARBAGE-NOT-A-CHECKPOINT" * 64)
    pin = _public_pin()
    actual = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert actual != pin, "fixture bytes match the pin"

    _plant_sidecar(raw, claimed_key=pin)

    with pytest.raises(WeightIntegrityError) as exc:
        load_verified_state_dict(raw, raw_digest=pin)

    message = str(exc.value)
    assert pin in message, "the refusal does not report the digest that was expected"
    assert actual in message, "the refusal does not report the digest the raw bytes actually have"


def test_no_branch_returns_tensors_on_a_mismatch(genuine: tuple[Path, str]) -> None:
    """covers: R:SILENTSERVE -- no branch that can reach a sidecar serves a mismatch.

    Three branches can reach a sidecar. Fixing only the pinned one would leave
    the hole open under `raw_digest=None`, which is what every `-cls` and `-obb`
    entry and every explicit `weights_path` loads with.

    On the pinned branch a mismatch is a refusal. On the unpinned branch there is
    no pin to refuse against, so the sidecar is simply not believed and the raw
    checkpoint is reconverted (M4) -- but under no circumstances do the planted
    tensors come back.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine

    # Branch 1 -- pinned, sidecar claims the pin, file does not hash to it.
    _plant_sidecar(path, claimed_key=pinned)
    path.write_bytes(b"substituted after resolution" * 32)
    with pytest.raises(WeightIntegrityError):
        load_verified_state_dict(path, raw_digest=pinned)

    # Branch 2 -- unpinned, sidecar claims a digest the file does not have.
    path.unlink()
    honest = _write_checkpoint(path, fill=0.25)
    _plant_sidecar(path, claimed_key="0" * 64)
    state = load_verified_state_dict(path, raw_digest=None)
    assert not _served_the_plant(state), "the unpinned branch served a mismatched sidecar"

    # Branch 3 -- unpinned, sidecar claims the PIN rather than the file's digest.
    # The pin is public; keying on it must not buy a sidecar any trust at all.
    _plant_sidecar(path, claimed_key=_public_pin())
    state = load_verified_state_dict(path, raw_digest=None)
    assert not _served_the_plant(state), "a pin-keyed sidecar was believed on the unpinned branch"
    assert honest == hashlib.sha256(path.read_bytes()).hexdigest(), "fixture bytes drifted"


# ---------------------------------------------------------------------------
# M3, A5, R:REHASH -- one measurement, two questions
# ---------------------------------------------------------------------------


def test_the_raw_file_is_hashed_exactly_once(genuine: tuple[Path, str]) -> None:
    """covers: M3, A5, R:REHASH -- closing the hole must not double the cost.

    A5 fixes the order for exactly this reason: hash the file once, then compare
    that one measurement to the sidecar's claim AND to the pin. Consulting the
    pin first -- or calling `verify_digest`, which hashes again from scratch --
    would read 5-110 MB twice for one load.

    Counted on both the fast path (sidecar hit) and the converting path, pinned
    and unpinned, because the obvious naive repair costs a second hash on only
    one of them.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine

    # Converting path, pinned: hash the file, compare to the pin, convert.
    counter = _Sha256Counter()
    with patch.object(hashlib, "sha256", counter):
        load_verified_state_dict(path, raw_digest=pinned)
    assert counter.count == 1, f"pinned conversion hashed the raw file {counter.count}x"

    # Fast path, pinned: the sidecar written above is now a hit.
    assert _sidecar_of(path).exists(), "no sidecar was written to exercise the fast path"
    counter = _Sha256Counter()
    with patch.object(hashlib, "sha256", counter):
        load_verified_state_dict(path, raw_digest=pinned)
    assert counter.count == 1, f"pinned sidecar hit hashed the raw file {counter.count}x"

    # Converting path, unpinned.
    _sidecar_of(path).unlink()
    counter = _Sha256Counter()
    with patch.object(hashlib, "sha256", counter):
        load_verified_state_dict(path, raw_digest=None)
    assert counter.count == 1, f"unpinned conversion hashed the raw file {counter.count}x"

    # Fast path, unpinned.
    counter = _Sha256Counter()
    with patch.object(hashlib, "sha256", counter):
        load_verified_state_dict(path, raw_digest=None)
    assert counter.count == 1, f"unpinned sidecar hit hashed the raw file {counter.count}x"


def test_the_hash_precedes_reading_the_sidecar_contents(genuine: tuple[Path, str]) -> None:
    """covers: A3 -- the measurement lands before the sidecar's contents are read.

    "Before it is returned" is not good enough. A sidecar is attacker-controlled
    bytes; reading it back and only then deciding whether it was allowed to speak
    means it was already read on the attacker's terms. The raw digest must exist
    before `torch.load` is ever pointed at the sidecar.

    Run on the branch where the sidecar IS read -- a correctly keyed one beside a
    genuine file -- so an ordering that never reads it cannot pass vacuously.
    """
    from yowo.arch import _weights as arch_weights

    path, pinned = genuine
    _plant_sidecar(path, claimed_key=pinned)

    order: list[str] = []
    real_digest = arch_weights._raw_digest
    real_load = torch.load
    sidecar = _sidecar_of(path)

    def _spy_digest(target: Path) -> str:
        order.append("hash")
        return real_digest(target)

    def _spy_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(f, (str, Path)) and Path(f) == sidecar:
            order.append("read-sidecar")
        return real_load(f, *args, **kwargs)

    with (
        patch.object(arch_weights, "_raw_digest", side_effect=_spy_digest),
        patch.object(torch, "load", side_effect=_spy_load),
    ):
        arch_weights.load_verified_state_dict(path, raw_digest=pinned)

    assert "read-sidecar" in order, "the sidecar was never read; this check proves nothing"
    assert order[0] == "hash", (
        f"the sidecar's contents were read before the raw file was measured: {order}"
    )


# ---------------------------------------------------------------------------
# M4, E2, E3 -- a genuine sidecar is still believed; a stale one is routine
# ---------------------------------------------------------------------------


def test_a_genuine_sidecar_still_takes_the_fast_path(genuine: tuple[Path, str]) -> None:
    """covers: M4, E2 -- hardening must not cost the fast path its whole point.

    The sidecar exists so the executing reader runs once, ever. A real sidecar
    beside a real file must still short-circuit, and `_extract_state_dict` -- the
    one deserialisation in the system that can execute code -- must not be
    reached. A tripwire, not a spy that returns: reaching it at all is the
    failure.
    """
    from yowo.arch import _weights as arch_weights

    path, pinned = genuine
    first = arch_weights.load_verified_state_dict(path, raw_digest=pinned)
    assert first, "the first (converting) load returned nothing"
    assert _sidecar_of(path).exists(), "the converting load wrote no sidecar"

    tripwire = AssertionError("the raw checkpoint reached the executing reader")
    with patch.object(arch_weights, "_extract_state_dict", side_effect=tripwire) as convert:
        served = arch_weights.load_verified_state_dict(path, raw_digest=pinned)

    convert.assert_not_called()
    assert set(served) == set(first), "the sidecar did not serve the second load"


def test_a_stale_sidecar_is_discarded_and_reconverted(genuine: tuple[Path, str]) -> None:
    """covers: M4, E3 -- a sidecar from an older pin is ordinary, not an attack.

    Raising here would break every user whose cache predates a re-pin. The
    sidecar is dropped, the checkpoint reconverted, and the sidecar rewritten
    against the bytes actually on disk -- so the next load is fast again.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine
    _plant_sidecar(path, claimed_key="0" * 64)  # keyed on a previous pin

    state = load_verified_state_dict(path, raw_digest=pinned)

    assert not _served_the_plant(state), "the stale sidecar was served"
    assert state, "the reconversion returned nothing"
    blob = torch.load(_sidecar_of(path), map_location="cpu", weights_only=True)
    assert blob["raw_sha256"] == pinned, "the sidecar was not re-keyed to the file's own bytes"


def test_a_discarded_sidecar_reports_why(
    genuine: tuple[Path, str], caplog: pytest.LogCaptureFixture
) -> None:
    """covers: A6 -- the operator must read this as housekeeping, not a break-in.

    The audience is whoever is running inference, and what they need is: which
    file, that its cached conversion was not believed, and that the checkpoint is
    being reconverted so nothing is broken. A bare "digest mismatch" makes a
    routine re-pin look like an intrusion; silence makes an intrusion look
    routine.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine
    _plant_sidecar(path, claimed_key="0" * 64)

    with caplog.at_level(logging.INFO, logger="yowo.arch._weights"):
        load_verified_state_dict(path, raw_digest=pinned)

    discarded = [r.getMessage() for r in caplog.records if "reconvert" in r.getMessage().lower()]
    assert discarded, "a sidecar was silently thrown away with no explanation"
    said = " ".join(discarded).lower()
    assert path.name.lower() in said or _sidecar_of(path).name.lower() in said, (
        "the message does not name the file whose cached conversion was discarded"
    )
    assert "discard" in said or "not used" in said or "ignor" in said, (
        "the message does not say the sidecar was discarded"
    )


# ---------------------------------------------------------------------------
# A4, E4 -- a sidecar alone is unauthenticatable
# ---------------------------------------------------------------------------


def test_a_sidecar_without_its_raw_checkpoint_is_refused(genuine: tuple[Path, str]) -> None:
    """covers: A4, E4 -- deleting the .pt does not promote the sidecar.

    With the raw file gone there is nothing left to measure, so the sidecar has
    nothing to be authenticated against. Serving it anyway is exactly R:SELFKEYED
    with the evidence deleted -- and it is the cheapest way to reach the hole:
    plant a sidecar, remove the checkpoint.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine
    _plant_sidecar(path, claimed_key=pinned)
    path.unlink()
    assert _sidecar_of(path).exists(), "fixture removed the sidecar too"

    with pytest.raises(ModelNotFoundError) as exc:
        load_verified_state_dict(path, raw_digest=pinned)

    assert path.name in str(exc.value), "the refusal does not name the missing checkpoint"


# ---------------------------------------------------------------------------
# A2, E5 -- the unpinned path keeps the behaviour it already had
# ---------------------------------------------------------------------------


def test_the_unpinned_path_is_unchanged(genuine: tuple[Path, str]) -> None:
    """covers: A2, E5 -- unpinned loads gain the guarantee without paying for it.

    `raw_digest=None` already hashed the raw file, because `sidecar_key` fell
    back to `_raw_digest` -- the accident that makes the pinned branch's omission
    so easy to miss. Nothing about this branch may change: it still converts, it
    still keys the sidecar on the file's own bytes, it still makes no
    verification claim, and it still hashes exactly once.
    """
    from yowo.arch import _weights as arch_weights

    path, _pinned = genuine
    computed = hashlib.sha256(path.read_bytes()).hexdigest()

    counter = _Sha256Counter()
    with (
        patch.object(hashlib, "sha256", counter),
        patch.object(arch_weights, "verify_digest") as verify,
    ):
        state = arch_weights.load_verified_state_dict(path, raw_digest=None)

    assert state, "an unpinned checkpoint must still load"
    verify.assert_not_called()
    assert counter.count == 1, f"the unpinned path hashed the raw file {counter.count}x"

    blob = torch.load(_sidecar_of(path), map_location="cpu", weights_only=True)
    assert blob["raw_sha256"] == computed, "the unpinned sidecar is not keyed on the file's bytes"


# ---------------------------------------------------------------------------
# M5 -- the callers keep working
# ---------------------------------------------------------------------------


def test_load_verified_state_dict_signature_is_unchanged() -> None:
    """covers: M5 -- this node hardens a body, not an interface.

    Three inference loaders (`load_weights`, `load_classify_weights`,
    `load_obb_weights`) and the export path all call this. A new parameter would
    imply the old shape was incapable, when the whole defect was that the value
    it already took was never allowed to reach the sidecar decision.
    """
    from yowo.arch._weights import load_verified_state_dict

    sig = inspect.signature(load_verified_state_dict)
    assert list(sig.parameters) == ["checkpoint_path", "raw_digest"]
    assert sig.parameters["raw_digest"].default is None
    assert sig.parameters["checkpoint_path"].default is inspect.Parameter.empty
