"""A global the state_dict does not need is stubbed or refused — never resolved.

`task-aware-weight-resolution` made a classify spec resolve and digest-verify its
own `yolo11n-cls.pt`, and an obb spec its `yolo11n-obb.pt`. The loader then
refused both — measured 2026-09-10 through the real restricted loader:

    _extract_state_dict(~/.cache/yowo/weights/yolo11/n/yolo11n-obb.pt)
        -> refused on __builtin__.getattr
    _extract_state_dict(~/.cache/yowo/weights/yolo11/n/yolo11n-cls.pt)
        -> refused on torchvision.transforms.transforms.Compose

Both refusals are CORRECT. `getattr` inside a restricted unpickler is a general
attribute-access primitive: a checkpoint that can call it can reach any attribute
of anything it can name and chain from there. `torchvision.transforms.*` is the
training-time preprocessing pipeline stored as objects, and `Compose` holds a
list of arbitrary callables. Neither is weights.

So neither name is ADMITTED. `torchvision.` joins the stubbed prefixes beside
`ultralytics.` and `models.`, and `__builtin__.getattr` is handed an inert
callable. The set of RESOLVED names — `_ALLOWED` — comes out of this node
unchanged, and the two permissions stay separately named in the source, because
collapsing them is how a stand-in becomes an admission.

Every check here goes through the real `_extract_state_dict` rather than
inspecting a constant. Authored red under ADD task `non-weight-globals-are-inert`.
"""

from __future__ import annotations

import ast
import builtins
import collections
import hashlib
import inspect
import io
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "scripts" / "checkpoint_globals_manifest.json"
_MEASURE_SCRIPT = _REPO_ROOT / "scripts" / "measure_checkpoint_globals.py"

# What a load must yield once the two refusals are answered. Measured through
# the real loader with an INERT `getattr` and `torchvision.` stubbed, which is
# what this node specifies; recorded on the frozen node at E1/E2.
_OBB_ENTRIES = 541
_CLS_ENTRIES = 236


# ---------------------------------------------------------------------------
# Reaching the shipped checkpoints without ever reaching the network
# ---------------------------------------------------------------------------


def _cached_checkpoint(task: str, family: str, size: str) -> Path:
    """Where `resolve_weights` would have put this variant. Never downloads."""
    from yowo.models._registry import get_for_task
    from yowo.models._weights import _CACHE_DIR
    from yowo.types import ModelFamily, ModelSize

    meta = get_for_task(task, ModelFamily(family), ModelSize(size))
    root = Path(os.environ.get("YOWO_CACHE_DIR", str(_CACHE_DIR)))
    return root / family / size / f"{meta.weight_stem}.pt"


def _require(path: Path) -> Path:
    """Skip rather than fetch: the unit suite never downloads a corpus."""
    if not path.is_file():
        pytest.skip(
            f"{path} is not in the weight cache. This check reads a shipped "
            "checkpoint and the unit suite never downloads one."
        )
    return path


# ---------------------------------------------------------------------------
# Fixtures — built in process, no network, no AGPL artifact
# ---------------------------------------------------------------------------


def _tensor_bearing() -> Any:
    """A module built only from allowlisted torch classes, so it always loads."""
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, bias=False))


class _Secret:
    """A real object with a real attribute, for the gadget probe to fail to read."""

    def __init__(self) -> None:
        self.secret = "sentinel"


class _GetattrPayload:
    """A REDUCE naming `getattr` — the attribute-access primitive itself."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (getattr, ({"a": 1}, "keys"))


_TORCHVISION_CONSTRUCTED: list[str] = []


class _FakeCompose:
    """Stands in for `torchvision.transforms.transforms.Compose` while saving.

    Records into `_TORCHVISION_CONSTRUCTED` if the loader ever restores THIS
    class from checkpoint bytes — which is the thing M2 forbids.
    """

    def __init__(self) -> None:
        self.transforms: list[object] = []

    def __setstate__(self, state: dict[str, object]) -> None:
        _TORCHVISION_CONSTRUCTED.append("Compose")
        self.__dict__.update(state)


def _install_fake_torchvision() -> None:
    pkg = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    inner = types.ModuleType("torchvision.transforms.transforms")
    _FakeCompose.__module__ = "torchvision.transforms.transforms"
    _FakeCompose.__qualname__ = "Compose"
    inner.Compose = _FakeCompose  # type: ignore[attr-defined]
    transforms.transforms = inner  # type: ignore[attr-defined]
    pkg.transforms = transforms  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "torchvision": pkg,
            "torchvision.transforms": transforms,
            "torchvision.transforms.transforms": inner,
        }
    )


def _remove_fake_torchvision() -> None:
    for name in [m for m in sys.modules if m == "torchvision" or m.startswith("torchvision.")]:
        del sys.modules[name]
    _FakeCompose.__module__ = __name__
    _FakeCompose.__qualname__ = "_FakeCompose"


@pytest.fixture
def torchvision_checkpoint(tmp_path: Path) -> Iterator[Path]:
    """A checkpoint whose metadata is a `torchvision.transforms` object."""
    _TORCHVISION_CONSTRUCTED.clear()
    _install_fake_torchvision()
    path = tmp_path / "transforms.pt"
    torch.save({"ema": _tensor_bearing(), "model": None, "transforms": _FakeCompose()}, path)
    try:
        yield path
    finally:
        _remove_fake_torchvision()
        _TORCHVISION_CONSTRUCTED.clear()


@pytest.fixture
def getattr_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint naming `__builtin__.getattr`, as `-obb` does."""
    path = tmp_path / "getattr.pt"
    torch.save({"ema": _tensor_bearing(), "model": None, "payload": _GetattrPayload()}, path)
    return path


def _find_class(checkpoint_path: Path, module: str, name: str) -> Any:
    """Ask the PRODUCTION unpickler what it hands back for one name."""
    from yowo.arch._weights import _restricted_unpickler_module

    shim = _restricted_unpickler_module(checkpoint_path)
    return shim.Unpickler(io.BytesIO(b"")).find_class(module, name)


class _AllowlistSpy:
    """Stands in for `_ALLOWED` and records every question and every verdict.

    A name reaches `super().find_class` — the REAL resolution — only on a True
    verdict here, because the exact set is the first branch and the only one
    that resolves. So the admitted list is exactly the list of names that were
    imported, and everything else the stream named was stood in for or refused.
    """

    def __init__(self, real: frozenset[tuple[str, str]]) -> None:
        self._real = real
        self.asked: list[tuple[Any, bool]] = []

    def __contains__(self, pair: object) -> bool:
        verdict = pair in self._real
        self.asked.append((pair, verdict))
        return verdict

    @property
    def seen(self) -> set[Any]:
        return {pair for pair, _ in self.asked}

    @property
    def admitted(self) -> set[Any]:
        return {pair for pair, verdict in self.asked if verdict}


def _state_dict_digest(state: Mapping[str, Any]) -> str:
    """SHA-256 over every key, dtype, shape and tensor BYTE, in sorted order."""
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        h.update(key.encode())
        h.update(str(tensor.dtype).encode())
        h.update(repr(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _find_class_source() -> str:
    from yowo.arch import _weights

    return inspect.getsource(_weights._restricted_unpickler_module)


def _comment_block_above(source: str, assignment_prefix: str) -> list[str]:
    """The `#` lines above a module-level assignment, up to the previous statement."""
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(assignment_prefix):
            block: list[str] = []
            cursor = index - 1
            while cursor >= 0:
                stripped = lines[cursor].strip()
                if stripped.startswith("#"):
                    block.append(stripped)
                elif stripped:
                    break
                cursor -= 1
            return list(reversed(block))
    return []


# ---------------------------------------------------------------------------
# E1, E2, M5 — the two checkpoints the loader refuses today
# ---------------------------------------------------------------------------


def test_the_obb_checkpoint_loads() -> None:
    """covers: M5, E1 — the measured refusal on `__builtin__.getattr` is gone.

    541 entries, measured through the real loader with an inert `getattr`. A
    count assertion, not a truthiness one: a stand-in that absorbed a call it
    should not have would yield a SHORTER state_dict and still look loaded.
    """
    from yowo.arch._weights import _extract_state_dict

    path = _require(_cached_checkpoint("obb", "yolo11", "n"))
    state = _extract_state_dict(path)

    assert len(state) == _OBB_ENTRIES, f"expected {_OBB_ENTRIES} entries, got {len(state)}"
    assert all(isinstance(v, torch.Tensor) for v in state.values()), "not every entry is a tensor"
    assert "model.0.conv.weight" in state, "the first conv is missing from the state_dict"


def test_the_classify_checkpoint_loads() -> None:
    """covers: M5, E2 — the same for the `torchvision` refusal.

    236 entries. E2 previously claimed a further refusal behind `Compose`;
    re-measured with an INERT `getattr`, there is none.
    """
    from yowo.arch._weights import _extract_state_dict

    path = _require(_cached_checkpoint("classify", "yolo11", "n"))
    state = _extract_state_dict(path)

    assert len(state) == _CLS_ENTRIES, f"expected {_CLS_ENTRIES} entries, got {len(state)}"
    assert all(isinstance(v, torch.Tensor) for v in state.values()), "not every entry is a tensor"
    assert "model.10.linear.bias" in state, "the classifier head is missing from the state_dict"


# ---------------------------------------------------------------------------
# M1 — the real resolution is reached only for the exact allowlist
# ---------------------------------------------------------------------------


def test_super_find_class_is_reached_only_for_allowlisted_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """covers: M1 — a spy records every name the real resolution is asked for.

    Across a load of each task's checkpoint, every admitted name is on the exact
    allowlist, and the names this node makes INERT or STUBBED are seen by
    `find_class` and never admitted. That difference — named, and not imported —
    is the difference between standing in for a class and importing it.
    """
    from yowo.arch import _weights

    real_allowed = _weights._ALLOWED
    real_import = builtins.__import__

    checked = 0
    for task, expect_named in (
        ("obb", ("__builtin__", "getattr")),
        ("classify", ("torchvision.transforms.transforms", "Compose")),
    ):
        path = _cached_checkpoint(task, "yolo11", "n")
        if not path.is_file():
            continue
        imported: list[str] = []

        def recording_import(name: str, *a: object, **k: object) -> Any:
            imported.append(name)  # noqa: B023
            return real_import(name, *a, **k)  # type: ignore[arg-type]

        spy = _AllowlistSpy(real_allowed)
        monkeypatch.setattr(_weights, "_ALLOWED", spy)
        monkeypatch.setattr(builtins, "__import__", recording_import)
        try:
            state = _weights._extract_state_dict(path)
        finally:
            monkeypatch.undo()

        assert state, f"{task}: nothing loaded, so the spy recorded nothing"
        assert spy.admitted, f"{task}: no name reached the real resolution at all"
        assert spy.admitted <= real_allowed, (
            f"{task}: a name outside the exact allowlist reached the real "
            f"resolution: {sorted(spy.admitted - real_allowed)}"
        )
        assert expect_named in spy.seen, (
            f"{task}: {expect_named} never reached find_class, so this check saw nothing"
        )
        assert expect_named not in spy.admitted, (
            f"{task}: {expect_named} was RESOLVED, not stood in for"
        )
        leaked = [m for m in imported if m.split(".")[0] in ("torchvision", "ultralytics")]
        assert not leaked, f"{task}: a stubbed namespace was actually imported: {leaked}"
        checked += 1

    if not checked:
        pytest.skip("neither the -obb nor the -cls checkpoint is in the weight cache")


# ---------------------------------------------------------------------------
# M3, R:RESOLVEGADGET — getattr is inert, the real primitives stay refused
# ---------------------------------------------------------------------------


def test_getattr_resolves_to_something_inert_not_the_builtin(getattr_checkpoint: Path) -> None:
    """covers: M3, R:RESOLVEGADGET — a stand-in, never the attribute primitive."""
    from yowo.arch._weights import _extract_state_dict

    state = _extract_state_dict(getattr_checkpoint)
    assert state, "the checkpoint naming getattr yielded no tensors"

    resolved = _find_class(getattr_checkpoint, "__builtin__", "getattr")
    assert resolved is not getattr, "the loader handed back the real getattr"
    assert resolved is not builtins.getattr, "the loader handed back builtins.getattr"

    probe = _Secret()
    out = resolved(probe, "secret")
    assert out is not probe.secret, "the stand-in read an attribute off a real object"
    assert out != "sentinel", "the stand-in returned the real attribute value"


@pytest.mark.parametrize(
    ("module", "name", "obj"),
    [
        ("builtins", "eval", eval),
        ("builtins", "exec", exec),
        ("builtins", "__import__", __import__),
        ("os", "system", os.system),
    ],
    ids=["eval", "exec", "import", "os-system"],
)
def test_a_general_purpose_primitive_is_still_refused(
    tmp_path: Path, module: str, name: str, obj: object
) -> None:
    """covers: E3, R:RESOLVEGADGET — inert is for metadata, not for primitives.

    Widening what may be STUBBED must not widen what may be REACHED. These four
    are refused BY NAME, before the module they came from is imported — both as
    a real checkpoint names them (a serialiser rewrites `builtins` to
    `__builtin__`, and the shell primitive to `posix.system`) and under the
    literal E3 spelling, so no rewriting can smuggle one past the boundary.
    """
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    path = tmp_path / f"{name.strip('_')}.pt"
    torch.save({"ema": None, "model": None, "payload": obj}, path)

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(path)
    message = str(excinfo.value)
    assert "Refusing to load" in message, f"not refused by name: {message}"
    assert f".{name}'" in message, f"the refusal must name the exact global: {message}"

    with pytest.raises(ModelLoadError) as direct:
        _find_class(path, module, name)
    assert f"{module}.{name}" in str(direct.value), (
        f"the literal spelling {module}.{name} was not refused by name: {direct.value}"
    )


# ---------------------------------------------------------------------------
# M2 — torchvision is stubbed, never constructed
# ---------------------------------------------------------------------------


def test_no_torchvision_class_is_ever_constructed(torchvision_checkpoint: Path) -> None:
    """covers: M2 — a stand-in absorbs the transform; the real class never runs.

    The fake `Compose` is left installed in `sys.modules` for the load, so a
    loader that resolved the name would restore THIS class and its
    `__setstate__` would record it. Nothing is recorded, and the tensors still
    come out.
    """
    from yowo.arch._weights import _extract_state_dict

    state = _extract_state_dict(torchvision_checkpoint)
    assert state, "the checkpoint naming a torchvision transform yielded no tensors"
    assert not _TORCHVISION_CONSTRUCTED, (
        "the real torchvision class was constructed from checkpoint bytes"
    )

    resolved = _find_class(torchvision_checkpoint, "torchvision.transforms.transforms", "Compose")
    assert resolved is not _FakeCompose, "the loader resolved the torchvision name"
    assert isinstance(resolved, type) and issubclass(resolved, torch.nn.Module), (
        "a stubbed name must yield the inert stand-in"
    )


# ---------------------------------------------------------------------------
# M4, R:SILENTWIDEN — stubbed is not resolved, and the two stay apart
# ---------------------------------------------------------------------------


def test_the_resolved_allowlist_is_unchanged() -> None:
    """covers: M4, R:SILENTWIDEN — nothing moved from refused to RESOLVED.

    This node widens what may be STUBBED. `_ALLOWED` is asserted against the
    exact set recorded before it, entry by entry, so a stub cannot be smuggled
    in as an admission — and the widening is asserted to have landed somewhere
    else, because "unchanged" proves nothing on its own.
    """
    from yowo.arch import _weights

    expected_exact = frozenset(
        {
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch._utils", "_rebuild_tensor"),
            ("torch._utils", "_rebuild_parameter"),
            ("torch", "Size"),
            ("collections", "OrderedDict"),
        }
        | {
            (mod, name)
            for mod in ("builtins", "__builtin__")
            for name in ("set", "frozenset", "dict", "list", "tuple")
        }
    )
    assert expected_exact == _weights._ALLOWED_EXACT, (
        f"the exact allowlist changed: added={sorted(_weights._ALLOWED_EXACT - expected_exact)} "
        f"removed={sorted(expected_exact - _weights._ALLOWED_EXACT)}"
    )
    assert _weights._ALLOWED == _weights._ALLOWED_EXACT | _weights._ALLOWED_TORCH, (
        "_ALLOWED is no longer exactly the two recorded halves"
    )

    for gadget in ("getattr", "setattr", "eval", "exec", "__import__", "compile", "open"):
        assert not any(name == gadget for _, name in _weights._ALLOWED), (
            f"{gadget} was admitted to the RESOLVED allowlist"
        )
    assert not any(module.startswith("torchvision") for module, _ in _weights._ALLOWED), (
        "a torchvision name was admitted to the RESOLVED allowlist"
    )

    # "Unchanged" is only meaningful once the widening landed elsewhere.
    assert "torchvision." in _weights._STUBBED_PREFIXES, (
        "torchvision was not stubbed, so nothing was widened and this check is vacuous"
    )
    assert ("__builtin__", "getattr") in _weights._INERT_EXACT, (
        "getattr was not made inert, so nothing was widened and this check is vacuous"
    )


def test_resolved_and_stubbed_are_separate_constants() -> None:
    """covers: M4 — a later reader must not be able to confuse the two.

    Separate objects, disjoint contents, separate comments, and — the part that
    is not source-reading — separate BEHAVIOUR: an allowlisted name yields the
    real object, an inert or stubbed one yields the stand-in.
    """
    from yowo.arch import _weights

    resolved_set = _weights._ALLOWED
    inert_set = _weights._INERT_EXACT
    stubbed = _weights._STUBBED_PREFIXES

    assert resolved_set is not inert_set, "the resolved and inert sets are the same object"
    assert not (resolved_set & inert_set), (
        f"a name is both RESOLVED and INERT: {sorted(resolved_set & inert_set)}"
    )
    assert not [m for m, _ in resolved_set if m.startswith(tuple(stubbed))], (
        "a RESOLVED name lives under a stubbed prefix"
    )

    source = inspect.getsource(_weights)
    comments = {
        "_ALLOWED_EXACT": _comment_block_above(source, "_ALLOWED_EXACT"),
        "_INERT_EXACT": _comment_block_above(source, "_INERT_EXACT"),
        "_STUBBED_PREFIXES": _comment_block_above(source, "_STUBBED_PREFIXES"),
    }
    for constant, block in comments.items():
        assert block, f"{constant} carries no comment saying which permission it grants"
    assert len({tuple(block) for block in comments.values()}) == 3, (
        "two of the three constants share a comment block; the permissions read as one"
    )

    # One unpickler, three questions: the allowlisted name is IMPORTED, the
    # inert name and the stubbed one are both STOOD IN FOR by the same object.
    from yowo.arch._weights import _restricted_unpickler_module

    unpickler = _restricted_unpickler_module(Path("unused.pt")).Unpickler(io.BytesIO(b""))
    real = unpickler.find_class("collections", "OrderedDict")
    assert real is collections.OrderedDict, "an allowlisted name did not resolve"
    stand_in = unpickler.find_class("__builtin__", "getattr")
    assert stand_in is not real and stand_in is not builtins.getattr
    assert unpickler.find_class("torchvision.transforms.transforms", "Compose") is stand_in, (
        "the inert and stubbed branches hand back different stand-ins"
    )


# ---------------------------------------------------------------------------
# A4, E6 — absorb and continue, but never absorb a malformed file into success
# ---------------------------------------------------------------------------


def test_a_stand_in_absorbs_calls_without_aborting_the_load(
    torchvision_checkpoint: Path,
) -> None:
    """covers: A4 — a stand-in that RAISES ends the load partway and reports the
    truncated result as a complete one. Calling, indexing and attribute-setting
    must all be absorbed."""
    from yowo.arch._weights import _extract_state_dict

    stand_in_cls = _find_class(torchvision_checkpoint, "torchvision.transforms.transforms", "C")
    stand_in = stand_in_cls()
    assert stand_in(1, 2, key="value") is not None, "a stand-in refused to be called"
    assert stand_in[0] is not None, "a stand-in refused to be indexed"
    assert stand_in["key"] is not None, "a stand-in refused a string key"
    stand_in.attribute = "value"

    state = _extract_state_dict(torchvision_checkpoint)
    assert state, "the load ended early despite the stand-in absorbing"


@pytest.mark.parametrize(
    "kind",
    ["no-model-key", "empty-checkpoint", "stand-in-without-tensors"],
)
def test_a_checkpoint_with_no_tensors_still_fails(tmp_path: Path, kind: str) -> None:
    """covers: A4, E6 — the probe A4 names.

    Absorb-and-continue is the right answer for a stand-in and the wrong answer
    for a file. A checkpoint whose tensors are absent must FAIL, not return an
    empty mapping that every caller downstream reads as a complete load. The
    third case is E6 exactly: a stubbed global standing where the state_dict
    would have needed it, yielding an object with nothing in it.
    """
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    path = tmp_path / "empty.pt"
    if kind == "no-model-key":
        torch.save({"ema": None, "model": None}, path)
    elif kind == "empty-checkpoint":
        torch.save({}, path)
    else:
        _install_fake_torchvision()
        try:
            torch.save({"ema": _FakeCompose(), "model": None}, path)
        finally:
            _remove_fake_torchvision()

    with pytest.raises((ValueError, ModelLoadError)) as excinfo:
        result = _extract_state_dict(path)
        raise AssertionError(f"a checkpoint with no tensors loaded as {result!r}")
    message = str(excinfo.value)
    assert "no tensors" in message or "no 'model'" in message, (
        f"the failure must say the tensors are missing: {message}"
    )


# ---------------------------------------------------------------------------
# E4, R:CORRUPT — inert means invisible to the result
# ---------------------------------------------------------------------------


def test_the_ten_detection_variants_are_byte_identical() -> None:
    """covers: E4, R:CORRUPT — every tensor of every pinned variant, unchanged.

    The baseline is a per-variant SHA-256 over every key, dtype, shape and
    tensor BYTE, recorded by `scripts/measure_checkpoint_globals.py` BEFORE this
    node changed the loader. A stand-in that changed which tensors come back, or
    their values, moves a digest.
    """
    import json

    from yowo.arch._weights import _extract_state_dict
    from yowo.models._registry import list_available
    from yowo.models._weights import _CACHE_DIR

    manifest = json.loads(_MANIFEST.read_text())
    recorded: dict[str, dict[str, Any]] = manifest.get("state_dict_digests", {})
    assert recorded, (
        "no per-tensor baseline in the recorded measurement; re-run "
        "scripts/measure_checkpoint_globals.py so R:CORRUPT has something to "
        "compare against"
    )
    assert len(recorded) == len(list_available()), (
        "the baseline does not cover all ten pinned detection variants"
    )

    root = Path(os.environ.get("YOWO_CACHE_DIR", str(_CACHE_DIR)))
    checked = 0
    for meta in list_available():
        assert meta.weight_stem in recorded, f"{meta.weight_stem} has no recorded baseline"
        path = root / meta.family.value / meta.size.value / f"{meta.weight_stem}.pt"
        if not path.is_file():
            continue
        state = _extract_state_dict(path)
        row = recorded[meta.weight_stem]
        assert len(state) == row["entries"], (
            f"{meta.weight_stem}: {len(state)} entries now, {row['entries']} before"
        )
        assert _state_dict_digest(state) == row["sha256"], (
            f"{meta.weight_stem}: the tensors this loader returns are no longer "
            "byte-identical to the ones recorded before this node"
        )
        checked += 1

    if not checked:
        pytest.skip("no pinned detection checkpoint is in the weight cache")


# ---------------------------------------------------------------------------
# A5 — most specific to least, refusal last
# ---------------------------------------------------------------------------


def test_find_class_consults_the_exact_set_before_any_prefix(
    monkeypatch: pytest.MonkeyPatch, getattr_checkpoint: Path
) -> None:
    """covers: A5 — a broad rule can never answer what a narrow rule answered.

    Behaviourally: with a prefix widened to cover an allowlisted module, the
    exact set must still win, or the narrowing `narrow-loader-allowlist`
    performed is silently undone. Structurally: the three tests appear in the
    frozen order and the refusal is last.
    """
    import torch._utils

    from yowo.arch import _weights

    monkeypatch.setattr(_weights, "_STUBBED_PREFIXES", (*_weights._STUBBED_PREFIXES, "torch."))
    resolved = _find_class(getattr_checkpoint, "torch._utils", "_rebuild_tensor_v2")
    assert resolved is torch._utils._rebuild_tensor_v2, (
        "a stubbed prefix outranked the exact allowlist; the boundary is undone"
    )
    state = _weights._extract_state_dict(getattr_checkpoint)
    assert state, "the exact set no longer reaches the tensor rebuild helpers"
    monkeypatch.undo()

    tree = ast.parse(_find_class_source())
    body: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "find_class":
            body = node.body
    assert body, "no find_class in the production unpickler"

    order: list[str] = []
    for statement in body:
        if isinstance(statement, ast.If):
            text = ast.dump(statement.test)
            if "_ALLOWED" in text:
                order.append("allowed")
            elif "_INERT_EXACT" in text:
                order.append("inert")
            elif "_STUBBED_PREFIXES" in text:
                order.append("stubbed")
        if isinstance(statement, ast.Raise):
            order.append("refuse")
    assert order == ["allowed", "inert", "stubbed", "refuse"], (
        f"find_class decides in the order {order}; A5 freezes exact allowlist, "
        "then inert names, then stubbed prefixes, then refuse"
    )


# ---------------------------------------------------------------------------
# E5 — the legacy non-zip path stays closed
# ---------------------------------------------------------------------------


class _MkdirPayload:
    """A REDUCE naming `os.mkdir` — the shape of an attack on the legacy path."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, ...]:
        return (os.mkdir, (str(self.marker),))


class _InertReachPayload:
    """A REDUCE naming the newly-INERT `getattr`, on the legacy header path."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (getattr, ({"a": 1}, "keys"))


def test_the_legacy_format_probe_still_refuses_before_anything_runs(tmp_path: Path) -> None:
    """covers: E5 — the path `checkpoint-loader` closed stays closed.

    `torch.load`'s legacy (non-zip) reader calls `pickle_module.load` on the file
    header BEFORE an `Unpickler` exists, so a stock `load` there ran a
    checkpoint's first REDUCE before `find_class` was ever consulted. Making a
    name inert must not reopen that: a legacy file reaching the newly-inert
    `getattr` is still rejected, and reaches nothing.
    """
    import pickle

    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    marker = tmp_path / "pwned"
    hostile = tmp_path / "legacy.pt"
    hostile.write_bytes(pickle.dumps(_MkdirPayload(marker), protocol=2))

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(hostile)
    assert not marker.exists(), "the first object ran before the allowlist was consulted"
    assert "Refusing to load" in str(excinfo.value), f"not refused by name: {excinfo.value}"
    assert "mkdir" in str(excinfo.value)

    # The same path, reaching a name this node makes INERT rather than refusing.
    # It must still be rejected — as a malformed file, since nothing ran — and
    # never read as a checkpoint.
    inert_reach = tmp_path / "legacy-inert.pt"
    inert_reach.write_bytes(pickle.dumps(_InertReachPayload(), protocol=2))
    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(inert_reach)
    assert "Refusing to load" not in str(excinfo.value), (
        "getattr is still refused outright, so the inert branch was never exercised here"
    )
    assert "Could not read checkpoint" in str(excinfo.value), (
        f"a legacy file must be rejected, not read: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# M6, A2, A6 — every decision names its measurement
# ---------------------------------------------------------------------------


def test_every_decision_here_names_its_measurement(tmp_path: Path) -> None:
    """covers: M6, A2, A6 — no name is stubbed or made inert by reasoning alone.

    Each one traces to an observation recorded by a committed script, naming the
    checkpoint that named it. And the refusal a maintainer reads must still name
    the exact global, say which of the two remedies applies, and point at the
    script that produces the observation.
    """
    import json

    from yowo.arch import _weights
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    assert _MEASURE_SCRIPT.is_file(), "the measurement is not regenerable by a committed script"
    manifest = json.loads(_MANIFEST.read_text())
    observations: dict[str, list[str]] = manifest["observations"]
    assert manifest["measured_by"] == "scripts/measure_checkpoint_globals.py"

    for module, name in sorted(_weights._INERT_EXACT):
        observers = observations.get(f"{module}.{name}", [])
        assert observers, f"{module}.{name} is inert with no recorded observation behind it"
        assert any(not o.startswith("arch:") for o in observers), (
            f"{module}.{name} was only ever named by our own serialised model, "
            "never by a checkpoint; A2 admits only what a checkpoint names"
        )

    torchvision_observers = [
        obs for global_name, obs in observations.items() if global_name.startswith("torchvision.")
    ]
    assert torchvision_observers, (
        "`torchvision.` was added to the stubbed prefixes with no recorded observation"
    )
    assert any("cls" in observer for obs in torchvision_observers for observer in obs), (
        "no -cls checkpoint is recorded as naming a torchvision global"
    )
    assert any("obb" in observer for observer in observations.get("__builtin__.getattr", [])), (
        "no -obb checkpoint is recorded as naming __builtin__.getattr"
    )

    path = tmp_path / "unknown.pt"
    torch.save({"ema": None, "model": None, "payload": torch.nn.LSTM(2, 2)}, path)
    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(path)
    message = str(excinfo.value)
    assert "torch.nn.modules.rnn.LSTM" in message, "the refusal must name the exact global"
    assert "scripts/measure_checkpoint_globals.py" in message, (
        "the refusal must point at the script that produces the observation"
    )
    assert "stub" in message.lower(), (
        "the refusal must say a stand-in is the other remedy, or the next maintainer "
        "reads widening the allowlist as the only fix"
    )
    assert "allowlist" in message.lower(), "the refusal must still name the allowlist remedy"


# ---------------------------------------------------------------------------
# A1 — the boundary cannot tell whose checkpoint it is
# ---------------------------------------------------------------------------


def test_the_threat_model_covers_a_user_supplied_checkpoint(tmp_path: Path) -> None:
    """covers: A1 — an unpinned local file is treated exactly as an official one.

    A pin says a file is the one that was published; it never says it is safe,
    and `spec.weights_path` bypasses pinning entirely. So the same bytes must
    get the same answer from the same boundary whatever their provenance.
    """
    from yowo.arch._weights import _extract_state_dict, load_verified_state_dict
    from yowo.errors import ModelLoadError

    official_looking = tmp_path / "cache" / "yolo11" / "n" / "yolo11n.pt"
    official_looking.parent.mkdir(parents=True)
    user_supplied = tmp_path / "my_own_weights.pt"

    torch.save({"ema": None, "model": None, "payload": os.system}, official_looking)
    user_supplied.write_bytes(official_looking.read_bytes())

    messages = []
    for path in (official_looking, user_supplied):
        with pytest.raises(ModelLoadError) as excinfo:
            _extract_state_dict(path)
        messages.append(str(excinfo.value).replace(str(path), "<path>"))
    assert messages[0] == messages[1], "the boundary answered differently by provenance"

    # And the unpinned entry point reaches the same boundary — `raw_digest=None`
    # means "nobody claims whose these are", never "nobody checks what they name".
    with pytest.raises(ModelLoadError):
        load_verified_state_dict(user_supplied, raw_digest=None)

    # The widening this node performs applies to a user file identically.
    _install_fake_torchvision()
    _TORCHVISION_CONSTRUCTED.clear()
    try:
        mine = tmp_path / "mine.pt"
        torch.save({"ema": _tensor_bearing(), "model": None, "t": _FakeCompose()}, mine)
        state = _extract_state_dict(mine)
        assert state, "a user-supplied checkpoint naming a stubbed global did not load"
        assert not _TORCHVISION_CONSTRUCTED, (
            "a user-supplied checkpoint got its torchvision class constructed"
        )
    finally:
        _remove_fake_torchvision()
        _TORCHVISION_CONSTRUCTED.clear()
