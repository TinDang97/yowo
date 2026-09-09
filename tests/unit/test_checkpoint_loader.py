"""The checkpoint loader's trust boundary.

An ultralytics `.pt` stores its weights inside a pickled `nn.Module`, so reading
one has always meant letting the file choose what gets imported. That makes
`ultralytics` an undeclared runtime dependency AND hands checkpoint authors an
import primitive. These checks bind the loader that fixes both.

Every fixture here is built in-process: a checkpoint naming
`ultralytics.nn.tasks.DetectionModel` is produced by registering that module only
while saving, then removing it. No real weights file, no network, no AGPL artifact.

Authored red under ADD task `checkpoint-loader` — see `.add/tasks/checkpoint-loader.md`.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


class _Stand_in(torch.nn.Module):
    """Stands in for ultralytics' DetectionModel while the fixture is written."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, bias=False))


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


@pytest.fixture
def ultralytics_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint whose `ema` is an `ultralytics.nn.tasks.DetectionModel`."""
    _install_fake_ultralytics()
    try:
        path = tmp_path / "fake.pt"
        torch.save({"ema": _Stand_in(), "model": None}, path)
    finally:
        _remove_fake_ultralytics()
    return path


@pytest.fixture
def hostile_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint carrying a global that is neither torch nor ultralytics."""
    import os

    path = tmp_path / "hostile.pt"
    torch.save({"ema": None, "model": None, "payload": os.system}, path)
    return path


@pytest.fixture
def _no_ultralytics(monkeypatch: pytest.MonkeyPatch):
    """Make `ultralytics` unimportable, as it is on any real install."""
    real_import = builtins.__import__

    def blocked(name: str, *a: object, **k: object):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    _remove_fake_ultralytics()
    monkeypatch.setattr(builtins, "__import__", blocked)


def test_loads_without_ultralytics_importable(
    ultralytics_checkpoint: Path, _no_ultralytics: None
) -> None:
    """covers: M1, A5 — the headline bug.

    `pip install yowo[pytorch]` does not install ultralytics, so this is the
    state every real user is in.
    """
    from yowo.arch._weights import _extract_state_dict

    sd = _extract_state_dict(ultralytics_checkpoint)
    assert sd, "no tensors extracted"
    assert all(isinstance(v, torch.Tensor) for v in sd.values())


def test_refuses_a_foreign_global_by_name(hostile_checkpoint: Path, _no_ultralytics: None) -> None:
    """covers: M2, R:ARBITRARY_IMPORT, E2 — the checkpoint must not pick imports."""
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(hostile_checkpoint)
    assert "system" in str(excinfo.value), "the refusal must name the offending global"


@pytest.fixture
def ema_and_model_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint carrying distinguishable `ema` and `model` weights."""
    _install_fake_ultralytics()
    try:
        ema, plain = _Stand_in(), _Stand_in()
        with torch.no_grad():
            ema.model[0].weight.fill_(1.0)
            plain.model[0].weight.fill_(2.0)
        path = tmp_path / "both.pt"
        torch.save({"ema": ema, "model": plain}, path)
    finally:
        _remove_fake_ultralytics()
    return path


def test_ema_weights_win_over_model_weights(
    ema_and_model_checkpoint: Path, _no_ultralytics: None
) -> None:
    """covers: M4 — EMA preference must survive the refactor.

    Runs with ultralytics blocked so it is red for the same reason as M1 today,
    and still guards the preference once the loader works.
    """
    from yowo.arch._weights import _extract_state_dict

    path = ema_and_model_checkpoint

    sd = _extract_state_dict(path)
    weight = next(v for k, v in sd.items() if k.endswith("weight"))
    assert torch.allclose(weight, torch.ones_like(weight)), "EMA weights were not preferred"


def test_unreadable_checkpoint_raises_a_typed_yowo_error(
    tmp_path: Path, _no_ultralytics: None
) -> None:
    """covers: M5, A6, R:SILENT_PARTIAL — a typed error, never a raw UnpicklingError."""
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    path = tmp_path / "truncated.pt"
    path.write_bytes(b"not a checkpoint at all")

    with pytest.raises(ModelLoadError):
        _extract_state_dict(path)


# ---------------------------------------------------------------------------
# The enumerated torch allowlist — ADD task `narrow-loader-allowlist`
# ---------------------------------------------------------------------------
#
# `checkpoint-loader` froze `torch.nn.modules.` as a PREFIX rule. Every class
# under it is data-bearing today, but a prefix admits a namespace where the
# boundary should name a class. These checks bind the narrowing: the list is
# an enumerated `(module, name)` set, every entry traces to a RECORDED
# observation (scripts/checkpoint_globals_manifest.json, produced by
# scripts/measure_checkpoint_globals.py), and anything outside it is refused
# by name — including look-alike namespaces and torch's own unlisted layers.
#
# Authored red under `.add/tasks/narrow-loader-allowlist.md`.

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "scripts" / "checkpoint_globals_manifest.json"
_MEASURE_SCRIPT = _REPO / "scripts" / "measure_checkpoint_globals.py"
_CHANGE_REQUEST_PATH = ".add/tasks/narrow-loader-allowlist.md"


def _checkpoint_naming(tmp_path: Path, module: str, name: str) -> Path:
    """A checkpoint whose serialised stream names ``module.name``.

    The module does not have to exist: a throwaway class is given that
    ``__module__`` / ``__qualname__`` and a matching module is registered in
    ``sys.modules`` only while ``torch.save`` resolves the global, then
    removed. So the file names e.g. ``torch.nn.parameterfoo.X`` without such a
    module ever being importable — exactly the shape a hostile file has.
    """
    cls = type(name, (), {})
    cls.__module__ = module
    cls.__qualname__ = name
    fake = types.ModuleType(module)
    setattr(fake, name, cls)
    previous = sys.modules.get(module)
    sys.modules[module] = fake
    try:
        path = tmp_path / f"{module}.{name}.pt"
        torch.save({"ema": None, "model": None, "payload": cls()}, path)
    finally:
        if previous is None:
            del sys.modules[module]
        else:
            sys.modules[module] = previous
    return path


def _string_literals(node: object, namespace: object) -> list[str]:
    """Every string a `startswith` argument could match, literal or via a module name."""
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Tuple):
        return [s for elt in node.elts for s in _string_literals(elt, namespace)]
    if isinstance(node, ast.Name):
        value = getattr(namespace, node.id, None)
        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, list, set, frozenset)):
            return [v for v in value if isinstance(v, str)]
    return []


def test_no_rule_admits_a_torch_namespace() -> None:
    """covers: M1, R:PREFIX — neither `_ALLOWED_PREFIXES` nor any `startswith`
    over a torch namespace survives in the enforcement point."""
    import ast
    import inspect

    from yowo.arch import _weights

    assert not hasattr(_weights, "_ALLOWED_PREFIXES"), "the prefix rule is still defined"
    allowed = getattr(_weights, "_ALLOWED_TORCH", None)
    assert isinstance(allowed, frozenset) and allowed, "no enumerated torch set"
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(s, str) for s in pair)
        for pair in allowed
    ), "the torch set must hold exact (module, name) pairs"

    tree = ast.parse(inspect.getsource(_weights._restricted_unpickler_module))
    admitted_by_prefix: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "startswith"
        ):
            for arg in node.args:
                admitted_by_prefix.extend(
                    s for s in _string_literals(arg, _weights) if s.startswith("torch")
                )
    assert not admitted_by_prefix, (
        f"a torch namespace is still admitted by prefix: {admitted_by_prefix}"
    )


def test_allowlist_matches_the_recorded_measurement() -> None:
    """covers: M2, M4, R:GUESSED — every entry traces to a recorded observation and
    no entry is unaccounted for, in both directions."""
    import json

    from yowo.arch._weights import _ALLOWED_TORCH

    assert _MANIFEST.exists(), f"no recorded measurement at {_MANIFEST}"
    manifest = json.loads(_MANIFEST.read_text())

    recorded = frozenset((m, n) for m, n in manifest["allowed_torch"])
    assert recorded == _ALLOWED_TORCH, (
        f"committed list drifted from the measurement: "
        f"unrecorded={sorted(_ALLOWED_TORCH - recorded)} "
        f"missing={sorted(recorded - _ALLOWED_TORCH)}"
    )

    observations: dict[str, list[str]] = manifest["observations"]
    for module, name in sorted(_ALLOWED_TORCH):
        observers = observations.get(f"{module}.{name}", [])
        assert observers, f"{module}.{name} has no recorded observation behind it"

    # The manifest must say WHAT was measured, or the observations are unanchored:
    # every pinned detection variant with its verified digest, plus the arch sweep.
    from yowo.models._registry import list_available

    measured = {c["stem"]: c["sha256"] for c in manifest["checkpoints"]}
    for meta in list_available():
        assert meta.sha256 is not None, (
            f"{meta.weight_stem} is unpinned; it cannot define a boundary"
        )
        assert measured.get(meta.weight_stem) == meta.sha256, (
            f"{meta.weight_stem} was not measured at its pinned digest; re-run the measurement"
        )
    assert manifest["arch_sweep"], "the in-process arch sweep is not recorded"
    assert manifest["measured_at"], "no timestamp on the measurement"


@pytest.mark.parametrize(
    ("module", "name", "make"),
    [
        # E3 — `Module` itself: the base class every layer subclasses, which no
        # checkpoint names and which the prefix rule admitted anyway.
        ("torch.nn.modules.module", "Module", lambda: torch.nn.Module()),
        # A data-bearing torch layer that no shipped variant uses.
        ("torch.nn.modules.rnn", "LSTM", lambda: torch.nn.LSTM(2, 2)),
    ],
)
def test_unlisted_torch_class_is_refused(
    tmp_path: Path, module: str, name: str, make: object
) -> None:
    """covers: M3, A4, E3 — an unlisted `torch.nn.modules.*` name is refused
    exactly as any other unlisted name is: no warn-and-admit path."""
    from yowo.arch._weights import _ALLOWED_TORCH, _extract_state_dict
    from yowo.errors import ModelLoadError

    assert (module, name) not in _ALLOWED_TORCH, "fixture class is listed; pick another"
    path = tmp_path / f"{name}.pt"
    torch.save({"ema": None, "model": None, "payload": make()}, path)  # type: ignore[operator]

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(path)
    assert "Refusing to load" in str(excinfo.value)


def test_refusal_names_the_class_and_the_change_path(tmp_path: Path) -> None:
    """covers: M3, A6 — the reader is someone whose new upstream checkpoint just
    stopped loading; the message names the exact class and where to take it."""
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    path = tmp_path / "module.pt"
    torch.save({"ema": None, "model": None, "payload": torch.nn.Module()}, path)

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(path)
    message = str(excinfo.value)
    assert "torch.nn.modules.module.Module" in message, "the refusal must name the exact class"
    assert _CHANGE_REQUEST_PATH in message, "the refusal must point at the change-request path"
    assert str(path) in message, "the refusal must name the file"


@pytest.fixture
def import_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every module name the process tried to import while the test ran."""
    real_import = builtins.__import__
    seen: list[str] = []

    def spy(name: str, *a: object, **k: object):
        seen.append(name)
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", spy)
    return seen


@pytest.mark.parametrize(
    ("module", "name"),
    [
        # Admitted by `startswith("torch.nn.parameter")` today: the file names a
        # module that does not exist and the loader tries to IMPORT it.
        ("torch.nn.parameterfoo", "X"),
        ("torch.nn.modulesX", "Y"),
        ("torch.nn.modules_extra.conv", "Conv2d"),
    ],
)
def test_lookalike_torch_prefixes_are_refused(
    tmp_path: Path, import_log: list[str], module: str, name: str
) -> None:
    """covers: A3, E2 — a look-alike namespace is refused BY NAME, before any
    import is attempted. A refusal that only happens because the import failed
    is the prefix hole with a friendlier error."""
    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    path = _checkpoint_naming(tmp_path, module, name)
    del import_log[:]

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(path)
    assert "Refusing to load" in str(excinfo.value), (
        f"not refused by name; the loader got as far as: {excinfo.value}"
    )
    assert f"{module}.{name}" in str(excinfo.value)
    assert module not in import_log, f"the loader tried to import {module!r}"


class _ClsStand_in(torch.nn.Module):
    """Stands in for ultralytics' ClassificationModel while the fixture is written.

    Its head carries the three torch layers no detection checkpoint names:
    `AdaptiveAvgPool2d`, `Dropout` and `Linear` (E1). There is deliberately no
    `nn.Flatten` — neither upstream's `Classify` nor ours constructs one (both
    call `.flatten(1)` on the tensor), so the measurement never observed it and
    the list must not admit it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1, bias=False),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Dropout(p=0.0, inplace=True),
            torch.nn.Linear(4, 10),
        )


@pytest.fixture
def classification_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint whose `ema` is an `ultralytics.nn.tasks.ClassificationModel`."""
    pkg = types.ModuleType("ultralytics")
    nn = types.ModuleType("ultralytics.nn")
    tasks = types.ModuleType("ultralytics.nn.tasks")
    _ClsStand_in.__module__ = "ultralytics.nn.tasks"
    _ClsStand_in.__qualname__ = "ClassificationModel"
    tasks.ClassificationModel = _ClsStand_in  # type: ignore[attr-defined]
    pkg.nn = nn  # type: ignore[attr-defined]
    nn.tasks = tasks  # type: ignore[attr-defined]
    sys.modules.update({"ultralytics": pkg, "ultralytics.nn": nn, "ultralytics.nn.tasks": tasks})
    try:
        path = tmp_path / "fake-cls.pt"
        torch.save({"ema": _ClsStand_in(), "model": None}, path)
    finally:
        _remove_fake_ultralytics()
        _ClsStand_in.__module__ = __name__
        _ClsStand_in.__qualname__ = "_ClsStand_in"
    return path


def test_classification_layers_are_admitted(
    classification_checkpoint: Path, _no_ultralytics: None
) -> None:
    """covers: A2, E1 — the cls head loads. No detection variant names `Linear`;
    the enumerated set must still admit it, from the in-process arch sweep."""
    from yowo.arch._weights import _ALLOWED_TORCH, _extract_state_dict

    sd = _extract_state_dict(classification_checkpoint)
    assert "model.3.weight" in sd, f"Linear weights were not extracted: {sorted(sd)}"
    assert sd["model.3.weight"].shape == (10, 4)

    # And the admission is by enumeration, not by a namespace that happens to
    # cover the head: each layer is a listed pair with a recorded observation.
    head = {
        ("torch.nn.modules.linear", "Linear"),
        ("torch.nn.modules.dropout", "Dropout"),
        ("torch.nn.modules.pooling", "AdaptiveAvgPool2d"),
    }
    assert head <= _ALLOWED_TORCH, f"cls head layers missing from the set: {head - _ALLOWED_TORCH}"


def _load_measure_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure_checkpoint_globals", _MEASURE_SCRIPT)
    assert spec is not None and spec.loader is not None, f"no script at {_MEASURE_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measurement_script_downloads_nothing_in_the_unit_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """covers: A7 — the measurement is a maintainer action, never a CI job. Under
    CI it refuses before touching the network or the registry; importing it does
    nothing at all. The unit suite asserts the list against the RECORDED
    manifest (above) and fetches no corpus."""
    import requests

    def explode(*a: object, **k: object):
        raise AssertionError("the unit suite reached for the network")

    monkeypatch.setattr(requests, "get", explode)
    monkeypatch.setenv("CI", "1")

    script = _load_measure_script()  # import must have no side effects
    monkeypatch.setattr(script, "resolve_weights", explode)

    with pytest.raises(SystemExit) as excinfo:
        script.main([])
    assert excinfo.value.code not in (0, None), "the script ran under CI"


class _MkdirPayload:
    """A REDUCE naming `os.mkdir` — the shape of an attack on the legacy file path."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        import os

        return (os.mkdir, (str(self.marker),))


def test_legacy_format_file_runs_nothing_before_find_class(tmp_path: Path) -> None:
    """covers: checkpoint-loader M2, R:ARBITRARY_IMPORT — the parent contract.

    torch's legacy (non-zip) reader calls ``pickle_module.load`` on the file
    header BEFORE it builds our ``Unpickler``. With a stock ``load`` in the shim
    that first read is unrestricted: a non-zip file whose first object is a
    ``REDUCE`` runs it, and only then fails on "Invalid magic number". Found by
    the refute lens on `narrow-loader-allowlist`; the parent's M2 already
    forbids it, so the shim's ``load`` / ``loads`` must go through the same
    restricted ``find_class`` as everything else.
    """
    import pickle

    from yowo.arch._weights import _extract_state_dict
    from yowo.errors import ModelLoadError

    marker = tmp_path / "pwned"
    path = tmp_path / "legacy.pt"
    path.write_bytes(pickle.dumps(_MkdirPayload(marker), protocol=2))

    with pytest.raises(ModelLoadError) as excinfo:
        _extract_state_dict(path)
    assert not marker.exists(), "the first object ran before the allowlist was consulted"
    assert "Refusing to load" in str(excinfo.value), f"not refused by name: {excinfo.value}"
    assert "mkdir" in str(excinfo.value)
