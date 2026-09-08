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
