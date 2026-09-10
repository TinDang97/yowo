"""The verified digest reaches the loader, closing the resolve-to-load window.

`weight-integrity` made `resolve_weights` verify a cached weight against its
pinned SHA-256. It then returns a bare `Path`, and `load_verified_state_dict`
re-hashes that path -- but only to KEY the converted sidecar, never to re-check it
against the pin. Anything with write access to the cache between those two calls
can substitute the file: the new bytes hash to a different key, the sidecar
therefore misses, and the substituted checkpoint is handed straight to
`_extract_state_dict` -- the one executing-reader deserialisation in the system --
without ever being compared to anything.

`load_verified_state_dict` already TAKES a `raw_digest`; nothing has ever passed
one. So on the default path it computes the value from the very file it is meant
to authenticate, which matches by construction. These checks bind the pin to the
loader and make that tautology impossible.

Every fixture is built in-process, following `test_checkpoint_loader.py`: a
checkpoint naming `ultralytics.nn.tasks.DetectionModel` is produced by registering
that module only while saving, then removing it. No real weights file, no network,
no AGPL artifact.

Authored red under ADD task `verified-digest-threading`.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yowo.models._weights import WeightIntegrityError
from yowo.types import ModelFamily, ModelSize, ModelSpec

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
    """The file resolution verified: its path and the digest it hashed to."""
    path = tmp_path / "yolo11n.pt"
    return path, _write_checkpoint(path, fill=0.25)


def _substitute(path: Path) -> str:
    """Overwrite an already-verified path, as anyone with cache write access could."""
    return _write_checkpoint(path, fill=0.75)


# ---------------------------------------------------------------------------
# R:SELFKEYED -- a digest computed from the file under test proves nothing
# ---------------------------------------------------------------------------


def test_computed_digest_never_satisfies_verification(genuine: tuple[Path, str]) -> None:
    """covers: R:SELFKEYED, M2 -- verification compares against the PIN, not the file.

    The whole defect in one check. The file at `path` was verified as `pinned`;
    it is then swapped. If the loader authenticates a checkpoint using a digest
    it computed from that same checkpoint, the comparison is a tautology and can
    never fail. Proving it is not self-keyed means proving the raised message
    reports two DIFFERENT digests: an expected one that came from the caller's
    pin, and an actual one that came from the bytes on disk.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine
    substituted = _substitute(path)
    assert substituted != pinned, "fixture failed to change the bytes"

    with pytest.raises(WeightIntegrityError) as exc:
        load_verified_state_dict(path, raw_digest=pinned)

    message = str(exc.value)
    assert pinned in message, "the expected digest is not the caller's pin"
    assert substituted in message, "the actual digest is not the bytes on disk"


def test_swapped_file_after_resolution_is_refused(genuine: tuple[Path, str]) -> None:
    """covers: M1 -- a file substituted in the resolve-to-load window must not load.

    No sidecar exists, so this is the path that converts. Today it converts the
    substituted bytes and returns them as if they were the pinned model.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine
    _substitute(path)
    assert not path.with_suffix(".state_dict.pt").exists(), "fixture left a sidecar behind"

    with pytest.raises(WeightIntegrityError):
        load_verified_state_dict(path, raw_digest=pinned)


def test_conversion_is_never_reached_on_a_mismatch(genuine: tuple[Path, str]) -> None:
    """covers: R:CONVERTFIRST -- the comparison runs strictly before the conversion.

    Order is the entire control here. `_extract_state_dict` is the one place in
    the system that reads a checkpoint with a reader that can execute; reaching
    it and *then* discovering the digest was wrong means the payload already ran.
    """
    from yowo.arch import _weights as arch_weights

    path, pinned = genuine
    _substitute(path)

    # side_effect rather than return_value: reaching the conversion at all is the
    # failure, and a spy that quietly returns would let the sidecar write succeed
    # and report a vaguer red.
    tripwire = AssertionError("conversion ran before the digest was compared")
    with (
        patch.object(arch_weights, "_extract_state_dict", side_effect=tripwire) as convert,
        pytest.raises(WeightIntegrityError),
    ):
        arch_weights.load_verified_state_dict(path, raw_digest=pinned)

    convert.assert_not_called()


# ---------------------------------------------------------------------------
# M5 -- re-verify only where a conversion would actually happen
# ---------------------------------------------------------------------------


def test_sidecar_hit_on_the_pin_does_not_rehash_the_raw_file(genuine: tuple[Path, str]) -> None:
    """covers: M5 -- the steady-state path pays for ONE hash of the raw file, never two.

    Partly superseded by ADD task `sidecar-not-self-attesting`, and the name is
    kept only so this node's gate can still find it. As first written this check
    asserted the pinned fast path touched the raw file at all -- neither hashing
    it nor comparing it to the pin. That turned out to be the defect rather than
    the invariant: the sidecar was then keyed on the PIN, which is public and
    printed in `_registry.py`, so anything able to write the weight cache could
    plant a `.state_dict.pt` naming the published digest and have arbitrary
    tensors served with no read of the checkpoint at all. The raw file is now
    measured once, unconditionally, and the sidecar is believed only when it
    names that measurement.

    What survives, and is what this check was really defending, is the cost
    bound: ONE read of the raw file per load. `verify_digest` must consume the
    measurement already taken rather than hashing 5-110 MB a second time.
    """
    from yowo.arch import _weights as arch_weights

    path, pinned = genuine
    tensors = {"model.0.conv.weight": torch.zeros(1)}
    torch.save({"raw_sha256": pinned, "tensors": tensors}, path.with_suffix(".state_dict.pt"))

    hashed: list[Path] = []
    real_digest = arch_weights._raw_digest

    def _spy(target: Path) -> str:
        hashed.append(target)
        return real_digest(target)

    # A tripwire rather than a stub: the whole point of the sidecar is that the
    # executing reader runs once ever, so reaching it here is the failure.
    tripwire = AssertionError("the fast path fell through to the executing reader")
    with (
        patch.object(arch_weights, "_raw_digest", side_effect=_spy),
        patch.object(arch_weights, "_extract_state_dict", side_effect=tripwire) as convert,
    ):
        got = arch_weights.load_verified_state_dict(path, raw_digest=pinned)

    assert set(got) == set(tensors), "the sidecar did not serve the load"
    convert.assert_not_called()
    assert hashed == [path], f"the raw file was hashed {len(hashed)}x on the fast path"


def test_stale_sidecar_falls_through_to_the_pin_comparison(genuine: tuple[Path, str]) -> None:
    """covers: M1, M5 -- a sidecar keyed on other bytes must not shortcut the check.

    The sidecar misses (it names a digest the file no longer has), so a
    conversion is about to happen -- which is exactly when the pin must be
    re-compared.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, pinned = genuine
    torch.save(
        {"raw_sha256": "0" * 64, "tensors": {"stale": torch.zeros(1)}},
        path.with_suffix(".state_dict.pt"),
    )
    _substitute(path)

    with pytest.raises(WeightIntegrityError):
        load_verified_state_dict(path, raw_digest=pinned)


# ---------------------------------------------------------------------------
# A4 / M3 -- `raw_digest=None` still means UNPINNED, and still loads
# ---------------------------------------------------------------------------


def test_unpinned_load_converts_without_claiming_verification(genuine: tuple[Path, str]) -> None:
    """covers: A4, M3, R:SELFKEYED -- an unpinned model loads, and verifies nothing.

    An explicit `spec.weights_path`, and every `-cls` / `-obb` registry entry,
    carry `sha256=None`. There is no pinned value to compare against, so the
    honest behaviour is to convert and key the sidecar on the computed digest --
    and to make no verification claim at all. Calling `verify_digest` with a
    self-computed value here is precisely the tautology R:SELFKEYED forbids.
    """
    from yowo.arch import _weights as arch_weights

    path, _pinned = genuine

    with patch.object(arch_weights, "verify_digest") as verify:
        state = arch_weights.load_verified_state_dict(path, raw_digest=None)

    assert state, "an unpinned checkpoint must still load"
    verify.assert_not_called()


def test_unpinned_sidecar_is_still_keyed_on_the_computed_digest(
    genuine: tuple[Path, str],
) -> None:
    """covers: A4, M5 -- unpinned loads keep today's sidecar reuse.

    Dropping the computed key would make every `-cls` and `-obb` load reconvert
    forever, which is a performance regression dressed as a security fix.
    """
    from yowo.arch._weights import load_verified_state_dict

    path, _pinned = genuine
    load_verified_state_dict(path, raw_digest=None)

    sidecar = path.with_suffix(".state_dict.pt")
    assert sidecar.exists(), "no sidecar written for an unpinned load"
    blob = torch.load(sidecar, map_location="cpu", weights_only=True)
    assert blob["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# S2 / A2 -- all three load paths take a pin and forward it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_name",
    ["load_weights", "load_classify_weights", "load_obb_weights"],
)
def test_every_load_path_forwards_the_pin(func_name: str, tmp_path: Path) -> None:
    """covers: A2, M1, S2 -- cls and obb pass the same gate as detection.

    All three call `load_verified_state_dict` and all three reach the executing
    reader. Covering detection alone would leave two open paths while claiming
    the window shut.
    """
    from yowo.arch import _weights as arch_weights

    func = getattr(arch_weights, func_name)
    assert "raw_digest" in inspect.signature(func).parameters, (
        f"{func_name} cannot carry a pin at all"
    )

    path = tmp_path / "w.pt"
    path.write_bytes(b"unread -- the loader is stubbed")
    model = torch.nn.Linear(1, 1)

    with patch.object(arch_weights, "load_verified_state_dict", return_value={}) as loader:
        func(model, path, raw_digest="deadbeef")

    assert loader.call_count == 1
    assert loader.call_args.kwargs.get("raw_digest") == "deadbeef", (
        f"{func_name} accepted a pin and dropped it on the floor"
    )


# ---------------------------------------------------------------------------
# S3 / A7 -- the backend reads the pin from the registry, not from a wider Protocol
# ---------------------------------------------------------------------------


def _hw() -> MagicMock:
    hw = MagicMock()
    hw.libraries.torch_version = "2.0.0"
    hw.has_nvidia_gpu = False
    hw.libraries.torch_cuda_available = False
    return hw


def _stub_model() -> MagicMock:
    model = MagicMock()
    model.fuse.return_value = model
    model.eval.return_value = model
    model.to.return_value = model
    return model


def _meta(sha256: str | None) -> MagicMock:
    meta = MagicMock()
    meta.sha256 = sha256
    meta.num_classes = 1000
    meta.input_height = 224
    meta.input_width = 224
    return meta


def test_backend_threads_the_registry_pin_for_detection() -> None:
    """covers: S3, A1 -- the compared digest is the registry pin for the loaded spec.

    Never the caller's word: `model_path` is an argument anyone can choose, so
    authenticating it against something derived from itself, or from the caller,
    authenticates nothing.
    """
    from yowo.backends._pytorch import PyTorchBackend
    from yowo.models._registry import get

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    backend = PyTorchBackend(_hw(), model_spec=spec)
    expected = get(ModelFamily.YOLO11, ModelSize.NANO).sha256
    assert expected, "yolo11n has no registry pin; fixture assumption broken"

    with (
        patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
        patch("yowo.arch.build_model", return_value=_stub_model()),
        patch("yowo.arch._weights.load_weights") as load_fn,
    ):
        backend.load("cached.pt", device="cpu")

    assert load_fn.call_args.kwargs.get("raw_digest") == expected


@pytest.mark.parametrize(
    ("task", "builder", "loader", "registry_get"),
    [
        (
            "classify",
            "yowo.arch.build_classify_model",
            "yowo.arch._weights.load_classify_weights",
            "yowo.models._registry.get_cls",
        ),
        (
            "obb",
            "yowo.arch.build_obb_model",
            "yowo.arch._weights.load_obb_weights",
            "yowo.models._registry.get_obb",
        ),
    ],
)
def test_backend_threads_the_registry_pin_for_cls_and_obb(
    task: str, builder: str, loader: str, registry_get: str
) -> None:
    """covers: S3, A2 -- the cls and obb branches thread their own metas' pins.

    Both are `sha256=None` in the registry today, so this asserts the wiring
    rather than a value: the day one of those entries is pinned, the gate closes
    on it without another change here.
    """
    from yowo.backends._pytorch import PyTorchBackend

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task=task)
    backend = PyTorchBackend(_hw(), model_spec=spec)

    with (
        patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
        patch(registry_get, return_value=_meta("a" * 64)),
        patch(builder, return_value=_stub_model()),
        patch(loader) as load_fn,
    ):
        backend.load("cached.pt", device="cpu")

    assert load_fn.call_args.kwargs.get("raw_digest") == "a" * 64


def test_backend_sends_no_pin_for_an_explicit_weights_path(tmp_path: Path) -> None:
    """covers: A4, S3 -- a user-supplied checkpoint is not the file the registry pinned.

    Comparing a fine-tuned local weight against the official yolo11n digest would
    refuse every custom deployment on upgrade. `weights_path` set means unpinned.
    """
    from yowo.backends._pytorch import PyTorchBackend

    local = tmp_path / "finetuned.pt"
    local.write_bytes(b"a user's own weights")
    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, weights_path=local)
    backend = PyTorchBackend(_hw(), model_spec=spec)

    with (
        patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
        patch("yowo.arch.build_model", return_value=_stub_model()),
        patch("yowo.arch._weights.load_weights") as load_fn,
    ):
        backend.load(local, device="cpu")

    # The kwarg must be PRESENT and None, not absent. Accepting absence would let
    # this pass against today's code, which threads no pin at all -- an assertion
    # that is already true proves nothing about the branch it claims to cover.
    assert "raw_digest" in load_fn.call_args.kwargs, "no pin decision was made at all"
    assert load_fn.call_args.kwargs["raw_digest"] is None


def test_integrity_failure_keeps_its_type_through_the_backend() -> None:
    """covers: A6 -- one integrity error, with one message, wherever it is raised.

    `PyTorchBackend.load` ends in a blanket `except Exception -> BackendLoadError`.
    Letting it swallow this would give the same substitution two different types
    depending only on whether it was caught during resolution or during load, and
    a reader could not tell which control fired.
    """
    from yowo.backends._pytorch import PyTorchBackend

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    backend = PyTorchBackend(_hw(), model_spec=spec)
    boom = WeightIntegrityError("Weight failed integrity check: cached.pt")

    with (
        patch("yowo.backends._pytorch.PyTorchBackend._resolve_device", return_value="cpu"),
        patch("yowo.arch.build_model", return_value=_stub_model()),
        patch("yowo.arch._weights.load_weights", side_effect=boom),
        pytest.raises(WeightIntegrityError),
    ):
        backend.load("cached.pt", device="cpu")


# ---------------------------------------------------------------------------
# Invariant guards -- green before this task and required to stay green
# ---------------------------------------------------------------------------


def test_backend_load_signatures_stay_identical() -> None:
    """covers: A7 -- the pin must not widen a Protocol five backends share.

    Only PyTorch can use a checkpoint digest. Threading it through `Backend.load`
    would put a parameter on ONNX, OpenVINO, CoreML and TensorRT that none of
    them can honour, and every future backend would inherit the lie.
    """
    from yowo.backends import InferenceBackend
    from yowo.backends._coreml import CoreMLBackend
    from yowo.backends._onnx import OnnxBackend
    from yowo.backends._openvino import OpenVinoBackend
    from yowo.backends._pytorch import PyTorchBackend
    from yowo.backends._tensorrt import TensorRTBackend

    reference = str(inspect.signature(InferenceBackend.load))
    for backend in (
        PyTorchBackend,
        OnnxBackend,
        OpenVinoBackend,
        CoreMLBackend,
        TensorRTBackend,
    ):
        assert str(inspect.signature(backend.load)) == reference, (
            f"{backend.__name__}.load drifted from the shared Protocol signature"
        )


def test_resolve_weights_signature_is_unchanged() -> None:
    """covers: A7 -- resolution stays a bare `-> Path`, mocked at ~60 test sites.

    Returning the pin from `resolve_weights` would have been the obvious design
    and would have broken every `return_value=Path(...)` mock in the suite. The
    pin is read from the registry instead precisely to avoid that.
    """
    from yowo.models._weights import resolve_weights

    sig = inspect.signature(resolve_weights)
    assert list(sig.parameters) == ["spec", "cache_dir"]
    assert sig.return_annotation in (Path, "Path")


def test_load_verified_state_dict_signature_is_unchanged() -> None:
    """covers: S1 -- the argument already existed; only its wiring was missing.

    Nothing about the fix needs a new parameter, and adding one would suggest the
    old signature was incapable rather than unused.
    """
    from yowo.arch._weights import load_verified_state_dict

    params = inspect.signature(load_verified_state_dict).parameters
    assert list(params) == ["checkpoint_path", "raw_digest"]
    assert params["raw_digest"].default is None


def test_no_second_integrity_message_was_authored() -> None:
    """covers: A6 -- reuse `verify_digest`, do not write a rival error.

    Two differently worded integrity errors leave a reader unable to tell whether
    they hit resolution or load, for the same underlying event.
    """
    from yowo.arch import _weights as arch_weights
    from yowo.models._weights import verify_digest

    # `hasattr`, not a substring: the name `verify_digest` already appears in
    # `_raw_digest`'s docstring explaining that resolution ran it. Matching that
    # would be a check that passes on prose while the loader verifies nothing --
    # the same trap `test_routine_load_uses_weights_only` was written to dodge.
    assert getattr(arch_weights, "verify_digest", None) is verify_digest, (
        "the loader does not hold the one integrity check the project already has"
    )

    # Strip comments before searching, for the same reason.
    code = "\n".join(line.split("#", 1)[0] for line in inspect.getsource(arch_weights).splitlines())
    assert "WeightIntegrityError(" not in code, (
        "a second integrity error was authored instead of reusing verify_digest"
    )
