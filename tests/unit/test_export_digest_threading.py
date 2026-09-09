"""The export path verifies its weights the way the inference path does.

`verified-digest-threading` closed the resolve-to-load window for the inference
path only: it threaded the registry's pinned SHA-256 from `PyTorchBackend.load`
into `load_weights` / `load_classify_weights` / `load_obb_weights`, so a file
substituted between `resolve_weights` and the unpickling conversion is refused
before `_extract_state_dict` ever runs.

`export_model` in `yowo.export._exporter` resolves the same weights, through
the same `resolve_weights`, and calls the same three loaders -- but never
passed a digest. All three loaders already accept `raw_digest: str | None`
(added by the parent task), so exporting a substituted file has always
converted it without ever comparing it to the pin. That is worse at export
time than at inference time in one specific way: an exported artifact is a
file that outlives the process and gets shipped somewhere else.

These checks thread the same pin through the same three call sites, exactly as
`test_verified_digest_threading.py` proved for the backend. The classify and
obb branches already hold the registry `meta` they need; the detection branch
does not call the registry today and must gain that lookup.

Authored red under ADD task `export-digest-threading`.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    fused = MagicMock()
    evaled = MagicMock()
    model.fuse.return_value = fused
    fused.eval.return_value = evaled
    evaled.half.return_value = evaled
    return model


def _meta(sha256: str | None) -> MagicMock:
    meta = MagicMock()
    meta.sha256 = sha256
    meta.num_classes = 15
    meta.input_height = 640
    meta.input_width = 640
    return meta


def _run_export(spec: ModelSpec, tmp_path: Path, *patches: object) -> None:
    """Drive export_model far enough to observe the load_* call, no further.

    Every export exercised here stops at ONNX -- `_export_onnx` is always
    stubbed -- because the digest threading these checks prove happens before
    any conversion work, on the model-build side of `export_model`.
    """
    from yowo.export._exporter import export_model

    def _write_fake_onnx(_model: object, _dummy: object, onnx_path: Path, **_kw: object) -> None:
        # `_export_onnx` writes to a model_stem-derived path (e.g.
        # "yolo11n.onnx" or "yolo11n-obb.onnx") -- match it exactly, since
        # `export_model` later `.stat()`s that same `Path` object.
        onnx_path.write_bytes(b"fake")

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.export._exporter._export_onnx", side_effect=_write_fake_onnx),
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
    ):
        fake_meta = MagicMock()
        fake_meta.file_path = "unused"
        fake_meta.file_size_bytes = 4
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32)


# ---------------------------------------------------------------------------
# A2/E3 -- the detection branch gains the registry lookup it does not make today
# ---------------------------------------------------------------------------


def test_export_detection_branch_makes_a_registry_lookup(tmp_path: Path) -> None:
    """covers: A2 -- detect export must call the registry, which it does not today.

    `classify` and `obb` already hold `meta` from `get_cls` / `get_obb`. Detect
    exports build straight from `spec.family` / `spec.size` with no lookup at
    all -- so there is nothing to read a pin from until this branch gains one.
    """
    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)

    with (
        patch("yowo.arch.build_model", return_value=_make_mock_model()),
        patch("yowo.arch._weights.load_weights"),
        patch("yowo.models._registry.get", return_value=_meta("a" * 64)) as mock_get,
    ):
        _run_export(spec, tmp_path)

    mock_get.assert_called_once_with(ModelFamily.YOLO11, ModelSize.NANO)


# ---------------------------------------------------------------------------
# A1/A2 -- all three branches thread the registry pin, never a caller value
# ---------------------------------------------------------------------------


def test_export_threads_the_registry_pin_for_detection(tmp_path: Path) -> None:
    """covers: A1, A2, M1 -- the digest reaching the loader is the registry's own.

    Uses the real registry entry for yolo11n rather than a mock, the same way
    the backend's equivalent check does: the pin must be the actual pinned
    value, not a strand that merely happens not to be None.
    """
    from yowo.models._registry import get

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    expected = get(ModelFamily.YOLO11, ModelSize.NANO).sha256
    assert expected, "yolo11n has no registry pin; fixture assumption broken"

    with (
        patch("yowo.arch.build_model", return_value=_make_mock_model()),
        patch("yowo.arch._weights.load_weights") as load_fn,
    ):
        _run_export(spec, tmp_path)

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
def test_export_threads_the_registry_pin_for_cls_and_obb(
    task: str, builder: str, loader: str, registry_get: str, tmp_path: Path
) -> None:
    """covers: A2, E3 -- cls and obb exports pass the same gate as detection.

    Both are `sha256=None` in the registry today; this asserts the wiring, not
    a value, exactly as the backend's sibling check does -- the day either
    entry is pinned, the export gate closes on it without another change here.
    """
    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task=task)

    with (
        patch(registry_get, return_value=_meta("b" * 64)) as mock_registry_get,
        patch(builder, return_value=_make_mock_model()),
        patch(loader) as load_fn,
    ):
        _run_export(spec, tmp_path)

    assert load_fn.call_args.kwargs.get("raw_digest") == "b" * 64
    # A5: exactly the lookup the branch already made for num_classes/input size --
    # no second registry call added to source the pin.
    mock_registry_get.assert_called_once()


# ---------------------------------------------------------------------------
# A1/A4 -- an explicit weights_path is never compared to the official digest
# ---------------------------------------------------------------------------


def test_export_sends_no_pin_for_an_explicit_weights_path(tmp_path: Path) -> None:
    """covers: A1, A4 -- a user's own checkpoint is not the file the registry pinned.

    Comparing a fine-tuned local weight against the official yolo11n digest
    would refuse every custom export. The kwarg must be PRESENT and None, not
    merely absent -- absence would pass against today's code too, which sends
    no pin at all, and would prove nothing about this branch.
    """
    local = tmp_path / "finetuned.pt"
    local.write_bytes(b"a user's own weights")
    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, weights_path=local)

    with (
        patch("yowo.arch.build_model", return_value=_make_mock_model()),
        patch("yowo.arch._weights.load_weights") as load_fn,
    ):
        _run_export(spec, tmp_path)

    assert "raw_digest" in load_fn.call_args.kwargs, "no pin decision was made at all"
    assert load_fn.call_args.kwargs["raw_digest"] is None


# ---------------------------------------------------------------------------
# A4/M3 -- unpinned models (cls, obb today) still export, and now say so
# ---------------------------------------------------------------------------


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
def test_export_unpinned_registry_model_still_makes_an_explicit_no_pin_decision(
    task: str, builder: str, loader: str, registry_get: str, tmp_path: Path
) -> None:
    """covers: A4, M3 -- an unpinned model is a regression risk, not a hard case.

    Every `-cls` and `-obb` registry entry carries `sha256=None` today, so
    unpinned is the COMMON case for two of the three export branches. Refusing
    them would be a regression dressed as a security fix. The kwarg must be
    PRESENT and None -- absence would pass against today's code too, which
    sends no pin at all and makes no decision about one either way.
    """
    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task=task)

    with (
        patch(registry_get, return_value=_meta(None)) as mock_registry_get,
        patch(builder, return_value=_make_mock_model()),
        patch(loader) as load_fn,
    ):
        _run_export(spec, tmp_path)

    mock_registry_get.assert_called_once()  # A5: the pin comes from the lookup already made
    load_fn.assert_called_once()
    assert "raw_digest" in load_fn.call_args.kwargs, "no pin decision was made at all"
    assert load_fn.call_args.kwargs["raw_digest"] is None


# ---------------------------------------------------------------------------
# A3/M4 -- no new comparison logic in the exporter; it calls the parent's gate
# ---------------------------------------------------------------------------


def test_no_new_comparison_logic_was_authored_in_exporter() -> None:
    """covers: A3, M4 -- the exporter supplies the value; it never re-implements

    the check that consumes it. `load_verified_state_dict` already calls
    `verify_digest` when handed a digest -- reimplementing that comparison in
    the exporter would give one rule two implementations that can disagree.
    """
    from yowo.export import _exporter

    assert not hasattr(_exporter, "verify_digest"), (
        "the exporter imported verify_digest directly -- it must never compare, only supply"
    )
    assert not hasattr(_exporter, "WeightIntegrityError"), (
        "the exporter authored its own integrity error type instead of letting the loader raise"
    )

    # Strip comments before searching, for the same reason
    # test_no_second_integrity_message_was_authored does in the parent suite.
    code = "\n".join(line.split("#", 1)[0] for line in inspect.getsource(_exporter).splitlines())
    assert "verify_digest(" not in code, "the exporter calls verify_digest directly"
    assert "WeightIntegrityError(" not in code, "the exporter authored a second integrity error"


def test_integrity_failure_passes_through_export_unwrapped(tmp_path: Path) -> None:
    """covers: A6 -- a mismatch surfaces as verify_digest's own message.

    `export_model` has no `try/except` around the model-build section, so a
    `WeightIntegrityError` raised by `load_weights` must reach the caller
    exactly as raised -- not re-wrapped in an `ExportError`, which would leave
    a reader unable to tell an integrity refusal from a real export failure.
    """
    from yowo.models._weights import WeightIntegrityError

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    boom = WeightIntegrityError("Weight failed integrity check: cached.pt")

    with (
        patch("yowo.arch.build_model", return_value=_make_mock_model()),
        patch("yowo.arch._weights.load_weights", side_effect=boom),
        pytest.raises(WeightIntegrityError),
    ):
        _run_export(spec, tmp_path)


def test_load_verified_state_dict_signature_is_unchanged() -> None:
    """covers: M4 -- this node adds a caller, not a parameter.

    Threading a value that already exists must not touch the signature the
    parent task settled; if it did, this node widened the gate instead of
    feeding it.
    """
    from yowo.arch._weights import load_verified_state_dict

    params = inspect.signature(load_verified_state_dict).parameters
    assert list(params) == ["checkpoint_path", "raw_digest"]
    assert params["raw_digest"].default is None
