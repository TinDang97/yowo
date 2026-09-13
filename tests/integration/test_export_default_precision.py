"""The default export invocation must produce a loadable artifact.

Measured 2026-09-13 at HEAD 1853ab0: ``yowo export yolo26n -f onnx`` — every
flag left at its default — exits 1 and writes ZERO files. ``export_model``
defaults to ``Precision.FP16`` and the CLI to ``-p fp16``; both fail for every
detection and OBB model, while ``-p fp32`` succeeds and writes 3 files.

Cause: ``Detect._init_strides`` builds its stride tensor in float32 and writes
it into the (half) ``stride`` buffer with ``copy_``. ``torch.export``
functionalizes that mutation, so ``make_anchors`` downstream reads the float32
pre-copy value and its ``.to(dtype=half)`` raises
``Tensor dtype mismatch! Expected: torch.float16, Got: torch.float32``.

Nothing caught this because every ``export_model`` call in the unit suite
patches ``_export_onnx`` and passes FP32 or INT8 — never the default. These
checks therefore run a REAL export and load the result back; a mocked exporter
cannot prove the default path works.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from yowo.export import export_model
from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

_FAMILY = ModelFamily.YOLO26
_SIZE = ModelSize.NANO


def test_the_default_precision_is_still_fp16() -> None:
    """Guards the check below from being 'fixed' by changing the default.

    If someone makes the default export work by switching the default to FP32,
    the artifact check would go green while FP16 export stayed broken. This
    check fails in that case and says so.
    """
    default = inspect.signature(export_model).parameters["precision"].default
    assert default is Precision.FP16, (
        f"export_model's default precision is {default!r}, not FP16. If this "
        "was changed deliberately, the FP16 export path still needs its own "
        "check — do not let the default-path check pass by narrowing what the "
        "default is."
    )


def test_the_default_export_invocation_produces_a_loadable_artifact(
    verified_weight: Path, tmp_path: Path
) -> None:
    """A real export at the DEFAULT precision, loaded back through onnxruntime."""
    ort = pytest.importorskip("onnxruntime")

    spec = ModelSpec(family=_FAMILY, size=_SIZE, weights_path=verified_weight)
    out = tmp_path / "default-export"
    out.mkdir()

    # No precision= argument: this is exactly what the CLI's default does.
    export_model(spec, ExportFormat.ONNX, out, imgsz=640)

    produced = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert produced, (
        "the default export invocation produced NO files. Measured 2026-09-13: "
        "export_model(..., precision=FP16) raises ExportError for every "
        "detection and OBB model, so the documented default writes nothing."
    )

    onnx_files = [p for p in out.rglob("*.onnx")]
    assert onnx_files, f"no .onnx among the produced files: {produced}"

    session = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: np.zeros((1, 3, 640, 640), dtype=np.float16)})
    assert outputs and outputs[0].size > 0, "the exported graph returned nothing"


def test_fp32_export_still_works(verified_weight: Path, tmp_path: Path) -> None:
    """The path that worked before the fix must be unaffected by it.

    For an fp32 `stride` buffer the added cast is a no-op, so this is the
    control that proves the default path was not bought at its expense.
    """
    spec = ModelSpec(family=_FAMILY, size=_SIZE, weights_path=verified_weight)
    out = tmp_path / "fp32-export"
    out.mkdir()

    export_model(spec, ExportFormat.ONNX, out, precision=Precision.FP32, imgsz=640)

    assert [p for p in out.rglob("*.onnx")], "FP32 export regressed"
