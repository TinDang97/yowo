"""A real ONNX session must bind the provider the caller asked for.

The unit checks drive `_select_providers` directly. This one goes through the
real `load()` path and reads the providers back off the live session, because
the defect was never in the selection function alone — it was that nothing
ever compared what was requested against what ORT actually bound.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from yowo.backends._onnx import OnnxBackend
from yowo.hardware import get_hardware_profile
from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

_FAMILY = ModelFamily.YOLO26
_SIZE = ModelSize.NANO


@pytest.fixture(scope="module")
def onnx_artifact(verified_weight: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("onnxruntime")
    from yowo.export import export_model

    out = tmp_path_factory.mktemp("onnx-provider")
    spec = ModelSpec(family=_FAMILY, size=_SIZE, weights_path=verified_weight)
    export_model(spec, ExportFormat.ONNX, out, precision=Precision.FP32, imgsz=640)
    return next(out.rglob("*.onnx"))


def test_a_cpu_request_actually_runs_on_the_cpu_provider(onnx_artifact: Path) -> None:
    """device='cpu' must bind exactly the CPU EP on the live session."""
    backend = OnnxBackend(get_hardware_profile())
    backend.load(onnx_artifact, device="cpu")
    try:
        bound = backend.active_providers
    finally:
        backend.unload()

    assert bound == ("CPUExecutionProvider",), (
        f"device='cpu' bound {bound!r}. On a CoreML host this is the FP16 "
        "substitution: CoreML computes in half precision, so an explicit CPU "
        "request silently ran on the Neural Engine while selection reported "
        "cpu/fp32. Host: "
        f"{platform.system()} {platform.machine()}."
    )


def test_the_reported_providers_come_from_the_session(onnx_artifact: Path) -> None:
    """The value is read back, not echoed from the request."""
    backend = OnnxBackend(get_hardware_profile())
    backend.load(onnx_artifact, device="cpu")
    try:
        assert backend.active_providers == tuple(backend._session.get_providers())
    finally:
        backend.unload()
