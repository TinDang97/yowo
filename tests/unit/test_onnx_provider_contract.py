"""An explicitly requested device must be the device that runs.

Measured 2026-09-13: `_select_providers` returned the CoreML EP when the
caller asked for `device="cpu"` — its own docstring documented this as
intentional — and CoreML computes in FP16. Same host, same weight, same
exported ONNX, yolo26n:

    providers ['CoreMLExecutionProvider', 'CPUExecutionProvider']  0.82116699 px
    providers ['CPUExecutionProvider'] forced                      0.00012207 px

All five CoreML-path confidences landed exactly on the float16 grid; none of
the CPU-path ones did. So `engine.selection` reported cpu/fp32 while the
Neural Engine computed in half precision, shifting confidences by up to
0.0073 against a default `confidence_threshold` of 0.25.

This figure was filed twice as a platform fact — first as a general
PyTorch-ONNX divergence, then as arm64 arithmetic — before anyone checked
which execution provider had actually run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from yowo.backends._onnx import OnnxBackend


def _hw(*, has_coreml: bool = False, has_cuda: bool = False) -> MagicMock:
    hw = MagicMock()
    hw.libraries.onnxruntime_version = "1.17.0"
    hw.libraries.onnxruntime_has_cuda = has_cuda
    hw.libraries.onnxruntime_has_coreml = has_coreml
    hw.has_nvidia_gpu = has_cuda
    return hw


def _names(providers: list[object]) -> list[str]:
    return [p[0] if isinstance(p, tuple) else p for p in providers]  # type: ignore[index]


def test_an_explicit_cpu_device_yields_only_the_cpu_provider() -> None:
    """The defect, stated as a check: asking for cpu must get the CPU EP."""
    backend = OnnxBackend(_hw(has_coreml=True))

    providers = backend._select_providers("cpu")

    assert _names(providers) == ["CPUExecutionProvider"], (
        f"device='cpu' selected {_names(providers)} on a CoreML host. CoreML "
        "computes in FP16, so an explicit CPU request silently ran on the "
        "Neural Engine at half precision while selection reported cpu/fp32."
    )


def test_auto_still_selects_coreml_where_available() -> None:
    """The speedup is not paid for the fix."""
    backend = OnnxBackend(_hw(has_coreml=True))

    providers = backend._select_providers("auto")

    assert "CoreMLExecutionProvider" in _names(providers), (
        "device='auto' stopped selecting CoreML — the 4-5x Apple Silicon "
        "speedup was traded away to fix the explicit-request path"
    )
    assert ("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}) in providers


def test_a_host_without_coreml_is_unaffected() -> None:
    """On a non-CoreML host the fix is a no-op, not a behaviour change."""
    backend = OnnxBackend(_hw(has_coreml=False))

    assert _names(backend._select_providers("cpu")) == ["CPUExecutionProvider"]
    assert _names(backend._select_providers("auto")) == ["CPUExecutionProvider"]


def test_cuda_is_still_honoured() -> None:
    """An explicit cuda request was already honoured and must stay so."""
    backend = OnnxBackend(_hw(has_cuda=True))

    providers = backend._select_providers("cuda")

    assert _names(providers)[0] == "CUDAExecutionProvider"


def test_an_unrecognised_device_still_yields_a_usable_provider() -> None:
    """A8: this node fixes a lie; it must not add validation that raises."""
    backend = OnnxBackend(_hw(has_coreml=False))

    providers = backend._select_providers("wibble")

    assert _names(providers) == ["CPUExecutionProvider"], (
        "an unknown device string stopped yielding a usable session — this "
        "node must not turn a working call into a crash for an unrelated reason"
    )


def test_active_providers_is_empty_before_load() -> None:
    """A4/E3: safe to read on the error path that most needs it."""
    backend = OnnxBackend(_hw())

    assert backend.active_providers == ()


def test_active_providers_is_read_back_from_the_session() -> None:
    """M3: report what ORT bound, never the list that was requested."""
    backend = OnnxBackend(_hw(has_coreml=True))
    session = MagicMock()
    # ORT bound ONLY the CPU EP, whatever was requested.
    session.get_providers.return_value = ["CPUExecutionProvider"]
    backend._session = session
    backend._record_active_providers()

    assert backend.active_providers == ("CPUExecutionProvider",), (
        "active_providers did not reflect the live session — reporting the "
        "requested list rather than the bound one is the same class of lie "
        "this node exists to fix"
    )


def test_active_providers_reports_the_whole_chain_in_order() -> None:
    """A7/A9: all bound providers, in ORT's own precedence order."""
    backend = OnnxBackend(_hw(has_coreml=True))
    session = MagicMock()
    session.get_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    backend._session = session
    backend._record_active_providers()

    assert backend.active_providers == (
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ), "the fallback chain must stay visible — it explains a number landing between two EPs"


def test_active_providers_is_cleared_on_unload() -> None:
    """A stale provider list describes a session that no longer exists."""
    backend = OnnxBackend(_hw(has_coreml=True))
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    backend._session = session
    backend._record_active_providers()
    assert backend.active_providers  # precondition: it was populated

    backend.unload()

    assert backend.active_providers == ()
