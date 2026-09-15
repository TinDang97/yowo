"""Which precision each backend can actually make its compute use, per device.

A backend *honours* a precision only if it can make its own runtime compute use
that precision. That is a deliberately strict reading, and it splits the roster
in two:

* ``pytorch`` decides its own numerics at inference time. It runs FP32 on every
  device, and FP16 on CUDA via ``torch.amp.autocast`` — which is gated on
  ``startswith("cuda")`` alone (``_pytorch.py``), so MPS gets no FP16 either.
* ``onnx``, ``tensorrt``, ``openvino`` and ``coreml`` load a **pre-compiled
  artifact**. Its numerics were fixed by whoever exported it, so they decide
  nothing at runtime — but deciding nothing is not the same as executing
  nothing, and the first version of this module conflated the two. It refused
  every explicit request to an artifact backend, including an ``fp32`` request
  against an artifact exported at FP32, which is the system doing exactly what
  was asked. Eight conformance tests caught it on PR #52.

  So the artifact is ASKED instead, at its I/O boundary
  (``artifact_precision``), and the refusal is reserved for a DEMONSTRATED
  mismatch. Where the artifact cannot say, the request is accepted and
  ``health_report()`` reports ``"unknown"`` — the anti-lie property lives in
  what is REPORTED, not in a refusal.

No backend executes INT8 at runtime. INT8 is an export-time property here.

The table is TOTAL over ``BackendType`` and ``DeviceType``, and the totality is
asserted against the enums at runtime rather than against a list someone has to
remember to update — the same discipline ``_roster.py`` uses for execution
verdicts. A sixth backend cannot reach ``create_backend()`` without a verdict.
"""

from __future__ import annotations

from yowo.errors import ConfigError
from yowo.types import BackendType, DeviceType, Precision

__all__ = [
    "RUNTIME_PRECISIONS",
    "artifact_precision",
    "device_type_of",
    "honoured_precision",
    "verify_artifact_precision",
]


_NONE_AT_RUNTIME: dict[DeviceType, frozenset[Precision]] = {
    device: frozenset() for device in DeviceType
}


#: ``{backend: {device: the precisions that backend's own compute can use}}``.
#: An empty set means the backend determines no precision at runtime.
RUNTIME_PRECISIONS: dict[BackendType, dict[DeviceType, frozenset[Precision]]] = {
    BackendType.PYTORCH: {
        DeviceType.CUDA: frozenset({Precision.FP32, Precision.FP16}),
        DeviceType.CPU: frozenset({Precision.FP32}),
        DeviceType.MPS: frozenset({Precision.FP32}),
    },
    BackendType.ONNX: dict(_NONE_AT_RUNTIME),
    BackendType.TENSORRT: dict(_NONE_AT_RUNTIME),
    BackendType.OPENVINO: dict(_NONE_AT_RUNTIME),
    BackendType.COREML: dict(_NONE_AT_RUNTIME),
}


def _device_refusal(
    backend: BackendType,
    device_type: DeviceType,
    requested: Precision,
    executes: Precision,
) -> ConfigError:
    remedy = (
        "use a CUDA device for fp16"
        if requested is Precision.FP16
        else f"no backend in yowo executes '{requested.value}' at runtime; it is "
        f"an export-time property"
    )
    return ConfigError(
        f"Backend '{backend.value}' on device '{device_type.value}' cannot honour "
        f"precision '{requested.value}'; it executes '{executes.value}'. It was "
        f"executing '{executes.value}' before this error existed too — the request "
        f"was accepted and ignored. To fix: {remedy}, or omit the precision "
        f"request to accept '{executes.value}'."
    )


def honoured_precision(
    backend: BackendType,
    device_type: DeviceType,
    requested: Precision | None,
    *,
    explicit: bool = False,
) -> Precision | None:
    """The precision *backend* will really execute on *device_type*.

    Args:
        backend: The backend that will execute.
        device_type: The device family the backend actually RESOLVED — never
            ``BackendSelection.device_type``, which can say CUDA while the
            backend runs on CPU.
        requested: The precision asked for, or ``None`` for no preference.
        explicit: ``True`` when a human asked for *requested* — the
            ``precision=`` argument, a non-``auto`` ``--precision``, or
            ``YOWO_PRECISION``. A tune profile is not explicit; it is a
            measurement. Only a request a human made is ever loud.

    Returns:
        The precision that will execute, or ``None`` when this backend
        determines no precision at runtime because its artifact does.

    Raises:
        ConfigError: *explicit* is True and *requested* is not what will run.
            An auto-selected precision is adjusted silently instead: only a
            request a human actually made is loud.
    """
    honourable = RUNTIME_PRECISIONS[backend][device_type]

    if not honourable:
        # An artifact backend decides no precision at runtime. That is ignorance,
        # not a mismatch, so nothing is refused here — `verify_artifact_precision`
        # asks the artifact, and only a demonstrated disagreement raises.
        return None

    executes = Precision.FP32 if Precision.FP32 in honourable else min(honourable)
    if requested is None:
        return executes
    if requested in honourable:
        return requested
    if explicit:
        raise _device_refusal(backend, device_type, requested, executes)
    return executes


def device_type_of(device: str) -> DeviceType:
    """The device family a resolved device STRING belongs to.

    Read from the string the backend resolved, deliberately. ``select_backend``
    can return ``device_type=CUDA`` alongside ``device_str="cpu"`` — it sets the
    type from ``torch_cuda_available`` while ``device_override`` sets the string
    — so a verdict taken from the selection would call fp16 honourable on a run
    that executes on CPU.
    """
    lowered = device.lower()
    if lowered.startswith("cuda"):
        return DeviceType.CUDA
    if lowered.startswith("mps"):
        return DeviceType.MPS
    return DeviceType.CPU


# ---------------------------------------------------------------------------
# What a pre-compiled artifact can be made to say about itself
# ---------------------------------------------------------------------------

#: Element-type spellings that mean half precision. This is an ORACLE, so it is
#: measured rather than assumed — both live spellings below were read off a real
#: artifact on this machine, 2026-09-15:
#:
#: * onnxruntime — which is ONNX *and* TensorRT, since ``_tensorrt.py`` drives
#:   an ORT session on the TensorRT EP — reports ``"tensor(float16)"`` against
#:   ``"tensor(float)"``;
#: * OpenVINO's ``get_element_type().get_type_name()`` reports ``"f16"``
#:   against ``"f32"``.
#:
#: The remaining members are defensive spellings for runtime versions not
#: measured here. They widen what is RECOGNISED as half, never what is refused:
#: a spelling this set misses yields ``None`` and reads ``"unknown"``.
_HALF_SPELLINGS = frozenset({"tensor(float16)", "float16", "f16", "fp16", "half"})


def artifact_precision(element_type: str | None) -> Precision | None:
    """What an artifact's I/O element type PROVES about the precision it runs.

    Measured 2026-09-15 on this project's own export path, yolo11n ONNX:

    ======================  ==================  ==================
    artifact                input               output
    ======================  ==================  ==================
    FP32 export             ``tensor(float)``   ``tensor(float)``
    FP16 export             ``tensor(float16)`` ``tensor(float16)``
    INT8 (QDQ, from FP32)   ``tensor(float)``   ``tensor(float)``
    ======================  ==================  ==================

    So a half-precision boundary is conclusive, and a float boundary is NOT:
    FP32 and an INT8 QDQ graph are indistinguishable there — identical element
    types at both ends, differing only in file size, and a byte count is not a
    precision oracle. ``None`` is therefore the honest answer for a float
    boundary, and the caller reports ``"unknown"`` rather than claiming FP32.

    Args:
        element_type: The runtime's own spelling of the input element type, or
            ``None`` when the artifact was not interrogated.

    Returns:
        ``Precision.FP16``, or ``None`` when the artifact does not settle it.
    """
    if element_type is None:
        return None
    return Precision.FP16 if element_type.strip().lower() in _HALF_SPELLINGS else None


def verify_artifact_precision(
    backend: BackendType,
    element_type: str | None,
    requested: Precision | None,
    *,
    explicit: bool = False,
) -> Precision | None:
    """The precision an artifact backend will execute, refusing a shown mismatch.

    M2 raises on a request "that is not what the loaded backend actually
    executes" — a demonstrated mismatch, never ignorance. Exactly ONE thing is
    demonstrated by the I/O element type: an artifact declaring a HALF boundary
    executes fp16, so a request for anything else disagrees with it.

    Nothing else is. A float boundary is SILENT, and reading it as "therefore
    not fp16" would be the strict A3 reading wearing a quieter hat. Measured on
    this machine 2026-09-15: ``onnxconverter_common.convert_float_to_float16(
    ..., keep_io_types=True)`` — a standard fp16 export — produces a graph whose
    compute is fp16 (inner ``Cast`` nodes) behind ``tensor(float)`` at BOTH
    ends. Refusing an explicit fp16 there would false-refuse a genuinely fp16
    artifact, which is the same false refusal PR #52 reopened this node for.

    So everything but the half-boundary mismatch is accepted and reported
    ``"unknown"`` — an explicit ``fp32`` against an INT8-quantized artifact
    included, the two being indistinguishable here (see
    :func:`artifact_precision`). The honesty is bought by what is REPORTED, not
    by a refusal: M4 keeps anything that is not executing from being reported.
    The cost, knowingly accepted: "asked fp16, got an FP32 export" is not loud.
    It is quiet and correctly labelled, rather than loud and sometimes wrong.

    Args:
        backend: The backend that loaded the artifact.
        element_type: The runtime's spelling of the input element type.
        requested: The precision asked for, or ``None``.
        explicit: ``True`` when a human asked for *requested*.

    Returns:
        The determined precision, or ``None`` when the artifact does not say.

    Raises:
        ConfigError: *explicit*, and the artifact demonstrably disagrees.
    """
    declared = artifact_precision(element_type)
    if not explicit or requested is None:
        return declared

    observed = element_type or "an element type it did not report"

    if declared is not None and requested is not declared:
        raise ConfigError(
            f"Backend '{backend.value}' cannot honour precision "
            f"'{requested.value}': the artifact it loaded declares "
            f"'{observed}' at its input, so it executes "
            f"'{declared.value}'. Re-export at '{requested.value}', or omit "
            f"the precision request to run the artifact you have."
        )

    return declared
