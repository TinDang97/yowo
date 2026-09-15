"""Which precision each backend can actually make its compute use, per device.

A backend *honours* a precision only if it can make its own runtime compute use
that precision. That is a deliberately strict reading, and it splits the roster
in two:

* ``pytorch`` decides its own numerics at inference time. It runs FP32 on every
  device, and FP16 on CUDA via ``torch.amp.autocast`` — which is gated on
  ``startswith("cuda")`` alone (``_pytorch.py``), so MPS gets no FP16 either.
* ``onnx``, ``tensorrt``, ``openvino`` and ``coreml`` load a **pre-compiled
  artifact**. Its numerics were fixed by whoever exported it. They honour
  nothing at runtime — ``fp32`` included, because passing an explicit ``fp32``
  to a backend holding an FP16 engine would report one number while another ran.

No backend executes INT8 at runtime. INT8 is an export-time property here.

The table is TOTAL over ``BackendType`` and ``DeviceType``, and the totality is
asserted against the enums at runtime rather than against a list someone has to
remember to update — the same discipline ``_roster.py`` uses for execution
verdicts. A sixth backend cannot reach ``create_backend()`` without a verdict.
"""

from __future__ import annotations

from yowo.errors import ConfigError
from yowo.types import BackendType, DeviceType, Precision

__all__ = ["RUNTIME_PRECISIONS", "device_type_of", "honoured_precision"]


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


def _artifact_refusal(backend: BackendType, requested: Precision) -> ConfigError:
    return ConfigError(
        f"Backend '{backend.value}' cannot honour precision '{requested.value}'. "
        f"Its numerics are fixed in the artifact it loads, chosen when that "
        f"artifact was exported — this backend was never executing "
        f"'{requested.value}' because of this flag, before or now. "
        f"Export at the precision you want "
        f"(`yowo export <model> -f {backend.value} --precision {requested.value}`) "
        f"and load that artifact, or omit the precision request."
    )


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
            measurement.

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
        if explicit and requested is not None:
            raise _artifact_refusal(backend, requested)
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
