"""What is actually executed, and what is not — with the runner it would take.

A backend nobody can run is not "untested", it is unshipped, and from outside
the only thing distinguishing the two is whether someone wrote it down.

Measured 2026-09-13, `pytest tests/unit --cov=src/yowo/backends`:

    _openvino.py   21%      _tensorrt.py   73%
    _pytorch.py    58%      _onnx.py       77%

with no `test_pytorch_backend.py` at all, and exactly one backend (`pytorch`)
executed unmocked anywhere. This module is the committed answer to m3 box 1,
and it is TOTAL over ``BackendType``: a sixth backend cannot be added without a
verdict, because the totality is asserted against the enum at runtime rather
than against a list someone has to remember to update.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from yowo.types import BackendType

__all__ = [
    "EXECUTED",
    "UNVERIFIED",
    "UnverifiedEntry",
    "Verdict",
    "verdict_for",
]


class Verdict(Enum):
    """Whether a backend is proved by execution or merely declared."""

    EXECUTED = "executed"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class UnverifiedEntry:
    """Why a backend is not executed, and what it would take to execute it.

    ``reason`` says what is actually missing. ``runner`` names the machine — a
    reason without a runner is indistinguishable from "nobody looked", which is
    the state this roster exists to end.
    """

    reason: str
    runner: str


#: Executed unmocked against the digest-verified weight by
#: ``tests/integration/test_backend_conformance.py``, which CI runs.
EXECUTED: tuple[BackendType, ...] = (
    BackendType.PYTORCH,
    BackendType.ONNX,
    BackendType.OPENVINO,
)


#: Not executed. Each entry names what is missing and the runner that would fix
#: it — never a bare "unverified".
UNVERIFIED: dict[BackendType, UnverifiedEntry] = {
    BackendType.TENSORRT: UnverifiedEntry(
        reason=(
            "TensorRT has no PyPI wheel on the standard index and requires an NVIDIA "
            "driver plus a CUDA-capable device at import time, so neither the "
            "dependency nor the execution can be obtained on a GitHub-hosted runner."
        ),
        runner="a self-hosted Linux runner with an NVIDIA GPU and CUDA",
    ),
    BackendType.COREML: UnverifiedEntry(
        reason=(
            "coremltools compiles and executes only on macOS; on Linux the model "
            "cannot be loaded at all, so a Linux job could construct the backend and "
            "still prove nothing about inference."
        ),
        runner="a macos-latest runner (GitHub offers one; adding a second OS to the "
        "matrix is sequenced into ci-matrix, deliberately last in m3)",
    ),
}


def verdict_for(backend: BackendType) -> Verdict | None:
    """Return this backend's roster verdict, or None when it has none.

    None is the answer the totality check exists to catch: a backend reachable
    from ``create_backend()`` that nobody has ruled on.
    """
    if backend in EXECUTED:
        return Verdict.EXECUTED
    if backend in UNVERIFIED:
        return Verdict.UNVERIFIED
    return None
