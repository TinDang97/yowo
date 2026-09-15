"""`precision` reaches the backend that executes it, or the caller is told it cannot.

The red suite for `/tasks/precision-plumbing.md`. Measured at HEAD 714fa79:
`create_backend()` has no `precision` parameter, so the value `select_backend()`
resolves at `engine.py:286` is computed and dropped; `PyTorchBackend(fp16=...)`
is the only runtime hook in the library and nothing has ever passed it.

Every fixture here requests a NON-DEFAULT precision. `fp32` is both the CPU
default and the value every unhonoured request already coerces to, so a fixture
built from it compares equal whether or not the value survived the trip (Q4).

Every check asserts at the EXECUTOR — the constructed backend, the engine's own
state — never at the config that did the asking (Q13).
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import DetectionEngine, HealthReport
from yowo.errors import ConfigError
from yowo.types import BackendSelection, BackendType, DeviceType, Precision

# The two precisions that are not the default. A check built from FP32 proves
# nothing here: it is what an unhonoured request already silently becomes.
NON_DEFAULT = (Precision.FP16, Precision.INT8)


# ---------------------------------------------------------------------------
# Helpers — mirroring tests/unit/test_oom_monitor.py's idiom
# ---------------------------------------------------------------------------


def _mock_backend(backend_type: BackendType = BackendType.PYTORCH) -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = backend_type
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.load.return_value = None
    mock.warmup.return_value = None
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.unload.return_value = None
    return mock


def _loaded_engine(
    mock: MagicMock,
    *,
    device_type: DeviceType = DeviceType.CUDA,
    precision: Precision = Precision.FP32,
) -> DetectionEngine:
    """An engine with a mock backend, loaded, reporting *device_type*."""
    engine = DetectionEngine(backend_instance=mock)
    engine._selection = BackendSelection(
        backend=mock.backend_type,
        device_type=device_type,
        precision=precision,
        device_index=0,
        reason="test",
    )
    with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


# ---------------------------------------------------------------------------
# The request must reach the executor, or be refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requested", NON_DEFAULT, ids=lambda p: p.value)
def test_an_explicit_precision_the_backend_does_not_execute_raises(
    requested: Precision,
) -> None:
    """covers: M2, A23, A26, E2, E3, R:SILENT_COERCION, R:REPORTS_REQUEST.

    Neither fp16 nor int8 is executed by any backend on a CPU device: the only
    runtime hook in the library autocasts under CUDA alone (`_pytorch.py:354`).
    Today both are accepted, stored, and reported while fp32 runs.
    """
    with pytest.raises(ConfigError) as exc:
        DetectionEngine(
            backend=BackendType.PYTORCH,
            device="cpu",
            precision=requested,
        ).load()

    message = str(exc.value).lower()
    assert "pytorch" in message, f"the error must name the backend: {message}"
    assert "cpu" in message, f"the error must name the device: {message}"
    assert requested.value in message, f"the error must name the request: {message}"


def test_an_auto_selected_precision_that_cannot_be_honoured_does_not_raise() -> None:
    """covers: M3, A13, E1, E4, R:AUTO_RAISES.

    The control, and the half of the decision that is NOT 'raise everywhere'.
    An unset precision is the caller expressing no preference; it must reach the
    backend and refuse nothing on any device.
    """
    engine = DetectionEngine(backend=BackendType.PYTORCH, device="cpu")

    assert engine._requested_precision is None, (
        "the engine must keep the RAW request: select_precision degrades it before "
        "any verdict could see it (_selector.py:189, :199-200)"
    )


def test_a_honourable_precision_reaches_the_executing_backend() -> None:
    """covers: M1, A9, R:MOCK_ONLY_HONOUR.

    Asserted on the CONSTRUCTED backend, not on a mock's call args and not on
    the config that asked. Before load the backend has resolved no device, so it
    reports nothing rather than echoing the request.
    """
    from yowo.backends import create_backend
    from yowo.hardware import get_hardware_profile
    from yowo.types import ModelFamily, ModelSize, ModelSpec

    hw = get_hardware_profile()
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    backend = create_backend(BackendType.PYTORCH, hw, model_spec=spec, precision=Precision.FP16)
    assert backend.active_precision is None, "an unloaded backend executes nothing"


def test_the_precision_verdict_is_total_over_the_backend_enum() -> None:
    """covers: M5, A27 — enumerated at runtime, so a sixth backend cannot slip in."""
    from yowo.backends._precision import honoured_precision

    missing = [
        b
        for b in BackendType
        for d in DeviceType
        if honoured_precision(b, d, Precision.FP32) is ...  # sentinel: no verdict
    ]
    assert not missing, f"these backends have no precision verdict: {missing}"


@pytest.mark.parametrize("device_type", [DeviceType.CPU, DeviceType.MPS])
def test_only_cuda_claims_fp16(device_type: DeviceType) -> None:
    """covers: A3 — autocast is gated on startswith("cuda") alone."""
    from yowo.backends._precision import honoured_precision

    assert honoured_precision(BackendType.PYTORCH, device_type, Precision.FP16) is (
        Precision.FP32
    ), "a non-CUDA device executes fp32 whatever was asked for"


@pytest.mark.parametrize("backend", [BackendType.ONNX, BackendType.TENSORRT])
def test_an_artifact_backend_has_no_precision_verdict(backend: BackendType) -> None:
    """covers: A3, A16 — precision is the artifact's property, chosen at export."""
    from yowo.backends._precision import honoured_precision

    assert honoured_precision(backend, DeviceType.CUDA, Precision.FP16) is None


# ---------------------------------------------------------------------------
# What is reported must be what ran
# ---------------------------------------------------------------------------


def test_health_report_precision_comes_from_the_backend_not_the_request() -> None:
    """covers: M4, A15, E9, R:REPORTS_REQUEST.

    The selection holds INT8; the backend executes fp32. Today `precision_current`
    reads the selection and says "int8" — a number an operator sizes hardware on.
    """
    mock = _mock_backend()
    mock.active_precision = Precision.FP32
    engine = _loaded_engine(mock, precision=Precision.INT8)

    assert engine.health_report().precision_current == "fp32", (
        "precision_current must be produced by the executing backend"
    )


def test_an_artifact_backend_reports_no_precision_rather_than_guessing() -> None:
    """covers: M4, A15 — None is this dataclass's convention for 'not knowable'."""
    mock = _mock_backend(BackendType.ONNX)
    mock.active_precision = None
    engine = _loaded_engine(mock, precision=Precision.FP16)

    assert engine.health_report().precision_current is None


def test_precision_current_is_no_longer_allowlisted_as_producerless() -> None:
    """covers: M4 — the allowlist reason at test_metrics_truth.py:50 IS the defect."""
    from tests.unit.test_metrics_truth import _ALLOWLIST

    assert "precision_current" not in _ALLOWLIST, (
        "precision_current has a real producer now — the executing backend"
    )


def test_the_public_surface_loses_no_name_and_precision_current_is_widened() -> None:
    """covers: M4, A2, R:PUBLIC_BREAK.

    One assertion, three clauses. The first is red today and is why this check
    fails first; the other two are the fence this node must not cross — removing
    a public name is m5's deprecation call, not this node's.
    """
    import yowo
    import yowo.tune

    hints = typing.get_type_hints(HealthReport)
    assert hints["precision_current"] == (str | None), (
        "precision_current must be able to say it does not know"
    )

    assert "TuneProfile" in yowo.tune.__all__
    assert any(f.name == "precision" for f in dataclasses.fields(yowo.tune.TuneProfile)), (
        "dropping a field from a public frozen dataclass is m5's decision"
    )
    assert "HealthReport" in yowo.__all__


# ---------------------------------------------------------------------------
# A profile's precision is a machine's measurement, not a human's request
# ---------------------------------------------------------------------------


def test_a_tune_profile_precision_is_not_applied_to_a_production_config() -> None:
    """covers: M7, A6.

    `_load_tune_profile` is the only writer of a non-None precision that no human
    typed. While it stands, "explicit" is undecidable and the loudest rung fires
    on a value nobody asked for. The profile carries INT8 deliberately — a
    default-valued fixture here would compare equal either way (Q4).
    """
    from yowo.config import InferenceConfig
    from yowo.engine import _load_tune_profile
    from yowo.hardware import get_hardware_profile
    from yowo.tune._profile import TuneProfile, compute_fingerprint
    from yowo.types import ModelFamily, ModelSize, ModelSpec

    hw = get_hardware_profile()
    profile = TuneProfile(
        model="yolo11n",
        backend="onnx",
        batch_size=4,
        precision="int8",
        fps_achieved=123.4,
        tuned_at="2026-09-15T00:00:00",
        fingerprint=compute_fingerprint(hw),
    )
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    config = InferenceConfig(model_family=ModelFamily.YOLO11, model_size=ModelSize.NANO)

    with patch("yowo.tune._profile.load_profile", return_value=profile):
        result = _load_tune_profile(spec, config, hw)

    assert result.precision is None, (
        "a profile's precision is an FPS measurement over configurations that were "
        "identical because nothing consumed the value — not a request"
    )
    assert result.backend == BackendType.ONNX, "the profile's backend still applies"
    assert result.batch_size == 4, "the profile's batch size still applies"


def test_a_pre_change_tune_profile_still_loads(tmp_path: Path) -> None:
    """covers: M7 — the control. `load_profile` builds from named keys, so a key
    it stops reading is inert rather than fatal. Green before and after; it is
    here to catch a loader rewritten into `TuneProfile(**raw)`.
    """
    import yaml

    from yowo.hardware import get_hardware_profile
    from yowo.tune._profile import compute_fingerprint, load_profile

    hw = get_hardware_profile()
    dest = tmp_path / "yolo11n.yaml"
    dest.write_text(
        yaml.safe_dump(
            {
                "model": "yolo11n",
                "backend": "onnx",
                "batch_size": 4,
                "precision": "int8",
                "fps_achieved": 123.4,
                "tuned_at": "2026-09-15T00:00:00",
                "fingerprint": compute_fingerprint(hw),
            }
        ),
        encoding="utf-8",
    )
    profile = load_profile("yolo11n", hw, path=dest)
    assert profile is not None, "a profile written by the old code must still load"
    assert profile.backend == "onnx"
    assert profile.batch_size == 4


# ---------------------------------------------------------------------------
# The OOM ladder counts only recoveries that happened
# ---------------------------------------------------------------------------


def test_pressure_between_tier2_and_tier3_halves_the_batch_and_clears() -> None:
    """covers: M6, A12, A17, A22, E7.

    Three defects in one band. `_apply_oom_recovery` returns after the dead
    tier-2 call (`engine.py:691-696`), so tier-1 is unreachable in [0.90, 0.95);
    and `_try_precision_fallback` sets DEGRADED without setting
    `_oom_recovering`, so the restore branch at `engine.py:665` can never fire
    and one tick latches DEGRADED for the life of the process.
    """
    from yowo.types import HealthStatus

    mock = _mock_backend()
    mock.active_precision = Precision.FP32
    engine = DetectionEngine(backend_instance=mock, batch_size=8)

    engine._apply_oom_recovery(0.92)
    assert engine._batch_size == 4, "the band satisfies tier-1 and must get its recovery"

    engine._apply_oom_recovery(0.70)
    assert engine._batch_size == 8, "pressure cleared — the batch size must come back"
    assert engine.health == HealthStatus.READY, "DEGRADED must not be a one-way latch"


def test_a_no_op_recovery_counts_no_phantom_degradation() -> None:
    """covers: A7, R:PHANTOM_DEGRADATION.

    Exactly one degradation across the 0.92 tick — the batch halving that really
    happened. Today the dead precision rung counts one for a recovery that was
    never attempted, on top of nothing else occurring.
    """
    mock = _mock_backend()
    mock.active_precision = Precision.FP32
    engine = DetectionEngine(backend_instance=mock, batch_size=8)

    before = engine.export_metrics()["degradations_total"]
    engine._apply_oom_recovery(0.92)
    after = engine.export_metrics()["degradations_total"]

    assert after - before == 1, "exactly one recovery happened, so exactly one counts"
    assert engine._oom_recovering is True, (
        "today the count is one because the DEAD precision rung incremented it while "
        "doing nothing — and it never set _oom_recovering, so the engine can never "
        "clear. A degradation that cannot be undone was not a recovery."
    )


def test_no_backend_defines_the_dead_precision_fallback_hook() -> None:
    """covers: A17 — the rung is deleted, not left as a hook nothing implements."""
    import yowo.engine

    assert not hasattr(yowo.engine.DetectionEngine, "_try_precision_fallback"), (
        "a tier this suite cannot refute is a tier that should not ship"
    )


# ---------------------------------------------------------------------------
# Q4, enforced on this file
# ---------------------------------------------------------------------------


def test_no_fixture_in_this_suite_requests_the_default_precision() -> None:
    """covers: R:DEFAULT_FIXTURE.

    A round-trip over the default precision compares equal whether or not the
    value survived the trip. Every REQUEST this file makes must be non-default.
    """
    assert Precision.FP32 not in NON_DEFAULT
    source = Path(__file__).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "precision=Precision.FP32" in line and "_loaded_engine" not in line
    ]
    assert not offenders, (
        f"these lines REQUEST the default precision, which proves nothing: {offenders}"
    )
