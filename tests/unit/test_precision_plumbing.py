"""`precision` reaches the backend that executes it, or the caller is told it cannot.

The check suite for `/tasks/precision-plumbing.md`. Measured at HEAD 714fa79:
`create_backend()` had no `precision` parameter, so the value `select_backend()`
resolved was computed and dropped; `PyTorchBackend(fp16=...)` was the only
runtime hook in the library and nothing ever passed it.

Every fixture here requests a NON-DEFAULT precision. `fp32` is both the CPU
default and the value every unhonoured request already coerced to, so a fixture
built from it compares equal whether or not the value survived the trip (Q4).

Every check asserts at the EXECUTOR — the constructed backend, the engine's own
state — never at the config that did the asking (Q13).

No check here is parametrised. The frozen `## CHECKS` cite bare names, and a
bare name resolves against the tail of a junit id: `test_x[fp16]` never matches
`test_x`, so a parametrised check cited by bare name binds nothing (M12).
"""

from __future__ import annotations

import ast
import dataclasses
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend, create_backend
from yowo.backends._precision import RUNTIME_PRECISIONS, honoured_precision
from yowo.backends._pytorch import PyTorchBackend
from yowo.engine import DetectionEngine, HealthReport
from yowo.errors import BackendLoadError, ConfigError
from yowo.hardware import HardwareProfile, get_hardware_profile
from yowo.types import (
    BackendSelection,
    BackendType,
    DeviceType,
    ModelFamily,
    ModelSize,
    ModelSpec,
    Precision,
)

# The two precisions that are not the default. A fixture built from FP32 proves
# nothing here: it is what an unhonoured request already silently became.
NON_DEFAULT = (Precision.FP16, Precision.INT8)

_SPEC = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)


# ---------------------------------------------------------------------------
# Helpers — mirroring tests/unit/test_oom_monitor.py's idiom
# ---------------------------------------------------------------------------


def _mock_backend(
    backend_type: BackendType = BackendType.PYTORCH,
    *,
    executing: Precision | None = Precision.FP32,
) -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = backend_type
    mock.is_loaded = False
    mock.input_shape = (640, 640)
    mock.executing_precision = executing
    mock.load.return_value = None
    mock.warmup.return_value = None
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.unload.return_value = None
    return mock


def _loaded_engine(
    mock: MagicMock,
    *,
    device_type: DeviceType = DeviceType.CUDA,
    selection_precision: Precision = Precision.FP32,
    batch_size: int = 1,
) -> DetectionEngine:
    """An engine holding *mock*, loaded, with a deliberately stale selection."""
    engine = DetectionEngine(backend_instance=mock, batch_size=batch_size)
    engine._selection = BackendSelection(
        backend=mock.backend_type,
        device_type=device_type,
        precision=selection_precision,
        device_index=0,
        reason="test",
    )
    with patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


def _cpu_pytorch_engine(precision: Precision | None) -> DetectionEngine:
    """An engine pinned to the PyTorch backend on a CPU device."""
    return DetectionEngine(
        model_family=ModelFamily.YOLO11,
        model_size=ModelSize.NANO,
        backend=BackendType.PYTORCH,
        device="cpu",
        precision=precision,
    )


# ---------------------------------------------------------------------------
# The request reaches the executor, or it is refused
# ---------------------------------------------------------------------------


# The name is fixed by the frozen `## CHECKS` and a bare citation must match it
# exactly, so it cannot be shortened to fit the line limit. The return
# annotation is dropped rather than wrapped, to keep the `def` on one line.
def test_an_explicit_precision_the_backend_does_not_execute_raises_and_names_backend_and_device():
    """covers: M2, A23, A26, E2, R:SILENT_COERCION.

    The COMMON case, not the edge: the only runtime precision hook in the
    library autocasts under CUDA alone, so an explicit fp16 on a CPU device is
    a request nothing executes. It used to be accepted and coerced in silence.
    """
    with pytest.raises(ConfigError) as exc:
        _cpu_pytorch_engine(Precision.FP16).load()

    message = str(exc.value).lower()
    assert "pytorch" in message, f"the error must name the backend: {message}"
    assert "cpu" in message, f"the error must name the device: {message}"
    assert "fp16" in message, f"the error must name the request: {message}"
    assert "fp32" in message, (
        "A26: the message must say what was executing instead — the request was "
        f"accepted and ignored before this error existed: {message}"
    )


def test_an_explicit_int8_request_raises_instead_of_being_reported() -> None:
    """covers: M2, E3, R:REPORTS_REQUEST.

    The shipping defect. `select_precision`'s CPU branch returned INT8 for an
    explicit int8 request, stored it on `BackendSelection`, and published it
    through `health_report()` while fp32 executed. Nothing in this library runs
    int8 at runtime, on any backend, on any device.
    """
    with pytest.raises(ConfigError) as exc:
        _cpu_pytorch_engine(Precision.INT8).load()

    assert "int8" in str(exc.value).lower()


def test_an_explicit_precision_on_an_artifact_backend_raises_including_fp32() -> None:
    """covers: M2, A3, A16.

    `fp32` is not a null request. A TensorRT engine built FP16 executes FP16;
    passing an explicit fp32 through to it would report one number while another
    ran — the int8 defect wearing a quieter hat. So the carve-out does not exist.
    """
    with pytest.raises(ConfigError) as exc:
        honoured_precision(BackendType.ONNX, DeviceType.CUDA, Precision.FP32, explicit=True)

    message = str(exc.value).lower()
    assert "onnx" in message
    assert "export" in message, f"the remedy for an artifact backend is to re-export: {message}"


def test_an_auto_selected_precision_that_cannot_be_honoured_does_not_raise() -> None:
    """covers: M3, A13, E1, E4, R:AUTO_RAISES.

    The other half of the decision, and the half that is NOT 'raise everywhere'.
    An unset precision is the caller expressing no preference: it must reach the
    backend and refuse nothing, on any device.
    """
    engine = _cpu_pytorch_engine(None)

    assert engine._requested_precision is None, (
        "the engine keeps the RAW request; None means the caller expressed none"
    )
    assert (
        honoured_precision(BackendType.PYTORCH, DeviceType.CPU, Precision.FP16) is Precision.FP32
    ), "an auto-selected fp16 on a CPU device is adjusted quietly, never raised"


def test_a_honourable_precision_reaches_and_changes_the_executing_backend() -> None:
    """covers: M1, A9, R:MOCK_ONLY_HONOUR.

    Asserted on the CONSTRUCTED backend's own state — not on a mock's call args,
    and not on the config that asked. Before load the backend has resolved no
    device, so it reports nothing rather than echoing the request.
    """
    hw = get_hardware_profile()
    backend = create_backend(BackendType.PYTORCH, hw, model_spec=_SPEC, precision=Precision.FP16)
    assert isinstance(backend, PyTorchBackend)
    assert backend.executing_precision is None, "an unloaded backend executes nothing"

    # The device the backend really resolved is what decides — never the
    # selection's device_type, which can disagree with it (A8).
    backend._resolve_precision("cuda:0")
    assert backend.executing_precision is Precision.FP16
    assert backend._fp16 is True, (
        "the backend's own numerics state changed; a mock that merely received "
        "the argument would prove nothing (R:MOCK_ONLY_HONOUR)"
    )

    cpu_backend = create_backend(
        BackendType.PYTORCH, hw, model_spec=_SPEC, precision=Precision.FP16
    )
    assert isinstance(cpu_backend, PyTorchBackend)
    cpu_backend._resolve_precision("cpu")
    assert cpu_backend.executing_precision is Precision.FP32
    assert cpu_backend._fp16 is False


def test_the_precision_verdict_is_total_over_the_backend_enum() -> None:
    """covers: M5, A27 — enumerated at runtime, so a sixth backend cannot slip in.

    A hand-written list would stay silent here. The enum is the source, exactly
    as `_roster.py` does it for execution verdicts.
    """
    missing = [b.value for b in BackendType if b not in RUNTIME_PRECISIONS]
    assert not missing, f"these backends have no precision verdict: {missing}"

    incomplete = [
        (b.value, d.value)
        for b in BackendType
        for d in DeviceType
        if d not in RUNTIME_PRECISIONS[b]
    ]
    assert not incomplete, f"these (backend, device) pairs have no verdict: {incomplete}"


def test_mps_and_cpu_do_not_claim_fp16() -> None:
    """covers: A3 — the autocast branch is gated on startswith("cuda") alone.

    Assuming MPS autocasts would ship a second silent coercion on Apple Silicon,
    a platform this project explicitly targets.
    """
    for device_type in (DeviceType.CPU, DeviceType.MPS):
        assert Precision.FP16 not in RUNTIME_PRECISIONS[BackendType.PYTORCH][device_type]
        with pytest.raises(ConfigError):
            honoured_precision(BackendType.PYTORCH, device_type, Precision.FP16, explicit=True)

    assert Precision.FP16 in RUNTIME_PRECISIONS[BackendType.PYTORCH][DeviceType.CUDA]


def test_the_verdict_reads_the_raw_request_not_the_degraded_selection() -> None:
    """covers: M8, E8.

    `select_precision` degrades an explicit request before any verdict could see
    it — `_degrade_from` on the GPU path, and an unconditional INT8 last resort.
    A verdict read from `BackendSelection.precision` would compare fp32 against
    fp32 and pass, having silently lost the request.
    """
    engine = _cpu_pytorch_engine(Precision.FP16)
    # Simulate exactly what _degrade_from does: the selection no longer carries
    # what the user asked for.
    engine._selection = dataclasses.replace(engine._selection, precision=Precision.FP32)

    assert engine._requested_precision is Precision.FP16
    with pytest.raises(ConfigError):
        engine.load()


def test_the_fallback_loop_does_not_discard_an_explicit_request() -> None:
    """covers: A20, E5, R:SWALLOWED_BY_FALLBACK.

    The loop used to re-derive the selection with `select_backend(..., backend_override=bt)`
    and NO `precision_override`, so an explicit request survived construction and
    was thrown away on fallback. Every candidate disqualified must surface an
    error that names precision — never a quiet landing on a backend that cannot
    honour the request either.
    """

    def _refuses_precision(*args: object, **kwargs: object) -> MagicMock:
        """A fallback candidate that cannot honour the request — i.e. any of the
        four artifact backends, which is what the loop actually reaches for."""
        assert kwargs.get("precision") is Precision.FP16, (
            "the loop must carry the RAW request to each candidate, not an auto value it re-derived"
        )
        assert kwargs.get("precision_explicit") is True
        backend = _mock_backend(BackendType.ONNX, executing=None)
        backend.load.side_effect = ConfigError("cannot honour precision 'fp16'")
        return backend

    # `pytorch` is the universal last resort and has NO fallbacks, so the
    # primary here is `onnx`, whose chain is [pytorch].
    engine = DetectionEngine(
        model_family=ModelFamily.YOLO11,
        model_size=ModelSize.NANO,
        backend=BackendType.ONNX,
        device="cpu",
        precision=Precision.FP16,
    )
    # The primary fails for an unrelated reason, so the loop reaches a fallback.
    primary = _mock_backend(BackendType.ONNX, executing=None)
    primary.load.side_effect = BackendLoadError("primary down")
    engine._backend = primary

    with (
        patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")),
        patch("yowo.engine.create_backend", side_effect=_refuses_precision),
        pytest.raises(BackendLoadError) as exc,
    ):
        engine.load()

    assert "precision" in str(exc.value).lower(), (
        "every candidate was disqualified for precision; the final error must "
        f"say so rather than merely 'all backends failed': {exc.value}"
    )


def test_a_user_supplied_backend_instance_is_judged_not_assumed_fp32() -> None:
    """covers: E6.

    The `backend_instance` branch hardcoded `precision=Precision.FP32` and
    `device_type=DeviceType.CPU`, so a caller passing both an instance and an
    explicit precision had the request silently discarded and never recorded.
    A user backend implements the Protocol, so it answers for itself.
    """
    mock = _mock_backend(executing=Precision.FP32)
    engine = DetectionEngine(backend_instance=mock, precision=Precision.FP16)

    assert engine._requested_precision is Precision.FP16, (
        "the request must be recorded, not replaced by a hardcoded FP32"
    )
    with (
        patch("yowo.engine.resolve_weights", return_value=Path("/fake/w.pt")),
        pytest.raises(ConfigError),
    ):
        engine.load()


# ---------------------------------------------------------------------------
# What is reported is what ran
# ---------------------------------------------------------------------------


def test_health_report_precision_comes_from_the_backend_not_the_request() -> None:
    """covers: M4, A15, A25, R:REPORTS_REQUEST.

    The selection holds INT8; the backend executes fp32. `precision_current`
    used to read the selection and say "int8" — a number an operator sizes
    hardware on.
    """
    mock = _mock_backend(executing=Precision.FP32)
    engine = _loaded_engine(mock, selection_precision=Precision.INT8)

    assert engine.health_report().precision_current == "fp32", (
        "precision_current must be produced by the executing backend"
    )


def test_an_unloaded_engine_reports_unknown_not_the_request() -> None:
    """A10's probe. NOT cited by the frozen `## CHECKS` — the ruling that added
    `· probe:` to A10 made it a gate-bound referent without giving it a
    citation, so `add gate` will list A10 unbound. The behaviour is here and
    passing, so the amendment is a one-line CHECKS edit rather than new work.

    An operator polling health during a slow load is exactly the reader who
    would act on a startup value, so the field must not echo the request while
    nothing is running yet. `HealthReport.status` carries the "not ready" half.
    """
    from yowo.types import HealthStatus

    engine = DetectionEngine(backend_instance=_mock_backend(executing=Precision.FP32))
    report = engine.health_report()

    assert report.precision_current == "unknown", (
        "before load there is no executing precision, only a request"
    )
    assert report.status is not HealthStatus.READY, (
        "the pair is complete without the precision field guessing"
    )


def test_precision_current_is_no_longer_allowlisted_as_producerless() -> None:
    """covers: M4 — the allowlist reason at test_metrics_truth.py:50 WAS the defect.

    That file exists to catch "reported without a consumer", and this field's
    exemption reason — "read straight off the resolved BackendSelection" — was a
    restatement of the thing it was exempting.
    """
    from tests.unit.test_metrics_truth import _ALLOWLIST

    assert "precision_current" not in _ALLOWLIST, (
        "precision_current has a real producer now — the executing backend"
    )


def test_the_public_surface_loses_no_name_and_changes_no_type() -> None:
    """covers: M4, A2, A15, R:PUBLIC_BREAK.

    Four clauses. `HealthReport` is in `yowo.__all__`, so widening this field to
    `str | None` — which would have matched `memory_pct: float | None` three
    lines above it — is a public type change m4's SCOPE defers to m5. The
    honesty is bought at the VALUE instead: the documented sentinel "unknown",
    used for nothing else.
    """
    import yowo
    import yowo.tune

    hints = typing.get_type_hints(HealthReport)
    assert hints["precision_current"] is str, (
        "the public annotation must not move — the sentinel is the cost of that"
    )

    mock = _mock_backend(BackendType.ONNX, executing=None)
    engine = _loaded_engine(mock, selection_precision=Precision.FP16)
    report = engine.health_report()
    assert report.precision_current == "unknown", (
        "a backend that cannot determine what it executes says so, and never echoes the request"
    )

    others = {
        name: value
        for name, value in dataclasses.asdict(report).items()
        if name != "precision_current" and value == "unknown"
    }
    assert not others, f"one sentinel, one meaning — 'unknown' leaked into {others}"

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

    `_load_tune_profile` was the only writer of a non-None precision that no
    human typed. While it stood, "explicit" was undecidable and the loudest rung
    would fire on a value nobody asked for. The profile carries INT8
    deliberately — a default-valued fixture would compare equal either way (Q4).
    """
    from yowo.config import InferenceConfig
    from yowo.engine import _load_tune_profile
    from yowo.tune._profile import TuneProfile, compute_fingerprint

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
    config = InferenceConfig(model_family=ModelFamily.YOLO11, model_size=ModelSize.NANO)

    with patch("yowo.tune._profile.load_profile", return_value=profile):
        result = _load_tune_profile(_SPEC, config, hw)

    assert result.precision is None, (
        "a profile's precision was an FPS measurement over configurations that "
        "were identical because nothing consumed the value — not a request"
    )
    assert result.backend == BackendType.ONNX, "the profile's backend still applies"
    assert result.batch_size == 4, "the profile's batch size still applies"


def test_a_pre_change_tune_profile_still_loads(tmp_path: Path) -> None:
    """covers: M7 — the control.

    `load_profile` builds from named keys, so a key it stops reading is inert
    rather than fatal: no migration is needed. This check exists to catch a
    loader rewritten into `TuneProfile(**raw)`, which would turn an old file on
    disk into a crash.
    """
    import yaml

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

    Three defects in one band. `_apply_oom_recovery` returned after the dead
    tier-2 call, so tier-1 was unreachable in [0.90, 0.95); and
    `_try_precision_fallback` set DEGRADED without setting `_oom_recovering`,
    so the restore branch could never fire and one tick latched DEGRADED for the
    life of the process.
    """
    from yowo.types import HealthStatus

    engine = _loaded_engine(_mock_backend(), batch_size=8)
    assert engine.health == HealthStatus.READY

    engine._apply_oom_recovery(0.92)
    assert engine._batch_size == 4, "the band satisfies tier-1 and must get its recovery"

    engine._apply_oom_recovery(0.70)
    assert engine._batch_size == 8, "pressure cleared — the batch size must come back"
    assert engine.health == HealthStatus.READY, "DEGRADED must not be a one-way latch"


def test_a_no_op_recovery_counts_no_degradation() -> None:
    """covers: A7, R:PHANTOM_DEGRADATION.

    Exactly one degradation across the 0.92 tick — the batch halving that really
    happened. The dead precision rung used to count one for a recovery that was
    never attempted, while nothing else occurred. An alert threshold of "any
    degradation" that ticks without a cause is an alert that gets muted.
    """
    engine = _loaded_engine(_mock_backend(), batch_size=8)

    before = engine.export_metrics()["degradations_total"]
    engine._apply_oom_recovery(0.92)
    after = engine.export_metrics()["degradations_total"]

    assert isinstance(before, int) and isinstance(after, int)
    assert after - before == 1, "exactly one recovery happened, so exactly one counts"
    assert engine._oom_recovering is True, (
        "the count was one because the DEAD rung incremented it while doing "
        "nothing, and it never set _oom_recovering — so the engine could never "
        "clear. A degradation that cannot be undone was not a recovery."
    )
    assert not hasattr(engine, "_try_precision_fallback"), (
        "A17: a tier this suite cannot refute is a tier that should not ship"
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

    # The four call sites through which this file ASKS for a precision. A
    # substring scan would also flag comments and the `_degrade_from`
    # simulation, which are not requests; the AST distinguishes them.
    requesters = {
        "_cpu_pytorch_engine": 0,
        "honoured_precision": 2,
        "DetectionEngine": None,
        "create_backend": None,
    }
    # The single exemption, with its reason stated — an allowlist without one
    # is how a genuine gap gets exempted. In this check fp32 IS the non-default
    # subject: the point is that an explicit fp32 to an artifact backend is not
    # a null request, so it has to be the value asked for.
    exempt = "test_an_explicit_precision_on_an_artifact_backend_raises_including_fp32"

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef) and func.name == exempt:
            for inner in ast.walk(func):
                inner._q4_exempt = True  # type: ignore[attr-defined]

    offenders: list[str] = []
    for node in ast.walk(tree):
        if getattr(node, "_q4_exempt", False):
            continue
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in requesters:
            continue
        asked: list[ast.expr] = [kw.value for kw in node.keywords if kw.arg == "precision"]
        index = requesters[name]
        if index is not None and len(node.args) > index:
            asked.append(node.args[index])
        for value in asked:
            if ast.unparse(value).endswith("Precision.FP32"):
                offenders.append(f"line {node.lineno}: {name}(... {ast.unparse(value)})")

    assert not offenders, (
        "these call sites REQUEST the default precision, where a value that "
        f"never travelled compares equal to one that did: {offenders}"
    )


def test_every_frozen_check_name_exists_in_this_module() -> None:
    """The node's `## CHECKS` cite bare names; a citation that matches nothing
    binds nothing (M12). This asserts the suite and the frozen bundle agree."""
    node = Path(__file__).parents[2] / ".add" / "tasks" / "precision-plumbing.md"
    if not node.is_file():  # pragma: no cover - bundle not present in sdist
        pytest.fail(f"the frozen node is missing: {node}")

    body = node.read_text(encoding="utf-8")
    section = body.split("## CHECKS", 1)[1].split("## EVIDENCE", 1)[0]
    cited = [
        line.strip().lstrip("- ").split(" ·", 1)[0]
        for line in section.splitlines()
        if line.strip().startswith("- test_")
    ]
    here = set(globals())
    missing = [name for name in cited if name not in here]
    assert not missing, f"frozen checks with no test of that name: {missing}"


def _unused(profile: HardwareProfile) -> None:  # pragma: no cover - typing anchor
    """Keeps the HardwareProfile import honest for type checkers."""
    assert profile is not None
