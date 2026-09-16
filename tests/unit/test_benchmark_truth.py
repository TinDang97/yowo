"""Every column in a benchmark or tune row must be measured for that row.

Re-verified by measurement 2026-09-16 (the milestone recorded these on
2026-09-13; all four still held). Requesting the onnx backend on this machine
logs `Primary backend onnx failed, trying pytorch`, and the row still reports
`format=onnx`:

    requested   row.format   model_size_mb   device   fps
    onnx        onnx                 5.354      cpu   16.97
    pytorch     pytorch              5.354      cpu   20.31

**Both rows executed pytorch.** A user reading that table to choose a
deployment format is comparing pytorch against pytorch and concluding that
onnx is 17% slower. `model_size_mb` is the source `.pt` in both rows — the
real `yolo11n.pt` is 5.354 MB — because `_get_model_size_mb` stats the spec's
weights, never the artifact that loaded.

Two more, from reading the same function:

  - the OBB branch constructs `OBBEngine` without `backend=`, while `classify`
    and `detect` both pass it, and every row is still labelled
    `format=backend_type.value`. `OBBEngine.__init__` accepts `backend` — it
    was simply omitted.
  - the no-latencies early return hardcodes `device="unknown"` although the
    real device is bound at that point.

And the tune sweep ranks configurations on `np.zeros((640, 640, 3))`, whose
winner `save_profile` persists and `yowo/engine.py:205-209` then applies to
production inference.

Authored red under ADD task `benchmark-truth`.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from yowo.benchmark._runner import BenchmarkResult, _get_model_size_mb, run_single_backend
from yowo.types import (
    BackendType,
    BoundingBox,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)

RUNNER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "yowo" / "benchmark" / "_runner.py"

#: Documentation of what was measured, NOT asserted (Q11). yolo11n / pytorch /
#: cpu, median of 15 runs, 2026-09-16 on darwin/arm64.
#:
#: The milestone box states "0.15 ms of postprocess, excluding 100% of the
#: 176-281 ms it is ranking". That did NOT reproduce here and is not repeated.
#: What reproduces is the KIND of error, not its magnitude: the synthetic frame
#: produces no detections at all, so the postprocess work the sweep is ranking
#: is absent from the measurement that ranks it.
OBSERVED_SYNTHETIC_FRAME = {"boxes": 0, "median_ms": 50.31}
OBSERVED_REAL_FRAME = {"boxes": 5, "median_ms": 63.63}


def _make_detection(inference_time_ms: float = 10.0) -> Detection:
    frame = Frame(pixels=np.zeros((480, 640, 3), dtype=np.uint8), source_id="bench")
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    return Detection(
        frame=frame,
        boxes=(BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0, confidence=0.9, class_id=0),),
        inference_time_ms=inference_time_ms,
        backend=BackendType.PYTORCH,
        model_spec=spec,
    )


def _mock_engine(executed: str = "pytorch", device: str = "cpu") -> MagicMock:
    """An engine that FELL BACK: asked for one backend, selected another."""
    engine = MagicMock()
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=False)
    engine.detect.return_value = [_make_detection()]
    engine.selection.backend.value = executed
    engine.selection.device_type.value = device
    return engine


def _run(backend: BackendType, engine_cls: MagicMock, task: str = "detect") -> BenchmarkResult:
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    return run_single_backend(
        model_spec=spec,
        backend_type=backend,
        images=[Path(f"/fake/img{i}.jpg") for i in range(3)],
        image_ids=None,
        gt_ann_path=None,
        task=task,
        warmup_passes=1,
    )


@patch("yowo.benchmark._runner.DetectionEngine")
def test_the_row_reports_the_backend_that_executed_not_the_one_requested(
    engine_cls: MagicMock,
) -> None:
    """binds R:REQUEST_AS_RESULT — onnx and pytorch rows both executed pytorch."""
    engine_cls.return_value = _mock_engine(executed="pytorch")
    result = _run(BackendType.ONNX, engine_cls)
    assert result.format == "pytorch", (
        f"the row reports format={result.format!r} after the engine selected pytorch. "
        f"A reader comparing formats would attribute pytorch's numbers to onnx."
    )


@patch("yowo.benchmark._runner.DetectionEngine")
def test_the_row_keeps_the_requested_backend_beside_the_executed_one(
    engine_cls: MagicMock,
) -> None:
    """Dropping the request would hide that a fallback happened at all."""
    engine_cls.return_value = _mock_engine(executed="pytorch")
    result = _run(BackendType.ONNX, engine_cls)
    assert result.requested_format == "onnx"
    assert result.format != result.requested_format, (
        "this row fell back; the two fields differing is what makes that legible"
    )


@patch("yowo.benchmark._runner.DetectionEngine")
def test_the_artifact_size_is_absent_rather_than_the_source_checkpoints(
    engine_cls: MagicMock,
) -> None:
    """binds R:SOURCE_FOR_ARTIFACT — 5.354 MB was reported for every format.

    No backend exposes the path of the artifact it loaded (measured 2026-09-16:
    `selection` carries only backend/device/precision/reason, and
    `PyTorchBackend` holds a spec, not a path). So for a row that did not run
    the source checkpoint, the honest size is no size.
    """
    engine_cls.return_value = _mock_engine(executed="onnx")
    result = _run(BackendType.ONNX, engine_cls)
    assert result.model_size_mb is None, (
        f"the row reports {result.model_size_mb} MB for an onnx execution, but the only "
        f"file this code can stat is the source .pt, which onnx did not load."
    )


def test_an_unmeasurable_size_is_none_and_never_zero() -> None:
    """binds R:PLACEHOLDER_AS_MEASUREMENT — a literal 0.0 renders as '0.0 MB'."""
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    with patch("yowo.benchmark._runner.resolve_weights", side_effect=OSError("no such file")):
        assert _get_model_size_mb(spec) is None
    with patch("yowo.benchmark._runner.Path.stat", side_effect=OSError("gone")):
        assert _get_model_size_mb(spec) is None


def _row(**over: object) -> BenchmarkResult:
    base = dict(
        format="onnx",
        requested_format="onnx",
        map_50_95=None,
        map_50=None,
        fps_avg=10.0,
        latency_p50_ms=1.0,
        latency_p95_ms=1.0,
        latency_p99_ms=1.0,
        model_size_mb=None,
        device="cpu",
        num_images=1,
    )
    base.update(over)
    return BenchmarkResult(**base)  # type: ignore[arg-type]


def test_the_report_renders_an_absent_size_as_a_dash_not_a_number(capsys) -> None:  # type: ignore[no-untyped-def]
    """The table is where a sentinel would be read as a finding."""
    from yowo.benchmark._report import render_table

    render_table([_row()])
    rendered = capsys.readouterr().out
    assert "0.0 MB" not in rendered, "an unmeasurable size rendered as a measurement"
    assert "-" in rendered

    # And a fallback must be visible to the person reading the table, not only
    # in the dataclass: two pytorch rows, one of them labelled onnx, is the
    # defect this task exists to remove.
    render_table([_row(format="pytorch", requested_format="onnx")])
    fallback = capsys.readouterr().out
    assert "onnx" in fallback and "pytorch" in fallback


def test_the_json_report_emits_null_for_an_absent_size() -> None:
    """A consumer must not parse 0.0 as a measurement."""
    from yowo.benchmark._report import results_to_json

    entry = results_to_json([_row(format="pytorch", requested_format="onnx")])["results"][0]
    assert entry["model_size_mb"] is None
    assert entry["format"] == "pytorch"
    assert entry["requested_format"] == "onnx"


@patch("yowo.benchmark._runner.DetectionEngine")
def test_a_row_without_latencies_still_reports_the_selected_device(
    engine_cls: MagicMock,
) -> None:
    """binds R:PLACEHOLDER_AS_MEASUREMENT — `device` is bound; 'unknown' was needless."""
    engine = _mock_engine(executed="pytorch", device="cuda")
    engine_cls.return_value = engine
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    result = run_single_backend(
        model_spec=spec,
        backend_type=BackendType.PYTORCH,
        images=[],  # no images -> no latencies -> the early return
        image_ids=None,
        gt_ann_path=None,
        task="detect",
        warmup_passes=0,
    )
    assert result.device == "cuda", f"device reported as {result.device!r}"


def _engine_constructions() -> dict[str, list[str]]:
    """Every `*Engine(...)` call in the runner, mapped to its keyword names."""
    tree = ast.parse(RUNNER_SOURCE.read_text())
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if not name.endswith("Engine"):
            continue
        found[name] = [kw.arg or "**" for kw in node.keywords]
    return found


def test_the_obb_path_passes_the_backend_it_is_labelled_with() -> None:
    """classify and detect pass `backend=`; obb did not, yet every row is labelled."""
    constructions = _engine_constructions()
    assert "OBBEngine" in constructions, "the OBB branch no longer constructs OBBEngine"
    assert "backend" in constructions["OBBEngine"], (
        "OBBEngine is constructed without `backend=`, so the requested backend is "
        "never executed — while the row is still labelled with it. "
        "OBBEngine.__init__ accepts `backend`; it was simply omitted."
    )


def test_every_engine_construction_in_the_benchmark_path_receives_the_backend() -> None:
    """Fixing one branch leaves the next branch free to repeat it."""
    missing = sorted(
        name for name, kwargs in _engine_constructions().items() if "backend" not in kwargs
    )
    assert not missing, (
        f"these engines are constructed in the benchmark path without `backend=`: "
        f"{missing}. Every row is labelled with the requested backend, so every "
        f"construction must receive it."
    )


def test_a_sweep_without_representative_frames_is_marked_unrepresentative() -> None:
    """binds R:RANK_WITHOUT_COST — the winner auto-applies to production.

    `save_profile` persists the sweep winner and `yowo/engine.py:205-209` loads
    and applies it. A ranking measured on a frame that yields no detections
    excluded the postprocess cost it was ranking.
    """
    from yowo.tune._profile import TuneProfile

    fields = {f.name for f in dataclasses.fields(TuneProfile)}
    assert "postprocess_representative" in fields, (
        "TuneProfile records no flag saying whether its measurement included the "
        "postprocess cost it ranked. A warning at sweep time is gone by the time "
        "the profile is applied."
    )
    profile = TuneProfile(
        model="yolo11n",
        backend="pytorch",
        batch_size=1,
        precision="fp32",
        fps_achieved=20.0,
        tuned_at="2026-09-16T00:00:00",
        fingerprint="abcd1234",
    )
    assert profile.postprocess_representative is False, (
        "the default must be the honest one: a sweep that was not given "
        "representative frames did not measure the cost it ranked"
    )


def test_a_sweep_given_representative_frames_is_marked_representative() -> None:
    """The flag must track the input, not be a constant."""
    import inspect

    from yowo.tune._sweep import _measure_config, run_sweep

    for fn in (_measure_config, run_sweep):
        params = inspect.signature(fn).parameters
        assert "sample_frames" in params, (
            f"{fn.__name__} takes no `sample_frames`, so a caller cannot supply the "
            f"representative input the ranking needs"
        )


def test_a_profile_written_before_the_flag_existed_still_loads(tmp_path: Path) -> None:
    """binds E6 — explicit `raw[...]` keys would raise on a missing one."""
    import yaml

    from yowo.tune._profile import load_profile

    old = {
        "model": "yolo11n",
        "backend": "pytorch",
        "batch_size": 1,
        "precision": "fp32",
        "fps_achieved": 20.0,
        "tuned_at": "2026-09-13T00:00:00",
        "fingerprint": "abcd1234",
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(old), encoding="utf-8")

    hw = MagicMock()
    with patch("yowo.tune._profile.compute_fingerprint", return_value="abcd1234"):
        loaded = load_profile("yolo11n", hw, path=path)
    assert loaded is not None, "a profile written before the flag existed must still load"
    assert loaded.postprocess_representative is False, (
        "a profile that predates the flag cannot have measured representatively"
    )


def test_the_recorded_postprocess_measurement_is_documentation_not_an_assertion() -> None:
    """Q11 — and the box's own 176-281 ms figure did not reproduce here.

    The OBSERVED_ constants say what one machine saw on one date. Asserting
    against them would make CI a referendum on runner hardware.
    """
    import inspect

    this = "test_the_recorded_postprocess_measurement_is_documentation_not_an_assertion"
    offenders = []
    for name, obj in list(globals().items()):
        if not name.startswith("test_") or not callable(obj) or name == this:
            continue
        if "OBSERVED_" in inspect.getsource(obj):
            offenders.append(name)
    assert not offenders, (
        f"{offenders} reference a recorded per-machine measurement. Timings belong "
        f"in the docstring; CI must not become a referendum on runner hardware."
    )
    # The STRUCTURAL finding those numbers record — no detections at all on the
    # synthetic frame, so no postprocess work to measure — is not a timing, and
    # it is the whole reason the sweep needs representative input.
    assert OBSERVED_SYNTHETIC_FRAME["boxes"] == 0
    assert OBSERVED_REAL_FRAME["boxes"] > 0


def test_the_profile_the_engine_applies_carries_the_flag() -> None:
    """binds A13 — the harmed party is every user of the tuned model, silently.

    Measured 2026-09-16: `yowo/engine.py` imports `load_profile` and applies the
    result, so a winner chosen on a frame that produced no detections becomes
    the production configuration. That is why the caveat lives on the persisted
    profile rather than in a warning printed once at sweep time: this consumer
    reads the profile, long after any sweep output has scrolled away.
    """
    import dataclasses
    import inspect

    import yowo.engine as engine_module
    from yowo.tune._profile import TuneProfile

    source = inspect.getsource(engine_module)
    assert "load_profile" in source, (
        "yowo.engine no longer applies a tuned profile. If that is deliberate, "
        "this check's premise changed and the flag's justification with it."
    )

    # The consumer can therefore see whether the ranking measured what it ranked.
    fields = {f.name for f in dataclasses.fields(TuneProfile)}
    assert "postprocess_representative" in fields
    assert "postprocess_representative" in inspect.getsource(
        inspect.getmodule(TuneProfile) or engine_module
    )
