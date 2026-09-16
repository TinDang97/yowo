"""Every cache key names the inputs its artifact depends on.

Three caches key artifacts on this machine: the tune profile, the INT8
calibration table and the TensorRT engine cache. Measured 2026-09-16, each
omitted inputs that change what the artifact IS. These checks pin the
invalidation set that closes those holes -- and pin it MECHANICALLY, so an
element added later is exercised without anyone remembering to add it here.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from yowo.cache._keys import ABSENT, InvalidationSet, calibration_digest
from yowo.hardware import get_hardware_profile

# A baseline with EVERY field populated. Perturbation needs a defined starting
# value per field, and `None` has no "different value of the same kind".
_BASELINE = InvalidationSet(
    model="yolo11n",
    backend="tensorrt",
    precision="int8",
    runtime_versions=("torch=2.4.0", "ort=1.18.0"),
    gpu_name="NVIDIA A10G",
    gpu_arch="8.6",
    driver_version="535.129.03",
    tensorrt_version="8.6.1",
    shape_profile="1x3x640x640",
    calibration_digest="deadbeef",
    cpu_count=8,
    platform="Linux-6.1-x86_64",
)


def _perturb(value: object) -> object:
    """Return a different value of the same kind.

    Raises on a kind it does not know. That is deliberate: a field added in a
    type this helper cannot vary must FAIL the sweep loudly, not be skipped
    into a green.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-perturbed"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, tuple):
        return (*value, "perturbed")
    raise AssertionError(
        f"no perturbation defined for {type(value).__name__} -- "
        "a new InvalidationSet field needs one here before it can be checked"
    )


def test_every_element_of_the_invalidation_set_changes_the_key() -> None:
    names = [f.name for f in dataclasses.fields(InvalidationSet)]
    assert names, "an empty invalidation set would pass every other check here"

    base = _BASELINE.key()
    for name in names:
        current = getattr(_BASELINE, name)
        other = dataclasses.replace(_BASELINE, **{name: _perturb(current)})
        assert other.key() != base, f"changing `{name}` did not change the key"


def test_the_same_inputs_give_the_same_key() -> None:
    # A key that never repeats satisfies every separation check above and makes
    # all three caches permanently useless.
    twin = dataclasses.replace(_BASELINE)
    assert twin.key() == _BASELINE.key()


def test_an_unprobeable_element_is_recorded_as_absent_not_as_empty() -> None:
    absent = dataclasses.replace(_BASELINE, driver_version=None)
    empty = dataclasses.replace(_BASELINE, driver_version="")
    assert absent.key() != empty.key()
    assert ABSENT in absent.render()


def test_reordering_the_declaration_does_not_change_the_key() -> None:
    # The rendering is sorted by field name, so the key survives a field
    # reorder. Otherwise every reorder silently flushes every cache on disk.
    rendered = _BASELINE.render()
    lines = rendered.splitlines()
    assert lines == sorted(lines), "rendering is not in canonical field order"


def test_two_elements_cannot_swap_values_into_the_same_key() -> None:
    swapped = dataclasses.replace(_BASELINE, backend="int8", precision="tensorrt")
    assert swapped.key() != _BASELINE.key()

    # A value carrying the delimiters must not be able to forge another
    # field's line. Note what a collision leg alone would NOT prove here:
    # rendering emits exactly one prefixed line per field in sorted order, so
    # an injected newline can only ADD a line, never displace one, and no
    # forged pair can collide even with escaping removed. The property the
    # escaping alone carries is this one -- the rendering stays one line per
    # field, so it can be read back unambiguously.
    forged = dataclasses.replace(_BASELINE, model="yolo11n\nbackend=onnx")
    assert len(forged.render().splitlines()) == len(dataclasses.fields(InvalidationSet))
    assert forged.key() != dataclasses.replace(_BASELINE, backend="onnx").key()


# ---------------------------------------------------------------------------
# (a) the tune profile
# ---------------------------------------------------------------------------


def test_a_profile_swept_on_one_backend_does_not_set_the_batch_size_for_another(
    tmp_path: Path,
) -> None:
    from yowo.hardware import get_hardware_profile
    from yowo.tune._profile import TuneProfile, compute_fingerprint, load_profile, save_profile

    hw = get_hardware_profile()
    dest = tmp_path / "yolo11n.yaml"
    save_profile(
        TuneProfile(
            model="yolo11n",
            backend="pytorch",
            batch_size=16,
            precision="fp32",
            fps_achieved=42.0,
            tuned_at="2026-09-16T00:00:00",
            fingerprint=compute_fingerprint(hw),
        ),
        dest,
    )

    # Same machine, same model -- but the sweep ran on pytorch. A tensorrt run
    # must not inherit pytorch's optimal batch size.
    assert load_profile("yolo11n", hw, dest, backend="pytorch") is not None
    assert load_profile("yolo11n", hw, dest, backend="tensorrt") is None


def test_a_profile_whose_stored_precision_differs_is_a_miss(tmp_path: Path) -> None:
    from yowo.hardware import get_hardware_profile
    from yowo.tune._profile import TuneProfile, compute_fingerprint, load_profile, save_profile

    hw = get_hardware_profile()
    dest = tmp_path / "yolo11n.yaml"
    save_profile(
        TuneProfile(
            model="yolo11n",
            backend="pytorch",
            batch_size=16,
            precision="fp32",
            fps_achieved=42.0,
            tuned_at="2026-09-16T00:00:00",
            fingerprint=compute_fingerprint(hw),
        ),
        dest,
    )
    assert load_profile("yolo11n", hw, dest, precision="fp32") is not None
    assert load_profile("yolo11n", hw, dest, precision="int8") is None


# ---------------------------------------------------------------------------
# (b) the INT8 calibration table
# ---------------------------------------------------------------------------


def test_the_calibration_cache_key_follows_the_images_not_the_directory_name(
    tmp_path: Path,
) -> None:
    first = tmp_path / "set-a"
    first.mkdir()
    (first / "0.jpg").write_bytes(b"image-one")
    (first / "1.jpg").write_bytes(b"image-two")

    baseline = calibration_digest(sorted(first.glob("*.jpg")))

    # Renaming the directory must NOT invalidate -- the images are the same.
    renamed = tmp_path / "set-b"
    first.rename(renamed)
    assert calibration_digest(sorted(renamed.glob("*.jpg"))) == baseline

    # Changing an image inside it MUST invalidate, which is the measured hole:
    # today the table is named from the engine path alone and never notices.
    (renamed / "1.jpg").write_bytes(b"image-two-but-different")
    assert calibration_digest(sorted(renamed.glob("*.jpg"))) != baseline


def test_the_export_names_its_calibration_table_after_the_images() -> None:
    from yowo.export import _exporter

    source = Path(_exporter.__file__).read_text(encoding="utf-8")
    assert 'with_suffix(".calib")' not in source, (
        "the calibration table is still keyed on the engine path alone, so a "
        "second export with different images reuses the first table"
    )
    assert "calibration_digest" in source


# ---------------------------------------------------------------------------
# (c) the TensorRT engine cache
# ---------------------------------------------------------------------------


def test_the_tensorrt_engine_cache_uses_a_prefix_this_package_computes() -> None:
    from yowo.backends._tensorrt import trt_provider_options

    options = trt_provider_options(0, Path("/models/yolo11n.onnx"), _BASELINE)
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_engine_cache_prefix"] == _BASELINE.key()

    other = dataclasses.replace(_BASELINE, precision="fp16")
    assert (
        trt_provider_options(0, Path("/models/yolo11n.onnx"), other)["trt_engine_cache_prefix"]
        != options["trt_engine_cache_prefix"]
    )


def test_the_tensorrt_engine_cache_effect_is_recorded_as_never_executed() -> None:
    # No runner in this project has a GPU, so the EP-level behaviour of the
    # prefix above has never run. The node must say so, or a future reader
    # takes this file's green as proof that it did.
    node = Path(".add/tasks/artifact-cache-keys.md").read_text(encoding="utf-8")
    assert "NEVER EXECUTED" in node


# ---------------------------------------------------------------------------
# the driver version -- the one element nothing probed at all
# ---------------------------------------------------------------------------


def test_the_driver_version_is_asked_for_and_survives_a_driver_that_cannot_report_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yowo.hardware import _detect

    assert "driver_version" in _detect._SMI_OPTIONAL_FIELDS

    asked: list[str] = []

    def fake_query(fields: str) -> subprocess.CompletedProcess[str]:
        asked.append(fields)
        if "driver_version" in fields:
            # A driver too old to know the field fails the WHOLE query.
            return subprocess.CompletedProcess(["nvidia-smi"], 2, "", "unknown field")
        return subprocess.CompletedProcess(
            ["nvidia-smi"], 0, "0, Tesla T4, 16384 MiB, 15000 MiB\n", ""
        )

    monkeypatch.setattr(_detect, "_run_smi_query", fake_query)
    devices = _detect._detect_gpus_smi()

    assert any("driver_version" in f for f in asked), "the query never asked for it"
    assert devices, "the fallback must still return the GPU it could see"
    assert devices[0].driver_version is None, "an unprobeable element is absent, not empty"


def test_the_caller_can_still_name_its_own_profile_path(tmp_path: Path) -> None:
    """The public escape hatch A11 leans on.

    Changing the DEFAULT key scheme is only harmless because a caller who
    wants a specific location can still name one. Both `save_profile` and
    `load_profile` take a path and both are exported from `yowo.tune`; drop
    either and the reasoning behind A11 stops holding.
    """
    import inspect

    import yowo.tune as tune

    for name in ("save_profile", "load_profile"):
        assert name in tune.__all__, f"{name} left the public surface"
        assert "path" in inspect.signature(getattr(tune, name)).parameters

    hw = get_hardware_profile()
    chosen = tmp_path / "somewhere" / "of-my-choosing.yaml"
    chosen.parent.mkdir()
    tune.save_profile(
        tune.TuneProfile(
            model="yolo11n",
            backend="pytorch",
            batch_size=4,
            precision="fp32",
            fps_achieved=1.0,
            tuned_at="2026-09-16T00:00:00",
            fingerprint=tune.compute_fingerprint(hw),
        ),
        chosen,
    )
    assert chosen.exists(), "the caller's path was ignored"
    assert tune.load_profile("yolo11n", hw, chosen) is not None
