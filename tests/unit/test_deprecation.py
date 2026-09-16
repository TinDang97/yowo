"""A public name can be retired without breaking anyone.

Before this, `src/` contained ZERO `DeprecationWarning` while the package sat
at 2.5.0 on PyPI with 191 public names. Two source comments already deferred
work to "the deprecation policy" -- a policy that did not exist -- and m4 was
constrained to additive-only as a direct consequence.

The rule, decided by the author 2026-09-16: a deprecated name warns for at
least one MINOR release and is removed no sooner than the next MAJOR; "public"
means a name in a module's `__all__`; and a recorded removal version is a
promise, not an enforcement.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import warnings
from pathlib import Path

import pytest

from yowo._deprecation import DEPRECATIONS, Deprecation, warn_deprecated
from yowo.types import ExportResult


def test_the_deprecation_warning_actually_fires() -> None:
    # CPython hides DeprecationWarning outside __main__, so a check that relies
    # on the ambient filter passes whether or not the warning was ever emitted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ExportResult(
            model_name="yolo11n",
            format="onnx",
            precision="fp16",
            output_path="/tmp/x.onnx",
            file_size_bytes=1,
            export_time_s=0.1,
            created_at="2026-09-16T00:00:00",
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "constructing a deprecated name emitted no DeprecationWarning"
    )


def test_the_warning_names_the_replacement_and_both_versions() -> None:
    record = DEPRECATIONS["ExportResult"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(record)
    message = str(caught[0].message)
    assert "ExportResult" in message
    assert record.since in message, "the message does not say when it was deprecated"
    assert record.removed_in in message, "the message does not say when it goes"
    assert record.instead in message, "a warning with no replacement leaves the caller stuck"


def test_a_deprecated_name_still_works_exactly_as_before() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = ExportResult(
            model_name="yolo11n",
            format="onnx",
            precision="fp16",
            output_path="/tmp/x.onnx",
            file_size_bytes=42,
            export_time_s=1.5,
            created_at="2026-09-16T00:00:00",
        )
    assert result.model_name == "yolo11n"
    assert result.file_size_bytes == 42
    fields = [f.name for f in dataclasses.fields(ExportResult)]
    assert fields == [
        "model_name",
        "format",
        "precision",
        "output_path",
        "file_size_bytes",
        "export_time_s",
        "created_at",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.model_name = "other"  # type: ignore[misc]


def test_the_warning_points_at_the_callers_line() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ExportResult(
            model_name="m",
            format="onnx",
            precision="fp16",
            output_path="/tmp/x",
            file_size_bytes=1,
            export_time_s=0.1,
            created_at="2026-09-16T00:00:00",
        )
    origin = Path(caught[0].filename).name
    assert origin == Path(__file__).name, (
        f"the warning blames {origin}; a stacklevel naming our own module tells "
        "a user nothing about their code"
    )


def test_nothing_inside_the_package_uses_a_deprecated_name() -> None:
    # Otherwise the package warns its own users about its own internals.
    offenders: list[str] = []
    for name in DEPRECATIONS:
        for path in Path("src/yowo").rglob("*.py"):
            if path.name in ("types.py", "_deprecation.py"):
                continue  # the definition and the mechanism
            text = path.read_text(encoding="utf-8")
            if re.search(rf"\b{re.escape(name)}\s*\(", text):
                offenders.append(f"{path}: constructs {name}")
    assert not offenders, "the package uses its own deprecated names:\n  " + "\n  ".join(offenders)


def test_every_deprecation_is_enumerable_from_one_record() -> None:
    assert DEPRECATIONS, "no deprecation is recorded, so the box's gate is unmet"
    for name, record in DEPRECATIONS.items():
        assert isinstance(record, Deprecation)
        assert record.name == name, "the mapping key and the record disagree"
        for field in dataclasses.fields(Deprecation):
            assert getattr(record, field.name), f"{name}.{field.name} is empty"
    # The record is the single source: the message is built from it, so a
    # message and its record cannot drift apart. Count EMISSION SITES across
    # the package, not mentions of the word -- the first version of this check
    # counted the string in prose and reported 3 for a module with one call.
    emitters: list[str] = []
    for path in Path("src/yowo").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name != "warn":
                continue
            if any(getattr(a, "id", None) == "DeprecationWarning" for a in node.args):
                emitters.append(f"{path}:{node.lineno}")
    assert len(emitters) == 1, (
        "the warning is emitted from more than one place, so the record is not "
        f"the single source: {emitters}"
    )


def test_a_removal_version_that_has_passed_still_warns_rather_than_raises() -> None:
    stale = Deprecation(name="Ancient", since="0.1.0", removed_in="0.2.0", instead="something.Else")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(stale)  # must not raise
    assert caught and issubclass(caught[0].category, DeprecationWarning)


def test_the_stability_policy_exists_and_names_the_mechanism_it_describes() -> None:
    policy = Path("docs/stability-policy.md")
    assert policy.exists(), "the policy two source comments already defer to does not exist"
    text = policy.read_text(encoding="utf-8")
    assert "__all__" in text, "the policy does not say what counts as public"
    assert "DeprecationWarning" in text, "the policy does not name the mechanism"
    assert "major" in text.lower() and "minor" in text.lower(), (
        "the policy does not state how long a deprecated name survives"
    )
    # Substring presence is a weak property -- measured, removing the bold
    # definition still passed because `__all__` appears elsewhere in the prose.
    # Tie the document to the RECORD instead: every deprecation must appear with
    # the exact versions the mechanism will announce, so the two cannot drift.
    for name, record in DEPRECATIONS.items():
        assert name in text, f"{name} is deprecated in code but absent from the policy"
        row = next(
            (line for line in text.splitlines() if f"`{name}`" in line and "|" in line), None
        )
        assert row is not None, f"{name} has no row in the policy's deprecation table"
        assert record.since in row, (
            f"the policy's row for {name} does not carry since={record.since}"
        )
        assert record.removed_in in row, (
            f"the policy's row for {name} does not carry removed_in={record.removed_in}"
        )
        assert record.instead in row, (
            f"the policy's row for {name} does not name the replacement {record.instead}"
        )
    # And the survival rule itself, stated as the author decided it.
    assert re.search(r"at least one[* ]+minor", text, re.I), (
        "the policy does not state the one-minor survival rule"
    )
    assert "promise, not an enforcement" in text.lower() or "not an enforcement" in text.lower(), (
        "the policy does not say a removal version is a promise rather than enforced"
    )


def test_the_mechanism_parses_under_the_claimed_python_floor() -> None:
    # The package claims requires-python >=3.8 and check_public_surface.py
    # already enforces that claim elsewhere.
    source = Path("src/yowo/_deprecation.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 8))


def test_the_mechanism_marks_a_name_and_not_a_parameter_or_a_behaviour() -> None:
    """covers A15 -- the scope this mechanism does NOT cover, stated by name.

    A `Deprecation` describes one public NAME. It cannot express "this
    parameter is going away", "this return type is widening", or "this
    behaviour changes in 3.0" -- those need different machinery, and the box
    asked for one warning that fires, not for all of it.

    This has a live consequence rather than a hypothetical one:
    `engine.py` records a public type change -- widening `PRECISION_UNKNOWN`
    from `str` to `str | None` -- explicitly deferred to "the deprecation
    policy". That policy now exists and STILL does not cover it, so the
    deferral stands. Better said here than discovered by someone who assumes
    this node unblocked it.
    """
    fields = {f.name for f in dataclasses.fields(Deprecation)}
    assert fields == {"name", "since", "removed_in", "instead"}, (
        "Deprecation grew a field; if it now describes more than a name, "
        "A15 and the stability policy both need revisiting"
    )

    engine = Path("src/yowo/engine.py").read_text(encoding="utf-8")
    assert "deferred to the deprecation policy" in engine, (
        "engine.py no longer defers that type change -- if it was resolved, "
        "this check and A15 should say so rather than silently pass"
    )
    assert "PRECISION_UNKNOWN" not in DEPRECATIONS, (
        "a type widening was recorded as a name deprecation; the mechanism "
        "cannot express it and the warning would mislead"
    )
