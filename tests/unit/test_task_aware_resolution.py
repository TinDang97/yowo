"""`resolve_weights` returns the asset the spec's TASK names, and every
registered model is pinned to a digest measured from its own URL.

Red-first for ADD task `task-aware-weight-resolution`.

Today `resolve_weights` calls `get(spec.family, spec.size)` regardless of
`spec.task`, so the DETECTION registry answers every question. A `classify`
spec and an `obb` spec both resolve `yolo11n.pt`. Reproduced 2026-09-10:

    load_classify_weights(model, yolo11n.pt)
      -> RuntimeError: Shape mismatches ... backbone.c2psa.cv1.conv.weight
    load_obb_weights(model, yolo11n.pt)
      -> WARNING 42 keys missing from checkpoint (model may produce incorrect
         results) -> RuntimeError: Shape mismatches

The OBB branch is the alarming one: it WARNS about 42 missing keys and only a
shape mismatch stops it. Had the shapes lined up it would have loaded a
silently wrong model behind a warning.

Every assertion below goes through the real `resolve_weights` entry point with
a real `ModelSpec`. Network I/O is the only thing stubbed, and it is stubbed at
`_download` / `_attempt_download` so the resolution decision under test --
which registry, which URL, which pin, which cache path -- is the production
one.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

from yowo.errors import ModelNotFoundError
from yowo.models import _registry as registry_mod
from yowo.models import _weights as weights_mod
from yowo.models._registry import ModelMeta, get, get_cls, get_obb, list_available
from yowo.types import ModelFamily, ModelSize, ModelSpec

REPO_ROOT = Path(__file__).parent.parent.parent

# Where `scripts/measure_weight_digests.py --out` records what it downloaded.
# A pin is only worth its measurement (R:UNMEASURED), so the digest in the
# registry has to be traceable to a download of that exact URL.
MEASUREMENTS = REPO_ROOT / "scripts" / "weight_digests.json"

REGENERATE = "python3 scripts/measure_weight_digests.py --out scripts/weight_digests.json"

Y11 = ModelFamily.YOLO11
Y26 = ModelFamily.YOLO26
NANO = ModelSize.NANO


def _record_downloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `_download` with a recorder that writes URL-derived stand-in bytes.

    The bytes differ per URL so a later assertion can tell "two files" from
    "one file written twice".
    """
    calls: list[dict[str, Any]] = []

    def fake_download(url: str, dest: Path, expected_sha256: str | None = None) -> None:
        calls.append({"url": url, "dest": dest, "sha256": expected_sha256})
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"stand-in bytes for {url}".encode())

    monkeypatch.setattr(weights_mod, "_download", fake_download)
    return calls


def _all_registered() -> list[tuple[str, ModelMeta]]:
    """Every entry in EVERY registry, as (task, meta).

    Deliberately not `list_available()`: that returns the ten detection metas
    and is blind to the fifteen this task exists to reach (M3).
    """
    return registry_mod.list_all_registered()


def test_a_classify_spec_resolves_the_cls_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M1, A3, E1 — the reproduction.

    A `task="classify"` spec must resolve `yolo11n-cls.pt`, from the `-cls`
    URL, carrying the `-cls` entry's OWN pin. One dispatch picks all three, so
    the URL and the pin cannot come from different entries.
    """
    calls = _record_downloads(monkeypatch)
    cls_meta = get_cls(Y11, NANO)
    det_meta = get(Y11, NANO)

    path = weights_mod.resolve_weights(
        ModelSpec(family=Y11, size=NANO, task="classify"), cache_dir=tmp_path
    )

    assert path.name == "yolo11n-cls.pt", (
        f"a classify spec resolved {path.name!r}; the detection registry answered a "
        "classification question"
    )
    assert len(calls) == 1, f"expected exactly one download, got {calls}"
    assert calls[0]["url"] == cls_meta.default_weights_url
    assert calls[0]["url"] != det_meta.default_weights_url
    assert calls[0]["sha256"] == cls_meta.sha256, (
        "the pin handed to the download is not the -cls entry's own; the URL and the "
        "pin came from different entries"
    )
    assert path.read_bytes() == f"stand-in bytes for {cls_meta.default_weights_url}".encode()


def test_an_obb_spec_resolves_the_obb_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M1, E2 — the branch that warns about 42 missing keys first.

    A silently-wrong OBB model behind a warning is worse than a hard failure,
    so this is asserted on the same terms as the classify branch.
    """
    calls = _record_downloads(monkeypatch)
    obb_meta = get_obb(Y11, NANO)
    det_meta = get(Y11, NANO)

    path = weights_mod.resolve_weights(
        ModelSpec(family=Y11, size=NANO, task="obb"), cache_dir=tmp_path
    )

    assert path.name == "yolo11n-obb.pt", (
        f"an obb spec resolved {path.name!r}; that checkpoint is missing 42 of the OBB model's keys"
    )
    assert len(calls) == 1, f"expected exactly one download, got {calls}"
    assert calls[0]["url"] == obb_meta.default_weights_url
    assert calls[0]["url"] != det_meta.default_weights_url
    assert calls[0]["sha256"] == obb_meta.sha256, (
        "the pin handed to the download is not the -obb entry's own"
    )


def test_a_detect_spec_and_a_taskless_spec_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: A1, A4, E3 — every existing caller keeps resolving what it did.

    Three spellings of "detect": explicit, defaulted (`ModelSpec.task` defaults
    to `"detect"`), and an explicit `None`. A4 makes absent mean `detect`, so a
    task-blind caller -- `benchmark/_runner.py:74` among them -- is unaffected.
    """
    det_meta = get(Y11, NANO)

    for label, spec in (
        ("explicit detect", ModelSpec(family=Y11, size=NANO, task="detect")),
        ("defaulted task", ModelSpec(family=Y11, size=NANO)),
        ("explicit None", ModelSpec(family=Y11, size=NANO, task=cast(str, None))),
    ):
        calls = _record_downloads(monkeypatch)
        cache = tmp_path / label.replace(" ", "-")
        path = weights_mod.resolve_weights(spec, cache_dir=cache)

        assert path.name == "yolo11n.pt", f"{label}: resolved {path.name!r}"
        assert len(calls) == 1, f"{label}: expected one download, got {calls}"
        assert calls[0]["url"] == det_meta.default_weights_url, label
        assert calls[0]["sha256"] == det_meta.sha256, label


def test_no_branch_returns_a_weight_for_another_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: R:WRONGASSET — swept over every registered task, not just the two
    that are broken today.

    For every (task, family, size) actually registered: the resolved file, the
    URL fetched and the pin enforced all belong to that task's entry, and none
    of them belongs to another task's entry for the same family and size.
    """
    calls = _record_downloads(monkeypatch)
    entries = _all_registered()
    assert len(entries) >= 25, (
        f"only {len(entries)} entries swept; the sweep is not seeing every registry"
    )

    for task, meta in entries:
        calls.clear()
        spec = ModelSpec(family=meta.family, size=meta.size, task=task)
        path = weights_mod.resolve_weights(spec, cache_dir=tmp_path)

        assert path.name == f"{meta.weight_stem}.pt", (
            f"task={task} {meta.family.value}/{meta.size.value} resolved {path.name!r}, "
            f"expected {meta.weight_stem}.pt"
        )
        assert len(calls) == 1, f"task={task} {meta.weight_stem}: downloads {calls}"
        assert calls[0]["url"] == meta.default_weights_url, f"task={task} {meta.weight_stem}"
        assert calls[0]["sha256"] == meta.sha256, f"task={task} {meta.weight_stem}"

        for other_task, other_registry in registry_mod._TASK_REGISTRIES.items():
            if other_task == task:
                continue
            other = other_registry.get((meta.family, meta.size))
            if other is None:
                continue
            assert path.name != f"{other.weight_stem}.pt", (
                f"task={task} resolved the {other_task} asset {other.weight_stem}"
            )
            assert calls[0]["url"] != other.default_weights_url, (
                f"task={task} fetched the {other_task} URL"
            )


def test_every_registry_entry_carries_a_pinned_digest() -> None:
    """covers: M2, A2 — all 25 entries across all three registries.

    EXPECTED RED until the measured pins land. The fifteen `-cls` / `-obb`
    entries carry `sha256=None`: those URLs were never fetched, so no digest
    for them may be invented, copied from another variant, or transcribed
    (R:UNMEASURED). They are measured by `scripts/measure_weight_digests.py`
    and pinned from what it downloaded.
    """
    entries = _all_registered()
    assert {task for task, _ in entries} == {"detect", "classify", "obb"}, (
        "the sweep does not reach every registry; see M3"
    )
    assert len(entries) >= 25, f"only {len(entries)} registered entries seen"

    unpinned = [f"{task}/{meta.weight_stem}" for task, meta in entries if not meta.sha256]
    assert unpinned == [], (
        f"registered models with no pinned digest: {unpinned}. Measure them with `{REGENERATE}`"
    )


def test_the_pinned_digest_check_cannot_pass_by_walking_one_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """covers: M3 — the check above is proved to SEE every registry.

    The old `test_every_builtin_carries_a_pinned_digest` walks `list_available()`,
    which returns the ten detection metas. It therefore reads as complete while
    covering ten of twenty-five, and stays green with fifteen URLs unpinned.
    This injects an unpinned entry into the two registries that enumeration
    cannot see and proves the M2 check fails on both.
    """
    sentinel_cls = ModelMeta(
        family=Y11,
        size=NANO,
        input_height=224,
        input_width=224,
        num_classes=1000,
        weight_stem="sentinel-cls",
        default_weights_url="https://example.invalid/sentinel-cls.pt",
        sha256=None,
    )
    sentinel_obb = ModelMeta(
        family=Y26,
        size=NANO,
        input_height=640,
        input_width=640,
        num_classes=15,
        weight_stem="sentinel-obb",
        default_weights_url="https://example.invalid/sentinel-obb.pt",
        sha256=None,
    )
    monkeypatch.setitem(registry_mod._CLS_REGISTRY, (Y11, NANO), sentinel_cls)
    monkeypatch.setitem(registry_mod._OBB_REGISTRY, (Y26, NANO), sentinel_obb)

    seen = {meta.weight_stem for _, meta in _all_registered()}
    assert "sentinel-cls" in seen, "the M2 sweep is blind to the classification registry"
    assert "sentinel-obb" in seen, "the M2 sweep is blind to the OBB registry"

    blind = {meta.weight_stem for meta in list_available()}
    assert "sentinel-cls" not in blind and "sentinel-obb" not in blind, (
        "list_available() unexpectedly reaches the other registries; this test's "
        "premise -- and M3's -- would need re-deriving"
    )
    assert len(blind) < len(seen), (
        f"list_available() sees {len(blind)} of {len(seen)} registered entries"
    )

    with pytest.raises(AssertionError) as excinfo:
        test_every_registry_entry_carries_a_pinned_digest()
    message = str(excinfo.value)
    assert "sentinel-cls" in message, "the M2 check did not fail on an unpinned -cls entry"
    assert "sentinel-obb" in message, "the M2 check did not fail on an unpinned -obb entry"


def test_every_recorded_digest_names_its_measurement() -> None:
    """covers: R:UNMEASURED — each pin traceable to a download of THAT URL.

    EXPECTED RED until the measurement record lands. A digest with no record of
    where it came from is indistinguishable from one copied off another variant
    or transcribed from a third party, which is the thing R:UNMEASURED rejects.
    Each record carries the URL that served the bytes, the byte count and the
    date, and its digest must be the one the registry pins.
    """
    assert MEASUREMENTS.exists(), (
        f"no measurement record at {MEASUREMENTS.relative_to(REPO_ROOT)}; "
        f"regenerate with `{REGENERATE}`"
    )
    records = json.loads(MEASUREMENTS.read_text())
    assert isinstance(records, list) and records, "the measurement record is empty"

    by_key = {(str(r["task"]), str(r["weight_stem"])): r for r in records}
    missing: list[str] = []
    for task, meta in _all_registered():
        record = by_key.get((task, meta.weight_stem))
        if record is None:
            missing.append(f"{task}/{meta.weight_stem}")
            continue
        assert record["url"] == meta.default_weights_url, (
            f"{task}/{meta.weight_stem}: measured {record['url']} but the registry "
            f"resolves {meta.default_weights_url}"
        )
        assert record["sha256"] == meta.sha256, (
            f"{task}/{meta.weight_stem}: the registry pin is not the digest that was "
            "measured from this URL"
        )
        assert isinstance(record["bytes"], int) and record["bytes"] > 0, (
            f"{task}/{meta.weight_stem}: no byte count, so nothing was actually read"
        )
        date.fromisoformat(str(record["measured"]))
    assert missing == [], f"pins with no recorded measurement: {missing}. Run `{REGENERATE}`"

    # A digest shared by two different URLs is a digest copied across variants.
    by_digest: dict[str, set[str]] = {}
    for record in records:
        by_digest.setdefault(str(record["sha256"]), set()).add(str(record["url"]))
    shared = {digest: urls for digest, urls in by_digest.items() if len(urls) > 1}
    assert shared == {}, f"one digest recorded for several URLs: {shared}"


def test_a_detection_and_a_classify_weight_do_not_share_a_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M4, R:COLLIDE, E4 — one family and size, three tasks, three files.

    Proved on disk rather than assumed from `weight_stem`: all three are
    resolved into one cache root and the directory is then listed. The final
    sweep extends it to every registered entry, so no pair anywhere collides.
    """
    _record_downloads(monkeypatch)

    det = weights_mod.resolve_weights(
        ModelSpec(family=Y11, size=NANO, task="detect"), cache_dir=tmp_path
    )
    cls = weights_mod.resolve_weights(
        ModelSpec(family=Y11, size=NANO, task="classify"), cache_dir=tmp_path
    )
    obb = weights_mod.resolve_weights(
        ModelSpec(family=Y11, size=NANO, task="obb"), cache_dir=tmp_path
    )

    assert len({det, cls, obb}) == 3, (
        f"tasks share a cache path: detect={det}, classify={cls}, obb={obb}"
    )
    assert det.exists() and cls.exists() and obb.exists()
    assert sorted(p.name for p in tmp_path.rglob("*.pt")) == [
        "yolo11n-cls.pt",
        "yolo11n-obb.pt",
        "yolo11n.pt",
    ], "one weight overwrote another on disk"
    assert len({det.read_bytes(), cls.read_bytes(), obb.read_bytes()}) == 3, (
        "the three cache entries hold the same bytes"
    )

    # No two registered entries anywhere resolve to the same cache path. A
    # separate root so the stand-in bytes written above are not re-verified
    # against the real detection pin, which is this fixture's artefact and not
    # the property under test.
    sweep_root = tmp_path / "sweep"
    paths: dict[Path, str] = {}
    for task, meta in _all_registered():
        spec = ModelSpec(family=meta.family, size=meta.size, task=task)
        path = weights_mod.resolve_weights(spec, cache_dir=sweep_root)
        owner = f"{task}/{meta.weight_stem}"
        assert path not in paths, f"{owner} and {paths[path]} share the cache path {path}"
        paths[path] = owner


def test_an_unrecognised_task_raises_naming_what_is_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M6, A4, A6, E5 — a typo'd task fails AT RESOLUTION.

    Not 200 lines later as a shape error in a conv layer. The message names the
    task that was asked for, the model, and the tasks that are registered. It
    never falls back to detection -- that fallback IS this defect.
    """
    calls = _record_downloads(monkeypatch)

    with pytest.raises(ModelNotFoundError) as excinfo:
        weights_mod.resolve_weights(
            ModelSpec(family=Y11, size=NANO, task="clasify"), cache_dir=tmp_path
        )

    message = str(excinfo.value)
    assert "clasify" in message, f"the message does not name the task asked for: {message}"
    assert "yolo11n" in message, f"the message does not name the model: {message}"
    for task in ("detect", "classify", "obb"):
        assert task in message, f"the message does not name registered task {task!r}: {message}"

    assert calls == [], "an unrecognised task fell back to a download"
    assert list(tmp_path.rglob("*.pt")) == [], "an unrecognised task wrote a weight to the cache"


def test_resolve_weights_signature_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: M5 — the ~60 mock sites and three real callers stay valid.

    `engine.py:447`, `export/_exporter.py:94` and `benchmark/_runner.py:74` all
    call it as `resolve_weights(spec)`; the tests that mock it substitute a
    one- or two-argument callable. Both the introspected signature and the
    three real call shapes are asserted.
    """
    signature = inspect.signature(weights_mod.resolve_weights)
    assert list(signature.parameters) == ["spec", "cache_dir"]

    spec_param = signature.parameters["spec"]
    assert spec_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert spec_param.default is inspect.Parameter.empty
    assert spec_param.annotation == "ModelSpec"

    cache_param = signature.parameters["cache_dir"]
    assert cache_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert cache_param.default is None
    assert cache_param.annotation == "Path | None"

    assert signature.return_annotation == "Path"

    _record_downloads(monkeypatch)
    spec = ModelSpec(family=Y11, size=NANO)
    assert isinstance(weights_mod.resolve_weights(spec, tmp_path / "a"), Path)
    assert isinstance(weights_mod.resolve_weights(spec, cache_dir=tmp_path / "b"), Path)
    assert isinstance(weights_mod.resolve_weights(spec=spec, cache_dir=tmp_path / "c"), Path)

    monkeypatch.setenv("YOWO_CACHE_DIR", str(tmp_path / "d"))
    assert isinstance(weights_mod.resolve_weights(spec), Path)


def test_an_explicit_weights_path_wins_over_any_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: A5, E6 — a user's own checkpoint, returned unpinned.

    `weights_path` short-circuits BEFORE any registry lookup, for every task
    including an unrecognised one. A fine-tuned checkpoint is a file no
    registry entry describes; comparing it to an official digest would refuse
    every custom run.
    """
    weight = tmp_path / "my-finetune.pt"
    weight.write_bytes(b"a user's own checkpoint")
    cache = tmp_path / "cache"

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an explicit weights_path consulted the registry or the network")

    for module, name in (
        (registry_mod, "get"),
        (registry_mod, "get_cls"),
        (registry_mod, "get_obb"),
        (registry_mod, "get_for_task"),
        (weights_mod, "get"),
        (weights_mod, "get_for_task"),
        (weights_mod, "_download"),
        (weights_mod, "verify_digest"),
        (weights_mod, "file_digest"),
    ):
        monkeypatch.setattr(module, name, boom, raising=False)

    for task in ("detect", "classify", "obb", "not-a-task", cast(str, None)):
        spec = ModelSpec(family=Y11, size=NANO, task=task, weights_path=weight)
        assert weights_mod.resolve_weights(spec, cache_dir=cache) == weight, (
            f"task={task!r} did not return the user's own checkpoint"
        )

    assert not cache.exists(), "an explicit weights_path touched the cache"


def test_a_pinned_cls_weight_that_fails_verification_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """covers: E7 — the new entries inherit the existing integrity control.

    A `-cls` entry that carries a pin is verified exactly as a detection weight
    is: the bad bytes never reach the cache, and the pin that is enforced is
    the `-cls` entry's own. Built with a patched registry entry carrying a
    known digest -- the real `-cls` pins are measured downloads and must not be
    invented here (R:UNMEASURED).
    """
    served = b"these are not the weights you are looking for"
    pinned = hashlib.sha256(b"what the -cls URL is supposed to serve").hexdigest()
    served_digest = hashlib.sha256(served).hexdigest()
    assert pinned != served_digest

    real = get_cls(Y11, NANO)
    monkeypatch.setitem(registry_mod._CLS_REGISTRY, (Y11, NANO), replace(real, sha256=pinned))

    def fake_attempt(url: str, tmp_target: Path, *, suppress_progress: bool) -> None:
        tmp_target.write_bytes(served)

    monkeypatch.setattr(weights_mod, "_attempt_download", fake_attempt)

    with pytest.raises(weights_mod.WeightIntegrityError) as excinfo:
        weights_mod.resolve_weights(
            ModelSpec(family=Y11, size=NANO, task="classify"), cache_dir=tmp_path
        )

    message = str(excinfo.value)
    assert pinned in message, (
        "the digest enforced is not the -cls entry's own pin; a classification spec was "
        f"verified against something else: {message}"
    )
    assert served_digest in message, (
        f"the message does not report what was actually served: {message}"
    )
    assert "yolo11n-cls" in message, f"the message does not name the -cls asset: {message}"

    assert list(tmp_path.rglob("*.pt")) == [], "the unverified file was left in the cache"
    assert list(tmp_path.rglob("*.tmp")) == [], "a partial download was left behind"
