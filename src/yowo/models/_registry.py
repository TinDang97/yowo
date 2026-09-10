"""Model family registry.

Maps (ModelFamily, ModelSize) -> ModelMeta with default weight URLs.
Register new model variants with register() without changing other code.
"""

from __future__ import annotations

from dataclasses import dataclass

from yowo.errors import ModelNotFoundError
from yowo.types import ModelFamily, ModelSize

_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


@dataclass(frozen=True)
class ModelMeta:
    """Metadata for a registered model variant.

    Attributes:
        family: YOLO model family.
        size: Size variant (nano … xlarge).
        input_height: Default spatial input height (pixels).
        input_width: Default spatial input width (pixels).
        num_classes: Number of output classes (80 for COCO).
        weight_stem: Name stem for weight files (e.g. ``"yolo11n"``).
        default_weights_url: HTTPS URL to the canonical ``.pt`` weights file.
        sha256: Pinned digest of the canonical file, verified on first download
            AND on every cache hit. ``None`` means unpinned — permitted for
            user-registered models, which load with a warning rather than a
            refusal, since a private bucket has no digest we could know.
    """

    family: ModelFamily
    size: ModelSize
    input_height: int
    input_width: int
    num_classes: int
    weight_stem: str
    default_weights_url: str
    sha256: str | None = None


_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}
_CLS_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}
_OBB_REGISTRY: dict[tuple[ModelFamily, ModelSize], ModelMeta] = {}

#: The task string a `ModelSpec` carries -> the registry that answers for it.
#: Holds the live dicts, not copies, so a runtime `register()` is visible here.
#: This is the ONLY place a task name is mapped to a registry: a second mapping
#: elsewhere is how a URL and a pin come to be read from different entries.
_TASK_REGISTRIES: dict[str, dict[tuple[ModelFamily, ModelSize], ModelMeta]] = {
    "detect": _REGISTRY,
    "classify": _CLS_REGISTRY,
    "obb": _OBB_REGISTRY,
}

#: What an absent or ``None`` task means -- the same default ``ModelSpec.task``
#: carries, so a task-blind caller keeps resolving detection weights (A4).
DEFAULT_TASK = "detect"


def register(meta: ModelMeta) -> None:
    """Add *meta* to the registry.

    Overwrites any existing entry for the same ``(family, size)`` key.
    """
    _REGISTRY[(meta.family, meta.size)] = meta


def get(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the ModelMeta for the given family/size combination.

    Raises:
        ModelNotFoundError: When the combination is not registered.
    """
    key = (family, size)
    if key not in _REGISTRY:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(_REGISTRY))
        raise ModelNotFoundError(
            f"Model {family.value}/{size.value} not found in registry. Available: {available}"
        )
    return _REGISTRY[key]


def get_cls(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the classification ModelMeta for the given family/size combination.

    Raises:
        ModelNotFoundError: When the combination is not registered.
    """
    key = (family, size)
    if key not in _CLS_REGISTRY:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(_CLS_REGISTRY))
        raise ModelNotFoundError(
            f"Classification model {family.value}/{size.value} not found in registry. "
            f"Available: {available}"
        )
    return _CLS_REGISTRY[key]


def get_obb(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the OBB ModelMeta for the given family/size combination.

    Raises:
        ModelNotFoundError: When the combination is not registered.
    """
    key = (family, size)
    if key not in _OBB_REGISTRY:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(_OBB_REGISTRY))
        raise ModelNotFoundError(
            f"OBB model {family.value}/{size.value} not found in registry. Available: {available}"
        )
    return _OBB_REGISTRY[key]


def registered_tasks() -> tuple[str, ...]:
    """The task names that select a registry, in a stable order."""
    return tuple(sorted(_TASK_REGISTRIES))


def get_for_task(task: str | None, family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Return the ModelMeta for *task*'s registry, family and size.

    The task picks the registry; family and size pick the entry within it. One
    lookup yields both the download URL and the pinned digest, so the two can
    never come from different entries (A3).

    ``None`` -- and, at the caller, an absent task -- means ``detect``, matching
    ``ModelSpec.task``'s own default, so every task-blind caller keeps resolving
    what it resolved before (A1, A4).

    An unrecognised task string is an error, never a fallback to detection:
    that fallback is precisely the defect this exists to close, and it surfaces
    two hundred lines later as a shape mismatch in a conv layer instead of here
    (M6, A6).

    Raises:
        ModelNotFoundError: When ``task`` is not a registered task, or when the
            family/size combination is not registered for that task.
    """
    resolved = DEFAULT_TASK if task is None else task
    registry = _TASK_REGISTRIES.get(resolved)
    if registry is None:
        raise ModelNotFoundError(
            f"Unknown task {resolved!r} for model {family.value}{size.value}. "
            f"Registered tasks: {', '.join(registered_tasks())}. "
            "No fallback to the detection weight is attempted -- handing a "
            f"{resolved!r} model the detection checkpoint is what this check exists "
            "to prevent."
        )
    key = (family, size)
    if key not in registry:
        available = ", ".join(f"{f.value}/{s.value}" for f, s in sorted(registry))
        raise ModelNotFoundError(
            f"No {resolved} model {family.value}/{size.value} in the registry. "
            f"Available for {resolved}: {available}"
        )
    return registry[key]


def list_available() -> list[ModelMeta]:
    """Return all registered DETECTION ModelMeta entries sorted by (family, size).

    Detection only. Use :func:`list_all_registered` to audit every registry --
    walking this one and calling the result complete is how fifteen ``-cls`` and
    ``-obb`` entries stayed unpinned behind a green check.
    """
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def list_all_registered() -> list[tuple[str, ModelMeta]]:
    """Every registered entry in EVERY registry, as ``(task, meta)`` pairs.

    Sorted by task, then family, then size. This is the enumeration an audit of
    the registries must use: `list_available()` returns the ten detection metas
    and cannot see the classification or OBB entries at all.
    """
    return [
        (task, _TASK_REGISTRIES[task][key])
        for task in registered_tasks()
        for key in sorted(_TASK_REGISTRIES[task])
    ]


# ---------------------------------------------------------------------------
# Built-in registrations -- 10 variants: YOLO11/26 x n/s/m/l/x
# ---------------------------------------------------------------------------

_ASSETS_V83 = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
_ASSETS_V84 = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


# Pinned SHA-256 of each canonical release asset, keyed by weight_stem so a
# detection, a `-cls` and an `-obb` asset for the same family and size are three
# separate keys and cannot borrow each other's digest.
#
# Every value is measured by downloading that entry's own `default_weights_url`
# and hashing the bytes it served — `scripts/measure_weight_digests.py`, whose
# output is recorded in `scripts/weight_digests.json` with the URL, the byte
# count and the date. A digest from any other source (another variant, a local
# file of unknown origin, a third party) is R:UNMEASURED and is not a pin.
#
# yolo11n's value was additionally cross-checked against a real cache entry and
# matched byte for byte, confirming a pin validates a cache hit and not merely a
# fresh download. Upstream republishing a tag with new bytes must FAIL loudly
# here and be re-pinned deliberately — auto-accepting new bytes under an old pin
# would make the pin decorative.
_PINS: dict[str, str] = {
    "yolo11l": "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f",
    "yolo11l-cls": "6b56513a5d8bdae6b8f0a36dacaf01b26d5a522ba1b34197c3bac9fa6463366c",
    "yolo11l-obb": "92dcf9face59a821cd4ee93828f4c19b51f6dee9b842b23c1dacab7aa89039fc",
    "yolo11m": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    "yolo11m-cls": "80583b974e44292d56fc5ba4d26cd0bb04e77edc8bbf4317f7ce678147156dd5",
    "yolo11m-obb": "41832a4349c08190335bbc11a8e64726750702eb49cf09abb262bc394a13498c",
    "yolo11n": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11n-cls": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "yolo11n-obb": "b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32",
    "yolo11s": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "yolo11s-cls": "e2b605d1c8c212b434a75a32759a6f7adf1d2b29c35f76bdccd4c794cb653cf2",
    "yolo11s-obb": "43fa63102922e0701501241b307420d24fc55e080816888b18bf8c6f96b1a45a",
    "yolo11x": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
    "yolo11x-cls": "5b99af0f075efcd88349746fad4ff0a88cd591712b578a887286790544bd449c",
    "yolo11x-obb": "1461342db1ce0f35c755278303febd1174604ccc375110076fa0ac155231b6a0",
    "yolo26l": "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
    "yolo26l-cls": "7a3aebb7de6f5d79f0153da627a5f68bd3d7df30e648c0fdb7f9217986ee129e",
    "yolo26m": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "yolo26m-cls": "9f6546f33a70d910e2cd6dcca5265c4617b5670b19a1f287c39e99258bade01a",
    "yolo26n": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "yolo26n-cls": "0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81",
    "yolo26s": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
    "yolo26s-cls": "816790029d5df3fef358f03c8144b96339d8824ee25577aeda8be0963e5c5f09",
    "yolo26x": "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
    "yolo26x-cls": "ee88a0c71e9596cdfdbb71892725929cb4489b874d24fb581d20bffe79724386",
}


def _make_meta(family: ModelFamily, size: ModelSize) -> ModelMeta:
    name = f"{family.value}{size.value}"
    # YOLO26 assets live in the v8.4.0 release; YOLO11's in v8.3.0. This used a
    # single base for both, so every YOLO26 detection weight 404'd — half the
    # advertised models were undownloadable. _make_cls_meta always chose
    # correctly; only this path was wrong.
    base = _ASSETS_V83 if family == ModelFamily.YOLO11 else _ASSETS_V84
    return ModelMeta(
        family=family,
        size=size,
        input_height=640,
        input_width=640,
        num_classes=80,
        weight_stem=name,
        default_weights_url=f"{base}{name}.pt",
        sha256=_PINS.get(name),
    )


def _make_obb_meta(size: ModelSize) -> ModelMeta:
    """Build ModelMeta for a YOLO11 OBB variant.

    All OBB variants use ultralytics assets v8.3.0 and DOTA v1 (15 classes).
    Input is 640x640 (same as detection).
    """
    name = f"yolo11{size.value}-obb"
    return ModelMeta(
        family=ModelFamily.YOLO11,
        size=size,
        input_height=640,
        input_width=640,
        num_classes=15,  # DOTA v1 — CRITICAL: not 80
        weight_stem=name,
        default_weights_url=f"{_ASSETS_V83}{name}.pt",
        sha256=_PINS.get(name),
    )


def _make_cls_meta(family: ModelFamily, size: ModelSize) -> ModelMeta:
    """Build ModelMeta for a classification variant.

    YOLO11-cls weights use ultralytics assets v8.3.0.
    YOLO26-cls weights use ultralytics assets v8.4.0.
    Input is 224x224 (ImageNet convention); 1000 classes.
    """
    name = f"{family.value}{size.value}-cls"
    base = _ASSETS_V83 if family == ModelFamily.YOLO11 else _ASSETS_V84
    url = f"{base}{name}.pt"
    return ModelMeta(
        family=family,
        size=size,
        input_height=224,
        input_width=224,
        num_classes=1000,
        weight_stem=name,
        default_weights_url=url,
        sha256=_PINS.get(name),
    )


def _register_builtins() -> None:
    for family in (ModelFamily.YOLO11, ModelFamily.YOLO26):
        for size in (
            ModelSize.NANO,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.XLARGE,
        ):
            register(_make_meta(family, size))
            _CLS_REGISTRY[(family, size)] = _make_cls_meta(family, size)
    # OBB variants: YOLO11 only (5 sizes), DOTA v1, nc=15
    for size in (
        ModelSize.NANO,
        ModelSize.SMALL,
        ModelSize.MEDIUM,
        ModelSize.LARGE,
        ModelSize.XLARGE,
    ):
        _OBB_REGISTRY[(ModelFamily.YOLO11, size)] = _make_obb_meta(size)


_register_builtins()

__all__ = [
    "DEFAULT_TASK",
    "_CLS_REGISTRY",
    "_OBB_REGISTRY",
    "ModelMeta",
    "get",
    "get_cls",
    "get_for_task",
    "get_obb",
    "list_all_registered",
    "list_available",
    "register",
    "registered_tasks",
]
