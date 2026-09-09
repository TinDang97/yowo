"""Load ultralytics ``.pt`` checkpoints into native YOLOModel and ClassifyModel.

Maps ultralytics sequential layer indices (``model.0``, ``model.1``, …)
to our semantic module paths (``backbone.stem``, ``neck.c3k2_fpn1``, …).

Usage::

    from yowo.arch import build_model
    from yowo.arch._weights import load_weights

    model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
    load_weights(model, "yolo11n.pt")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from yowo.models._weights import verify_digest

if TYPE_CHECKING:
    import torch as _torch

    from yowo.arch._obb import OBBModel
    from yowo.arch._yolo import ClassifyModel, YOLOModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer index → semantic path mapping
# ---------------------------------------------------------------------------

# Ultralytics DetectionModel stores layers in ``self.model = nn.Sequential(...)``.
# State-dict keys are ``model.{layer_idx}.{submodule}.{param}``.
# Layers 11, 12, 14, 15, 18, 21 are parameterless (Upsample / Concat).

_LAYER_MAP: dict[str, str] = {
    # Backbone (layers 0-10)
    "model.0.": "backbone.stem.",
    "model.1.": "backbone.conv1.",
    "model.2.": "backbone.c3k2_1.",
    "model.3.": "backbone.conv2.",
    "model.4.": "backbone.c3k2_2.",
    "model.5.": "backbone.conv3.",
    "model.6.": "backbone.c3k2_3.",
    "model.7.": "backbone.conv4.",
    "model.8.": "backbone.c3k2_4.",
    "model.9.": "backbone.sppf.",
    "model.10.": "backbone.c2psa.",
    # Neck (layers 13, 16, 17, 19, 20, 22 — parameterised only)
    "model.13.": "neck.c3k2_fpn1.",
    "model.16.": "neck.c3k2_fpn2.",
    "model.17.": "neck.down1.",
    "model.19.": "neck.c3k2_pan1.",
    "model.20.": "neck.down2.",
    "model.22.": "neck.c3k2_pan2.",
    # Detection head (layer 23)
    "model.23.": "head.",
}

# Sorted by longest prefix first for correct matching
_SORTED_PREFIXES = sorted(_LAYER_MAP.keys(), key=len, reverse=True)


def _remap_key(key: str) -> str | None:
    """Remap a single ultralytics state-dict key to our naming.

    Returns None if the key belongs to a parameterless layer (Upsample/Concat).
    """
    for prefix in _SORTED_PREFIXES:
        if key.startswith(prefix):
            return _LAYER_MAP[prefix] + key[len(prefix) :]
    return None


# ---------------------------------------------------------------------------
# Restricted checkpoint reading
# ---------------------------------------------------------------------------

# An ultralytics `.pt` stores its weights inside a pickled `nn.Module`, so reading
# one with a stock unpickler means the FILE chooses what gets imported. That had two
# consequences: `ultralytics` became an undeclared runtime dependency (a clean
# install raised `ModuleNotFoundError: No module named 'ultralytics.nn.tasks'`), and
# a checkpoint author held an import primitive.
#
# `weights_only=True` cannot fix it — it refuses the `nn.Module` the weights live
# inside. So we drive `torch.load` with our own unpickler instead: torch's tensor
# rebuild helpers and real `torch.nn` layers are constructed normally, every
# `ultralytics.*` name resolves to an inert stub so the pickle stream can be walked
# to the tensors it carries, and anything else is refused BY NAME without importing
# the module it came from.
#
# Frozen by ADD task `checkpoint-loader`; the torch layer set enumerated by
# `narrow-loader-allowlist`. Widening `_ALLOWED_*` is a change-request, not a build
# detail — each entry carries why it is here, and each torch entry carries the
# observation that put it here.

_ALLOWED_EXACT: frozenset[tuple[str, str]] = frozenset(
    {
        # Rebuilds a tensor from a storage + stride. The single global every
        # torch checkpoint needs; constructs data, not behaviour.
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_parameter"),
        # A tuple subclass carrying tensor shapes. Data, not behaviour — and the
        # one global the hermetic fixtures did not exercise, caught by the
        # equivalence floor against a real checkpoint.
        ("torch", "Size"),
        # Plain container used by older checkpoints for the state_dict itself.
        ("collections", "OrderedDict"),
    }
    # Plain data containers. `nn.Module` keeps `_non_persistent_buffers_set` and
    # friends, so a checkpoint legitimately names these. They construct data and
    # nothing else — note `eval`, `exec`, `getattr` and `__import__` are absent
    # from this list on purpose, and adding one is a change-request.
    | {
        (mod, name)
        for mod in ("builtins", "__builtin__")
        for name in ("set", "frozenset", "dict", "list", "tuple")
    }
)

# Real torch layers — the modules the weights hang off. ENUMERATED, not a
# namespace: `torch.nn.modules.` as a prefix admitted every class under it, and a
# prefix is a rule about a namespace where the boundary should name a class.
# Every pair below was OBSERVED, never reasoned into place — named either by one
# of the 10 digest-pinned detection checkpoints, or by a serialised `YOLOModel` /
# `ClassifyModel` / `OBBModel` of our own, walked in process. The observation
# behind each entry is recorded in `scripts/checkpoint_globals_manifest.json`,
# written by `scripts/measure_checkpoint_globals.py`; the unit suite holds this
# set to that manifest in both directions, so an entry cannot appear here without
# a measurement behind it.
#
# Absent on purpose, because nothing named them: `torch.nn.parameter.Parameter`
# (parameters travel as `torch._utils._rebuild_parameter`, so the old
# `torch.nn.parameter` prefix admitted a namespace no checkpoint ever used) and
# `torch.nn.modules.module.Module` itself. An unlisted torch class is refused
# exactly like any other unlisted name — a new upstream layer type fails loudly
# and is re-measured and re-pinned deliberately.
_ALLOWED_TORCH: frozenset[tuple[str, str]] = frozenset(
    {
        # Named by every detection checkpoint and every model we build: the
        # Conv → BatchNorm → SiLU block, its containers, and the `Identity` that
        # stands in for an absent activation or shortcut.
        ("torch.nn.modules.activation", "SiLU"),
        ("torch.nn.modules.batchnorm", "BatchNorm2d"),
        ("torch.nn.modules.container", "ModuleList"),
        ("torch.nn.modules.container", "Sequential"),
        ("torch.nn.modules.conv", "Conv2d"),
        ("torch.nn.modules.linear", "Identity"),
        # Named by every detection checkpoint: SPPF's pooling and the neck's
        # upsampling. Parameterless, so `.state_dict()` never sees them, but the
        # stream constructs them on the way to the tensors.
        ("torch.nn.modules.pooling", "MaxPool2d"),
        ("torch.nn.modules.upsampling", "Upsample"),
        # The classification head: Conv → AdaptiveAvgPool2d → Dropout → Linear.
        # No detection checkpoint names these; the in-process sweep of
        # `ClassifyModel` does, at every size. Without them every `-cls`
        # checkpoint would be refused at its pooling layer (E1).
        ("torch.nn.modules.dropout", "Dropout"),
        ("torch.nn.modules.linear", "Linear"),
        ("torch.nn.modules.pooling", "AdaptiveAvgPool2d"),
    }
)

# One membership test at `find_class` time. `_ALLOWED_EXACT` and `_ALLOWED_TORCH`
# are kept apart so each carries its own reasons, and joined here so the check
# stays a single set lookup — no slower than the prefix scan it replaced.
_ALLOWED: frozenset[tuple[str, str]] = _ALLOWED_EXACT | _ALLOWED_TORCH

# Typed storages live directly on `torch` (torch.FloatStorage, torch.HalfStorage, …)
# and used to be admitted here by a suffix rule — the last entry in this boundary
# that admitted by the SHAPE of a name rather than by the name. Removed by ADD task
# `storage-suffix-enumeration`: `scripts/checkpoint_globals_manifest.json` records
# 53 globals observed across the 10 digest-pinned checkpoints and the 25-model
# in-process sweep, and not one is a storage class — torch's own unpickler wrapper
# intercepts `*Storage` names before this `find_class` ever sees them for a real
# checkpoint. A storage class is now refused exactly like any other unlisted name;
# if one is ever genuinely needed, it joins `_ALLOWED_EXACT` carrying the
# observation that put it there, the same change-request path as every other entry.

# Third-party model classes. Never constructed — stubbed, see `_InertModule`.
_STUBBED_PREFIXES: tuple[str, ...] = ("ultralytics.", "models.")


def _inert_module_cls() -> type:
    """A stand-in for a checkpoint-named class, built lazily so torch stays deferred."""
    import torch

    class _InertModule(torch.nn.Module):
        """Constructed in place of a foreign class; runs none of its code.

        Subclasses ``nn.Module`` so that unpickling restores the original object's
        ``_parameters`` / ``_buffers`` / ``_modules`` into something whose inherited
        ``.float()`` and ``.state_dict()`` behave exactly as before. No method of
        the class named by the checkpoint is ever reachable.
        """

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    return _InertModule


def _restricted_unpickler_module(checkpoint_path: Path):
    """A `pickle_module` for `torch.load` whose `find_class` enforces the allowlist."""
    import io
    import pickle
    import types

    from yowo.errors import ModelLoadError

    inert = _inert_module_cls()

    class _RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> object:
            if (module, name) in _ALLOWED:
                return super().find_class(module, name)
            if module.startswith(_STUBBED_PREFIXES):
                return inert
            raise ModelLoadError(
                f"Refusing to load {checkpoint_path}: the checkpoint asks for "
                f"'{module}.{name}', which is not in the loader's allowlist of "
                f"weights primitives. A checkpoint that names arbitrary code is not "
                f"just weights — treat this file as untrusted rather than looking for "
                f"a flag to disable this check. If this is a genuine new upstream "
                f"layer type, the allowlist is widened by a change-request carrying a "
                f"recorded observation: see .add/tasks/narrow-loader-allowlist.md and "
                f"scripts/measure_checkpoint_globals.py."
            )

    # torch's legacy (non-zip) reader calls `load` on the file header three times
    # BEFORE it builds an `Unpickler` — so a stock `load` here would be an
    # unrestricted read of attacker bytes, and a non-zip file whose first object
    # is a REDUCE would run it and only then fail on "Invalid magic number".
    # Every read of the file, header included, goes through the same find_class.
    def _restricted_load(file: IO[bytes], **kwargs: Any) -> object:
        return _RestrictedUnpickler(file, **kwargs).load()

    def _restricted_loads(data: bytes, **kwargs: Any) -> object:
        return _RestrictedUnpickler(io.BytesIO(data), **kwargs).load()

    shim = types.ModuleType("yowo_restricted_pickle")
    shim.Unpickler = _RestrictedUnpickler  # type: ignore[attr-defined]
    shim.load = _restricted_load  # type: ignore[attr-defined]
    shim.loads = _restricted_loads  # type: ignore[attr-defined]
    shim.Pickler = pickle.Pickler  # type: ignore[attr-defined]
    shim.dump = pickle.dump  # type: ignore[attr-defined]
    shim.dumps = pickle.dumps  # type: ignore[attr-defined]
    return shim


def _safe_load(checkpoint_path: Path) -> object:
    """`torch.load` the checkpoint through the restricted unpickler.

    Raises:
        ModelLoadError: If the file cannot be read, or names a global that is not
            a weights primitive.
    """
    import torch

    from yowo.errors import ModelLoadError

    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_restricted_unpickler_module(checkpoint_path),
        )
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            f"Could not read checkpoint {checkpoint_path}: {type(exc).__name__}: {exc}"
        ) from exc


def _extract_state_dict(checkpoint_path: Path) -> dict[str, _torch.Tensor]:
    """Load checkpoint and extract the float32 state_dict.

    Handles ultralytics checkpoint format:
    - Prefers EMA weights (``ckpt['ema']``) over training weights.
    - Converts to float32 (ultralytics may store FP16).
    - Handles both full checkpoint dicts and raw state_dicts.
    """
    import torch

    ckpt = _safe_load(checkpoint_path)

    # Raw state_dict (unlikely but handle gracefully)
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return {k: v.float() for k, v in ckpt.items()}

    # Standard ultralytics checkpoint format
    if isinstance(ckpt, dict):
        model_obj = ckpt.get("ema") or ckpt.get("model")
        if model_obj is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} has no 'model' or 'ema' key. "
                f"Available keys: {list(ckpt.keys())}"
            )
        # model_obj is an nn.Module — extract state_dict
        if hasattr(model_obj, "state_dict"):
            model_obj = model_obj.float()
            return dict(model_obj.state_dict())
        # model_obj is already a dict
        if isinstance(model_obj, dict):
            return {k: v.float() for k, v in model_obj.items()}

    raise ValueError(
        f"Unrecognised checkpoint format at {checkpoint_path}. "
        f"Expected ultralytics .pt checkpoint or raw state_dict."
    )


def _sidecar_path(checkpoint_path: Path) -> Path:
    """Where the converted, tensor-only state_dict lives for this checkpoint."""
    return checkpoint_path.with_suffix(".state_dict.pt")


def _raw_digest(path: Path) -> str:
    """SHA-256 of the raw checkpoint, used ONLY to key its converted sidecar.

    This is never the security check, and it structurally cannot be one: a digest
    computed from the file under test matches that file by construction, so
    comparing the two proves only that SHA-256 is deterministic. The value exists
    to key the sidecar to the exact bytes it was derived from, so a re-fetched or
    re-pinned weight regenerates it instead of silently serving old tensors.

    Authentication is `verify_digest` against a value that came from the
    REGISTRY, and the two must never be allowed to meet — see
    :func:`load_verified_state_dict`.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_verified_state_dict(
    checkpoint_path: Path, raw_digest: str | None = None
) -> dict[str, _torch.Tensor]:
    """Return the checkpoint's state_dict, unpickling at most once, ever.

    m1 requires ``torch.load(weights_only=True)``, which cannot read an
    ultralytics checkpoint at all — it refuses the ``nn.Module`` the weights live
    inside. So the dangerous read happens exactly once, against bytes whose
    SHA-256 already matched the pin, and its output is re-serialised as a plain
    tensor mapping. Every later load reads that sidecar with
    ``weights_only=True`` and executes nothing.

    The sidecar stores the raw file's digest alongside the tensors and is only
    trusted when that digest matches. A re-pinned or re-fetched weight therefore
    regenerates it automatically, and a stale sidecar from an earlier version can
    never serve old weights under a new pin. A missing or mismatched sidecar
    falls back to verify-and-reconvert — never to loading it unchecked.

    ``resolve_weights`` verified this path against the pin before returning it,
    but that was a separate open at a separate moment. Anything with write access
    to the weight cache in between can substitute the file, and the substitution
    is self-concealing: the new bytes key to a different sidecar, so the sidecar
    misses and the attacker's checkpoint goes straight to the reader that can
    execute. So when a pin is supplied, it is compared again here — immediately
    before the conversion, and only on the calls that would actually convert.

    Args:
        checkpoint_path: The raw ``.pt``, already digest-verified by the caller.
        raw_digest: The digest the REGISTRY pins for this model. ``None`` means
            genuinely unpinned — an explicit ``spec.weights_path``, or a registry
            entry carrying ``sha256=None`` — in which case there is nothing to
            authenticate against and the sidecar is keyed on the file's own bytes.
    """
    import torch

    # Two values, deliberately never merged. `pin` authenticates and may only
    # ever come from the caller's registry lookup; `sidecar_key` addresses a
    # cache entry and may be computed from the file. Collapsing them into one
    # variable is exactly how this function came to "verify" a file against
    # itself, so the separation is the control, not a style choice.
    pin = raw_digest
    sidecar_key = pin if pin is not None else _raw_digest(checkpoint_path)

    sidecar = _sidecar_path(checkpoint_path)
    if sidecar.exists():
        try:
            blob = torch.load(sidecar, map_location="cpu", weights_only=True)
        except Exception:
            blob = None  # unreadable or written by an older torch — reconvert
        if isinstance(blob, dict) and blob.get("raw_sha256") == sidecar_key:
            tensors = blob.get("tensors")
            if isinstance(tensors, dict):
                # Served without touching the raw file. A sidecar keyed on the
                # pin was produced from bytes that hashed to the pin, and it
                # loads with weights_only=True, so there is nothing left to
                # authenticate — and re-hashing 5-110 MB on every steady-state
                # load would tax the exact fast path this sidecar exists to be.
                return tensors

    # Past this point the raw file WILL be read by the executing reader, so this
    # is the last moment a substitution can still be refused. Order is the whole
    # control: converting first would run the payload the comparison exists to
    # reject. Reuses verify_digest so a mismatch reads identically whether it was
    # caught during resolution or here — one event, one message.
    if pin is not None:
        verify_digest(checkpoint_path, pin)

    state = _extract_state_dict(checkpoint_path)
    logger.info(
        "Converted %s to a tensor-only state_dict; later loads skip the unpickler.",
        checkpoint_path.name,
    )
    try:
        tmp = sidecar.with_suffix(".tmp")
        torch.save({"raw_sha256": sidecar_key, "tensors": state}, tmp)
        tmp.replace(sidecar)
    except OSError:
        # A read-only or full cache directory must not break loading; the only
        # cost is that the next load converts again.
        logger.debug("Could not write %s; will reconvert next time.", sidecar)
    return state


def load_weights(model: YOLOModel, weights_path: str | Path, raw_digest: str | None = None) -> None:
    """Load ultralytics checkpoint weights into a native YOLOModel.

    Args:
        model: An initialised ``YOLOModel`` (from ``build_model()``).
        weights_path: Path to an ultralytics ``.pt`` checkpoint file.
        raw_digest: The SHA-256 the registry pins for this model, re-compared
            against the file immediately before conversion. ``None`` for an
            unpinned model — see :func:`load_verified_state_dict`.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        WeightIntegrityError: If the file no longer matches ``raw_digest``.
        ValueError: If the checkpoint format is unrecognised.
        RuntimeError: If weight shapes do not match the model architecture.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    src_state = load_verified_state_dict(path, raw_digest=raw_digest)

    # Remap keys
    mapped: dict[str, _torch.Tensor] = {}
    skipped: list[str] = []

    for src_key, tensor in src_state.items():
        dst_key = _remap_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue
        mapped[dst_key] = tensor

    if skipped:
        logger.debug(
            "Skipped %d checkpoint keys (parameterless layers): %s",
            len(skipped),
            skipped[:5],
        )

    # Load into model
    dst_state = model.state_dict()

    # Check for missing / unexpected keys
    missing = set(dst_state.keys()) - set(mapped.keys())
    unexpected = set(mapped.keys()) - set(dst_state.keys())

    if unexpected:
        logger.warning(
            "Ignoring %d unexpected keys from checkpoint: %s",
            len(unexpected),
            sorted(unexpected)[:5],
        )
        for k in unexpected:
            del mapped[k]

    if missing:
        # Separate truly missing from stride/anchor buffers (non-critical)
        critical_missing = [
            k for k in missing if "stride" not in k and "anchor" not in k and "dfl.weight" not in k
        ]
        if critical_missing:
            logger.warning(
                "%d keys missing from checkpoint (model may produce incorrect results): %s",
                len(critical_missing),
                sorted(critical_missing)[:10],
            )

    # Shape validation
    shape_mismatches: list[str] = []
    for key in list(mapped.keys()):
        if key in dst_state and mapped[key].shape != dst_state[key].shape:
            shape_mismatches.append(
                f"  {key}: checkpoint {mapped[key].shape} vs model {dst_state[key].shape}"
            )
            del mapped[key]

    if shape_mismatches:
        raise RuntimeError(
            "Shape mismatches between checkpoint and model:\n" + "\n".join(shape_mismatches)
        )

    # Detach before loading: EMA checkpoint tensors may retain autograd
    # computation graphs from training (non-leaf), causing PyTorch's
    # load_state_dict to warn when accessing .grad. Detach breaks the graph.
    model.load_state_dict({k: v.detach() for k, v in mapped.items()}, strict=False)
    logger.info(
        "Loaded %d/%d parameters from %s",
        len(mapped),
        len(dst_state),
        path.name,
    )


# ---------------------------------------------------------------------------
# Classification weight mapping
# ---------------------------------------------------------------------------

# Ultralytics cls checkpoints have 11 layers (0-10).
# Backbone layers 0-8 match detection; SPPF (detection layer 9) is absent,
# so C2PSA sits at layer 9 instead of 10.
# Layer 10 is the Classify head (Conv + pool + dropout + linear).

_CLS_LAYER_MAP: dict[str, str] = {
    # Backbone (layers 0-9) — same as detection except SPPF is omitted:
    # ultralytics cls backbone goes C3k2_4 (layer 8) → C2PSA (layer 9) directly.
    "model.0.": "backbone.stem.",
    "model.1.": "backbone.conv1.",
    "model.2.": "backbone.c3k2_1.",
    "model.3.": "backbone.conv2.",
    "model.4.": "backbone.c3k2_2.",
    "model.5.": "backbone.conv3.",
    "model.6.": "backbone.c3k2_3.",
    "model.7.": "backbone.conv4.",
    "model.8.": "backbone.c3k2_4.",
    # No SPPF in classification models (detection layer 9 is absent)
    "model.9.": "backbone.c2psa.",
    # Classification head (layer 10) — replaces neck+detect head
    "model.10.": "head.",
}

# Sorted by longest prefix first for correct matching
_CLS_SORTED_PREFIXES: tuple[str, ...] = tuple(sorted(_CLS_LAYER_MAP, key=len, reverse=True))


def _remap_cls_key(key: str) -> str | None:
    """Remap a single ultralytics cls state-dict key to our naming.

    Returns None if the key does not match any known prefix.
    """
    for prefix in _CLS_SORTED_PREFIXES:
        if key.startswith(prefix):
            return _CLS_LAYER_MAP[prefix] + key[len(prefix) :]
    return None


def load_classify_weights(
    model: ClassifyModel, weights_path: str | Path, raw_digest: str | None = None
) -> None:
    """Load an ultralytics classification checkpoint into a ClassifyModel.

    Args:
        model: An initialised ``ClassifyModel`` (from ``build_classify_model()``).
        weights_path: Path to an ultralytics ``-cls.pt`` checkpoint file.
        raw_digest: The SHA-256 the registry pins for this model. Every ``-cls``
            entry carries ``sha256=None`` today, so this is normally ``None`` —
            but the gate is wired regardless, and closes the day one is pinned.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        WeightIntegrityError: If the file no longer matches ``raw_digest``.
        ValueError: If the checkpoint format is unrecognised.
        RuntimeError: If weight shapes do not match the model architecture.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    src_state = load_verified_state_dict(path, raw_digest=raw_digest)

    # Remap keys
    mapped: dict[str, _torch.Tensor] = {}
    skipped: list[str] = []

    for src_key, tensor in src_state.items():
        dst_key = _remap_cls_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue
        mapped[dst_key] = tensor

    if skipped:
        logger.debug(
            "Skipped %d checkpoint keys (unremapped/neck/detect layers): %s",
            len(skipped),
            skipped[:5],
        )

    # Shape validation — warn on mismatch, remove offending key (don't crash)
    dst_state = model.state_dict()

    missing = set(dst_state.keys()) - set(mapped.keys())
    if missing:
        logger.warning(
            "%d keys missing from checkpoint (possible version/family mismatch): %s",
            len(missing),
            sorted(missing)[:5],
        )

    shape_mismatches: list[str] = []
    for key in list(mapped.keys()):
        if key in dst_state and mapped[key].shape != dst_state[key].shape:
            shape_mismatches.append(
                f"  {key}: checkpoint {mapped[key].shape} vs model {dst_state[key].shape}"
            )
            del mapped[key]

    if shape_mismatches:
        raise RuntimeError(
            "Shape mismatches between checkpoint and model:\n" + "\n".join(shape_mismatches)
        )

    # Detach before loading: EMA checkpoint tensors may retain autograd graphs
    model.load_state_dict({k: v.detach() for k, v in mapped.items()}, strict=False)
    logger.info(
        "Loaded %d/%d parameters from %s",
        len(mapped),
        len(dst_state),
        path.name,
    )


# ---------------------------------------------------------------------------
# OBB weight mapping
# ---------------------------------------------------------------------------

# OBB models share the same layer structure as detection models (layers 0-23).
# The OBB head sits at layer 23, identical to the detection head except for the
# added cv4 angle branches.  The existing _LAYER_MAP already maps
# ``model.23.`` → ``head.`` so we reuse _remap_key directly.


def load_obb_weights(
    model: OBBModel, weights_path: str | Path, raw_digest: str | None = None
) -> None:
    """Load an ultralytics OBB checkpoint into a native OBBModel.

    Uses the same ``_LAYER_MAP`` as :func:`load_weights`.  The OBB head's
    angle branches (``model.23.cv4.*``) map automatically to ``head.cv4.*``
    via the shared ``"model.23." → "head."`` prefix rule.

    Args:
        model: An initialised ``OBBModel`` (from ``build_obb_model()``).
        weights_path: Path to an ultralytics ``-obb.pt`` checkpoint file.
        raw_digest: The SHA-256 the registry pins for this model. Every ``-obb``
            entry carries ``sha256=None`` today, so this is normally ``None`` —
            but the gate is wired regardless, and closes the day one is pinned.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        WeightIntegrityError: If the file no longer matches ``raw_digest``.
        ValueError: If the checkpoint format is unrecognised.
        RuntimeError: If weight shapes do not match the model architecture.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    src_state = load_verified_state_dict(path, raw_digest=raw_digest)

    # Remap keys using the same _LAYER_MAP as detection loading
    mapped: dict[str, _torch.Tensor] = {}
    skipped: list[str] = []

    for src_key, tensor in src_state.items():
        dst_key = _remap_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue
        mapped[dst_key] = tensor

    if skipped:
        logger.debug(
            "Skipped %d checkpoint keys (parameterless layers): %s",
            len(skipped),
            skipped[:5],
        )

    dst_state = model.state_dict()

    missing = set(dst_state.keys()) - set(mapped.keys())
    unexpected = set(mapped.keys()) - set(dst_state.keys())

    if unexpected:
        logger.warning(
            "Ignoring %d unexpected keys from checkpoint: %s",
            len(unexpected),
            sorted(unexpected)[:5],
        )
        for k in unexpected:
            del mapped[k]

    if missing:
        critical_missing = [
            k for k in missing if "stride" not in k and "anchor" not in k and "dfl.weight" not in k
        ]
        if critical_missing:
            logger.warning(
                "%d keys missing from checkpoint (model may produce incorrect results): %s",
                len(critical_missing),
                sorted(critical_missing)[:10],
            )

    # Shape validation
    shape_mismatches: list[str] = []
    for key in list(mapped.keys()):
        if key in dst_state and mapped[key].shape != dst_state[key].shape:
            shape_mismatches.append(
                f"  {key}: checkpoint {mapped[key].shape} vs model {dst_state[key].shape}"
            )
            del mapped[key]

    if shape_mismatches:
        raise RuntimeError(
            "Shape mismatches between checkpoint and OBB model:\n" + "\n".join(shape_mismatches)
        )

    model.load_state_dict({k: v.detach() for k, v in mapped.items()}, strict=False)
    logger.info(
        "Loaded %d/%d OBB parameters from %s",
        len(mapped),
        len(dst_state),
        path.name,
    )


__all__ = ["load_classify_weights", "load_obb_weights", "load_weights"]
