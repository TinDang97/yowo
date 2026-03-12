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

import logging
from pathlib import Path
from typing import TYPE_CHECKING

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


def _extract_state_dict(checkpoint_path: Path) -> dict[str, _torch.Tensor]:
    """Load checkpoint and extract the float32 state_dict.

    Handles ultralytics checkpoint format:
    - Prefers EMA weights (``ckpt['ema']``) over training weights.
    - Converts to float32 (ultralytics may store FP16).
    - Handles both full checkpoint dicts and raw state_dicts.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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


def load_weights(model: YOLOModel, weights_path: str | Path) -> None:
    """Load ultralytics checkpoint weights into a native YOLOModel.

    Args:
        model: An initialised ``YOLOModel`` (from ``build_model()``).
        weights_path: Path to an ultralytics ``.pt`` checkpoint file.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        ValueError: If the checkpoint format is unrecognised.
        RuntimeError: If weight shapes do not match the model architecture.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    src_state = _extract_state_dict(path)

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


def load_classify_weights(model: ClassifyModel, weights_path: str | Path) -> None:
    """Load an ultralytics classification checkpoint into a ClassifyModel.

    Args:
        model: An initialised ``ClassifyModel`` (from ``build_classify_model()``).
        weights_path: Path to an ultralytics ``-cls.pt`` checkpoint file.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        ValueError: If the checkpoint format is unrecognised.
        RuntimeError: If weight shapes do not match the model architecture.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    src_state = _extract_state_dict(path)

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


def load_obb_weights(model: OBBModel, weights_path: str | Path) -> None:
    """Load an ultralytics OBB checkpoint into a native OBBModel.

    Uses the same ``_LAYER_MAP`` as :func:`load_weights`.  The OBB head's
    angle branches (``model.23.cv4.*``) map automatically to ``head.cv4.*``
    via the shared ``"model.23." → "head."`` prefix rule.

    Args:
        model: An initialised ``OBBModel`` (from ``build_obb_model()``).
        weights_path: Path to an ultralytics ``-obb.pt`` checkpoint file.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        ValueError: If the checkpoint format is unrecognised.
        RuntimeError: If weight shapes do not match the model architecture.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    src_state = _extract_state_dict(path)

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
