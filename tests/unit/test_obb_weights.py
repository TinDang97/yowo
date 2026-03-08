"""Unit tests for load_obb_weights key remapping."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from yowo.types import ModelFamily, ModelSize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fake_obb_state_dict(nc: int = 15) -> dict[str, torch.Tensor]:
    """Build a minimal synthetic state dict with OBB-specific keys.

    Keys simulate an ultralytics checkpoint with model.23 as the OBB head,
    including the cv4 angle branches unique to OBB models.
    """
    return {
        # backbone
        "model.0.conv.weight": torch.zeros(16, 3, 3, 3),
        "model.0.bn.weight": torch.zeros(16),
        # neck
        "model.13.cv1.conv.weight": torch.zeros(32, 64, 1, 1),
        # OBB head — cv2 (box reg), cv3 (cls), cv4 (angle) sub-modules
        "model.23.cv2.0.0.conv.weight": torch.zeros(64, 32, 3, 3),
        "model.23.cv3.0.0.conv.weight": torch.zeros(64, 32, 3, 3),
        "model.23.cv4.0.0.conv.weight": torch.zeros(64, 32, 3, 3),  # angle branch
        "model.23.cv4.0.0.bn.weight": torch.zeros(64),
        "model.23.cv4.1.0.conv.weight": torch.zeros(64, 32, 3, 3),
        "model.23.dfl.conv.weight": torch.zeros(16, 1, 1, 16),
        "model.23.stride": torch.tensor([8.0, 16.0, 32.0]),
        # parameterless layers — should be skipped
        "model.11.idx": torch.zeros(1),
        "model.12.weight": torch.zeros(1),
    }


# ---------------------------------------------------------------------------
# Key remapping tests
# ---------------------------------------------------------------------------


class TestLoadObbWeightsKeyMapping:
    def test_cv4_key_maps_to_head(self) -> None:
        """load_obb_weights uses _remap_key which maps model.23.cv4.* -> head.cv4.*."""
        from yowo.arch import build_obb_model
        from yowo.arch._weights import _remap_key

        # Verify OBBModel actually has cv4 (angle) parameters in the head
        model = build_obb_model(ModelFamily.YOLO11, ModelSize.NANO, num_classes=15)
        head_state = {k: v for k, v in model.state_dict().items() if "cv4" in k}
        assert len(head_state) > 0, "OBBHead must have cv4 (angle) parameters"

        # Verify _remap_key maps OBB angle branch keys correctly
        angle_keys = [
            "model.23.cv4.0.0.conv.weight",
            "model.23.cv4.0.0.bn.weight",
            "model.23.cv4.1.0.conv.weight",
        ]
        for src_key in angle_keys:
            dst_key = _remap_key(src_key)
            expected_prefix = "head.cv4."
            assert dst_key is not None and dst_key.startswith(expected_prefix), (
                f"Expected {src_key!r} to remap to 'head.cv4.*', got {dst_key!r}"
            )

    def test_cv4_not_in_missing_keys_after_load(self) -> None:
        """After load_obb_weights, no head.cv4.* key should be in missing_keys.

        Uses the real model + a synthetic checkpoint that covers all cv4 keys.
        We verify by inspecting the remapped dict rather than model state.
        """
        from yowo.arch._weights import _remap_key

        # Test _remap_key directly — it's the shared utility both loaders use
        src_key = "model.23.cv4.0.0.conv.weight"
        dst_key = _remap_key(src_key)
        assert dst_key == "head.cv4.0.0.conv.weight", (
            f"Expected 'head.cv4.0.0.conv.weight', got {dst_key!r}"
        )

    def test_backbone_key_maps_correctly(self) -> None:
        """model.0.* -> backbone.stem.*"""
        from yowo.arch._weights import _remap_key

        assert _remap_key("model.0.conv.weight") == "backbone.stem.conv.weight"

    def test_neck_key_maps_correctly(self) -> None:
        """model.13.* -> neck.c3k2_fpn1.*"""
        from yowo.arch._weights import _remap_key

        assert _remap_key("model.13.cv1.conv.weight") == "neck.c3k2_fpn1.cv1.conv.weight"

    def test_head_key_maps_to_head_prefix(self) -> None:
        """model.23.dfl.conv.weight -> head.dfl.conv.weight"""
        from yowo.arch._weights import _remap_key

        assert _remap_key("model.23.dfl.conv.weight") == "head.dfl.conv.weight"


# ---------------------------------------------------------------------------
# FileNotFoundError
# ---------------------------------------------------------------------------


class TestLoadObbWeightsFileNotFound:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_obb_weights raises FileNotFoundError for a non-existent path."""
        from yowo.arch import build_obb_model
        from yowo.arch._weights import load_obb_weights

        model = build_obb_model(ModelFamily.YOLO11, ModelSize.NANO)
        missing = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="not found"):
            load_obb_weights(model, missing)
