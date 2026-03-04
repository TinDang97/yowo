"""Unit tests for _CLS_LAYER_MAP and load_classify_weights()."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from yowo.arch import build_classify_model
from yowo.arch._config import get_classify_config
from yowo.arch._weights import _CLS_LAYER_MAP, _CLS_SORTED_PREFIXES, load_classify_weights
from yowo.arch._yolo import ClassifyModel
from yowo.types import ModelFamily, ModelSize


class TestClsLayerMap:
    def test_backbone_keys_present(self) -> None:
        """All backbone layer keys (model.0. through model.10.) are in _CLS_LAYER_MAP."""
        for i in range(11):
            key = f"model.{i}."
            assert key in _CLS_LAYER_MAP, f"Missing backbone key: {key}"

    def test_head_key_present(self) -> None:
        """'model.11.' maps to 'head.'."""
        assert "model.11." in _CLS_LAYER_MAP
        assert _CLS_LAYER_MAP["model.11."] == "head."

    def test_no_neck_keys(self) -> None:
        """Detection neck/head keys (model.12 through model.22) are NOT in _CLS_LAYER_MAP."""
        neck_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        for i in neck_indices:
            key = f"model.{i}."
            assert key not in _CLS_LAYER_MAP, f"Neck key should not be in map: {key}"

    def test_sorted_prefixes_longest_first(self) -> None:
        """_CLS_SORTED_PREFIXES are in longest-first order for correct matching."""
        lengths = [len(p) for p in _CLS_SORTED_PREFIXES]
        assert lengths == sorted(lengths, reverse=True)

    def test_all_map_values_semantic(self) -> None:
        """All map values start with 'backbone.' or 'head.'."""
        for dst in _CLS_LAYER_MAP.values():
            assert dst.startswith(("backbone.", "head.")), f"Unexpected dst: {dst}"


class TestLoadClassifyWeights:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when weights path does not exist."""
        model = build_classify_model(ModelFamily.YOLO11, ModelSize.NANO)
        missing = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError):
            load_classify_weights(model, missing)

    def test_key_remapping_loads_successfully(self, tmp_path: Path) -> None:
        """Synthetic ultralytics-format state dict loads via key remapping."""
        config = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO)
        model = ClassifyModel(config)
        yowo_state = model.state_dict()

        # Build inverse map: yowo key prefix -> ultralytics key prefix
        inverse: dict[str, str] = {v: k for k, v in _CLS_LAYER_MAP.items()}

        # Remap yowo keys back to ultralytics format
        ult_state: dict[str, torch.Tensor] = {}
        for yowo_key, tensor in yowo_state.items():
            remapped = None
            # Try to find longest matching yowo prefix
            for yowo_prefix in sorted(inverse.keys(), key=len, reverse=True):
                if yowo_key.startswith(yowo_prefix):
                    ult_prefix = inverse[yowo_prefix]
                    remapped = ult_prefix + yowo_key[len(yowo_prefix) :]
                    break
            if remapped is not None:
                ult_state[remapped] = tensor.clone()

        # Save as a raw state dict (all tensor values)
        ckpt_path = tmp_path / "fake_cls.pt"
        torch.save(ult_state, ckpt_path)

        # Build a fresh model and load — should succeed without error
        fresh_model = ClassifyModel(config)
        load_classify_weights(fresh_model, ckpt_path)

        # Verify at least backbone stem loaded correctly
        loaded_state = fresh_model.state_dict()
        for key, expected in yowo_state.items():
            if key in loaded_state:
                assert torch.allclose(loaded_state[key], expected), f"Mismatch at {key}"

    def test_shape_mismatch_raises_runtime_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Shape mismatch raises RuntimeError (consistent with load_weights behaviour)."""
        config = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO)
        model = ClassifyModel(config)

        # Build a state dict with a wrong-shape tensor for the head linear weight
        yowo_state = model.state_dict()
        inverse: dict[str, str] = {v: k for k, v in _CLS_LAYER_MAP.items()}
        ult_state: dict[str, torch.Tensor] = {}
        for yowo_key, tensor in yowo_state.items():
            remapped = None
            for yowo_prefix in sorted(inverse.keys(), key=len, reverse=True):
                if yowo_key.startswith(yowo_prefix):
                    ult_prefix = inverse[yowo_prefix]
                    remapped = ult_prefix + yowo_key[len(yowo_prefix) :]
                    break
            if remapped is not None:
                # Inject wrong shape for head linear weight
                if yowo_key == "head.linear.weight":
                    ult_state[remapped] = torch.zeros(999, 1280)  # wrong nc
                else:
                    ult_state[remapped] = tensor.clone()

        ckpt_path = tmp_path / "mismatched.pt"
        torch.save(ult_state, ckpt_path)

        fresh_model = ClassifyModel(config)
        with pytest.raises(RuntimeError, match="Shape mismatches"):
            load_classify_weights(fresh_model, ckpt_path)

    def test_missing_keys_emits_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Partial checkpoint (backbone.stem.* only) emits WARNING containing 'missing'."""
        config = get_classify_config(ModelFamily.YOLO11, ModelSize.NANO)
        model = ClassifyModel(config)
        yowo_state = model.state_dict()

        # Build inverse map: yowo prefix -> ultralytics prefix
        inverse: dict[str, str] = {v: k for k, v in _CLS_LAYER_MAP.items()}

        # Only include backbone.stem.* keys — all others will be missing
        partial_ult_state: dict[str, torch.Tensor] = {}
        for yowo_key, tensor in yowo_state.items():
            if not yowo_key.startswith("backbone.stem."):
                continue
            for yowo_prefix in sorted(inverse.keys(), key=len, reverse=True):
                if yowo_key.startswith(yowo_prefix):
                    ult_prefix = inverse[yowo_prefix]
                    ult_key = ult_prefix + yowo_key[len(yowo_prefix) :]
                    partial_ult_state[ult_key] = tensor.clone()
                    break

        ckpt_path = tmp_path / "partial_cls.pt"
        torch.save(partial_ult_state, ckpt_path)

        fresh_model = ClassifyModel(config)
        with caplog.at_level(logging.WARNING, logger="yowo.arch._weights"):
            load_classify_weights(fresh_model, ckpt_path)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("missing" in msg for msg in warning_messages), (
            f"Expected a WARNING containing 'missing'; got: {warning_messages}"
        )
