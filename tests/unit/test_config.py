"""Unit tests for yowo.config.load_config and env-var overrides.

Covers the YAML loading path, env override logic, and error propagation
that are not exercised by test_types.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from yowo.config import InferenceConfig, load_config
from yowo.errors import ConfigError
from yowo.types import BackendType, ModelFamily, ModelSize, Precision

# ---------------------------------------------------------------------------
# load_config — no file
# ---------------------------------------------------------------------------


class TestLoadConfigNoFile:
    def test_returns_defaults_when_path_is_none(self) -> None:
        cfg = load_config(path=None)
        assert cfg.model_family == ModelFamily.YOLO26
        assert cfg.model_size == ModelSize.NANO
        assert cfg.confidence_threshold == 0.25

    def test_returns_defaults_when_path_does_not_exist(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        cfg = load_config(path=missing)
        assert isinstance(cfg, InferenceConfig)


# ---------------------------------------------------------------------------
# load_config — YAML file
# ---------------------------------------------------------------------------


class TestLoadConfigFromYaml:
    def test_loads_model_family_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model_family: yolo11\n")
        cfg = load_config(path=cfg_file)
        assert cfg.model_family == ModelFamily.YOLO11

    def test_loads_model_size_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model_size: l\n")
        cfg = load_config(path=cfg_file)
        assert cfg.model_size == ModelSize.LARGE

    def test_loads_confidence_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("confidence_threshold: 0.7\n")
        cfg = load_config(path=cfg_file)
        assert cfg.confidence_threshold == pytest.approx(0.7)

    def test_loads_backend_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("backend: onnx\n")
        cfg = load_config(path=cfg_file)
        assert cfg.backend == BackendType.ONNX

    def test_loads_precision_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("precision: fp32\n")
        cfg = load_config(path=cfg_file)
        assert cfg.precision == Precision.FP32

    def test_loads_weights_path_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("weights_path: /models/yolo26n.pt\n")
        cfg = load_config(path=cfg_file)
        assert cfg.weights_path == Path("/models/yolo26n.pt")

    def test_loads_batch_size_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("batch_size: 4\n")
        cfg = load_config(path=cfg_file)
        assert cfg.batch_size == 4

    def test_loads_max_memory_mb_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("max_memory_mb: 8192\n")
        cfg = load_config(path=cfg_file)
        assert cfg.max_memory_mb == 8192

    def test_loads_frame_skip_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("frame_skip: 2\n")
        cfg = load_config(path=cfg_file)
        assert cfg.frame_skip == 2

    def test_loads_max_frames_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("max_frames: 1000\n")
        cfg = load_config(path=cfg_file)
        assert cfg.max_frames == 1000

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        cfg = load_config(path=cfg_file)
        assert cfg.model_family == ModelFamily.YOLO26

    def test_full_yaml_round_trip(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            model_family: yolo12
            model_size: s
            backend: pytorch
            device: cuda:0
            precision: fp16
            confidence_threshold: 0.6
            iou_threshold: 0.5
            batch_size: 2
            frame_skip: 1
            reconnect_timeout_s: 10.0
        """)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(content)
        cfg = load_config(path=cfg_file)
        assert cfg.model_family == ModelFamily.YOLO12
        assert cfg.model_size == ModelSize.SMALL
        assert cfg.backend == BackendType.PYTORCH
        assert cfg.device == "cuda:0"
        assert cfg.precision == Precision.FP16
        assert cfg.confidence_threshold == pytest.approx(0.6)
        assert cfg.iou_threshold == pytest.approx(0.5)
        assert cfg.batch_size == 2
        assert cfg.frame_skip == 1

    def test_invalid_enum_value_in_yaml_raises_config_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model_family: yolo99\n")
        with pytest.raises(ConfigError, match="Invalid value in config file"):
            load_config(path=cfg_file)

    def test_invalid_yaml_syntax_raises_config_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("key: [unclosed\n")
        with pytest.raises(ConfigError, match="Failed to parse config file"):
            load_config(path=cfg_file)

    def test_out_of_range_confidence_in_yaml_raises_config_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("confidence_threshold: 2.5\n")
        with pytest.raises(ConfigError, match="confidence_threshold"):
            load_config(path=cfg_file)

    def test_max_frames_null_in_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("max_frames: null\n")
        cfg = load_config(path=cfg_file)
        assert cfg.max_frames is None

    def test_max_memory_null_in_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("max_memory_mb: null\n")
        cfg = load_config(path=cfg_file)
        assert cfg.max_memory_mb is None

    def test_backend_null_in_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("backend: null\n")
        cfg = load_config(path=cfg_file)
        assert cfg.backend is None


# ---------------------------------------------------------------------------
# load_config — environment variable overrides
# ---------------------------------------------------------------------------


class TestLoadConfigEnvOverrides:
    def test_env_overrides_model_family(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YOWO_MODEL_FAMILY", "yolo11")
        cfg = load_config()
        assert cfg.model_family == ModelFamily.YOLO11

    def test_env_overrides_model_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_MODEL_SIZE", "x")
        cfg = load_config()
        assert cfg.model_size == ModelSize.XLARGE

    def test_env_overrides_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_BACKEND", "onnx")
        cfg = load_config()
        assert cfg.backend == BackendType.ONNX

    def test_env_overrides_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_DEVICE", "cuda:1")
        cfg = load_config()
        assert cfg.device == "cuda:1"

    def test_env_overrides_precision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_PRECISION", "fp32")
        cfg = load_config()
        assert cfg.precision == Precision.FP32

    def test_env_overrides_confidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_CONFIDENCE", "0.8")
        cfg = load_config()
        assert cfg.confidence_threshold == pytest.approx(0.8)

    def test_env_overrides_iou(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_IOU", "0.3")
        cfg = load_config()
        assert cfg.iou_threshold == pytest.approx(0.3)

    def test_env_overrides_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_BATCH_SIZE", "8")
        cfg = load_config()
        assert cfg.batch_size == 8

    def test_env_overrides_max_memory_mb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_MAX_MEMORY_MB", "4096")
        cfg = load_config()
        assert cfg.max_memory_mb == 4096

    def test_env_overrides_reconnect_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_RECONNECT_TIMEOUT", "60.0")
        cfg = load_config()
        assert cfg.reconnect_timeout_s == pytest.approx(60.0)

    def test_env_overrides_frame_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_FRAME_SKIP", "3")
        cfg = load_config()
        assert cfg.frame_skip == 3

    def test_env_overrides_max_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_MAX_FRAMES", "500")
        cfg = load_config()
        assert cfg.max_frames == 500

    def test_env_overrides_weights_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_WEIGHTS_PATH", "/data/model.pt")
        cfg = load_config()
        assert cfg.weights_path == Path("/data/model.pt")

    def test_invalid_env_backend_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_BACKEND", "notabackend")
        with pytest.raises(ConfigError, match="Invalid YOWO_ environment variable"):
            load_config()

    def test_env_override_wins_over_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model_family: yolo11\n")
        monkeypatch.setenv("YOWO_MODEL_FAMILY", "yolo12")
        cfg = load_config(path=cfg_file)
        # env var wins
        assert cfg.model_family == ModelFamily.YOLO12

    def test_env_confidence_out_of_range_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YOWO_CONFIDENCE", "1.5")
        with pytest.raises(ConfigError, match="confidence_threshold"):
            load_config()
