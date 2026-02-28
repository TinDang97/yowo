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
from yowo.types import BackendType, FrameDropPolicy, ModelFamily, ModelSize, Precision

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

    def test_loads_cache_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("cache: true\n")
        cfg = load_config(path=cfg_file)
        assert cfg.cache is True

    def test_loads_cache_dir_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("cache_dir: /tmp/yowo_cache\n")
        cfg = load_config(path=cfg_file)
        assert cfg.cache_dir == Path("/tmp/yowo_cache")

    def test_loads_kv_cache_from_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("kv_cache: true\n")
        cfg = load_config(path=cfg_file)
        assert cfg.kv_cache is True

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        cfg = load_config(path=cfg_file)
        assert cfg.model_family == ModelFamily.YOLO26

    def test_full_yaml_round_trip(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            model_family: yolo11
            model_size: s
            backend: pytorch
            device: cuda:0
            precision: fp16
            confidence_threshold: 0.6
            iou_threshold: 0.5
            batch_size: 2
            cache: true
            kv_cache: true
        """)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(content)
        cfg = load_config(path=cfg_file)
        assert cfg.model_family == ModelFamily.YOLO11
        assert cfg.model_size == ModelSize.SMALL
        assert cfg.backend == BackendType.PYTORCH
        assert cfg.device == "cuda:0"
        assert cfg.precision == Precision.FP16
        assert cfg.confidence_threshold == pytest.approx(0.6)
        assert cfg.iou_threshold == pytest.approx(0.5)
        assert cfg.batch_size == 2
        assert cfg.cache is True
        assert cfg.kv_cache is True

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

    def test_cache_dir_null_in_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("cache_dir: null\n")
        cfg = load_config(path=cfg_file)
        assert cfg.cache_dir is None

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

    def test_env_overrides_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_CACHE", "true")
        cfg = load_config()
        assert cfg.cache is True

    def test_env_overrides_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_CACHE_DIR", "/tmp/yowo_cache")
        cfg = load_config()
        assert cfg.cache_dir == Path("/tmp/yowo_cache")

    def test_env_overrides_kv_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_KV_CACHE", "yes")
        cfg = load_config()
        assert cfg.kv_cache is True

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
        monkeypatch.setenv("YOWO_MODEL_FAMILY", "yolo26")
        cfg = load_config(path=cfg_file)
        # env var wins
        assert cfg.model_family == ModelFamily.YOLO26

    def test_env_confidence_out_of_range_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YOWO_CONFIDENCE", "1.5")
        with pytest.raises(ConfigError, match="confidence_threshold"):
            load_config()


# ---------------------------------------------------------------------------
# Streaming + pipeline config fields
# ---------------------------------------------------------------------------


class TestStreamingConfig:
    def test_defaults(self) -> None:
        cfg = InferenceConfig(model_family=ModelFamily.YOLO26, model_size=ModelSize.NANO)
        assert cfg.frame_drop_policy == FrameDropPolicy.LATEST
        assert cfg.max_queue_size == 2
        assert cfg.prefetch is True
        assert cfg.pipeline_workers == 0

    def test_max_queue_size_validation(self) -> None:
        with pytest.raises(ConfigError, match="max_queue_size"):
            InferenceConfig(
                model_family=ModelFamily.YOLO26,
                model_size=ModelSize.NANO,
                max_queue_size=0,
            )

    def test_pipeline_workers_validation(self) -> None:
        with pytest.raises(ConfigError, match="pipeline_workers"):
            InferenceConfig(
                model_family=ModelFamily.YOLO26,
                model_size=ModelSize.NANO,
                pipeline_workers=-1,
            )

    def test_frame_drop_policy_latest(self) -> None:
        cfg = InferenceConfig(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            frame_drop_policy=FrameDropPolicy.LATEST,
        )
        assert cfg.frame_drop_policy == FrameDropPolicy.LATEST

    def test_frame_drop_policy_skip_oldest(self) -> None:
        cfg = InferenceConfig(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            frame_drop_policy=FrameDropPolicy.SKIP_OLDEST,
        )
        assert cfg.frame_drop_policy == FrameDropPolicy.SKIP_OLDEST

    def test_max_queue_size_minimum_valid(self) -> None:
        cfg = InferenceConfig(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            max_queue_size=1,
        )
        assert cfg.max_queue_size == 1

    def test_pipeline_workers_explicit_zero(self) -> None:
        cfg = InferenceConfig(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            pipeline_workers=0,
        )
        assert cfg.pipeline_workers == 0

    def test_prefetch_disabled(self) -> None:
        cfg = InferenceConfig(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            prefetch=False,
        )
        assert cfg.prefetch is False

    def test_yaml_loads_frame_drop_policy(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("frame_drop_policy: latest\n")
        cfg = load_config(path=cfg_file)
        assert cfg.frame_drop_policy == FrameDropPolicy.LATEST

    def test_yaml_loads_max_queue_size(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("max_queue_size: 8\n")
        cfg = load_config(path=cfg_file)
        assert cfg.max_queue_size == 8

    def test_yaml_loads_pipeline_workers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("pipeline_workers: 4\n")
        cfg = load_config(path=cfg_file)
        assert cfg.pipeline_workers == 4

    def test_yaml_loads_prefetch_false(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("prefetch: false\n")
        cfg = load_config(path=cfg_file)
        assert cfg.prefetch is False

    def test_yaml_invalid_frame_drop_policy_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("frame_drop_policy: bogus\n")
        with pytest.raises(ConfigError, match="Invalid value in config file"):
            load_config(path=cfg_file)

    def test_env_overrides_frame_drop_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_FRAME_DROP_POLICY", "none")
        cfg = load_config()
        assert cfg.frame_drop_policy == FrameDropPolicy.NONE

    def test_env_overrides_max_queue_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_MAX_QUEUE_SIZE", "16")
        cfg = load_config()
        assert cfg.max_queue_size == 16

    def test_env_overrides_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_PREFETCH", "false")
        cfg = load_config()
        assert cfg.prefetch is False

    def test_env_overrides_prefetch_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_PREFETCH", "1")
        cfg = load_config()
        assert cfg.prefetch is True

    def test_env_overrides_pipeline_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_PIPELINE_WORKERS", "2")
        cfg = load_config()
        assert cfg.pipeline_workers == 2

    def test_env_invalid_frame_drop_policy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_FRAME_DROP_POLICY", "bogus")
        with pytest.raises(ConfigError, match="Invalid YOWO_ environment variable"):
            load_config()

    def test_env_max_queue_size_zero_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_MAX_QUEUE_SIZE", "0")
        with pytest.raises(ConfigError, match="max_queue_size"):
            load_config()

    def test_env_metrics_enabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_METRICS_ENABLED", "false")
        cfg = load_config()
        assert cfg.metrics_enabled is False

    def test_env_metrics_enabled_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_METRICS_ENABLED", "1")
        cfg = load_config()
        assert cfg.metrics_enabled is True

    def test_env_error_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YOWO_ERROR_THRESHOLD", "3")
        cfg = load_config()
        assert cfg.error_threshold == 3


# ---------------------------------------------------------------------------
# error_threshold validation
# ---------------------------------------------------------------------------


class TestErrorThresholdValidation:
    def test_error_threshold_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="error_threshold"):
            InferenceConfig(error_threshold=0)

    def test_error_threshold_negative_raises(self) -> None:
        with pytest.raises(ConfigError, match="error_threshold"):
            InferenceConfig(error_threshold=-1)

    def test_error_threshold_one_is_valid(self) -> None:
        cfg = InferenceConfig(error_threshold=1)
        assert cfg.error_threshold == 1


# ---------------------------------------------------------------------------
# classify_source
# ---------------------------------------------------------------------------


class TestClassifySource:
    def test_image_jpg(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("photo.jpg") == SourceCategory.IMAGE

    def test_image_png(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("photo.png") == SourceCategory.IMAGE

    def test_video_mp4(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("clip.mp4") == SourceCategory.VIDEO

    def test_video_avi(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("clip.avi") == SourceCategory.VIDEO

    def test_rtsp_stream(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("rtsp://host/stream") == SourceCategory.LIVE_STREAM

    def test_webcam_index_zero(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("0") == SourceCategory.LIVE_STREAM

    def test_webcam_index_one(self) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source("1") == SourceCategory.LIVE_STREAM

    def test_directory_is_image(self, tmp_path: Path) -> None:
        from yowo.config import classify_source
        from yowo.types import SourceCategory

        assert classify_source(tmp_path) == SourceCategory.IMAGE

    def test_unknown_extension_raises(self) -> None:
        from yowo.config import classify_source

        with pytest.raises(ConfigError, match="Cannot classify"):
            classify_source("data.xyz")


# ---------------------------------------------------------------------------
# classify_device
# ---------------------------------------------------------------------------


class TestClassifyDevice:
    def test_jetson_device(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import classify_device
        from yowo.types import DeviceCategory

        hw = MagicMock()
        hw.is_jetson = True
        assert classify_device(hw) == DeviceCategory.JETSON

    def test_cuda_high(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import classify_device
        from yowo.types import DeviceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = True
        hw.primary_gpu.memory_total_mb = 16384
        assert classify_device(hw) == DeviceCategory.CUDA_HIGH

    def test_cuda_low(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import classify_device
        from yowo.types import DeviceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = True
        hw.primary_gpu.memory_total_mb = 4096
        assert classify_device(hw) == DeviceCategory.CUDA_LOW

    def test_apple_silicon(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import classify_device
        from yowo.types import CPUArch, DeviceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = False
        hw.libraries.onnxruntime_has_coreml = True
        hw.cpu.cpu_arch = CPUArch.AARCH64
        assert classify_device(hw) == DeviceCategory.APPLE_SILICON

    def test_cpu_x86(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import classify_device
        from yowo.types import CPUArch, DeviceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = False
        hw.libraries.onnxruntime_has_coreml = False
        hw.cpu.cpu_arch = CPUArch.X86_64
        assert classify_device(hw) == DeviceCategory.CPU_X86

    def test_cpu_arm(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import classify_device
        from yowo.types import CPUArch, DeviceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = False
        hw.libraries.onnxruntime_has_coreml = False
        hw.cpu.cpu_arch = CPUArch.AARCH64
        assert classify_device(hw) == DeviceCategory.CPU_ARM


# ---------------------------------------------------------------------------
# preset_config
# ---------------------------------------------------------------------------


class TestPresetConfig:
    def test_returns_inference_config(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import preset_config
        from yowo.types import CPUArch, SourceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = False
        hw.libraries.onnxruntime_has_coreml = False
        hw.cpu.cpu_arch = CPUArch.X86_64
        cfg = preset_config(hw, SourceCategory.VIDEO)
        assert isinstance(cfg, InferenceConfig)

    def test_cuda_high_video_uses_batch_4(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import preset_config
        from yowo.types import SourceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = True
        hw.primary_gpu.memory_total_mb = 16384
        cfg = preset_config(hw, SourceCategory.VIDEO)
        assert cfg.batch_size == 4
        assert cfg.cache is True

    def test_cuda_high_live_uses_kv_cache(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import preset_config
        from yowo.types import SourceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = True
        hw.primary_gpu.memory_total_mb = 16384
        cfg = preset_config(hw, SourceCategory.LIVE_STREAM)
        assert cfg.kv_cache is True
        assert cfg.frame_drop_policy == FrameDropPolicy.LATEST

    def test_overrides_win_over_preset(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import preset_config
        from yowo.types import CPUArch, SourceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = False
        hw.libraries.onnxruntime_has_coreml = False
        hw.cpu.cpu_arch = CPUArch.X86_64
        cfg = preset_config(hw, SourceCategory.VIDEO, batch_size=8)
        assert cfg.batch_size == 8

    def test_unknown_override_key_raises(self) -> None:
        from unittest.mock import MagicMock

        from yowo.config import preset_config
        from yowo.types import CPUArch, SourceCategory

        hw = MagicMock()
        hw.is_jetson = False
        hw.has_nvidia_gpu = False
        hw.libraries.onnxruntime_has_coreml = False
        hw.cpu.cpu_arch = CPUArch.X86_64
        with pytest.raises(ConfigError, match="Unknown InferenceConfig fields"):
            preset_config(hw, SourceCategory.IMAGE, nonexistent_field=True)
