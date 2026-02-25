# Preset Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add opt-in `preset_config()` factory and `--preset` CLI flag that auto-tunes pipeline/caching knobs based on detected hardware + input source type.

**Architecture:** A static lookup table in `config.py` maps `(DeviceCategory, SourceCategory)` pairs to `_PresetOverrides`. Two new StrEnum types in `types.py`. CLI wires `--preset` flag to the factory with explicit-flag override layering.

**Tech Stack:** Python 3.11+, dataclasses, StrEnum, Click CLI, pytest

---

### Task 1: Add SourceCategory and DeviceCategory enums to types.py

**Files:**
- Modify: `src/yowo/types.py` (insert after `StreamState` at line 110, update `__all__`)
- Test: `tests/unit/test_preset.py`

**Step 1: Write the failing test**

Create `tests/unit/test_preset.py`:

```python
"""Tests for preset inference auto-tuning."""

from __future__ import annotations

from yowo.types import DeviceCategory, SourceCategory


class TestSourceCategory:
    def test_values(self) -> None:
        assert SourceCategory.IMAGE == "image"
        assert SourceCategory.VIDEO == "video"
        assert SourceCategory.LIVE_STREAM == "live"

    def test_is_str_enum(self) -> None:
        assert isinstance(SourceCategory.IMAGE, str)


class TestDeviceCategory:
    def test_values(self) -> None:
        assert DeviceCategory.CUDA_HIGH == "cuda_high"
        assert DeviceCategory.CUDA_LOW == "cuda_low"
        assert DeviceCategory.JETSON == "jetson"
        assert DeviceCategory.APPLE_SILICON == "apple_silicon"
        assert DeviceCategory.CPU_X86 == "cpu_x86"
        assert DeviceCategory.CPU_ARM == "cpu_arm"

    def test_is_str_enum(self) -> None:
        assert isinstance(DeviceCategory.CUDA_HIGH, str)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_preset.py -x -q`
Expected: FAIL with `ImportError: cannot import name 'DeviceCategory'`

**Step 3: Write minimal implementation**

In `src/yowo/types.py`, insert after `StreamState` (after line 110):

```python
class SourceCategory(enum.StrEnum):
    """Input source classification for preset selection."""

    IMAGE = "image"
    VIDEO = "video"
    LIVE_STREAM = "live"


class DeviceCategory(enum.StrEnum):
    """Hardware classification for preset selection."""

    CUDA_HIGH = "cuda_high"
    CUDA_LOW = "cuda_low"
    JETSON = "jetson"
    APPLE_SILICON = "apple_silicon"
    CPU_X86 = "cpu_x86"
    CPU_ARM = "cpu_arm"
```

Update `__all__` in `src/yowo/types.py` — add `"DeviceCategory"` and `"SourceCategory"` in alphabetical position.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_preset.py -x -q`
Expected: PASS (2 tests)

**Step 5: Commit**

```
feat(types): add SourceCategory and DeviceCategory enums

New StrEnum types for preset inference auto-tuning. SourceCategory
classifies input sources (image/video/live). DeviceCategory classifies
hardware (cuda_high/cuda_low/jetson/apple_silicon/cpu_x86/cpu_arm).
```

---

### Task 2: Implement classify_source()

**Files:**
- Modify: `src/yowo/config.py` (add function + import `SourceCategory`)
- Modify: `src/yowo/config.py` `__all__` (add `"classify_source"`)
- Test: `tests/unit/test_preset.py` (append tests)

**Step 1: Write the failing tests**

Append to `tests/unit/test_preset.py`:

```python
from pathlib import Path

from yowo.config import classify_source


class TestClassifySource:
    def test_image_jpg(self) -> None:
        assert classify_source("photo.jpg") == SourceCategory.IMAGE

    def test_image_png(self) -> None:
        assert classify_source(Path("photo.png")) == SourceCategory.IMAGE

    def test_image_bmp(self) -> None:
        assert classify_source("scan.bmp") == SourceCategory.IMAGE

    def test_image_webp(self) -> None:
        assert classify_source("img.webp") == SourceCategory.IMAGE

    def test_video_mp4(self) -> None:
        assert classify_source("clip.mp4") == SourceCategory.VIDEO

    def test_video_avi(self) -> None:
        assert classify_source("clip.avi") == SourceCategory.VIDEO

    def test_video_mov(self) -> None:
        assert classify_source(Path("clip.mov")) == SourceCategory.VIDEO

    def test_video_mkv(self) -> None:
        assert classify_source("clip.mkv") == SourceCategory.VIDEO

    def test_rtsp_url(self) -> None:
        assert classify_source("rtsp://192.168.1.1/stream") == SourceCategory.LIVE_STREAM

    def test_rtsps_url(self) -> None:
        assert classify_source("rtsps://cam.example.com/feed") == SourceCategory.LIVE_STREAM

    def test_webcam_zero(self) -> None:
        assert classify_source("0") == SourceCategory.LIVE_STREAM

    def test_webcam_one(self) -> None:
        assert classify_source("1") == SourceCategory.LIVE_STREAM

    def test_directory(self, tmp_path: Path) -> None:
        assert classify_source(tmp_path) == SourceCategory.IMAGE

    def test_unknown_extension_raises(self) -> None:
        import pytest
        from yowo.errors import ConfigError

        with pytest.raises(ConfigError):
            classify_source("data.xyz")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_preset.py::TestClassifySource -x -q`
Expected: FAIL with `ImportError: cannot import name 'classify_source'`

**Step 3: Write minimal implementation**

In `src/yowo/config.py`, add import at line 35:

```python
from yowo.types import (
    BackendType, ExportFormat, FrameDropPolicy, ModelFamily, ModelSize, Precision,
    SourceCategory,
)
```

Add the function before `load_config` (around line 240, after `_dict_to_inference_config`):

```python
# ---------------------------------------------------------------------------
# Preset inference: source classification
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".ts"})
_RTSP_SCHEMES = ("rtsp://", "rtsps://")


def classify_source(source: str | Path) -> SourceCategory:
    """Classify a source string into a SourceCategory without opening it.

    Uses the same dispatch rules as ``open_source()`` in ``yowo.io``.

    Args:
        source: File path, URL string, webcam index string, or directory.

    Returns:
        The matching SourceCategory.

    Raises:
        ConfigError: If the source type cannot be determined.
    """
    source_str = str(source)

    # Webcam: pure digit string → live
    if isinstance(source, str) and source.isdigit():
        return SourceCategory.LIVE_STREAM

    # RTSP → live
    if source_str.startswith(_RTSP_SCHEMES):
        return SourceCategory.LIVE_STREAM

    path = Path(source_str)
    suffix = path.suffix.lower()

    # Existing directory → image batch
    if path.is_dir():
        return SourceCategory.IMAGE

    if suffix in _IMAGE_EXTS:
        return SourceCategory.IMAGE

    if suffix in _VIDEO_EXTS:
        return SourceCategory.VIDEO

    raise ConfigError(
        f"Cannot classify source type for: {source!r}. "
        f"Supported: image files {sorted(_IMAGE_EXTS)}, "
        f"video files {sorted(_VIDEO_EXTS)}, "
        f'RTSP URLs (rtsp://), webcam indices ("0", "1", ...).'
    )
```

Update `__all__` in `src/yowo/config.py`:

```python
__all__ = [
    "ExportConfig",
    "InferenceConfig",
    "classify_source",
    "load_config",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_preset.py::TestClassifySource -x -q`
Expected: PASS (14 tests)

**Step 5: Commit**

```
feat(config): add classify_source() for preset inference

Pure function that classifies source strings into SourceCategory
(IMAGE/VIDEO/LIVE_STREAM) using the same dispatch rules as open_source()
without opening the source. Raises ConfigError for unrecognized types.
```

---

### Task 3: Implement classify_device()

**Files:**
- Modify: `src/yowo/config.py` (add function + import `DeviceCategory`, `CPUArch`)
- Modify: `src/yowo/config.py` `__all__` (add `"classify_device"`)
- Test: `tests/unit/test_preset.py` (append tests)

**Step 1: Write the failing tests**

Append to `tests/unit/test_preset.py`. These tests construct synthetic `HardwareProfile` objects. Reference `src/yowo/hardware/_device.py` for `Device` and `src/yowo/hardware/_capabilities.py` for `InstalledLibraries`.

```python
from yowo.config import classify_device
from yowo.hardware import HardwareProfile
from yowo.hardware._device import Device
from yowo.hardware._capabilities import InstalledLibraries
from yowo.types import CPUArch, DeviceCategory, DeviceType, GPUArch


def _cpu_device(
    arch: CPUArch = CPUArch.X86_64,
    is_jetson: bool = False,
    memory_mb: int = 16384,
) -> Device:
    return Device(
        type=DeviceType.CPU,
        index=0,
        name="test-cpu",
        arch=None,
        cpu_arch=arch,
        memory_total_mb=memory_mb,
        memory_available_mb=memory_mb // 2,
        is_jetson=is_jetson,
    )


def _gpu_device(
    vram_mb: int = 8192,
    gpu_arch: GPUArch = GPUArch.AMPERE,
    is_jetson: bool = False,
) -> Device:
    return Device(
        type=DeviceType.CUDA,
        index=0,
        name="test-gpu",
        arch=gpu_arch,
        cpu_arch=CPUArch.X86_64,
        memory_total_mb=vram_mb,
        memory_available_mb=vram_mb // 2,
        is_jetson=is_jetson,
    )


def _libs(**kwargs: object) -> InstalledLibraries:
    return InstalledLibraries(**kwargs)  # type: ignore[arg-type]


def _hw(
    gpus: tuple[Device, ...] = (),
    cpu: Device | None = None,
    libs: InstalledLibraries | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        gpus=gpus,
        cpu=cpu or _cpu_device(),
        cpu_features=frozenset(),
        libraries=libs or _libs(),
    )


class TestClassifyDevice:
    def test_jetson(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.AARCH64, is_jetson=True))
        assert classify_device(hw) == DeviceCategory.JETSON

    def test_cuda_high_vram(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=24576),))
        assert classify_device(hw) == DeviceCategory.CUDA_HIGH

    def test_cuda_high_boundary(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=8192),))
        assert classify_device(hw) == DeviceCategory.CUDA_HIGH

    def test_cuda_low_vram(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=4096),))
        assert classify_device(hw) == DeviceCategory.CUDA_LOW

    def test_apple_silicon(self) -> None:
        hw = _hw(
            cpu=_cpu_device(arch=CPUArch.AARCH64),
            libs=_libs(onnxruntime_version="1.17.0", onnxruntime_has_coreml=True),
        )
        assert classify_device(hw) == DeviceCategory.APPLE_SILICON

    def test_cpu_x86(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.X86_64))
        assert classify_device(hw) == DeviceCategory.CPU_X86

    def test_cpu_arm_no_coreml(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.AARCH64))
        assert classify_device(hw) == DeviceCategory.CPU_ARM

    def test_jetson_priority_over_cuda(self) -> None:
        """Jetson has a GPU but should be classified as JETSON, not CUDA."""
        jetson_gpu = _gpu_device(vram_mb=8192, is_jetson=True)
        hw = _hw(
            gpus=(jetson_gpu,),
            cpu=_cpu_device(arch=CPUArch.AARCH64, is_jetson=True),
        )
        assert classify_device(hw) == DeviceCategory.JETSON
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_preset.py::TestClassifyDevice -x -q`
Expected: FAIL with `ImportError: cannot import name 'classify_device'`

**Step 3: Write minimal implementation**

In `src/yowo/config.py`, update the import from `yowo.types`:

```python
from yowo.types import (
    BackendType, CPUArch, ExportFormat, FrameDropPolicy, ModelFamily, ModelSize,
    Precision, SourceCategory, DeviceCategory,
)
```

Add a new import at the top of `config.py` (lazy to avoid circular import):

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yowo.hardware import HardwareProfile
```

Add the function after `classify_source`:

```python
_VRAM_HIGH_THRESHOLD_MB = 8192


def classify_device(hw: HardwareProfile) -> DeviceCategory:
    """Classify hardware into a DeviceCategory for preset selection.

    Priority order: Jetson > CUDA_HIGH > CUDA_LOW > APPLE_SILICON > CPU_X86 > CPU_ARM.

    Args:
        hw: Hardware profile snapshot.

    Returns:
        The matching DeviceCategory.
    """
    if hw.is_jetson:
        return DeviceCategory.JETSON

    if hw.has_nvidia_gpu:
        gpu = hw.primary_gpu
        assert gpu is not None  # guarded by has_nvidia_gpu
        if gpu.memory_total_mb >= _VRAM_HIGH_THRESHOLD_MB:
            return DeviceCategory.CUDA_HIGH
        return DeviceCategory.CUDA_LOW

    if hw.libraries.onnxruntime_has_coreml and hw.cpu.cpu_arch == CPUArch.AARCH64:
        return DeviceCategory.APPLE_SILICON

    if hw.cpu.cpu_arch == CPUArch.X86_64:
        return DeviceCategory.CPU_X86

    return DeviceCategory.CPU_ARM
```

Update `__all__`:

```python
__all__ = [
    "ExportConfig",
    "InferenceConfig",
    "classify_device",
    "classify_source",
    "load_config",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_preset.py::TestClassifyDevice -x -q`
Expected: PASS (8 tests)

**Step 5: Commit**

```
feat(config): add classify_device() for preset inference

Pure function that classifies HardwareProfile into DeviceCategory
with priority: Jetson > CUDA_HIGH (>=8GB) > CUDA_LOW > APPLE_SILICON
(AARCH64 + CoreML) > CPU_X86 > CPU_ARM.
```

---

### Task 4: Implement _PresetOverrides, _PRESET_TABLE, and preset_config()

**Files:**
- Modify: `src/yowo/config.py` (add dataclass, table, and factory function)
- Modify: `src/yowo/config.py` `__all__` (add `"preset_config"`)
- Test: `tests/unit/test_preset.py` (append tests)

**Step 1: Write the failing tests**

Append to `tests/unit/test_preset.py`:

```python
import pytest

from yowo.config import preset_config
from yowo.errors import ConfigError


class TestPresetConfig:
    def test_cuda_high_video(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=24576),))
        cfg = preset_config(hw, SourceCategory.VIDEO)
        assert cfg.batch_size == 4
        assert cfg.cache is True
        assert cfg.prefetch is True

    def test_cuda_high_live(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=24576),))
        cfg = preset_config(hw, SourceCategory.LIVE_STREAM)
        assert cfg.batch_size == 1
        assert cfg.kv_cache is True
        assert cfg.frame_drop_policy == FrameDropPolicy.LATEST
        assert cfg.max_queue_size == 4

    def test_cpu_x86_image(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.X86_64))
        cfg = preset_config(hw, SourceCategory.IMAGE)
        assert cfg.batch_size == 1
        assert cfg.prefetch is False

    def test_cpu_x86_live_no_kv_cache(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.X86_64))
        cfg = preset_config(hw, SourceCategory.LIVE_STREAM)
        assert cfg.kv_cache is False
        assert cfg.frame_drop_policy == FrameDropPolicy.LATEST

    def test_apple_silicon_video(self) -> None:
        hw = _hw(
            cpu=_cpu_device(arch=CPUArch.AARCH64),
            libs=_libs(onnxruntime_version="1.17.0", onnxruntime_has_coreml=True),
        )
        cfg = preset_config(hw, SourceCategory.VIDEO)
        assert cfg.batch_size == 2
        assert cfg.cache is True
        assert cfg.prefetch is True

    def test_override_batch_size(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=24576),))
        cfg = preset_config(hw, SourceCategory.VIDEO, batch_size=8)
        assert cfg.batch_size == 8
        # other preset values still applied
        assert cfg.cache is True
        assert cfg.prefetch is True

    def test_override_model_family(self) -> None:
        from yowo.types import ModelFamily

        hw = _hw()
        cfg = preset_config(hw, SourceCategory.IMAGE, model_family=ModelFamily.YOLO11)
        assert cfg.model_family == ModelFamily.YOLO11

    def test_unknown_override_raises(self) -> None:
        hw = _hw()
        with pytest.raises(ConfigError, match="Unknown"):
            preset_config(hw, SourceCategory.IMAGE, nonexistent_field=42)

    def test_does_not_set_backend(self) -> None:
        """Preset must not override backend/device/precision."""
        hw = _hw(gpus=(_gpu_device(vram_mb=24576),))
        cfg = preset_config(hw, SourceCategory.VIDEO)
        assert cfg.backend is None
        assert cfg.device == "auto"
        assert cfg.precision is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_preset.py::TestPresetConfig -x -q`
Expected: FAIL with `ImportError: cannot import name 'preset_config'`

**Step 3: Write minimal implementation**

In `src/yowo/config.py`, add the `_PresetOverrides` dataclass, lookup table, and factory function after `classify_device`:

```python
@dataclass(frozen=True, slots=True)
class _PresetOverrides:
    """Pipeline/caching knobs that differ from InferenceConfig defaults."""

    batch_size: int | None = None
    cache: bool | None = None
    kv_cache: bool | None = None
    prefetch: bool | None = None
    frame_drop_policy: FrameDropPolicy | None = None
    max_queue_size: int | None = None
    pipeline_workers: int | None = None
    confidence_threshold: float | None = None


# fmt: off
_PRESET_TABLE: dict[tuple[DeviceCategory, SourceCategory], _PresetOverrides] = {
    # ── CUDA HIGH (>= 8GB VRAM) ──
    (DeviceCategory.CUDA_HIGH, SourceCategory.IMAGE):       _PresetOverrides(batch_size=1, prefetch=False),
    (DeviceCategory.CUDA_HIGH, SourceCategory.VIDEO):       _PresetOverrides(batch_size=4, cache=True, prefetch=True),
    (DeviceCategory.CUDA_HIGH, SourceCategory.LIVE_STREAM): _PresetOverrides(batch_size=1, kv_cache=True, frame_drop_policy=FrameDropPolicy.LATEST, max_queue_size=4),
    # ── CUDA LOW (< 8GB VRAM) ──
    (DeviceCategory.CUDA_LOW, SourceCategory.IMAGE):        _PresetOverrides(batch_size=1, prefetch=False),
    (DeviceCategory.CUDA_LOW, SourceCategory.VIDEO):        _PresetOverrides(batch_size=2, prefetch=True),
    (DeviceCategory.CUDA_LOW, SourceCategory.LIVE_STREAM):  _PresetOverrides(batch_size=1, kv_cache=True, frame_drop_policy=FrameDropPolicy.LATEST, max_queue_size=2),
    # ── JETSON ──
    (DeviceCategory.JETSON, SourceCategory.IMAGE):          _PresetOverrides(batch_size=1, prefetch=False),
    (DeviceCategory.JETSON, SourceCategory.VIDEO):          _PresetOverrides(batch_size=1, prefetch=True),
    (DeviceCategory.JETSON, SourceCategory.LIVE_STREAM):    _PresetOverrides(batch_size=1, kv_cache=True, frame_drop_policy=FrameDropPolicy.LATEST, max_queue_size=2),
    # ── APPLE SILICON (CoreML) ──
    (DeviceCategory.APPLE_SILICON, SourceCategory.IMAGE):       _PresetOverrides(batch_size=1, prefetch=False),
    (DeviceCategory.APPLE_SILICON, SourceCategory.VIDEO):       _PresetOverrides(batch_size=2, cache=True, prefetch=True),
    (DeviceCategory.APPLE_SILICON, SourceCategory.LIVE_STREAM): _PresetOverrides(batch_size=1, kv_cache=True, frame_drop_policy=FrameDropPolicy.LATEST, max_queue_size=2),
    # ── CPU x86 ──
    (DeviceCategory.CPU_X86, SourceCategory.IMAGE):         _PresetOverrides(batch_size=1, prefetch=False),
    (DeviceCategory.CPU_X86, SourceCategory.VIDEO):         _PresetOverrides(batch_size=2, prefetch=True),
    (DeviceCategory.CPU_X86, SourceCategory.LIVE_STREAM):   _PresetOverrides(batch_size=1, frame_drop_policy=FrameDropPolicy.LATEST, max_queue_size=2),
    # ── CPU ARM (no CoreML) ──
    (DeviceCategory.CPU_ARM, SourceCategory.IMAGE):         _PresetOverrides(batch_size=1, prefetch=False),
    (DeviceCategory.CPU_ARM, SourceCategory.VIDEO):         _PresetOverrides(batch_size=1, prefetch=True),
    (DeviceCategory.CPU_ARM, SourceCategory.LIVE_STREAM):   _PresetOverrides(batch_size=1, frame_drop_policy=FrameDropPolicy.LATEST, max_queue_size=2),
}
# fmt: on

_INFERENCE_CONFIG_FIELDS = frozenset(f.name for f in __import__("dataclasses").fields(InferenceConfig))


def preset_config(
    hw: HardwareProfile,
    source_type: SourceCategory,
    **overrides: Any,
) -> InferenceConfig:
    """Build a device+source-optimized InferenceConfig.

    Looks up optimal pipeline/caching knobs for the ``(device, source)``
    combination, then applies any explicit ``**overrides`` on top.

    Backend, device, and precision are **not** set by presets — those
    are resolved later by ``select_backend()``.

    Args:
        hw: Hardware profile snapshot.
        source_type: Classified source category.
        **overrides: Any ``InferenceConfig`` field name. Values replace
            preset values. Unknown keys raise ``ConfigError``.

    Returns:
        A validated ``InferenceConfig`` with preset + override values.

    Raises:
        ConfigError: If an unknown override key is passed.
    """
    bad_keys = set(overrides) - _INFERENCE_CONFIG_FIELDS
    if bad_keys:
        raise ConfigError(f"Unknown InferenceConfig fields: {sorted(bad_keys)}")

    device_cat = classify_device(hw)
    preset = _PRESET_TABLE[(device_cat, source_type)]

    # Start from preset values (only non-None fields)
    kwargs: dict[str, Any] = {}
    for f in __import__("dataclasses").fields(preset):
        val = getattr(preset, f.name)
        if val is not None:
            kwargs[f.name] = val

    # Explicit overrides win
    kwargs.update(overrides)

    return InferenceConfig(**kwargs)
```

Update `__all__`:

```python
__all__ = [
    "ExportConfig",
    "InferenceConfig",
    "classify_device",
    "classify_source",
    "load_config",
    "preset_config",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_preset.py::TestPresetConfig -x -q`
Expected: PASS (9 tests)

**Step 5: Run all preset tests + quality gates**

Run: `uv run pytest tests/unit/test_preset.py -x -q`
Expected: PASS (all ~33 tests)

Run: `uv run ruff check src/yowo/config.py src/yowo/types.py`
Run: `uv run pyright src/yowo/config.py src/yowo/types.py`

**Step 6: Commit**

```
feat(config): add preset_config() factory with lookup table

Static lookup table maps (DeviceCategory, SourceCategory) to optimal
pipeline/caching knobs. preset_config() builds InferenceConfig with
preset values, explicit **overrides win. Does not set backend/device/
precision — those remain delegated to select_backend().
```

---

### Task 5: Wire --preset flag into CLI detect command

**Files:**
- Modify: `src/yowo/cli/_main.py` (add `--preset` flag, conditional config path)
- Test: `tests/unit/test_preset.py` (append CLI tests)

**Step 1: Write the failing tests**

Append to `tests/unit/test_preset.py`:

```python
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from yowo.cli._main import cli


class TestPresetCLI:
    def test_preset_flag_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["detect", "--help"])
        assert "--preset" in result.output

    @patch("yowo.cli._main.InferenceEngine")
    @patch("yowo.cli._main.open_source")
    @patch("yowo.cli._main.get_hardware_profile")
    @patch("yowo.cli._main.preset_config")
    def test_preset_calls_preset_config(
        self,
        mock_preset: MagicMock,
        mock_hw: MagicMock,
        mock_open: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        from yowo.config import InferenceConfig

        mock_preset.return_value = InferenceConfig()
        mock_hw.return_value = _hw()
        mock_open.return_value = iter([])
        engine = MagicMock()
        engine.stream.return_value = iter([])
        mock_engine_cls.return_value.__enter__ = MagicMock(return_value=engine)
        mock_engine_cls.return_value.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        runner.invoke(cli, ["detect", "video.mp4", "--preset"])

        mock_preset.assert_called_once()
        call_args = mock_preset.call_args
        assert call_args[0][1] == SourceCategory.VIDEO

    @patch("yowo.cli._main.InferenceEngine")
    @patch("yowo.cli._main.open_source")
    @patch("yowo.cli._main.get_hardware_profile")
    @patch("yowo.cli._main.preset_config")
    def test_preset_with_batch_override(
        self,
        mock_preset: MagicMock,
        mock_hw: MagicMock,
        mock_open: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        from yowo.config import InferenceConfig

        mock_preset.return_value = InferenceConfig(batch_size=8)
        mock_hw.return_value = _hw()
        mock_open.return_value = iter([])
        engine = MagicMock()
        engine.stream.return_value = iter([])
        mock_engine_cls.return_value.__enter__ = MagicMock(return_value=engine)
        mock_engine_cls.return_value.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        runner.invoke(cli, ["detect", "video.mp4", "--preset", "--batch", "8"])

        call_kwargs = mock_preset.call_args[1]
        assert call_kwargs["batch_size"] == 8
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_preset.py::TestPresetCLI -x -q`
Expected: FAIL (--preset flag not recognized or preset_config not called)

**Step 3: Write minimal implementation**

In `src/yowo/cli/_main.py`, add the `--preset` option and modify `detect_command`:

Add after the `--save-frames` option (line 50):

```python
@click.option("--preset", is_flag=True, default=False, help="Auto-tune config for device + source type")
```

Add `preset: bool` parameter to `detect_command` signature.

Replace the config-building block (lines 66-83) with:

```python
    from yowo.config import InferenceConfig
    from yowo.engine import InferenceEngine
    from yowo.io import open_source

    spec = _parse_model_spec(model)
    weights_path = Path(weights) if weights else spec.weights_path

    if preset:
        from yowo.config import classify_source, preset_config
        from yowo.hardware import get_hardware_profile

        hw = get_hardware_profile()
        source_cat = classify_source(source)

        # Collect explicit CLI overrides (non-default values)
        cli_overrides: dict[str, object] = {
            "model_family": spec.family,
            "model_size": spec.size,
            "weights_path": weights_path,
        }
        if backend != "auto":
            cli_overrides["backend"] = BackendType(backend)
        if device != "auto":
            cli_overrides["device"] = device
        if precision != "auto":
            cli_overrides["precision"] = Precision(precision)
        if batch != 1:
            cli_overrides["batch_size"] = batch
        if confidence != 0.25:
            cli_overrides["confidence_threshold"] = confidence
        if iou != 0.45:
            cli_overrides["iou_threshold"] = iou

        config = preset_config(hw, source_cat, **cli_overrides)
    else:
        config = InferenceConfig(
            model_family=spec.family,
            model_size=spec.size,
            weights_path=weights_path,
            batch_size=batch,
            confidence_threshold=confidence,
            iou_threshold=iou,
            backend=BackendType(backend) if backend != "auto" else None,
            device=device,
            precision=Precision(precision) if precision != "auto" else None,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_preset.py::TestPresetCLI -x -q`
Expected: PASS (3 tests)

**Step 5: Run full quality gates**

```bash
uv run ruff check src/ tests/ --quiet
uv run ruff format src/ tests/ --check --quiet
uv run pyright src/yowo/
uv run pytest tests/unit/ -x -q --tb=short
```

**Step 6: Commit**

```
feat(cli): add --preset flag for device+source auto-tuning

When --preset is passed, detect command auto-classifies the source
and hardware, builds an optimized InferenceConfig via preset_config(),
then applies any explicit CLI flags as overrides. Without --preset,
behavior is unchanged (full backward compatibility).
```

---

### Task 6: Run mandatory agents and final quality gates

**Files:**
- Review: `src/yowo/types.py`, `src/yowo/config.py`, `src/yowo/cli/_main.py`

**Step 1: Run api-contract-guardian**

Required because `types.py` and `config.py` `__all__` changed.

Run api-contract-guardian on the diff.

**Step 2: Run test-coverage-guardian**

Required for new feature implementation.

Run test-coverage-guardian on `config.py` (new functions).

**Step 3: Fix any issues raised by agents**

Address P0/P1 findings.

**Step 4: Run final quality gates**

```bash
uv run ruff check src/ tests/ --quiet
uv run ruff format src/ tests/ --check --quiet
uv run pyright src/yowo/
uv run pytest tests/unit/ -x -q --tb=short
```

All must pass before declaring done.

**Step 5: Commit any fixes from agent review**

```
fix(preset): address agent review findings

<describe specific fixes>
```
