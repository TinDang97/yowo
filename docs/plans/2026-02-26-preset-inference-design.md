# Preset Inference: Device + Source Auto-Tuning

**Date**: 2026-02-26
**Status**: Approved

## Goal

Automatically tune pipeline/caching knobs based on detected hardware and input source type. Exposed as an opt-in `preset_config()` factory (Python API) and `--preset` CLI flag. Does not replace existing defaults — composes on top.

## Source Categories

```python
class SourceCategory(StrEnum):
    IMAGE = "image"         # total_frames == 1
    VIDEO = "video"         # total_frames > 1, is_live == False
    LIVE_STREAM = "live"    # is_live == True (RTSP, webcam)
```

`classify_source(source: str | Path) -> SourceCategory` inspects the source string using the same dispatch rules as `open_source()` without opening the source.

## Device Categories

```python
class DeviceCategory(StrEnum):
    CUDA_HIGH = "cuda_high"         # NVIDIA GPU, VRAM >= 8GB
    CUDA_LOW = "cuda_low"           # NVIDIA GPU, VRAM < 8GB
    JETSON = "jetson"               # NVIDIA Jetson SoC
    APPLE_SILICON = "apple_silicon" # AARCH64 + CoreML EP available
    CPU_X86 = "cpu_x86"            # x86_64 CPU
    CPU_ARM = "cpu_arm"            # AARCH64 CPU without CoreML
```

`classify_device(hw: HardwareProfile) -> DeviceCategory` evaluates in priority order:

1. `hw.is_jetson` -> JETSON
2. `hw.has_nvidia_gpu` and VRAM >= 8192MB -> CUDA_HIGH
3. `hw.has_nvidia_gpu` -> CUDA_LOW
4. `onnxruntime_has_coreml` and `cpu_arch == AARCH64` -> APPLE_SILICON
5. `cpu_arch == X86_64` -> CPU_X86
6. fallback -> CPU_ARM

## Preset Lookup Table

Only pipeline/caching knobs are tuned. `backend`, `device`, and `precision` are excluded (handled by `select_backend()`).

| Device | Image | Video | Live Stream |
|--------|-------|-------|-------------|
| CUDA_HIGH | batch=1, prefetch=False | batch=4, cache=True, prefetch=True | batch=1, kv_cache=True, frame_drop=LATEST, max_queue=4 |
| CUDA_LOW | batch=1, prefetch=False | batch=2, prefetch=True | batch=1, kv_cache=True, frame_drop=LATEST, max_queue=2 |
| JETSON | batch=1, prefetch=False | batch=1, prefetch=True | batch=1, kv_cache=True, frame_drop=LATEST, max_queue=2 |
| APPLE_SILICON | batch=1, prefetch=False | batch=2, cache=True, prefetch=True | batch=1, kv_cache=True, frame_drop=LATEST, max_queue=2 |
| CPU_X86 | batch=1, prefetch=False | batch=2, prefetch=True | batch=1, frame_drop=LATEST, max_queue=2 |
| CPU_ARM | batch=1, prefetch=False | batch=1, prefetch=True | batch=1, frame_drop=LATEST, max_queue=2 |

Rationale:
- **Image**: no batching, no prefetch (single frame)
- **Video**: batch scales with device power; cache on high-memory devices; prefetch always on
- **Live**: batch=1 (latency-critical); kv_cache on GPU/Apple Silicon only (CPU KV is 2-8x slower); frame_drop=LATEST
- **CPU live**: no kv_cache (performance regression)

## Public API

```python
def preset_config(
    hw: HardwareProfile,
    source_type: SourceCategory,
    **overrides: Any,
) -> InferenceConfig:
```

`**overrides` accept any `InferenceConfig` field name. Explicit values replace preset values. Unknown keys raise `ConfigError`.

## CLI Integration

```
yowo detect rtsp://cam1 --model yolo26n --preset
yowo detect video.mp4 --model yolo26n --preset --batch 8
```

When `--preset`:
1. `classify_source(SOURCE)` determines `SourceCategory`
2. `get_hardware_profile()` gets device info
3. `preset_config(hw, source_cat)` builds base config
4. Explicit CLI flags override preset values
5. Merged config passed to `InferenceEngine`

Without `--preset`: behavior unchanged.

## File Changes

| File | Change |
|------|--------|
| `src/yowo/types.py` | Add `SourceCategory`, `DeviceCategory` enums |
| `src/yowo/config.py` | Add `_PresetOverrides`, `_PRESET_TABLE`, `classify_source()`, `classify_device()`, `preset_config()` |
| `src/yowo/cli/_main.py` | Add `--preset` flag, wire up `preset_config()` |
| `tests/unit/test_preset.py` | ~15-20 test cases |

## Not In Scope

- Does not modify `select_backend()` or priority chain
- Does not add new dependencies
- Does not change existing `InferenceConfig` defaults
- Does not affect behavior when `--preset` is not used
