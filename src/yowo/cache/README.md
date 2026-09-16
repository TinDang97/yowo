# yowo.cache — Feature Map Caching for Sequential Inference

Skip backbone + neck on similar consecutive frames. **Measured 2026-09-16 — the
saving and the blind spot are one number seen from two sides, so both are stated:**

| device | threshold | cache off | cache on | saving | object it cannot see |
|---|---|---|---|---|---|
| cpu | 0.01 (default) | 34.50 ms | 40.22 ms | −16.6% | 40x40 px |
| cpu | 0.05 | 33.17 ms | 18.55 ms | +44.1% | 92x92 px |
| cpu | 0.10 | 36.02 ms | 17.14 ms | +52.4% | 130x130 px |
| mps | 0.01 (default) | 6.27 ms | 19.52 ms | −211.3% | 40x40 px |
| mps | 0.10 | 6.38 ms | 11.75 ms | −84.1% | 130x130 px |

yolo11n, fixed camera, 40 frames, median. A hit copies ~6.4 MB of neck features
host→device, so a faster device loses harder — **on Apple Silicon the cache costs
time at every threshold**, which is why it is no longer on in that preset. The
earlier "60–85%" claim had no experiment behind it and is not reachable at any
threshold measured. Full grid and method:
[feature cache recall and savings](../../../docs/experiments/2026-09-16-feature-cache-recall-and-savings.md).

**What it cannot see.** The fingerprint is an 8x8 grid of per-channel means
compared by the worst cell, so a change confined to one cell is measured against
that cell. At the default threshold an object of up to **40x40 px** at realistic
contrast (22x22 px at maximum contrast) can appear without the cache noticing.
Raising the threshold to buy hit rate raises that bound in step.

## Architecture

```
FeatureCache (coordinator)
├── FeatureStore (bounded cache, in-memory or mmap)
└── spatial_fingerprint() + fingerprint_distance() (8x8 grid, worst cell)
```

```
src/yowo/cache/
├── __init__.py        # FeatureCache — public API
├── _store.py          # FeatureStore — in-memory / mmap dual-mode storage
├── _similarity.py     # spatial_fingerprint() / fingerprint_distance()
└── README.md
```

## Quick Start

### In-Memory (default)

```python
from yowo.cache import FeatureCache

cache = FeatureCache()  # bounded in-memory dict

# In inference loop:
cached = cache.check_and_load(source_id, preprocessed_tensor, device)
if cached is not None:
    output = model.forward_head(cached)  # head-only; see the table above for what that saves
else:
    output = model(preprocessed_tensor)  # full inference
    cache.update(source_id, preprocessed_tensor, neck_features)
```

### mmap-Backed (for many cameras / limited RAM)

```python
from pathlib import Path
from yowo.cache import FeatureCache

cache = FeatureCache(cache_dir=Path("/tmp/yowo_cache"))
```

OS page cache handles memory pressure: active entries stay in RAM, cold entries are paged to disk.

### Via InferenceEngine

```python
from yowo.engine import InferenceEngine

# In-memory cache
engine = InferenceEngine(spec, cache=True)

# mmap-backed cache
engine = InferenceEngine(spec, cache_dir=Path("/tmp/yowo_cache"))
```

## How It Works

1. **Frame arrives** → `check_and_load(source_id, tensor, device)`
2. **Compare** current tensor vs last tensor for this source (L1 mean absolute diff)
3. **If similar** (diff < threshold) → load cached neck features → return as PyTorch tensors
4. **If different** → return `None` → caller runs full inference → `update()` stores new features

### Cache Hit Path (head-only)

```
preprocessed_tensor → similarity check → load cached P3/P4/P5 → detection head → output
                      ~0.01ms             ~0.1ms (mem)           ~5ms
```

### Cache Miss Path (full inference + store)

```
preprocessed_tensor → backbone → neck → detection head → output
                      ~15ms      ~8ms    ~5ms
                                  ↓
                            store P3/P4/P5 in cache
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.01` | Max L1 diff to reuse cache. Lower = stricter. |
| `max_entries` | `32` | Max cached sources. FIFO eviction at capacity. |
| `cache_dir` | `None` | Set for mmap persistence. `None` = in-memory only. |

## Design Decisions

| Decision | Rationale |
|---|---|
| In-memory default | Zero disk I/O overhead for most users |
| No `flush()` on mmap writes | OS page cache handles lazy writeback; this is a transient cache, not a database |
| SHA-256 hash for directory names | Collision-free filesystem keys for URLs/paths as source_ids |
| `_last_inputs` bounded to `max_entries` | Prevents unbounded RAM growth in multi-camera deployments |
| Partial write cleanup | `shutil.rmtree` on write failure prevents orphaned directories |
| FIFO eviction (not LRU) | Simplicity; in sequential video, oldest source is least likely to be revisited |

## Integration with PyTorch Backend

The `PyTorchBackend` registers a forward hook on `model.neck` to capture intermediate features without modifying the model's forward path:

```
model.neck.register_forward_hook(_capture_neck)
          ↓
infer() → cache miss → full model(t) → hook captures neck output → cache.update()
infer() → cache hit  → model.forward_head(cached) → skip backbone + neck
```

The engine sets `source_id` per frame via `backend.set_source_id()` before calling `infer()`.
