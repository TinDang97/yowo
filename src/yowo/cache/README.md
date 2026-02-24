# yowo.cache — Feature Map Caching for Sequential Inference

Skip backbone + neck computation when consecutive frames are similar. Only the lightweight detection head runs on cache hits, reducing per-frame compute by 60-85%.

## Architecture

```
FeatureCache (coordinator)
├── FeatureStore (bounded cache, in-memory or mmap)
└── frame_similarity() (L1 mean pixel diff)
```

```
src/yowo/cache/
├── __init__.py        # FeatureCache — public API
├── _store.py          # FeatureStore — in-memory / mmap dual-mode storage
├── _similarity.py     # frame_similarity() — lightweight pixel diff
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
    output = model.forward_head(cached)  # head-only (~15% of total compute)
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
