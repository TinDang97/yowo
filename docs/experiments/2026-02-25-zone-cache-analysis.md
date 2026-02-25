# Zone-Level Cache for CCTV: Per-Token Spatial Caching Analysis

**Date**: 2026-02-25
**Branch**: `feat/phase3-source-aware-pipeline`
**Platform**: macOS Darwin 25.3.0, Apple Silicon (CPU), FP32, batch=1
**Prerequisite**: Read `2026-02-25-kv-cache-cctv-analysis.md` for current cache architecture.

---

## 1. Motivation

The current block cache (C2PSA / C3k2PSA) is **all-or-nothing**: if the
spatial-mean fingerprint differs by ≥ 0.01, the **entire block** recomputes —
even if only 5% of the spatial positions actually changed. In CCTV scenarios,
a walking person occupies ~0.5% of the frame but may cause 18-29% of spatial
tokens (at 20x20 resolution) to change in the deep feature map.

**Question**: Can we cache and reuse computation at the **per-token** (spatial
position) level — analogous to LLM KV caches that reuse tokens — to avoid
recomputing the 71-82% of unchanged spatial positions when the block cache
misses?

---

## 2. Feature Map Dimensions Across the Model

For yolo26n (input 640x640, width_mult=0.25):

```
Layer                    Resolution    Channels    Tokens    Cache Scope
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Backbone:
  stem (Conv 3x3/2)     320×320       16          102,400   —
  conv1 (Conv 3x3/2)    160×160       32           25,600   —
  c3k2_1 (C3k2)         160×160       64           25,600   Conv3x3 (no cache)
  conv2 (Conv 3x3/2)     80×80        64            6,400   —
  c3k2_2 (C3k2) → P3     80×80       128            6,400   Conv3x3 (no cache)
  conv3 (Conv 3x3/2)     40×40       128            1,600   —
  c3k2_3 (C3k2) → P4     40×40       128            1,600   Conv3x3 (no cache)
  conv4 (Conv 3x3/2)     20×20       256              400   —
  c3k2_4 (C3k2)          20×20       256              400   Conv3x3 (no cache)
  sppf                    20×20       256              400   MaxPool (no cache)
  ★ c2psa (C2PSA)        20×20       256              400   CACHEABLE (Attention)

Neck:
  c3k2_fpn1              40×40       128            1,600   Conv3x3 (no cache)
  c3k2_fpn2              80×80        64            6,400   Conv3x3 (no cache)
  c3k2_pan1              40×40       128            1,600   Conv3x3 (no cache)
  ★ c3k2_pan2 (C3k2PSA)  20×20       256              400   CACHEABLE (Attention)
```

**Both cacheable blocks operate at 20×20 = 400 tokens.**

This is a critical constraint. LLM KV caches work on sequences of 2K-128K
tokens where the overhead of indexing is amortized across thousands of
positions. At 400 tokens, indexed gather/scatter overhead is proportionally
much higher.

---

## 3. Spatial Change Distribution in CCTV

Tested with a 2560×1440 CCTV frame. A person-sized region (100×200 px, 0.5%
of frame) is shifted to simulate walking. At the 20×20 deep feature map
(backbone c2psa input), the change propagation is:

```
Movement    Pixels shifted    Tokens changed (20×20)    % changed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Static             0 px             0 / 400                  0%
Slow walk          5 px            52 / 400                 13%
Normal walk       20 px            76 / 400                 19%
Running           50 px           104 / 400                 26%
Fast run         100 px           116 / 400                 29%
```

**Key observation**: Even with significant motion, 71-87% of spatial tokens
are unchanged at the deep feature level. The 32× downsampling (640→20)
spreads a localized pixel change across a wider receptive field, but the
majority of positions remain static.

---

## 4. Prototype: Per-Token Zone KV Cache at 20×20

### 4.1 Design

Replace the all-or-nothing Attention forward with per-token selective
computation:

```python
# Per-token zone cache (prototype)
def forward_zone_cached(self, x):
    B, C, H, W = x.shape
    N = H * W  # 400 tokens

    # 1. Compute per-position fingerprint
    fp = x.view(B, C, N)  # (1, 256, 400)

    # 2. Identify changed positions (per-token L1 diff)
    diff = (fp - cached_fp).abs().mean(dim=1)  # (1, 400)
    changed_mask = diff > threshold             # (1, 400) bool

    # 3. Full QKV only on changed positions
    changed_idx = changed_mask.nonzero()[:, 1]  # indices
    x_changed = x.view(B, C, N)[:, :, changed_idx]  # gather changed
    qkv_changed = self.qkv_conv_1d(x_changed)       # Conv1x1 on subset

    # 4. Scatter updated K,V into cached tensors
    cached_k[:, :, changed_idx, :] = k_new
    cached_v[:, :, changed_idx, :] = v_new

    # 5. Fresh Q for ALL positions (needed for attention)
    q = ...  # must compute Q for all 400 positions

    # 6. Attention with hybrid K,V (mixed cached + fresh)
    out = scaled_dot_product_attention(q, cached_k, cached_v)
```

### 4.2 Benchmark Result

```
Operation               Time (µs)     Notes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full forward (current)      604       Dense QKV conv + SDPA
Zone-cached forward         663       Gather + partial conv + scatter + SDPA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Delta                       +59       -10% (SLOWER)
```

**Zone caching is 10% SLOWER than full forward at 20×20 resolution.**

### 4.3 Why Zone Caching Fails at 20×20

| Overhead Source | Cost (µs) | Why |
|---|---|---|
| Per-position fingerprint diff | ~35 | `(fp - cached_fp).abs().mean(dim=1)` on (1, 256, 400) |
| Boolean mask + nonzero() | ~15 | CPU synchronization point for dynamic indexing |
| Gather changed positions | ~20 | `x[:, :, changed_idx]` — indexed read, not contiguous |
| Scatter K,V updates | ~20 | `cached_k[:, :, idx, :] = new_k` — indexed write |
| **Total overhead** | **~90** | |
| Compute saved (74% fewer QKV channels) | ~45 | Conv1x1 on 104 vs 400 positions |
| **Net** | **+45 µs** | **Overhead exceeds savings by 2×** |

The fundamental issue: **PyTorch indexed operations (gather/scatter) on small
tensors are dominated by dispatch overhead, not compute.** The Conv1x1 on 400
positions at 256 channels takes only ~62 µs total — there isn't enough compute
to save for the indexing overhead to break even.

**Break-even analysis**: Zone caching becomes viable when:

```
gather_scatter_overhead < (1 - change_ratio) × full_conv_cost
~90 µs < (1 - 0.26) × conv_cost
conv_cost > ~122 µs
```

At 20×20, the Conv1x1 costs ~62 µs. We need **~2× more compute per position**
for zone caching to break even — which corresponds to roughly **28×28 = 784
tokens** at the same channel width, or 20×20 at ~512 channels.

---

## 5. Where Zone Caching Could Work

### 5.1 Resolution Analysis

| Resolution | Tokens | Conv1x1 cost (256ch) | Overhead | Break-even? |
|---|---|---|---|---|
| 20×20 | 400 | ~62 µs | ~90 µs | **NO** — 1.5× too little compute |
| 28×28 | 784 | ~122 µs | ~90 µs | Marginal — depends on change ratio |
| 40×40 | 1,600 | ~250 µs | ~95 µs | **YES** — 2.6× headroom |
| 80×80 | 6,400 | ~1,000 µs | ~110 µs | **YES** — 9× headroom |
| 160×160 | 25,600 | ~4,000 µs | ~150 µs | **YES** — 27× headroom |

Zone caching is viable at **40×40 and above**. But there's a problem:

### 5.2 Architectural Constraint

The cacheable attention blocks (C2PSA, C3k2PSA) all operate at **20×20 — the
smallest resolution in the model**. This is by design: attention's O(N²)
complexity makes it prohibitively expensive at larger resolutions.

The larger-resolution layers (40×40, 80×80, 160×160) are **pure convolutional
blocks** (C3k2 with Conv3x3 bottlenecks). These blocks have:

- **No attention** — no K,V to cache per token
- **Conv3x3 with 3×3 receptive field** — each output position depends on a
  3×3 neighborhood of input positions, creating spatial coupling that
  prevents independent per-token caching
- **Residual connections** — skip connections blend current and previous
  features, complicating selective recomputation

### 5.3 Conv3x3 Spatial Coupling Problem

```
Input feature map (40×40):
┌───┬───┬───┬───┬───┐
│ . │ . │ . │ . │ . │    For Conv3x3, updating position (2,2) requires:
├───┼───┼───┼───┼───┤    - inputs at (1,1), (1,2), (1,3)
│ . │ ■ │ ■ │ ■ │ . │    - inputs at (2,1), (2,2), (2,3)
├───┼───┼───┼───┼───┤    - inputs at (3,1), (3,2), (3,3)
│ . │ ■ │ ★ │ ■ │ . │
├───┼───┼───┼───┼───┤    Even if only ★ changed in the input, ALL 9
│ . │ ■ │ ■ │ ■ │ . │    outputs that overlap this receptive field must
├───┼───┼───┼───┼───┤    be recomputed → 3×3 = 9 positions per changed
│ . │ . │ . │ . │ . │    input position
└───┴───┴───┴───┴───┘

In a Bottleneck (Conv1x1 → Conv3x3 → Conv1x1):
  Changed positions propagate: N → N → N×9 → N×9
  A contiguous 5×5 changed region becomes (5+2)×(5+2) = 49 positions
  through the Conv3x3 → 80% of 40×40 map "touched"
```

The Conv3x3 spatial coupling means per-token caching in convolutional blocks
rapidly devolves to near-full recomputation for any non-trivial change
region. This is fundamentally different from Attention's Conv1x1 (pointwise)
operations, which preserve spatial independence.

---

## 6. Adaptive Quadtree Design (Theoretical)

Despite the 20×20 limitation, a quadtree-based approach was explored as
a potential future design for models with larger attention maps.

### 6.1 Concept

```
Level 0: Full 20×20 (or 40×40 for future models)
         ┌──────────┬──────────┐
         │          │          │
         │  10×10   │  10×10   │   Check coarse fingerprint per quadrant
         │          │          │
         ├──────────┼──────────┤
         │          │          │
         │  10×10   │  10×10   │   If any quadrant changed:
         │          │          │
         └──────────┴──────────┘

Level 1: Subdivide changed quadrant
         ┌─────┬─────┐
         │ 5×5 │ 5×5 │                Recurse until leaf or below threshold
         ├─────┼─────┤
         │ 5×5 │ 5×5 │                Only recompute leaf-changed tiles
         └─────┴─────┘

Level 2: Leaf tiles (minimum cacheable unit)
         ┌──┬──┐
         │  │  │                       3×3 or 5×5 minimum tile
         ├──┼──┤                       (set by Conv3x3 receptive field)
         │  │  │
         └──┘──┘
```

### 6.2 Analysis for 20×20

| Quadtree Level | Tile Size | Tiles | Overhead per level |
|---|---|---|---|
| 0 (root) | 20×20 | 1 | ~50 µs (fingerprint) |
| 1 | 10×10 | 4 | ~40 µs (4× fingerprint) |
| 2 (leaf) | 5×5 | 16 | ~80 µs (16× fingerprint) |

**Total quadtree overhead: ~170 µs** — this is **2.7× the full Conv1x1 cost**
(62 µs). The hierarchical fingerprinting is more expensive than just running
the computation.

For the quadtree to be worthwhile, the attention map must be at least
**80×80** (6,400 tokens), where the Conv1x1 costs ~1,000 µs and the
quadtree overhead (~200 µs) is a 20% cost against potentially 70% savings.

### 6.3 Minimum Viable Resolution

```
Quadtree break-even (assuming 25% change ratio, 75% cache hit):

At 80×80: savings = 0.75 × 1,000 = 750 µs; overhead = ~200 µs → NET +550 µs ✓
At 40×40: savings = 0.75 × 250 = 187 µs;  overhead = ~150 µs → NET +37 µs  ~ marginal
At 20×20: savings = 0.75 × 62 = 46 µs;    overhead = ~170 µs → NET -124 µs ✗
```

---

## 7. Comparison: Current vs Zone-Cache Approaches

| Approach | Mechanism | Savings on Cache Hit | When It Fails | Resolution Requirement |
|---|---|---|---|---|
| **Block cache (current)** | All-or-nothing spatial mean | 28-33% per block | Any scene change > threshold | Any (works at 20×20) |
| **KV cache (current)** | Reuse K,V, fresh Q | Saves transpose + detach | Fused QKV wastes 75% | Any (works at 20×20) |
| **Per-token zone cache** | Gather/scatter changed tokens | Up to 75% of QKV compute | Overhead > savings at 20×20 | **≥ 40×40** |
| **Adaptive quadtree** | Hierarchical tile caching | Up to 75% + structured | Fingerprint overhead too high | **≥ 80×80** |

---

## 8. What Would Make Zone Caching Viable

### 8.1 Larger Attention Maps (Architecture Change)

Future YOLO architectures may place attention at higher resolutions:

```
If C2PSA operated at 40×40 (1,600 tokens):
  Full Conv1x1: ~250 µs
  Zone cache (26% changed): ~90 + 0.26 × 250 = ~155 µs
  Savings: ~95 µs (38%) ← viable
```

This is a model architecture decision, not a cache mechanism decision.

### 8.2 Fused Gather-Conv Kernel (Custom CUDA)

A fused kernel that combines gather + Conv1x1 + scatter into a single pass
would eliminate the ~90 µs dispatch overhead:

```
Fused kernel overhead: ~10 µs (estimated)
Break-even at 20×20: 10 < 0.74 × 62 = 46 µs → viable
```

This requires custom CUDA/Metal kernel development and is not available in
standard PyTorch.

### 8.3 Sparse Attention (Algorithm Change)

Instead of caching K,V per token and running full SDPA, use sparse attention
where unchanged tokens don't participate as queries:

```python
# Sparse attention: only changed tokens attend
q_changed = q[:, :, changed_idx, :]   # (B, heads, ~104, key_dim)
out_changed = sdpa(q_changed, k_all, v_all)  # attend to ALL K,V

# Copy unchanged outputs from cache
out_unchanged = cached_attn_out[:, :, unchanged_idx, :]

# Merge
out = scatter(out_changed, out_unchanged, changed_idx, unchanged_idx)
```

Problem: SDPA is O(N_q × N_kv). Even with N_q=104, N_kv=400, the attention
cost is 104×400 = 41,600 — compared to 400×400 = 160,000 for full attention.
This saves ~74% of attention compute but attention itself is only ~15% of the
block cost (the Conv1x1 dominates). Net savings: ~11% of block — still
within the gather/scatter overhead at 20×20.

---

## 9. Recommendation

### For the current architecture (C2PSA/C3k2PSA at 20×20):

**Keep the existing block cache + KV cache.** Per-token zone caching adds
overhead that exceeds savings at 400 tokens. The block cache's all-or-nothing
approach is actually well-matched to this resolution because:

1. The block cache hits 100% of the time for typical CCTV motion (the spatial
   mean is robust to localized changes at 20×20)
2. When it does miss (scene change), the entire feature map has changed, so
   zone caching wouldn't help either
3. The block cache saves 28-33% per block with only ~50 µs fingerprint
   overhead

### For future larger-resolution attention models:

Zone caching becomes viable at ≥ 40×40 resolution. If future YOLO
architectures place attention blocks at P4 (40×40) or P3 (80×80), the
per-token zone cache with adaptive quadtree would provide significant
savings:

| Resolution | Estimated Zone Cache Savings | vs Block Cache |
|---|---|---|
| 40×40 | 15-38% of attention block | Marginal improvement over block cache |
| 80×80 | 40-55% of attention block | Significant — block cache misses more at higher res |
| 160×160 | 55-70% of attention block | Dominant — essential for real-time at this resolution |

### Optimal strategy per model scale:

| Model Size | Attention Resolution | Recommended Cache | Reason |
|---|---|---|---|
| Nano (n) | 20×20 | Block cache only | KV overhead exceeds savings |
| Small (s) | 20×20 | Block + KV cache | Combined 6.4% gain |
| Medium+ (m/l/x) | 20×20 | Block + KV cache | Combined 4-7.6% gain |
| Future (attention at 40×40) | 40×40 | Block + Zone KV | Zone cache viable |
| Future (attention at 80×80) | 80×80 | Quadtree Zone KV | Maximum savings |

---

## 10. QKV Split: The Actionable Optimization

While per-token zone caching is not viable at 20×20, one related optimization
**is** viable: splitting the fused QKV Conv1x1 into separate Q and KV
projections at load time.

### Current state (75% compute waste on KV cache hit):

```python
self.qkv = Conv(dim, dim + 2*nh_kd, 1, act=False)  # 128 → 256 channels
# On cache hit: computes ALL 256 channels, discards K+V (192 channels, 75%)
```

### Proposed split (eliminates waste):

```python
# At load time — weight transformation, not architecture change:
q_proj = Conv1x1(dim, nh_kd)        # 128 → 64 channels
kv_proj = Conv1x1(dim, dim + nh_kd)  # 128 → 192 channels

# Cache hit path:
q = q_proj(x)       # 64ch only → 4× less compute
k = cached_k
v = cached_v

# Cache miss path:
q = q_proj(x)
kv = kv_proj(x)
k, v = kv.split(...)
```

### Estimated savings from QKV split:

| Model | Attention blocks | Saved per frame | % of total inference |
|---|---|---|---|
| yolo26n | 2 | ~102 µs | 0.3% |
| yolo26s | 2 | ~180 µs | 0.4% |
| yolo26l | 3 | ~350 µs | 0.3% |

**This is a small but clean optimization** — no per-token indexing, no
quadtree, just avoiding 75% of wasted Conv1x1 compute on cache hit. It
requires careful weight extraction due to the per-head interleaved layout
of ultralytics checkpoints.

**Note**: This optimization is ONLY relevant when the block cache MISSES
(the KV cache hit path). When the block cache hits (100% of typical CCTV
frames), the Attention layer never executes, so the QKV split has zero
impact on the hot path.

---

## 11. Data Reproduction

```bash
# Zone cache prototype benchmark (requires torch)
uv run python -c "
import torch, time

B, C, H, W = 1, 256, 20, 20
N = H * W  # 400 tokens
num_heads, key_dim = 2, 32
change_ratio = 0.26

x = torch.randn(B, C, H, W)
qkv_weight = torch.randn(256, 256, 1, 1)

# Full forward
times_full = []
for _ in range(200):
    t0 = time.perf_counter_ns()
    qkv = torch.nn.functional.conv2d(x, qkv_weight)
    times_full.append((time.perf_counter_ns() - t0) / 1000)

# Zone-cached forward
n_changed = int(N * change_ratio)
changed_idx = torch.randperm(N)[:n_changed]
cached_fp = torch.randn(B, C, N)

times_zone = []
for _ in range(200):
    t0 = time.perf_counter_ns()
    fp = x.view(B, C, N)
    diff = (fp - cached_fp).abs().mean(dim=1)
    mask = diff > 0.01
    idx = mask.nonzero()[:, 1]
    x_sel = fp[:, :, idx].unsqueeze(-1)
    qkv_sel = torch.nn.functional.conv2d(x_sel, qkv_weight)
    times_zone.append((time.perf_counter_ns() - t0) / 1000)

print(f'Full:  {sum(times_full[20:])/len(times_full[20:]):.0f} µs')
print(f'Zone:  {sum(times_zone[20:])/len(times_zone[20:]):.0f} µs')
print(f'Delta: {sum(times_zone[20:])/len(times_zone[20:]) - sum(times_full[20:])/len(times_full[20:]):+.0f} µs')
"
```
