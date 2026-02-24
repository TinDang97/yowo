# arch — Native YOLO Architecture

Native PyTorch implementations of YOLO11 and YOLO26, reproduced exactly from the ultralytics YAML specifications — no ultralytics dependency. Covers backbone, FPN-PAN neck, detection head, scaling, weight loading, and inference optimizations.

---

## Public API

```python
from yowo.arch import build_model, load_weights
from yowo.types import ModelFamily, ModelSize

# Build an untrained model
model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
model = build_model(ModelFamily.YOLO26, ModelSize.MEDIUM)

# Custom class count (fine-tuned weights)
model = build_model(ModelFamily.YOLO11, ModelSize.SMALL, num_classes=7)

# Load ultralytics .pt checkpoint weights
load_weights(model, "yolo11n.pt")

# Fuse Conv+BN layers and set eval mode (required before inference)
model = model.fuse().eval()

# Run inference
import torch
x = torch.zeros(1, 3, 640, 640)
with torch.inference_mode():
    output = model(x)

# Optional: compile for CUDA (do NOT use on CPU — 30–43% regression)
model.compile_for_inference(mode="reduce-overhead")
```

---

## Module Structure

```
arch/
├── __init__.py      — build_model(), load_weights(), public re-exports
├── _config.py       — ModelConfig dataclass + get_config() + scaling helpers
├── _blocks.py       — Conv, DWConv, Bottleneck, C2f, C3, C3k, C3k2, SPPF, Concat
├── _attention.py    — Attention, PSABlock, C2PSA, C3k2PSA
├── _neck.py         — FPNPANNeck (layers 11–22)
├── _heads.py        — DFL, Detect (layers 23+), make_anchors, dist2bbox
└── _yolo.py         — Backbone (layers 0–10), YOLOModel, fuse(), compile_for_inference()
```

| File | Exports | Role |
|------|---------|------|
| `__init__.py` | `build_model`, `load_weights`, `ModelConfig`, `YOLOModel`, `get_config`, `scale_channels`, `scale_repeats` | Public surface |
| `_config.py` | `ModelConfig`, `get_config`, `scale_channels`, `scale_repeats` | Architecture hyperparameters per family × size |
| `_blocks.py` | `Conv`, `DWConv`, `Bottleneck`, `C2f`, `C3`, `C3k`, `C3k2`, `SPPF`, `Concat` | Core CSP building blocks |
| `_attention.py` | `Attention`, `PSABlock`, `C2PSA`, `C3k2PSA` | Self-attention blocks (YOLO26 neck, both-family backbone) |
| `_neck.py` | `FPNPANNeck` | Feature Pyramid + Path Aggregation neck |
| `_heads.py` | `DFL`, `Detect`, `make_anchors`, `dist2bbox` | Detection head, DFL decode, anchor generation |
| `_yolo.py` | `Backbone`, `YOLOModel` | Full model assembly |
| `_weights.py` | `load_weights` | Checkpoint key remapping, shape validation |

---

## Architecture Overview

```
Input (B, 3, 640, 640)
        │
        ▼
   ┌─────────────────────────────────┐
   │         BACKBONE (layers 0–10) │
   │  stem → conv1 → c3k2 → conv2   │
   │  c3k2 → conv3 → c3k2 → conv4   │
   │  c3k2 → sppf → c2psa           │
   └────────┬────────┬──────────────┘
            │        │
            P3       │
           (c4)      │
            │        P4       P5 (c5)
            │       (c4)       │
            ▼        ▼         ▼
   ┌─────────────────────────────────┐
   │     FPN-PAN NECK (layers 11–22) │
   │  FPN: P5↑→concat(P4)→C3k2→P4'  │
   │       P4'↑→concat(P3)→C3k2→P3' │
   │  PAN: P3'↓→concat(P4')→C3k2    │
   │       P4''↓→concat(P5)→C3k2PSA │
   └────────┬────────┬──────────────┘
            │        │
           P3'      P4''     P5''
            │        │        │
            ▼        ▼        ▼
   ┌─────────────────────────────────┐
   │         DETECT HEAD            │
   │  cv2: box regression (DFL)     │
   │  cv3: class scores             │
   └─────────────────────────────────┘
            │
            ▼
   YOLO11: (B, 4+nc, 8400)         — raw, needs NMS
   YOLO26: (B, max_det=300, 6)     — NMS-free top-k
```

For a 640×640 input: 80×80 (P3) + 40×40 (P4) + 20×20 (P5) = **8400 total anchors**.

---

## Backbone Layers (layers 0–10)

| Layer | Name | Op | Input → Output channels | Stride | Notes |
|-------|------|----|------------------------|--------|-------|
| 0 | `stem` | `Conv(3×3)` | 3 → c1 | /2 | P1/2 |
| 1 | `conv1` | `Conv(3×3)` | c1 → c2 | /2 | P2/4 |
| 2 | `c3k2_1` | `C3k2` | c2 → c3 | 1 | `c3k=backbone_c3k`, e=0.25 |
| 3 | `conv2` | `Conv(3×3)` | c3 → c3 | /2 | P3/8 |
| 4 | `c3k2_2` | `C3k2` | c3 → c4 | 1 | `c3k=backbone_c3k`, e=0.25 |
| 5 | `conv3` | `Conv(3×3)` | c4 → c4 | /2 | P4/16 |
| 6 | `c3k2_3` | `C3k2` | c4 → c4 | 1 | `c3k=True` always |
| 7 | `conv4` | `Conv(3×3)` | c4 → c5 | /2 | P5/32 |
| 8 | `c3k2_4` | `C3k2` | c5 → c5 | 1 | `c3k=True` always |
| 9 | `sppf` | `SPPF(k=5)` | c5 → c5 | 1 | `shortcut=sppf_shortcut` |
| 10 | `c2psa` | `C2PSA` | c5 → c5 | 1 | Self-attention block |

P3 feature map (layer 4 output, c4 channels) and P4 (layer 6 output, c4 channels) and P5 (layer 10 output, c5 channels) are forwarded to the neck.

---

## Neck Layers (layers 11–22)

### FPN top-down path

| Layer | Name | Op | Input | Output |
|-------|------|----|-------|--------|
| 11 | `up1` | `Upsample(×2)` | P5 (c5) | P5↑ |
| 12 | `concat1` | `Concat` | P5↑ + P4 | (c5+c4) |
| 13 | `c3k2_fpn1` | `C3k2` | (c5+c4) → c4 | P4' |
| 14 | `up2` | `Upsample(×2)` | P4' (c4) | P4'↑ |
| 15 | `concat2` | `Concat` | P4'↑ + P3 | (c4+c4) |
| 16 | `c3k2_fpn2` | `C3k2` | (c4+c4) → c3 | P3' |

### PAN bottom-up path

| Layer | Name | Op | Input | Output |
|-------|------|----|-------|--------|
| 17 | `down1` | `Conv(3×3, s=2)` | P3' (c3) | P3'↓ |
| 18 | `concat3` | `Concat` | P3'↓ + P4' | (c3+c4) |
| 19 | `c3k2_pan1` | `C3k2` | (c3+c4) → c4 | P4'' |
| 20 | `down2` | `Conv(3×3, s=2)` | P4'' (c4) | P4''↓ |
| 21 | `concat4` | `Concat` | P4''↓ + P5 | (c4+c5) |
| 22 | `c3k2_pan2` | `C3k2` or `C3k2PSA` | (c4+c5) → c5 | P5'' |

> Layer 22 is **`C3k2PSA`** for YOLO26 (Bottleneck + PSABlock), **`C3k2(c3k=True)`** for YOLO11.

Head inputs: **[P3', P4'', P5'']** at channels [c3, c4, c5].

---

## Detection Head

`Detect` in `_heads.py` implements decoupled box + class branches:

```
For each scale (P3', P4'', P5''):
  cv2[i]:  Conv(3) → Conv(3) → Conv2d(1)   → (B, 4*reg_max, H*W)   box
  cv3[i]:  DWConv+Conv → DWConv+Conv → Conv2d(1) → (B, nc, H*W)   class

Concatenated: (B, 4*reg_max + nc, total_anchors)
  ↓
DFL decode: (B, 4, total_anchors) — expected distance from anchor centre
  ↓
dist2bbox + strides → pixel-space [cx,cy,w,h] (YOLO11) or [x1,y1,x2,y2] (YOLO26)
  ↓
sigmoid(class_logits)
  ↓
YOLO11: (B, 4+nc, 8400)           — pass to NMS postprocess
YOLO26: top-k → (B, 300, 6)      — [x1,y1,x2,y2,conf,class_id], NMS-free
```

### DFL (Distribution Focal Loss decode)

```python
# (B, 4*reg_max, N) → (B, 4, N) via softmax-weighted sum over bins [0..reg_max-1]
t = x.view(b, 4, reg_max, a).softmax(2)
return (t * weight[None, None, :, None]).sum(2)
```

Implemented as `register_buffer("weight", arange(reg_max))` — no learned parameters, no Conv2d overhead.

---

## Scaling Table

All 10 variants share the same topology; only channel widths and layer repeat counts differ.

| Size | depth\_mult | width\_mult | max\_channels | c1 | c2 | c3 | c4 | c5 |
|------|------------|------------|--------------|-----|-----|-----|-----|-----|
| nano | 0.50 | 0.25 | 1024 | 16 | 32 | 64 | 128 | 256 |
| small | 0.50 | 0.50 | 1024 | 32 | 64 | 128 | 256 | 512 |
| medium | 0.50 | 1.00 | 512 | 64 | 128 | 256 | 512 | 512 |
| large | 1.00 | 1.00 | 512 | 64 | 128 | 256 | 512 | 512 |
| xlarge | 1.00 | 1.50 | 512 | 96 | 192 | 384 | 512 | 512 |

Channel formula: `min(round(base * width_mult), max_channels)`

Repeat formula: `max(round(n * depth_mult), 1)` — n=2 → 1 (nano/small), 2 (medium/large/xlarge)

---

## Family Differences (YOLO11 vs YOLO26)

| Property | YOLO11 | YOLO26 |
|----------|--------|--------|
| `reg_max` | 16 | 1 |
| DFL | Softmax over 16 bins | `nn.Identity()` (pass-through) |
| `end2end` | False | True |
| Head output | `(B, 84, 8400)` raw → NMS | `(B, 300, 6)` NMS-free top-k |
| `sppf_shortcut` | False | True (residual in SPPF) |
| `backbone_c3k` | False for n/s; True for m/l/x | False for n/s; True for m/l/x |
| `neck_c3k` | False for n/s; True for m/l/x | **True for ALL sizes** (`neck_c3k_force`) |
| Neck layer 22 | `C3k2(c3k=True)` | `C3k2PSA` |
| End-to-end heads | — | `one2one_cv2`, `one2one_cv3` (deepcopy of cv2/cv3) |

---

## C3k2 / C3k / Bottleneck Hierarchy

```
Bottleneck         — standard residual block (shortcut=True adds skip connection)
C2f                — CSP with two-branch split: hidden + n × Bottleneck
C3                 — CSP with full-channel branches: cv1 → n × Bottleneck + cv2 → cat → cv3
C3k   (: C3)       — C3 but uses C3k-sized Bottleneck (k=3 convs)
C3k2  (: C2f)      — C2f but with switchable Bottleneck type:
                     c3k=False → plain Bottleneck (n/s backbone layers 2, 4)
                     c3k=True  → C3k Bottleneck (m/l/x backbone; all neck layers)
```

`c3k=True` adds one more 3×3 conv per bottleneck, increasing representational capacity for larger sizes.

---

## BatchNorm Convention

All `Conv` blocks use `eps=1e-3, momentum=0.03` (ultralytics convention, not PyTorch default `1e-5`). This is critical for correct weight loading from `.pt` checkpoints — incorrect values cause systematic normalisation errors.

---

## Performance Notes

Optimizations applied in the hot path (see `docs/experiments/2026-02-24-arch-inference-optimization-benchmark.md`):

| Optimization | File | Detail |
|-------------|------|--------|
| DFL register\_buffer | `_heads.py` `DFL` | `arange(c1)` buffer + weighted sum, no Conv2d |
| In-place sigmoid | `_heads.py` `Detect._decode` | `cls_cat.sigmoid_()` — saves ~2.7 MB alloc/frame |
| Stride init flag | `_heads.py` `Detect._decode` | `_strides_initialized: bool` — no GPU `.sum()` per frame |
| Anchor cache pre-init | `_heads.py` `Detect.__init__` | `_anchor_cache_key = None` — no `hasattr` per frame |
| `compile_for_inference()` | `_yolo.py` `YOLOModel` | `torch.compile(fullgraph=False)` — **CUDA only; CPU is 30–43% slower** |

---

## Weight Loading

`load_weights(model, path)` handles ultralytics `.pt` checkpoints:

1. **Key remapping**: ultralytics checkpoint keys (`model.0.conv.weight`) → yowo keys (`backbone.stem.conv.weight`) via `_LAYER_MAP` table in `_weights.py`.
2. **Skipped keys**: parameterless layers (activations, upsample) produce no state_dict entries and are silently skipped.
3. **Non-critical missing keys**: `stride`, `anchor_*`, and `dfl.weight` buffers are computed at runtime — not present in checkpoints and not warnings.
4. **Shape validation**: mismatched shapes raise `RuntimeError` before `load_state_dict`.
5. **Detach**: EMA checkpoint tensors are detached before loading to avoid non-leaf parameter warnings.
6. **DFL weight key**: old checkpoints have `head.dfl.conv.weight`; the remapped key is excluded from the unexpected-key warning — expected and harmless.

```
Checkpoint key (ultralytics)        yowo key
model.0.conv.weight              →  backbone.stem.conv.weight
model.0.bn.weight                →  backbone.stem.bn.weight
model.23.cv2.0.0.conv.weight     →  head.cv2.0.0.conv.weight
model.23.dfl.conv.weight         →  (skipped — replaced by register_buffer)
```

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `types.py` | `ModelFamily`, `ModelSize` |
| Downstream | `backends/_pytorch.py` | calls `build_model()`, `load_weights()`, `model.fuse()`, `model.compile_for_inference()` |
| Downstream | `export/_exporter.py` | calls `build_model()`, `load_weights()`, then `torch.onnx.export(model)` |
