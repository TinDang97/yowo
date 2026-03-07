# Phase 4: OBB Detection - Research

**Researched:** 2026-03-08
**Domain:** Oriented Bounding Box Detection (YOLO11-OBB architecture, rotated NMS, weight mapping)
**Confidence:** HIGH

## Summary

YOLO11-OBB is an architectural extension of standard YOLO11 detection: the backbone and FPN/PAN neck are **identical** (layers 0-22, same weight keys), and only the detection head is replaced with an OBB head. The OBB head adds a third branch (`cv4`) alongside the existing box (`cv2`) and class (`cv3`) branches to predict rotation angles. The angle output is `ne=1` channel per anchor (default: 1 extra parameter = one angle per box), encoded as `(sigmoid(raw) - 0.25) * pi`, which gives angles in the range `[-pi/4, 3pi/4]`.

The OBB head forward pass produces `(B, 4+nc+1, total_anchors)` at inference: 4 box coords + nc class logits + 1 rotation angle. Rotated NMS uses **probabilistic IoU (probiou)** based on Gaussian covariance matrices derived from box dimensions and angle, operating on `(cx, cy, w, h, theta)` format (xywhr). The `dist2rbox` function decodes DFL box distances + angle into rotated box coordinates. Since the backbone and neck layer numbering is identical to standard YOLO11 detection (layers 0-22, head at layer 23), OBB weights map with the same `_LAYER_MAP` except the head key maps to an `OBBHead` module instead of `Detect`.

**Primary recommendation:** Implement `OBBHead(nn.Module)` in `arch/_heads.py`, add `OBBModel` in `arch/_yolo.py` (reusing existing `Backbone` + `FPNPANNeck`), add `dist2rbox` + `probiou_nms` in `postprocess/_obb_nms.py`, add `OBBDetection` type in `types.py`, add `OBBEngine` in `obb_engine.py` following the same `BaseEngine` pattern as `ClassificationEngine`, register OBB variants in the model registry, add `detect-obb` CLI command, and add `spec.task == "obb"` branch to `export_model()`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| OBB-01 | OBB detection head produces oriented bounding boxes with rotation angle | OBBHead with cv4 angle branch; dist2rbox decoding; output includes `angle` field |
| OBB-02 | OBB postprocessing includes rotation-aware NMS | probiou NMS using Gaussian covariance; operates on xywhr format; Fast-NMS algorithm |
| OBB-03 | OBB model variants match ultralytics OBB architecture for yolo11 family | yolo11-obb.yaml confirms identical backbone+neck, same 5 scales n/s/m/l/x |
| OBB-04 | OBB weights load correctly from ultralytics-trained checkpoints | Layer map layers 0-22 identical; only head key changes to OBBHead |
| OBB-05 | CLI supports `yowo detect-obb SOURCE --model yolo11n-obb` | New `detect-obb` Click command; `parse_model_name` needs `-obb` suffix support |
| OBB-06 | OBB models export to ONNX and TensorRT correctly | `export_model()` needs `spec.task == "obb"` branch; ONNX-compatible forward |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch (torch) | >=2.0 | OBBHead forward pass, dist2rbox, angle encoding | Already in project; all arch uses torch |
| NumPy | >=1.24 | OBB postprocessing arrays (confidence filtering, coordinate transforms) | Already used in `_nms.py` |
| OpenCV (cv2) | >=4.8 | cv2.dnn.NMSBoxes (axis-aligned NMS) — NOT used for rotated NMS | Already used; rotated NMS uses probiou instead |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ONNX + onnxruntime | >=1.16 | OBB export validation | Required for OBB-06 |
| TensorRT (tensorrt) | >=8.6 | OBB TRT export | Required for OBB-06 on NVIDIA |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure Python probiou | torchvision.ops.box_iou for rotated | torchvision rotated NMS is not shipped in all builds; probiou is dependency-free and more accurate for OBB |
| cv2.dnn.NMSBoxes | Pure Python NMS | cv2 NMSBoxes does not support rotated boxes at all — must use probiou approach |

**Installation:** No new dependencies required. All needed packages already in project.

## Architecture Patterns

### Recommended Project Structure

```
src/yowo/
├── arch/
│   ├── _heads.py          # ADD: OBBHead class (next to Detect, Classify)
│   └── _yolo.py           # ADD: OBBModel class (next to YOLOModel, ClassifyModel)
│   └── __init__.py        # ADD: build_obb_model(), OBBModel exports
├── postprocess/
│   ├── _obb_nms.py        # NEW: dist2rbox, probiou NMS, postprocess_obb()
│   └── __init__.py        # ADD: exports for postprocess_obb
├── models/
│   └── _registry.py       # ADD: OBB registry + _OBB_REGISTRY dict, get_obb()
├── types.py               # ADD: OBBBox dataclass (x,y,w,h,angle,confidence,class_id)
├── obb_engine.py          # NEW: OBBEngine(BaseEngine) — mirrors classify_engine.py
├── cli/
│   └── _main.py           # ADD: detect-obb command + -obb suffix in parse_model_name
└── export/
    └── _exporter.py       # ADD: task == "obb" branch in export_model()
```

### Pattern 1: OBBHead — Extends Detect with angle branch

**What:** OBBHead subclasses the existing `Detect` head, adding `cv4` convolution branch for angle prediction. Output is `(B, 4+nc+ne, total_anchors)` where `ne=1` is the single angle channel.

**When to use:** Only for OBB task. Standard detection still uses `Detect`.

**Key architectural fact (from yolo11-obb.yaml, verified against ultralytics source):** The OBB backbone and neck are **identical** to standard YOLO11 detection. Layers 0-22 have the same layer numbers and parameters. Only layer 23 is `OBB` instead of `Detect`. This means `_LAYER_MAP` for weight loading can be reused for layers 0-22, only needing an OBB-specific head mapping.

**Example:**
```python
# Source: ultralytics/nn/modules/head.py (verified 2026-03-08)
class OBBHead(Detect):
    """OBB detection head: adds angle branch cv4 alongside cv2 (box) and cv3 (cls)."""

    def __init__(
        self,
        nc: int = 80,
        ne: int = 1,        # number of angle parameters (always 1 for YOLO11-OBB)
        reg_max: int = 16,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne) if ch else self.ne
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(c, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for c in ch
        )

    def forward(self, x: list[Tensor]) -> Tensor:
        # cv2 = box, cv3 = cls, cv4 = angle
        box_feats = [self.cv2[i](x[i]) for i in range(self.nl)]
        cls_feats = [self.cv3[i](x[i]) for i in range(self.nl)]
        angle_feats = [self.cv4[i](x[i]) for i in range(self.nl)]

        box_cat = torch.cat([b.flatten(2) for b in box_feats], dim=2)  # (B, 4*reg_max, A)
        cls_cat = torch.cat([c.flatten(2) for c in cls_feats], dim=2)  # (B, nc, A)
        ang_cat = torch.cat([a.flatten(2) for a in angle_feats], dim=2)  # (B, ne, A)

        # [init strides + anchor cache same as Detect._decode()]
        dfl_out = self.dfl(box_cat)
        # dist2rbox decodes using angle — see postprocess/_obb_nms.py
        ang_cat = (ang_cat.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        dbox = dist2rbox(dfl_out, ang_cat, self._cached_anchors) * self._cached_strides_t

        cls_cat.sigmoid_()
        return torch.cat([dbox, cls_cat, ang_cat], dim=1)  # (B, 4+nc+ne, A)
```

### Pattern 2: dist2rbox — Decode DFL distances + angle to rotated box

**What:** Converts `(left, top, right, bottom)` DFL predictions and a rotation angle into a rotated bounding box in `(cx, cy, w, h)` format. This is the core geometry operation for OBB.

**When to use:** Inside OBBHead.forward() to decode raw DFL predictions to spatial box coordinates.

**Example:**
```python
# Source: ultralytics/utils/tal.py (verified 2026-03-08)
# NOTE: ultralytics uses dim=-1 (transposed layout) but our arch uses dim=1 (B, C, A)
# Adapt: operate on (B, 4, A) input directly
def dist2rbox(
    pred_dist: Tensor,   # (B, 4, A) — left, top, right, bottom DFL-decoded
    pred_angle: Tensor,  # (B, 1, A) — angle in radians [-pi/4, 3pi/4]
    anchor_points: Tensor,  # (A, 2) — anchor grid centroids
    *,
    anchors_t: Tensor | None = None,  # (1, 2, A) pre-transposed cache
) -> Tensor:
    """Returns (B, 4, A) rotated box in (cx, cy, w, h) format."""
    lt, rb = pred_dist.chunk(2, dim=1)          # each (B, 2, A)
    cos = torch.cos(pred_angle)                  # (B, 1, A)
    sin = torch.sin(pred_angle)                  # (B, 1, A)
    # Half-extents in rotated frame
    xf = (rb[:, 0:1] - lt[:, 0:1]) * 0.5       # (B, 1, A)
    yf = (rb[:, 1:2] - lt[:, 1:2]) * 0.5       # (B, 1, A)
    # Rotate half-extents back to world frame and add anchor
    if anchors_t is None:
        anchors_t = anchor_points.T.unsqueeze(0)  # (1, 2, A)
    cx = xf * cos - yf * sin + anchors_t[:, 0:1]
    cy = xf * sin + yf * cos + anchors_t[:, 1:2]
    w = lt[:, 0:1] + rb[:, 0:1]                 # (B, 1, A)
    h = lt[:, 1:2] + rb[:, 1:2]                 # (B, 1, A)
    return torch.cat([cx, cy, w, h], dim=1)      # (B, 4, A)
```

### Pattern 3: Probabilistic IoU NMS for Rotated Boxes

**What:** Replaces `cv2.dnn.NMSBoxes` (only valid for axis-aligned boxes) with probiou-based greedy NMS. Models OBBs as 2D Gaussians and uses Wasserstein distance → Hellinger distance → IoU.

**When to use:** All OBB postprocessing. Axis-aligned NMS produces WRONG results for rotated boxes.

**Example:**
```python
# Source: ultralytics/utils/metrics.py + ultralytics/utils/nms.py (verified 2026-03-08)
import math
import torch
from torch import Tensor

def _get_covariance_matrix(boxes_xywhr: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Covariance matrix components from xywhr boxes. Returns (a, b, c) for [[a,c],[c,b]] matrix."""
    # boxes_xywhr: (N, 5)
    gbbs = torch.cat((boxes_xywhr[:, 2:4].pow(2) / 12, boxes_xywhr[:, 4:5]), dim=-1)
    a, b, c = gbbs.unbind(-1)
    cos, sin = torch.cos(c), torch.sin(c)
    cos2, sin2, sincos = cos.pow(2), sin.pow(2), cos * sin
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * sincos

def probiou(obb1: Tensor, obb2: Tensor, eps: float = 1e-7) -> Tensor:
    """Probabilistic IoU for xywhr OBBs. Returns (N, M) similarity matrix."""
    # ... (Wasserstein distance → Hellinger distance → 1 - hd)

def nms_rotated(
    boxes_xywhr: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_ids: NDArray[np.intp],
    iou_threshold: float,
) -> NDArray[np.intp]:
    """Greedy NMS using probiou; class-aware via offset trick."""
    # Sort by score descending; compute probiou pairwise; suppress overlapping
```

### Pattern 4: OBBEngine — Mirrors ClassificationEngine

**What:** `OBBEngine(BaseEngine)` follows the exact same structure as `ClassificationEngine`. It overrides `_build_model_spec()`, `_validate_output_values()`, and adds an `detect_obb()` method that calls `postprocess_obb()`.

**When to use:** All OBB inference through the engine API.

**Key methods:**
- `detect_obb(frames) -> list[OBBDetection]` — synchronous batch detect
- `stream_obb(source) -> Iterator[OBBDetection]` — streaming detect (inherits from BaseEngine)
- `_result_event_name` property returns `"obb_detection"`

### Pattern 5: Model Registry for OBB

**What:** Separate `_OBB_REGISTRY` dict and `get_obb()` / `_make_obb_meta()` factory following the existing `_CLS_REGISTRY` pattern. OBB variants: `yolo11{n,s,m,l,x}-obb`.

**Weights URL:** `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{n|s|m|l|x}-obb.pt` (confirmed 200 OK for v8.3.0).

**Classes:** OBB models trained on DOTA v1 (15 categories). Default `nc=15`.

### Anti-Patterns to Avoid

- **Using cv2.dnn.NMSBoxes for OBB:** Only valid for axis-aligned boxes. Rotated boxes require probiou NMS.
- **Trying to reuse the standard Detect head:** The stride/anchor initialization must still happen (same as Detect), but the decode step calls `dist2rbox` (not `dist2bbox`).
- **Applying sigmoid twice to angles:** The OBBHead must apply `(sigmoid - 0.25) * pi` exactly once. Postprocessing does NOT re-apply sigmoid.
- **Expecting `(cx,cy,w,h,angle)` output from dist2rbox:** dist2rbox returns `(cx,cy,w,h)` — angle is concatenated separately from `ang_cat`.
- **Using `backbone.sppf_shortcut=False` for OBB:** OBB uses YOLO11 family (not YOLO26), so `sppf_shortcut=False` is already correct.
- **OBB weights for YOLO26:** YOLO26-OBB does not exist in ultralytics as of 2026-03. Only YOLO11-OBB variants exist.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rotated IoU | Custom polygon intersection | probiou (Gaussian-based) | Polygon intersection is O(n^2) with degenerate edge cases; probiou is O(1) per pair, differentiable, matches ultralytics behavior exactly |
| OBB NMS | OpenCV/Shapely-based rotated NMS | Pure PyTorch fast_nms + probiou iou_func | cv2.dnn.NMSBoxes ignores rotation; Shapely adds heavy dep |
| Angle encoding | Custom sinusoidal encoding | `(sigmoid(raw) - 0.25) * pi` | This exact encoding matches ultralytics weights — any other encoding breaks weight compatibility |
| Covariance matrix | Manual 2x2 matrix construction | `_get_covariance_matrix(boxes)` | Already proven correct in ultralytics; avoids subtle math errors |

**Key insight:** All OBB math (dist2rbox, probiou, angle encoding) must match ultralytics EXACTLY — any deviation breaks weight compatibility even if architecture loads correctly.

## Common Pitfalls

### Pitfall 1: Angle Range Confusion
**What goes wrong:** DOTA labels use angles in `[0, pi/2)` (OpenCV convention), but OBBHead outputs angles in `[-pi/4, 3pi/4]`. They are NOT the same convention.
**Why it happens:** The sigmoid encoding `(sigmoid - 0.25) * pi` centers the angle range at 0, not 0. This is intentional for training stability.
**How to avoid:** At inference time, work with the raw model output angle `[-pi/4, 3pi/4]`. For visualisation, convert using `regularize_rboxes` if needed: swap w/h and clamp angle to `[0, pi/2)`.
**Warning signs:** Boxes appear systematically rotated by 90 degrees on symmetric objects.

### Pitfall 2: Wrong Output Shape Orientation in Postprocessing
**What goes wrong:** Raw OBB head output is `(B, 4+nc+1, A)`. Postprocessing code transposes to `(B, A, 4+nc+1)` for per-anchor indexing. If the transpose is done before or after NMS at the wrong point, indices break.
**Why it happens:** Standard Detect postprocessing (`_nms.py`) already handles the transpose. OBB postprocessing needs the angle column appended AFTER box+cls columns, so the slice indices are `raw[:, :4]` (box), `raw[:, 4:4+nc]` (cls), `raw[:, 4+nc:]` (angle).
**How to avoid:** Define clear column-slice constants in `_obb_nms.py`.
**Warning signs:** angle values appearing as class scores or vice versa.

### Pitfall 3: Weight Key Mapping for OBB Head
**What goes wrong:** OBB checkpoint has head at `model.23.cv4.*` (angle branch). If weight loading only maps `model.23.` → `head.`, the cv4 weights are correctly mapped to `head.cv4.*`. This is correct if the OBBModel uses `OBBHead` under the attribute name `head`.
**Why it happens:** Identical to DetectionEngine approach — the `_LAYER_MAP` entry `"model.23."` → `"head."` already covers this. No separate mapping needed for cv4 sub-keys.
**How to avoid:** Verify with a shape check that `head.cv4.0.0.conv.weight` exists and loads.
**Warning signs:** "Missing keys" warnings specifically for `head.cv4.*` keys.

### Pitfall 4: DOTA Class Count Mismatch
**What goes wrong:** OBB models are trained on DOTA v1 with 15 classes. If the engine defaults to 80 classes (COCO), the model's head shape doesn't match and weight loading fails.
**Why it happens:** Registry default `num_classes=80` conflicts with OBB checkpoint `nc=15`.
**How to avoid:** OBB registry entries must set `num_classes=15`. OBBConfig must default to `nc=15`.
**Warning signs:** `RuntimeError: shape mismatch for head.cv3.*.weight`.

### Pitfall 5: Export Shape Mismatch for OBB
**What goes wrong:** ONNX export produces output `(B, 4+nc+1, A)` but downstream code expects `(B, A, 5+nc)`. Dynamic axes must account for the extra angle dimension.
**Why it happens:** OBB output has `ne=1` extra channel vs detection. ONNX dynamic axis config needs updating.
**How to avoid:** Add `ne` to output shape documentation and verify with `onnxruntime` sample.
**Warning signs:** ONNX runtime dimension mismatch errors during validation.

## Code Examples

Verified patterns from official sources:

### OBBBox type definition (new frozen dataclass)
```python
# In types.py — alongside BoundingBox
@dataclass(frozen=True, slots=True)
class OBBBox:
    """Oriented bounding box in (cx, cy, w, h, angle) format.

    angle: rotation in radians, range [-pi/4, 3pi/4] (ultralytics convention).
    All coordinates in original-frame pixel space.
    """
    cx: float
    cy: float
    w: float
    h: float
    angle: float      # radians, [-pi/4, 3pi/4]
    confidence: float
    class_id: int
    class_name: str = ""
```

### probiou NMS (in postprocess/_obb_nms.py, torch-based greedy)
```python
# Source: ultralytics/utils/metrics.py + nms.py (verified 2026-03-08)
def _get_covariance_matrix(
    boxes: Tensor,  # (N, 5) xywhr
) -> tuple[Tensor, Tensor, Tensor]:
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:5]), dim=-1)
    a, b, c_angle = gbbs.unbind(-1)
    cos, sin = torch.cos(c_angle), torch.sin(c_angle)
    cos2, sin2, sincos = cos.pow(2), sin.pow(2), cos * sin
    a_cov = a * cos2 + b * sin2
    b_cov = a * sin2 + b * cos2
    c_cov = (a - b) * sincos
    return a_cov, b_cov, c_cov  # each (N,)


def probiou_matrix(
    obb1: Tensor,  # (N, 5)
    obb2: Tensor,  # (M, 5)
    eps: float = 1e-7,
) -> Tensor:
    """Returns (N, M) IoU matrix for xywhr OBBs."""
    x1, y1 = obb1[..., :2].split(1, dim=-1)   # (N, 1) each
    x2, y2 = obb2[..., :2].split(1, dim=-1)   # (M, 1) each — will broadcast
    a1, b1, c1 = _get_covariance_matrix(obb1)  # (N,)
    a2, b2, c2 = _get_covariance_matrix(obb2)  # (M,)
    # Broadcast to (N, M)
    a1, b1, c1 = a1[:, None], b1[:, None], c1[:, None]
    a2, b2, c2 = a2[None, :], b2[None, :], c2[None, :]
    x1, y1 = x1, y1  # (N, 1)
    x2, y2 = x2.T, y2.T  # (1, M)
    # Wasserstein distance terms
    t1 = ((a1+a2)*(y1-y2)**2 + (b1+b2)*(x1-x2)**2) / (
        (a1+a2)*(b1+b2) - (c1+c2)**2 + eps) * 0.25
    t2 = ((c1+c2)*(x2-x1)*(y1-y2)) / (
        (a1+a2)*(b1+b2) - (c1+c2)**2 + eps) * 0.5
    t3 = (((a1+a2)*(b1+b2) - (c1+c2)**2) / (
        4 * ((a1*b1 - c1**2).clamp_(0) * (a2*b2 - c2**2).clamp_(0)).sqrt() + eps
    ) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd
```

### OBB weight key mapping (extends existing _LAYER_MAP)
```python
# In arch/_weights.py — OBB uses IDENTICAL backbone/neck map as detection
# Only difference: "model.23." -> "head."  (already correct — OBBHead sits at head attr)
# No new _LAYER_MAP needed; same _LAYER_MAP works because the layer numbering is identical.
# OBBModel attribute layout:
#   model.backbone.*   <-- same
#   model.neck.*       <-- same
#   model.head.*       <-- maps from model.23.* (cv2, cv3, cv4, dfl, stride)

def load_obb_weights(model: OBBModel, weights_path: str | Path) -> None:
    """Load ultralytics yolo11n-obb.pt into OBBModel. Reuses _LAYER_MAP."""
    # Same implementation as load_weights() — layer map is identical
    # OBBHead weights: model.23.cv2.*, model.23.cv3.*, model.23.cv4.*
    # All map via "model.23." -> "head." prefix
```

### CLI detect-obb command structure
```python
# In cli/_main.py — mirrors detect_command exactly
@cli.command("detect-obb")
@click.argument("source")
@click.option("--model", "-m", default="yolo11n-obb", help="OBB model, e.g. yolo11n-obb")
@click.option("--weights", "-w", default=None, type=click.Path(exists=True))
@click.option("--confidence", default=0.25, type=float)
@click.option("--iou", default=0.45, type=float)
@click.option("--batch", default=1, type=int)
@click.option("--output", "-o", default=None, type=click.Path())
@click.option("--json", "json_output", is_flag=True)
def detect_obb_command(source, model, weights, confidence, iou, batch, output, json_output):
    """Run OBB detection on SOURCE."""
    from yowo.obb_engine import OBBEngine
    spec = _parse_obb_model_spec(model)  # handles "yolo11n-obb" -> ModelSpec(YOLO11, NANO, task="obb")
    ...
```

### parse_model_name extension for OBB
```python
# In _convenience.py — add -obb suffix handling alongside -cls
def parse_model_name(name: str) -> ModelSpec:
    task = "detect"
    if name.endswith("-cls"):
        task = "classify"
        name = name[:-4]
    elif name.endswith("-obb"):
        task = "obb"
        name = name[:-4]
    # ... existing family/size parsing
    return ModelSpec(family, _SIZE_MAP[suffix], task=task)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Polygon intersection for rotated IoU | Probabilistic IoU (probiou) via Gaussian covariance | YOLOv8 OBB launch (2023) | Faster, differentiable, no degenerate edge cases |
| Separate OBB training pipeline | OBB head is just Detect + angle branch | YOLOv8 onward | Same backbone/neck/training loop, OBB drops in |
| cv2.minAreaRect for box regularization | `regularize_rboxes()` tensor operation | YOLOv8 | GPU-compatible, no CPU synchronization |

**Note:** YOLO26-OBB does not exist. Only YOLO11 family has OBB variants in ultralytics as of 2026-03-08.

## Open Questions

1. **OBBEngine streaming mode**
   - What we know: BaseEngine provides `stream()` + `astream()` that both work by calling `_infer_batch()`. OBBEngine needs to override `_infer_batch()` to call `postprocess_obb()` instead of `postprocess()`.
   - What's unclear: Whether the event bus should emit `"obb_detection"` or `"detection"` for downstream consumers.
   - Recommendation: Emit `"obb_detection"` (separate event name, same pattern as ClassificationEngine emitting `"classification"`).

2. **OBBDetection type vs Detection type**
   - What we know: `Detection.boxes` is `tuple[BoundingBox, ...]`. OBB needs `tuple[OBBBox, ...]`.
   - What's unclear: Whether to create `OBBDetection` as a parallel type or extend `Detection` with an optional angle field.
   - Recommendation: New `OBBDetection` dataclass with `boxes: tuple[OBBBox, ...]` — clean separation, avoids making BoundingBox optional-angle.

3. **Export task routing for OBB**
   - What we know: `export_model()` has `if spec.task == "classify"` branch. OBB needs `elif spec.task == "obb"` branch.
   - What's unclear: Whether TensorRT OBB ONNX output shape requires special dynamic axis handling for the angle dimension.
   - Recommendation: Use same `dynamic_batch` axis as detection; the output shape is `(B, 4+nc+1, A)` which has the same dynamic structure.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/ -x -q -k obb` |
| Full suite command | `uv run pytest tests/unit/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OBB-01 | OBBHead forward produces `(B, 4+nc+1, A)` output | unit | `uv run pytest tests/unit/test_obb_head.py -x -q` | Wave 0 |
| OBB-01 | dist2rbox produces correct (cx,cy,w,h) from distances+angle | unit | `uv run pytest tests/unit/test_obb_nms.py::TestDist2Rbox -x -q` | Wave 0 |
| OBB-02 | probiou NMS suppresses overlapping rotated boxes | unit | `uv run pytest tests/unit/test_obb_nms.py::TestProbiouNMS -x -q` | Wave 0 |
| OBB-02 | probiou NMS does not suppress non-overlapping rotated boxes | unit | `uv run pytest tests/unit/test_obb_nms.py::TestProbiouNMS -x -q` | Wave 0 |
| OBB-03 | OBBModel forward shape matches expected `(1, 5+nc, 8400)` for nano | unit | `uv run pytest tests/unit/test_obb_model.py -x -q` | Wave 0 |
| OBB-03 | All 5 OBB scales (n/s/m/l/x) build without error | unit | `uv run pytest tests/unit/test_obb_model.py::TestOBBModelAllSizes -x -q` | Wave 0 |
| OBB-04 | load_obb_weights maps all head.cv4.* keys correctly (no missing) | unit | `uv run pytest tests/unit/test_obb_weights.py -x -q` | Wave 0 |
| OBB-04 | OBB registry resolves yolo11n-obb URL to v8.3.0 | unit | `uv run pytest tests/unit/test_obb_registry.py -x -q` | Wave 0 |
| OBB-05 | CLI `detect-obb` exits 0 on a valid image | unit (mock engine) | `uv run pytest tests/unit/test_obb_cli.py -x -q` | Wave 0 |
| OBB-05 | parse_model_name("yolo11n-obb") returns ModelSpec with task="obb" | unit | `uv run pytest tests/unit/test_obb_cli.py::TestParseOBBModelName -x -q` | Wave 0 |
| OBB-06 | export_model with task="obb" calls build_obb_model() branch | unit (mock torch.onnx) | `uv run pytest tests/unit/test_obb_export.py -x -q` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q -k obb`
- **Per wave merge:** `uv run ruff check src/ tests/ --quiet && uv run pyright src/yowo/ && uv run pytest tests/unit/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_obb_head.py` — covers OBB-01 (OBBHead forward shape, angle encoding)
- [ ] `tests/unit/test_obb_nms.py` — covers OBB-01 (dist2rbox) + OBB-02 (probiou NMS)
- [ ] `tests/unit/test_obb_model.py` — covers OBB-03 (OBBModel builds + forward shape)
- [ ] `tests/unit/test_obb_weights.py` — covers OBB-04 (weight key mapping for cv4)
- [ ] `tests/unit/test_obb_registry.py` — covers OBB-04 (OBB registry entries, URLs)
- [ ] `tests/unit/test_obb_cli.py` — covers OBB-05 (detect-obb command, parse_model_name)
- [ ] `tests/unit/test_obb_export.py` — covers OBB-06 (export_model OBB branch)
- [ ] `src/yowo/postprocess/_obb_nms.py` — new module
- [ ] `src/yowo/obb_engine.py` — new module

## Sources

### Primary (HIGH confidence)
- `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/yolo11-obb.yaml` — exact YAML spec: identical backbone+neck layers 0-22, OBB head at layer 23, nc=15 default DOTA, scales n/s/m/l/x
- `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/nn/modules/head.py` — OBB class `__init__`, angle branch `cv4`, `forward_head`, `ne=1`, c4 calculation `max(ch[0]//4, ne)`
- `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/tal.py` — `dist2rbox` complete implementation with `(lt, rb)` split + cos/sin rotation
- `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/metrics.py` — `batch_probiou`, `_get_covariance_matrix` complete math
- `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/nms.py` — rotated NMS branch in `non_max_suppression` using `TorchNMS.fast_nms` with `batch_probiou`
- `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt` — HTTP 200 confirms weight URL at v8.3.0

### Secondary (MEDIUM confidence)
- `https://docs.ultralytics.com/tasks/obb/` — OBB task overview, DOTA v1 dataset (15 classes), inference examples
- Existing yowo codebase analysis — confirmed identical backbone/neck layer map reuse for OBB weight loading

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — All OBB-specific math verified against ultralytics source
- Architecture: HIGH — yolo11-obb.yaml and OBB class source both verified
- Pitfalls: HIGH — All derived from actual source code behavior (angle range, class count, weight mapping)
- Weight URLs: HIGH — HTTP 200 confirmed for v8.3.0

**Research date:** 2026-03-08
**Valid until:** 2026-06-08 (stable architecture; ultralytics OBB has not changed materially since YOLOv8 launch)
