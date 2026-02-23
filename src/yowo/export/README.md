# export — Model Export and Quantization

Converts PyTorch `.pt` weights to optimized inference formats (ONNX, TensorRT, OpenVINO) with optional quantization. Uses native `yowo.arch` models with `torch.onnx.export()` — no external dependencies beyond PyTorch. Adds: weight resolution, calibration data handling, output validation, and metadata sidecar writing.

---

## Design

The export pipeline has one entry point (`export_model`) and three internal responsibilities split across three private modules:

```
export_model(spec, target_format, output_dir, precision, dynamic_batch)
    │
    ├── models.resolve_weights(spec)        → .pt path
    ├── arch.build_model() + load_weights() → fused native model
    ├── _exporter.run_export()              → torch.onnx.export, returns output path
    ├── _calibration.resolve_calibration()  → calibration data (INT8 only)
    └── _metadata.save()                    → writes .yowo.json sidecar
```

---

## Export Pipeline

```
.pt weights
    │
    ▼ (build native model, fuse Conv+BN, torch.onnx.export)
  ONNX (.onnx)  ──────────────────────────────► ExportResult
    │
    ├──► TensorRT (.engine)  ────────────────► ExportResult
    │      (trtexec or tensorrt Python API)
    │
    └──► OpenVINO (_openvino_model/ directory) ► ExportResult
           (openvino.convert_model Python API)
```

TensorRT and OpenVINO exports always go through ONNX as an intermediate.

---

## Module Structure

```
export/
├── __init__.py       — public surface: export_model(), ExportResult
├── _exporter.py      — native torch.onnx.export pipeline
├── _calibration.py   — calibration source resolution for INT8
└── _metadata.py      — ExportMetadata dataclass, sidecar write/read
```

### `__init__.py`

```python
@dataclass(frozen=True, slots=True)
class ExportResult:
    file_path:   Path           # path to the exported model file or directory
    format:      str            # "onnx" | "tensorrt" | "openvino"
    precision:   Precision
    metadata:    ExportMetadata

def export_model(
    spec:          ModelSpec,
    target_format: str,                  # "onnx" | "tensorrt" | "openvino"
    output_dir:    Path | None = None,   # default: ~/.yowo/exports/
    precision:     Precision = Precision.FP16,
    dynamic_batch: bool = False,
    calibration_data: Path | None = None,  # required when precision == INT8
) -> ExportResult:
    """
    Export spec to target_format at precision.

    Raises ExportError if:
      - target_format is not in {"onnx", "tensorrt", "openvino"}
      - precision == INT8 and calibration_data is None
      - torch.onnx.export or downstream conversion fails
      - expected output file is not found after export
    """
```

### `_exporter.py`

Uses native `yowo.arch` models with `torch.onnx.export()`. Responsibilities:

- Resolve weights path via `models.resolve_weights(spec)`.
- Build native model via `arch.build_model()`, load weights, fuse Conv+BN.
- Export to ONNX via `torch.onnx.export()` with opset 17, optional dynamic batch axes.
- Optionally simplify ONNX graph with `onnxslim`.
- Convert ONNX to TensorRT (via `trtexec`) or OpenVINO (via `openvino.convert_model`) as needed.
- Wrap any exception in `ExportError`.

```python
def run_export(
    weights_path:     Path,
    target_format:    str,
    precision:        Precision,
    dynamic_batch:    bool,
    calibration_yaml: Path | None,
    output_dir:       Path,
    hw_profile:       HardwareProfile,
) -> Path:
    """Internal. Returns path to the exported artifact."""
```

### `_calibration.py`

```python
def resolve_calibration_source(
    source: Path,
    min_images: int = 300,
) -> Path:
    """
    Validate and return a calibration data directory path.

    If source is a directory:
      - Count supported image files (.jpg, .png, .bmp, .webp).
      - Warn (not raise) if count < min_images.
      - Return the directory path.
    Raises InputError if source is not a directory.
    Raises InputError if source directory contains zero supported images.
    """
```

Recommended minimum of 300 images for calibration. Using fewer images produces an INT8 engine but may show higher accuracy degradation than the ~1-3% typical figure.

### `_metadata.py`

```python
@dataclass(frozen=True, slots=True)
class ExportMetadata:
    model_name:          str            # "{family}{size}" e.g. "yolo26n"
    format:              str            # "onnx" | "tensorrt" | "openvino"
    precision:           Precision
    input_shape:         tuple[int, int, int, int]   # (1, 3, H, W)
    file_path:           Path
    file_size_bytes:     int
    created_at:          datetime       # UTC
    export_duration_sec: float
    yowo_version:        str
    gpu_name:            str | None     # None if CPU export
    calibration_data:    Path | None    # None for FP32/FP16

    def save(self) -> Path:
        """
        Write metadata as JSON to file_path.with_suffix('.yowo.json').
        Returns path to the written sidecar file.
        """

    @classmethod
    def load(cls, model_path: Path) -> ExportMetadata:
        """
        Read metadata from model_path.with_suffix('.yowo.json').
        Raises FileNotFoundError if sidecar is absent.
        """
```

---

## Quantization Guide

| Precision | Use case | Accuracy impact | Calibration required |
|-----------|----------|----------------|---------------------|
| FP32 | Baseline, debugging, CPU-only | None (reference) | No |
| FP16 | Default GPU production | Negligible (<0.5% mAP) | No |
| INT8 | Edge/Jetson, maximum throughput | ~1-3% mAP | Yes — 300+ images |

---

## INT8 Calibration Workflow

```
1. Gather 300+ representative images from the deployment domain.
   (Images should match the lighting, resolution, and content distribution
    of production input — not generic COCO validation images.)

2. Call export_model(..., precision=INT8, calibration_data=Path("./calib_images/"))

3. yowo._calibration validates the directory and image count.

4. TensorRT calibration runs during engine build:
   - Feeds images through the network one batch at a time.
   - Collects per-layer activation range statistics.
   - Selects INT8 quantization scales that minimize accuracy loss.

5. Resulting .engine file embeds the calibration scales.

6. Engine is GPU-arch-specific: re-export if deploying to a different GPU family.
   (sm_86 engine will not load on sm_89 hardware and vice versa.)
```

---

## Metadata Sidecar

Every exported model gets a `.yowo.json` sidecar file alongside it:

```
~/.yowo/exports/
└── yolo26n_tensorrt_fp16/
    ├── yolo26n.engine
    └── yolo26n.yowo.json
```

Example sidecar content:

```json
{
  "model_name": "yolo26n",
  "format": "tensorrt",
  "precision": "fp16",
  "input_shape": [1, 3, 640, 640],
  "file_path": "/home/user/.yowo/exports/yolo26n_tensorrt_fp16/yolo26n.engine",
  "file_size_bytes": 12582912,
  "created_at": "2026-02-23T14:30:00Z",
  "export_duration_sec": 142.3,
  "yowo_version": "0.1.0",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "calibration_data": null
}
```

---

## Dependencies

- **torch**: model building and `torch.onnx.export()` (not imported at module level — imported inside `_exporter.run_export()`)
- **onnxslim** (optional): ONNX graph simplification
- **yowo imports**: `arch/` (`build_model`, `load_weights`), `models/` (`resolve_weights`), `hardware/` (`get_hardware_profile`), `types.py`, `errors.py` (`ExportError`, `InputError`)

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `models/` | `resolve_weights()` to get `.pt` path |
| Upstream | `hardware/` | `HardwareProfile.primary_gpu` for GPU name in metadata |
| Upstream | `types.py` | `ModelSpec`, `Precision` |
| Upstream | `errors.py` | `ExportError`, `InputError` |
| Downstream | `cli/_main.py` | `yowo export` command calls `export_model()` |
