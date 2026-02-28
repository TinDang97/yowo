# cli — Command-Line Interface

Click-based CLI entry point exposing the yowo library as the `yowo` command. Framework choice: Click (not Typer) — mature, lightweight, excellent nested command groups, full control over help formatting.

---

## Module Structure

```
cli/
├── __init__.py    — registers the Click group as package entry point
└── _main.py       — all command definitions and option parsing
```

Entry point registration in `pyproject.toml`:

```toml
[project.scripts]
yowo = "yowo.cli._main:cli"
```

---

## Command Reference

### `yowo detect <source>`

Run inference on any supported input source.

```
yowo detect <source>
            [--model MODEL]
            [--backend BACKEND]
            [--device DEVICE]
            [--precision PRECISION]
            [--confidence FLOAT]
            [--iou FLOAT]
            [--batch INT]
            [--output PATH]
            [--save-frames DIR]
            [--show]
            [--json]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `yolo26n` | Model name in `{family}{size}` format, e.g. `yolo11m`, `yolo26n` |
| `--backend` | `auto` | `auto \| tensorrt \| onnx \| openvino \| pytorch` |
| `--device` | `auto` | `auto \| cuda \| cuda:0 \| cuda:1 \| cpu` |
| `--precision` | `auto` | `auto \| fp32 \| fp16 \| int8` |
| `--confidence` | `0.25` | Discard detections below this confidence score |
| `--iou` | `0.45` | NMS IoU suppression threshold |
| `--batch` | `1` | Number of frames per inference batch |
| `--output` | none | Save detections to a JSON file at PATH |
| `--save-frames` | none | Save annotated frames as JPEGs to DIR |
| `--show` | false | Display frames in an OpenCV window (requires `$DISPLAY`) |
| `--json` | false | Machine-readable output: print JSON to stdout instead of human text |

`<source>` accepts: image file path, video file path, directory path, `rtsp://...` URL, or webcam index integer.

### `yowo export <model>`

Export PyTorch weights to an optimized inference format.

```
yowo export <model>
            [--format FORMAT]
            [--precision PRECISION]
            [--calibration-data PATH]
            [--output-dir DIR]
            [--dynamic-batch]
            [--json]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `onnx` | Target format: `onnx \| tensorrt \| openvino \| coreml` |
| `--precision` | `fp16` | `fp32 \| fp16 \| int8` |
| `--calibration-data` | none | Required for `--precision int8`: path to image directory or calibration YAML |
| `--output-dir` | `~/.yowo/exports/` | Destination directory for exported model and sidecar |
| `--dynamic-batch` | true | Export with dynamic batch dimension (ONNX/TensorRT). Use `--no-dynamic-batch` to disable |
| `--batch-sizes` | none | Comma-separated batch sizes for CoreML EnumeratedShapes (e.g. `1,4,8`) |
| `--json` | false | Print `ExportResult` as JSON to stdout |

`<model>` is a model name in `{family}{size}` format, e.g. `yolo26n`.

### `yowo info`

Print detected hardware profile and available backends. No arguments.

```
yowo info [--json]
```

Output includes:

- Detected GPUs: name, arch, VRAM total/available, FP16/INT8 support
- CPU: arch, detected features (AVX2, NEON, VNNI)
- Installed SDKs: TensorRT, ONNX Runtime, OpenVINO, PyTorch (with versions)
- Selected default backend for this machine

### `yowo models`

List registered model variants. Optionally filter output.

```
yowo models [--family FAMILY]
            [--format FORMAT]
            [--json]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--family` | none | Filter to a specific model family: `yolo11 \| yolo26` |
| `--format` | none | Filter to models that have a cached export in this format |
| `--json` | none | Print as JSON array |

---

## Output Formats

### Human-readable (default)

```
yowo detect photo.jpg
─────────────────────────────────────────
Source       : photo.jpg
Backend      : TensorRT (fp16, cuda:0)
Model        : yolo26n  [640×640]
Inference    : 3.2 ms/frame
─────────────────────────────────────────
Frame 0001   2 detections
  person       0.92   [120, 45, 380, 610]
  car          0.87   [500, 200, 900, 480]
```

### Machine-readable (`--json`)

```json
{
  "source": "photo.jpg",
  "backend": "tensorrt",
  "precision": "fp16",
  "model": "yolo26n",
  "detections": [
    {
      "frame_index": 0,
      "inference_time_ms": 3.2,
      "boxes": [
        { "x1": 120, "y1": 45, "x2": 380, "y2": 610,
          "class_id": 0, "class_name": "person", "confidence": 0.92 },
        { "x1": 500, "y1": 200, "x2": 900, "y2": 480,
          "class_id": 2, "class_name": "car", "confidence": 0.87 }
      ]
    }
  ]
}
```

---

## Exit Codes

| Code | Meaning | Examples |
|------|---------|---------|
| `0` | Success | Normal completion |
| `1` | User error | Bad argument, model not found, unsupported source format |
| `2` | System error | Backend crash, OOM, GPU driver failure, RTSP timeout |

The CLI maps `YowoError` subclasses to exit codes:

```
InputError, ModelNotFoundError  → exit 1
BackendError, HardwareError     → exit 2
DependencyError                 → exit 1 (with install instructions printed)
```

Unhandled exceptions (unexpected failures) print a traceback to stderr and exit `2`.

---

## Model Name Parsing

`<model>` arguments in the form `yolo26n` are split into `(family, size)` by stripping the trailing size character:

```python
SIZES = {"n", "s", "m", "l", "x"}
# "yolo26n" → family="yolo26", size="n"
# "yolo11xl" is invalid — raises click.BadParameter
```

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `engine.py` | `Engine` class — `detect` command instantiates and calls it |
| Upstream | `export/` | `export_model()` — `export` command calls it |
| Upstream | `hardware/` | `get_hardware_profile()` — `info` command reads it |
| Upstream | `models/` | `list_available()` — `models` command reads it |
| Upstream | `__init__.py` | public API types used for annotation |
