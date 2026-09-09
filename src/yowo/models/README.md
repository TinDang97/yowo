# models — Model Registry and Weight Management

Registry of supported YOLO model families and sizes, plus weight file download and cache management.

---

## Responsibilities

- Maintain a registry of known model families, sizes, and their metadata.
- Resolve a `ModelSpec` to a local `.pt` weights file, downloading and caching it on first use.
- Expose a single `register()` hook so new model variants can be added without modifying any other module.

---

## Module Structure

```
models/
├── __init__.py    — public surface: get(), list_available(), resolve_weights()
├── _registry.py   — ModelMeta dataclass, in-memory registry, built-in registrations
└── _weights.py    — download, hash verification, local cache management
```

### `_registry.py`

```python
@dataclass(frozen=True, slots=True)
class ModelMeta:
    family:               str          # "yolo11" | "yolo26"
    size:                 str          # "n" | "s" | "m" | "l" | "x"
    input_height:         int          # default input resolution height (e.g. 640)
    input_width:          int          # default input resolution width  (e.g. 640)
    num_classes:          int          # 80 for COCO
    default_weights_url:  str          # HTTPS URL to canonical .pt file
    weight_stem:          str          # e.g. "yolo11n" — used for weight file resolution
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `register` | `(meta: ModelMeta) -> None` | Add entry to the in-memory registry. Overwrites if `(family, size)` key already exists. |
| `get` | `(family: str, size: str) -> ModelMeta` | Return entry or raise `ModelNotFoundError`. |
| `list_available` | `() -> list[ModelMeta]` | All registered entries, sorted by `(family, size)`. |

The registry is a module-level `dict[tuple[str, str], ModelMeta]` populated at import time by `_register_builtins()`.

### `_weights.py`

```python
def resolve_weights(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """
    Return path to a local .pt file for spec.

    Resolution order:
      1. spec.weights_path if set and file exists → return as-is
      2. Cache hit at cache_dir / family / size / filename → return cached path
      3. Download from ModelMeta.default_weights_url → verify hash → cache → return

    cache_dir defaults to ~/.cache/yowo/weights/
    Raises ModelNotFoundError if spec.family/size not in registry.
    Raises IOError on download failure after retries.
    """
```

Download behavior:
- Retries: 3 attempts with exponential backoff (2s, 4s, 8s).
- Progress: `tqdm` progress bar to stderr (suppressed when `CI=true`).
- Hash verification: SHA-256 checked against `ModelMeta.sha256`, a digest pinned in the
  registry beside the URL it verifies. Checked on first download **and on every cache
  hit** — a file swapped after download would otherwise stay trusted forever.
- Atomic write: download to `.tmp`, verify the digest, then `os.replace()`. A mismatch
  leaves nothing behind: not the bad file, not a partial. It is never retried (the same
  wrong bytes would return) and never downgraded to a warning.
- Pre-existing cache entries: a cached weight that fails verification is re-downloaded
  once and replaced, with a warning. Not a hard failure — that would break working and
  air-gapped installs on upgrade — and not grandfathered.
- Unpinned models: a user-registered `ModelMeta` may set `sha256=None`. Its weights load
  with a warning rather than a refusal, since a private bucket has no digest we could know.
- Deserialisation: after verification a checkpoint is converted once to a tensor-only
  `state_dict` sidecar. Every later load reads that with `torch.load(weights_only=True)`
  and executes nothing. The sidecar is keyed to the raw file's digest, so a re-fetched or
  re-pinned weight regenerates it instead of serving stale tensors.

---

## Supported Models

| Family | Size | Input (H×W) | Classes | Weight stem |
|--------|------|------------|---------|-------------|
| yolo11 | n | 640×640 | 80 | yolo11n |
| yolo11 | s | 640×640 | 80 | yolo11s |
| yolo11 | m | 640×640 | 80 | yolo11m |
| yolo11 | l | 640×640 | 80 | yolo11l |
| yolo11 | x | 640×640 | 80 | yolo11x |
| yolo26 | n | 640×640 | 80 | yolo26n |
| yolo26 | s | 640×640 | 80 | yolo26s |
| yolo26 | m | 640×640 | 80 | yolo26m |
| yolo26 | l | 640×640 | 80 | yolo26l |
| yolo26 | x | 640×640 | 80 | yolo26x |

---

## Public Interface

```python
from yowo.models import get, list_available, resolve_weights, ModelMeta

# Look up metadata
meta: ModelMeta = get("yolo26", "n")

# List all registered models
all_models: list[ModelMeta] = list_available()

# Resolve to a local .pt path (downloads if needed)
weights_path: Path = resolve_weights(ModelSpec(family="yolo26", size="n"))
```

---

## Adding New Models

No other module needs to change. Call `register()` at import time in your extension module:

```python
from yowo.models._registry import register, ModelMeta

register(ModelMeta(
    family="yolo_custom",
    size="n",
    input_height=640,
    input_width=640,
    num_classes=10,
    default_weights_url="https://my-bucket.s3.amazonaws.com/yolo_custom_n.pt",
    weight_stem="yolo_custom_n",
))
```

If your bucket requires a credential, put it in `default_weights_url` as normal userinfo
(`https://KEY:SECRET@my-bucket.s3.amazonaws.com/yolo_custom_n.pt`) — the registry stores it
intact so the download can authenticate. It is never displayed: `yowo.io.redact_url()` strips
the credential everywhere `default_weights_url` is shown, including `yowo models` output and
download-failure messages.

---

## Cache Layout

```
~/.cache/yowo/weights/
└── yolo26/
    ├── n/
    │   ├── yolo26n.pt
    │   └── yolo26n.pt.sha256
    └── m/
        ├── yolo26m.pt
        └── yolo26m.pt.sha256
```

Cache directory is configurable via `YOWO_CACHE_DIR` environment variable or `cache_dir` argument to `resolve_weights()`.

---

## Dependencies

- **Stdlib**: `hashlib`, `os`, `pathlib`, `urllib.request`, `time`
- **Optional**: `tqdm` (progress bar; silently skipped if not installed)
- **yowo imports**: `types.py` (`ModelSpec`), `errors.py` (`ModelNotFoundError`)

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `types.py` | `ModelSpec` dataclass |
| Upstream | `errors.py` | `ModelNotFoundError` |
| Downstream | `engine.py` | calls `resolve_weights()` during `Engine.__init__` |
| Downstream | `export/_exporter.py` | calls `resolve_weights()` before running `torch.onnx.export()` |
