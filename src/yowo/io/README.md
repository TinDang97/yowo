# io — Input Sources, Preprocessing, and Output Sinks

Opens any input source (image file, video file, RTSP stream, webcam, directory) and yields `Frame` objects; preprocesses `Frame` batches into `PreprocessedTensor`; writes detection output.

---

## Module Structure

```
io/
├── __init__.py    — public surface: open_source(), preprocess(), write_json(), write_annotated_frames()
├── _source.py     — FrameSource Protocol and all source implementations
├── _decode.py     — letterbox resize, normalization, BCHW stacking
└── _sink.py       — JSON and annotated frame output writers
```

---

## FrameSource Protocol

```python
from typing import Protocol, Iterator

class FrameSource(Protocol):
    is_live:       bool          # True for RTSP/webcam — disables progress bar
    total_frames:  int | None    # None for live sources and unknown-length videos

    def __iter__(self) -> Iterator[Frame]:
        """Yield Frame objects in capture order. Raises InputError on unrecoverable failure."""

    def close(self) -> None:
        """Release underlying resource (file handle, socket, capture device). Idempotent."""
```

All source implementations are context managers (`__enter__` / `__exit__` call `close()`).

---

## Source Dispatch

```python
def open_source(
    source:               str | int,
    loop:                 bool = False,
    frame_skip:           int = 0,
    max_frames:           int | None = None,
    reconnect_timeout_s:  float = 30.0,
) -> FrameSource:
    """
    Factory function. Inspects source and returns the appropriate FrameSource.

    source:              file path string, URL string, or integer webcam index
    loop:                repeat video file from beginning when exhausted
    frame_skip:          yield every (frame_skip+1)th frame (0 = no skip)
    max_frames:          stop after this many yielded frames
    reconnect_timeout_s: RTSPStreamSource only — total seconds to retry before raising
    """
```

### Dispatch Table

| Pattern | Source class |
|---------|-------------|
| `.jpg` / `.jpeg` / `.png` / `.bmp` / `.webp` file | `ImageFileSource` |
| Directory path | `ImageDirectorySource` (sorted by filename) |
| `.mp4` / `.avi` / `.mov` / `.mkv` / `.ts` file | `VideoFileSource` |
| `rtsp://...` | `RTSPStreamSource` |
| `"0"`, `"1"`, ... or `int` | `WebcamSource` |

`open_source` raises `InputError` immediately if the pattern does not match any known source type, or if the file/directory does not exist.

---

## Source Implementations

### `ImageFileSource`

- Reads single image via `cv2.imread()`.
- Yields exactly one `Frame`.
- `total_frames = 1`, `is_live = False`.

### `ImageDirectorySource`

- Iterates all supported image files in directory, sorted by filename.
- `total_frames = len(files)`, `is_live = False`.
- Raises `InputError` if directory contains no supported images.

### `VideoFileSource`

- Opens via `cv2.VideoCapture(path)`.
- `total_frames` from `cv2.CAP_PROP_FRAME_COUNT` (may be `None` for some containers).
- `is_live = False`.
- Supports `loop=True` — seeks to frame 0 on `StopIteration`.

### `RTSPStreamSource`

- Opens via `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`.
- `is_live = True`, `total_frames = None`.
- Reconnect strategy: exponential backoff on read failure.

```
reconnect attempt 1: wait  2s
reconnect attempt 2: wait  4s
reconnect attempt 3: wait  8s
reconnect attempt 4: wait 10s  (cap)
...continues until reconnect_timeout_s is exceeded
```

- Raises `InputError` when total elapsed reconnect time exceeds `reconnect_timeout_s`.

### `WebcamSource`

- Opens `cv2.VideoCapture(int(source))`.
- `is_live = True`, `total_frames = None`.
- Raises `InputError` immediately if device index is unavailable.

---

## Preprocessing — `_decode.py`

```python
@dataclass(frozen=True, slots=True)
class TensorMeta:
    """Metadata needed to reverse the letterbox transform in postprocess."""
    scale_factors:  tuple[float, float]    # (scale_h, scale_w) applied to original frame
    pad_offsets:    tuple[int, int]        # (pad_top, pad_left) pixels added
    original_sizes: tuple[tuple[int, int], ...]  # per-frame (H, W) before resize

def preprocess(
    frames:      list[Frame],
    target_size: tuple[int, int],          # (height, width) — from ModelMeta
) -> tuple[PreprocessedTensor, TensorMeta]:
    """
    Convert a batch of raw BGR frames to a float32 BCHW tensor ready for inference.

    Steps applied to each frame:
      1. Letterbox resize: scale to fit target_size while preserving aspect ratio.
      2. Pad: fill remaining area with gray (114, 114, 114).
      3. Color convert: BGR → RGB.
      4. Transpose: HWC → CHW.
      5. Normalize: divide by 255.0 → values in [0, 1].
    All frames stacked on axis 0 → shape (B, 3, H_target, W_target).

    Returns (tensor, meta) where meta carries the transform parameters
    needed by postprocess to recover original-frame coordinates.
    """
```

### Letterbox Algorithm

```
original frame: H_orig × W_orig

scale = min(H_target / H_orig, W_target / W_orig)
new_h = int(round(H_orig * scale))
new_w = int(round(W_orig * scale))

pad_top    = (H_target - new_h) // 2
pad_bottom = H_target - new_h - pad_top
pad_left   = (W_target - new_w) // 2
pad_right  = W_target - new_w - pad_left

resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
padded  = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
```

`scale_factors = (scale, scale)` and `pad_offsets = (pad_top, pad_left)` are stored in `TensorMeta` for the inverse transform in `postprocess/_nms.py`.

---

## Output Sinks — `_sink.py`

```python
def write_json(
    detections: list[Detection],
    path:       Path,
) -> None:
    """
    Serialize detections to JSON.

    Each Detection serializes as:
      { "frame_index": int, "boxes": [ { "x1", "y1", "x2", "y2",
                                         "class_id", "class_name", "confidence" } ] }
    Writes atomically via temp file + os.replace().
    """

def write_annotated_frames(
    detections:  list[Detection],
    output_dir:  Path,
) -> None:
    """
    Draw bounding boxes on each frame and save as JPEG.

    Filenames: 000000.jpg, 000001.jpg, ...
    Box color: class_id % 20 mapped to a fixed color palette.
    Label: "{class_name} {confidence:.2f}" drawn above box.
    output_dir is created if it does not exist.
    """
```

---

## Dependencies

- **cv2** (`opencv-python-headless`): capture, resize, color convert, annotate, write
- **numpy**: array operations in `_decode.py`
- **stdlib**: `pathlib`, `json`, `os`, `time`
- **yowo imports**: `types.py` (`Frame`, `PreprocessedTensor`, `Detection`), `errors.py` (`InputError`)

---

## Cross-References

| Direction | Module | What it provides / consumes |
|-----------|--------|-----------------------------|
| Upstream | `types.py` | `Frame`, `PreprocessedTensor`, `Detection`, `BoundingBox` |
| Upstream | `errors.py` | `InputError` |
| Downstream | `engine.py` | calls `open_source()`, `preprocess()`, `write_json()`, `write_annotated_frames()` |
| Downstream | `cli/_main.py` | passes user CLI args directly to `open_source()` |
