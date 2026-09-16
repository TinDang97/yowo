"""Measure what the feature cache actually saves, and what it cannot see.

The README claimed "60-85% compute savings" with no experiment behind it,
unlike every other headline figure in this repo. This produces the figure that
replaces it, on the fingerprint the release ships.

Two measurements, both on a real photograph rather than a synthetic frame --
`tune`'s blank-frame mistake is documented elsewhere in this milestone and is
not repeated here:

  RECALL   the largest object, in pixels, that the fingerprint cannot see, at
           two contrasts and at both placements (inside one cell, and
           straddling a corner where only a quarter of it lands in each of
           four cells).
  SAVINGS  hit rate and per-frame latency over a panning sequence built from
           that photograph, with the cache on and off.

Run:  uv run python scripts/measure_feature_cache.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from yowo.cache import FeatureCache
from yowo.cache._similarity import DEFAULT_GRID

THRESHOLD = 0.01
FRAME = 640


def _feats() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((1, 144, 80, 80), np.float32),
        np.zeros((1, 288, 40, 40), np.float32),
        np.zeros((1, 576, 20, 20), np.float32),
    )


def _base(value: float = 0.5) -> np.ndarray:
    return np.full((1, 3, FRAME, FRAME), value, np.float32)


def measure_recall(grid: int) -> dict[tuple[float, str], int]:
    """Largest invisible square, per contrast and placement."""
    out: dict[tuple[float, str], int] = {}
    for contrast in (0.5, 0.30, 0.15):
        for placement in ("aligned", "straddling"):
            largest = 0
            for size in range(1, 400):
                probe = _base()
                if placement == "straddling":
                    centre = (FRAME // grid) * (grid // 2)
                    top = max(0, centre - size // 2)
                else:
                    top = 0
                probe[:, :, top : top + size, top : top + size] = 0.5 + contrast

                cache = FeatureCache(similarity_threshold=THRESHOLD, grid=grid)
                cache.update("cam", _base(), _feats())
                if cache.check_and_load("cam", probe) is None:
                    break
                largest = size
            out[(contrast, placement)] = largest
    return out


def _panning_sequence(image: Path, frames: int, step: int) -> list[np.ndarray]:
    """A real photograph panned by `step` px per frame, letterboxed to BCHW."""
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"cannot read {image}")
    big = cv2.resize(img, (FRAME + frames * step, FRAME))
    seq = []
    for i in range(frames):
        crop = big[:, i * step : i * step + FRAME]
        t = crop[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        seq.append(np.ascontiguousarray(t))
    return seq


def measure_hit_rate(seq: list[np.ndarray], grid: int, threshold: float = THRESHOLD) -> float:
    cache = FeatureCache(similarity_threshold=threshold, grid=grid)
    hits = 0
    for frame in seq:
        if cache.check_and_load("cam", frame) is not None:
            hits += 1
        else:
            cache.update("cam", frame, _feats())
    return 100.0 * hits / len(seq)


def _surveillance_sequence(image: Path, frames: int, obj: int, speed: int) -> list[np.ndarray]:
    """The case the docstring claims: a FIXED camera, one small thing moving.

    A panning sequence moves every pixel and is the worst case for any
    fingerprint. This is the case the default was chosen for, so it is the one
    the default has to be judged on.
    """
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"cannot read {image}")
    base = cv2.resize(img, (FRAME, FRAME))
    seq = []
    for i in range(frames):
        f = base.copy()
        x = (i * speed) % max(1, FRAME - obj)
        patch = base[FRAME // 2 : FRAME // 2 + obj, :obj]
        f[FRAME // 2 : FRAME // 2 + obj, x : x + obj] = 255 - patch
        t = f[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        seq.append(np.ascontiguousarray(t))
    return seq


def measure_blind_spot(grid: int, threshold: float, contrast: float) -> int:
    """Largest straddling square invisible at this threshold."""
    largest = 0
    for size in range(1, 400):
        probe = _base()
        centre = (FRAME // grid) * (grid // 2)
        top = max(0, centre - size // 2)
        probe[:, :, top : top + size, top : top + size] = 0.5 + contrast
        cache = FeatureCache(similarity_threshold=threshold, grid=grid)
        cache.update("cam", _base(), _feats())
        if cache.check_and_load("cam", probe) is None:
            break
        largest = size
    return largest


def measure_savings(
    image: Path, frames: int, grid: int, device: str = "cpu", threshold: float = THRESHOLD
) -> dict[str, float]:
    """Per-frame latency on the REAL backend path, cache on and off.

    Driven through ``PyTorchBackend`` with ``set_source_id`` rather than
    ``engine.detect``, because the cache is only consulted when a frame
    carries a source id (``engine.py:1030-1033``) and ``detect()`` on a bare
    array supplies none. Measuring through ``detect()`` would have reported
    the cost of the feature with none of its benefit, and reported it as 0%.
    """
    from yowo.backends._pytorch import PyTorchBackend
    from yowo.hardware import get_hardware_profile
    from yowo.io._decode import preprocess
    from yowo.models import resolve_weights
    from yowo.types import Frame, ModelFamily, ModelSize, ModelSpec

    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    hw = get_hardware_profile()
    weights = resolve_weights(spec)

    img = cv2.imread(str(image))
    base = cv2.resize(img, (FRAME, FRAME))
    obj = 48
    crops = []
    for i in range(frames):
        f = base.copy()
        x = (i * 6) % max(1, FRAME - obj)
        patch = base[FRAME // 2 : FRAME // 2 + obj, :obj]
        f[FRAME // 2 : FRAME // 2 + obj, x : x + obj] = 255 - patch
        crops.append(f)
    tensors = [
        preprocess([Frame(pixels=c, source_id="cam", frame_index=i)], (FRAME, FRAME))
        for i, c in enumerate(crops)
    ]

    out: dict[str, float] = {}
    for label, cache in (
        ("cache_off", None),
        ("cache_on", FeatureCache(similarity_threshold=threshold, grid=grid)),
    ):
        backend = PyTorchBackend(hw, model_spec=spec, feature_cache=cache)
        backend.load(weights, device=device)
        backend.set_source_id("cam")
        backend.infer(tensors[0])
        times = []
        for t in tensors:
            t0 = time.perf_counter()
            backend.infer(t)
            times.append((time.perf_counter() - t0) * 1000)
        backend.unload()
        out[label] = statistics.median(times)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=Path.home() / ".cache/yowo/test-assets/bus.jpg")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--step", type=int, default=2, help="pan in px per frame")
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    print(f"grid={args.grid}  threshold={THRESHOLD}  frame={FRAME}x{FRAME}")
    print()
    print("RECALL — largest object the fingerprint cannot see")
    for (contrast, placement), size in measure_recall(args.grid).items():
        pct = 100 * size**2 / FRAME**2
        print(
            f"  contrast {contrast:<5} {placement:<11} "
            f"{size:>3}x{size:<3} px  ({pct:.2f}% of frame)"
        )

    print()
    panning = _panning_sequence(args.image, args.frames, args.step)
    fixed = _surveillance_sequence(args.image, args.frames, obj=48, speed=6)
    print(f"HIT RATE and the bound that comes with it — {args.frames} frames of {args.image.name}")
    print("  threshold   fixed camera   panning    invisible object (straddling, 0.15 contrast)")
    for thr in (0.005, 0.01, 0.02, 0.05, 0.10):
        b = measure_blind_spot(args.grid, thr, 0.15)
        print(
            f"  {thr:<11.3f} {measure_hit_rate(fixed, args.grid, thr):>6.1f}%"
            f"       {measure_hit_rate(panning, args.grid, thr):>6.1f}%"
            f"     {b:>3}x{b:<3} px ({100 * b**2 / FRAME**2:.2f}% of frame)"
        )

    if not args.skip_model:
        print()
        print("SAVINGS — yolo11n, fixed camera, median per-frame latency")
        print("  device  threshold   cache off   cache on    saving")
        for device in args.devices.split(","):
            for thr in (0.01, 0.05, 0.10):
                try:
                    s = measure_savings(args.image, args.frames, args.grid, device, thr)
                except Exception as exc:
                    print(f"  {device:<7} {thr:<11.3f} unavailable: {type(exc).__name__}: {exc}")
                    break
                saved = 100 * (1 - s["cache_on"] / s["cache_off"]) if s["cache_off"] else 0.0
                print(
                    f"  {device:<7} {thr:<11.3f} {s['cache_off']:8.2f} ms"
                    f" {s['cache_on']:8.2f} ms  {saved:+7.1f}%"
                )


if __name__ == "__main__":
    main()
