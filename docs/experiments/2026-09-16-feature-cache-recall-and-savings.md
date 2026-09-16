# Feature cache: what it cannot see, and what it actually saves

**Date:** 2026-09-16
**Machine:** darwin / arm64, Apple Silicon
**Model:** yolo11n, PyTorch backend, fp32
**Script:** `scripts/measure_feature_cache.py` (re-run it; every number here comes from it)
**Why:** `README.md` and `src/yowo/cache/README.md` both claimed "60–85% compute
savings" with no experiment behind them, unlike every other headline figure in
this repo. This is that experiment. It does not confirm the claim.

## What the cache decides on

The cache skips the backbone and neck when it judges the current frame similar
enough to the last one. Before this release that judgement came from
`tensor.mean(axis=(2, 3))` — **three numbers** for a 640×640 RGB frame. Three
numbers cannot say where anything is.

Measured against that fingerprint at the shipped threshold of 0.01, the largest
object that could appear without the cache noticing:

| contrast | invisible object | share of a 640×640 frame |
|---|---|---|
| 0.50 (grey → white) | 90×90 px | 1.98% |
| 0.30 | 116×116 px | 3.29% |
| 0.15 (realistic) | 165×165 px | 6.65% |

Three further failures had no bound at all:

- a **640×640 entry answered a 320×320 query**, returning 80×80 features for an
  input needing 40×40 — both frames fingerprint to three numbers, so the resize
  was invisible;
- a **B=1 entry answered a B=4 query**, because `np.abs((1,3) - (4,3))`
  broadcasts rather than raising. `cache=True` ships in a preset that also sets
  `batch_size=4`, so any video whose frame count is not a multiple of 4 ends on
  a partial batch;
- a **black-over-white frame matched a uniform grey one exactly**, because both
  average 0.5.

The guard that would have stopped the first of those existed —
`_similarity.py:26` returned 1.0 on a shape mismatch — and `frame_similarity`
had **zero callers in `src/`**. It was not untested dead code: `test_cache.py`
tested it. That is how a shape guard stayed covered, passing and unreachable
while the live path had none.

## The fingerprint this release ships

Per-cell means over an 8×8 grid, compared by the **worst cell** rather than the
frame average. Cell edges come from `np.linspace` over the real extent, so every
pixel lands in exactly one cell whatever `H` and `W` are.

Measured, grid 8, threshold 0.01. "Straddling" centres the object on a cell
corner so only a quarter of it lands in each of four cells — the worst placement,
and therefore the number that gets quoted:

| contrast | aligned | straddling | share of frame |
|---|---|---|---|
| 0.50 | 11×11 px | **22×22 px** | 0.12% |
| 0.30 | 14×14 px | **28×28 px** | 0.19% |
| 0.15 | 20×20 px | **40×40 px** | 0.39% |

Against 90×90 / 116×116 / 165×165 before: about 8× smaller per side, 64× by area.

**This bounds the blind spot; it does not remove it.** A difference spread
thinly enough to leave every cell mean intact is invisible to any pooled
fingerprint. Claiming otherwise would be the same overclaim in a new size.

## Hit rate, and the bound that buys it

40 frames built from `bus.jpg`. *Fixed camera* is the case the default was
chosen for — a still scene with one 48×48 px object moving 6 px per frame.
*Panning* moves every pixel by 2 px per frame and is the worst case for any
fingerprint.

| threshold | fixed camera | panning | invisible object (straddling, 0.15 contrast) |
|---|---|---|---|
| 0.005 | 2.5% | 0.0% | 28×28 px |
| **0.010** (default) | **17.5%** | **0.0%** | **40×40 px** |
| 0.020 | 37.5% | 50.0% | 58×58 px |
| 0.050 | 70.0% | 75.0% | 92×92 px |
| 0.100 | 85.0% | 85.0% | 130×130 px |

Every row of hit rate is bought with a row of blindness. They are one number
seen from two sides, and the original claim stated only the flattering side.

## What it actually saves

Median per-frame latency, fixed camera, same 40 frames, through the real
backend path (`PyTorchBackend` with `set_source_id`, because the cache is only
consulted when a frame carries a source id — `engine.py:1030-1033` — and
`detect()` on a bare array supplies none).

| device | threshold | cache off | cache on | saving | blind spot |
|---|---|---|---|---|---|
| cpu | 0.01 | 34.50 ms | 40.22 ms | **−16.6%** | 40×40 px |
| cpu | 0.05 | 33.17 ms | 18.55 ms | **+44.1%** | 92×92 px |
| cpu | 0.10 | 36.02 ms | 17.14 ms | **+52.4%** | 130×130 px |
| mps | 0.01 | 6.27 ms | 19.52 ms | **−211.3%** | 40×40 px |
| mps | 0.05 | 6.27 ms | 12.08 ms | **−92.7%** | 92×92 px |
| mps | 0.10 | 6.38 ms | 11.75 ms | **−84.1%** | 130×130 px |

**"60–85% compute savings" is not reachable at any threshold measured.** The
best result is +52.4%, CPU only, at a 130×130 px blind spot — which is roughly
the hole the old fingerprint had in the first place.

**On MPS the cache costs time at every threshold**, and costs most where it is
safest. A hit copies roughly 6.4 MB of neck features host→device
(P3 1×144×80×80, P4 1×288×40×40, P5 1×576×20×20 at fp32) and a miss copies them
back, so a faster device loses harder: MPS inference is 6.3 ms and the transfer
alone exceeds it.

## What changed as a result

- `cache=True` was **removed from the `(APPLE_SILICON, VIDEO)` preset**, where
  it was measured 3.1× slower at the shipped threshold.
- `(CUDA_HIGH, VIDEO)` **keeps** `cache=True` and is **unmeasured** — there is no
  CUDA device here. It is recorded as unmeasured rather than changed on a guess,
  which is the thing this milestone exists to stop. Someone with a CUDA device
  should re-run `scripts/measure_feature_cache.py --devices cuda`.

## Caveats

One model (yolo11n), one machine, one image, fp32, 40 frames, median. A number
measured on one machine is not a property of the code. What reproduces is the
shape of the trade — hit rate and blindness move together, and the host↔device
copy dominates on a fast device — not the exact percentages.
