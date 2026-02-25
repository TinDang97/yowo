"""Benchmark yowo native YOLO architecture inference latency.

Measures wall-clock latency per model variant:
  - eager vs torch.compile (CUDA only; compile regresses on CPU)
  - fp32 vs fp16 (CUDA only)

Usage:
    uv run python bench/bench_arch.py --all --device cpu
    uv run python bench/bench_arch.py --all --device cuda --compile
    uv run python bench/bench_arch.py --all --device cuda --fp16
    uv run python bench/bench_arch.py --model yolo26n --weights bench/weights/yolo26n.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median, quantiles
from typing import NamedTuple

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yowo.arch import build_model
from yowo.arch._weights import load_weights
from yowo.types import ModelFamily, ModelSize

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, tuple[ModelFamily, ModelSize]] = {
    "yolo11n": (ModelFamily.YOLO11, ModelSize.NANO),
    "yolo11s": (ModelFamily.YOLO11, ModelSize.SMALL),
    "yolo11m": (ModelFamily.YOLO11, ModelSize.MEDIUM),
    "yolo11l": (ModelFamily.YOLO11, ModelSize.LARGE),
    "yolo11x": (ModelFamily.YOLO11, ModelSize.XLARGE),
    "yolo26n": (ModelFamily.YOLO26, ModelSize.NANO),
    "yolo26s": (ModelFamily.YOLO26, ModelSize.SMALL),
    "yolo26m": (ModelFamily.YOLO26, ModelSize.MEDIUM),
    "yolo26l": (ModelFamily.YOLO26, ModelSize.LARGE),
    "yolo26x": (ModelFamily.YOLO26, ModelSize.XLARGE),
}

_WEIGHTS_DIR = Path(__file__).parent / "weights"

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class BenchResult(NamedTuple):
    name: str
    device: str
    compile: bool
    fp16: bool
    n_iters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    error: str


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def _load_model(
    family: ModelFamily,
    size: ModelSize,
    weights: Path,
    device: torch.device,
    *,
    use_compile: bool,
    use_fp16: bool,
) -> torch.nn.Module:
    model = build_model(family, size)
    load_weights(model, weights)
    model = model.fuse()
    model.eval()
    model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

    if use_compile:
        try:
            model.compile_for_inference(mode="reduce-overhead")
        except Exception as exc:
            raise RuntimeError(f"torch.compile failed: {exc}") from exc

    return model


def _make_input(device: torch.device) -> Tensor:
    x = torch.rand(1, 3, 640, 640, dtype=torch.float32, device=device)
    if device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    return x


def _run_bench(
    model: torch.nn.Module,
    x: Tensor,
    *,
    n_warmup: int,
    n_iters: int,
    use_fp16: bool,
    device: torch.device,
) -> list[float]:
    is_cuda = device.type == "cuda"

    def _forward() -> None:
        with torch.inference_mode():
            if use_fp16 and is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):  # type: ignore[attr-defined]
                    model(x)
            else:
                model(x)

    for _ in range(n_warmup):
        _forward()
        if is_cuda:
            torch.cuda.synchronize()  # type: ignore[attr-defined]

    latencies: list[float] = []
    for _ in range(n_iters):
        if is_cuda:
            t0 = torch.cuda.Event(enable_timing=True)  # type: ignore[attr-defined]
            t1 = torch.cuda.Event(enable_timing=True)  # type: ignore[attr-defined]
            t0.record()  # type: ignore[attr-defined]
            _forward()
            t1.record()  # type: ignore[attr-defined]
            torch.cuda.synchronize()  # type: ignore[attr-defined]
            latencies.append(t0.elapsed_time(t1))  # type: ignore[attr-defined]
        else:
            s = time.perf_counter()
            _forward()
            latencies.append((time.perf_counter() - s) * 1000)

    return latencies


def bench_model(
    name: str,
    weights: Path,
    device: torch.device,
    *,
    n_warmup: int,
    n_iters: int,
    use_compile: bool,
    use_fp16: bool,
) -> BenchResult:
    if name not in _VARIANTS:
        return BenchResult(
            name, str(device), use_compile, use_fp16, 0, 0, 0, 0, 0, 0, f"unknown model '{name}'"
        )

    family, size = _VARIANTS[name]
    try:
        model = _load_model(
            family, size, weights, device, use_compile=use_compile, use_fp16=use_fp16
        )
        x = _make_input(device)
        latencies = _run_bench(
            model, x, n_warmup=n_warmup, n_iters=n_iters, use_fp16=use_fp16, device=device
        )

        q = quantiles(latencies, n=100)
        return BenchResult(
            name=name,
            device=str(device),
            compile=use_compile,
            fp16=use_fp16,
            n_iters=n_iters,
            mean_ms=mean(latencies),
            p50_ms=median(latencies),
            p95_ms=q[94],
            p99_ms=q[98],
            fps=1000.0 / mean(latencies),
            error="",
        )
    except Exception as exc:
        return BenchResult(name, str(device), use_compile, use_fp16, 0, 0, 0, 0, 0, 0, str(exc))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HDR = (
    f"{'model':<12}{'device':<10}{'compile':<10}{'fp16':<8}"
    f"{'mean(ms)':<12}{'p50(ms)':<12}{'p95(ms)':<12}{'p99(ms)':<12}{'fps':<10}"
)
_SEP = "-" * len(_HDR)


def _print_result(r: BenchResult) -> None:
    if r.error:
        print(f"  {r.name:<10}  ERROR: {r.error}")
        return
    print(
        f"{r.name:<12}{r.device:<10}"
        f"{'yes' if r.compile else 'no':<10}"
        f"{'yes' if r.fp16 else 'no':<8}"
        f"{r.mean_ms:<12.2f}{r.p50_ms:<12.2f}{r.p95_ms:<12.2f}{r.p99_ms:<12.2f}{r.fps:<10.1f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark yowo YOLO architecture inference latency.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(_VARIANTS), help="Single model variant.")
    group.add_argument("--all", action="store_true", help="All variants in bench/weights/.")
    p.add_argument("--weights", type=Path, help="Path to .pt file (required with --model).")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu).")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (CUDA only).")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 autocast (CUDA only).")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10).")
    p.add_argument("--iters", type=int, default=100, help="Benchmark iterations (default: 100).")
    return p


def _resolve_jobs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.model:
        if args.weights is None:
            raise SystemExit(f"--weights required with --model {args.model}")
        return [(args.model, args.weights)]

    jobs: list[tuple[str, Path]] = []
    for name in sorted(_VARIANTS):
        pt = _WEIGHTS_DIR / f"{name}.pt"
        if pt.exists():
            jobs.append((name, pt))
    if not jobs:
        raise SystemExit(
            f"No .pt files found in {_WEIGHTS_DIR}\nRun:  uv run python bench/setup.py"
        )
    return jobs


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    jobs = _resolve_jobs(args)

    if args.fp16 and device.type != "cuda":
        print("Warning: --fp16 has no effect on non-CUDA devices.")
    if args.compile and device.type == "cpu":
        print("Warning: torch.compile typically regresses on CPU (30-43%). Consider --device cuda.")

    print(_HDR)
    print(_SEP)

    results: list[BenchResult] = []
    for name, weights in jobs:
        r = bench_model(
            name,
            weights,
            device,
            n_warmup=args.warmup,
            n_iters=args.iters,
            use_compile=args.compile,
            use_fp16=args.fp16,
        )
        _print_result(r)
        results.append(r)

    print(_SEP)
    passed = [r for r in results if not r.error]
    if passed:
        avg_fps = mean(r.fps for r in passed)
        print(
            f"\nSummary: {len(passed)}/{len(results)} variants"
            f"  avg FPS: {avg_fps:.1f}"
            f"  device: {args.device}"
            f"  compile: {'yes' if args.compile else 'no'}"
            f"  fp16: {'yes' if args.fp16 else 'no'}"
        )

    if any(r.error for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
