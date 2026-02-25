"""Side-by-side benchmark: yowo vs ultralytics — latency, CPU, memory.

Loads identical weights into both implementations and reports:
  - Latency: mean/p50/p95/p99 + FPS
  - CPU utilization (%) sampled during inference loop     [--profile]
  - RSS memory footprint (MB) of each loaded model        [--profile]
  - Model parameter count (M)                             [--profile]

Usage:
    uv run python bench/bench_compare.py --all --device cpu
    uv run python bench/bench_compare.py --all --profile
    uv run python bench/bench_compare.py --all --compile
    uv run python bench/bench_compare.py --model yolo26n --weights bench/weights/yolo26n.pt

Requirements:
    ultralytics:  uv add ultralytics --group dev
    psutil:       uv add psutil --group dev   (for --profile)
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import threading
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
# Result types
# ---------------------------------------------------------------------------


class LatencyStats(NamedTuple):
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float


class ResourceStats(NamedTuple):
    rss_mb: float
    cpu_pct: float
    params_m: float


class CompareResult(NamedTuple):
    name: str
    device: str
    compile: bool
    ul_lat: LatencyStats | None
    yowo_lat: LatencyStats | None
    ul_res: ResourceStats | None
    yowo_res: ResourceStats | None
    speedup: float
    error: str


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------


class _CpuSampler:
    def __init__(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process()
            self._proc.cpu_percent(interval=None)
            self._available = True
        except ImportError:
            self._available = False
            self._proc = None
        self._samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._available:
            return
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return mean(self._samples) if self._samples else 0.0

    def _loop(self) -> None:
        while self._running:
            try:
                self._samples.append(self._proc.cpu_percent(interval=0.05))
            except Exception:
                break


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _param_count_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_ultralytics(weights: Path, device: torch.device) -> torch.nn.Module:
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics not installed. Run: uv add ultralytics --group dev"
        ) from exc

    ul = YOLO(str(weights))
    model: torch.nn.Module = ul.model  # type: ignore[assignment]
    model.eval()
    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    return model


def _load_yowo(
    family: ModelFamily,
    size: ModelSize,
    weights: Path,
    device: torch.device,
    *,
    use_compile: bool,
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
        model.compile_for_inference(mode="reduce-overhead")
    return model


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _make_input(device: torch.device) -> Tensor:
    torch.manual_seed(42)
    x = torch.rand(1, 3, 640, 640, dtype=torch.float32, device=device)
    if device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    return x


def _time_model(
    model: torch.nn.Module,
    x: Tensor,
    *,
    n_warmup: int,
    n_iters: int,
    device: torch.device,
    do_profile: bool,
) -> tuple[LatencyStats, ResourceStats | None]:
    is_cuda = device.type == "cuda"

    def _fwd() -> None:
        with torch.inference_mode():
            model(x)

    for _ in range(n_warmup):
        _fwd()
        if is_cuda:
            torch.cuda.synchronize()  # type: ignore[attr-defined]

    sampler = _CpuSampler() if do_profile else None
    rss_before = _rss_mb() if do_profile else 0.0

    if sampler is not None:
        sampler.start()

    latencies: list[float] = []
    for _ in range(n_iters):
        if is_cuda:
            t0 = torch.cuda.Event(enable_timing=True)  # type: ignore[attr-defined]
            t1 = torch.cuda.Event(enable_timing=True)  # type: ignore[attr-defined]
            t0.record()  # type: ignore[attr-defined]
            _fwd()
            t1.record()  # type: ignore[attr-defined]
            torch.cuda.synchronize()  # type: ignore[attr-defined]
            latencies.append(t0.elapsed_time(t1))  # type: ignore[attr-defined]
        else:
            s = time.perf_counter()
            _fwd()
            latencies.append((time.perf_counter() - s) * 1000.0)

    cpu_pct = sampler.stop() if sampler is not None else 0.0
    rss_after = _rss_mb() if do_profile else 0.0

    m = mean(latencies)
    if len(latencies) >= 2:
        q = quantiles(latencies, n=100)
        p50 = median(latencies)
        p95 = q[94]
        p99 = q[98]
    else:
        p50 = p95 = p99 = latencies[0]

    lat = LatencyStats(mean_ms=m, p50_ms=p50, p95_ms=p95, p99_ms=p99, fps=1000.0 / m)
    res = (
        ResourceStats(
            rss_mb=max(rss_after, rss_before), cpu_pct=cpu_pct, params_m=_param_count_m(model)
        )
        if do_profile
        else None
    )
    return lat, res


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compare_model(
    name: str,
    weights: Path,
    device: torch.device,
    *,
    n_warmup: int,
    n_iters: int,
    use_compile: bool,
    do_profile: bool,
) -> CompareResult:
    if name not in _VARIANTS:
        return CompareResult(
            name, str(device), use_compile, None, None, None, None, 0.0, f"unknown '{name}'"
        )

    family, size = _VARIANTS[name]
    try:
        x = _make_input(device)

        ul_model = _load_ultralytics(weights, device)
        ul_lat, ul_res = _time_model(
            ul_model, x, n_warmup=n_warmup, n_iters=n_iters, device=device, do_profile=do_profile
        )
        del ul_model
        gc.collect()

        yowo_model = _load_yowo(family, size, weights, device, use_compile=use_compile)
        yowo_lat, yowo_res = _time_model(
            yowo_model, x, n_warmup=n_warmup, n_iters=n_iters, device=device, do_profile=do_profile
        )
        del yowo_model
        gc.collect()

        speedup = yowo_lat.fps / ul_lat.fps if ul_lat.fps > 0 else 0.0
        return CompareResult(
            name=name,
            device=str(device),
            compile=use_compile,
            ul_lat=ul_lat,
            yowo_lat=yowo_lat,
            ul_res=ul_res,
            yowo_res=yowo_res,
            speedup=speedup,
            error="",
        )
    except Exception as exc:
        return CompareResult(name, str(device), use_compile, None, None, None, None, 0.0, str(exc))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_C = 11
_HDR_LAT = (
    f"{'model':<{_C}}{'impl':<14}"
    f"{'mean(ms)':<{_C}}{'p50(ms)':<{_C}}{'p95(ms)':<{_C}}"
    f"{'fps':<9}{'speedup'}"
)
_HDR_RES = (
    f"{'model':<{_C}}{'impl':<14}"
    f"{'params(M)':<{_C}}{'rss(MB)':<{_C}}{'cpu%':<{_C}}"
    f"{'delta rss':<{_C}}{'delta cpu%'}"
)
_SEP = "-" * (len(_HDR_LAT) + 2)


def _print_latency(results: list[CompareResult]) -> None:
    print("\n--- LATENCY ---")
    print(_HDR_LAT)
    print(_SEP)
    for r in results:
        if r.error:
            print(f"  {r.name:<{_C}}  ERROR: {r.error}")
            continue
        assert r.ul_lat and r.yowo_lat
        tag = "+compile" if r.compile else ""
        speedup = f"{r.speedup:.2f}x {'(faster)' if r.speedup >= 1.0 else '(slower)'}"
        print(
            f"{r.name:<{_C}}{'ultralytics':<14}"
            f"{r.ul_lat.mean_ms:<{_C}.2f}{r.ul_lat.p50_ms:<{_C}.2f}{r.ul_lat.p95_ms:<{_C}.2f}"
            f"{r.ul_lat.fps:<9.1f}"
        )
        print(
            f"{'':_<{_C}}{f'yowo{tag}':<14}"
            f"{r.yowo_lat.mean_ms:<{_C}.2f}{r.yowo_lat.p50_ms:<{_C}.2f}{r.yowo_lat.p95_ms:<{_C}.2f}"
            f"{r.yowo_lat.fps:<9.1f}{speedup}"
        )
        print()


def _print_resources(results: list[CompareResult]) -> None:
    valid = [r for r in results if not r.error and r.ul_res and r.yowo_res]
    if not valid:
        return
    print("\n--- CPU & MEMORY ---")
    print(_HDR_RES)
    print(_SEP)
    for r in valid:
        assert r.ul_res and r.yowo_res
        tag = "+compile" if r.compile else ""
        d_rss = r.yowo_res.rss_mb - r.ul_res.rss_mb
        d_cpu = r.yowo_res.cpu_pct - r.ul_res.cpu_pct
        print(
            f"{r.name:<{_C}}{'ultralytics':<14}"
            f"{r.ul_res.params_m:<{_C}.1f}{r.ul_res.rss_mb:<{_C}.0f}{r.ul_res.cpu_pct:<{_C}.1f}"
        )
        print(
            f"{'':_<{_C}}{f'yowo{tag}':<14}"
            f"{r.yowo_res.params_m:<{_C}.1f}{r.yowo_res.rss_mb:<{_C}.0f}{r.yowo_res.cpu_pct:<{_C}.1f}"
            f"{f'{d_rss:+.0f} MB':<{_C}}{d_cpu:+.1f}%"
        )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Latency + resource comparison: yowo vs ultralytics.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(_VARIANTS), help="Single variant.")
    group.add_argument("--all", action="store_true", help="All found in bench/weights/.")
    p.add_argument("--weights", type=Path, help="Path to .pt (required with --model).")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu).")
    p.add_argument("--compile", action="store_true", help="torch.compile on yowo.")
    p.add_argument(
        "--profile", action="store_true", help="Measure RSS and CPU%% (requires psutil)."
    )
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

    print(
        f"Device: {device}  warmup={args.warmup}  iters={args.iters}"
        f"  compile={'yes' if args.compile else 'no'}"
        f"  profile={'yes' if args.profile else 'no'}"
    )

    logging.disable(logging.WARNING)

    results: list[CompareResult] = []
    for name, weights in jobs:
        sys.stdout.write(f"  {name}... ")
        sys.stdout.flush()
        r = compare_model(
            name,
            weights,
            device,
            n_warmup=args.warmup,
            n_iters=args.iters,
            use_compile=args.compile,
            do_profile=args.profile,
        )
        sys.stdout.write("done\n")
        sys.stdout.flush()
        results.append(r)

    _print_latency(results)
    if args.profile:
        _print_resources(results)

    valid = [r for r in results if not r.error]
    if valid:
        avg_sp = mean(r.speedup for r in valid)
        n_faster = sum(1 for r in valid if r.speedup >= 1.0)
        print(
            f"Summary: {len(valid)}/{len(results)} compared"
            f"  avg speedup: {avg_sp:.2f}x"
            f"  yowo faster: {n_faster}/{len(valid)}"
        )

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            print(f"  {r.name}: {r.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
