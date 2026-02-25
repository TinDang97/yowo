"""Benchmark PyTorch vs ONNX Runtime across all yowo model variants.

Exports each variant to ONNX on first run (cached to bench/exports/).
Reports mean/p50/p95/p99 latency and FPS for each backend.

Usage:
    uv run python bench/bench_onnx.py                     # all 10 variants
    uv run python bench/bench_onnx.py --model yolo26n
    uv run python bench/bench_onnx.py --no-export          # use cached ONNX
    uv run python bench/bench_onnx.py --coreml             # include CoreML EP (macOS arm64)
    uv run python bench/bench_onnx.py --kv-cache           # include KV-cache ONNX
    uv run python bench/bench_onnx.py --coreml --kv-cache  # all backends
"""

from __future__ import annotations

import argparse
import gc
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yowo.arch import build_model  # noqa: E402
from yowo.arch._weights import load_weights  # noqa: E402
from yowo.backends._onnx import OnnxBackend  # noqa: E402
from yowo.backends._pytorch import PyTorchBackend  # noqa: E402
from yowo.hardware import get_hardware_profile  # noqa: E402
from yowo.types import ModelFamily, ModelSize, ModelSpec, PreprocessedTensor  # noqa: E402

WEIGHTS_DIR = ROOT / "bench" / "weights"
EXPORTS_DIR = ROOT / "bench" / "exports"
WARMUP = 5
RUNS = 50
IMGSZ = 640

VARIANTS: list[tuple[str, ModelFamily, ModelSize]] = [
    ("yolo11n", ModelFamily.YOLO11, ModelSize.NANO),
    ("yolo11s", ModelFamily.YOLO11, ModelSize.SMALL),
    ("yolo11m", ModelFamily.YOLO11, ModelSize.MEDIUM),
    ("yolo11l", ModelFamily.YOLO11, ModelSize.LARGE),
    ("yolo11x", ModelFamily.YOLO11, ModelSize.XLARGE),
    ("yolo26n", ModelFamily.YOLO26, ModelSize.NANO),
    ("yolo26s", ModelFamily.YOLO26, ModelSize.SMALL),
    ("yolo26m", ModelFamily.YOLO26, ModelSize.MEDIUM),
    ("yolo26l", ModelFamily.YOLO26, ModelSize.LARGE),
    ("yolo26x", ModelFamily.YOLO26, ModelSize.XLARGE),
]


def _rss_mb() -> float:
    """Current process RSS in MB (macOS: bytes; Linux: KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1_048_576 if sys.platform == "darwin" else usage / 1_024


def _make_dummy_tensor() -> PreprocessedTensor:
    data = np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
    return PreprocessedTensor(
        data=data,
        original_shapes=((IMGSZ, IMGSZ),),
        input_shape=(IMGSZ, IMGSZ),
        scale_factors=((1.0, 1.0),),
        pad_offsets=((0, 0),),
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _export_onnx(stem: str, family: ModelFamily, size: ModelSize) -> Path:
    """Export model to ONNX + simplify with onnxslim. Skips if cached."""
    import torch

    out = EXPORTS_DIR / f"{stem}.onnx"
    if out.exists():
        return out

    weights = WEIGHTS_DIR / f"{stem}.pt"
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(family, size)
    load_weights(model, weights)
    model = model.fuse().eval()

    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=18,
        input_names=["images"],
        output_names=["output0"],
    )

    try:
        import onnx
        import onnxslim

        slim = onnxslim.slim(str(out))
        onnx.save(slim, str(out))
        print(f"  {stem:<10} onnxslim OK", flush=True)
    except Exception as e:
        print(f"  {stem:<10} onnxslim SKIP — {e}", flush=True)

    return out


def _export_onnx_kv(stem: str, family: ModelFamily, size: ModelSize) -> Path:
    """Export model with KV cache as explicit I/O. Skips if cached."""
    import torch

    from yowo.export._kv_wrapper import YOLOKVWrapper

    kv_dir = EXPORTS_DIR / "kv"
    out = kv_dir / f"{stem}.onnx"
    if out.exists():
        return out

    weights = WEIGHTS_DIR / f"{stem}.pt"
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    kv_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(family, size)
    load_weights(model, weights)
    model = model.fuse().eval()

    wrapper = YOLOKVWrapper(model)
    if wrapper.num_attn == 0:
        raise ValueError(f"{stem} has no Attention modules — KV cache N/A")

    dummy_images = torch.zeros(1, 3, IMGSZ, IMGSZ)
    dummy_kvs = wrapper.build_dummy_kv_inputs(IMGSZ, dtype=dummy_images.dtype)
    use_cache_t = torch.tensor(0.0, dtype=dummy_images.dtype)
    dummy_inputs = (dummy_images, use_cache_t, *dummy_kvs)

    input_names = ["images", *wrapper.kv_input_names]
    output_names = ["output0", *wrapper.kv_output_names]

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(out),
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
    )

    # Internalize external tensor data (CoreML EP needs single file).
    try:
        import onnx

        m = onnx.load(str(out), load_external_data=True)
        tmp = out.with_suffix(".onnx.tmp")
        onnx.save(m, str(tmp))
        tmp.replace(out)
        out.with_suffix(".onnx.data").unlink(missing_ok=True)
    except ImportError:
        pass
    except Exception as e:
        out.with_suffix(".onnx.tmp").unlink(missing_ok=True)
        print(f"  {stem:<10} kv internalize WARN — {e}", flush=True)

    # Simplify with onnxslim — KV I/O nodes survive since they have real data deps.
    try:
        import onnx
        import onnxslim

        slim_model = onnxslim.slim(str(out))
        slim_ins = {i.name for i in slim_model.graph.input}
        slim_outs = {o.name for o in slim_model.graph.output}
        kv_io_ok = all(n in slim_ins for n in input_names) and all(
            n in slim_outs for n in output_names
        )
        if kv_io_ok:
            onnx.save(slim_model, str(out))
            print(f"  {stem:<10} kv onnxslim OK ({wrapper.num_attn} attn)", flush=True)
        else:
            print(f"  {stem:<10} kv onnxslim reverted — KV I/O stripped", flush=True)
    except Exception as e:
        print(f"  {stem:<10} kv onnxslim SKIP — {e}", flush=True)

    return out


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench(backend: object, tensor: PreprocessedTensor, runs: int) -> list[float]:
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        backend.infer(tensor)  # type: ignore[union-attr]
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _bench_kv(
    session: object,
    tensor: PreprocessedTensor,
    runs: int,
) -> tuple[float, list[float]]:
    """Benchmark KV-cache ORT session. Returns (cold_ms, warm_times_ms)."""
    import onnxruntime as ort

    assert isinstance(session, ort.InferenceSession)
    inputs = session.get_inputs()
    kv_inputs = [i for i in inputs if i.name.startswith("past_")]
    kv_output_names = [o.name for o in session.get_outputs() if o.name.startswith("present_")]

    kv_state: dict[str, np.ndarray] = {
        ki.name: np.zeros(ki.shape, dtype=np.float32) for ki in kv_inputs
    }

    def _run(use_cache_val: float) -> tuple[float, dict[str, np.ndarray]]:
        feed: dict[str, np.ndarray] = {
            "images": tensor.data,
            "use_cache": np.array(use_cache_val, dtype=np.float32),
            **{ki.name: kv_state[ki.name] for ki in kv_inputs},
        }
        t0 = time.perf_counter()
        outputs = session.run(None, feed)
        elapsed = (time.perf_counter() - t0) * 1000.0
        new_state = {
            present_name.replace("present_", "past_"): outputs[1 + idx]
            for idx, present_name in enumerate(kv_output_names)
        }
        return elapsed, new_state

    cold_ms, kv_state = _run(0.0)
    warm_times: list[float] = []
    for _ in range(runs):
        ms, kv_state = _run(1.0)
        warm_times.append(ms)
    return cold_ms, warm_times


def _stats(times: list[float]) -> tuple[float, float, float, float, float]:
    s = sorted(times)
    n = len(s)
    m = statistics.mean(s)
    return m, s[n // 2], s[int(n * 0.95)], s[int(n * 0.99)], 1000.0 / m


def _make_ort_session(model_path: Path, providers: list[str], cpu_count: int) -> object:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, cpu_count // 2)
    opts.inter_op_num_threads = max(1, cpu_count // 4)
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HDR = (
    f"{'model':<10} {'backend':<14}"
    f" {'mean(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10} {'fps':>8}"
)
_SEP = "-" * len(_HDR)


def _fmt_row(
    stem: str,
    label: str,
    stats: tuple[float, float, float, float, float],
    ref: tuple[float, float, float, float, float] | None = None,
    suffix: str = "",
) -> str:
    m, p50, p95, p99, fps = stats
    speedup = f"   ({ref[0] / m:+.2f}x vs pytorch)" if ref else ""
    return (
        f"{stem:<10} {label:<14}"
        f" {m:>10.2f} {p50:>10.2f} {p95:>10.2f} {p99:>10.2f} {fps:>8.1f}{speedup}{suffix}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark PyTorch vs ONNX Runtime (+ CoreML, KV cache)."
    )
    p.add_argument("--model", default=None, help="Single model (e.g. yolo26n).")
    p.add_argument("--no-export", action="store_true", help="Skip export; use cached ONNX.")
    p.add_argument("--coreml", action="store_true", help="Include CoreML EP (macOS arm64 only).")
    p.add_argument("--kv-cache", action="store_true", help="Include KV-cache ONNX benchmarks.")
    p.add_argument(
        "--runs", type=int, default=RUNS, help=f"Inference iterations (default: {RUNS})."
    )
    p.add_argument(
        "--warmup", type=int, default=WARMUP, help=f"Warmup iterations (default: {WARMUP})."
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    targets = VARIANTS
    if args.model:
        targets = [v for v in VARIANTS if v[0] == args.model]
        if not targets:
            print(f"Unknown model: {args.model}. Valid: {[v[0] for v in VARIANTS]}")
            sys.exit(1)

    import os

    import torch

    cpu_count = os.cpu_count() or 1
    try:
        torch.set_num_threads(max(1, cpu_count // 2))
        torch.set_num_interop_threads(max(1, min(2, cpu_count // 4)))
    except RuntimeError:
        pass

    hw = get_hardware_profile()
    runs = args.runs
    warmup = args.warmup

    # Phase 1: PyTorch benchmarks + ONNX exports
    pt_results: dict[str, tuple[float, float, float, float, float] | None] = {}
    onnx_paths: dict[str, Path] = {}
    kv_paths: dict[str, Path] = {}

    for stem, family, size in targets:
        pt_path = WEIGHTS_DIR / f"{stem}.pt"
        if not pt_path.exists():
            continue

        if not args.no_export:
            try:
                onnx_paths[stem] = _export_onnx(stem, family, size)
            except Exception as e:
                print(f"  {stem:<10} SKIP export — {e}", flush=True)
        else:
            cached = EXPORTS_DIR / f"{stem}.onnx"
            if cached.exists():
                onnx_paths[stem] = cached

        if args.kv_cache:
            if not args.no_export:
                try:
                    kv_paths[stem] = _export_onnx_kv(stem, family, size)
                except Exception as e:
                    print(f"  {stem:<10} SKIP kv export — {e}", flush=True)
            else:
                cached_kv = EXPORTS_DIR / "kv" / f"{stem}.onnx"
                if cached_kv.exists():
                    kv_paths[stem] = cached_kv

        tensor = _make_dummy_tensor()
        try:
            spec = ModelSpec(family=family, size=size)
            pt_backend = PyTorchBackend(hw, model_spec=spec)
            pt_backend.load(pt_path, device="cpu")
            for _ in range(warmup):
                pt_backend.infer(tensor)
            pt_results[stem] = _stats(_bench(pt_backend, tensor, runs))
        except Exception as e:
            pt_results[stem] = None
            print(f"  {stem:<10} pytorch ERROR — {e}", flush=True)

    # Phase 2: ORT benchmarks
    print()
    print(_HDR)
    print(_SEP)

    for stem, _family, _size in targets:
        tensor = _make_dummy_tensor()
        pt_r = pt_results.get(stem)

        if pt_r:
            print(_fmt_row(stem, "pytorch", pt_r))

        if stem in onnx_paths:
            try:
                ort_backend = OnnxBackend(hw)
                ort_backend.load(onnx_paths[stem], device="cpu")
                for _ in range(warmup):
                    ort_backend.infer(tensor)
                print(_fmt_row(stem, "onnx-cpu", _stats(_bench(ort_backend, tensor, runs)), pt_r))
                ort_backend.unload()
                del ort_backend
                gc.collect()
            except Exception as e:
                print(f"  {stem:<10} onnx-cpu ERROR — {e}")

        if args.coreml and stem in onnx_paths:
            try:
                session = _make_ort_session(
                    onnx_paths[stem],
                    ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                    cpu_count,
                )
                import onnxruntime as ort

                assert isinstance(session, ort.InferenceSession)
                input_name = session.get_inputs()[0].name
                for _ in range(warmup):
                    session.run(None, {input_name: tensor.data})
                cml_times: list[float] = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    session.run(None, {input_name: tensor.data})
                    cml_times.append((time.perf_counter() - t0) * 1000.0)
                print(_fmt_row(stem, "coreml", _stats(cml_times), pt_r))
                del session
                gc.collect()
            except Exception as e:
                print(f"  {stem:<10} coreml ERROR — {e}")

        if args.kv_cache and stem in kv_paths:
            try:
                session = _make_ort_session(kv_paths[stem], ["CPUExecutionProvider"], cpu_count)
                _bench_kv(session, tensor, warmup)
                cold_ms, warm_times = _bench_kv(session, tensor, runs)
                print(_fmt_row(stem, "kv-cpu", _stats(warm_times), pt_r, f"  cold={cold_ms:.1f}ms"))
                del session
                gc.collect()
            except Exception as e:
                print(f"  {stem:<10} kv-cpu ERROR — {e}")

        if args.coreml and args.kv_cache and stem in kv_paths:
            try:
                session = _make_ort_session(
                    kv_paths[stem],
                    ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                    cpu_count,
                )
                _bench_kv(session, tensor, warmup)
                cold_ms, warm_times = _bench_kv(session, tensor, runs)
                print(
                    _fmt_row(stem, "kv-coreml", _stats(warm_times), pt_r, f"  cold={cold_ms:.1f}ms")
                )
                del session
                gc.collect()
            except Exception as e:
                print(f"  {stem:<10} kv-coreml ERROR — {e}")

        print()

    print(_SEP)
    backends = "pytorch, onnx-cpu"
    if args.coreml:
        backends += ", coreml"
    if args.kv_cache:
        backends += ", kv-cpu"
    if args.coreml and args.kv_cache:
        backends += ", kv-coreml"
    rss = _rss_mb()
    print(
        f"device: cpu  |  warmup: {warmup}  |  runs: {runs}"
        f"  |  imgsz: {IMGSZ}  |  RSS: {rss:.0f} MB"
    )
    print(f"backends: {backends}")
    if args.kv_cache:
        print("kv-* rows show WARM frame latency (steady-state); cold= shows first-frame cost")


if __name__ == "__main__":
    main()
