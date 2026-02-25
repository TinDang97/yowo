"""Setup script for yowo benchmarks.

Downloads model weights from ultralytics GitHub releases and optionally
installs benchmark-specific packages (onnxruntime, onnxslim, psutil).

Usage:
    uv run python bench/setup.py                        # all 10 variants
    uv run python bench/setup.py --models yolo26n       # single model
    uv run python bench/setup.py --no-weights           # check packages only
    uv run python bench/setup.py --install-extras       # download + install packages
    uv run python bench/setup.py --force                # re-download existing files
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Weight registry
# ---------------------------------------------------------------------------

_WEIGHT_URLS: dict[str, str] = {
    "yolo11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolo11m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolo11l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolo11x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    "yolo26n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolo26s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
    "yolo26m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
    "yolo26l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "yolo26x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
}

# ---------------------------------------------------------------------------
# Optional benchmark packages
# ---------------------------------------------------------------------------

_EXTRAS: dict[str, str] = {
    "onnxruntime": "onnxruntime>=1.17",
    "onnxslim": "onnxslim>=0.1.85",
    "onnx": "onnx>=1.12,<2.0",
    "psutil": "psutil",
}

_DEFAULT_WEIGHTS_DIR = Path(__file__).parent / "weights"

# ---------------------------------------------------------------------------
# Package checks
# ---------------------------------------------------------------------------


def _check_python() -> bool:
    ok = sys.version_info >= (3, 11)
    status = "OK" if ok else "FAIL"
    print(f"  [python {status}] {sys.version.split()[0]}  (required: >=3.11)")
    return ok


def _check_package(name: str) -> bool:
    try:
        __import__(name.replace("-", "_").split(">=")[0].split("<")[0])
        print(f"  [  OK  ] {name}")
        return True
    except ImportError:
        print(f"  [ MISS ] {name}")
        return False


def check_packages() -> dict[str, bool]:
    print("\n=== Package Check ===")
    results: dict[str, bool] = {}
    for name in ["torch", "numpy", "cv2", *_EXTRAS]:
        results[name] = _check_package(name)
    return results


# ---------------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------------


def _pip_install(packages: list[str]) -> bool:
    """Install packages using uv pip (fallback: pip)."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", *packages]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *packages]

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def install_extras(pkg_status: dict[str, bool]) -> None:
    missing = [spec for name, spec in _EXTRAS.items() if not pkg_status.get(name, False)]
    if not missing:
        print("\n  All optional packages already installed.")
        return

    print(f"\n  Installing: {', '.join(missing)}")
    if not _pip_install(missing):
        print("  ERROR: installation failed — check pip output above", file=sys.stderr)
        sys.exit(1)
    print("  Done.")


# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------


def _download_weight(name: str, url: str, dest: Path, *, force: bool) -> bool:
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  [SKIP] {name:<10}  already exists ({size_mb:.1f} MB)")
        return True

    print(f"  [DOWN] {name:<10}  {url}", flush=True)
    try:
        import requests
        from tqdm import tqdm

        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            try:
                with (
                    open(tmp, "wb") as fh,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"    {name}",
                        leave=False,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
                        bar.update(len(chunk))
                tmp.replace(dest)
                size_mb = dest.stat().st_size / 1_048_576
                print(f"         → {dest} ({size_mb:.1f} MB)")
                return True
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

    except ImportError as exc:
        print(f"  ERROR: {exc} — run 'uv sync --group dev' first", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  ERROR: {name}: {exc}", file=sys.stderr)
        return False


def download_weights(
    models: list[str],
    weights_dir: Path,
    *,
    force: bool,
) -> None:
    print(f"\n=== Downloading Weights → {weights_dir} ===")
    weights_dir.mkdir(parents=True, exist_ok=True)

    passed = failed = 0
    for name in models:
        if name not in _WEIGHT_URLS:
            print(f"  [ ERR ] Unknown model: {name}")
            failed += 1
            continue
        url = _WEIGHT_URLS[name]
        dest = weights_dir / f"{name}.pt"
        if _download_weight(name, url, dest, force=force):
            passed += 1
        else:
            failed += 1

    print(f"\n  Downloaded: {passed}  |  Failed/Skipped: {failed}")
    if failed and passed == 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download yowo benchmark weights and verify packages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models",
        default=",".join(sorted(_WEIGHT_URLS)),
        help=(
            "Comma-separated model names to download "
            f"(default: all 10). Available: {', '.join(sorted(_WEIGHT_URLS))}"
        ),
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=_DEFAULT_WEIGHTS_DIR,
        help=f"Destination directory for .pt files (default: {_DEFAULT_WEIGHTS_DIR}).",
    )
    p.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip weight download; only check packages.",
    )
    p.add_argument(
        "--install-extras",
        action="store_true",
        help="Install missing optional packages (onnxruntime, onnxslim, onnx, psutil).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download weights even if the file already exists.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    print("=== yowo Benchmark Setup ===")
    ok = _check_python()
    if not ok:
        sys.exit(1)

    pkg_status = check_packages()

    if args.install_extras:
        install_extras(pkg_status)

    if not args.no_weights:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models if m not in _WEIGHT_URLS]
        if unknown:
            print(f"\nERROR: unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Valid: {', '.join(sorted(_WEIGHT_URLS))}", file=sys.stderr)
            sys.exit(1)
        download_weights(models, args.weights_dir, force=args.force)

    print("\n=== Setup Complete ===")
    print(f"Weights dir : {args.weights_dir}")
    print("Next steps  :")
    print("  uv run python bench/bench_arch.py --all --device cpu")
    print("  uv run python bench/bench_onnx.py --all")
    print("  uv run python bench/bench_compare.py --all")


if __name__ == "__main__":
    main()
