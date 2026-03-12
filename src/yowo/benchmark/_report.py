"""Benchmark report rendering: rich tables and JSON serialization.

Renders benchmark results as colored terminal tables using ``rich``
and produces structured JSON for programmatic consumption.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table

from yowo.benchmark._runner import BenchmarkResult


def render_table(
    results: list[BenchmarkResult],
    ultralytics_results: dict[str, float] | None = None,
    model_name: str = "",
) -> None:
    """Render benchmark results as a rich terminal table.

    Args:
        results: List of benchmark results to display.
        ultralytics_results: Optional ultralytics reference metrics.
        model_name: Model name for the table title.
    """
    title = f"YOWO Benchmark: {model_name}" if model_name else "YOWO Benchmark"
    table = Table(title=title)

    table.add_column("Format", style="cyan")
    table.add_column("mAP@0.5:0.95", justify="right")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("Latency p50", justify="right")
    table.add_column("Model Size", justify="right")
    table.add_column("Device", style="magenta")

    if ultralytics_results is not None:
        table.add_column("Ultralytics mAP", justify="right", style="yellow")

    ultra_map = ultralytics_results.get("mAP_50_95") if ultralytics_results else None

    for r in results:
        map_str = f"{r.map_50_95:.4f}" if r.map_50_95 is not None else "N/A"
        fps_str = f"{r.fps_avg:.1f}"
        lat_str = f"{r.latency_p50_ms:.1f} ms"
        size_str = f"{r.model_size_mb:.1f} MB"

        row: list[str] = [r.format, map_str, fps_str, lat_str, size_str, r.device]

        if ultralytics_results is not None:
            if ultra_map is not None and r.map_50_95 is not None:
                delta = r.map_50_95 - ultra_map
                sign = "+" if delta >= 0 else ""
                row.append(f"{ultra_map:.4f} ({sign}{delta:.4f})")
            else:
                row.append("N/A")

        table.add_row(*row)

    console = Console()
    console.print(table)


def results_to_json(
    results: list[BenchmarkResult],
    ultralytics_results: dict[str, float] | None = None,
    model_name: str = "",
    device_info: str = "",
) -> dict[str, Any]:
    """Serialize benchmark results to a JSON-compatible dict.

    Args:
        results: List of benchmark results.
        ultralytics_results: Optional ultralytics reference metrics.
        model_name: Model name identifier.
        device_info: Device description string.

    Returns:
        Dict that is safe for ``json.dumps()``.
    """
    ultra_map = ultralytics_results.get("mAP_50_95") if ultralytics_results else None

    result_dicts: list[dict[str, Any]] = []
    for r in results:
        rd: dict[str, Any] = {
            "format": r.format,
            "map_50_95": r.map_50_95,
            "map_50": r.map_50,
            "fps_avg": r.fps_avg,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_p99_ms": r.latency_p99_ms,
            "model_size_mb": r.model_size_mb,
            "device": r.device,
            "num_images": r.num_images,
        }
        if ultra_map is not None and r.map_50_95 is not None:
            rd["delta_map"] = round(r.map_50_95 - ultra_map, 6)
        result_dicts.append(rd)

    output: dict[str, Any] = {
        "model": model_name,
        "device": device_info,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "results": result_dicts,
    }

    if ultralytics_results is not None:
        output["ultralytics"] = ultralytics_results

    return output


__all__ = ["render_table", "results_to_json"]
