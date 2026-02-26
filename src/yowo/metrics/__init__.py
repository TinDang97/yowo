"""Inference metrics collection for yowo.

Provides zero-dependency, thread-safe metrics for monitoring InferenceEngine
latency, throughput, and error rates in production.

Usage::

    engine = InferenceEngine(metrics_enabled=True)
    engine.load()
    # ... run inference ...
    m = engine.metrics
    print(f"{m.fps:.1f} FPS  p99={m.inference_p99_ms:.1f}ms")
"""

from yowo.metrics._collector import EngineMetrics, MetricsCollector

__all__ = ["EngineMetrics", "MetricsCollector"]
