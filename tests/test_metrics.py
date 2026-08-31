import pytest
from pagedserve.metrics import EngineMetrics, LatencyStats


def test_latency_stats_empty():
    stats = LatencyStats()
    d = stats.to_dict()
    assert d["count"] == 0
    assert d["mean"] is None


def test_latency_stats_recording():
    stats = LatencyStats()
    stats.record(0.1)
    stats.record(0.2)
    stats.record(0.3)
    d = stats.to_dict()
    assert d["count"] == 3
    assert d["mean"] == 0.2
    assert d["p50"] is not None


def test_engine_metrics_lifecycle():
    metrics = EngineMetrics()
    metrics.kv_blocks_total = 100
    metrics.kv_blocks_used = 40

    metrics.on_request_submitted(prompt_tokens=10)
    assert metrics.requests_total == 1
    assert metrics.prompt_tokens_total == 10

    metrics.on_request_completed(generated_tokens=5, ttft=0.05, total_latency=0.25)
    assert metrics.requests_completed == 1
    assert metrics.generated_tokens_total == 5
    assert metrics.kv_blocks_free == 60
    assert metrics.kv_utilization == 0.4

    d = metrics.to_dict()
    assert d["requests"]["total"] == 1
    assert d["requests"]["completed"] == 1
    assert d["kv_cache"]["free_blocks"] == 60
