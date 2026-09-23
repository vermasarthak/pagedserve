"""TTFT & ITL Serving Performance Benchmark Harness for PagedServe.

Simulates Poisson arrival request loads and measures Time-To-First-Token (TTFT)
and Inter-Token Latency (ITL) P50, P90, and P99 metrics.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class ServingMetrics:
    """Aggregated LLM serving performance telemetry metrics."""
    total_requests: int
    completed_requests: int
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p90_ms: float
    itl_p99_ms: float
    throughput_tokens_per_sec: float


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s_data = sorted(data)
    idx = int(len(s_data) * pct / 100.0)
    idx = min(max(idx, 0), len(s_data) - 1)
    return s_data[idx]


class ServingBenchmarkHarness:
    """Harness for benchmarking LLM serving TTFT and ITL under Poisson traffic pattern."""

    def __init__(self, request_rate_qps: float = 10.0, num_requests: int = 50):
        self.request_rate_qps = request_rate_qps
        self.num_requests = num_requests

    def run_simulation(self) -> ServingMetrics:
        """Simulates synthetic Poisson request dispatch and measures TTFT and ITL telemetry."""
        ttfts: list[float] = []
        itls: list[float] = []
        total_output_tokens = 0

        start_wall_time = time.perf_counter()

        for i in range(self.num_requests):
            # Inter-arrival delay according to Poisson process (exponential distribution)
            arrival_delay = random.expovariate(self.request_rate_qps)
            time.sleep(min(arrival_delay, 0.005)) # Scaled for fast unit testing

            # Simulate prefill (TTFT)
            prompt_len = random.randint(32, 256)
            ttft_sim = (prompt_len * 0.0001) + random.uniform(0.002, 0.008)
            ttfts.append(ttft_sim * 1000.0)

            # Simulate decode steps (ITL)
            output_len = random.randint(16, 64)
            total_output_tokens += output_len

            for step in range(output_len):
                itl_sim = random.uniform(0.001, 0.004)
                itls.append(itl_sim * 1000.0)

        elapsed = time.perf_counter() - start_wall_time
        throughput = total_output_tokens / max(elapsed, 1e-6)

        return ServingMetrics(
            total_requests=self.num_requests,
            completed_requests=self.num_requests,
            ttft_p50_ms=percentile(ttfts, 50),
            ttft_p90_ms=percentile(ttfts, 90),
            ttft_p99_ms=percentile(ttfts, 99),
            itl_p50_ms=percentile(itls, 50),
            itl_p90_ms=percentile(itls, 90),
            itl_p99_ms=percentile(itls, 99),
            throughput_tokens_per_sec=throughput,
        )


if __name__ == "__main__":
    harness = ServingBenchmarkHarness(request_rate_qps=20.0, num_requests=30)
    metrics = harness.run_simulation()
    print("=" * 60)
    print("PAGEDSERVE TTFT & ITL BENCHMARK REPORT")
    print("=" * 60)
    print(f"Total Requests:      {metrics.total_requests}")
    print(f"TTFT P50:            {metrics.ttft_p50_ms:.2f} ms")
    print(f"TTFT P90:            {metrics.ttft_p90_ms:.2f} ms")
    print(f"TTFT P99:            {metrics.ttft_p99_ms:.2f} ms")
    print(f"ITL P50:             {metrics.itl_p50_ms:.2f} ms")
    print(f"ITL P90:             {metrics.itl_p90_ms:.2f} ms")
    print(f"ITL P99:             {metrics.itl_p99_ms:.2f} ms")
    print(f"Throughput:          {metrics.throughput_tokens_per_sec:.2f} tok/s")
    print("=" * 60)
