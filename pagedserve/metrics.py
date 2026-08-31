"""Internal metrics collection for PagedServe engine telemetry."""

import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class LatencyStats:
    """Aggregated latency statistics for a sequence of measurements."""

    count: int = 0
    total: float = 0.0
    _samples: List[float] = field(default_factory=list, repr=False)

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self._samples.append(value)

    def to_dict(self) -> Dict:
        if not self._samples:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        sorted_s = sorted(self._samples)
        n = len(sorted_s)

        def pct(p: float) -> float:
            idx = max(0, int(p * n / 100) - 1)
            return round(sorted_s[idx], 6)

        return {
            "count": self.count,
            "mean": round(self.total / self.count, 6),
            "p50": pct(50),
            "p95": pct(95),
            "p99": pct(99),
        }


class EngineMetrics:
    """Central telemetry store for the PagedServe engine."""

    def __init__(self) -> None:
        self._start_time: float = time.monotonic()

        # Request counters
        self.requests_total: int = 0
        self.requests_completed: int = 0
        self.requests_cancelled: int = 0
        self.requests_failed: int = 0

        # Token counters
        self.prompt_tokens_total: int = 0
        self.generated_tokens_total: int = 0

        # Prefix cache
        self.prefix_cache_hits: int = 0
        self.prefix_cache_misses: int = 0

        # Latency distributions (seconds)
        self.ttft: LatencyStats = LatencyStats()  # time to first token
        self.tpot: LatencyStats = LatencyStats()  # time per output token (decode)
        self.e2e_latency: LatencyStats = LatencyStats()  # full request latency

        # Running state (set externally by engine at each step)
        self.requests_waiting: int = 0
        self.requests_running: int = 0
        self.kv_blocks_total: int = 0
        self.kv_blocks_used: int = 0

    def on_request_submitted(self, prompt_tokens: int) -> None:
        self.requests_total += 1
        self.prompt_tokens_total += prompt_tokens

    def on_request_completed(
        self, generated_tokens: int, ttft: Optional[float], total_latency: Optional[float]
    ) -> None:
        self.requests_completed += 1
        self.generated_tokens_total += generated_tokens
        if ttft is not None:
            self.ttft.record(ttft)
        if total_latency is not None:
            self.e2e_latency.record(total_latency)
            if generated_tokens > 1 and ttft is not None:
                decode_time = total_latency - ttft
                decode_tokens = max(1, generated_tokens - 1)
                self.tpot.record(decode_time / decode_tokens)

    def on_request_cancelled(self) -> None:
        self.requests_cancelled += 1

    def on_request_failed(self) -> None:
        self.requests_failed += 1

    def on_prefix_cache_hit(self) -> None:
        self.prefix_cache_hits += 1

    def on_prefix_cache_miss(self) -> None:
        self.prefix_cache_misses += 1

    @property
    def kv_blocks_free(self) -> int:
        return max(0, self.kv_blocks_total - self.kv_blocks_used)

    @property
    def kv_utilization(self) -> float:
        if self.kv_blocks_total == 0:
            return 0.0
        return self.kv_blocks_used / self.kv_blocks_total

    @property
    def prefix_cache_hit_rate(self) -> float:
        total = self.prefix_cache_hits + self.prefix_cache_misses
        if total == 0:
            return 0.0
        return self.prefix_cache_hits / total

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def throughput_tokens_per_sec(self) -> float:
        uptime = self.uptime_seconds
        if uptime <= 0:
            return 0.0
        return self.generated_tokens_total / uptime

    def to_dict(self) -> Dict:
        return {
            "uptime_seconds": round(self.uptime_seconds, 3),
            "requests": {
                "total": self.requests_total,
                "completed": self.requests_completed,
                "cancelled": self.requests_cancelled,
                "failed": self.requests_failed,
                "waiting": self.requests_waiting,
                "running": self.requests_running,
            },
            "tokens": {
                "prompt_total": self.prompt_tokens_total,
                "generated_total": self.generated_tokens_total,
                "throughput_per_sec": round(self.throughput_tokens_per_sec, 3),
            },
            "kv_cache": {
                "total_blocks": self.kv_blocks_total,
                "used_blocks": self.kv_blocks_used,
                "free_blocks": self.kv_blocks_free,
                "utilization": round(self.kv_utilization, 4),
            },
            "prefix_cache": {
                "hits": self.prefix_cache_hits,
                "misses": self.prefix_cache_misses,
                "hit_rate": round(self.prefix_cache_hit_rate, 4),
            },
            "latency": {
                "ttft_seconds": self.ttft.to_dict(),
                "tpot_seconds": self.tpot.to_dict(),
                "e2e_seconds": self.e2e_latency.to_dict(),
            },
        }
