"""Scheduling policy interfaces and implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pagedserve.engine.request import InferenceRequest


class SchedulingPolicy(ABC):
    """Abstract interface for request prioritization and scheduling policies."""

    @abstractmethod
    def sort_waiting(self, requests: list[InferenceRequest]) -> list[InferenceRequest]:
        """Order waiting requests for admission consideration."""

    @abstractmethod
    def sort_running_decode(self, requests: list[InferenceRequest]) -> list[InferenceRequest]:
        """Order active decode requests for execution priority."""


class FCFSPolicy(SchedulingPolicy):
    """First-Come, First-Served (FCFS) Scheduling Policy.
    
    Requests are admitted strictly in order of their arrival timestamp.
    This guarantees fairness, absence of starvation under moderate load,
    and highly deterministic testing behavior.
    """

    def sort_waiting(self, requests: list[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(requests, key=lambda r: r.arrival_time)

    def sort_running_decode(self, requests: list[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(requests, key=lambda r: r.arrival_time)


@dataclass
class KVPressureLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MemoryAwarePolicy(SchedulingPolicy):
    """Experimental memory-pressure-adaptive scheduling policy.
    
    Adapts admission and prefill chunk size based on KV block utilization:
    - LOW (< low_threshold): Normal FCFS behavior.
    - MEDIUM (low_threshold to high_threshold): Reduce new admissions and chunk sizes.
    - HIGH (> high_threshold): Strongly restrict admissions; prioritize near-complete sequences.
    
    Starvation prevention: requests gain priority weight after waiting many iterations.
    This policy is EXPERIMENTAL. FCFS remains the default and is always available.
    """

    def __init__(
        self,
        low_threshold: float = 0.60,
        high_threshold: float = 0.85,
        starvation_penalty_iters: int = 50,
    ) -> None:
        if not (0.0 < low_threshold < high_threshold < 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < low_threshold < high_threshold < 1. "
                f"Got low={low_threshold}, high={high_threshold}"
            )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.starvation_penalty_iters = starvation_penalty_iters
        self._wait_iters: dict[str, int] = {}

    def pressure_level(self, kv_utilization: float) -> str:
        if kv_utilization >= self.high_threshold:
            return KVPressureLevel.HIGH
        elif kv_utilization >= self.low_threshold:
            return KVPressureLevel.MEDIUM
        return KVPressureLevel.LOW

    def record_wait_iteration(self, request_id: str) -> None:
        self._wait_iters[request_id] = self._wait_iters.get(request_id, 0) + 1

    def clear_wait_record(self, request_id: str) -> None:
        self._wait_iters.pop(request_id, None)

    def _priority_key(self, r: InferenceRequest) -> float:
        """Priority key: lower is better. Aging reduces priority score for long-waiting requests."""
        wait_iters = self._wait_iters.get(r.request_id, 0)
        aging_bonus = min(wait_iters / self.starvation_penalty_iters, 1.0)
        return r.arrival_time - aging_bonus

    def sort_waiting(self, requests: list[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(requests, key=self._priority_key)

    def sort_running_decode(self, requests: list[InferenceRequest]) -> list[InferenceRequest]:
        # Under pressure, prefer requests closest to completion (shortest remaining tokens)
        return sorted(requests, key=lambda r: r.remaining_tokens)

    def should_admit(
        self,
        kv_utilization: float,
        candidate: InferenceRequest,
        num_running: int,
        max_num_sequences: int,
    ) -> bool:
        """Whether to admit a new request given current KV pressure."""
        if num_running >= max_num_sequences:
            return False
        level = self.pressure_level(kv_utilization)
        wait_iters = self._wait_iters.get(candidate.request_id, 0)
        has_aged_out = wait_iters >= self.starvation_penalty_iters

        if level == KVPressureLevel.LOW:
            return True
        elif level == KVPressureLevel.MEDIUM:
            return num_running < max_num_sequences // 2 or has_aged_out
        else:  # HIGH
            is_small = candidate.num_prompt_tokens <= 32
            return (is_small and num_running < max(1, max_num_sequences // 4)) or has_aged_out

    def adjusted_prefill_chunk(self, kv_utilization: float, base_chunk: int) -> int:
        """Reduce prefill chunk size under memory pressure to avoid KV spikes."""
        level = self.pressure_level(kv_utilization)
        if level == KVPressureLevel.MEDIUM:
            return max(1, base_chunk // 2)
        elif level == KVPressureLevel.HIGH:
            return max(1, base_chunk // 4)
        return base_chunk
