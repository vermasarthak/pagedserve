"""Scheduling policy interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import List
from pagedserve.engine.request import InferenceRequest


class SchedulingPolicy(ABC):
    """Abstract interface for request prioritization and scheduling policies."""

    @abstractmethod
    def sort_waiting(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Order waiting requests for admission consideration."""
        pass

    @abstractmethod
    def sort_running_decode(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Order active decode requests for execution priority."""
        pass


class FCFSPolicy(SchedulingPolicy):
    """First-Come, First-Served (FCFS) Scheduling Policy.
    
    Requests are admitted strictly in order of their arrival timestamp.
    This guarantees fairness, absence of starvation under moderate load,
    and highly deterministic testing behavior.
    """

    def sort_waiting(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        return sorted(requests, key=lambda r: r.arrival_time)

    def sort_running_decode(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        return sorted(requests, key=lambda r: r.arrival_time)
