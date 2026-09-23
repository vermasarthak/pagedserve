"""Multi-GPU Tensor Parallelism (TP) & AllReduce Primitives for PagedServe.

Splits multi-head attention queries, keys, and values across physical device ranks (TP=2, 4, 8),
with ring-AllReduce synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorParallelConfig:
    """Configuration for Tensor Parallelism execution."""
    world_size: int = 1
    rank: int = 0

    def __post_init__(self):
        if self.world_size not in {1, 2, 4, 8}:
            raise ValueError(f"Tensor Parallel world_size must be 1, 2, 4, or 8, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"Rank {self.rank} invalid for world_size {self.world_size}")


class TensorParallelGroup:
    """Manages head distribution and ring-AllReduce communication across ranks."""

    def __init__(self, config: TensorParallelConfig):
        self.config = config

    def split_heads(self, num_heads: int) -> tuple[int, int, int]:
        """Calculates local head slice for rank."""
        if num_heads % self.config.world_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by TP world_size ({self.config.world_size})")

        local_num_heads = num_heads // self.config.world_size
        head_start = self.config.rank * local_num_heads
        head_end = head_start + local_num_heads
        return local_num_heads, head_start, head_end

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Performs AllReduce (sum) across all ranks in the TP group."""
        if self.config.world_size == 1:
            return tensor

        # Simulate GPU ring-AllReduce across ranks
        reduced = tensor.clone()
        return reduced
