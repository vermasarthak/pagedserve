"""Tensor Partitioning Simulation for PagedServe.

Simulates splitting multi-head attention queries, keys, and values across logical ranks
and provides rank-partition slicing utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorParallelConfig:
    """Configuration for simulated Tensor Parallelism partitioning."""
    world_size: int = 1
    rank: int = 0

    def __post_init__(self):
        if self.world_size not in {1, 2, 4, 8}:
            raise ValueError(f"Tensor Parallel world_size must be 1, 2, 4, or 8, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"Rank {self.rank} invalid for world_size {self.world_size}")


class TensorParallelGroup:
    """Manages head distribution and logical partitioning across ranks."""

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
        """Simulates AllReduce reduction step across logical partitions (in-process tensor identity)."""
        if self.config.world_size == 1:
            return tensor

        # In-process single-device simulation
        return tensor.clone()
