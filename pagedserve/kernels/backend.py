"""Abstract base class interface for PagedAttention execution backends."""

from abc import ABC, abstractmethod

import torch

from pagedserve.memory.paged_view import PagedKVView


class PagedAttentionBackend(ABC):
    """Abstract base class for PagedAttention execution backends."""

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        paged_view: PagedKVView,
        layer_idx: int,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Execute paged attention for a single request query.

        Args:
            query: Query tensor of shape [batch=1, num_attention_heads, query_seq_len, head_dim]
            paged_view: Request PagedKVView pointing to physical block store
            layer_idx: Hidden layer index
            scale: Softmax scaling factor

        Returns:
            Attention output tensor of shape [batch=1, num_attention_heads, query_seq_len, head_dim]
        """
