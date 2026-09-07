"""PyTorch reference backend executing direct blockwise online-softmax paged attention."""

from typing import Optional
import torch

from pagedserve.kernels.backend import PagedAttentionBackend
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.attention import paged_attention_blockwise


class PyTorchPagedAttentionBackend(PagedAttentionBackend):
    """Pure PyTorch execution backend for blockwise paged attention."""

    def forward(
        self,
        query: torch.Tensor,
        paged_view: PagedKVView,
        layer_idx: int,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward pass delegating to paged_attention_blockwise."""
        return paged_attention_blockwise(
            query=query,
            paged_view=paged_view,
            layer_idx=layer_idx,
            scale=scale,
        )
