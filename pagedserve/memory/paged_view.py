"""Request-level paged cache view object providing gathered access to non-contiguous physical blocks."""

import torch
from transformers.cache_utils import DynamicCache

from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.scatter_gather import CacheScatterGather
from pagedserve.memory.tensor_block_store import TensorBlockStore
from pagedserve.model.cache_adapter import CacheAdapter


class PagedKVView:
    """Request-level view object exposing access to non-contiguous physical KV blocks.

    Architectural & Implementation Limitation:
    This milestone gathers non-contiguous physical K/V blocks into temporary contiguous
    tensors on-demand during view operations. It does NOT yet perform zero-copy paged attention.
    """

    def __init__(
        self,
        block_table: BlockTable,
        seq_length: int,
        store: TensorBlockStore,
    ):
        self.block_table: BlockTable = block_table
        self.seq_length: int = seq_length
        self.store: TensorBlockStore = store
        self._scatter_gather: CacheScatterGather = CacheScatterGather(store)

    @property
    def block_size(self) -> int:
        """Token capacity per physical block."""
        return self.store.geometry.block_size

    def gather_keys(self, layer_idx: int) -> torch.Tensor:
        """Gather canonical Key sequence for a given layer.

        Returns:
            Key tensor of shape (seq_length, num_kv_heads, head_dim).
        """
        k, _ = self._scatter_gather.gather_sequence_layer(
            layer_idx=layer_idx, block_table=self.block_table, seq_length=self.seq_length
        )
        return k

    def gather_values(self, layer_idx: int) -> torch.Tensor:
        """Gather canonical Value sequence for a given layer.

        Returns:
            Value tensor of shape (seq_length, num_kv_heads, head_dim).
        """
        _, v = self._scatter_gather.gather_sequence_layer(
            layer_idx=layer_idx, block_table=self.block_table, seq_length=self.seq_length
        )
        return v

    def gather_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather both Key and Value canonical sequences for a given layer."""
        return self._scatter_gather.gather_sequence_layer(
            layer_idx=layer_idx, block_table=self.block_table, seq_length=self.seq_length
        )

    def to_hf_dynamic_cache(self) -> DynamicCache:
        """Gather all layers from non-contiguous physical blocks into a Hugging Face DynamicCache."""
        layer_kv_pairs = self._scatter_gather.gather_sequence_all_layers(
            block_table=self.block_table, seq_length=self.seq_length
        )
        return CacheAdapter.canonical_to_hf_cache(layer_kv_pairs, self.store.geometry)
