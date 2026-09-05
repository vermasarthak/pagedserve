"""Scatter and Gather operations for non-contiguous physical KV block storage."""

from typing import List, Tuple
import torch

from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.tensor_block_store import TensorBlockStore, PhysicalBlockStoreError


class CacheScatterGather:
    """Handles scatter (exporting logical KV sequence into physical blocks) and gather (reconstructing logical KV sequence from physical blocks)."""

    def __init__(self, store: TensorBlockStore):
        self.store: TensorBlockStore = store
        self.geometry = store.geometry
        self.block_size = store.geometry.block_size

    def scatter_sequence_layer(
        self,
        layer_idx: int,
        block_table: BlockTable,
        key_seq: torch.Tensor,
        value_seq: torch.Tensor,
    ) -> None:
        """Scatter a logical sequence of Key and Value tensors for one layer into physical blocks.
        
        Args:
            layer_idx: Target hidden layer index.
            block_table: Request's BlockTable specifying logical-to-physical block mapping.
            key_seq: Canonical Key tensor of shape (seq_length, num_kv_heads, head_dim).
            value_seq: Canonical Value tensor of shape (seq_length, num_kv_heads, head_dim).
        """
        seq_len = key_seq.shape[0]
        if seq_len == 0:
            return

        if value_seq.shape[0] != seq_len:
            raise PhysicalBlockStoreError(
                f"Key and Value sequence length mismatch: {seq_len} vs {value_seq.shape[0]}"
            )

        num_required_blocks = (seq_len + self.block_size - 1) // self.block_size
        if block_table.num_blocks < num_required_blocks:
            raise PhysicalBlockStoreError(
                f"BlockTable has {block_table.num_blocks} blocks, but sequence length {seq_len} requires {num_required_blocks}"
            )

        for log_idx in range(num_required_blocks):
            phys_id = block_table.physical_block_for(log_idx)
            start_tok = log_idx * self.block_size
            end_tok = min(seq_len, (log_idx + 1) * self.block_size)
            chunk_len = end_tok - start_tok

            k_chunk = key_seq[start_tok:end_tok]
            v_chunk = value_seq[start_tok:end_tok]

            self.store.write_block(
                layer_idx=layer_idx,
                physical_block_id=phys_id,
                key_block=k_chunk,
                value_block=v_chunk,
                num_tokens=chunk_len,
            )

    def scatter_sequence_all_layers(
        self,
        block_table: BlockTable,
        layer_kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Scatter canonical Key and Value tensors for all layers into physical blocks."""
        if len(layer_kv_pairs) != self.geometry.num_layers:
            raise PhysicalBlockStoreError(
                f"Expected {self.geometry.num_layers} layer pairs, got {len(layer_kv_pairs)}"
            )

        for layer_idx, (k_seq, v_seq) in enumerate(layer_kv_pairs):
            self.scatter_sequence_layer(layer_idx, block_table, k_seq, v_seq)

    def gather_sequence_layer(
        self,
        layer_idx: int,
        block_table: BlockTable,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather non-contiguous physical blocks into a single contiguous logical sequence for one layer.
        
        Args:
            layer_idx: Target hidden layer index.
            block_table: Request's BlockTable specifying physical block mapping.
            seq_length: Exact token count of the active sequence.
            
        Returns:
            Tuple of (gathered_key_seq, gathered_value_seq) each of shape (seq_length, num_kv_heads, head_dim).
        """
        if seq_length == 0:
            empty_shape = (0, self.geometry.num_kv_heads, self.geometry.head_dim)
            return (
                torch.empty(empty_shape, dtype=self.store.dtype, device=self.store.device),
                torch.empty(empty_shape, dtype=self.store.dtype, device=self.store.device),
            )

        num_required_blocks = (seq_length + self.block_size - 1) // self.block_size
        if block_table.num_blocks < num_required_blocks:
            raise PhysicalBlockStoreError(
                f"BlockTable has {block_table.num_blocks} blocks, but seq_length {seq_length} requires {num_required_blocks}"
            )

        k_out = torch.empty(
            (seq_length, self.geometry.num_kv_heads, self.geometry.head_dim),
            dtype=self.store.dtype,
            device=self.store.device,
        )
        v_out = torch.empty(
            (seq_length, self.geometry.num_kv_heads, self.geometry.head_dim),
            dtype=self.store.dtype,
            device=self.store.device,
        )

        for log_idx in range(num_required_blocks):
            phys_id = block_table.physical_block_for(log_idx)
            start_tok = log_idx * self.block_size
            end_tok = min(seq_length, (log_idx + 1) * self.block_size)
            chunk_len = end_tok - start_tok

            k_blk, v_blk = self.store.read_block(
                layer_idx=layer_idx, physical_block_id=phys_id, num_tokens=chunk_len
            )
            k_out[start_tok:end_tok] = k_blk
            v_out[start_tok:end_tok] = v_blk

        return k_out, v_out

    def gather_sequence_all_layers(
        self,
        block_table: BlockTable,
        seq_length: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Gather non-contiguous physical blocks into logical sequences across all layers."""
        layers = []
        for layer_idx in range(self.geometry.num_layers):
            layers.append(self.gather_sequence_layer(layer_idx, block_table, seq_length))
        return layers
