"""Memory overhead estimation utilities for full-sequence gather vs direct blockwise paged attention."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryEstimate:
    """Theoretical memory consumption estimate for an attention execution path."""
    sequence_length: int
    num_heads: int
    head_dim: int
    block_size: int
    dtype_bytes: int
    gather_memory_bytes: int
    blockwise_memory_bytes: int

    @property
    def gather_memory_kb(self) -> float:
        return self.gather_memory_bytes / 1024.0

    @property
    def blockwise_memory_kb(self) -> float:
        return self.blockwise_memory_bytes / 1024.0

    @property
    def reduction_factor(self) -> float:
        if self.blockwise_memory_bytes == 0:
            return 1.0
        return self.gather_memory_bytes / self.blockwise_memory_bytes


def estimate_attention_memory(
    sequence_length: int,
    num_heads: int = 12,
    head_dim: int = 64,
    block_size: int = 16,
    dtype_bytes: int = 4,  # float32 = 4 bytes, float16 = 2 bytes
) -> MemoryEstimate:
    """Estimate peak temporary working memory for full-sequence gather vs direct blockwise attention.
    
    Args:
        sequence_length: Total active sequence length S.
        num_heads: Number of attention heads H.
        head_dim: Dimension per head D.
        block_size: Tokens per physical block B.
        dtype_bytes: Bytes per element (e.g. 4 for float32).
        
    Returns:
        MemoryEstimate dataclass containing byte allocations and reduction factor.
    """
    # Full gather allocates contiguous K and V tensors: 2 * (S * H * D * dtype_bytes)
    gather_bytes = 2 * sequence_length * num_heads * head_dim * dtype_bytes

    # Blockwise attention only allocates temporary per-block K and V slices: 2 * (B * H * D * dtype_bytes)
    effective_block_len = min(block_size, sequence_length)
    blockwise_bytes = 2 * effective_block_len * num_heads * head_dim * dtype_bytes

    return MemoryEstimate(
        sequence_length=sequence_length,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype_bytes=dtype_bytes,
        gather_memory_bytes=gather_bytes,
        blockwise_memory_bytes=blockwise_bytes,
    )


def compare_sequence_scaling(
    sequence_lengths: Optional[list[int]] = None,
    num_heads: int = 12,
    head_dim: int = 64,
    block_size: int = 16,
    dtype_bytes: int = 4,
) -> list[MemoryEstimate]:
    """Compare memory scaling across multiple sequence lengths."""
    if sequence_lengths is None:
        sequence_lengths = [128, 512, 2048, 8192]
    return [
        estimate_attention_memory(
            sequence_length=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype_bytes=dtype_bytes,
        )
        for seq_len in sequence_lengths
    ]
