"""Correctness reference paged attention math implementation."""

import math
from typing import Optional
import torch

from pagedserve.memory.paged_view import PagedKVView


def paged_attention_reference(
    query: torch.Tensor,
    paged_view: PagedKVView,
    layer_idx: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Correctness reference implementation for paged attention.
    
    IMPORTANT TECHNICAL NOTICE:
    This is a correctness reference implementation. It is NOT an optimized PagedAttention
    kernel because it gathers paged K/V into contiguous tensors before attention computation.
    It does NOT claim zero-copy paged attention.
    
    Args:
        query: Single-token or multi-token Query tensor for one sequence.
               Shape: [batch=1, num_attention_heads, query_seq_len, head_dim]
        paged_view: Request-level PagedKVView pointing to physical block storage.
        layer_idx: Target hidden layer index.
        scale: Optional softmax scaling factor. Defaults to 1.0 / sqrt(head_dim).
        
    Returns:
        Attention output tensor of shape [batch=1, num_attention_heads, query_seq_len, head_dim].
    """
    # 1. Gather non-contiguous paged K and V blocks into contiguous tensors for this layer
    # Output canonical shapes: [seq_length, num_kv_heads, head_dim]
    k_can, v_can = paged_view.gather_layer(layer_idx)

    num_attn_heads = query.shape[1]
    query_seq_len = query.shape[2]
    head_dim = query.shape[3]
    num_kv_heads = k_can.shape[1]
    kv_seq_len = k_can.shape[0]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # 2. Reshape gathered K and V to match PyTorch multi-head attention shapes:
    # Canonical: [kv_seq_len, num_kv_heads, head_dim] -> Transpose -> [1, num_kv_heads, kv_seq_len, head_dim]
    k_seq = k_can.transpose(0, 1).unsqueeze(0)  # (1, num_kv_heads, kv_seq_len, head_dim)
    v_seq = v_can.transpose(0, 1).unsqueeze(0)  # (1, num_kv_heads, kv_seq_len, head_dim)

    # 3. Support Grouped-Query Attention (GQA) / Multi-Query Attention (MQA) head expansion if needed
    if num_attn_heads != num_kv_heads:
        num_queries_per_kv = num_attn_heads // num_kv_heads
        k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=1)
        v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=1)

    # 4. Attention math:
    # Query: (1, num_heads, query_seq_len, head_dim)
    # Key:   (1, num_heads, kv_seq_len, head_dim) -> transpose -> (1, num_heads, head_dim, kv_seq_len)
    scores = torch.matmul(query, k_seq.transpose(-2, -1)) * scale  # (1, num_heads, query_seq_len, kv_seq_len)

    # 5. Apply Causal Masking if query has multiple tokens (e.g. prompt prefill)
    if query_seq_len > 1:
        # Construct causal mask
        causal_mask = torch.triu(
            torch.full((query_seq_len, kv_seq_len), float("-inf"), device=query.device, dtype=query.dtype),
            diagonal=kv_seq_len - query_seq_len + 1,
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_seq)  # (1, num_heads, query_seq_len, head_dim)

    return output
