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


def paged_attention_blockwise(
    query: torch.Tensor,
    paged_view: PagedKVView,
    layer_idx: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Direct blockwise paged attention using incremental online softmax.
    
    Reads physical KV blocks directly from TensorBlockStore without gathering
    the entire sequence into a contiguous K/V tensor.
    
    Memory Complexity:
        O(block_size * num_heads * head_dim) temporary working memory per block,
        compared to O(seq_length * num_heads * head_dim) for full-sequence gather.
        
    Args:
        query: Query tensor for one sequence.
               Shape: [batch=1, num_attention_heads, query_seq_len, head_dim]
        paged_view: Request-level PagedKVView pointing to physical block storage.
        layer_idx: Target hidden layer index.
        scale: Optional softmax scaling factor. Defaults to 1.0 / sqrt(head_dim).
        
    Returns:
        Attention output tensor of shape [batch=1, num_attention_heads, query_seq_len, head_dim].
    """
    block_table = paged_view.block_table
    seq_length = paged_view.seq_length
    store = paged_view.store
    block_size = store.geometry.block_size
    num_kv_heads = store.geometry.num_kv_heads

    num_attn_heads = query.shape[1]
    query_seq_len = query.shape[2]
    head_dim = query.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    num_queries_per_kv = num_attn_heads // num_kv_heads

    # Online softmax accumulators:
    # max_score: [1, num_attn_heads, query_seq_len, 1]
    # sum_exp:   [1, num_attn_heads, query_seq_len, 1]
    # acc:       [1, num_attn_heads, query_seq_len, head_dim]
    max_score: Optional[torch.Tensor] = None
    sum_exp: Optional[torch.Tensor] = None
    acc: Optional[torch.Tensor] = None

    for block_idx, phys_block_id in enumerate(block_table.blocks):
        start_token_idx = block_idx * block_size
        end_token_idx = min((block_idx + 1) * block_size, seq_length)
        valid_tokens = end_token_idx - start_token_idx

        if valid_tokens <= 0:
            break

        # Direct block lookup without gather
        # Shapes: [valid_tokens, num_kv_heads, head_dim]
        k_block_raw = store.get_key_block(phys_block_id, layer_idx)[:valid_tokens]
        v_block_raw = store.get_value_block(phys_block_id, layer_idx)[:valid_tokens]

        # Reshape to [1, num_kv_heads, valid_tokens, head_dim]
        k_block = k_block_raw.transpose(0, 1).unsqueeze(0)
        v_block = v_block_raw.transpose(0, 1).unsqueeze(0)

        # Handle GQA / MQA head expansion if needed
        if num_queries_per_kv > 1:
            k_block = k_block.repeat_interleave(num_queries_per_kv, dim=1)
            v_block = v_block.repeat_interleave(num_queries_per_kv, dim=1)

        # Query: [1, num_attn_heads, query_seq_len, head_dim]
        # k_block: [1, num_attn_heads, valid_tokens, head_dim]
        # scores_block: [1, num_attn_heads, query_seq_len, valid_tokens]
        scores_block = torch.matmul(query, k_block.transpose(-2, -1)) * scale

        # Apply causal masking if query has multiple tokens (e.g. prompt prefill)
        if query_seq_len > 1:
            # Query token position i is at absolute index: (seq_length - query_seq_len + i)
            # Block token position j is at absolute index: (start_token_idx + j)
            q_pos = torch.arange(
                seq_length - query_seq_len, seq_length, device=query.device
            ).unsqueeze(1) # [query_seq_len, 1]
            kv_pos = torch.arange(
                start_token_idx, start_token_idx + valid_tokens, device=query.device
            ).unsqueeze(0) # [1, valid_tokens]

            mask = kv_pos > q_pos # [query_seq_len, valid_tokens] boolean mask
            scores_block = scores_block.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Online softmax update
        block_max = torch.max(scores_block, dim=-1, keepdim=True).values  # [1, num_heads, q_len, 1]

        if max_score is None:
            max_score = block_max
            exp_scores = torch.exp(scores_block - max_score)
            exp_scores = torch.nan_to_num(exp_scores, nan=0.0) # Handle masked out -inf
            sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
            acc = torch.matmul(exp_scores, v_block)
        else:
            new_max = torch.maximum(max_score, block_max)
            alpha = torch.exp(max_score - new_max)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            exp_scores = torch.exp(scores_block - new_max)
            exp_scores = torch.nan_to_num(exp_scores, nan=0.0)

            sum_exp = alpha * sum_exp + torch.sum(exp_scores, dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(exp_scores, v_block)
            max_score = new_max

    if acc is None or sum_exp is None:
        raise ValueError(f"No valid blocks processed for sequence length {seq_length}")

    # Output: [1, num_attn_heads, query_seq_len, head_dim]
    output = acc / torch.clamp(sum_exp, min=1e-12)
    return output

