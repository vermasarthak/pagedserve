#!/usr/bin/env python3
"""Microbenchmarks for physical KV block storage operations: scatter, gather, token append, and reference attention."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_utils import synchronize_device
from pagedserve.memory.attention import paged_attention_reference
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.geometry import KVCacheGeometry
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.scatter_gather import CacheScatterGather
from pagedserve.memory.tensor_block_store import TensorBlockStore


def benchmark_physical_store_ops(device_str: str = "cpu", trials: int = 100):
    device = torch.device(device_str)
    dtype = torch.float32

    geometry = KVCacheGeometry(
        num_layers=12,
        num_kv_heads=12,
        num_attention_heads=12,
        head_dim=64,
        block_size=16,
        dtype=dtype,
        device=device,
    )
    num_blocks = 128
    store = TensorBlockStore(geometry, num_blocks=num_blocks)
    scatter_gather = CacheScatterGather(store)

    # 1. Prepare synthetic sequence of length 128 (8 blocks)
    seq_len = 128
    key_seq = torch.randn((seq_len, geometry.num_kv_heads, geometry.head_dim), dtype=dtype, device=device)
    val_seq = torch.randn((seq_len, geometry.num_kv_heads, geometry.head_dim), dtype=dtype, device=device)

    # Non-contiguous scrambled blocks
    scrambled_ids = [17, 3, 42, 9, 88, 12, 65, 2]
    table = BlockTable("bench_req")
    for b in scrambled_ids:
        table.append_block(b)

    # Warmup
    for _ in range(10):
        scatter_gather.scatter_sequence_layer(0, table, key_seq, val_seq)
        _ = scatter_gather.gather_sequence_layer(0, table, seq_len)

    synchronize_device(device)
    
    # Measure Scatter
    t0 = time.monotonic()
    for _ in range(trials):
        scatter_gather.scatter_sequence_layer(0, table, key_seq, val_seq)
    synchronize_device(device)
    scatter_time_ms = ((time.monotonic() - t0) / trials) * 1000.0

    # Measure Gather
    t0 = time.monotonic()
    for _ in range(trials):
        _ = scatter_gather.gather_sequence_layer(0, table, seq_len)
    synchronize_device(device)
    gather_time_ms = ((time.monotonic() - t0) / trials) * 1000.0

    # Measure Single Token Append
    single_k = torch.randn((geometry.num_kv_heads, geometry.head_dim), dtype=dtype, device=device)
    single_v = torch.randn((geometry.num_kv_heads, geometry.head_dim), dtype=dtype, device=device)
    t0 = time.monotonic()
    for _ in range(trials):
        store.write_token(0, 17, 5, single_k, single_v)
    synchronize_device(device)
    write_token_time_ms = ((time.monotonic() - t0) / trials) * 1000.0

    # Measure Reference Paged Attention
    paged_view = PagedKVView(table, seq_len, store)
    query = torch.randn((1, geometry.num_attention_heads, 1, geometry.head_dim), dtype=dtype, device=device)
    
    # Warmup attention
    _ = paged_attention_reference(query, paged_view, layer_idx=0)
    synchronize_device(device)

    t0 = time.monotonic()
    for _ in range(trials):
        _ = paged_attention_reference(query, paged_view, layer_idx=0)
    synchronize_device(device)
    attn_time_ms = ((time.monotonic() - t0) / trials) * 1000.0

    print(f"\n--- PHYSICAL KV BLOCK STORE MICROBENCHMARKS ({device_str.upper()}) ---")
    print(f"  Scatter Layer (seq_len={seq_len}):        {scatter_time_ms:.4f} ms")
    print(f"  Gather Layer (seq_len={seq_len}):         {gather_time_ms:.4f} ms")
    print(f"  Write Single Decode Token:            {write_token_time_ms:.4f} ms")
    print(f"  Reference Paged Attention (1 decode): {attn_time_ms:.4f} ms")

    return {
        "device": device_str,
        "scatter_time_ms": scatter_time_ms,
        "gather_time_ms": gather_time_ms,
        "write_token_time_ms": write_token_time_ms,
        "attn_time_ms": attn_time_ms,
    }


if __name__ == "__main__":
    benchmark_physical_store_ops("cpu")
    if torch.backends.mps.is_available():
        benchmark_physical_store_ops("mps")
