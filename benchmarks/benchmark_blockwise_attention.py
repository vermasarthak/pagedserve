"""Microbenchmark comparing Contiguous Attention, Gather-based Paged Attention, and Direct Blockwise Paged Attention."""

import math
import time

import torch

from pagedserve.memory.attention import (
    paged_attention_blockwise,
    paged_attention_reference,
)
from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.geometry import KVCacheGeometry
from pagedserve.memory.memory_estimator import estimate_attention_memory
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.tensor_block_store import TensorBlockStore


def run_benchmark(
    device_str: str = "cpu",
    sequence_lengths: list[int] = [128, 512, 2048],
    num_heads: int = 12,
    head_dim: int = 64,
    block_size: int = 16,
    num_warmup: int = 5,
    num_iters: int = 50,
):
    device = torch.device(device_str)
    print("\n==========================================================")
    print(f" BLOCKWISE ATTENTION BENCHMARK — Device: {device_str.upper()}")
    print("==========================================================\n")
    print("| Seq Len | Mode | Mean Latency (ms) | Peak Temp Memory (KB) | Memory Reduction |")
    print("|---------|------|-------------------|-----------------------|------------------|")

    for seq_len in sequence_lengths:
        geom = KVCacheGeometry(
            num_layers=1,
            num_kv_heads=num_heads,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=torch.float32,
            device=device,
        )
        num_blocks = math.ceil(seq_len / block_size)
        pool = BlockPool(num_blocks=num_blocks + 4, block_size=block_size)
        store = TensorBlockStore(geometry=geom, num_blocks=num_blocks + 4)

        phys_blocks = [pool.allocate().block_id for _ in range(num_blocks)]
        block_table = BlockTable(request_id="bench_req")
        for p_id in phys_blocks:
            block_table.append_block(p_id)

        for b_idx, p_id in enumerate(phys_blocks):
            start = b_idx * block_size
            end = min((b_idx + 1) * block_size, seq_len)
            v_len = end - start
            k_data = torch.randn(v_len, num_heads, head_dim, device=device)
            v_data = torch.randn(v_len, num_heads, head_dim, device=device)
            store.write_block(0, p_id, k_data, v_data, num_tokens=v_len)

        paged_view = PagedKVView(block_table, seq_len, store)
        query = torch.randn(1, num_heads, 1, head_dim, device=device)

        mem_est = estimate_attention_memory(seq_len, num_heads, head_dim, block_size)

        # 1. Contiguous Attention Benchmark
        k_contiguous = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        v_contiguous = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        scale = 1.0 / math.sqrt(head_dim)

        for _ in range(num_warmup):
            scores = torch.matmul(query, k_contiguous.transpose(-2, -1)) * scale
            _ = torch.matmul(torch.softmax(scores, dim=-1), v_contiguous)

        t0 = time.perf_counter()
        for _ in range(num_iters):
            scores = torch.matmul(query, k_contiguous.transpose(-2, -1)) * scale
            _ = torch.matmul(torch.softmax(scores, dim=-1), v_contiguous)
        t_cont = (time.perf_counter() - t0) / num_iters * 1000.0

        # 2. Gather-based Reference Attention Benchmark
        for _ in range(num_warmup):
            _ = paged_attention_reference(query, paged_view, layer_idx=0)

        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = paged_attention_reference(query, paged_view, layer_idx=0)
        t_ref = (time.perf_counter() - t0) / num_iters * 1000.0

        # 3. Direct Blockwise Attention Benchmark
        for _ in range(num_warmup):
            _ = paged_attention_blockwise(query, paged_view, layer_idx=0)

        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = paged_attention_blockwise(query, paged_view, layer_idx=0)
        t_bw = (time.perf_counter() - t0) / num_iters * 1000.0

        print(f"| {seq_len:<7} | Contiguous | {t_cont:17.3f} | {mem_est.gather_memory_kb:21.1f} | 1.0x             |")
        print(f"| {seq_len:<7} | Gather Ref | {t_ref:17.3f} | {mem_est.gather_memory_kb:21.1f} | 1.0x             |")
        print(f"| {seq_len:<7} | Blockwise  | {t_bw:17.3f} | {mem_est.blockwise_memory_kb:21.1f} | {mem_est.reduction_factor:4.1f}x            |")
        print("|---------|------|-------------------|-----------------------|------------------|")


if __name__ == "__main__":
    run_benchmark(device_str="cpu")
    if torch.backends.mps.is_available():
        run_benchmark(device_str="mps")
