"""Microbenchmark comparing Contiguous Attention, PyTorch Gather Ref, PyTorch Blockwise, and Fused Metal Direct Paged Attention."""

import math
import statistics
import time
import torch

from pagedserve.kernels import PyTorchPagedAttentionBackend, MetalPagedAttentionBackend, is_metal_available
from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.geometry import KVCacheGeometry
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.tensor_block_store import TensorBlockStore
from pagedserve.memory.attention import paged_attention_reference
from pagedserve.memory.memory_estimator import estimate_attention_memory


def run_benchmark(
    sequence_lengths: list[int] = [128, 512, 2048],
    block_sizes: list[int] = [16, 32],
    num_heads: int = 12,
    head_dim: int = 64,
    warmups: int = 10,
    trials: int = 100,
):
    print("\n==========================================================================================")
    print(" FUSED APPLE METAL PAGED ATTENTION MICROBENCHMARK")
    print(f" Warmups: {warmups}, Trials: {trials}, Metal Available: {is_metal_available()}")
    print("==========================================================================================\n")

    if not is_metal_available():
        print("Metal execution is not supported in this environment. Exiting benchmark.")
        return

    # Benchmark kernel compilation time separately
    t_comp_start = time.perf_counter()
    metal_backend = MetalPagedAttentionBackend()
    _ = metal_backend._get_pipeline()
    t_comp_ms = (time.perf_counter() - t_comp_start) * 1000.0
    print(f"Metal MSL Shader Pipeline Initial Compilation Time: {t_comp_ms:.3f} ms\n")

    pytorch_backend = PyTorchPagedAttentionBackend()

    print("| Seq Len | Block Size | Mode | Median Latency (ms) | p95 Latency (ms) | Peak Temp Mem (KB) | Latency vs PT Blockwise |")
    print("|---------|------------|------|---------------------|------------------|--------------------|-------------------------|")

    for seq_len in sequence_lengths:
        for block_size in block_sizes:
            geom = KVCacheGeometry(
                num_layers=1,
                num_kv_heads=num_heads,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
                dtype=torch.float32,
                device=torch.device("cpu"),
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
                val_len = end - start
                k_data = torch.randn(val_len, num_heads, head_dim)
                v_data = torch.randn(val_len, num_heads, head_dim)
                store.write_block(0, p_id, k_data, v_data, num_tokens=val_len)

            paged_view = PagedKVView(block_table, seq_len, store)
            query = torch.randn(1, num_heads, 1, head_dim)
            mem_est = estimate_attention_memory(seq_len, num_heads, head_dim, block_size)

            # 1. Contiguous PyTorch Attention
            k_cont = torch.randn(1, num_heads, seq_len, head_dim)
            v_cont = torch.randn(1, num_heads, seq_len, head_dim)
            scale = 1.0 / math.sqrt(head_dim)

            for _ in range(warmups):
                s = torch.matmul(query, k_cont.transpose(-2, -1)) * scale
                _ = torch.matmul(torch.softmax(s, dim=-1), v_cont)

            cont_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                s = torch.matmul(query, k_cont.transpose(-2, -1)) * scale
                _ = torch.matmul(torch.softmax(s, dim=-1), v_cont)
                cont_times.append((time.perf_counter() - t0) * 1000.0)

            # 2. PyTorch Gather Reference
            for _ in range(warmups):
                _ = paged_attention_reference(query, paged_view, layer_idx=0)

            ref_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                _ = paged_attention_reference(query, paged_view, layer_idx=0)
                ref_times.append((time.perf_counter() - t0) * 1000.0)

            # 3. PyTorch Blockwise Backend
            for _ in range(warmups):
                _ = pytorch_backend.forward(query, paged_view, layer_idx=0)

            pt_bw_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                _ = pytorch_backend.forward(query, paged_view, layer_idx=0)
                pt_bw_times.append((time.perf_counter() - t0) * 1000.0)

            # 4. Metal Direct Paged Attention Backend
            for _ in range(warmups):
                _ = metal_backend.forward(query, paged_view, layer_idx=0)

            metal_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                _ = metal_backend.forward(query, paged_view, layer_idx=0)
                metal_times.append((time.perf_counter() - t0) * 1000.0)

            def stats(t_list):
                sorted_t = sorted(t_list)
                med = statistics.median(sorted_t)
                p95 = sorted_t[int(0.95 * len(sorted_t))]
                return med, p95

            med_cont, p95_cont = stats(cont_times)
            med_ref, p95_ref = stats(ref_times)
            med_pt_bw, p95_pt_bw = stats(pt_bw_times)
            med_metal, p95_metal = stats(metal_times)

            ratio_metal_vs_pt = med_metal / med_pt_bw
            if ratio_metal_vs_pt < 1.0:
                rel_str = f"{(1.0 - ratio_metal_vs_pt)*100.0:.1f}% latency reduction"
            else:
                rel_str = f"{ratio_metal_vs_pt:.2f}x higher latency"

            print(f"| {seq_len:<7} | {block_size:<10} | Contiguous | {med_cont:19.3f} | {p95_cont:16.3f} | {mem_est.gather_memory_kb:18.1f} | 1.00x                   |")
            print(f"| {seq_len:<7} | {block_size:<10} | Gather Ref | {med_ref:19.3f} | {p95_ref:16.3f} | {mem_est.gather_memory_kb:18.1f} | 1.00x                   |")
            print(f"| {seq_len:<7} | {block_size:<10} | PT Block  | {med_pt_bw:19.3f} | {p95_pt_bw:16.3f} | {mem_est.blockwise_memory_kb:18.1f} | Baseline                |")
            print(f"| {seq_len:<7} | {block_size:<10} | Metal Direct| {med_metal:19.3f} | {p95_metal:16.3f} | {mem_est.blockwise_memory_kb:18.1f} | {rel_str:<23} |")
            print("|---------|------------|------|---------------------|------------------|--------------------|-------------------------|")


if __name__ == "__main__":
    run_benchmark(sequence_lengths=[128, 512, 2048], block_sizes=[16, 32], warmups=10, trials=100)
