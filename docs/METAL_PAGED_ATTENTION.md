# Fused Apple Metal Paged Attention Subsystem

This document details the design, Metal Shading Language (MSL) shader architecture, threadgroup dispatch, online softmax implementation, fallback rules, and empirical benchmark results for PagedServe's experimental Apple Metal execution backend (`MetalPagedAttentionBackend`).

---

## 1. System Overview & Architecture

The Metal paged attention backend (`pagedserve/kernels/metal_backend.py` & `pagedserve/kernels/metal_shader.metal`) provides custom GPU compute kernel execution for single-token decode attention on Apple Silicon (M1/M2/M3/M4 GPUs).

```
Query [1, H_q, 1, D]
        |
        v
Metal Shader (paged_attention_decode_kernel)
        |---> Reads block_table [p0, p1, p2, ...]
        |---> Direct 5D physical tensor lookup: K_store & V_store
        |---> Online softmax accumulation (m, l, acc)
        v
Output [1, H_q, 1, D]
```

### Key Properties
- **Zero Full-Sequence Gather**: Reads non-contiguous physical block IDs (`phys_block_id = block_table[b]`) directly inside the Metal compute shader.
- **Online Softmax**: Computes numerically stable online softmax ($m, l, \text{acc}$) across physical blocks on GPU.
- **Head Mapping (MHA, GQA, MQA)**: Maps query heads $h_q$ to KV heads $h_{kv} = h_q // \text{queries\_per\_kv}$ without duplicating KV memory.
- **Partial Block Support**: Handles partial final blocks for arbitrary sequence lengths (`valid_tokens = min(block_size, seq_len - b * block_size)`).
- **Runtime Compilation & Pipeline Caching**: MSL source is compiled ONCE at runtime using macOS `Metal.framework` via Objective-C runtime bindings. Compiled `MTLComputePipelineState` objects are cached.

---

## 2. Metal Shading Language (MSL) Kernel Design

```metal
#include <metal_stdlib>
using namespace metal;

kernel void paged_attention_decode_kernel(
    device const float* query              [[buffer(0)]],
    device const float* k_store            [[buffer(1)]],
    device const float* v_store            [[buffer(2)]],
    device const int*   block_table        [[buffer(3)]],
    device float*       output             [[buffer(4)]],
    constant int&       num_attn_heads     [[buffer(5)]],
    constant int&       num_kv_heads       [[buffer(6)]],
    constant int&       head_dim           [[buffer(7)]],
    constant int&       block_size         [[buffer(8)]],
    constant int&       total_blocks       [[buffer(9)]],
    constant int&       num_blocks_in_table[[buffer(10)]],
    constant int&       seq_length         [[buffer(11)]],
    constant int&       layer_idx          [[buffer(12)]],
    constant float&     scale              [[buffer(13)]],
    uint thread_id                         [[thread_position_in_grid]]
)
```

### Threadgroup Mapping
- **Grid Size**: `MTLSize(width=num_attn_heads, height=1, depth=1)` — 1 GPU thread per Query attention head.
- **Threadgroup Size**: `MTLSize(width=min(num_attn_heads, 32), height=1, depth=1)`.
- **Memory Access Pattern**: Vectorized strided indexing across 5D physical storage `[num_layers, total_blocks, block_size, num_kv_heads, head_dim]`.

---

## 3. Microbenchmark & Performance Analysis

Measured on Apple Silicon (M1 Mac, macOS 15.5):

```
==========================================================================================
 FUSED APPLE METAL PAGED ATTENTION MICROBENCHMARK
 Warmups: 10, Trials: 100, Metal Available: True
==========================================================================================

Metal MSL Shader Pipeline Initial Compilation Time: 4.734 ms

| Seq Len | Block Size | Mode | Median Latency (ms) | p95 Latency (ms) | Peak Temp Mem (KB) | Latency vs PT Blockwise |
|---------|------------|------|---------------------|------------------|--------------------|-------------------------|
| 128     | 16         | Contiguous |               0.054 |            0.069 |              768.0 | 1.00x                   |
| 128     | 16         | Gather Ref |               0.150 |            0.207 |              768.0 | 1.00x                   |
| 128     | 16         | PT Block  |               0.605 |            0.689 |               96.0 | Baseline                |
| 128     | 16         | Metal Direct|               3.570 |            6.984 |               96.0 | 5.90x higher latency    |
|---------|------------|------|---------------------|------------------|--------------------|-------------------------|
| 512     | 16         | Contiguous |               0.129 |            0.158 |             3072.0 | 1.00x                   |
| 512     | 16         | Gather Ref |               0.564 |            0.869 |             3072.0 | 1.00x                   |
| 512     | 16         | PT Block  |               2.111 |            2.746 |               96.0 | Baseline                |
| 512     | 16         | Metal Direct|              13.146 |          101.545 |               96.0 | 6.23x higher latency    |
|---------|------------|------|---------------------|------------------|--------------------|-------------------------|
| 2048    | 16         | Contiguous |               0.744 |            1.739 |            12288.0 | 1.00x                   |
| 2048    | 16         | Gather Ref |               3.768 |            5.975 |            12288.0 | 1.00x                   |
| 2048    | 16         | PT Block  |              19.284 |          103.977 |               96.0 | Baseline                |
| 2048    | 16         | Metal Direct|              46.935 |           56.345 |               96.0 | 2.43x higher latency    |
|---------|------------|------|---------------------|------------------|--------------------|-------------------------|
```

### Technical Root Cause of Latency Characteristics
1. **Per-Invocation Buffer Allocation Overhead**: In high-level Python/PyTorch host code, transferring CPU memory buffers to Metal via `newBufferWithBytes` incurs buffer allocation and memory copy overhead on each token step.
2. **Dispatch Granularity**: For small single-token decode shapes (e.g. 12 heads), CPU C++ SIMD vectorization in PyTorch avoids host-to-device command buffer encoding overhead.
3. **Memory Footprint Parity**: Both PyTorch blockwise and Metal direct paged attention achieve $96\text{ KB}$ peak temporary working memory ($128\times$ lower than contiguous gather attention).

---

## 4. Automatic Fallback Rules

`MetalPagedAttentionBackend` automatically falls back to `PyTorchPagedAttentionBackend` if:
- `query_seq_len > 1` (prefill mode).
- Environment lacks Apple Silicon Metal runtime support.
- `head_dim > 128` (exceeds thread register array capacity).
