# Direct Blockwise Paged Attention Subsystem (PyTorch Reference)

This document details the design, mathematical formulation, memory complexity, and empirical benchmark results of the direct blockwise paged attention implementation (`paged_attention_blockwise`) in PagedServe.

## 1. Context & Architectural Evolution

Prior to this milestone, PagedServe managed KV blocks using two reference access patterns:
1. **Contiguous Full KV Cache**: Standard Hugging Face transformer execution (`DynamicCache`).
2. **Gather-Based Reference Paged Attention** (`paged_attention_reference`): A correctness reference that reads non-contiguous physical blocks from `TensorBlockStore` and reconstructs them into a single contiguous sequence tensor before computing standard PyTorch multi-head attention.

While `paged_attention_reference` proves physical block mapping correctness, it requires allocating a temporary contiguous KV sequence tensor of size $O(S \cdot H \cdot D)$ per layer during attention computation, where $S$ is the total sequence length, $H$ is the head count, and $D$ is the head dimension.

`paged_attention_blockwise` introduces an **incremental online softmax algorithm** that iterates directly over physical blocks in logical order without ever constructing a contiguous full-sequence KV tensor.

---

## 2. Online Softmax Formulation across Physical Blocks

For a decode Query tensor $q \in \mathbb{R}^{1 \times H \times 1 \times D}$ and sequence length $S$ mapped across physical blocks $B_0, B_1, \dots, B_{N-1}$:

For each physical block $i \in [0, N-1]$ storing Key slice $K_i$ and Value slice $V_i$ of length $b_i = \min(\text{block\_size}, S - i \cdot \text{block\_size})$:

1. **Block Logits Computation**:
   $$\text{scores}_i = \frac{q K_i^T}{\sqrt{D}} \in \mathbb{R}^{1 \times H \times 1 \times b_i}$$

2. **Causal Masking** (if prefill query length $> 1$):
   For prefill token position $p_q$ and block token position $p_{\text{kv}} = i \cdot \text{block\_size} + j$:
   $$\text{scores}_i(p_q, j) = -\infty \quad \text{if } p_{\text{kv}} > p_q$$

3. **Block Maximum**:
   $$m_i = \max_{j} (\text{scores}_i)$$

4. **Running Maximum Update**:
   $$m_{\text{new}} = \max(m_{\text{prev}}, m_i)$$

5. **Rescaling Factors**:
   $$\alpha = \exp(m_{\text{prev}} - m_{\text{new}})$$
   $$P_i = \exp(\text{scores}_i - m_{\text{new}})$$

6. **Accumulator Updates**:
   $$l_{\text{new}} = \alpha \cdot l_{\text{prev}} + \sum_{j} P_i$$
   $$\text{acc}_{\text{new}} = \alpha \cdot \text{acc}_{\text{prev}} + P_i V_i$$

7. **Final Normalization**:
   $$\text{Output} = \frac{\text{acc}_{\text{final}}}{l_{\text{final}}}$$

---

## 3. Support for GQA / MQA

For Grouped-Query Attention (GQA) and Multi-Query Attention (MQA), the number of Query heads $H_q$ exceeds the number of KV heads $H_{kv}$.
Rather than replicating $K_i$ and $V_i$ physical blocks in memory across layers, `paged_attention_blockwise` expands key/value head dimensions per block on-the-fly (`repeat_interleave`), retaining $O(\text{block\_size} \cdot H_q \cdot D)$ temporary memory overhead per block.

---

## 4. Memory Complexity Comparison

| Dimension | Full-Sequence Gather (`paged_attention_reference`) | Direct Blockwise (`paged_attention_blockwise`) |
| :--- | :--- | :--- |
| **Peak Temporary KV Buffer Memory** | $O(S \cdot H_{kv} \cdot D)$ | $O(B \cdot H_q \cdot D)$ |
| **Peak Attention Matrix Memory** | $O(H_q \cdot S)$ | $O(H_q \cdot B)$ |
| **Memory Reduction at $S=2048, B=16$** | $1.0\times$ | **$128.0\times$** |

---

## 5. Microbenchmark Results

Measured on Apple Silicon (M-series CPU & MPS):

```
==========================================================
 BLOCKWISE ATTENTION BENCHMARK — Device: CPU
==========================================================

| Seq Len | Mode | Mean Latency (ms) | Peak Temp Memory (KB) | Memory Reduction |
|---------|------|-------------------|-----------------------|------------------|
| 128     | Contiguous |             0.064 |                 768.0 | 1.0x             |
| 128     | Gather Ref |             0.198 |                 768.0 | 1.0x             |
| 128     | Blockwise  |             0.804 |                  96.0 |  8.0x            |
|---------|------|-------------------|-----------------------|------------------|
| 512     | Contiguous |             0.135 |                3072.0 | 1.0x             |
| 512     | Gather Ref |             0.669 |                3072.0 | 1.0x             |
| 512     | Blockwise  |             3.258 |                  96.0 | 32.0x            |
|---------|------|-------------------|-----------------------|------------------|
| 2048    | Contiguous |             0.677 |               12288.0 | 1.0x             |
| 2048    | Gather Ref |             5.859 |               12288.0 | 1.0x             |
| 2048    | Blockwise  |            14.974 |                  96.0 | 128.0x            |
|---------|------|-------------------|-----------------------|------------------|
```

> [!NOTE]
> In pure PyTorch, blockwise attention executes a Python loop over physical blocks, introducing PyTorch kernel dispatch overhead relative to contiguous GEMM. This is an expected trade-off in high-level reference implementations before introducing custom CUDA/Triton kernels. Peak working memory is dramatically bounded to $96\text{ KB}$ regardless of sequence length.
