# Physical Block-Backed KV Cache Storage

This document details the architecture, tensor geometry, allocation policies, scatter/gather mechanisms, physical prefix sharing, and reference paged attention implemented in PagedServe.

---

## 1. System Overview & Architecture Evolution

PagedServe has evolved from metadata-only KV cache management to a **real physical block-backed KV tensor store**:

```
BEFORE (Phases 1-20):
Request -> BlockTable -> KVBlock Metadata -> HF DynamicCache -> K/V Tensors

AFTER THIS MILESTONE (Phases 21-28):
Request -> BlockTable -> TensorBlockStore (Physical Tensor Pool) -> Scatter / Gather -> Reference Paged Attention
```

> [!NOTE]
> **Technical Honesty & Architectural Scope**  
> Physical Key and Value tensors are now allocated and stored in contiguous memory buffers partitioned into fixed-size physical blocks (`TensorBlockStore`). To perform attention computation in PyTorch without custom C++/CUDA/Triton kernels, non-contiguous physical blocks are gathered on-demand into temporary contiguous sequence tensors (`PagedKVView` / `paged_attention_reference`). This implementation establishes correctness and provides a reference baseline before developing optimized physical kernels.

---

## 2. KV Cache Geometry (`KVCacheGeometry`)

The physical tensor dimensions are dynamically derived from the loaded model configuration (`LoadedModel`).

### Attributes & Field Resolution
- `num_layers`: Extracted from `config.num_hidden_layers` or `config.n_layer`.
- `num_attention_heads`: Extracted from `config.num_attention_heads` or `config.n_head`.
- `num_kv_heads`: Extracted from `config.num_key_value_heads` (defaults to `num_attention_heads` for Multi-Head Attention, supports Grouped-Query and Multi-Query Attention).
- `head_dim`: Computed as `hidden_size // num_attention_heads` or `config.head_dim`.
- `block_size`: Fixed token capacity per block (default: 16).
- `dtype` & `device`: Inherited from model precision and target device (CPU, MPS, CUDA).

---

## 3. Physical KV Tensor Block Store (`TensorBlockStore`)

### Physical Tensor Layout & Dimensions
Both Key ($K$) and Value ($V$) storage are pre-allocated ONCE as unified contiguous 5D PyTorch tensors:

$$\text{Shape: } [\text{num\_layers},\; \text{num\_blocks},\; \text{block\_size},\; \text{num\_kv\_heads},\; \text{head\_dim}]$$

### Layout Rationale & Contiguity
1. **Layer Dimension (0)**: Indexing by layer index yields contiguous multi-block memory for that layer.
2. **Block Dimension (1)**: Physical block IDs `[0..num_blocks-1]` index directly into dimension 1.
3. **Block Offset (2), KV Heads (3), Head Dim (4)**: Contiguous placement per block enables vectorized block writes/reads and efficient batch slicing without per-element copies.

### Memory Footprint & Telemetry

$$\text{Bytes per Token (K+V)} = 2 \times \text{num\_layers} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{element\_size}$$

$$\text{Bytes per Block} = \text{Bytes per Token} \times \text{block\_size}$$

$$\text{Total Storage Bytes} = \text{Bytes per Block} \times \text{num\_blocks}$$

---

## 4. Cache Scatter, Gather, and View Abstractions

### Cache Scatter (`CacheScatterGather.scatter_sequence_layer`)
Scatters canonical sequence tensors of shape `[seq_len, num_kv_heads, head_dim]` into non-contiguous physical block locations determined by the request's `BlockTable`.

### Cache Gather (`CacheScatterGather.gather_sequence_layer`)
Reconstructs the logical sequence `[seq_len, num_kv_heads, head_dim]` by reading non-contiguous physical blocks from `TensorBlockStore`. Tested across exact token lengths `1`, `15`, `16`, `17`, `31`, `32`, `33`.

### Request Paged View (`PagedKVView`)
Provides a request-level view bundling `BlockTable`, `seq_length`, and `TensorBlockStore`. Exposes `.gather_keys(layer)`, `.gather_values(layer)`, and `.to_hf_dynamic_cache()`.

---

---

## 7. Direct Blockwise Paged Attention (`paged_attention_blockwise`)

PagedServe supports both gather-based reference attention and direct blockwise paged attention:
- `paged_attention_reference`: Gathers non-contiguous physical blocks into a full contiguous tensor $O(S \cdot H \cdot D)$ before attention.
- `paged_attention_blockwise`: Iterates directly over physical blocks in logical sequence using online softmax, maintaining $O(\text{block\_size} \cdot H \cdot D)$ temporary memory overhead per block.

See [BLOCKWISE_ATTENTION.md](file:///Users/sarthak/.gemini/antigravity/scratch/pagedserve/docs/BLOCKWISE_ATTENTION.md) for full mathematical formulation and benchmark comparisons.

