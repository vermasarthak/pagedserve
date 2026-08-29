# PagedServe: KV Cache & Memory Management

## 1. What is a Transformer KV Cache?

During autoregressive generation in a decoder Transformer, computing attention for a newly generated token at step $t$ requires query-key dot products:
$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{softmax}\left(\frac{Q_t K_{\le t}^T}{\sqrt{d_k}}\right) V_{\le t}$$

Instead of recomputing key and value vectors for all previous tokens $1 \dots t-1$ on every single step (which would cause $O(t^2)$ redundant compute), the model caches past Key and Value activation tensors in memory:
$$\text{Memory per token} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$

For a model with 32 layers, 32 heads, head dimension 128 in FP16 (2 bytes), each token consumes:
$$2 \times 32 \times 32 \times 128 \times 2 = 524,288 \text{ bytes} \approx 0.5 \text{ MB per token}$$
A 4096-token sequence requires $2.0 \text{ GB}$ of KV cache memory alone!

---

## 2. Why Variable-Length Requests Complicate Memory Allocation

In naive inference runtimes, systems pre-allocate a contiguous maximum-length buffer (e.g. 2048 or 4096 tokens) for each sequence. This leads to two critical inefficiencies:
1. **Internal Fragmentation**: If a sequence finishes after generating only 50 tokens, the remaining 2000+ slots in the contiguous buffer remain allocated and wasted.
2. **External Fragmentation**: Because memory buffers are contiguous and of varying lifetime, allocations and frees create gaps in memory that cannot fit larger sequences, causing out-of-memory errors despite abundant aggregate free memory.
3. **Reservation Waste**: Even when sequences eventually reach maximum length, memory allocated for future tokens sits unused for hundreds of iterations.

Studies have shown traditional contiguous allocators waste **60% to 80%** of GPU memory.

---

## 3. PagedServe Architecture: Fixed-Size Block Memory Virtualization

PagedServe solves memory fragmentation by implementing a paged memory architecture:

### 3.1 Logical Blocks vs. Physical Blocks
- **Physical Memory** is partitioned into fixed-size blocks (e.g. `block_size = 16` tokens) pre-allocated in a global `BlockPool`.
- **Logical Sequence Memory** is maintained through a `BlockTable` mapping logical block indices to non-contiguous physical block IDs:
  $$\text{Logical Block } 0 \to \text{Physical Block } 17$$
  $$\text{Logical Block } 1 \to \text{Physical Block } 3$$
  $$\text{Logical Block } 2 \to \text{Physical Block } 42$$
- Physical blocks can be located anywhere in memory.
- Blocks are allocated strictly on demand as the sequence expands across block boundaries ($\text{len} \pmod{\text{block\_size}} == 1$).
- Internal fragmentation is strictly bounded to at most one partially filled block per sequence ($< \text{block\_size}$ tokens).

---

## 4. Prefix Caching & Chained Hashing

### 4.1 Content-Addressed Chained Hashing
To support shared prompt prefixes (such as system prompts or few-shot examples), PagedServe uses a deterministic chained cryptographic hash:
$$\text{Hash}_0 = \text{SHA256}(\text{tokens in block 0})$$
$$\text{Hash}_k = \text{SHA256}(\text{Hash}_{k-1} \,\|\, \text{tokens in block } k)$$

Chaining guarantees that block identity is strictly dependent on the entire preceding prefix history. Identical token subsequences appearing in different conversational contexts produce distinct hashes, preventing false cache hits.

### 4.2 Caching Rules
- **Full Blocks Only**: Only complete blocks ($\text{num\_tokens} == \text{block\_size}$) are cached. Trailing partial blocks are not cached because subsequent tokens in the prompt or decode phase will mutate their contents.

---

## 5. Three-Tier Ownership & Reference Counting Model

To eliminate memory leaks, double-frees, and premature deallocations, PagedServe implements an explicit three-tier reference model:

```mermaid
graph TD
    subgraph Ownership Tiers
        REQ_A[Active Request A] -->|table[0] ref| BLOCK[Physical Block ID: 17]
        REQ_B[Active Request B] -->|table[0] ref| BLOCK
        CACHE[PrefixCache Entry] -->|cache ref| BLOCK
        BLOCK -->|ref_count == 3| POOL[BlockPool]
    end
```

1. **Active Request Ownership**: Each active request holds 1 reference to every physical block registered in its `BlockTable`.
2. **PrefixCache Ownership**: When a physical block is inserted into the `PrefixCache`, the cache retains 1 reference (`pool.retain(block_id)`).
3. **BlockPool Reclaim**: A physical block is only returned to the pool's free queue when its `ref_count` drops to zero:
   $$\text{ref\_count} = \sum \text{Active Request References} + \mathbb{I}(\text{Cached in PrefixCache})$$

### Lifecycle Scenarios:
- **Request A finishes, Request B still active**: Request A calls `release_request()`. Block 17's `ref_count` drops from 3 to 2. Request B continues executing without disruption.
- **Both Request A and Request B finish**: Ref count drops to 1. The block remains allocated and cached in `PrefixCache`.
- **New Request C arrives with same prefix**: `PrefixCache.lookup()` finds block 17. Request C retains it (`ref_count` increases from 1 to 2). Zero physical memory allocated!
- **Cache Eviction**: Under memory pressure, `PrefixCache.evict_lru()` removes the entry and calls `pool.release(block_id)`. If no active requests are using it, `ref_count` drops to 0 and the block returns to the `BlockPool` free queue immediately.

---

## 6. Limitations of Current Implementation
- **CPU/MPS Metadata vs. Custom CUDA PagedAttention Kernel**: In this phase, the metadata management (allocation, translation, reference counting, hash deduplication) is 100% functional and verified. When hooked to PyTorch transformers, tensor gather/scatter or KV cache splicing is utilized rather than a custom Triton/CUDA paged attention kernel.
