# PagedServe Limitations & Operational Boundaries

PagedServe is an experimental LLM-serving systems reference with paged KV allocation, continuous-batching simulation and attention-kernel experiments.

## Architectural Scope & Subsystems

1. **KV Memory Virtualization**:
   - Implements block-based virtual memory allocation (`KVBlock`, `BlockPool`, `BlockTable`, `PrefixCache`).
   - Manages logical-to-physical block mappings, reference counting, and LRU prefix eviction.

2. **Continuous Batching & Scheduling**:
   - Token-by-token continuous batching scheduler with FCFS and memory-aware admission control.
   - Chunked prefill support and cancellation cleanup.

3. **Attention Kernel Experiments**:
   - Blockwise online softmax attention reference implementation in PyTorch (`paged_attention_blockwise`).
   - Custom Metal Shading Language (`metal_shader.metal`) kernel for single-token decode on Apple Silicon GPUs.

4. **Model Execution**:
   - Model execution relies on PyTorch and Hugging Face Transformers (`AutoModelForCausalLM`).
   - Validated on `distilgpt2` and `TinyLlama-1.1B`.

5. **Simulated Components**:
   - **Tensor Parallelism**: Partitioning slice simulation across logical ranks; does not implement distributed multi-node GPU ring-AllReduce network collectives.
   - **Speculative Decoding**: Standalone verification and rejection sampling routine on probability distributions; not an integrated multi-model speculative serving pipeline.
   - **Benchmarks**: Benchmarks are clearly separated into synthetic scheduler simulations, microbenchmarks, and real-model HuggingFace throughput measurements.
