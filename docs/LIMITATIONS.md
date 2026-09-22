# Architecture Boundaries & Limitations

PagedServe is an educational and research runtime designed to explore virtual memory management (Paged KV Cache), continuous batching scheduling, and model execution boundaries.

## Core Architectural Scope

1. **Memory Subsystem (Metadata Layer)**
   - Implements block-based virtual memory allocation (`KVBlock`, `BlockPool`, `BlockTable`, `PrefixCache`).
   - Manages logical-to-physical block mappings, reference counting, and LRU prefix eviction.

2. **Execution Subsystem (Framework Layer)**
   - Model execution relies on PyTorch and Hugging Face Transformers (`AutoModelForCausalLM`).
   - Auto-regressive decode steps utilize the framework's native `past_key_values` tensor allocation for sequence generation.

3. **Metal Attention Shaders**
   - Custom Metal Shading Language (`metal_shader.metal`) kernels provide standalone single-token decode benchmarks.
   - Current PyTorch HuggingFace integration uses standard tensor evaluation baselines rather than in-place kernel overrides.

## Future Engineering Roadmap

- Direct C++/CUDA & Metal framework extension bindings (`torch.ops`).
- Zero-copy tensor indexing inside transformer attention layers.
- Chunked prefill integration across pipeline parallel workers.
