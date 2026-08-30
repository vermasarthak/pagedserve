# PagedServe

PagedServe is an original experimental LLM inference runtime built from scratch to demonstrate modern systems engineering principles for high-throughput LLM serving.

## Features Currently Implemented

- **KV Memory Subsystem**: Block-based continuous batching and memory management, including `KVBlock`, `BlockPool`, `BlockTable`, and strict ref-counting.
- **Prefix Caching**: Content-addressed deduplication of full blocks using chained SHA-256 cryptographic hashing.
- **Continuous Batching Scheduler**: Iteration-level scheduling with FCFS, chunked prefill, token budgets, and memory-aware admission.
- **Model Execution**: Genuine transformer inference engine integrated!
  - `ModelLoader`: Device/dtype resolution.
  - `ModelRunner`: Real prefill and single-token decode utilizing `DynamicCache`.
  - `Sampler`: Temperature, top-k, top-p, and greedy sampling.
  - `PagedServeEngine`: Central orchestration handling multi-request concurrency and scheduling.

*Note: Genuine model execution is working, but performance claims are intentionally omitted as we are not yet implementing physical PagedAttention kernels or optimizing for raw latency/throughput.*

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [KV Cache](docs/KV_CACHE.md)
- [Scheduler](docs/SCHEDULER.md)
- [Model Execution](docs/MODEL_EXECUTION.md)
