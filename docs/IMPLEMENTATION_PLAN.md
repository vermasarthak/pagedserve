# PagedServe: Implementation Plan & Milestone Roadmap

## Milestone Overview

The goal of PagedServe is to build an original, high-performance, and verifiable experimental LLM serving runtime from scratch in Python and PyTorch.

---

## Phase Breakdown

### Phase 1: Request Lifecycle & State Model (`pagedserve/engine/request.py`, `state.py`)
- Define `RequestState` enum: `WAITING`, `PREFILL`, `DECODING`, `FINISHED`, `CANCELLED`, `FAILED`.
- Define `InferenceRequest` dataclass:
  - Identification: `request_id`, `prompt`, `prompt_token_ids`, `arrival_time`.
  - Sampling params: `temperature`, `top_k`, `top_p`, `max_new_tokens`, `stop_token_ids`.
  - Execution tracking: `generated_token_ids`, `num_prompt_tokens_processed`, `state`.
  - Memory tracking: reference to request's `BlockTable`.
  - Properties: `num_generated_tokens`, `remaining_tokens`, `is_finished`, `ttft`, `total_latency`.

### Phase 2: KV Block Abstraction (`pagedserve/memory/block.py`)
- Define `KVBlock`:
  - Fields: `block_id: int`, `capacity: int`, `ref_count: int`, `num_tokens: int`, `is_allocated: bool`.
  - Invariants:
    - `ref_count == 0` if and only if block is unallocated.
    - `ref_count >= 1` when allocated.
    - `ref_count` strictly non-negative.
    - Cannot be allocated twice without retain; cannot be released below zero.

### Phase 3: Physical Block Pool (`pagedserve/memory/block_pool.py`)
- Define `BlockPool`:
  - Pre-allocates a fixed pool of `num_blocks` `KVBlock` instances.
  - Manages free list (LIFO/FIFO queue for locality).
  - Methods: `allocate() -> KVBlock`, `allocate_many(count: int) -> list[KVBlock]`, `retain(block_id: int)`, `release(block_id: int)`.
  - Statistics: `total_blocks`, `free_count`, `used_count`, `utilization`.
  - Error conditions: `KVCacheExhaustedError` on out-of-memory, double-free validation, invalid block ID checking.

### Phase 4: Logical Block Table (`pagedserve/memory/block_table.py`)
- Define `BlockTable`:
  - Logical-to-physical block mapping for an individual sequence.
  - `append_block(physical_block_id: int)`
  - `physical_block_for(logical_index: int) -> int`
  - `release_all(pool: BlockPool)`
  - `num_blocks`, `logical_indices`, `physical_indices`
  - Demonstrates why logical-to-physical indirection decouples continuous sequence growth from contiguous physical tensor layouts.

### Phase 5: KV Cache Manager (`pagedserve/memory/kv_cache.py`)
- Define `KVCacheManager`:
  - Owns `BlockPool` and maps `request_id -> BlockTable`.
  - `allocate_for_prompt(request_id: str, prompt_len: int) -> BlockTable`
  - `append_token(request_id: str, current_total_len: int) -> Optional[int]` (allocates new block when crossing `block_size` boundary).
  - `release_request(request_id: str)`
  - Utilization telemetry: `get_stats() -> CacheStats`.

### Phase 6: Prefix Cache (`pagedserve/memory/prefix_cache.py`)
- Content-addressed block deduplication:
  - Cryptographic chained hashing: $\text{Hash}_k = H(\text{Hash}_{k-1}, \text{tokens}_k)$.
  - Maps `block_hash -> physical_block_id`.
  - `lookup(prefix_hashes: list[str]) -> list[KVBlock]`
  - `insert(prefix_hash: str, block: KVBlock)`
  - LRU eviction of unreferenced blocks when cache capacity is exceeded.
  - Verification: identical prompt prefixes share physical blocks; releasing one request does not affect active peers.

### Phase 7: Scheduler & Continuous Batching (`pagedserve/scheduler/`)
- Queue management: `waiting_queue`, `running_requests`, `completed_requests`.
- `ContinuousBatch`:
  - Prefill batch: list of requests in `PREFILL` state.
  - Decode batch: list of requests in `DECODING` state.
- Iteration step:
  - Step 1: Drain finished/cancelled requests and release KV blocks.
  - Step 2: Check KV block capacity for running decode sequences.
  - Step 3: Schedule decode tokens.
  - Step 4: If budget permits (`max_batch_tokens`, `max_num_sequences`, KV free blocks), admit waiting requests for prefill.
- Base policy: First-Come, First-Served (`FCFS`).

### Phase 8: Prefill vs. Decode Explicit Separation
- Explicit batch structures and execution branches in scheduler and runner.
- Separate tracking of prefill token compute vs. decode token generation.

### Phase 9: Chunked Prefill
- Configurable `max_prefill_tokens_per_iteration` (e.g. 512 tokens).
- Chunking large prompts across iterations while interleaving single-token decodes from already active sequences.
- Prevents TPOT spikes and decode latency degradation.

### Phase 10: Model Loader (`pagedserve/model/loader.py`)
- Clean Hugging Face causal LM loader.
- Auto-detects device: `cuda`, `mps`, or `cpu`.
- Provides easy loading for tiny test models (`sshleifer/tiny-gpt2`) and standard models.

### Phase 11: Model Runner (`pagedserve/model/runner.py`)
- Runs prefill forward pass (calculates logits for prompt tokens).
- Runs single-token decode pass for batched sequences.
- Employs PyTorch KV caching with clear hooks for block mapping.

### Phase 12: Sampler (`pagedserve/model/sampler.py`)
- Modular sampler:
  - Greedy argmax.
  - Temperature scaling.
  - Top-k filtering.
  - Top-p (nucleus) filtering.
  - Repetition penalty support.

### Phase 13: Main Engine Loop (`pagedserve/engine/engine.py`)
- Orchestrates scheduler, runner, sampler, and KV cache manager.
- Thread-safe / async-friendly: `submit()`, `step()`, `step_async()`, `cancel()`.
- Guarantees complete block reclamation on normal finish, exception/failure, or client cancellation.

### Phase 14: Sequential Baseline Implementation (`pagedserve/engine/baseline.py`)
- Naive sequential inference implementation using standard Hugging Face pipeline for side-by-side throughput, latency, and memory comparison.

### Phase 15: Streaming & Async Generation (`pagedserve/server/streaming.py`)
- Asynchronous generator for SSE (Server-Sent Events).
- Yields tokens as they are produced by the engine step loop.
- Instant cancellation detection when HTTP connection aborts.

### Phase 16: HTTP API Server (`pagedserve/server/api.py`, `schemas.py`)
- Endpoints:
  - `GET /health`: Engine status, loaded model, device, memory pool.
  - `GET /metrics`: Latency, throughput, block pool statistics.
  - `POST /v1/completions`: OpenAI-compatible text completion.
  - `POST /v1/chat/completions`: OpenAI-compatible chat completion.

### Phase 17: Metrics Collector (`pagedserve/metrics/collector.py`)
- Monotonic clock timing for TTFT, TPOT, requests/sec, tokens/sec.
- Block utilization tracking over time.

### Phase 18: Benchmarking Suite (`benchmarks/`)
- `benchmark_latency.py`: TTFT and TPOT across sequence lengths.
- `benchmark_throughput.py`: Requests/sec and tokens/sec under concurrencies (1, 2, 4, 8, 16, 32).
- `benchmark_scheduler.py`: Continuous batching vs. sequential baseline.
- `benchmark_memory.py`: KV block pool vs contiguous allocation simulation.

### Phase 19: Memory Allocator Fragmentation Simulation
- Synthetic simulator comparing:
  - Contiguous buffer allocation (fails under fragmentation).
  - Paged/block-based allocator (near-zero external fragmentation).
- Measures reserved memory vs. used memory.

### Phase 20: Original Systems Experiment: `MemoryAwareScheduler`
- Experimental scheduling policy that adjusts prefill admissions and chunk sizes based on real-time KV block utilization and remaining decode steps.
- Rigorous comparative benchmark against standard FCFS under memory stress.

### Phase 21: Hardware/GPU Work & Kernels (`pagedserve/kernels/`)
- Hardware feature detection (CUDA, MPS, CPU).
- CPU/MPS fallbacks for all operations.
- Clean vector sampling or block gather implementations.

### Phase 22: Comprehensive Test Suite (`tests/`)
- Unit tests for:
  - `test_block_pool.py`: allocation, double-free, exhaustion, ref counting.
  - `test_block_table.py`: logical-physical indirection, multi-block mapping.
  - `test_kv_cache.py`: prompt reservation, decode token boundary growth, release.
  - `test_prefix_cache.py`: prefix deduplication, ref-count integrity, LRU eviction.
  - `test_scheduler.py`: continuous batching, prefill chunking, FCFS order.
  - `test_sampler.py`: greedy determinism, top-k/top-p filtering.
  - `test_engine.py`: end-to-end request lifecycle, cancellation, error recovery.
  - `test_api.py`: FastAPI endpoints, SSE streaming, input validation.

### Phase 23: Concurrency & Thread Safety Audit
- Validates async request submission, concurrent batching, and zero race conditions in block reclamation.

### Phase 24: Error Handling & Invariants Audit
- Explicit exception classes: `KVCacheExhaustedError`, `InvalidRequestError`, `EngineCancelledError`, etc.
- Audits repository for `pass`, `TODO`, `NotImplementedError`.

### Phase 25: Comprehensive Documentation & Report
- `README.md`: Architecture, benchmarks, installation, API guide.
- `docs/KV_CACHE.md`, `docs/SCHEDULER.md`, `docs/DESIGN_DECISIONS.md`, `docs/BENCHMARKING.md`.
- Final engineering report.
