# PagedServe: Continuous Batching & Scheduling Architecture

## 1. Static Batching vs. Continuous Batching

### 1.1 The Static Batching Bottleneck
In traditional inference servers (e.g. standard Hugging Face pipelines or TorchServe), requests are grouped into a static batch:
- If Request A generates 10 tokens and Request B generates 500 tokens, Request A's slot sits completely idle for 490 iterations while the batch waits for Request B to finish.
- New waiting requests cannot be admitted until all sequences in the batch have completed.
- Result: Low GPU utilization, high queuing delay, and poor throughput.

### 1.2 Continuous (Iteration-Level) Batching
Continuous batching operates at the granularity of individual engine iterations:
- At step $t$, if Request A finishes, it is retired immediately, freeing its KV memory and sequence slot.
- At step $t+1$, waiting Request C is admitted immediately to fill the vacated slot, running alongside the remaining decode steps of Request B.
- Result: Constant high device occupancy, drastically reduced queuing delays, and up to $4\times$ to $10\times$ higher throughput.

---

## 2. Queue Management

The `Scheduler` maintains five explicit request sets:
1. `waiting`: Queue of newly arrived requests awaiting compute and memory admission.
2. `running`: Active requests currently undergoing prompt prefill or autoregressive decode.
3. `completed`: Requests that successfully reached stop tokens or `max_new_tokens`.
4. `cancelled`: Requests aborted due to caller cancellation or client disconnect.
5. `failed`: Requests terminated due to unrecoverable runtime errors.

---

## 3. Prefill vs. Decode Separation

An inference step consists of two fundamentally distinct computational regimes:
1. **Prefill (Compute-Bound)**:
   - Processes all or part of a request's prompt tokens in parallel.
   - High compute intensity (GEMM matrix multiplications saturate tensor cores).
   - Generates the initial KV cache activations.
2. **Decode (Memory-Bandwidth Bound)**:
   - Generates exactly 1 new token per sequence at each step.
   - Low arithmetic intensity: reads all past KV cache activations from HBM/RAM to compute attention for a single query token.
   - Bound by memory read bandwidth.

PagedServe represents these explicitly in `ScheduledItem` (`WorkType.PREFILL` vs. `WorkType.DECODE`).

---

## 4. Chunked Prefill

### 4.1 The Problem: Prefill Starvation
A long prompt (e.g. 2048 or 4096 tokens) scheduled in a single iteration takes hundreds of milliseconds to compute. If running decode sequences must wait for that massive forward pass to complete, their Time-Per-Output-Token (TPOT) spikes dramatically, causing stuttering in interactive streaming sessions.

### 4.2 The Solution: Chunking
PagedServe chunks prompts into configurable budgets (e.g. `prefill_chunk_size = 512` tokens):
- Iteration 1: Decode sequences run 1 token + Request X runs tokens 0..512.
- Iteration 2: Decode sequences run 1 token + Request X runs tokens 512..1024.
- Iteration 3: Decode sequences run 1 token + Request X runs tokens 1024..1536.
- Iteration 4: Decode sequences run 1 token + Request X runs tokens 1536..1800 (prefill complete $\to$ transitions to `DECODING`).

Active decodes execute continuously without starvation, keeping TPOT stable and predictable.

---

## 5. Token Budgets & Memory Admission

Each scheduler iteration enforces three strict safety constraints:
1. `max_num_sequences`: Limits active concurrent sequences to avoid thrashing.
2. `max_batch_tokens`: Limits aggregate token compute in one forward pass.
3. `max_prefill_tokens`: Limits prompt tokens scheduled per step, reserving remaining budget for decodes.
4. **Memory Admission Check**:
   Before admitting a waiting request, the scheduler calls:
   ```python
   kv_cache_mgr.can_allocate_prompt_tokens(candidate.prompt_token_ids)
   ```
   If insufficient free blocks exist, the request remains safely in the `waiting` queue without mutating or corrupting allocator state.

---

## 6. Default FCFS Policy vs. Future MemoryAwareScheduler

- **First-Come, First-Served (FCFS)**: The default policy orders waiting and running queues by `arrival_time`. It guarantees fairness, predictable ordering, and absence of starvation under nominal load.
- **`MemoryAwareScheduler` (Phase 20 Experiment)**: An original experimental policy that monitors real-time KV block pressure. When memory utilization exceeds high watermarks (e.g. 85%), it throttles new admissions, reduces prefill chunk sizes, and prioritizes requests closest to completion to quickly reclaim memory.
