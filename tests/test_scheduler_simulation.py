"""High-level continuous batching simulation test with heterogeneous mixed workloads."""

import time
from pagedserve.engine.request import InferenceRequest, SamplingParams
from pagedserve.engine.state import RequestState
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.scheduler.batch import WorkType
from pagedserve.scheduler.scheduler import Scheduler


def test_heterogeneous_scheduler_simulation():
    """Simulate engine scheduling across requests with widely varying prompt and output lengths:
    
    A: prompt 50,   output 5
    B: prompt 2000, output 3  (Long prompt, multi-chunk prefill)
    C: prompt 20,   output 8  (Short prompt, long decode)
    D: prompt 500,  output 2  (Medium prompt)
    
    Demonstrates:
    - Long prefill is chunked without monopolizing iterations.
    - Shorter decoding requests (A, C) interleave and make progress while B is still prefilling.
    - Continuous batching dynamically admits and retires sequences.
    - KV memory usage strictly stays within pool bounds.
    - All requests eventually finish with exact expected token counts.
    - Full KV block recovery to the free pool with zero leaks.
    """
    block_size = 16
    total_blocks = 200  # 3200 tokens total capacity
    mgr = KVCacheManager(num_blocks=total_blocks, block_size=block_size)

    prefill_chunk_size = 256
    scheduler = Scheduler(
        kv_cache_mgr=mgr,
        max_num_sequences=8,
        max_batch_tokens=512,
        max_prefill_tokens=256,
        prefill_chunk_size=prefill_chunk_size,
    )

    # Construct the 4 requests
    t0 = time.monotonic()
    req_a = InferenceRequest(
        request_id="A",
        prompt="A" * 50,
        prompt_token_ids=list(range(1, 51)),
        sampling_params=SamplingParams(max_new_tokens=5),
        arrival_time=t0 + 0.1,
    )
    req_b = InferenceRequest(
        request_id="B",
        prompt="B" * 2000,
        prompt_token_ids=list(range(1, 2001)),
        sampling_params=SamplingParams(max_new_tokens=3),
        arrival_time=t0 + 0.2,
    )
    req_c = InferenceRequest(
        request_id="C",
        prompt="C" * 20,
        prompt_token_ids=list(range(1, 21)),
        sampling_params=SamplingParams(max_new_tokens=8),
        arrival_time=t0 + 0.3,
    )
    req_d = InferenceRequest(
        request_id="D",
        prompt="D" * 500,
        prompt_token_ids=list(range(1, 501)),
        sampling_params=SamplingParams(max_new_tokens=2),
        arrival_time=t0 + 0.4,
    )

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler.add_request(req_c)
    scheduler.add_request(req_d)

    # Execution telemetry to verify interleaving
    max_memory_observed = 0
    b_chunks_count = 0
    interleaved_decodes_while_b_prefilling = 0
    a_completed_step = None
    b_prefill_done_step = None

    step = 0
    max_steps = 100

    while step < max_steps:
        step += 1
        batch = scheduler.schedule()

        # Check termination condition
        if batch.is_empty and scheduler.num_running == 0 and scheduler.num_waiting == 0:
            break

        # Record memory stats
        max_memory_observed = max(max_memory_observed, mgr.used_blocks)
        assert mgr.used_blocks <= total_blocks, "Memory exceeded pool capacity!"

        # Track state of B
        b_in_prefill = "B" in scheduler.running_requests and scheduler.running_requests["B"].state == RequestState.PREFILL

        # Execute the scheduled items (simulated executor step)
        for item in batch.items:
            req = scheduler.running_requests[item.request_id]

            if item.work_type == WorkType.PREFILL:
                if item.request_id == "B":
                    b_chunks_count += 1
                req.advance_prompt_tokens(item.num_tokens)
                if req.is_prefill_complete:
                    req.mark_decoding()
                    if item.request_id == "B":
                        b_prefill_done_step = step

            elif item.work_type == WorkType.DECODE:
                if b_in_prefill and item.request_id in ("A", "C"):
                    interleaved_decodes_while_b_prefilling += 1

                # Append physical block if boundary crossed
                mgr.append_token(req.request_id, req.total_tokens + 1)
                # Sample token
                req.append_generated_token(token_id=2000 + req.num_generated_tokens)

                if req.request_id == "A" and req.is_finished:
                    a_completed_step = step

    # Final drain step to release any remaining terminal requests
    scheduler.schedule()

    # --- Assertions on Systems Engineering Behavior ---

    # 1. All requests must have completed successfully
    assert scheduler.num_completed == 4
    assert scheduler.num_running == 0
    assert scheduler.num_waiting == 0

    completed_map = {r.request_id: r for r in scheduler.completed_requests}
    assert completed_map["A"].num_generated_tokens == 5
    assert completed_map["B"].num_generated_tokens == 3
    assert completed_map["C"].num_generated_tokens == 8
    assert completed_map["D"].num_generated_tokens == 2

    # 2. Long prefill B was chunked across multiple steps:
    # On admission, A consumed 50 of the 256 prefill budget, leaving 206 for B's first chunk.
    # The remaining 1794 tokens were processed in 7 chunks of 256 (1792 tokens) + 1 chunk of 2 tokens = 9 chunks total.
    assert b_chunks_count == 9, f"Expected 9 prefill chunks for B, got {b_chunks_count}"

    # 3. Interleaving: Shorter decoding requests (A and C) MUST have progressed while B was prefilling!
    assert interleaved_decodes_while_b_prefilling > 0, (
        "Decode starvation detected: no decode steps were interleaved while B was prefilling"
    )

    # 4. Request A must have finished before Request B even finished prefilling!
    assert a_completed_step is not None
    assert b_prefill_done_step is not None
    assert a_completed_step <= b_prefill_done_step, (
        f"Expected A to finish (step {a_completed_step}) before B finished prefilling (step {b_prefill_done_step})"
    )

    # 5. Zero memory leaks: All blocks recovered to free pool
    mgr.verify_clean()
    assert mgr.used_blocks == 0
    assert mgr.free_blocks == total_blocks
