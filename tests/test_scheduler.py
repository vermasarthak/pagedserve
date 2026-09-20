"""Comprehensive unit tests for the Continuous Batching Scheduler, Prefill/Decode, and Chunking."""

import time

from pagedserve.engine.request import InferenceRequest, SamplingParams
from pagedserve.engine.state import RequestState
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.scheduler.batch import WorkType
from pagedserve.scheduler.scheduler import Scheduler


def make_request(req_id: str, prompt_len: int, max_new_tokens: int = 5, arrival_offset: float = 0.0) -> InferenceRequest:
    """Helper to construct dummy inference requests with distinct arrival times."""
    return InferenceRequest(
        request_id=req_id,
        prompt=f"Prompt for {req_id}",
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        arrival_time=time.monotonic() + arrival_offset,
    )


def test_scheduler_first_request_admission():
    """Verify a single request is admitted from waiting and scheduled for prefill."""
    mgr = KVCacheManager(num_blocks=10, block_size=16)
    scheduler = Scheduler(kv_cache_mgr=mgr, max_num_sequences=4)

    req = make_request("req-1", prompt_len=20)
    scheduler.add_request(req)
    assert scheduler.num_waiting == 1
    assert scheduler.num_running == 0

    batch = scheduler.schedule()
    assert not batch.is_empty
    assert len(batch.items) == 1
    item = batch.items[0]
    assert item.request_id == "req-1"
    assert item.work_type == WorkType.PREFILL
    assert item.num_tokens == 20
    assert item.prompt_token_range == (0, 20)

    assert scheduler.num_waiting == 0
    assert scheduler.num_running == 1
    assert req.state == RequestState.PREFILL
    assert mgr.used_blocks == 2  # ceil(20/16) = 2


def test_scheduler_fcfs_ordering():
    """Verify waiting requests are admitted strictly according to arrival_time."""
    mgr = KVCacheManager(num_blocks=20, block_size=16)
    scheduler = Scheduler(kv_cache_mgr=mgr, max_num_sequences=2)

    # Submit r2 first, then r1 with earlier arrival_time, then r3
    r1 = make_request("r1", prompt_len=10, arrival_offset=-10.0)
    r2 = make_request("r2", prompt_len=10, arrival_offset=0.0)
    r3 = make_request("r3", prompt_len=10, arrival_offset=10.0)

    scheduler.add_request(r2)
    scheduler.add_request(r1)
    scheduler.add_request(r3)

    # max_num_sequences = 2, so r1 and r2 should be admitted first (FCFS)
    batch = scheduler.schedule()
    admitted_ids = [item.request_id for item in batch.items]
    assert admitted_ids == ["r1", "r2"]
    assert scheduler.num_waiting == 1
    assert scheduler.waiting_requests[0].request_id == "r3"


def test_scheduler_max_sequence_capacity():
    """Verify scheduler respects max_num_sequences ceiling."""
    mgr = KVCacheManager(num_blocks=50, block_size=16)
    scheduler = Scheduler(kv_cache_mgr=mgr, max_num_sequences=3)

    for i in range(5):
        scheduler.add_request(make_request(f"r{i}", prompt_len=10, arrival_offset=float(i)))

    batch = scheduler.schedule()
    assert len(batch.items) == 3
    assert scheduler.num_running == 3
    assert scheduler.num_waiting == 2


def test_continuous_batching_immediate_admission():
    """Verify continuous batching: when sequence C finishes, D enters immediately without waiting for A & B."""
    mgr = KVCacheManager(num_blocks=30, block_size=16)
    scheduler = Scheduler(kv_cache_mgr=mgr, max_num_sequences=3)

    # Step 1: Enqueue A, B, C, D
    req_a = make_request("A", prompt_len=10, arrival_offset=1.0)
    req_b = make_request("B", prompt_len=10, arrival_offset=2.0)
    req_c = make_request("C", prompt_len=10, arrival_offset=3.0)
    req_d = make_request("D", prompt_len=10, arrival_offset=4.0)

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler.add_request(req_c)
    scheduler.add_request(req_d)

    # Iteration 1: Admits A, B, C (running = 3, waiting = 1 [D])
    batch_1 = scheduler.schedule()
    assert [item.request_id for item in batch_1.items] == ["A", "B", "C"]
    assert scheduler.num_running == 3
    assert scheduler.num_waiting == 1

    # Simulate prefill complete and transition to DECODING
    for r in [req_a, req_b, req_c]:
        r.advance_prompt_tokens(10)
        r.mark_decoding()

    # Sequence C generates its last token and finishes!
    req_c.append_generated_token(100)
    req_c.mark_finished(reason="stop")
    assert req_c.is_finished

    # Iteration 2:
    # - C is drained and its KV blocks released
    # - A and B decode 1 token
    # - D enters IMMEDIATELY into the vacated slot (Continuous Batching!)
    batch_2 = scheduler.schedule()

    scheduled_ids = [item.request_id for item in batch_2.items]
    assert "A" in scheduled_ids
    assert "B" in scheduled_ids
    assert "D" in scheduled_ids
    assert "C" not in scheduled_ids

    # Verify A & B got DECODE work, and D got PREFILL work in the SAME iteration
    item_a = next(it for it in batch_2.items if it.request_id == "A")
    item_d = next(it for it in batch_2.items if it.request_id == "D")
    assert item_a.work_type == WorkType.DECODE
    assert item_d.work_type == WorkType.PREFILL

    assert scheduler.num_running == 3  # A, B, D
    assert scheduler.num_waiting == 0  # D was admitted
    assert scheduler.num_completed == 1


def test_chunked_prefill_2500_tokens_into_1024_chunks():
    """Verify a 2500-token prompt chunks into 1024, 1024, 452 tokens across iterations."""
    mgr = KVCacheManager(num_blocks=200, block_size=16)
    # prefill_chunk_size = 1024, max_prefill_tokens = 1024
    scheduler = Scheduler(
        kv_cache_mgr=mgr,
        max_num_sequences=4,
        max_batch_tokens=2048,
        max_prefill_tokens=1024,
        prefill_chunk_size=1024,
    )

    req = make_request("req-long", prompt_len=2500)
    scheduler.add_request(req)

    # Iteration 1: Chunk 1 (0..1024)
    b1 = scheduler.schedule()
    assert len(b1.items) == 1
    assert b1.items[0].num_tokens == 1024
    assert b1.items[0].prompt_token_range == (0, 1024)
    # Simulate executor advancing prompt tokens
    req.advance_prompt_tokens(1024)
    assert req.state == RequestState.PREFILL
    assert not req.is_prefill_complete

    # Iteration 2: Chunk 2 (1024..2048)
    b2 = scheduler.schedule()
    assert len(b2.items) == 1
    assert b2.items[0].num_tokens == 1024
    assert b2.items[0].prompt_token_range == (1024, 2048)
    req.advance_prompt_tokens(1024)
    assert req.state == RequestState.PREFILL
    assert not req.is_prefill_complete

    # Iteration 3: Final Chunk (2048..2500 -> 452 tokens)
    b3 = scheduler.schedule()
    assert len(b3.items) == 1
    assert b3.items[0].num_tokens == 452
    assert b3.items[0].prompt_token_range == (2048, 2500)
    req.advance_prompt_tokens(452)
    assert req.is_prefill_complete

    # Transition to decode
    req.mark_decoding()

    # Iteration 4: Auto-regressive decode step (1 token)
    b4 = scheduler.schedule()
    assert len(b4.items) == 1
    assert b4.items[0].work_type == WorkType.DECODE
    assert b4.items[0].num_tokens == 1

    # Cleanup
    scheduler.cancel_request("req-long")
    scheduler.schedule()  # drain
    mgr.verify_clean()


def test_interleaved_prefill_and_decode():
    """Verify decode sequences continue executing while a long prompt is chunking."""
    mgr = KVCacheManager(num_blocks=100, block_size=16)
    scheduler = Scheduler(
        kv_cache_mgr=mgr,
        max_num_sequences=4,
        max_batch_tokens=2048,
        max_prefill_tokens=500,
        prefill_chunk_size=500,
    )

    # Request A is an active decoding request
    req_a = make_request("req-A", prompt_len=10)
    scheduler.add_request(req_a)
    scheduler.schedule()
    req_a.advance_prompt_tokens(10)
    req_a.mark_decoding()

    # Request B arrives with a 1200-token prompt
    req_b = make_request("req-B", prompt_len=1200)
    scheduler.add_request(req_b)

    # Next iteration: Both A (decode) AND B (chunk 1 of prefill) must be scheduled!
    batch = scheduler.schedule()
    assert len(batch.items) == 2

    decode_item = next(it for it in batch.items if it.request_id == "req-A")
    prefill_item = next(it for it in batch.items if it.request_id == "req-B")

    assert decode_item.work_type == WorkType.DECODE
    assert decode_item.num_tokens == 1

    assert prefill_item.work_type == WorkType.PREFILL
    assert prefill_item.num_tokens == 500  # bounded by prefill_chunk_size

    # Clean up
    scheduler.cancel_request("req-A")
    scheduler.cancel_request("req-B")
    scheduler.schedule()
    mgr.verify_clean()


def test_token_budget_enforcement():
    """Verify max_batch_tokens and max_prefill_tokens budgets are strictly enforced."""
    mgr = KVCacheManager(num_blocks=100, block_size=16)
    # max_batch_tokens = 50, max_prefill_tokens = 40, prefill_chunk_size = 30
    scheduler = Scheduler(
        kv_cache_mgr=mgr,
        max_num_sequences=10,
        max_batch_tokens=50,
        max_prefill_tokens=40,
        prefill_chunk_size=30,
    )

    r1 = make_request("r1", prompt_len=100, arrival_offset=1.0)
    r2 = make_request("r2", prompt_len=100, arrival_offset=2.0)
    scheduler.add_request(r1)
    scheduler.add_request(r2)

    batch = scheduler.schedule()
    # r1 gets min(100, 30, 50, 40) = 30 tokens
    # Remaining batch_budget = 20, remaining prefill_budget = 10
    # r2 gets min(100, 30, 20, 10) = 10 tokens
    # Total prefill scheduled = 40 <= max_prefill_tokens (40)
    assert batch.total_tokens <= 50
    assert batch.num_prefill_tokens <= 40

    scheduler.cancel_request("r1")
    scheduler.cancel_request("r2")
    scheduler.schedule()
    mgr.verify_clean()


def test_memory_exhaustion_leaves_request_safely_waiting():
    """Verify when KV blocks are insufficient, candidate request safely waits without side effects."""
    # Only 2 blocks = 32 tokens max capacity
    mgr = KVCacheManager(num_blocks=2, block_size=16)
    scheduler = Scheduler(kv_cache_mgr=mgr, max_num_sequences=4)

    # r1 uses 2 blocks (30 tokens)
    r1 = make_request("r1", prompt_len=30, arrival_offset=1.0)
    # r2 needs 2 blocks (25 tokens), but pool only has 0 free blocks
    r2 = make_request("r2", prompt_len=25, arrival_offset=2.0)

    scheduler.add_request(r1)
    scheduler.add_request(r2)

    # Iteration 1: admits r1, r2 cannot be admitted due to memory exhaustion
    b1 = scheduler.schedule()
    assert b1.has_request("r1")
    assert not b1.has_request("r2")
    assert scheduler.num_running == 1
    assert scheduler.num_waiting == 1
    assert scheduler.waiting_requests[0].request_id == "r2"
    assert mgr.used_blocks == 2

    # Advance and finish r1
    r1.advance_prompt_tokens(30)
    r1.mark_decoding()
    r1.append_generated_token(1)
    r1.mark_finished()

    # Iteration 2: r1 drains, frees 2 blocks, r2 is NOW admitted!
    b2 = scheduler.schedule()
    assert b2.has_request("r2")
    assert scheduler.num_running == 1
    assert scheduler.num_waiting == 0
    assert scheduler.running_requests["r2"].state == RequestState.PREFILL

    # Finish r2 and verify zero leaks
    scheduler.cancel_request("r2")
    scheduler.schedule()
    mgr.verify_clean()


def test_cancellation_and_failure_reclaims_memory():
    """Verify cancelling or failing a running request reclaims KV blocks immediately."""
    mgr = KVCacheManager(num_blocks=10, block_size=16)
    scheduler = Scheduler(kv_cache_mgr=mgr, max_num_sequences=4)

    r1 = make_request("r1", prompt_len=32)
    scheduler.add_request(r1)
    scheduler.schedule()
    assert mgr.used_blocks == 2
    assert scheduler.num_running == 1

    # Cancel while running
    cancelled = scheduler.cancel_request("r1")
    assert cancelled
    assert scheduler.num_running == 0
    assert scheduler.num_cancelled == 1
    assert mgr.used_blocks == 0
    mgr.verify_clean()

    # Test failure while running
    r2 = make_request("r2", prompt_len=16)
    scheduler.add_request(r2)
    scheduler.schedule()
    assert mgr.used_blocks == 1

    failed = scheduler.fail_request("r2", error="Simulated OOM")
    assert failed
    assert scheduler.num_running == 0
    assert scheduler.num_failed == 1
    assert mgr.used_blocks == 0
    mgr.verify_clean()
