"""Exhaustive tests for KVCacheManager: prompt allocation, boundary crossing, and resource cleanup."""

import pytest

from pagedserve.errors import KVCacheExhaustedError
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.kv_cache import KVCacheManager


def test_kv_cache_prompt_allocation():
    """Verify prompt block allocation and token distribution."""
    # block_size = 16, num_blocks = 10
    mgr = KVCacheManager(num_blocks=10, block_size=16)
    assert mgr.total_blocks == 10
    assert mgr.free_blocks == 10
    assert mgr.used_blocks == 0
    assert mgr.utilization == 0.0

    # Prompt length = 35 tokens -> ceil(35 / 16) = 3 blocks
    table = mgr.allocate_for_prompt("req-1", prompt_len=35)
    assert table.num_blocks == 3
    assert mgr.used_blocks == 3
    assert mgr.free_blocks == 7
    assert mgr.utilization == 0.3
    assert mgr.num_active_requests == 1

    # Verify token distributions in blocks:
    # Block 0: 16 tokens (full)
    # Block 1: 16 tokens (full)
    # Block 2: 3 tokens
    b0 = mgr._pool.get_block(table.physical_block_for(0))
    b1 = mgr._pool.get_block(table.physical_block_for(1))
    b2 = mgr._pool.get_block(table.physical_block_for(2))

    assert b0.num_tokens == 16
    assert b0.is_full
    assert b1.num_tokens == 16
    assert b1.is_full
    assert b2.num_tokens == 3
    assert not b2.is_full
    assert b2.available_slots == 13


def test_kv_cache_exact_block_boundary_prompt():
    """Verify prompt length exactly matching a block multiple."""
    mgr = KVCacheManager(num_blocks=8, block_size=16)
    # Prompt length = 32 -> exactly 2 blocks
    table = mgr.allocate_for_prompt("req-exact", prompt_len=32)
    assert table.num_blocks == 2
    b0 = mgr._pool.get_block(table.physical_block_for(0))
    b1 = mgr._pool.get_block(table.physical_block_for(1))
    assert b0.num_tokens == 16
    assert b1.num_tokens == 16
    assert b0.is_full and b1.is_full


def test_kv_cache_sequence_expansion_across_boundary():
    """Verify decode steps crossing block boundaries allocate exactly 1 new block."""
    # block_size = 4, num_blocks = 6
    mgr = KVCacheManager(num_blocks=6, block_size=4)

    # Initial prompt length = 3 tokens (fits in 1 block, 1 slot free)
    table = mgr.allocate_for_prompt("req-expand", prompt_len=3)
    assert table.num_blocks == 1
    assert mgr.used_blocks == 1

    # Decode step 1: total len = 4 (fills remaining slot in Block 0)
    new_blk = mgr.append_token("req-expand", current_total_len=4)
    assert new_blk is None  # No new block needed
    assert table.num_blocks == 1
    b0 = mgr._pool.get_block(table.physical_block_for(0))
    assert b0.num_tokens == 4
    assert b0.is_full

    # Decode step 2: total len = 5 -> Boundary crossed! Allocates Block 1
    new_blk_id = mgr.append_token("req-expand", current_total_len=5)
    assert new_blk_id is not None
    assert table.num_blocks == 2
    assert mgr.used_blocks == 2
    assert table.physical_block_for(1) == new_blk_id
    b1 = mgr._pool.get_block(new_blk_id)
    assert b1.num_tokens == 1

    # Decode step 3: total len = 6 (fits in Block 1)
    new_blk = mgr.append_token("req-expand", current_total_len=6)
    assert new_blk is None
    assert table.num_blocks == 2
    assert b1.num_tokens == 2


def test_kv_cache_exhaustion_at_prompt():
    """Verify prompt allocation fails cleanly when pool has insufficient blocks."""
    mgr = KVCacheManager(num_blocks=3, block_size=16)

    # Prompt requiring 4 blocks (64 tokens requires 4 blocks)
    assert not mgr.can_allocate_prompt(64)
    with pytest.raises(KVCacheExhaustedError):
        mgr.allocate_for_prompt("req-oom", prompt_len=64)

    # State must be completely clean (no leaks)
    mgr.verify_clean()


def test_kv_cache_exhaustion_during_decode():
    """Verify OOM during decode token append raises KVCacheExhaustedError."""
    mgr = KVCacheManager(num_blocks=2, block_size=4)

    # Allocate request 1: prompt_len = 4 (1 block)
    mgr.allocate_for_prompt("req-1", prompt_len=4)
    # Allocate request 2: prompt_len = 4 (1 block)
    mgr.allocate_for_prompt("req-2", prompt_len=4)

    # Pool is now full (0 free blocks)
    assert mgr.free_blocks == 0

    # Request 1 attempts decode step crossing boundary -> total len = 5
    assert not mgr.can_append_token("req-1", current_total_len=5)
    with pytest.raises(KVCacheExhaustedError):
        mgr.append_token("req-1", current_total_len=5)

    # Cleanup request 2 frees 1 block
    mgr.release_request("req-2")
    assert mgr.free_blocks == 1

    # Now Request 1 can append
    assert mgr.can_append_token("req-1", current_total_len=5)
    new_blk_id = mgr.append_token("req-1", current_total_len=5)
    assert new_blk_id is not None

    # Cleanup request 1 -> perfectly clean
    mgr.release_request("req-1")
    mgr.verify_clean()


def test_kv_cache_idempotent_release_and_double_cleanup():
    """Verify releasing an already released or non-existent request is safe."""
    mgr = KVCacheManager(num_blocks=5, block_size=16)
    mgr.allocate_for_prompt("req-safe", prompt_len=20)
    assert mgr.used_blocks == 2

    # First release
    mgr.release_request("req-safe")
    assert mgr.used_blocks == 0
    mgr.verify_clean()

    # Second release of same request (safe no-op)
    mgr.release_request("req-safe")
    mgr.verify_clean()

    # Release unknown request (safe no-op)
    mgr.release_request("req-unknown")
    mgr.verify_clean()


def test_kv_cache_concurrent_multi_request_lifecycle():
    """Simulate multiple requests concurrently allocating, expanding, and finishing."""
    mgr = KVCacheManager(num_blocks=20, block_size=8)

    # Submit 3 requests
    mgr.allocate_for_prompt("r1", prompt_len=12)  # 2 blocks
    mgr.allocate_for_prompt("r2", prompt_len=20)  # 3 blocks
    mgr.allocate_for_prompt("r3", prompt_len=7)   # 1 block

    assert mgr.used_blocks == 6
    assert mgr.free_blocks == 14

    # Expand r3 by 2 tokens -> len = 9 (crosses boundary from 8 to 9, needs 2nd block)
    mgr.append_token("r3", current_total_len=8)   # fits in block 0
    mgr.append_token("r3", current_total_len=9)   # allocates block 1
    assert mgr.used_blocks == 7

    # Cancel r2 early (e.g. client disconnect simulation)
    mgr.release_request("r2")
    assert mgr.used_blocks == 4

    # r1 finishes normally
    mgr.release_request("r1")
    assert mgr.used_blocks == 2

    # r3 finishes normally
    mgr.release_request("r3")
    assert mgr.used_blocks == 0

    # Verify zero leaks
    mgr.verify_clean()
    stats = mgr.get_stats()
    assert stats.used_blocks == 0
    assert stats.free_blocks == 20
    assert stats.utilization == 0.0
    assert stats.num_active_requests == 0


def test_kv_cache_blocks_by_request_tracking():
    """Verify blocks_by_request accurately reflects mapping."""
    mgr = KVCacheManager(num_blocks=10, block_size=16)
    t1 = mgr.allocate_for_prompt("r1", prompt_len=16)
    t2 = mgr.allocate_for_prompt("r2", prompt_len=32)

    by_req = mgr.blocks_by_request
    assert by_req["r1"] == t1.blocks
    assert by_req["r2"] == t2.blocks

    mgr.release_request("r1")
    mgr.release_request("r2")
    mgr.verify_clean()


def test_kv_cache_shared_blocks_across_requests():
    """Verify shared physical block retention and reference counting across requests.

    When Request A and Request B share a physical block (e.g. prefix sharing),
    releasing Request A must NOT return the block to the pool until Request B also finishes.
    """
    mgr = KVCacheManager(num_blocks=4, block_size=16)

    # Request A allocates 1 block (block 0)
    tA = mgr.allocate_for_prompt("req-A", prompt_len=16)
    shared_block_id = tA.physical_block_for(0)
    assert mgr.used_blocks == 1
    assert mgr.free_blocks == 3

    # Request B shares the same physical block directly
    mgr._pool.retain(shared_block_id)
    table_B = BlockTable("req-B")
    table_B.append_block(shared_block_id)
    mgr._tables["req-B"] = table_B
    assert mgr.used_blocks == 1  # 1 unique block used, but ref_count is 2
    assert mgr._pool.get_block(shared_block_id).ref_count == 2

    # Release Request A
    mgr.release_request("req-A")
    # Shared block 0 is still retained by Request B!
    assert mgr._pool.get_block(shared_block_id).ref_count == 1
    assert mgr._pool.get_block(shared_block_id).is_allocated

    # Release Request B
    mgr.release_request("req-B")
    # Now all blocks are freed
    mgr.verify_clean()


def test_kv_cache_failure_and_cancellation_cleanup():
    """Verify that simulated request failures and cancellations leave zero orphaned blocks."""
    mgr = KVCacheManager(num_blocks=8, block_size=8)

    # 3 requests in flight
    mgr.allocate_for_prompt("req-fail", prompt_len=16)    # 2 blocks
    mgr.allocate_for_prompt("req-cancel", prompt_len=24)  # 3 blocks
    mgr.allocate_for_prompt("req-normal", prompt_len=8)   # 1 block
    assert mgr.used_blocks == 6

    # Expand req-fail and simulate exception during execution
    mgr.append_token("req-fail", current_total_len=17)    # 3rd block allocated
    assert mgr.used_blocks == 7

    # Failure handler triggers release
    mgr.release_request("req-fail")
    assert mgr.used_blocks == 4

    # Cancellation handler triggers release
    mgr.release_request("req-cancel")
    assert mgr.used_blocks == 1

    # Normal finish triggers release
    mgr.release_request("req-normal")
    assert mgr.used_blocks == 0
    mgr.verify_clean()

