"""Exhaustive tests for BlockPool: allocation, exhaustion, double-free, and ref counts."""

import pytest
from pagedserve.memory.block_pool import BlockPool
from pagedserve.errors import (
    KVCacheExhaustedError,
    InvalidBlockError,
    DoubleFreeError,
    BlockInvariantError,
)


def test_pool_initialization():
    """Verify pool setup and initial metrics."""
    pool = BlockPool(num_blocks=10, block_size=16)
    assert pool.total_blocks == 10
    assert pool.block_size == 16
    assert pool.free_count == 10
    assert pool.used_count == 0
    assert pool.utilization == 0.0

    # Invalid pool parameters
    with pytest.raises(BlockInvariantError):
        BlockPool(num_blocks=0, block_size=16)
    with pytest.raises(BlockInvariantError):
        BlockPool(num_blocks=10, block_size=-1)


def test_pool_single_allocation_and_release():
    """Verify single block allocation and release mechanics."""
    pool = BlockPool(num_blocks=4, block_size=16)

    b0 = pool.allocate()
    assert b0.block_id == 0
    assert pool.free_count == 3
    assert pool.used_count == 1
    assert pool.utilization == 0.25

    b1 = pool.allocate()
    assert b1.block_id == 1
    assert pool.free_count == 2
    assert pool.used_count == 2
    assert pool.utilization == 0.5

    # Release b0
    rem = pool.release(b0.block_id)
    assert rem == 0
    assert pool.free_count == 3
    assert pool.used_count == 1
    assert pool.utilization == 0.25

    # Re-allocating should return b0 (since it was returned to free queue)
    b_new = pool.allocate()
    # Free queue had [2, 3] and b0 appended, so next was 2
    assert b_new.block_id == 2
    assert pool.free_count == 2


def test_pool_allocate_many_atomic():
    """Verify allocate_many allocates all blocks or raises without side effects."""
    pool = BlockPool(num_blocks=5, block_size=16)

    # Allocate 0 blocks
    empty = pool.allocate_many(0)
    assert empty == []
    assert pool.free_count == 5

    # Allocate 3 blocks
    blocks = pool.allocate_many(3)
    assert len(blocks) == 3
    assert [b.block_id for b in blocks] == [0, 1, 2]
    assert pool.free_count == 2
    assert pool.used_count == 3

    # Attempt to allocate 3 blocks when only 2 are free (must raise and not allocate partial)
    with pytest.raises(KVCacheExhaustedError) as exc_info:
        pool.allocate_many(3)
    assert exc_info.value.requested == 3
    assert exc_info.value.available == 2
    assert exc_info.value.total == 5

    # Verify state was untouched
    assert pool.free_count == 2
    assert pool.used_count == 3
    pool.verify_invariants()


def test_pool_exhaustion():
    """Verify pool completely exhausts and raises KVCacheExhaustedError."""
    pool = BlockPool(num_blocks=2, block_size=8)
    b0 = pool.allocate()
    b1 = pool.allocate()
    assert pool.free_count == 0
    assert pool.used_count == 2
    assert pool.utilization == 1.0

    with pytest.raises(KVCacheExhaustedError) as exc:
        pool.allocate()
    assert exc.value.requested == 1
    assert exc.value.available == 0

    # Release one block, then allocation succeeds
    pool.release(b0.block_id)
    assert pool.free_count == 1
    b2 = pool.allocate()
    assert b2.block_id == b0.block_id
    assert pool.free_count == 0


def test_pool_double_free_protection():
    """Verify double freeing a block is immediately rejected."""
    pool = BlockPool(num_blocks=3, block_size=16)
    b0 = pool.allocate()

    # Free once
    pool.release(b0.block_id)

    # Free twice
    with pytest.raises(DoubleFreeError, match="already in the free pool"):
        pool.release(b0.block_id)


def test_pool_invalid_block_ids():
    """Verify out of bounds or negative block IDs raise InvalidBlockError."""
    pool = BlockPool(num_blocks=4, block_size=16)

    with pytest.raises(InvalidBlockError):
        pool.get_block(-1)

    with pytest.raises(InvalidBlockError):
        pool.get_block(4)

    with pytest.raises(InvalidBlockError):
        pool.release(-1)

    with pytest.raises(InvalidBlockError):
        pool.release(99)


def test_pool_shared_retain_and_release():
    """Verify retain / release multi-reference mechanics inside the pool."""
    pool = BlockPool(num_blocks=4, block_size=16)
    b = pool.allocate()
    block_id = b.block_id

    # Retain for a second sequence
    ref2 = pool.retain(block_id)
    assert ref2 == 2
    assert pool.used_count == 1
    assert pool.free_count == 3

    # Release from sequence 1: block remains used
    rem1 = pool.release(block_id)
    assert rem1 == 1
    assert pool.used_count == 1
    assert pool.free_count == 3

    # Release from sequence 2: block now freed
    rem2 = pool.release(block_id)
    assert rem2 == 0
    assert pool.used_count == 0
    assert pool.free_count == 4
    pool.verify_invariants()


def test_pool_cannot_retain_free_block():
    """Cannot retain a block that is in the free list."""
    pool = BlockPool(num_blocks=4, block_size=16)
    with pytest.raises(BlockInvariantError, match="free list"):
        pool.retain(0)


def test_pool_zero_leaks_after_heavy_churn():
    """Verify invariants and zero leaked blocks after heavy allocate/free churn."""
    pool = BlockPool(num_blocks=16, block_size=16)

    allocated_sets = []
    # Allocate in waves
    for _ in range(4):
        batch = pool.allocate_many(4)
        allocated_sets.append(batch)

    assert pool.free_count == 0
    assert pool.used_count == 16

    # Release half
    for b in allocated_sets[0] + allocated_sets[2]:
        pool.release(b.block_id)

    assert pool.free_count == 8
    assert pool.used_count == 8
    pool.verify_invariants()

    # Re-allocate
    new_batch = pool.allocate_many(8)
    assert pool.free_count == 0
    pool.verify_invariants()

    # Free everything
    for b in allocated_sets[1] + allocated_sets[3] + new_batch:
        pool.release(b.block_id)

    assert pool.free_count == 16
    assert pool.used_count == 0
    assert pool.utilization == 0.0
    pool.verify_invariants()
