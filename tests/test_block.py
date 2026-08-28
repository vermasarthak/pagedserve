"""Comprehensive unit tests for KVBlock metadata model and invariants."""

import pytest
from pagedserve.memory.block import KVBlock
from pagedserve.errors import BlockInvariantError, DoubleFreeError


def test_block_initial_state():
    """Verify default initial unallocated state and invariants."""
    block = KVBlock(block_id=0, capacity=16)
    assert block.block_id == 0
    assert block.capacity == 16
    assert block.num_tokens == 0
    assert not block.is_allocated
    assert block.ref_count == 0
    assert block.is_empty
    assert not block.is_full
    assert block.available_slots == 16


def test_block_invalid_construction():
    """Verify negative ID or non-positive capacity are rejected."""
    with pytest.raises(BlockInvariantError, match="negative"):
        KVBlock(block_id=-1, capacity=16)

    with pytest.raises(BlockInvariantError, match="positive"):
        KVBlock(block_id=0, capacity=0)

    with pytest.raises(BlockInvariantError, match="positive"):
        KVBlock(block_id=0, capacity=-8)


def test_block_allocation_and_release_lifecycle():
    """Verify allocate -> append -> release cycle and state resets."""
    block = KVBlock(block_id=5, capacity=4)

    # Allocate
    block.allocate()
    assert block.is_allocated
    assert block.ref_count == 1
    assert block.num_tokens == 0

    # Cannot allocate again when already allocated
    with pytest.raises(BlockInvariantError, match="already allocated"):
        block.allocate()

    # Append tokens
    block.append_tokens(3)
    assert block.num_tokens == 3
    assert block.available_slots == 1
    assert not block.is_full

    block.append_tokens(1)
    assert block.num_tokens == 4
    assert block.is_full
    assert block.available_slots == 0

    # Exceed capacity
    with pytest.raises(BlockInvariantError, match="capacity"):
        block.append_tokens(1)

    # Release to zero
    new_refs = block.release()
    assert new_refs == 0
    assert not block.is_allocated
    assert block.num_tokens == 0
    assert block.ref_count == 0

    # Double free detection
    with pytest.raises(DoubleFreeError, match="already freed"):
        block.release()


def test_block_reference_counting_and_shared_ownership():
    """Verify retain and multi-reference lifecycle (prefix sharing)."""
    block = KVBlock(block_id=10, capacity=16)
    block.allocate()
    block.append_tokens(16)
    assert block.ref_count == 1

    # First shared reference (Request B shares block with Request A)
    ref2 = block.retain()
    assert ref2 == 2
    assert block.ref_count == 2
    assert block.is_allocated

    # Second shared reference (Request C)
    ref3 = block.retain()
    assert ref3 == 3
    assert block.ref_count == 3

    # Release Request A: block must remain allocated with 2 references
    rem1 = block.release()
    assert rem1 == 2
    assert block.is_allocated
    assert block.ref_count == 2
    assert block.num_tokens == 16  # data preserved

    # Release Request B: block must remain allocated with 1 reference
    rem2 = block.release()
    assert rem2 == 1
    assert block.is_allocated
    assert block.ref_count == 1

    # Release Request C: final release resets block
    rem3 = block.release()
    assert rem3 == 0
    assert not block.is_allocated
    assert block.ref_count == 0
    assert block.num_tokens == 0


def test_block_cannot_retain_unallocated():
    """Cannot retain an unallocated block."""
    block = KVBlock(block_id=1, capacity=8)
    with pytest.raises(BlockInvariantError, match="unallocated"):
        block.retain()


def test_block_cannot_append_to_unallocated():
    """Cannot append tokens to an unallocated block."""
    block = KVBlock(block_id=1, capacity=8)
    with pytest.raises(BlockInvariantError, match="unallocated"):
        block.append_tokens(2)
