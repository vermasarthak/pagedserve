"""Exhaustive tests for BlockTable logical-to-physical block mapping."""

import pytest
from pagedserve.memory.block_table import BlockTable
from pagedserve.errors import InvalidBlockError


def test_block_table_initial_state():
    """Verify empty block table initialization."""
    table = BlockTable(request_id="req-1")
    assert table.request_id == "req-1"
    assert table.num_blocks == 0
    assert len(table) == 0
    assert table.blocks == []
    assert table.logical_indices == []


def test_block_table_non_contiguous_mapping():
    """Verify non-contiguous physical block mapping and logical index lookup."""
    table = BlockTable(request_id="req-2")

    # Non-contiguous block assignment: logical 0 -> 17, logical 1 -> 3, logical 2 -> 42
    table.append_block(17)
    table.append_block(3)
    table.append_block(42)

    assert table.num_blocks == 3
    assert len(table) == 3
    assert table.blocks == [17, 3, 42]
    assert table.logical_indices == [0, 1, 2]

    # Logical to physical lookup
    assert table.physical_block_for(0) == 17
    assert table.physical_block_for(1) == 3
    assert table.physical_block_for(2) == 42

    # Negative block ID rejected
    with pytest.raises(InvalidBlockError):
        table.append_block(-5)

    # Out of range logical indices
    with pytest.raises(IndexError):
        table.physical_block_for(-1)
    with pytest.raises(IndexError):
        table.physical_block_for(3)


def test_block_table_token_offset_resolution():
    """Verify mapping arbitrary token positions to physical blocks and offsets."""
    table = BlockTable(request_id="req-tokens")
    # block_size = 16
    # Logical 0 -> block 10 (tokens 0..15)
    # Logical 1 -> block 25 (tokens 16..31)
    # Logical 2 -> block 4  (tokens 32..47)
    table.append_block(10)
    table.append_block(25)
    table.append_block(4)

    block_size = 16

    # Token 0 -> Block 10, offset 0
    assert table.get_physical_block_and_offset(0, block_size) == (10, 0)
    # Token 15 -> Block 10, offset 15
    assert table.get_physical_block_and_offset(15, block_size) == (10, 15)

    # Token 16 -> Block 25, offset 0 (boundary crossing)
    assert table.get_physical_block_and_offset(16, block_size) == (25, 0)
    # Token 20 -> Block 25, offset 4
    assert table.get_physical_block_and_offset(20, block_size) == (25, 4)

    # Token 32 -> Block 4, offset 0
    assert table.get_physical_block_and_offset(32, block_size) == (4, 0)
    # Token 47 -> Block 4, offset 15
    assert table.get_physical_block_and_offset(47, block_size) == (4, 15)

    # Token 48 is beyond allocated blocks (IndexError)
    with pytest.raises(IndexError):
        table.get_physical_block_and_offset(48, block_size)

    # Negative token index
    with pytest.raises(ValueError):
        table.get_physical_block_and_offset(-1, block_size)


def test_block_table_contains_and_clear():
    """Verify membership check and clear mechanics."""
    table = BlockTable(request_id="req-clear")
    table.append_block(12)
    table.append_block(19)

    assert table.contains_block(12)
    assert table.contains_block(19)
    assert not table.contains_block(99)

    cleared = table.clear()
    assert cleared == [12, 19]
    assert table.num_blocks == 0
    assert table.blocks == []
