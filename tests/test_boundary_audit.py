"""Pre-flight audit test verifying exact KV block boundary semantics for block_size=16.

Specifically verifies:
- Lengths 1, 15, 16, 17, 31, 32, 33
- Expected physical block counts:
  1  -> 1 block, final block occupancy 1
  15 -> 1 block, final block occupancy 15
  16 -> 1 block, final block occupancy 16
  17 -> 2 blocks, final block occupancy 1
  31 -> 2 blocks, final block occupancy 15
  32 -> 2 blocks, final block occupancy 16
  33 -> 3 blocks, final block occupancy 1
- Decode-step sequence expansion across 16 -> 17 and 32 -> 33
- Full return of all physical blocks to the free pool on release.
"""

import pytest

from pagedserve.memory.kv_cache import KVCacheManager


@pytest.mark.parametrize(
    "length,expected_blocks,expected_final_occupancy",
    [
        (1, 1, 1),
        (15, 1, 15),
        (16, 1, 16),
        (17, 2, 1),
        (31, 2, 15),
        (32, 2, 16),
        (33, 3, 1),
    ],
)
def test_kv_cache_prompt_boundary_semantics(
    length: int, expected_blocks: int, expected_final_occupancy: int
):
    """Verify prompt block allocation at critical boundary lengths."""
    block_size = 16
    mgr = KVCacheManager(num_blocks=10, block_size=block_size)
    req_id = f"req-{length}"

    table = mgr.allocate_for_prompt(req_id, prompt_len=length)

    # 1. Verify physical block count
    assert table.num_blocks == expected_blocks
    assert mgr.used_blocks == expected_blocks
    assert mgr.free_blocks == 10 - expected_blocks

    # 2. Verify num_tokens occupancy for the final block
    final_block_id = table.physical_block_for(expected_blocks - 1)
    final_block = mgr._pool.get_block(final_block_id)
    assert final_block.num_tokens == expected_final_occupancy

    # 3. Verify prior blocks (if any) are 100% full
    for i in range(expected_blocks - 1):
        prior_block_id = table.physical_block_for(i)
        prior_block = mgr._pool.get_block(prior_block_id)
        assert prior_block.num_tokens == block_size
        assert prior_block.is_full

    # 4. Verify release returns every single physical block to the free pool
    mgr.release_request(req_id)
    mgr.verify_clean()


def test_kv_cache_decode_boundary_crossing_16_to_17_and_32_to_33():
    """Verify step-by-step decode growth crossing 16 -> 17 and 32 -> 33."""
    block_size = 16
    mgr = KVCacheManager(num_blocks=10, block_size=block_size)
    req_id = "req-decode-boundary"

    # Start with prompt_len = 15 (1 block, 15 tokens)
    table = mgr.allocate_for_prompt(req_id, prompt_len=15)
    assert table.num_blocks == 1
    b0_id = table.physical_block_for(0)
    assert mgr._pool.get_block(b0_id).num_tokens == 15

    # Decode step: length 16 (fits in block 0)
    new_blk = mgr.append_token(req_id, current_total_len=16)
    assert new_blk is None
    assert table.num_blocks == 1
    assert mgr._pool.get_block(b0_id).num_tokens == 16
    assert mgr._pool.get_block(b0_id).is_full

    # Decode step: length 17 -> Boundary crossed! Allocates Block 1
    new_blk_1 = mgr.append_token(req_id, current_total_len=17)
    assert new_blk_1 is not None
    assert table.num_blocks == 2
    assert mgr.used_blocks == 2
    b1_id = table.physical_block_for(1)
    assert b1_id == new_blk_1
    assert mgr._pool.get_block(b1_id).num_tokens == 1

    # Fast forward decode up to length 31
    for length in range(18, 32):
        res = mgr.append_token(req_id, current_total_len=length)
        assert res is None
    assert table.num_blocks == 2
    assert mgr._pool.get_block(b1_id).num_tokens == 15

    # Decode step: length 32 (fills block 1)
    res = mgr.append_token(req_id, current_total_len=32)
    assert res is None
    assert table.num_blocks == 2
    assert mgr._pool.get_block(b1_id).num_tokens == 16
    assert mgr._pool.get_block(b1_id).is_full

    # Decode step: length 33 -> Boundary crossed! Allocates Block 2
    new_blk_2 = mgr.append_token(req_id, current_total_len=33)
    assert new_blk_2 is not None
    assert table.num_blocks == 3
    assert mgr.used_blocks == 3
    b2_id = table.physical_block_for(2)
    assert b2_id == new_blk_2
    assert mgr._pool.get_block(b2_id).num_tokens == 1

    # Cleanup and verify zero leaks
    mgr.release_request(req_id)
    mgr.verify_clean()
