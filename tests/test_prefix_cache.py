"""Exhaustive tests for PrefixCache: hashing, reference ownership, LRU eviction, and zero leaks."""

import pytest
from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.memory.prefix_cache import PrefixCache, compute_prefix_block_hash


def test_prefix_hash_determinism_and_chaining():
    """Verify hash is deterministic and dependent on full preceding history (chaining)."""
    block_tokens = [10, 20, 30, 40]

    # Deterministic across calls
    h1 = compute_prefix_block_hash(block_tokens, prev_hash="")
    h2 = compute_prefix_block_hash(block_tokens, prev_hash="")
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64  # SHA-256 hex string

    # Chaining: same tokens with different previous hash yield different hashes!
    h_chain_a = compute_prefix_block_hash(block_tokens, prev_hash="prefix_a")
    h_chain_b = compute_prefix_block_hash(block_tokens, prev_hash="prefix_b")
    assert h_chain_a != h_chain_b
    assert h_chain_a != h1


def test_prefix_cache_identical_first_block_reuse():
    """Verify identical full first block is reused by subsequent request."""
    # block_size = 4, pool of 6 blocks
    mgr = KVCacheManager(num_blocks=6, block_size=4, enable_prefix_caching=True)

    # Request 1: prompt [1, 2, 3, 4] -> exactly 1 full block
    t1 = mgr.allocate_for_prompt_tokens("req-1", [1, 2, 3, 4])
    b1_id = t1.physical_block_for(0)
    assert mgr.used_blocks == 1
    # Block has ref_count = 2 (1 owned by req-1, 1 owned by prefix_cache)
    assert mgr._pool.get_block(b1_id).ref_count == 2
    assert mgr.prefix_cache.size == 1
    assert mgr.prefix_cache.hits == 0
    assert mgr.prefix_cache.misses == 1

    # Request 2: identical prompt [1, 2, 3, 4]
    t2 = mgr.allocate_for_prompt_tokens("req-2", [1, 2, 3, 4])
    b2_id = t2.physical_block_for(0)
    assert b2_id == b1_id  # REUSED SAME PHYSICAL BLOCK!
    assert mgr.used_blocks == 1  # No additional physical blocks used
    # Ref count now 3 (req-1, req-2, prefix_cache)
    assert mgr._pool.get_block(b1_id).ref_count == 3
    assert mgr.prefix_cache.hits == 1
    assert mgr.prefix_cache.misses == 1
    assert mgr.prefix_cache.hit_rate == 0.5

    # Release req-1: block remains allocated for req-2 and cache
    mgr.release_request("req-1")
    assert mgr.used_blocks == 1
    assert mgr._pool.get_block(b1_id).ref_count == 2

    # Release req-2: block remains in cache with ref_count = 1
    mgr.release_request("req-2")
    assert mgr.used_blocks == 1
    assert mgr._pool.get_block(b1_id).ref_count == 1

    # Clear prefix cache -> fully returned to pool
    mgr.clear_prefix_cache()
    mgr.verify_clean()


def test_prefix_cache_identical_two_block_prefix_reuse():
    """Verify multi-block identical prefix shares all full blocks."""
    mgr = KVCacheManager(num_blocks=6, block_size=4, enable_prefix_caching=True)

    # 2 full blocks (8 tokens)
    prefix_tokens = [10, 11, 12, 13, 20, 21, 22, 23]
    t1 = mgr.allocate_for_prompt_tokens("req-1", prefix_tokens)
    assert t1.num_blocks == 2
    b0_id = t1.physical_block_for(0)
    b1_id = t1.physical_block_for(1)
    assert mgr.used_blocks == 2
    assert mgr.prefix_cache.size == 2

    # Request 2 with identical prefix
    t2 = mgr.allocate_for_prompt_tokens("req-2", prefix_tokens)
    assert t2.num_blocks == 2
    assert t2.physical_block_for(0) == b0_id
    assert t2.physical_block_for(1) == b1_id
    assert mgr.used_blocks == 2  # Zero additional blocks used!
    assert mgr.prefix_cache.hits == 2

    mgr.release_request("req-1")
    mgr.release_request("req-2")
    mgr.verify_clean(clear_prefix=True)


def test_prefix_cache_first_block_same_second_different():
    """Verify first block is reused, but second distinct block is allocated separately."""
    mgr = KVCacheManager(num_blocks=6, block_size=4, enable_prefix_caching=True)

    # Shared system prompt (block 0: [1, 2, 3, 4])
    # Request 1 continues with [5, 6, 7, 8]
    t1 = mgr.allocate_for_prompt_tokens("req-1", [1, 2, 3, 4, 5, 6, 7, 8])
    assert t1.num_blocks == 2
    b0_req1 = t1.physical_block_for(0)
    b1_req1 = t1.physical_block_for(1)
    assert mgr.used_blocks == 2

    # Request 2 continues with [9, 9, 9, 9]
    t2 = mgr.allocate_for_prompt_tokens("req-2", [1, 2, 3, 4, 9, 9, 9, 9])
    assert t2.num_blocks == 2
    b0_req2 = t2.physical_block_for(0)
    b1_req2 = t2.physical_block_for(1)

    # First block is shared!
    assert b0_req2 == b0_req1
    # Second block is DIFFERENT!
    assert b1_req2 != b1_req1
    assert mgr.used_blocks == 3

    mgr.release_request("req-1")
    mgr.release_request("req-2")
    mgr.verify_clean(clear_prefix=True)


def test_prefix_cache_different_first_block_no_reuse():
    """Verify completely different prompts do not reuse any blocks."""
    mgr = KVCacheManager(num_blocks=6, block_size=4, enable_prefix_caching=True)

    t1 = mgr.allocate_for_prompt_tokens("req-1", [1, 2, 3, 4])
    t2 = mgr.allocate_for_prompt_tokens("req-2", [5, 6, 7, 8])

    assert t1.physical_block_for(0) != t2.physical_block_for(0)
    assert mgr.used_blocks == 2
    assert mgr.prefix_cache.hits == 0
    assert mgr.prefix_cache.misses == 2

    mgr.release_request("req-1")
    mgr.release_request("req-2")
    mgr.verify_clean(clear_prefix=True)


def test_prefix_cache_partial_trailing_blocks_not_cached():
    """Verify partial trailing blocks are not inserted into the PrefixCache."""
    mgr = KVCacheManager(num_blocks=6, block_size=4, enable_prefix_caching=True)

    # 6 tokens: 1 full block (4 tokens) + 1 partial block (2 tokens)
    t1 = mgr.allocate_for_prompt_tokens("req-1", [1, 2, 3, 4, 5, 6])
    assert t1.num_blocks == 2
    # Prefix cache size must be 1 (only the full block is cached)
    assert mgr.prefix_cache.size == 1

    # Request 2 has same first 4 tokens, but different partial tokens [7, 8]
    t2 = mgr.allocate_for_prompt_tokens("req-2", [1, 2, 3, 4, 7, 8])
    assert t2.num_blocks == 2
    # First block shared
    assert t2.physical_block_for(0) == t1.physical_block_for(0)
    # Second block not shared
    assert t2.physical_block_for(1) != t1.physical_block_for(1)
    assert mgr.prefix_cache.size == 1

    mgr.release_request("req-1")
    mgr.release_request("req-2")
    mgr.verify_clean(clear_prefix=True)


def test_prefix_cache_lru_eviction():
    """Verify bounded cache capacity and LRU eviction mechanics."""
    pool = BlockPool(num_blocks=6, block_size=4)
    # PrefixCache capacity = 2 blocks
    cache = PrefixCache(pool=pool, max_cached_blocks=2)

    # Allocate and insert Block 0
    b0 = pool.allocate()
    cache.insert("hash-0", b0.block_id)
    assert b0.ref_count == 2  # allocated + cache
    assert cache.size == 1

    # Allocate and insert Block 1
    b1 = pool.allocate()
    cache.insert("hash-1", b1.block_id)
    assert cache.size == 2

    # Touch Block 0 to make Block 1 least recently used
    assert cache.lookup("hash-0") == b0.block_id

    # Allocate and insert Block 2 -> exceeds capacity (2), evicts LRU (hash-1)
    b2 = pool.allocate()
    cache.insert("hash-2", b2.block_id)
    assert cache.size == 2
    assert cache.contains("hash-0")
    assert cache.contains("hash-2")
    assert not cache.contains("hash-1")

    # Block 1 had its cache reference released (ref_count drops to 1)
    assert b1.ref_count == 1
    pool.release(b1.block_id)  # release initial allocation
    assert b1.ref_count == 0   # returned to free pool

    # Clear cache and release remaining
    cache.clear()
    pool.release(b0.block_id)
    pool.release(b2.block_id)
    assert pool.free_count == 6
    pool.verify_invariants()
