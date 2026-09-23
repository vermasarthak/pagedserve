"""Content-Addressed Prefix Caching with Chained Hashing and LRU Eviction."""

import hashlib
from collections import OrderedDict
from collections.abc import Sequence

from pagedserve.memory.block_pool import BlockPool


def compute_prefix_block_hash(token_ids: Sequence[int], prev_hash: str = "") -> str:
    """Compute a deterministic, content-addressed cryptographic hash for a block of tokens.

    Chained Hashing:
    The hash is computed over the previous block's hash concatenated with the current
    block's sequence of token IDs. This guarantees that block identity is strictly
    dependent on the entire prefix history leading up to this point, preventing false
    collisions between identical token subsequences appearing at different positions
    or under different prompt contexts.

    Args:
        token_ids: Sequence of token IDs in the current block.
        prev_hash: The cryptographic hash of the immediately preceding block (or "" for block 0).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    # Ingest predecessor hash
    hasher.update(prev_hash.encode("utf-8"))
    # Ingest 4-byte big-endian binary representation of each token ID
    for token_id in token_ids:
        hasher.update(int(token_id).to_bytes(4, byteorder="big", signed=True))
    return hasher.hexdigest()


class PrefixCache:
    """Maintains a bounded, content-addressed cache of reusable full KV blocks.

    Ownership Semantics:
    1. Active Request: Holds 1 reference to each physical block in its BlockTable.
    2. PrefixCache Entry: When a block is registered in the cache, the PrefixCache
       explicitly retains 1 reference via `pool.retain(physical_block_id)`.
    3. BlockPool: Reclaims the physical block to the free list only when `ref_count == 0`
       (i.e. neither any active request nor the PrefixCache holds a reference).
    4. Eviction: Evicting an LRU entry from PrefixCache releases ONLY the cache's
       reference (`pool.release(block_id)`). If an active request is currently using the
       block, its ref_count drops to 1 and the block remains valid for that request.
    """

    def __init__(self, pool: BlockPool, max_cached_blocks: int = 128):
        if max_cached_blocks <= 0:
            raise ValueError(f"max_cached_blocks must be positive, got {max_cached_blocks}")
        self._pool: BlockPool = pool
        self._max_cached_blocks: int = max_cached_blocks

        # OrderedDict maps prefix_hash -> physical_block_id (LRU ordered: least recent at front)
        self._cache: OrderedDict[str, int] = OrderedDict()

        # Telemetry
        self._hits: int = 0
        self._misses: int = 0

    @property
    def capacity(self) -> int:
        """Maximum number of blocks retained in prefix cache."""
        return self._max_cached_blocks

    @property
    def size(self) -> int:
        """Current number of entries in the prefix cache."""
        return len(self._cache)

    @property
    def hits(self) -> int:
        """Total number of successful prefix cache lookups."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total number of missed prefix cache lookups."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were hits in [0.0, 1.0]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def contains(self, prefix_hash: str) -> bool:
        """Check if a prefix hash is present in the cache without altering LRU order or metrics."""
        return prefix_hash in self._cache

    def lookup(self, prefix_hash: str) -> int | None:
        """Look up a cached physical block by its chained prefix hash.

        If found: marks entry as most-recently-used, increments hits, returns physical_block_id.
        If missing: increments misses, returns None.
        """
        if prefix_hash in self._cache:
            self._hits += 1
            self._cache.move_to_end(prefix_hash)
            return self._cache[prefix_hash]
        else:
            self._misses += 1
            return None

    def insert(self, prefix_hash: str, physical_block_id: int) -> None:
        """Insert a full physical block into the cache under its prefix hash.

        The PrefixCache increments the block's reference count in the pool to guarantee
        that the physical block will not be reclaimed while cached.

        If the cache exceeds max_cached_blocks, the least recently used entry is evicted.
        """
        if prefix_hash in self._cache:
            # Already cached under this hash; update LRU position
            self._cache.move_to_end(prefix_hash)
            return

        # Ensure valid physical block
        self._pool.get_block(physical_block_id)

        # Retain ownership reference for the prefix cache
        self._pool.retain(physical_block_id)
        self._cache[prefix_hash] = physical_block_id

        # Enforce bounded capacity via LRU eviction
        if len(self._cache) > self._max_cached_blocks:
            self.evict_lru()

    def evict_lru(self) -> tuple[str, int] | None:
        """Evict the least recently used cached block and release the cache's reference."""
        if not self._cache:
            return None

        evicted_hash, evicted_block_id = self._cache.popitem(last=False)
        # Release the cache's ownership reference back to the pool
        self._pool.release(evicted_block_id)
        return evicted_hash, evicted_block_id

    def clear(self) -> None:
        """Clear all entries and release all cache-held references in the pool."""
        while self._cache:
            _, block_id = self._cache.popitem(last=False)
            self._pool.release(block_id)
        self._hits = 0
        self._misses = 0
