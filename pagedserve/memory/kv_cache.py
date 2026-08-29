"""High-level Key-Value Cache Manager orchestrating BlockPool and Request BlockTables."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.prefix_cache import PrefixCache, compute_prefix_block_hash
from pagedserve.errors import KVCacheExhaustedError, InvalidRequestError


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of KV cache memory statistics."""

    total_blocks: int
    used_blocks: int
    free_blocks: int
    utilization: float
    num_active_requests: int
    prefix_cache_size: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_cache_hit_rate: float = 0.0


class KVCacheManager:
    """Manages the lifecycle of KV cache memory across all active inference requests.
    
    Coordinates the physical BlockPool, per-request BlockTables, and PrefixCache:
    - Allocates initial physical blocks needed for prompt prefill.
    - Deduplicates identical prompt prefixes across requests via PrefixCache.
    - Dynamically expands block allocations during auto-regressive decode steps.
    - Recovers and recycles physical blocks when requests finish or are cancelled.
    - Exposes real-time memory utilization telemetry without needing GPU execution.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = False,
        max_prefix_cached_blocks: int = 128,
    ):
        self._block_size: int = block_size
        self._pool: BlockPool = BlockPool(num_blocks=num_blocks, block_size=block_size)
        self._tables: Dict[str, BlockTable] = {}
        self._enable_prefix_caching: bool = enable_prefix_caching
        self._prefix_cache: Optional[PrefixCache] = (
            PrefixCache(pool=self._pool, max_cached_blocks=max_prefix_cached_blocks)
            if enable_prefix_caching
            else None
        )

    @property
    def prefix_cache(self) -> Optional[PrefixCache]:
        """The attached PrefixCache instance, if enabled."""
        return self._prefix_cache

    @property
    def block_size(self) -> int:
        """Capacity in tokens of each physical block."""
        return self._block_size

    @property
    def total_blocks(self) -> int:
        """Total number of physical blocks."""
        return self._pool.total_blocks

    @property
    def used_blocks(self) -> int:
        """Number of currently allocated physical blocks."""
        return self._pool.used_count

    @property
    def free_blocks(self) -> int:
        """Number of free physical blocks remaining."""
        return self._pool.free_count

    @property
    def utilization(self) -> float:
        """Fraction of total physical blocks in use [0.0, 1.0]."""
        return self._pool.utilization

    @property
    def num_active_requests(self) -> int:
        """Number of requests with active block allocations."""
        return len(self._tables)

    @property
    def blocks_by_request(self) -> Dict[str, List[int]]:
        """Mapping from request_id to list of allocated physical block IDs."""
        return {req_id: table.blocks for req_id, table in self._tables.items()}

    def get_stats(self) -> CacheStats:
        """Return an immutable snapshot of KV cache memory metrics."""
        p_size = self._prefix_cache.size if self._prefix_cache else 0
        p_hits = self._prefix_cache.hits if self._prefix_cache else 0
        p_misses = self._prefix_cache.misses if self._prefix_cache else 0
        p_rate = self._prefix_cache.hit_rate if self._prefix_cache else 0.0
        return CacheStats(
            total_blocks=self.total_blocks,
            used_blocks=self.used_blocks,
            free_blocks=self.free_blocks,
            utilization=self.utilization,
            num_active_requests=self.num_active_requests,
            prefix_cache_size=p_size,
            prefix_cache_hits=p_hits,
            prefix_cache_misses=p_misses,
            prefix_cache_hit_rate=p_rate,
        )

    def num_blocks_needed_for_tokens(self, num_tokens: int) -> int:
        """Calculate the number of blocks needed to hold a given token count."""
        if num_tokens <= 0:
            return 0
        return math.ceil(num_tokens / self._block_size)

    def can_allocate_prompt(self, prompt_len: int) -> bool:
        """Check if there are enough free blocks to allocate for a prompt of given length."""
        needed = self.num_blocks_needed_for_tokens(prompt_len)
        return self._pool.free_count >= needed

    def can_allocate_prompt_tokens(self, prompt_token_ids: Sequence[int]) -> bool:
        """Check if sufficient free blocks exist to admit a prompt, accounting for cached prefix blocks."""
        if not prompt_token_ids:
            return False
        if not self._enable_prefix_caching or not self._prefix_cache:
            return self.can_allocate_prompt(len(prompt_token_ids))

        needed_new_blocks = 0
        prev_hash = ""
        num_full_blocks = len(prompt_token_ids) // self._block_size
        for i in range(num_full_blocks):
            chunk = prompt_token_ids[i * self._block_size : (i + 1) * self._block_size]
            h = compute_prefix_block_hash(chunk, prev_hash)
            prev_hash = h
            if not self._prefix_cache.contains(h):
                needed_new_blocks += 1

        if len(prompt_token_ids) % self._block_size != 0:
            needed_new_blocks += 1

        return self._pool.free_count >= needed_new_blocks

    def can_append_token(self, request_id: str, current_total_len: int) -> bool:
        """Check if a decode step can be accommodated without causing an unhandled OOM."""
        table = self._tables.get(request_id)
        if table is None:
            raise KeyError(f"No active BlockTable for request '{request_id}'")

        current_capacity = table.num_blocks * self._block_size
        if current_total_len > current_capacity:
            return self._pool.free_count >= 1
        return True

    def allocate_for_prompt(self, request_id: str, prompt_len: int) -> BlockTable:
        """Allocate initial physical blocks for a new prompt sequence.
        
        Args:
            request_id: Unique identifier for the inference request.
            prompt_len: Length of the prompt in tokens (must be >= 1).
            
        Returns:
            The initialized BlockTable populated with physical block IDs.
            
        Raises:
            InvalidRequestError: if prompt_len <= 0 or request_id is already active.
            KVCacheExhaustedError: if insufficient free blocks in pool.
        """
        if prompt_len <= 0:
            raise InvalidRequestError(f"prompt_len must be positive, got {prompt_len}")
        if request_id in self._tables:
            raise InvalidRequestError(f"Request '{request_id}' already has active KV cache allocations")

        num_blocks_needed = self.num_blocks_needed_for_tokens(prompt_len)
        allocated_blocks = self._pool.allocate_many(num_blocks_needed)

        # Distribute token counts across allocated blocks
        tokens_remaining = prompt_len
        for block in allocated_blocks:
            tokens_in_this_block = min(self._block_size, tokens_remaining)
            block.append_tokens(tokens_in_this_block)
            tokens_remaining -= tokens_in_this_block

        table = BlockTable(request_id=request_id)
        for block in allocated_blocks:
            table.append_block(block.block_id)

        self._tables[request_id] = table
        return table

    def allocate_for_prompt_tokens(
        self, request_id: str, prompt_token_ids: Sequence[int]
    ) -> BlockTable:
        """Allocate physical blocks for a prompt, deduplicating full prefix blocks via PrefixCache if enabled.
        
        Args:
            request_id: Unique identifier for the inference request.
            prompt_token_ids: Sequence of token IDs in the prompt.
            
        Returns:
            The initialized BlockTable populated with physical block IDs.
        """
        if not prompt_token_ids:
            raise InvalidRequestError("prompt_token_ids cannot be empty")
        if request_id in self._tables:
            raise InvalidRequestError(f"Request '{request_id}' already has active KV cache allocations")

        if not self._enable_prefix_caching or not self._prefix_cache:
            return self.allocate_for_prompt(request_id, len(prompt_token_ids))

        # Atomic allocation with rollback on failure
        allocated_in_call: List[int] = []
        table = BlockTable(request_id=request_id)
        prev_hash = ""
        num_full_blocks = len(prompt_token_ids) // self._block_size

        try:
            # 1. Process full blocks (eligible for prefix caching)
            for i in range(num_full_blocks):
                chunk = prompt_token_ids[i * self._block_size : (i + 1) * self._block_size]
                h = compute_prefix_block_hash(chunk, prev_hash)
                prev_hash = h

                cached_block_id = self._prefix_cache.lookup(h)
                if cached_block_id is not None:
                    # Cache hit! Retain for this active request
                    self._pool.retain(cached_block_id)
                    allocated_in_call.append(cached_block_id)
                    table.append_block(cached_block_id)
                else:
                    # Cache miss! Allocate a new physical block
                    new_block = self._pool.allocate()
                    new_block.append_tokens(self._block_size)
                    allocated_in_call.append(new_block.block_id)
                    table.append_block(new_block.block_id)
                    # Register into prefix cache
                    self._prefix_cache.insert(h, new_block.block_id)

            # 2. Process partial trailing block (NOT cached)
            trailing_len = len(prompt_token_ids) % self._block_size
            if trailing_len > 0:
                partial_block = self._pool.allocate()
                partial_block.append_tokens(trailing_len)
                allocated_in_call.append(partial_block.block_id)
                table.append_block(partial_block.block_id)

            self._tables[request_id] = table
            return table

        except Exception:
            # Roll back any retained or allocated blocks in this call
            for block_id in reversed(allocated_in_call):
                try:
                    self._pool.release(block_id)
                except Exception:
                    pass
            raise

    def append_token(self, request_id: str, current_total_len: int) -> Optional[int]:
        """Record the generation of a new token for an existing sequence.
        
        If sequence length exceeds currently mapped block capacity, allocates
        a new physical block and appends it to the request's BlockTable.
        
        Args:
            request_id: Identifier of the request.
            current_total_len: Total length of the sequence including the new token.
            
        Returns:
            The newly allocated physical block ID if a block boundary was crossed, else None.
            
        Raises:
            KeyError: if request_id is not found.
            KVCacheExhaustedError: if a new block is required but the pool is exhausted.
        """
        table = self._tables.get(request_id)
        if table is None:
            raise KeyError(f"No active BlockTable for request '{request_id}'")

        current_capacity = table.num_blocks * self._block_size

        if current_total_len > current_capacity:
            # Boundary crossed: allocate a new physical block
            new_block = self._pool.allocate()
            new_block.append_tokens(1)
            table.append_block(new_block.block_id)
            return new_block.block_id
        else:
            # Token fits in the currently mapped last block
            last_physical_block_id = table.physical_block_for(table.num_blocks - 1)
            last_block = self._pool.get_block(last_physical_block_id)
            # Update token count for the block
            expected_in_block = ((current_total_len - 1) % self._block_size) + 1
            if expected_in_block > last_block.num_tokens:
                last_block.append_tokens(expected_in_block - last_block.num_tokens)
            return None

    def release_request(self, request_id: str) -> None:
        """Release all physical blocks allocated to the given request.
        
        Idempotent: if request_id is not in active tables, this call safely does nothing.
        """
        table = self._tables.pop(request_id, None)
        if table is None:
            return

        block_ids = table.clear()
        for block_id in block_ids:
            self._pool.release(block_id)

    def get_block_table(self, request_id: str) -> BlockTable:
        """Retrieve the BlockTable for an active request."""
        table = self._tables.get(request_id)
        if table is None:
            raise KeyError(f"No active BlockTable for request '{request_id}'")
        return table

    def clear_prefix_cache(self) -> None:
        """Clear all entries in the prefix cache and release their ownership references."""
        if self._prefix_cache is not None:
            self._prefix_cache.clear()

    def verify_clean(self, clear_prefix: bool = True) -> None:
        """Verify that all requests have been released and the pool is 100% free with no leaks."""
        if clear_prefix and self._prefix_cache is not None:
            self._prefix_cache.clear()

        if len(self._tables) > 0:
            raise AssertionError(
                f"Memory leak detected: {len(self._tables)} active request tables remaining: {list(self._tables.keys())}"
            )
        if self.used_blocks != 0 or self.free_blocks != self.total_blocks:
            raise AssertionError(
                f"Memory leak detected: used_blocks={self.used_blocks}, "
                f"free_blocks={self.free_blocks}, total_blocks={self.total_blocks}"
            )
        self._pool.verify_invariants()
