"""Physical Block Pool managing allocation, retention, and freeing of KV blocks."""

from collections import deque
from typing import List, Deque, Set

from pagedserve.memory.block import KVBlock
from pagedserve.errors import (
    KVCacheExhaustedError,
    InvalidBlockError,
    DoubleFreeError,
    BlockInvariantError,
)


class BlockPool:
    """Manages a pre-allocated pool of fixed-size physical KV cache blocks.
    
    The pool maintains an array of all blocks and a free-list deque for O(1) allocation
    and deallocation. It enforces strict reference counting invariants and prevents
    double frees, invalid block references, and silent memory leaks.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise BlockInvariantError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise BlockInvariantError(f"block_size must be positive, got {block_size}")

        self._num_blocks: int = num_blocks
        self._block_size: int = block_size

        # Pre-allocate all block metadata instances
        self._blocks: List[KVBlock] = [
            KVBlock(block_id=i, capacity=block_size)
            for i in range(num_blocks)
        ]

        # Free list: queue of available block IDs (LIFO/FIFO)
        # Using deque for efficient pops and appends
        self._free_queue: Deque[int] = deque(range(num_blocks))
        self._free_set: Set[int] = set(range(num_blocks))

        self.verify_invariants()

    @property
    def total_blocks(self) -> int:
        """Total number of physical blocks managed by this pool."""
        return self._num_blocks

    @property
    def block_size(self) -> int:
        """Token capacity per individual block."""
        return self._block_size

    @property
    def free_count(self) -> int:
        """Number of currently available unallocated blocks."""
        return len(self._free_queue)

    @property
    def used_count(self) -> int:
        """Number of blocks currently allocated and in use."""
        return self._num_blocks - len(self._free_queue)

    @property
    def utilization(self) -> float:
        """Current pool utilization ratio in [0.0, 1.0]."""
        return self.used_count / self._num_blocks

    def get_block(self, block_id: int) -> KVBlock:
        """Retrieve the KVBlock metadata for a given physical block ID."""
        if block_id < 0 or block_id >= self._num_blocks:
            raise InvalidBlockError(
                f"Block ID {block_id} is out of bounds [0, {self._num_blocks - 1}]"
            )
        return self._blocks[block_id]

    def allocate(self) -> KVBlock:
        """Allocate a single physical block from the free list.
        
        Raises:
            KVCacheExhaustedError: if no free blocks are available.
        """
        if not self._free_queue:
            raise KVCacheExhaustedError(
                requested=1, available=0, total=self._num_blocks
            )

        block_id = self._free_queue.popleft()
        self._free_set.remove(block_id)

        block = self._blocks[block_id]
        block.allocate()
        return block

    def allocate_many(self, count: int) -> List[KVBlock]:
        """Atomically allocate multiple physical blocks.
        
        Guarantees that either all `count` blocks are allocated, or none are.
        
        Raises:
            ValueError: if count < 0.
            KVCacheExhaustedError: if available free blocks < count.
        """
        if count < 0:
            raise ValueError(f"Cannot allocate negative block count: {count}")
        if count == 0:
            return []

        if len(self._free_queue) < count:
            raise KVCacheExhaustedError(
                requested=count,
                available=len(self._free_queue),
                total=self._num_blocks,
            )

        allocated: List[KVBlock] = []
        for _ in range(count):
            block_id = self._free_queue.popleft()
            self._free_set.remove(block_id)
            block = self._blocks[block_id]
            block.allocate()
            allocated.append(block)

        return allocated

    def retain(self, block_id: int) -> int:
        """Increment the reference count for a block when shared by another sequence."""
        block = self.get_block(block_id)
        if block_id in self._free_set:
            raise BlockInvariantError(
                f"Cannot retain block {block_id}: it is currently in the free list"
            )
        return block.retain()

    def release(self, block_id: int) -> int:
        """Release one reference to a block.
        
        If the reference count reaches 0, the block is reset and returned to the free list.
        
        Raises:
            InvalidBlockError: if block_id is invalid.
            DoubleFreeError: if the block is already free.
        """
        if block_id < 0 or block_id >= self._num_blocks:
            raise InvalidBlockError(
                f"Cannot release invalid block ID {block_id}"
            )

        if block_id in self._free_set:
            raise DoubleFreeError(
                f"Double free detected: block {block_id} is already in the free pool"
            )

        block = self._blocks[block_id]
        new_ref_count = block.release()

        if new_ref_count == 0:
            self._free_queue.append(block_id)
            self._free_set.add(block_id)

        return new_ref_count

    def verify_invariants(self) -> None:
        """Exhaustively verify structural invariants of the pool."""
        if len(self._free_queue) != len(self._free_set):
            raise BlockInvariantError("Free queue and free set size mismatch (possible duplicate free block)")

        if len(self._free_queue) + self.used_count != self._num_blocks:
            raise BlockInvariantError(
                f"Pool block count mismatch: free ({len(self._free_queue)}) + used ({self.used_count}) "
                f"!= total ({self._num_blocks})"
            )

        for b in self._blocks:
            b.verify_invariants()
            if b.block_id in self._free_set:
                if b.is_allocated or b.ref_count != 0 or b.num_tokens != 0:
                    raise BlockInvariantError(
                        f"Free list block {b.block_id} is in inconsistent state: "
                        f"allocated={b.is_allocated}, ref_count={b.ref_count}, num_tokens={b.num_tokens}"
                    )
            else:
                if not b.is_allocated or b.ref_count <= 0:
                    raise BlockInvariantError(
                        f"Allocated block {b.block_id} is in inconsistent state: "
                        f"allocated={b.is_allocated}, ref_count={b.ref_count}"
                    )
