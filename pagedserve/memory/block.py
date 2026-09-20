"""Fixed-size Key-Value Cache Block metadata abstraction."""

from dataclasses import dataclass

from pagedserve.errors import BlockInvariantError, DoubleFreeError


@dataclass
class KVBlock:
    """Represents the metadata for a single fixed-size physical KV cache block.
    
    A physical block holds KV cache activations for up to `capacity` tokens.
    Logical sequences map their tokens into one or more physical blocks.
    Blocks can be shared across multiple requests (e.g. for prefix caching),
    tracked via `ref_count`.
    """

    block_id: int
    capacity: int
    num_tokens: int = 0
    is_allocated: bool = False
    ref_count: int = 0

    def __post_init__(self) -> None:
        if self.block_id < 0:
            raise BlockInvariantError(f"block_id cannot be negative: {self.block_id}")
        if self.capacity <= 0:
            raise BlockInvariantError(f"block capacity must be positive: {self.capacity}")
        self.verify_invariants()

    def verify_invariants(self) -> None:
        """Verify internal consistency and reference counting invariants."""
        if self.ref_count < 0:
            raise BlockInvariantError(
                f"Block {self.block_id} invariant violated: ref_count ({self.ref_count}) < 0"
            )

        if self.num_tokens < 0 or self.num_tokens > self.capacity:
            raise BlockInvariantError(
                f"Block {self.block_id} invariant violated: num_tokens ({self.num_tokens}) "
                f"out of range [0, {self.capacity}]"
            )

        if not self.is_allocated and self.ref_count != 0:
            raise BlockInvariantError(
                f"Block {self.block_id} invariant violated: unallocated block has ref_count={self.ref_count} != 0"
            )

        if self.is_allocated and self.ref_count <= 0:
            raise BlockInvariantError(
                f"Block {self.block_id} invariant violated: allocated block has ref_count={self.ref_count} <= 0"
            )

        if not self.is_allocated and self.num_tokens != 0:
            raise BlockInvariantError(
                f"Block {self.block_id} invariant violated: unallocated block has num_tokens={self.num_tokens} != 0"
            )

    @property
    def is_full(self) -> bool:
        """True if the block contains exactly `capacity` tokens."""
        return self.num_tokens == self.capacity

    @property
    def is_empty(self) -> bool:
        """True if no tokens have been written to this block yet."""
        return self.num_tokens == 0

    @property
    def available_slots(self) -> int:
        """Number of remaining token positions available in this block."""
        return self.capacity - self.num_tokens

    def allocate(self) -> None:
        """Transition an unallocated block to allocated with initial reference count 1."""
        if self.is_allocated or self.ref_count > 0:
            raise BlockInvariantError(
                f"Cannot allocate block {self.block_id}: already allocated (ref_count={self.ref_count})"
            )
        self.is_allocated = True
        self.ref_count = 1
        self.num_tokens = 0
        self.verify_invariants()

    def retain(self) -> int:
        """Increment the reference count when an additional request shares this block (e.g. prefix sharing)."""
        if not self.is_allocated:
            raise BlockInvariantError(
                f"Cannot retain unallocated block {self.block_id}"
            )
        self.ref_count += 1
        self.verify_invariants()
        return self.ref_count

    def release(self) -> int:
        """Decrement the reference count.
        
        When reference count reaches 0, the block is reset to unallocated and its
        token count is cleared so it can safely return to the free pool.
        """
        if not self.is_allocated or self.ref_count <= 0:
            raise DoubleFreeError(
                f"Cannot release block {self.block_id}: already freed or unallocated "
                f"(is_allocated={self.is_allocated}, ref_count={self.ref_count})"
            )

        self.ref_count -= 1
        if self.ref_count == 0:
            self.is_allocated = False
            self.num_tokens = 0

        self.verify_invariants()
        return self.ref_count

    def append_tokens(self, count: int) -> None:
        """Record newly added tokens to this block."""
        if not self.is_allocated:
            raise BlockInvariantError(
                f"Cannot append tokens to unallocated block {self.block_id}"
            )
        if count <= 0:
            raise ValueError(f"Token count to append must be positive, got {count}")
        if self.num_tokens + count > self.capacity:
            raise BlockInvariantError(
                f"Cannot append {count} tokens to block {self.block_id}: "
                f"capacity={self.capacity}, current={self.num_tokens}"
            )
        self.num_tokens += count
        self.verify_invariants()
