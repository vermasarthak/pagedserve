"""Logical-to-Physical Block Table mapping for individual inference requests."""


from pagedserve.errors import InvalidBlockError


class BlockTable:
    """Maintains the logical-to-physical block mapping for a single inference request.
    
    Why Logical/Physical Indirection Matters:
    Traditional inference systems pre-allocate a contiguous maximum-length KV buffer
    for every request (e.g. 2048 or 4096 tokens). Because actual generation lengths vary
    widely and cannot be known in advance, this causes severe external and internal
    fragmentation (60-80% wasted KV cache memory).
    
    By decoupling the logical sequence of tokens from physical memory, PagedServe
    allows physical blocks to be scattered non-contiguously throughout GPU/CPU memory.
    Blocks are allocated strictly on demand as the sequence expands, eliminating external
    fragmentation and bounding internal fragmentation to at most one partially filled
    block per request.
    """

    def __init__(self, request_id: str):
        self._request_id: str = request_id
        self._physical_blocks: list[int] = []

    @property
    def request_id(self) -> str:
        """Identifier of the request owning this block table."""
        return self._request_id

    @property
    def num_blocks(self) -> int:
        """Total number of physical blocks currently mapped in this table."""
        return len(self._physical_blocks)

    @property
    def blocks(self) -> list[int]:
        """A copy of the ordered physical block IDs mapped to this sequence."""
        return list(self._physical_blocks)

    @property
    def logical_indices(self) -> list[int]:
        """Logical block indices [0, 1, ..., num_blocks - 1]."""
        return list(range(len(self._physical_blocks)))

    def append_block(self, physical_block_id: int) -> None:
        """Append a new physical block to the end of the logical sequence."""
        if physical_block_id < 0:
            raise InvalidBlockError(f"Cannot append negative block ID: {physical_block_id}")
        self._physical_blocks.append(physical_block_id)

    def physical_block_for(self, logical_index: int) -> int:
        """Translate a logical block index (0, 1, ...) to its corresponding physical block ID.
        
        Raises:
            IndexError: if logical_index is out of range.
        """
        if logical_index < 0 or logical_index >= len(self._physical_blocks):
            raise IndexError(
                f"Logical block index {logical_index} out of range for request '{self._request_id}' "
                f"(allocated logical blocks: 0..{len(self._physical_blocks) - 1})"
            )
        return self._physical_blocks[logical_index]

    def get_physical_block_and_offset(
        self, token_index: int, block_size: int
    ) -> tuple[int, int]:
        """Given a zero-indexed token position in the sequence, return (physical_block_id, offset_in_block).
        
        Args:
            token_index: 0-indexed position of the token in the sequence.
            block_size: number of tokens stored per physical block.
            
        Returns:
            Tuple of (physical_block_id, offset_in_block).
        """
        if token_index < 0:
            raise ValueError(f"token_index cannot be negative: {token_index}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive: {block_size}")

        logical_block_idx = token_index // block_size
        offset = token_index % block_size

        physical_block_id = self.physical_block_for(logical_block_idx)
        return physical_block_id, offset

    def contains_block(self, physical_block_id: int) -> bool:
        """Check if this block table currently maps the specified physical block ID."""
        return physical_block_id in self._physical_blocks

    def clear(self) -> list[int]:
        """Clear all block mappings and return the physical block IDs that were mapped.
        
        The returned block IDs must subsequently be released in the BlockPool by the caller.
        """
        released = list(self._physical_blocks)
        self._physical_blocks.clear()
        return released

    def __len__(self) -> int:
        return self.num_blocks

    def __repr__(self) -> str:
        return f"BlockTable(request_id='{self._request_id}', blocks={self._physical_blocks})"
