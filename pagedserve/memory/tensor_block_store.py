"""Physical block-backed KV tensor store allocating unified Key and Value storage."""

from typing import Tuple
import torch

from pagedserve.memory.geometry import KVCacheGeometry
from pagedserve.errors import InvalidBlockError, PagedServeError


class PhysicalBlockStoreError(PagedServeError):
    """Raised on invalid physical block store operations or tensor mismatch."""
    pass


class TensorBlockStore:
    """Manages pre-allocated physical PyTorch tensors for Key and Value block storage.
    
    Layout Rationale & Dimensions:
    Shape for K & V tensors:
        [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        
    Dimensions & Contiguity Rationale:
    1. Layer dimension (0): indexing by layer index yields contiguous multi-block memory per layer.
    2. Block dimension (1): physical block IDs [0..num_blocks-1] index directly into dimension 1.
    3. Token offset (2), KV heads (3), Head Dim (4): contiguous layout per block allows vectorized
       token scatter/gather, batch slicing, and direct head operations without per-element copies.
       
    Memory Allocation:
    Storage is allocated ONCE at initialization on the specified target device and dtype.
    Avoids dynamic tensor allocation during decode steps.
    """

    def __init__(self, geometry: KVCacheGeometry, num_blocks: int):
        if num_blocks <= 0:
            raise PhysicalBlockStoreError(f"num_blocks must be positive, got {num_blocks}")

        self._geometry: KVCacheGeometry = geometry
        self._num_blocks: int = num_blocks

        # Pre-allocate single contiguous tensor buffers for Key and Value storage across all blocks & layers.
        # Shape: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        shape = (
            geometry.num_layers,
            num_blocks,
            geometry.block_size,
            geometry.num_kv_heads,
            geometry.head_dim,
        )

        # Allocate once without unneeded zero-fill overhead where uninitialized memory is fine,
        # but use empty tensor allocation on target device.
        self._k_store: torch.Tensor = torch.empty(
            shape, dtype=geometry.dtype, device=geometry.device
        )
        self._v_store: torch.Tensor = torch.empty(
            shape, dtype=geometry.dtype, device=geometry.device
        )

    @property
    def geometry(self) -> KVCacheGeometry:
        """KV cache geometry configuration."""
        return self._geometry

    @property
    def num_blocks(self) -> int:
        """Total physical blocks managed by this tensor store."""
        return self._num_blocks

    @property
    def dtype(self) -> torch.dtype:
        """Tensor data type."""
        return self._geometry.dtype

    @property
    def device(self) -> torch.device:
        """PyTorch execution device."""
        return self._geometry.device

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        """Physical shape of K and V storage buffers [layers, blocks, block_size, kv_heads, head_dim]."""
        return self._k_store.shape

    @property
    def bytes_per_block(self) -> int:
        """Memory footprint in bytes for a single physical block across all layers."""
        return self._geometry.bytes_per_block

    @property
    def total_bytes(self) -> int:
        """Total allocated physical memory footprint in bytes across K & V stores."""
        return self.bytes_per_block * self._num_blocks

    def _validate_indices(self, layer_idx: int, physical_block_id: int, block_offset: Optional[int] = None) -> None:
        if layer_idx < 0 or layer_idx >= self._geometry.num_layers:
            raise PhysicalBlockStoreError(
                f"Layer index {layer_idx} out of bounds [0, {self._geometry.num_layers - 1}]"
            )
        if physical_block_id < 0 or physical_block_id >= self._num_blocks:
            raise InvalidBlockError(
                f"Physical block ID {physical_block_id} out of bounds [0, {self._num_blocks - 1}]"
            )
        if block_offset is not None:
            if block_offset < 0 or block_offset >= self._geometry.block_size:
                raise PhysicalBlockStoreError(
                    f"Block offset {block_offset} out of bounds [0, {self._geometry.block_size - 1}]"
                )

    def write_token(
        self,
        layer_idx: int,
        physical_block_id: int,
        block_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write single-token Key and Value vectors into a physical block location.
        
        Args:
            layer_idx: Target transformer layer [0..num_layers-1].
            physical_block_id: Target physical block ID [0..num_blocks-1].
            block_offset: Token position within block [0..block_size-1].
            key: Key tensor of shape (num_kv_heads, head_dim).
            value: Value tensor of shape (num_kv_heads, head_dim).
        """
        self._validate_indices(layer_idx, physical_block_id, block_offset)

        if key.device.type != self.device.type or value.device.type != self.device.type:
            raise PhysicalBlockStoreError(
                f"Device mismatch: write tensors on ({key.device}, {value.device}), store on {self.device}"
            )
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise PhysicalBlockStoreError(
                f"Dtype mismatch: write tensors ({key.dtype}, {value.dtype}), store {self.dtype}"
            )

        expected_shape = (self._geometry.num_kv_heads, self._geometry.head_dim)
        if key.shape != expected_shape or value.shape != expected_shape:
            raise PhysicalBlockStoreError(
                f"Shape mismatch: expected {expected_shape}, got key {key.shape}, val {value.shape}"
            )

        self._k_store[layer_idx, physical_block_id, block_offset] = key
        self._v_store[layer_idx, physical_block_id, block_offset] = value

    def write_block(
        self,
        layer_idx: int,
        physical_block_id: int,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        num_tokens: Optional[int] = None,
    ) -> None:
        """Write a full or partial block of Key and Value tensors.
        
        Args:
            layer_idx: Target layer index.
            physical_block_id: Target physical block ID.
            key_block: Key tensor of shape (num_tokens, num_kv_heads, head_dim).
            value_block: Value tensor of shape (num_tokens, num_kv_heads, head_dim).
            num_tokens: Optional explicit token count <= block_size.
        """
        self._validate_indices(layer_idx, physical_block_id)

        tokens_to_write = num_tokens if num_tokens is not None else key_block.shape[0]
        if tokens_to_write <= 0 or tokens_to_write > self._geometry.block_size:
            raise PhysicalBlockStoreError(
                f"Invalid num_tokens to write: {tokens_to_write} (block_size={self._geometry.block_size})"
            )

        expected_suffix_shape = (self._geometry.num_kv_heads, self._geometry.head_dim)
        if key_block.shape[1:] != expected_suffix_shape or value_block.shape[1:] != expected_suffix_shape:
            raise PhysicalBlockStoreError(
                f"Shape mismatch: expected suffix {expected_suffix_shape}, got key {key_block.shape}, val {value_block.shape}"
            )
        if key_block.device.type != self.device.type or value_block.device.type != self.device.type:
            raise PhysicalBlockStoreError(
                f"Device mismatch: write tensors on ({key_block.device}, {value_block.device}), store on {self.device}"
            )

        self._k_store[layer_idx, physical_block_id, :tokens_to_write] = key_block[:tokens_to_write]
        self._v_store[layer_idx, physical_block_id, :tokens_to_write] = value_block[:tokens_to_write]

    def read_token(
        self, layer_idx: int, physical_block_id: int, block_offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read single-token Key and Value tensors from a physical block position."""
        self._validate_indices(layer_idx, physical_block_id, block_offset)
        k = self._k_store[layer_idx, physical_block_id, block_offset]
        v = self._v_store[layer_idx, physical_block_id, block_offset]
        return k, v

    def read_block(
        self, layer_idx: int, physical_block_id: int, num_tokens: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a full or partial block of Key and Value tensors."""
        self._validate_indices(layer_idx, physical_block_id)
        n = num_tokens if num_tokens is not None else self._geometry.block_size
        k = self._k_store[layer_idx, physical_block_id, :n]
        v = self._v_store[layer_idx, physical_block_id, :n]
        return k, v
