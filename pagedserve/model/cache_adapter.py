"""Adapter translating between Hugging Face model cache structures and canonical PagedServe cache tensors.

Canonical Internal PagedServe Cache Shape:
  [layer_idx, seq_length, num_kv_heads, head_dim] for K and V separately
  or Tuple[Tensor, Tensor] per layer: (Key, Value)
"""

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from pagedserve.errors import PagedServeError
from pagedserve.memory.geometry import KVCacheGeometry


class CacheAdapterError(PagedServeError):
    """Raised when Hugging Face cache conversion fails."""


class CacheAdapter:
    """Translates Hugging Face cache objects (DynamicCache, tuple of tuples) to and from canonical PagedServe tensors.
    
    Canonical Internal PagedServe Representation per Layer:
        key:   [seq_length, num_kv_heads, head_dim]  (unbatched single-sequence)
        value: [seq_length, num_kv_heads, head_dim]  (unbatched single-sequence)
    """

    @staticmethod
    def hf_cache_to_canonical(
        past_key_values: Any, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract canonical single-sequence Key and Value tensors for a specified layer from HF cache.
        
        Supports:
        - `DynamicCache` (standard in modern Transformers)
        - Legacy tuple of `(key, value)` tuples
        
        Args:
            past_key_values: Hugging Face cache structure.
            layer_idx: Layer index to extract.
            
        Returns:
            Tuple of (key_tensor, value_tensor) each of shape [seq_length, num_kv_heads, head_dim].
        """
        if past_key_values is None:
            raise CacheAdapterError("past_key_values is None")

        if isinstance(past_key_values, DynamicCache):
            # DynamicCache in transformers >= 4.36 or transformers >= 4.45
            if hasattr(past_key_values, "key_cache"):
                if layer_idx < 0 or layer_idx >= len(past_key_values.key_cache):
                    raise CacheAdapterError(f"Layer index {layer_idx} out of range for DynamicCache key_cache")
                k_raw = past_key_values.key_cache[layer_idx]
                v_raw = past_key_values.value_cache[layer_idx]
            elif hasattr(past_key_values, "layers"):
                if layer_idx < 0 or layer_idx >= len(past_key_values.layers):
                    raise CacheAdapterError(f"Layer index {layer_idx} out of range for DynamicCache layers")
                k_raw = past_key_values.layers[layer_idx].keys
                v_raw = past_key_values.layers[layer_idx].values
            else:
                raise CacheAdapterError(f"Unrecognized DynamicCache structure: {dir(past_key_values)}")

        elif isinstance(past_key_values, (tuple, list)):
            if layer_idx < 0 or layer_idx >= len(past_key_values):
                raise CacheAdapterError(
                    f"Layer index {layer_idx} out of range for tuple cache (len={len(past_key_values)})"
                )
            layer_pair = past_key_values[layer_idx]
            k_raw, v_raw = layer_pair[0], layer_pair[1]
        else:
            raise CacheAdapterError(f"Unsupported HF cache type: {type(past_key_values)}")

        # Squeeze batch dim (assume batch_size=1) and transpose to canonical [seq_length, num_kv_heads, head_dim]
        # HF input shape: (1, num_kv_heads, seq_len, head_dim) -> squeeze(0) -> (num_kv_heads, seq_len, head_dim)
        # transpose(0, 1) -> (seq_len, num_kv_heads, head_dim)
        if k_raw.dim() == 4:
            k_raw = k_raw.squeeze(0)
            v_raw = v_raw.squeeze(0)

        k_canonical = k_raw.transpose(0, 1).contiguous()
        v_canonical = v_raw.transpose(0, 1).contiguous()

        return k_canonical, v_canonical

    @staticmethod
    def canonical_to_hf_cache(
        layer_kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        geometry: KVCacheGeometry,
    ) -> DynamicCache:
        """Convert a list of canonical layer (key, value) tensors back into a Hugging Face DynamicCache.
        
        Args:
            layer_kv_pairs: List of (key, value) tuples per layer, where key/value are
                            shape [seq_length, num_kv_heads, head_dim].
            geometry: Target KV cache geometry.
            
        Returns:
            Populated Hugging Face DynamicCache instance compatible with model forward calls.
        """
        cache = DynamicCache()
        for layer_idx, (k_can, v_can) in enumerate(layer_kv_pairs):
            # Input shape: [seq_length, num_kv_heads, head_dim]
            # Desired HF shape: [batch_size=1, num_kv_heads, seq_length, head_dim]
            k_hf = k_can.transpose(0, 1).unsqueeze(0).contiguous()
            v_hf = v_can.transpose(0, 1).unsqueeze(0).contiguous()
            cache.update(k_hf, v_hf, layer_idx)

        return cache
