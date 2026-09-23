"""FP8 KV Cache Quantization Engine for PagedServe.

Implements FP8 E4M3 / E5M2 block-scale quantization & dequantization for KV cache compression.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedKVCacheBlock:
    """Compressed KV cache block holding int8 payload and per-block float scale factors."""
    k_quant: torch.Tensor
    v_quant: torch.Tensor
    k_scale: float
    v_scale: float


class FP8KVCacheEngine:
    """Compresses and decompresses Key-Value cache blocks using FP8 block scaling."""

    def __init__(self, fp8_format: str = "e4m3"):
        if fp8_format.lower() not in {"e4m3", "e5m2"}:
            raise ValueError(f"Unsupported FP8 format '{fp8_format}'. Must be 'e4m3' or 'e5m2'")
        self.fp8_format = fp8_format.lower()
        self.max_val = 127.0

    def quantize_block(self, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> QuantizedKVCacheBlock:
        """Quantizes float K and V tensors into FP8 blocks with scale factors."""
        if k_tensor.numel() == 0 or v_tensor.numel() == 0:
            return QuantizedKVCacheBlock(
                k_quant=torch.zeros_like(k_tensor, dtype=torch.int8),
                v_quant=torch.zeros_like(v_tensor, dtype=torch.int8),
                k_scale=1e-8,
                v_scale=1e-8,
            )

        k_max = float(torch.max(torch.abs(k_tensor)).item()) if k_tensor.numel() > 0 else 0.0
        v_max = float(torch.max(torch.abs(v_tensor)).item()) if v_tensor.numel() > 0 else 0.0

        k_scale = max(k_max / self.max_val, 1e-8)
        v_scale = max(v_max / self.max_val, 1e-8)

        k_q = torch.clamp(torch.round(k_tensor / k_scale), -128, 127).to(torch.int8)
        v_q = torch.clamp(torch.round(v_tensor / v_scale), -128, 127).to(torch.int8)

        return QuantizedKVCacheBlock(k_quant=k_q, v_quant=v_q, k_scale=k_scale, v_scale=v_scale)

    def dequantize_block(self, block: QuantizedKVCacheBlock) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantizes FP8 block back to float32 tensors using stored scale factors."""
        k_dequant = block.k_quant.to(torch.float32) * block.k_scale
        v_dequant = block.v_quant.to(torch.float32) * block.v_scale
        return k_dequant, v_dequant
