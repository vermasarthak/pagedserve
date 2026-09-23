"""Unit tests for FP8 KV Cache Quantization Engine."""

import torch

from pagedserve.memory.fp8_cache import FP8KVCacheEngine


def test_fp8_quantize_dequantize_e4m3():
    engine = FP8KVCacheEngine(fp8_format="e4m3")
    k = torch.randn(16, 8, 64, dtype=torch.float32)
    v = torch.randn(16, 8, 64, dtype=torch.float32)

    block = engine.quantize_block(k, v)

    assert block.k_quant.dtype == torch.int8
    assert block.v_quant.dtype == torch.int8
    assert block.k_scale > 0.0
    assert block.v_scale > 0.0

    k_deq, v_deq = engine.dequantize_block(block)

    # Low error between original and dequantized
    assert torch.allclose(k, k_deq, atol=0.05, rtol=0.05)
    assert torch.allclose(v, v_deq, atol=0.05, rtol=0.05)


def test_fp8_quantize_dequantize_e5m2():
    engine = FP8KVCacheEngine(fp8_format="e5m2")
    k = torch.randn(16, 4, 32, dtype=torch.float32) * 100.0
    v = torch.randn(16, 4, 32, dtype=torch.float32) * 100.0

    block = engine.quantize_block(k, v)
    k_deq, v_deq = engine.dequantize_block(block)

    assert torch.allclose(k, k_deq, atol=5.0, rtol=0.05)
    assert torch.allclose(v, v_deq, atol=5.0, rtol=0.05)


def test_fp8_empty_tensors():
    engine = FP8KVCacheEngine(fp8_format="e4m3")
    k = torch.empty(0, 8, 64, dtype=torch.float32)
    v = torch.empty(0, 8, 64, dtype=torch.float32)

    block = engine.quantize_block(k, v)
    assert block.k_quant.numel() == 0
    assert block.v_quant.numel() == 0
    k_deq, v_deq = engine.dequantize_block(block)
    assert k_deq.numel() == 0
