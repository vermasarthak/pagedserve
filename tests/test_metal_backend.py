"""Comprehensive correctness and equivalence unit tests for fused Apple Metal paged attention backend."""

import math
import pytest
import torch

from pagedserve.kernels import get_backend, MetalPagedAttentionBackend, PyTorchPagedAttentionBackend, is_metal_available
from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.geometry import KVCacheGeometry
from pagedserve.memory.tensor_block_store import TensorBlockStore
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.attention import paged_attention_reference


# Skip all Metal tests if Metal execution is not available in the current environment
pytestmark = pytest.mark.skipif(not is_metal_available(), reason="Metal execution backend is not supported in this environment.")


@pytest.mark.parametrize("seq_length", [1, 7, 15, 16, 17, 31, 32, 33, 64, 127, 256])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("head_config", [
    (12, 12, 64),  # MHA 64-dim
    (8, 8, 32),    # MHA 32-dim
    (12, 4, 64),   # GQA 64-dim
    (12, 1, 64),   # MQA 64-dim
])
def test_metal_attention_correctness_matrix(seq_length, block_size, head_config):
    """Test Metal backend against PyTorch reference across sequence lengths, block sizes, and head configurations."""
    torch.manual_seed(42 + seq_length)
    num_attn_heads, num_kv_heads, head_dim = head_config
    num_layers = 1

    geom = KVCacheGeometry(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attn_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    num_blocks_needed = math.ceil(seq_length / block_size)
    pool = BlockPool(num_blocks=num_blocks_needed + 4, block_size=block_size)
    store = TensorBlockStore(geometry=geom, num_blocks=num_blocks_needed + 4)

    phys_blocks = [pool.allocate().block_id for _ in range(num_blocks_needed)]
    block_table = BlockTable(request_id="metal_matrix_req")
    for p_id in phys_blocks:
        block_table.append_block(p_id)

    for b_idx, p_id in enumerate(phys_blocks):
        start = b_idx * block_size
        end = min((b_idx + 1) * block_size, seq_length)
        val_len = end - start
        k_data = torch.randn(val_len, num_kv_heads, head_dim)
        v_data = torch.randn(val_len, num_kv_heads, head_dim)
        store.write_block(layer_idx=0, physical_block_id=p_id, key_block=k_data, value_block=v_data, num_tokens=val_len)

    paged_view = PagedKVView(block_table=block_table, seq_length=seq_length, store=store)
    decode_query = torch.randn(1, num_attn_heads, 1, head_dim)

    # 1. Oracle: PyTorch blockwise attention
    backend_pt = PyTorchPagedAttentionBackend()
    out_pt = backend_pt.forward(decode_query, paged_view, layer_idx=0)

    # 2. Target: Metal paged attention
    backend_metal = MetalPagedAttentionBackend()
    out_metal = backend_metal.forward(decode_query, paged_view, layer_idx=0)

    # 3. Correctness check
    assert torch.allclose(out_metal, out_pt, atol=1e-4), (
        f"Metal mismatch for seq_length={seq_length}, block_size={block_size}, head_config={head_config}. "
        f"Max diff: {(out_metal - out_pt).abs().max().item():.6e}"
    )


def test_metal_no_gather_enforcement(monkeypatch):
    """Verify Metal backend executes without calling PagedKVView gather methods."""
    torch.manual_seed(123)
    geom = KVCacheGeometry(num_layers=1, num_kv_heads=4, num_attention_heads=4, head_dim=32, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    pool = BlockPool(num_blocks=4, block_size=16)
    store = TensorBlockStore(geometry=geom, num_blocks=4)

    p_id = pool.allocate().block_id
    store.write_block(0, p_id, torch.randn(16, 4, 32), torch.randn(16, 4, 32))

    block_table = BlockTable(request_id="req_metal_nogather")
    block_table.append_block(p_id)
    paged_view = PagedKVView(block_table=block_table, seq_length=16, store=store)

    def forbidden_gather(*args, **kwargs):
        raise RuntimeError("GATHER METHOD WAS CALLED IN METAL BACKEND!")

    monkeypatch.setattr(paged_view, "gather_keys", forbidden_gather)
    monkeypatch.setattr(paged_view, "gather_values", forbidden_gather)
    monkeypatch.setattr(paged_view, "gather_layer", forbidden_gather)

    query = torch.randn(1, 4, 1, 32)
    backend_metal = MetalPagedAttentionBackend()

    # Must complete without calling gather
    out = backend_metal.forward(query, paged_view, layer_idx=0)
    assert out.shape == (1, 4, 1, 32)


def test_metal_scrambled_physical_blocks():
    """Verify Metal kernel correctness under non-contiguous scrambled physical block table indexing."""
    torch.manual_seed(999)
    geom = KVCacheGeometry(num_layers=1, num_kv_heads=2, num_attention_heads=2, head_dim=16, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    pool = BlockPool(num_blocks=50, block_size=16)
    store = TensorBlockStore(geometry=geom, num_blocks=50)

    all_blocks = [pool.allocate().block_id for _ in range(50)]
    scrambled = [all_blocks[29], all_blocks[3], all_allocated := all_blocks[44], all_blocks[12]]

    seq_length = 60
    block_table = BlockTable(request_id="req_scrambled_metal")
    for p_id in scrambled:
        block_table.append_block(p_id)

    for b_idx, p_id in enumerate(scrambled):
        start = b_idx * 16
        end = min((b_idx + 1) * 16, seq_length)
        v_len = end - start
        store.write_block(0, p_id, torch.randn(v_len, 2, 16), torch.randn(v_len, 2, 16), num_tokens=v_len)

    paged_view = PagedKVView(block_table=block_table, seq_length=seq_length, store=store)
    query = torch.randn(1, 2, 1, 16)

    out_pt = paged_attention_reference(query, paged_view, layer_idx=0)
    backend_metal = MetalPagedAttentionBackend()
    out_metal = backend_metal.forward(query, paged_view, layer_idx=0)

    assert torch.allclose(out_metal, out_pt, atol=1e-4)


def test_metal_real_model_geometry_distilgpt2():
    """Verify Metal backend using real model geometry (distilgpt2: 12 heads, 64 dim, float32)."""
    # distilgpt2 specs: n_head=12, n_embd=768 -> head_dim = 64
    geom = KVCacheGeometry(
        num_layers=6,
        num_kv_heads=12,
        num_attention_heads=12,
        head_dim=64,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool = BlockPool(num_blocks=16, block_size=16)
    store = TensorBlockStore(geometry=geom, num_blocks=16)

    seq_len = 100
    num_blocks = math.ceil(seq_len / 16)
    phys_blocks = [pool.allocate().block_id for _ in range(num_blocks)]

    block_table = BlockTable(request_id="distilgpt2_req")
    for p_id in phys_blocks:
        block_table.append_block(p_id)

    for b_idx, p_id in enumerate(phys_blocks):
        v_len = min(16, seq_len - b_idx * 16)
        k_data = torch.randn(v_len, 12, 64)
        v_data = torch.randn(v_len, 12, 64)
        store.write_block(layer_idx=0, physical_block_id=p_id, key_block=k_data, value_block=v_data, num_tokens=v_len)

    paged_view = PagedKVView(block_table=block_table, seq_length=seq_len, store=store)
    decode_query = torch.randn(1, 12, 1, 64)

    out_pt = PyTorchPagedAttentionBackend().forward(decode_query, paged_view, layer_idx=0)
    out_metal = MetalPagedAttentionBackend().forward(decode_query, paged_view, layer_idx=0)

    assert torch.allclose(out_metal, out_pt, atol=1e-4)
