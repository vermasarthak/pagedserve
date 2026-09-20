"""Comprehensive correctness tests for direct blockwise paged attention."""

import math

import pytest
import torch

from pagedserve.memory.attention import (
    paged_attention_blockwise,
    paged_attention_reference,
)
from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.geometry import KVCacheGeometry
from pagedserve.memory.memory_estimator import (
    compare_sequence_scaling,
    estimate_attention_memory,
)
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.tensor_block_store import TensorBlockStore


def test_naive_per_block_softmax_fails():
    """Prove that performing independent softmax per block produces mathematically incorrect attention outputs."""
    torch.manual_seed(42)
    num_heads = 4
    head_dim = 16
    block_size = 16

    # Query: single token decode
    query = torch.randn(1, num_heads, 1, head_dim)

    # 2 physical blocks of K and V
    k_block0 = torch.randn(block_size, num_heads, head_dim)
    v_block0 = torch.randn(block_size, num_heads, head_dim)
    k_block1 = torch.randn(block_size, num_heads, head_dim)
    v_block1 = torch.randn(block_size, num_heads, head_dim)

    # 1. Correct contiguous attention
    k_all = torch.cat([k_block0, k_block1], dim=0).transpose(0, 1).unsqueeze(0)  # (1, num_heads, 32, head_dim)
    v_all = torch.cat([v_block0, v_block1], dim=0).transpose(0, 1).unsqueeze(0)  # (1, num_heads, 32, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    scores_correct = torch.matmul(query, k_all.transpose(-2, -1)) * scale
    probs_correct = torch.softmax(scores_correct, dim=-1)
    correct_output = torch.matmul(probs_correct, v_all)

    # 2. Naive per-block independent softmax (incorrect sum of unnormalized/locally normalized outputs)
    k0_t = k_block0.transpose(0, 1).unsqueeze(0)
    v0_t = v_block0.transpose(0, 1).unsqueeze(0)
    scores0 = torch.matmul(query, k0_t.transpose(-2, -1)) * scale
    probs0 = torch.softmax(scores0, dim=-1)
    out0 = torch.matmul(probs0, v0_t)

    k1_t = k_block1.transpose(0, 1).unsqueeze(0)
    v1_t = v_block1.transpose(0, 1).unsqueeze(0)
    scores1 = torch.matmul(query, k1_t.transpose(-2, -1)) * scale
    probs1 = torch.softmax(scores1, dim=-1)
    out1 = torch.matmul(probs1, v1_t)

    naive_output = 0.5 * (out0 + out1)  # Average of independent block outputs

    # Assert naive output fails to equal correct attention
    assert not torch.allclose(naive_output, correct_output, atol=1e-3), (
        "Naive per-block softmax unexpectedly matched correct attention!"
    )


@pytest.mark.parametrize("seq_length", [1, 7, 15, 16, 17, 31, 32, 33, 64, 127])
@pytest.mark.parametrize("head_config", [
    (12, 12),  # MHA
    (12, 4),   # GQA
    (12, 1),   # MQA
])
def test_paged_attention_blockwise_equivalence(seq_length, head_config):
    """Verify paged_attention_blockwise matches paged_attention_reference across sequence lengths and head configurations."""
    torch.manual_seed(42 + seq_length)
    num_attn_heads, num_kv_heads = head_config
    head_dim = 64
    block_size = 16
    num_layers = 2

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

    # Allocate physical blocks
    phys_blocks = [pool.allocate().block_id for _ in range(num_blocks_needed)]
    block_table = BlockTable(request_id="req1")
    for p_id in phys_blocks:
        block_table.append_block(p_id)

    # Populate store with known random K/V data
    for b_idx, p_id in enumerate(phys_blocks):
        start = b_idx * block_size
        end = min((b_idx + 1) * block_size, seq_length)
        val_len = end - start
        k_data = torch.randn(val_len, num_kv_heads, head_dim)
        v_data = torch.randn(val_len, num_kv_heads, head_dim)
        store.write_block(layer_idx=0, physical_block_id=p_id, key_block=k_data, value_block=v_data, num_tokens=val_len)

    paged_view = PagedKVView(block_table=block_table, seq_length=seq_length, store=store)

    # Test single-token decode query
    decode_query = torch.randn(1, num_attn_heads, 1, head_dim)

    ref_out = paged_attention_reference(decode_query, paged_view, layer_idx=0)
    bw_out = paged_attention_blockwise(decode_query, paged_view, layer_idx=0)

    assert torch.allclose(bw_out, ref_out, atol=1e-5), f"Decode mismatch for seq_length={seq_length}, heads={head_config}"

    # Test multi-token prefill query if seq_length > 1
    if seq_length > 1:
        prefill_query = torch.randn(1, num_attn_heads, seq_length, head_dim)
        ref_prefill = paged_attention_reference(prefill_query, paged_view, layer_idx=0)
        bw_prefill = paged_attention_blockwise(prefill_query, paged_view, layer_idx=0)
        assert torch.allclose(bw_prefill, ref_prefill, atol=1e-5), f"Prefill mismatch for seq_length={seq_length}, heads={head_config}"


def test_no_gather_enforcement(monkeypatch):
    """Verify paged_attention_blockwise executes without calling any PagedKVView gather methods."""
    torch.manual_seed(123)
    geom = KVCacheGeometry(
        num_layers=1,
        num_kv_heads=4,
        num_attention_heads=4,
        head_dim=32,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool = BlockPool(num_blocks=4, block_size=16)
    store = TensorBlockStore(geometry=geom, num_blocks=4)

    p_id = pool.allocate().block_id
    k_data = torch.randn(16, 4, 32)
    v_data = torch.randn(16, 4, 32)
    store.write_block(layer_idx=0, physical_block_id=p_id, key_block=k_data, value_block=v_data)

    block_table = BlockTable(request_id="req_nogather")
    block_table.append_block(p_id)

    paged_view = PagedKVView(block_table=block_table, seq_length=16, store=store)

    # Monkeypatch gather methods to raise RuntimeError
    def forbidden_gather(*args, **kwargs):
        raise RuntimeError("GATHER METHOD WAS CALLED!")

    monkeypatch.setattr(paged_view, "gather_keys", forbidden_gather)
    monkeypatch.setattr(paged_view, "gather_values", forbidden_gather)
    monkeypatch.setattr(paged_view, "gather_layer", forbidden_gather)

    query = torch.randn(1, 4, 1, 32)
    # Must run to completion without triggering RuntimeError
    output = paged_attention_blockwise(query, paged_view, layer_idx=0)
    assert output.shape == (1, 4, 1, 32)


def test_scrambled_physical_block_table():
    """Verify correct attention output when physical block IDs are non-contiguous and scrambled."""
    torch.manual_seed(999)
    geom = KVCacheGeometry(
        num_layers=1,
        num_kv_heads=2,
        num_attention_heads=2,
        head_dim=16,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool = BlockPool(num_blocks=50, block_size=16)
    store = TensorBlockStore(geometry=geom, num_blocks=50)

    # Allocate scattered physical blocks
    all_allocated = [pool.allocate().block_id for _ in range(50)]
    scrambled_p_ids = [all_allocated[29], all_allocated[3], all_allocated[44], all_allocated[12]]

    seq_length = 60  # spans 4 blocks: 16, 16, 16, 12
    block_table = BlockTable(request_id="req_scrambled")
    for p_id in scrambled_p_ids:
        block_table.append_block(p_id)

    for b_idx, p_id in enumerate(scrambled_p_ids):
        start = b_idx * 16
        end = min((b_idx + 1) * 16, seq_length)
        v_len = end - start
        k_data = torch.randn(v_len, 2, 16)
        v_data = torch.randn(v_len, 2, 16)
        store.write_block(layer_idx=0, physical_block_id=p_id, key_block=k_data, value_block=v_data, num_tokens=v_len)

    paged_view = PagedKVView(block_table=block_table, seq_length=seq_length, store=store)
    query = torch.randn(1, 2, 1, 16)

    ref_out = paged_attention_reference(query, paged_view, layer_idx=0)
    bw_out = paged_attention_blockwise(query, paged_view, layer_idx=0)

    assert torch.allclose(bw_out, ref_out, atol=1e-5)


def test_physical_prefix_sharing_cow_under_blockwise_attention():
    """Verify blockwise attention correctness under physical prefix sharing and Copy-on-Write."""
    geom = KVCacheGeometry(
        num_layers=1,
        num_kv_heads=2,
        num_attention_heads=2,
        head_dim=16,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool = BlockPool(num_blocks=10, block_size=16)
    store = TensorBlockStore(geometry=geom, num_blocks=10)

    shared_p_id = pool.allocate().block_id
    k_data = torch.randn(16, 2, 16)
    v_data = torch.randn(16, 2, 16)
    store.write_block(layer_idx=0, physical_block_id=shared_p_id, key_block=k_data, value_block=v_data)
    pool.retain(shared_p_id)  # ref_count = 2

    # Request A
    bt_a = BlockTable(request_id="req_a")
    bt_a.append_block(shared_p_id)

    # Request B
    bt_b = BlockTable(request_id="req_b")
    bt_b.append_block(shared_p_id)

    pv_a = PagedKVView(bt_a, seq_length=16, store=store)
    pv_b = PagedKVView(bt_b, seq_length=16, store=store)

    query = torch.randn(1, 2, 1, 16)

    out_a = paged_attention_blockwise(query, pv_a, layer_idx=0)
    out_b = paged_attention_blockwise(query, pv_b, layer_idx=0)

    # Outputs for both requests must be identical on shared prefix
    assert torch.allclose(out_a, out_b, atol=1e-6)

    # Request B appends a private block (Copy-on-Write decode step)
    priv_p_id = pool.allocate().block_id
    k_priv = torch.randn(1, 2, 16)
    v_priv = torch.randn(1, 2, 16)
    store.write_block(layer_idx=0, physical_block_id=priv_p_id, key_block=k_priv, value_block=v_priv, num_tokens=1)
    bt_b.append_block(priv_p_id)
    pv_b.seq_length = 17

    out_b_new = paged_attention_blockwise(query, pv_b, layer_idx=0)
    out_a_still = paged_attention_blockwise(query, pv_a, layer_idx=0)

    # Request A output must remain completely unchanged
    assert torch.allclose(out_a, out_a_still, atol=1e-6)
    # Request B output reflects updated sequence length
    assert not torch.allclose(out_b, out_b_new, atol=1e-3)


def test_memory_estimator_calculations():
    """Verify memory estimator reduction calculations."""
    est = estimate_attention_memory(sequence_length=1024, num_heads=12, head_dim=64, block_size=16)
    # Gather: 2 * 1024 * 12 * 64 * 4 = 6,291,456 bytes
    assert est.gather_memory_bytes == 6291456
    # Blockwise: 2 * 16 * 12 * 64 * 4 = 98,304 bytes
    assert est.blockwise_memory_bytes == 98304
    assert round(est.reduction_factor, 1) == 64.0

    scaling = compare_sequence_scaling(sequence_lengths=[128, 512, 2048])
    assert len(scaling) == 3
    assert scaling[-1].reduction_factor == 128.0
