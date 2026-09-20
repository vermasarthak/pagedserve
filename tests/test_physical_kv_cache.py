"""Unit tests for Phase 21-28 physical block-backed KV cache subsystem."""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from pagedserve.memory.attention import paged_attention_reference
from pagedserve.memory.block_pool import BlockPool
from pagedserve.memory.block_table import BlockTable
from pagedserve.memory.geometry import KVCacheGeometry, UnsupportedArchitectureError
from pagedserve.memory.paged_view import PagedKVView
from pagedserve.memory.prefix_cache import PrefixCache
from pagedserve.memory.scatter_gather import CacheScatterGather
from pagedserve.memory.tensor_block_store import (
    PhysicalBlockStoreError,
    TensorBlockStore,
)
from pagedserve.model.cache_adapter import CacheAdapter
from pagedserve.model.loader import ModelLoader

# -----------------------------------------------------------------------------
# 1. KV Cache Geometry Tests
# -----------------------------------------------------------------------------

def test_geometry_from_loaded_model_cpu():
    loaded = ModelLoader.load("sshleifer/tiny-gpt2", device="cpu")
    geom = KVCacheGeometry.from_loaded_model(loaded, block_size=16)

    assert geom.num_layers == 2
    assert geom.num_attention_heads == 2
    assert geom.num_kv_heads == 2
    assert geom.head_dim == 1  # n_embd=2 // n_head=2 = 1
    assert geom.block_size == 16
    assert geom.dtype == torch.float32
    assert geom.device.type == "cpu"

    # Memory calculation test
    # 2 (K&V) * 2 layers * 2 heads * 1 head_dim * 4 bytes (float32) = 32 bytes / token
    assert geom.bytes_per_token_kv == 32
    assert geom.bytes_per_block == 32 * 16  # 512 bytes per block


def test_geometry_unsupported_arch():
    class FakeConfig:
        pass

    class FakeModel:
        config = FakeConfig()

    class FakeLoaded:
        model = FakeModel()
        dtype = torch.float32
        device = torch.device("cpu")

    with pytest.raises(UnsupportedArchitectureError):
        KVCacheGeometry.from_loaded_model(FakeLoaded())


# -----------------------------------------------------------------------------
# 2. TensorBlockStore & Memory Accounting Tests
# -----------------------------------------------------------------------------

def test_tensor_block_store_allocation_and_accounting():
    geom = KVCacheGeometry(
        num_layers=4,
        num_kv_heads=8,
        num_attention_heads=8,
        head_dim=64,
        block_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    num_blocks = 32
    store = TensorBlockStore(geom, num_blocks=num_blocks)

    # Shape: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    assert store.shape == (4, 32, 16, 8, 64)
    assert store.num_blocks == 32

    # Expected bytes per block: 2 * 4 layers * 8 heads * 64 dim * 4 bytes * 16 tokens = 655,360 bytes
    expected_bytes_per_block = 2 * 4 * 8 * 64 * 4 * 16
    assert store.bytes_per_block == expected_bytes_per_block
    assert store.total_bytes == expected_bytes_per_block * 32


def test_tensor_block_store_invalid_write_reads():
    geom = KVCacheGeometry(num_layers=2, num_kv_heads=2, num_attention_heads=2, head_dim=32, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    store = TensorBlockStore(geom, num_blocks=4)

    # Invalid layer index
    with pytest.raises(PhysicalBlockStoreError):
        store.read_token(layer_idx=5, physical_block_id=0, block_offset=0)

    # Invalid block ID
    with pytest.raises(Exception):
        store.read_token(layer_idx=0, physical_block_id=10, block_offset=0)

    # Invalid shape write
    bad_key = torch.randn((5, 5), dtype=torch.float32)
    bad_val = torch.randn((5, 5), dtype=torch.float32)
    with pytest.raises(PhysicalBlockStoreError):
        store.write_token(layer_idx=0, physical_block_id=0, block_offset=0, key=bad_key, value=bad_val)


# -----------------------------------------------------------------------------
# 3. Scatter, Gather, and Non-Contiguous Round-Trip Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("device_str", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_non_contiguous_scatter_gather_round_trip(device_str):
    device = torch.device(device_str)
    dtype = torch.float32

    geom = KVCacheGeometry(num_layers=2, num_kv_heads=4, num_attention_heads=4, head_dim=16, block_size=16, dtype=dtype, device=device)
    store = TensorBlockStore(geom, num_blocks=64)
    scatter_gather = CacheScatterGather(store)

    # Generate synthetic sequence of length 50 across 2 layers
    seq_len = 50
    k_layer0 = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim), dtype=dtype, device=device)
    v_layer0 = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim), dtype=dtype, device=device)
    k_layer1 = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim), dtype=dtype, device=device)
    v_layer1 = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim), dtype=dtype, device=device)

    # Scrambled non-contiguous physical block assignment (50 tokens -> 4 blocks: 16, 16, 16, 2)
    scrambled_blocks = [19, 3, 42, 8]
    table = BlockTable("scramble_req")
    for b in scrambled_blocks:
        table.append_block(b)

    # Scatter into physical store
    scatter_gather.scatter_sequence_all_layers(table, [(k_layer0, v_layer0), (k_layer1, v_layer1)])

    # Gather back from non-contiguous physical blocks
    gathered_layers = scatter_gather.gather_sequence_all_layers(table, seq_len)

    # Assert exact numerical match using torch.testing.assert_close
    torch.testing.assert_close(gathered_layers[0][0], k_layer0)
    torch.testing.assert_close(gathered_layers[0][1], v_layer0)
    torch.testing.assert_close(gathered_layers[1][0], k_layer1)
    torch.testing.assert_close(gathered_layers[1][1], v_layer1)


@pytest.mark.parametrize("seq_len, expected_blocks, final_occupancy", [
    (1, 1, 1),
    (15, 1, 15),
    (16, 1, 16),
    (17, 2, 1),
    (31, 2, 15),
    (32, 2, 16),
    (33, 3, 1),
])
def test_physical_block_boundary_lengths(seq_len, expected_blocks, final_occupancy):
    geom = KVCacheGeometry(num_layers=1, num_kv_heads=2, num_attention_heads=2, head_dim=16, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    store = TensorBlockStore(geom, num_blocks=16)
    scatter_gather = CacheScatterGather(store)

    table = BlockTable(f"boundary_req_{seq_len}")
    phys_blocks = list(range(expected_blocks))
    for b in phys_blocks:
        table.append_block(b)

    k_seq = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim))
    v_seq = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim))

    scatter_gather.scatter_sequence_layer(0, table, k_seq, v_seq)
    g_k, g_v = scatter_gather.gather_sequence_layer(0, table, seq_len)

    torch.testing.assert_close(g_k, k_seq)
    torch.testing.assert_close(g_v, v_seq)


# -----------------------------------------------------------------------------
# 4. Real Hugging Face Model Cache Import & Export Test
# -----------------------------------------------------------------------------

def test_real_hf_model_cache_import_export_round_trip():
    loaded = ModelLoader.load("sshleifer/tiny-gpt2", device="cpu")
    geom = KVCacheGeometry.from_loaded_model(loaded, block_size=16)
    store = TensorBlockStore(geom, num_blocks=32)
    scatter_gather = CacheScatterGather(store)

    # Perform real prefill pass with Hugging Face model
    inputs = loaded.tokenizer("Hello, world! PagedServe physical KV test.", return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = loaded.model(**inputs, use_cache=True)
    
    hf_cache = outputs.past_key_values
    assert isinstance(hf_cache, DynamicCache)

    # Extract canonical KV pairs from HF cache
    canonical_layers = [
        CacheAdapter.hf_cache_to_canonical(hf_cache, layer_idx)
        for layer_idx in range(geom.num_layers)
    ]

    # Map to scrambled physical blocks
    num_blocks_needed = (seq_len + 15) // 16
    scrambled = [11, 2, 29][:num_blocks_needed]
    table = BlockTable("hf_import_req")
    for b in scrambled:
        table.append_block(b)

    # Import canonical tensors into physical block store
    scatter_gather.scatter_sequence_all_layers(table, canonical_layers)

    # Reconstruct via PagedKVView and convert back to HF DynamicCache
    view = PagedKVView(table, seq_len, store)
    reconstructed_hf_cache = view.to_hf_dynamic_cache()

    # Assert numerical equivalence between original HF cache and gathered HF cache
    for layer_idx in range(geom.num_layers):
        orig_k, orig_v = CacheAdapter.hf_cache_to_canonical(hf_cache, layer_idx)
        rec_k, rec_v = CacheAdapter.hf_cache_to_canonical(reconstructed_hf_cache, layer_idx)

        torch.testing.assert_close(rec_k, orig_k)
        torch.testing.assert_close(rec_v, orig_v)


# -----------------------------------------------------------------------------
# 5. Physical Prefix Sharing & Copy-on-Write Invariant Tests
# -----------------------------------------------------------------------------

def test_physical_prefix_sharing_and_cow_protection():
    pool = BlockPool(num_blocks=16, block_size=16)
    prefix_cache = PrefixCache(pool=pool, max_cached_blocks=8)

    geom = KVCacheGeometry(num_layers=1, num_kv_heads=2, num_attention_heads=2, head_dim=16, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    store = TensorBlockStore(geom, num_blocks=16)
    scatter_gather = CacheScatterGather(store)

    # Common prefix of 16 tokens (1 full block)
    prefix_tokens = list(range(100, 116))
    prefix_k = torch.randn((16, geom.num_kv_heads, geom.head_dim))
    prefix_v = torch.randn((16, geom.num_kv_heads, geom.head_dim))

    # Request A allocates prefix block
    block_a = pool.allocate()
    block_a.append_tokens(16)
    prefix_hash = "prefix_hash_1"
    prefix_cache.insert(prefix_hash, block_a.block_id)

    # Scatter prefix KV into physical store block_a
    table_a = BlockTable("ReqA")
    table_a.append_block(block_a.block_id)
    scatter_gather.scatter_sequence_layer(0, table_a, prefix_k, prefix_v)

    # Request B reuses same prefix block from cache
    cached_id = prefix_cache.lookup(prefix_hash)
    assert cached_id == block_a.block_id
    pool.retain(cached_id)

    table_b = BlockTable("ReqB")
    table_b.append_block(cached_id)

    # Verify physical sharing
    assert table_a.blocks[0] == table_b.blocks[0]
    block_obj = pool.get_block(table_a.blocks[0])
    # Ref count should be 3: (ReqA + ReqB + PrefixCache)
    assert block_obj.ref_count == 3
    assert block_obj.ref_count > 1

    # COW Invariant Verification: Shared block MUST NOT be mutated during decode steps
    # When ReqA generates a new token (token 17), it MUST allocate a NEW private block for block 2,
    # rather than overwriting the shared immutable prefix block.
    req_a_new_block = pool.allocate()
    table_a.append_block(req_a_new_block.block_id)

    # Private block belongs solely to ReqA
    assert table_a.num_blocks == 2
    assert table_b.num_blocks == 1
    assert table_a.blocks[1] != table_b.blocks[0]
    assert pool.get_block(req_a_new_block.block_id).ref_count == 1


# -----------------------------------------------------------------------------
# 6. Reference Paged Attention Equivalence Test
# -----------------------------------------------------------------------------

def test_reference_paged_attention_equivalence():
    geom = KVCacheGeometry(num_layers=1, num_kv_heads=4, num_attention_heads=4, head_dim=32, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    store = TensorBlockStore(geom, num_blocks=16)
    scatter_gather = CacheScatterGather(store)

    seq_len = 32
    k_seq = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim))
    v_seq = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim))

    scrambled_blocks = [12, 4]
    table = BlockTable("attn_req")
    for b in scrambled_blocks:
        table.append_block(b)

    scatter_gather.scatter_sequence_layer(0, table, k_seq, v_seq)

    paged_view = PagedKVView(table, seq_len, store)
    query = torch.randn((1, geom.num_attention_heads, 1, geom.head_dim))

    # Path A: Ordinary contiguous attention math
    k_contig = k_seq.transpose(0, 1).unsqueeze(0)  # (1, num_heads, seq_len, head_dim)
    v_contig = v_seq.transpose(0, 1).unsqueeze(0)
    scores = torch.matmul(query, k_contig.transpose(-2, -1)) / (32 ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    expected_attn_output = torch.matmul(probs, v_contig)

    # Path B: Physical store -> PagedKVView -> reference paged attention
    actual_attn_output = paged_attention_reference(query, paged_view, layer_idx=0)

    # Assert outputs match
    torch.testing.assert_close(actual_attn_output, expected_attn_output)


# -----------------------------------------------------------------------------
# 7. Stress Test: Deterministic Physical Paging Under Churn
# -----------------------------------------------------------------------------

def test_deterministic_physical_paging_stress():
    torch.manual_seed(42)
    geom = KVCacheGeometry(num_layers=2, num_kv_heads=2, num_attention_heads=2, head_dim=16, block_size=16, dtype=torch.float32, device=torch.device("cpu"))
    pool = BlockPool(num_blocks=32, block_size=16)
    store = TensorBlockStore(geom, num_blocks=32)
    scatter_gather = CacheScatterGather(store)

    active_requests = {}

    # Run 50 iterations of random allocation, scatter, gather, and release
    for i in range(50):
        req_id = f"stress_req_{i}"
        seq_len = ((i % 5) + 1) * 7  # varying sequence lengths (7, 14, 21, 28, 35)
        blocks_needed = (seq_len + 15) // 16

        if pool.free_count >= blocks_needed:
            allocated = pool.allocate_many(blocks_needed)
            table = BlockTable(req_id)
            for b in allocated:
                table.append_block(b.block_id)

            k_data = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim))
            v_data = torch.randn((seq_len, geom.num_kv_heads, geom.head_dim))

            scatter_gather.scatter_sequence_all_layers(table, [(k_data, v_data), (k_data, v_data)])
            active_requests[req_id] = (table, seq_len, k_data, v_data, [b.block_id for b in allocated])

        # Periodically release oldest requests
        if len(active_requests) > 3 or i % 4 == 0:
            if active_requests:
                oldest_id = next(iter(active_requests))
                table_to_free, s_len, orig_k, orig_v, b_ids = active_requests.pop(oldest_id)
                # Verify data integrity before release
                gathered = scatter_gather.gather_sequence_layer(0, table_to_free, s_len)
                torch.testing.assert_close(gathered[0], orig_k)

                released = table_to_free.clear()
                for bid in released:
                    pool.release(bid)

    # Clean up remaining requests
    for req_id, (table_to_free, s_len, orig_k, orig_v, b_ids) in list(active_requests.items()):
        released = table_to_free.clear()
        for bid in released:
            pool.release(bid)

    # Verify zero leaks
    assert pool.free_count == pool.total_blocks
    assert pool.used_count == 0
