"""Tests for ModelRunner: prefill, decode with past_key_values reuse, and chunked prefill equivalence."""

import pytest
import torch
from pagedserve.model.loader import LoadedModel
from pagedserve.model.runner import ModelRunner, ModelSequenceState


def test_runner_prefill_and_single_token_decode_with_cache(cpu_loaded_model: LoadedModel):
    """Verify prefill generates initial KV cache and decode reuses cache with 1 token."""
    runner = ModelRunner(loaded_model=cpu_loaded_model)
    prompt_tokens = [101, 102, 103, 104]  # 4 tokens

    # 1. Prefill
    logits, state = runner.prefill("req-1", prompt_tokens)
    assert logits.dim() == 1
    assert logits.size(0) == cpu_loaded_model.vocab_size
    assert state.past_key_values is not None
    assert state.seq_length == 4

    # Verify cache sequence length in framework
    if hasattr(state.past_key_values, "get_seq_length"):
        assert state.past_key_values.get_seq_length() == 4

    # 2. Decode Step 1: feed ONLY a single new token (e.g. 500)
    next_token = 500
    step_logits_1, state_1 = runner.decode("req-1", next_token, state)
    assert step_logits_1.dim() == 1
    assert step_logits_1.size(0) == cpu_loaded_model.vocab_size
    assert state_1.seq_length == 5
    if hasattr(state_1.past_key_values, "get_seq_length"):
        assert state_1.past_key_values.get_seq_length() == 5

    # 3. Decode Step 2: feed ONLY the next token (e.g. 501)
    step_logits_2, state_2 = runner.decode("req-1", 501, state_1)
    assert state_2.seq_length == 6
    if hasattr(state_2.past_key_values, "get_seq_length"):
        assert state_2.past_key_values.get_seq_length() == 6


def test_runner_chunked_prefill_matches_full_prefill(cpu_loaded_model: LoadedModel):
    """Verify that chunked prefill produces identical logits to a single-pass full prefill."""
    runner = ModelRunner(loaded_model=cpu_loaded_model)
    full_prompt = [40, 284, 13, 1204, 318, 257, 1332, 286, 262, 995]  # 10 tokens

    # A. Full prefill in one forward pass
    full_logits, full_state = runner.prefill("full", full_prompt)

    # B. Chunked prefill: Chunk 1 (6 tokens) + Chunk 2 (4 tokens)
    chunk_1 = full_prompt[:6]
    chunk_2 = full_prompt[6:]

    c1_logits, c1_state = runner.prefill("chunked", chunk_1)
    assert c1_state.seq_length == 6

    c2_logits, c2_state = runner.prefill("chunked", chunk_2, existing_state=c1_state)
    assert c2_state.seq_length == 10

    # Numerical equivalence assertion
    torch.testing.assert_close(
        full_logits,
        c2_logits,
        atol=1e-4,
        rtol=1e-4,
        msg="Chunked prefill logits must match full prefill within tolerance",
    )


def test_runner_decode_without_prefill_raises(cpu_loaded_model: LoadedModel):
    """Verify calling decode with an empty or uninitialized past_key_values raises ValueError."""
    runner = ModelRunner(loaded_model=cpu_loaded_model)
    empty_state = ModelSequenceState(request_id="err", past_key_values=None)

    with pytest.raises(ValueError, match="past_key_values is None"):
        runner.decode("err", next_token_id=10, state=empty_state)
