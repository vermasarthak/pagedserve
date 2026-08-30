"""Integration tests for PagedServeEngine: end-to-end generation, reference correctness, and cleanup."""

import pytest
import torch
from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams
from pagedserve.engine.state import RequestState
from pagedserve.model.loader import LoadedModel


def test_engine_single_request_generation(cpu_loaded_model: LoadedModel):
    """Verify end-to-end generation of a single request on a real causal LM."""
    config = EngineConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu",
        num_blocks=20,
        block_size=16,
    )
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)

    prompt = "Hello, my name is"
    req_id = engine.submit(
        prompt=prompt,
        sampling_params=SamplingParams(max_new_tokens=6, temperature=0.0),
    )

    req = engine.run_until_complete(req_id, max_steps=50)

    # 1. Assertions on request state and output
    assert req.state == RequestState.FINISHED
    assert req.num_generated_tokens == 6
    assert len(req.generated_token_ids) == 6

    # 2. Non-empty decoded text produced
    text = engine.get_output_text(req_id)
    assert isinstance(text, str) and len(text) > 0

    # 3. Post-completion resource cleanup
    engine.step()  # drain step
    assert req_id not in engine._model_states
    assert engine.scheduler.num_running == 0
    assert engine.kv_cache_mgr.used_blocks == 0
    engine.kv_cache_mgr.verify_clean(clear_prefix=True)


def test_engine_reference_greedy_equivalence(cpu_loaded_model: LoadedModel):
    """Critical correctness test: verify PagedServe greedy output matches HF standard loop bit-for-bit."""
    config = EngineConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu",
        num_blocks=20,
        block_size=16,
    )
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)
    model = cpu_loaded_model.model
    tokenizer = cpu_loaded_model.tokenizer

    prompt = "The quick brown fox"
    max_tokens = 5

    # 1. Reference generation using direct PyTorch/HuggingFace autoregressive loop
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
    reference_tokens = []
    curr_input = input_ids.clone()
    with torch.inference_mode():
        for _ in range(max_tokens):
            out = model(curr_input)
            next_token = int(torch.argmax(out.logits[0, -1, :]).item())
            reference_tokens.append(next_token)
            curr_input = torch.cat([curr_input, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    # 2. PagedServe generation
    req_id = engine.submit(
        prompt=prompt,
        sampling_params=SamplingParams(max_new_tokens=max_tokens, temperature=0.0),
    )
    req = engine.run_until_complete(req_id)

    # 3. Bitwise token identity assertion
    assert req.generated_token_ids == reference_tokens, (
        f"Greedy generation mismatch! PagedServe: {req.generated_token_ids} vs Reference: {reference_tokens}"
    )

    engine.step()
    engine.kv_cache_mgr.verify_clean(clear_prefix=True)


def test_engine_multi_request_concurrency(cpu_loaded_model: LoadedModel):
    """Verify concurrent execution and continuous batching across multiple real requests."""
    config = EngineConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu",
        num_blocks=50,
        block_size=16,
        max_num_sequences=4,
    )
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)

    prompts = [
        ("req-A", "Artificial intelligence", 4),
        ("req-B", "The future of computing", 6),
        ("req-C", "Once upon a time", 3),
    ]

    for req_id, p, max_toks in prompts:
        engine.submit(
            prompt=p,
            sampling_params=SamplingParams(max_new_tokens=max_toks, temperature=0.0),
            request_id=req_id,
        )

    # Step engine until all requests finish
    steps = 0
    while engine.has_active_work and steps < 100:
        engine.step()
        steps += 1

    assert not engine.has_active_work

    for req_id, _, max_toks in prompts:
        req = engine.get_request(req_id)
        assert req.state == RequestState.FINISHED
        assert req.num_generated_tokens == max_toks
        text = engine.get_output_text(req_id)
        assert len(text) > 0

    # Verify zero leaks across model states and memory manager
    assert len(engine._model_states) == 0
    engine.kv_cache_mgr.verify_clean(clear_prefix=True)


def test_engine_cancellation_reclaims_all_resources(cpu_loaded_model: LoadedModel):
    """Verify cancelling an in-flight request frees both metadata blocks and model KV tensors."""
    config = EngineConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu",
        num_blocks=20,
        block_size=16,
    )
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)

    req_id = engine.submit("Long running prompt text", sampling_params=SamplingParams(max_new_tokens=50))
    # Step once to admit and run prefill
    engine.step()

    # Cancel while active
    cancelled = engine.cancel(req_id)
    assert cancelled

    # Drain step
    engine.step()

    req = engine.get_request(req_id)
    assert req.state == RequestState.CANCELLED
    assert req_id not in engine._model_states
    assert engine.kv_cache_mgr.used_blocks == 0
    engine.kv_cache_mgr.verify_clean(clear_prefix=True)
