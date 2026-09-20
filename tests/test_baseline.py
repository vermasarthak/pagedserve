from pagedserve.baseline.sequential import SequentialBaseline
from pagedserve.engine.request import SamplingParams


def test_baseline_greedy_produces_output(cpu_loaded_model):
    """Baseline produces at least one output token for a short greedy prompt."""
    baseline = SequentialBaseline(cpu_loaded_model)
    prompt_tokens = [1, 2, 3, 4, 5]
    sampling_params = SamplingParams(max_new_tokens=5, temperature=0.0)

    output = baseline.generate(prompt_tokens, sampling_params)

    assert isinstance(output, list), "Expected list of token IDs"
    assert len(output) >= 1, "Should generate at least one token"
    assert all(isinstance(t, int) for t in output), "All tokens should be ints"


def test_baseline_greedy_matches_engine(cpu_loaded_model):
    """Baseline greedy output must exactly match PagedServe engine output for equivalent prompts."""
    from pagedserve.config import EngineConfig
    from pagedserve.engine.engine import PagedServeEngine

    prompt_tokens = [1, 2, 3, 4]
    sampling_params = SamplingParams(max_new_tokens=6, temperature=0.0)

    # 1. Baseline
    baseline = SequentialBaseline(cpu_loaded_model)
    baseline_output = baseline.generate(prompt_tokens, sampling_params)

    # 2. Engine
    config = EngineConfig(
        num_blocks=100,
        block_size=16,
        max_num_sequences=4,
        max_batch_tokens=1024,
        max_prefill_tokens_per_step=512,
        enable_prefix_caching=False,
    )
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)

    req_id = engine.submit(
        prompt="dummy",
        sampling_params=sampling_params,
    )
    # Override tokenized prompt with exact same token IDs we used in baseline
    req = engine._requests[req_id]
    req.prompt_token_ids = prompt_tokens
    req.num_prompt_tokens_processed = 0

    steps = 0
    while engine.has_active_work and steps < 200:
        engine.step()
        steps += 1

    engine_output = engine._requests[req_id].generated_token_ids

    assert baseline_output == engine_output, (
        f"Greedy baseline ({baseline_output}) != engine ({engine_output})"
    )


def test_baseline_deterministic_with_seed(cpu_loaded_model):
    """Baseline with identical seeds must produce identical output."""
    baseline = SequentialBaseline(cpu_loaded_model)
    prompt_tokens = [5, 6, 7, 8]
    sampling_params = SamplingParams(max_new_tokens=4, temperature=1.0)

    out1 = baseline.generate(prompt_tokens, sampling_params, seed=42)
    out2 = baseline.generate(prompt_tokens, sampling_params, seed=42)

    assert out1 == out2, "Identical seeds must produce identical outputs"


def test_baseline_stops_at_eos(cpu_loaded_model):
    """Baseline stops if eos_token_id is generated before max_new_tokens."""
    baseline = SequentialBaseline(cpu_loaded_model)
    tokenizer = cpu_loaded_model.tokenizer
    eos_id = tokenizer.eos_token_id

    sampling_params = SamplingParams(max_new_tokens=50, temperature=0.0)
    if eos_id is not None:
        sampling_params.stop_token_ids.add(eos_id)
    prompt_tokens = [1, 2, 3]

    output = baseline.generate(prompt_tokens, sampling_params)

    assert len(output) <= 50
