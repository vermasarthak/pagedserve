import pytest
from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams


def test_engine_streaming_correctness(cpu_loaded_model):
    """Verify that streamed token IDs match full engine generated token IDs."""
    config = EngineConfig(num_blocks=100)
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)

    prompt = "Test prompt for streaming"
    params = SamplingParams(max_new_tokens=8, temperature=0.0)

    # 1. Run via streaming
    req_id = engine.submit(prompt, sampling_params=params)
    events = list(engine.run_and_stream(req_id))

    assert len(events) > 0
    streamed_token_ids = [e.token_id for e in events]

    # 2. Get generated token IDs from engine request record
    req = engine.get_request(req_id)
    assert req is not None
    assert streamed_token_ids == req.generated_token_ids
    assert events[-1].finished is True
    assert events[-1].finish_reason in ("stop", "length")


def test_streaming_cancellation_terminates_stream(cpu_loaded_model):
    """Verify that cancelling a request terminates the streaming event generator."""
    config = EngineConfig(num_blocks=100)
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)

    params = SamplingParams(max_new_tokens=20, temperature=0.0)
    req_id = engine.submit("Short prompt", sampling_params=params)

    # Step once to produce prefill token
    engine.step()

    # Cancel request
    cancelled = engine.cancel(req_id)
    assert cancelled is True

    # Remaining stream events should yield queued tokens and terminate
    events = list(engine.run_and_stream(req_id))
    # Stream is now closed and done
    events_after = list(engine.run_and_stream(req_id))
    assert len(events_after) == 0
