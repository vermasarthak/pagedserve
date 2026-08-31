import pytest
from pagedserve.scheduler.policy import MemoryAwarePolicy, KVPressureLevel
from pagedserve.engine.request import InferenceRequest, SamplingParams


def test_memory_aware_pressure_levels():
    policy = MemoryAwarePolicy(low_threshold=0.60, high_threshold=0.85)

    assert policy.pressure_level(0.40) == KVPressureLevel.LOW
    assert policy.pressure_level(0.65) == KVPressureLevel.MEDIUM
    assert policy.pressure_level(0.90) == KVPressureLevel.HIGH


def test_memory_aware_chunk_adjustment():
    policy = MemoryAwarePolicy(low_threshold=0.60, high_threshold=0.85)
    base_chunk = 512

    assert policy.adjusted_prefill_chunk(0.40, base_chunk) == 512
    assert policy.adjusted_prefill_chunk(0.70, base_chunk) == 256
    assert policy.adjusted_prefill_chunk(0.90, base_chunk) == 128


def test_memory_aware_admission():
    policy = MemoryAwarePolicy(low_threshold=0.60, high_threshold=0.85)
    req = InferenceRequest(
        request_id="r1",
        prompt="hello",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )

    # Low pressure -> admit
    assert policy.should_admit(0.40, req, num_running=2, max_num_sequences=10) is True

    # High pressure -> restrict large requests unless starved
    large_req = InferenceRequest(
        request_id="r2",
        prompt="large prompt",
        prompt_token_ids=list(range(100)),
        sampling_params=SamplingParams(),
    )
    assert policy.should_admit(0.90, large_req, num_running=2, max_num_sequences=10) is False

    # Starved request gets admitted even under high pressure
    for _ in range(60):
        policy.record_wait_iteration("r2")
    assert policy.should_admit(0.90, large_req, num_running=2, max_num_sequences=10) is True
