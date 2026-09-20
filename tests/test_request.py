"""Comprehensive tests for Request, RequestState, SamplingParams, and state transitions."""

import pytest

from pagedserve.engine.request import InferenceRequest, SamplingParams
from pagedserve.engine.state import RequestState
from pagedserve.errors import InvalidRequestError, InvalidRequestStateError


def test_sampling_params_validation():
    """Verify SamplingParams parameter constraints and validations."""
    # Valid defaults
    params = SamplingParams()
    assert params.temperature == 1.0
    assert params.top_k == 50
    assert params.top_p == 1.0
    assert params.max_new_tokens == 64

    # Invalid temperature
    with pytest.raises(InvalidRequestError, match="temperature"):
        SamplingParams(temperature=-0.5)

    # Invalid top_k
    with pytest.raises(InvalidRequestError, match="top_k"):
        SamplingParams(top_k=-1)

    # Invalid top_p
    with pytest.raises(InvalidRequestError, match="top_p"):
        SamplingParams(top_p=0.0)
    with pytest.raises(InvalidRequestError, match="top_p"):
        SamplingParams(top_p=1.5)

    # Invalid max_new_tokens
    with pytest.raises(InvalidRequestError, match="max_new_tokens"):
        SamplingParams(max_new_tokens=0)

    # Invalid repetition_penalty
    with pytest.raises(InvalidRequestError, match="repetition_penalty"):
        SamplingParams(repetition_penalty=0.0)


def test_request_initialization_and_validation():
    """Verify request construction and input guards."""
    req = InferenceRequest(
        request_id="req-1",
        prompt="Hello world",
        prompt_token_ids=[101, 102, 103],
    )
    assert req.request_id == "req-1"
    assert req.num_prompt_tokens == 3
    assert req.num_generated_tokens == 0
    assert req.num_prompt_tokens_processed == 0
    assert req.state == RequestState.WAITING
    assert not req.is_finished
    assert not req.is_prefill_complete
    assert req.remaining_tokens == 64
    assert req.total_tokens == 0

    # Empty request_id
    with pytest.raises(InvalidRequestError, match="request_id"):
        InferenceRequest(request_id="", prompt="Hi", prompt_token_ids=[1])

    # Empty prompt_token_ids
    with pytest.raises(InvalidRequestError, match="prompt_token_ids"):
        InferenceRequest(request_id="req-2", prompt="Hi", prompt_token_ids=[])


def test_request_normal_lifecycle():
    """Verify normal progression: WAITING -> PREFILL -> DECODING -> FINISHED."""
    req = InferenceRequest(
        request_id="req-lifecycle",
        prompt="Explain KV caching",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_new_tokens=3, stop_token_ids={999}),
    )

    # 1. Admit to prefill
    req.mark_prefilling()
    assert req.state == RequestState.PREFILL
    assert not req.is_prefill_complete

    # 2. Chunked prefill progress
    req.advance_prompt_tokens(2)
    assert req.num_prompt_tokens_processed == 2
    assert not req.is_prefill_complete

    req.advance_prompt_tokens(2)
    assert req.num_prompt_tokens_processed == 4
    assert req.is_prefill_complete

    # 3. Transition to decode
    req.mark_decoding()
    assert req.state == RequestState.DECODING

    # 4. Generate first token (records TTFT)
    req.append_generated_token(10)
    assert req.num_generated_tokens == 1
    assert req.ttft is not None
    assert req.state == RequestState.DECODING
    assert not req.is_finished

    # 5. Generate second token
    req.append_generated_token(20)
    assert req.num_generated_tokens == 2
    assert not req.is_finished

    # 6. Generate third token: hits max_new_tokens (3)
    req.append_generated_token(30)
    assert req.num_generated_tokens == 3
    assert req.state == RequestState.FINISHED
    assert req.finish_reason == "length"
    assert req.is_finished
    assert req.total_latency is not None


def test_request_stop_token_termination():
    """Verify generation stops immediately upon emitting a stop token."""
    req = InferenceRequest(
        request_id="req-stop",
        prompt="Hi",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=10, stop_token_ids={50256}),
    )
    req.mark_prefilling()
    req.advance_prompt_tokens(2)
    req.mark_decoding()

    req.append_generated_token(100)
    assert not req.is_finished

    # Emit stop token
    req.append_generated_token(50256)
    assert req.is_finished
    assert req.state == RequestState.FINISHED
    assert req.finish_reason == "stop"
    assert req.num_generated_tokens == 2


def test_request_cancellation_and_failure():
    """Verify cancellation and failure transitions from active states."""
    # Cancellation from WAITING
    req1 = InferenceRequest(request_id="c1", prompt="p", prompt_token_ids=[1])
    req1.mark_cancelled()
    assert req1.state == RequestState.CANCELLED
    assert req1.finish_reason == "cancelled"
    assert req1.is_finished

    # Failure from PREFILL
    req2 = InferenceRequest(request_id="f1", prompt="p", prompt_token_ids=[1, 2])
    req2.mark_prefilling()
    req2.mark_failed("Out of memory in prefill")
    assert req2.state == RequestState.FAILED
    assert req2.error_message == "Out of memory in prefill"
    assert req2.is_finished


def test_request_invalid_transitions():
    """Verify illegal transitions throw explicit errors."""
    req = InferenceRequest(request_id="err", prompt="p", prompt_token_ids=[1, 2, 3])

    # Cannot jump from WAITING directly to DECODING
    with pytest.raises(InvalidRequestStateError):
        req.mark_decoding()

    req.mark_prefilling()
    # Cannot decode before prefill completes
    req.advance_prompt_tokens(1)  # only 1 of 3
    with pytest.raises(InvalidRequestStateError):
        req.mark_decoding()

    # Cannot advance more than total prompt tokens
    with pytest.raises(ValueError, match="exceed"):
        req.advance_prompt_tokens(5)

    # Complete prefill and decode
    req.advance_prompt_tokens(2)
    req.mark_decoding()
    req.mark_finished()

    # Cannot mutate after terminal
    with pytest.raises(InvalidRequestStateError):
        req.append_generated_token(99)
    with pytest.raises(InvalidRequestStateError):
        req.mark_cancelled()
    with pytest.raises(InvalidRequestStateError):
        req.mark_failed("fail")
