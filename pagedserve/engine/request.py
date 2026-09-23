"""Inference request representation, sampling parameters, and lifecycle tracking."""

import time
from dataclasses import dataclass, field

from pagedserve.engine.state import RequestState
from pagedserve.errors import InvalidRequestError, InvalidRequestStateError


@dataclass
class SamplingParams:
    """Sampling hyperparameters for text generation."""

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 64
    stop_token_ids: set[int] = field(default_factory=set)
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise InvalidRequestError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.top_k < 0:
            raise InvalidRequestError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise InvalidRequestError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.max_new_tokens < 1:
            raise InvalidRequestError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.repetition_penalty <= 0.0:
            raise InvalidRequestError(
                f"repetition_penalty must be > 0.0, got {self.repetition_penalty}"
            )


@dataclass
class InferenceRequest:
    """Represents a single generative inference request in the PagedServe runtime."""

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.monotonic)

    # Output tracking
    generated_token_ids: list[int] = field(default_factory=list)

    # State & lifecycle
    state: RequestState = RequestState.WAITING
    num_prompt_tokens_processed: int = 0
    finish_reason: str | None = None
    error_message: str | None = None

    # Telemetry timestamps (monotonic)
    first_token_time: float | None = None
    completion_time: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise InvalidRequestError("request_id cannot be empty")
        if not isinstance(self.prompt_token_ids, list):
            raise InvalidRequestError("prompt_token_ids must be a list of integers")
        if len(self.prompt_token_ids) == 0:
            raise InvalidRequestError("prompt_token_ids cannot be empty")
        if any(not isinstance(t, int) or t < 0 for t in self.prompt_token_ids):
            raise InvalidRequestError("prompt_token_ids must contain non-negative integers")

    # --- Computed Properties ---

    @property
    def num_prompt_tokens(self) -> int:
        """Total number of tokens in the original prompt."""
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        """Number of tokens generated so far."""
        return len(self.generated_token_ids)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens currently processed or generated."""
        return self.num_prompt_tokens_processed + self.num_generated_tokens

    @property
    def remaining_tokens(self) -> int:
        """Number of new tokens remaining before reaching max_new_tokens."""
        return max(0, self.sampling_params.max_new_tokens - self.num_generated_tokens)

    @property
    def is_finished(self) -> bool:
        """True if the request has terminated (FINISHED, CANCELLED, or FAILED)."""
        return self.state.is_terminal

    @property
    def is_prefill_complete(self) -> bool:
        """True if all prompt tokens have been ingested into the KV cache."""
        return self.num_prompt_tokens_processed == self.num_prompt_tokens

    @property
    def ttft(self) -> float | None:
        """Time to First Token (TTFT) in seconds, if first token has been generated."""
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def total_latency(self) -> float | None:
        """Total end-to-end request latency in seconds, if request is completed."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    # --- State Mutations & Lifecycle Transitions ---

    def mark_prefilling(self) -> None:
        """Transition from WAITING to PREFILL."""
        if self.state != RequestState.WAITING:
            raise InvalidRequestStateError(
                f"Cannot transition request '{self.request_id}' to PREFILL from '{self.state}'"
            )
        self.state = RequestState.PREFILL

    def advance_prompt_tokens(self, count: int) -> None:
        """Record newly processed prompt tokens during chunked or full prefill."""
        if self.state != RequestState.PREFILL:
            raise InvalidRequestStateError(
                f"Cannot advance prompt tokens while in state '{self.state}'"
            )
        if count <= 0:
            raise ValueError(f"Advance count must be positive, got {count}")
        if self.num_prompt_tokens_processed + count > self.num_prompt_tokens:
            raise ValueError(
                f"Cannot advance {count} tokens: would exceed total prompt length "
                f"({self.num_prompt_tokens_processed} + {count} > {self.num_prompt_tokens})"
            )
        self.num_prompt_tokens_processed += count

    def mark_decoding(self) -> None:
        """Transition from PREFILL to DECODING once prefill is fully completed."""
        if self.state != RequestState.PREFILL:
            raise InvalidRequestStateError(
                f"Cannot transition request '{self.request_id}' to DECODING from '{self.state}'"
            )
        if not self.is_prefill_complete:
            raise InvalidRequestStateError(
                f"Cannot transition to DECODING before prefill is complete "
                f"({self.num_prompt_tokens_processed}/{self.num_prompt_tokens} tokens processed)"
            )
        self.state = RequestState.DECODING

    def append_generated_token(self, token_id: int, timestamp: float | None = None) -> None:
        """Append a newly sampled token during decode or prefill final step."""
        if self.state not in (RequestState.PREFILL, RequestState.DECODING):
            raise InvalidRequestStateError(
                f"Cannot append token to request '{self.request_id}' in state '{self.state}'"
            )

        now = timestamp if timestamp is not None else time.monotonic()
        if self.first_token_time is None:
            self.first_token_time = now

        self.generated_token_ids.append(token_id)

        # Check termination criteria
        if token_id in self.sampling_params.stop_token_ids:
            self.mark_finished(reason="stop", timestamp=now)
        elif self.num_generated_tokens >= self.sampling_params.max_new_tokens:
            self.mark_finished(reason="length", timestamp=now)

    def mark_finished(self, reason: str = "stop", timestamp: float | None = None) -> None:
        """Mark the request as successfully finished."""
        if self.state.is_terminal:
            raise InvalidRequestStateError(
                f"Request '{self.request_id}' is already terminal ({self.state})"
            )
        self.state = RequestState.FINISHED
        self.finish_reason = reason
        self.completion_time = timestamp if timestamp is not None else time.monotonic()

    def mark_cancelled(self, timestamp: float | None = None) -> None:
        """Mark the request as cancelled by caller or client disconnect."""
        if self.state.is_terminal:
            raise InvalidRequestStateError(
                f"Request '{self.request_id}' is already terminal ({self.state})"
            )
        self.state = RequestState.CANCELLED
        self.finish_reason = "cancelled"
        self.completion_time = timestamp if timestamp is not None else time.monotonic()

    def mark_failed(self, error: str, timestamp: float | None = None) -> None:
        """Mark the request as failed due to runtime error or OOM."""
        if self.state.is_terminal:
            raise InvalidRequestStateError(
                f"Request '{self.request_id}' is already terminal ({self.state})"
            )
        self.state = RequestState.FAILED
        self.finish_reason = "error"
        self.error_message = error
        self.completion_time = timestamp if timestamp is not None else time.monotonic()
