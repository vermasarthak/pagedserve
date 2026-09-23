"""Request state definitions for PagedServe inference lifecycle."""

from enum import Enum


class RequestState(str, Enum):
    """Lifecycle states of an inference request in PagedServe.

    State transition graph:
        WAITING --> PREFILL --> DECODING --> FINISHED
           |           |           |
           +-----------+-----------+---------> CANCELLED
           |           |           |
           +-----------+-----------+---------> FAILED
    """

    WAITING = "WAITING"      # Queued in scheduler, awaiting admission and memory allocation
    PREFILL = "PREFILL"      # Active prompt prefill (may span multiple chunked steps)
    DECODING = "DECODING"    # Generating tokens iteratively one step at a time
    FINISHED = "FINISHED"    # Successfully completed (stop token or max_new_tokens reached)
    CANCELLED = "CANCELLED"  # Aborted by client disconnect or caller request
    FAILED = "FAILED"        # Terminated due to runtime exception or memory exhaustion

    @property
    def is_active(self) -> bool:
        """Returns True if the request is actively consuming or awaiting compute resources."""
        return self in (RequestState.WAITING, RequestState.PREFILL, RequestState.DECODING)

    @property
    def is_terminal(self) -> bool:
        """Returns True if the request has reached a terminal state."""
        return self in (RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED)
