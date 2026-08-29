"""Batch and scheduled item representations for inference execution."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class WorkType(str, Enum):
    """Type of computational work scheduled for an engine step."""
    PREFILL = "PREFILL"  # Processing prompt tokens (saturates compute/matrix-multiplication)
    DECODE = "DECODE"    # Generating one token auto-regressively (memory-bandwidth bound)


@dataclass(frozen=True)
class ScheduledItem:
    """Represents a discrete slice of work assigned to a specific request for one engine iteration."""

    request_id: str
    work_type: WorkType
    num_tokens: int

    # For PREFILL: range of token offsets in the prompt [start_idx, end_idx)
    prompt_token_range: Optional[Tuple[int, int]] = None

    # For DECODE: the zero-indexed output token position being generated
    decode_step_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {self.num_tokens}")
        if self.work_type == WorkType.PREFILL and self.prompt_token_range is None:
            raise ValueError("PREFILL work items must specify prompt_token_range")
        if self.work_type == WorkType.DECODE and self.num_tokens != 1:
            raise ValueError(f"DECODE work items must have num_tokens=1, got {self.num_tokens}")


@dataclass
class SchedulerBatch:
    """The aggregate batch of work scheduled for execution in a single engine step."""

    items: List[ScheduledItem] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if no work was scheduled in this step."""
        return len(self.items) == 0

    @property
    def total_tokens(self) -> int:
        """Total token budget across all prefill and decode items in this batch."""
        return sum(item.num_tokens for item in self.items)

    @property
    def num_prefill_tokens(self) -> int:
        """Total prompt tokens scheduled across all prefill chunks."""
        return sum(item.num_tokens for item in self.items if item.work_type == WorkType.PREFILL)

    @property
    def num_decode_tokens(self) -> int:
        """Total decode tokens scheduled across all active decode sequences."""
        return sum(item.num_tokens for item in self.items if item.work_type == WorkType.DECODE)

    @property
    def prefill_items(self) -> List[ScheduledItem]:
        """List of scheduled prefill operations."""
        return [item for item in self.items if item.work_type == WorkType.PREFILL]

    @property
    def decode_items(self) -> List[ScheduledItem]:
        """List of scheduled decode operations."""
        return [item for item in self.items if item.work_type == WorkType.DECODE]

    def has_request(self, request_id: str) -> bool:
        """Check if a specific request has work scheduled in this batch."""
        return any(item.request_id == request_id for item in self.items)

    def __len__(self) -> int:
        return len(self.items)
