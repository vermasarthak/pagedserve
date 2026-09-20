"""Global engine configuration and memory hyperparameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Configuration parameters for PagedServe inference engine."""

    # Model parameters
    model_name_or_path: str = "sshleifer/tiny-gpt2"
    device: str | None = None  # None for auto-detect (mps -> cuda -> cpu)
    dtype: str = "float32"

    # KV memory block hyperparameters
    block_size: int = 16  # Number of tokens stored per physical block
    num_blocks: int = 256  # Total physical blocks pre-allocated in pool

    # Scheduler parameters
    max_num_sequences: int = 32  # Maximum concurrently running sequences
    max_batch_tokens: int = 2048  # Maximum total tokens scheduled in one engine step
    max_prefill_tokens_per_step: int = 512  # Budget for chunked prefill per iteration

    # Prefix caching
    enable_prefix_caching: bool = True

    # Telemetry
    log_stats_interval_steps: int = 100
