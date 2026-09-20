"""Model runner executing prefill, chunked prefill, and auto-regressive decode steps."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from pagedserve.model.loader import LoadedModel


@dataclass
class ModelSequenceState:
    """Per-sequence model execution state holding PyTorch transformer activations.
    
    Architectural Note:
    PagedServe maintains a clean separation between:
    1. The Systems Metadata KV Subsystem (KVBlock, BlockPool, BlockTable, PrefixCache),
       which virtualizes memory, tracks reference counts, and drives scheduling.
    2. The Framework KV State (past_key_values), which holds the actual PyTorch activation
       tensors managed by Hugging Face Transformers for auto-regressive decode reuse.
    
    At this milestone, Hugging Face's native past_key_values is used for tensor caching,
    while PagedServe's block-based memory manager governs admission, sequence capacity,
    and lifecycle reclamation.
    """

    request_id: str
    past_key_values: Any | None = None
    last_logits: torch.Tensor | None = None
    seq_length: int = 0


class ModelRunner:
    """Executes transformer forward passes for prompt prefill and single-token decode steps."""

    def __init__(self, loaded_model: LoadedModel):
        self.loaded_model: LoadedModel = loaded_model
        self.model = loaded_model.model
        self.tokenizer = loaded_model.tokenizer
        self.device: torch.device = loaded_model.device
        self.dtype: torch.dtype = loaded_model.dtype

    @torch.inference_mode()
    def prefill(
        self,
        request_id: str,
        token_ids: Sequence[int],
        existing_state: ModelSequenceState | None = None,
    ) -> tuple[torch.Tensor, ModelSequenceState]:
        """Execute a prefill forward pass on a full prompt or a prompt chunk.
        
        Args:
            request_id: Identifier of the request.
            token_ids: Slice of token IDs to process in this prefill step.
            existing_state: Prior ModelSequenceState if this is an incremental chunk of a long prompt.
            
        Returns:
            Tuple of (last_token_logits, updated_model_sequence_state).
        """
        if not token_ids:
            raise ValueError(f"Cannot run prefill with empty token_ids for request '{request_id}'")

        input_tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        past_kv = existing_state.past_key_values if existing_state is not None else None

        # Execute causal LM forward pass with KV caching enabled
        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=past_kv,
            use_cache=True,
        )

        # Logits shape: (batch=1, seq_len, vocab_size) -> extract final position (vocab_size,)
        last_logits = outputs.logits[0, -1, :]

        prev_len = existing_state.seq_length if existing_state is not None else 0
        new_seq_len = prev_len + len(token_ids)

        new_state = ModelSequenceState(
            request_id=request_id,
            past_key_values=outputs.past_key_values,
            last_logits=last_logits,
            seq_length=new_seq_len,
        )
        return last_logits, new_state

    @torch.inference_mode()
    def decode(
        self,
        request_id: str,
        next_token_id: int,
        state: ModelSequenceState,
    ) -> tuple[torch.Tensor, ModelSequenceState]:
        """Execute a single-token auto-regressive decode step reusing past KV activations.
        
        Mandatory: Feeds ONLY the single new token into the transformer, relying on
        past_key_values rather than recomputing the full sequence.
        
        Args:
            request_id: Identifier of the request.
            next_token_id: The single newly generated token ID from the previous step.
            state: The current ModelSequenceState containing prior past_key_values.
            
        Returns:
            Tuple of (next_token_logits, updated_model_sequence_state).
        """
        if state.past_key_values is None:
            raise ValueError(
                f"Cannot decode request '{request_id}': past_key_values is None (prefill must be run first)"
            )

        # Single-token input tensor: shape (1, 1)
        input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

        # Forward pass with 1 token + past_key_values
        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=state.past_key_values,
            use_cache=True,
        )

        # Logits shape: (1, 1, vocab_size) -> extract (vocab_size,)
        step_logits = outputs.logits[0, -1, :]
        new_seq_len = state.seq_length + 1

        updated_state = ModelSequenceState(
            request_id=request_id,
            past_key_values=outputs.past_key_values,
            last_logits=step_logits,
            seq_length=new_seq_len,
        )
        return step_logits, updated_state
