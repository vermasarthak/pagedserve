from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pagedserve.engine.request import SamplingParams


class Sampler:
    """Provides modular token sampling from raw model logits."""

    @staticmethod
    def sample(
        logits: torch.Tensor,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample the next token ID from a 1D or (1, V) logits tensor.

        Sampling Pipeline:
            Raw Logits
                |
            Greedy Check (temp == 0) -> argmax
                |
            Temperature Scaling (logits / temp)
                |
            Top-K Filtering (mask below rank K)
                |
            Top-P Nucleus Filtering (mask below cumulative probability P)
                |
            Softmax (convert to categorical distribution)
                |
            Multinomial Sampling -> next_token_id

        Args:
            logits: Output logits tensor from the model runner.
            sampling_params: Generation hyperparameters.
            generator: Optional torch.Generator for deterministic reproducibility.

        Returns:
            Sampled token ID as an integer in [0, vocab_size - 1].
        """
        # Ensure logits is a 1D float tensor
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
        elif logits.dim() != 1:
            raise ValueError(f"Expected 1D or (1, V) logits tensor, got shape {tuple(logits.shape)}")

        # Clone to prevent in-place mutation of runner tensors
        filtered_logits = logits.clone().float()

        # 1. Greedy argmax decoding (temperature <= 0)
        if sampling_params.temperature <= 0.0:
            return int(torch.argmax(filtered_logits).item())

        # 2. Temperature scaling
        filtered_logits = filtered_logits / sampling_params.temperature

        # 3. Top-K filtering
        top_k = sampling_params.top_k
        vocab_size = filtered_logits.size(-1)
        if 0 < top_k < vocab_size:
            # Get threshold at K-th largest logit
            kth_val, _ = torch.kthvalue(filtered_logits, vocab_size - top_k + 1)
            filtered_logits[filtered_logits < kth_val] = -float("inf")

        # 4. Top-P (Nucleus) filtering
        top_p = sampling_params.top_p
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens where cumulative probability exceeds top_p
            # Shift mask to ensure the highest-probability token is always preserved
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            filtered_logits[indices_to_remove] = -float("inf")

        # 5. Softmax & Categorical Multinomial Sampling
        probs = torch.softmax(filtered_logits, dim=-1)

        # Numerical safety: if distribution collapsed to NaNs or all zeros, fall back to argmax
        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0.0:
            return int(torch.argmax(logits).item())

        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(sampled.item())
