
import torch
from transformers.cache_utils import DynamicCache

from pagedserve.engine.request import SamplingParams
from pagedserve.model.loader import LoadedModel
from pagedserve.model.sampler import Sampler


class SequentialBaseline:
    """A conventional baseline inference implementation for fair comparison.
    
    Processes one request at a time using a standard autoregressive loop
    with Hugging Face DynamicCache. It intentionally does not use
    continuous batching or custom KV memory management.
    """

    def __init__(self, loaded_model: LoadedModel):
        self.model = loaded_model.model
        self.tokenizer = loaded_model.tokenizer
        self.device = loaded_model.device

    def generate(
        self,
        prompt_tokens: list[int],
        sampling_params: SamplingParams,
        seed: int | None = None,
    ) -> list[int]:
        """Generate output tokens for a single prompt sequentially.
        
        Args:
            prompt_tokens: Input token IDs.
            sampling_params: Generation hyperparameters.
            seed: Optional random seed for deterministic generation.
            
        Returns:
            List of generated output token IDs (excluding prompt).
        """
        generator = torch.Generator(device=self.device) if seed is not None else None
        if generator is not None:
            generator.manual_seed(seed)

        input_ids = torch.tensor([prompt_tokens], device=self.device)
        past_key_values = DynamicCache()
        generated_tokens = []

        with torch.inference_mode():
            for _ in range(sampling_params.max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Get logits for the last token in the sequence
                next_token_logits = outputs.logits[0, -1, :]

                # Sample the next token
                next_token_id = Sampler.sample(next_token_logits, sampling_params, generator)
                generated_tokens.append(next_token_id)

                if next_token_id in sampling_params.stop_token_ids or (
                    self.tokenizer.eos_token_id is not None
                    and next_token_id == self.tokenizer.eos_token_id
                ):
                    break

                # Update input_ids for the next decode step
                input_ids = torch.tensor([[next_token_id]], device=self.device)

        return generated_tokens
