"""KV Cache Geometry metadata structure derived from loaded transformer models."""

from dataclasses import dataclass
from typing import Any, Optional
import torch

from pagedserve.model.loader import LoadedModel
from pagedserve.errors import PagedServeError


class UnsupportedArchitectureError(PagedServeError):
    """Raised when an unsupported model architecture is supplied for physical KV cache geometry."""
    pass


@dataclass(frozen=True)
class KVCacheGeometry:
    """Encapsulates physical Key-Value cache tensor geometry for a transformer model.
    
    Attributes:
        num_layers: Total number of transformer hidden layers.
        num_kv_heads: Number of Key-Value attention heads per layer (supports MHA, GQA, MQA).
        num_attention_heads: Number of Query attention heads per layer.
        head_dim: Dimensionality of each attention head (hidden_size // num_attention_heads).
        block_size: Number of tokens stored per physical block.
        dtype: Data type of stored KV tensors (e.g., torch.float32, torch.float16).
        device: PyTorch device where physical KV tensors reside (cpu, mps, cuda).
    """

    num_layers: int
    num_kv_heads: int
    num_attention_heads: int
    head_dim: int
    block_size: int
    dtype: torch.dtype
    device: torch.device

    @property
    def bytes_per_token_kv(self) -> int:
        """Total memory in bytes required to store Key and Value for 1 token across all layers.
        
        Formula: 2 (K & V) * num_layers * num_kv_heads * head_dim * element_size
        """
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * element_size

    @property
    def bytes_per_block(self) -> int:
        """Total memory in bytes required to store Key and Value for 1 block across all layers."""
        return self.bytes_per_token_kv * self.block_size

    @classmethod
    def from_loaded_model(
        cls,
        loaded_model: LoadedModel,
        block_size: int = 16,
    ) -> "KVCacheGeometry":
        """Extract KV cache geometry from a LoadedModel instance.
        
        Currently supports:
        - GPT-2 family (`GPT2LMHeadModel`, `GPT2Model`, etc.)
        - Llama / Mistral / Qwen / generic CausalLM with standard HF configs (`n_layer`/`num_hidden_layers`, etc.)
        """
        config = loaded_model.model.config

        # 1. Resolve number of layers
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(config, "n_layer", None)
        if num_layers is None:
            raise UnsupportedArchitectureError(
                f"Cannot determine layer count for model class '{loaded_model.model.__class__.__name__}'. "
                "Expected config attribute 'num_hidden_layers' or 'n_layer'."
            )

        # 2. Resolve Query attention heads
        num_attention_heads = getattr(config, "num_attention_heads", None)
        if num_attention_heads is None:
            num_attention_heads = getattr(config, "n_head", None)
        if num_attention_heads is None:
            raise UnsupportedArchitectureError(
                f"Cannot determine attention head count for model class '{loaded_model.model.__class__.__name__}'. "
                "Expected config attribute 'num_attention_heads' or 'n_head'."
            )

        # 3. Resolve Key-Value attention heads (supports MHA, GQA, MQA)
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads  # Default MHA (Multi-Head Attention)

        # 4. Resolve head dimension
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(config, "n_embd", None)
            if hidden_size is None:
                raise UnsupportedArchitectureError(
                    f"Cannot determine head dimension for model class '{loaded_model.model.__class__.__name__}'. "
                    "Expected config attribute 'head_dim', 'hidden_size', or 'n_embd'."
                )
            head_dim = hidden_size // num_attention_heads

        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=loaded_model.dtype,
            device=loaded_model.device,
        )
