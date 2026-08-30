"""Model and Tokenizer loading abstraction for Hugging Face causal language models."""

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from pagedserve.errors import PagedServeError


class ModelLoadError(PagedServeError):
    """Raised when model or tokenizer loading fails."""
    pass


@dataclass
class LoadedModel:
    """Encapsulates a loaded Hugging Face model, tokenizer, and execution metadata."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    model_name: str
    vocab_size: int
    context_length: int
    parameter_count: int

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"


class ModelLoader:
    """Responsible for resolving devices, dtypes, and loading causal language models."""

    @staticmethod
    def resolve_device(requested_device: Optional[str] = None) -> torch.device:
        """Select execution device following: explicit user choice -> CUDA -> MPS -> CPU."""
        if requested_device is not None:
            return torch.device(requested_device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @staticmethod
    def resolve_dtype(device: torch.device, requested_dtype: Optional[str] = None) -> torch.dtype:
        """Select safe floating point precision for the chosen device."""
        if requested_dtype is not None:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if requested_dtype not in dtype_map:
                raise ValueError(
                    f"Unsupported dtype '{requested_dtype}'. Choose from {list(dtype_map.keys())}"
                )
            return dtype_map[requested_dtype]

        # Device-specific defaults:
        # CPU: float32 for stable vectorized execution
        # MPS: float32 default for broad operator stability on Apple Silicon
        # CUDA: float16 for tensor-core acceleration
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    @classmethod
    def load(
        cls,
        model_name_or_path: str = "sshleifer/tiny-gpt2",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> LoadedModel:
        """Load tokenizer and causal LM weights from Hugging Face or local path.
        
        Args:
            model_name_or_path: HF model identifier or local directory.
            device: Explicit device string ("cpu", "mps", "cuda"). Defaults to auto-detect.
            dtype: Precision string ("float32", "float16", "bfloat16"). Defaults to device-safe.
            
        Returns:
            LoadedModel container with model, tokenizer, and metadata.
        """
        target_device = cls.resolve_device(device)
        target_dtype = cls.resolve_dtype(target_device, dtype)

        try:
            # 1. Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            # For GPT-2 style tokenizers, pad_token is typically undefined
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 2. Load Causal Language Model
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=target_dtype,
            )
            model.to(target_device)
            model.eval()

            # 3. Extract Metadata
            vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
            context_length = (
                getattr(model.config, "max_position_embeddings", None)
                or getattr(model.config, "n_positions", None)
                or getattr(model.config, "n_ctx", 2048)
            )
            parameter_count = sum(p.numel() for p in model.parameters())

            return LoadedModel(
                model=model,
                tokenizer=tokenizer,
                device=target_device,
                dtype=target_dtype,
                model_name=model_name_or_path,
                vocab_size=vocab_size,
                context_length=context_length,
                parameter_count=parameter_count,
            )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model '{model_name_or_path}' on device '{target_device}': {str(e)}"
            ) from e
