"""Tests for ModelLoader, device selection, and model metadata extraction."""

import pytest
import torch

from pagedserve.model.loader import LoadedModel, ModelLoader


def test_model_loader_cpu(cpu_loaded_model: LoadedModel):
    """Verify loading model on CPU with complete metadata."""
    loaded = cpu_loaded_model

    assert loaded.device.type == "cpu"
    assert loaded.dtype == torch.float32
    assert loaded.model_name == "sshleifer/tiny-gpt2"
    assert loaded.vocab_size > 0
    assert loaded.parameter_count == 102714
    assert loaded.context_length > 0
    assert loaded.is_cpu
    assert not loaded.is_cuda

    # Verify tokenizer pad token
    assert loaded.tokenizer.pad_token is not None
    assert loaded.tokenizer.pad_token == loaded.tokenizer.eos_token


def test_model_loader_device_and_dtype_resolution():
    """Verify device priority and dtype rules."""
    # Explicit CPU
    cpu_dev = ModelLoader.resolve_device("cpu")
    assert cpu_dev.type == "cpu"
    cpu_dtype = ModelLoader.resolve_dtype(cpu_dev)
    assert cpu_dtype == torch.float32

    # Custom dtype
    custom_dtype = ModelLoader.resolve_dtype(cpu_dev, "float32")
    assert custom_dtype == torch.float32

    # Invalid dtype rejected
    with pytest.raises(ValueError, match="Unsupported dtype"):
        ModelLoader.resolve_dtype(cpu_dev, "invalid_dtype_name")


def test_model_loader_mps_if_available():
    """Verify MPS loading if hardware supports it."""
    if torch.backends.mps.is_available():
        loaded = ModelLoader.load("sshleifer/tiny-gpt2", device="mps")
        assert loaded.device.type == "mps"
        assert loaded.is_mps
        assert not loaded.is_cpu
