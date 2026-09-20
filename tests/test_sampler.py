"""Comprehensive unit tests for Sampler (Greedy, Temperature, Top-K, Top-P)."""

import pytest
import torch

from pagedserve.engine.request import SamplingParams
from pagedserve.model.sampler import Sampler


def test_sampler_greedy_argmax():
    """Verify temperature=0.0 always deterministically returns argmax."""
    logits = torch.tensor([1.2, 5.8, 0.4, 3.1, -2.0])
    params = SamplingParams(temperature=0.0)

    sampled = Sampler.sample(logits, params)
    assert sampled == 1  # Index of 5.8


def test_sampler_seeded_determinism():
    """Verify passing a torch.Generator produces perfectly reproducible samples."""
    logits = torch.tensor([2.0, 2.1, 2.05, 1.95, 2.02])
    params = SamplingParams(temperature=1.0)

    gen1 = torch.Generator().manual_seed(42)
    s1 = [Sampler.sample(logits, params, generator=gen1) for _ in range(10)]

    gen2 = torch.Generator().manual_seed(42)
    s2 = [Sampler.sample(logits, params, generator=gen2) for _ in range(10)]

    assert s1 == s2


def test_sampler_top_k_filtering():
    """Verify top_k restricts sampling strictly to the K highest logits."""
    # Logits: index 3 (10.0), index 1 (8.0) are top 2.
    # Indices 0, 2, 4 have much lower logits (-100.0)
    logits = torch.tensor([-100.0, 8.0, -100.0, 10.0, -100.0])
    params = SamplingParams(temperature=1.0, top_k=2)

    gen = torch.Generator().manual_seed(123)
    sampled_tokens = {Sampler.sample(logits, params, generator=gen) for _ in range(50)}

    # Must only sample from {1, 3}
    assert sampled_tokens.issubset({1, 3})
    assert 0 not in sampled_tokens
    assert 2 not in sampled_tokens
    assert 4 not in sampled_tokens


def test_sampler_top_p_nucleus_filtering():
    """Verify top_p masks the tail of the probability distribution."""
    # High probability on token 0 and 1, tiny on 2, 3, 4
    logits = torch.tensor([10.0, 9.5, 0.0, -2.0, -5.0])
    params = SamplingParams(temperature=1.0, top_p=0.8)

    gen = torch.Generator().manual_seed(999)
    sampled_tokens = {Sampler.sample(logits, params, generator=gen) for _ in range(50)}

    # Tokens 2, 3, 4 must never be sampled under top_p=0.8
    assert sampled_tokens.issubset({0, 1})


def test_sampler_vocab_bounds():
    """Verify sampled token ID is always within valid vocabulary range."""
    vocab_size = 50257
    logits = torch.randn(vocab_size)
    params = SamplingParams(temperature=0.8, top_k=50, top_p=0.9)

    token_id = Sampler.sample(logits, params)
    assert 0 <= token_id < vocab_size


def test_sampler_invalid_logits_shape():
    """Verify multidimensional logits (>2D or batch > 1) raise ValueError."""
    with pytest.raises(ValueError, match="Expected 1D"):
        Sampler.sample(torch.randn(2, 5, 10), SamplingParams())
    with pytest.raises(ValueError, match="Expected 1D"):
        Sampler.sample(torch.randn(2, 10), SamplingParams())
