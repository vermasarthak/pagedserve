"""Shared test fixtures for PagedServe test suite."""

import pytest

from pagedserve.model.loader import LoadedModel, ModelLoader


@pytest.fixture(scope="session")
def cpu_loaded_model() -> LoadedModel:
    """Session-scoped fixture providing a pre-loaded tiny-gpt2 model on CPU for fast testing."""
    return ModelLoader.load("sshleifer/tiny-gpt2", device="cpu")
