"""Shared test fixtures for Relaxed Recursive Transformers test suite."""

import jax
import jax.numpy as jnp
import pytest

from src.model.config import get_test_config


@pytest.fixture
def small_config():
    """Return the small test configuration for CPU-based testing."""
    return get_test_config()


@pytest.fixture
def rng_key():
    """Return a deterministic JAX PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def dummy_input(rng_key):
    """Return random token IDs of shape (batch=2, seq_len=32)."""
    return jax.random.randint(rng_key, shape=(2, 32), minval=0, maxval=256)


@pytest.fixture
def dummy_hidden(rng_key):
    """Return random hidden states of shape (batch=2, seq_len=32, hidden_dim=128)."""
    return jax.random.normal(rng_key, shape=(2, 32, 128))
