"""Tests for SVD-based LoRA initialization on various matrix shapes."""

import jax
import jax.numpy as jnp
import pytest

from src.model.lora import init_lora_from_svd


class TestSVDDecomposition:
    """Tests for SVD decomposition on random matrices of various shapes."""

    @pytest.mark.parametrize(
        "shape",
        [(32, 32), (64, 32), (32, 64), (128, 16), (16, 128)],
    )
    def test_full_rank_recovery_various_shapes(self, shape: tuple[int, int]):
        """Full-rank SVD should exactly recover the residual for any shape."""
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        original = jax.random.normal(k1, shape)
        shared = jax.random.normal(k2, shape)
        full_rank = min(shape)

        lora_a, lora_b = init_lora_from_svd(original, shared, rank=full_rank)
        reconstructed = lora_b.T @ lora_a
        residual = original - shared

        assert jnp.allclose(reconstructed, residual, atol=1e-4)

    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    def test_truncated_rank_error_decreases_with_rank(self, rank: int):
        """Higher rank should give lower reconstruction error."""
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        original = jax.random.normal(k1, (64, 32))
        shared = jax.random.normal(k2, (64, 32))
        residual = original - shared

        lora_a, lora_b = init_lora_from_svd(original, shared, rank=rank)
        reconstructed = lora_b.T @ lora_a
        error = jnp.mean((reconstructed - residual) ** 2)

        lora_a_prev, lora_b_prev = init_lora_from_svd(
            original, shared, rank=max(1, rank - 1)
        )
        reconstructed_prev = lora_b_prev.T @ lora_a_prev
        error_prev = jnp.mean((reconstructed_prev - residual) ** 2)

        if rank > 1:
            assert error < error_prev

    def test_rank_one_is_best_rank_one_approximation(self):
        """Rank-1 SVD should capture the largest singular value direction."""
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        original = jax.random.normal(k1, (32, 16))
        shared = jax.random.normal(k2, (32, 16))
        residual = original - shared

        lora_a, lora_b = init_lora_from_svd(original, shared, rank=1)
        reconstructed = lora_b.T @ lora_a

        u, s, vt = jnp.linalg.svd(residual, full_matrices=False)
        expected_reconstruction = jnp.outer(u[:, 0] * s[0], vt[0, :])

        assert jnp.allclose(reconstructed, expected_reconstruction, atol=1e-4)
