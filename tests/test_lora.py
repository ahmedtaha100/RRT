"""Tests for LoRA modules and SVD initialization."""

import jax
import jax.numpy as jnp
import pytest

from src.model.lora import LoRAAdapter, LoRALayerSet, LoRALinear, init_lora_from_svd


class TestLoRALinear:
    """Tests for the LoRALinear module."""

    def test_output_shape(self, rng_key):
        """LoRALinear output should have shape (..., out_features)."""
        layer = LoRALinear(out_features=64, rank=8, alpha=16.0)
        x = jax.random.normal(rng_key, (2, 16, 128))
        params = layer.init(rng_key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16, 64)

    def test_zero_rank_equals_base(self, rng_key):
        """With rank=0, LoRALinear should produce identical output to the base dense."""
        layer = LoRALinear(out_features=64, rank=0)
        x = jax.random.normal(rng_key, (2, 16, 128))
        params = layer.init(rng_key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16, 64)

    def test_zeroed_lora_equals_base(self, rng_key):
        """With LoRA weights zeroed out, output should equal base dense output."""
        x = jax.random.normal(rng_key, (2, 8, 32))

        base_layer = LoRALinear(out_features=16, rank=0)
        base_params = base_layer.init(rng_key, x)
        base_output = base_layer.apply(base_params, x)

        lora_layer = LoRALinear(out_features=16, rank=4, alpha=8.0)
        lora_params = lora_layer.init(rng_key, x)

        zeroed_params = {
            "params": {
                **lora_params["params"],
                "lora_a": jnp.zeros_like(lora_params["params"]["lora_a"]),
                "lora_b": jnp.zeros_like(lora_params["params"]["lora_b"]),
            }
        }

        lora_output = lora_layer.apply(zeroed_params, x)
        assert jnp.allclose(lora_output, base_output, atol=1e-5)

    def test_gradient_flows(self, rng_key):
        """Gradients should flow through both base and LoRA parameters.

        Since lora_b is zero-initialized and lora_a is random-initialized,
        at initialization only lora_b receives nonzero gradients (the gradient
        through lora_a depends on lora_b which is zero). The base kernel
        always receives gradients.
        """
        layer = LoRALinear(out_features=16, rank=4, alpha=8.0)
        x = jax.random.normal(rng_key, (1, 4, 32))
        params = layer.init(rng_key, x)

        nonzero_lora_b = {
            "params": {
                **params["params"],
                "lora_b": jax.random.normal(rng_key, params["params"]["lora_b"].shape),
            }
        }

        def loss_fn(p):
            """Compute sum of squared outputs as a scalar loss."""
            return jnp.sum(layer.apply(p, x) ** 2)

        grads = jax.grad(loss_fn)(nonzero_lora_b)
        assert jnp.any(grads["params"]["base"]["kernel"] != 0)
        assert jnp.any(grads["params"]["lora_a"] != 0)
        assert jnp.any(grads["params"]["lora_b"] != 0)

    def test_dropout_active_in_training(self, rng_key):
        """With dropout > 0 and deterministic=False, LoRA should use dropout."""
        layer = LoRALinear(
            out_features=16, rank=4, alpha=8.0,
            dropout_rate=0.5, deterministic=False,
        )
        x = jax.random.normal(rng_key, (1, 8, 32))
        params = layer.init({"params": rng_key, "dropout": rng_key}, x)

        nonzero_b = {
            "params": {
                **params["params"],
                "lora_b": jax.random.normal(rng_key, params["params"]["lora_b"].shape),
            }
        }

        out1 = layer.apply(nonzero_b, x, rngs={"dropout": jax.random.PRNGKey(0)})
        out2 = layer.apply(nonzero_b, x, rngs={"dropout": jax.random.PRNGKey(1)})
        assert not jnp.allclose(out1, out2, atol=1e-6)


class TestLoRAAdapter:
    """Tests for the standalone LoRA adapter."""

    def test_output_shape(self, rng_key):
        """LoRAAdapter should produce output of shape (..., out_features)."""
        adapter = LoRAAdapter(
            in_features=128, out_features=64, rank=8, alpha=16.0
        )
        x = jax.random.normal(rng_key, (2, 16, 128))
        params = adapter.init(rng_key, x)
        output = adapter.apply(params, x)
        assert output.shape == (2, 16, 64)

    def test_zero_rank_returns_zeros(self, rng_key):
        """With rank=0, LoRAAdapter should return zeros."""
        adapter = LoRAAdapter(
            in_features=32, out_features=16, rank=0
        )
        x = jax.random.normal(rng_key, (1, 4, 32))
        params = adapter.init(rng_key, x)
        output = adapter.apply(params, x)
        assert jnp.allclose(output, jnp.zeros((1, 4, 16)))


class TestLoRALayerSet:
    """Tests for the complete LoRA adapter set for one layer-depth pair."""

    def test_all_adapters_initialized(self, rng_key, small_config):
        """Relaxed model with LoRA should have per-loop LoRA adapter params."""
        from src.model.relaxed_recursive_transformer import RelaxedRecursiveTransformer

        model = RelaxedRecursiveTransformer.from_config(small_config)
        dummy_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, dummy_ids)

        lora_key = "lora_loop_0_layer_0"
        assert lora_key in params["params"]
        adapter_params = params["params"][lora_key]
        expected_adapters = {
            "query_lora", "key_lora", "value_lora", "output_lora",
            "gate_lora", "up_lora", "down_lora",
        }
        assert expected_adapters.issubset(set(adapter_params.keys()))


class TestSVDInit:
    """Tests for SVD-based LoRA initialization."""

    def test_full_rank_recovery(self):
        """At full rank, SVD init should exactly recover the residual matrix."""
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        original = jax.random.normal(k1, (64, 32))
        shared = jax.random.normal(k2, (64, 32))
        full_rank = min(64, 32)

        lora_a, lora_b = init_lora_from_svd(original, shared, rank=full_rank)
        reconstructed = lora_b.T @ lora_a
        residual = original - shared

        assert jnp.allclose(reconstructed, residual, atol=1e-4)

    def test_truncated_rank_has_error(self):
        """Truncated rank should have nonzero reconstruction error."""
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        original = jax.random.normal(k1, (64, 32))
        shared = jax.random.normal(k2, (64, 32))

        lora_a, lora_b = init_lora_from_svd(original, shared, rank=4)
        reconstructed = lora_b.T @ lora_a
        residual = original - shared

        error = jnp.mean((reconstructed - residual) ** 2)
        assert error > 1e-3

    def test_output_shapes(self):
        """A and B matrices should have correct shapes."""
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        original = jax.random.normal(k1, (128, 64))
        shared = jax.random.normal(k2, (128, 64))

        lora_a, lora_b = init_lora_from_svd(original, shared, rank=16)
        assert lora_a.shape == (16, 64)
        assert lora_b.shape == (16, 128)
