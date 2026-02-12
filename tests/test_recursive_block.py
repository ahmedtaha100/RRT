"""Tests for the recursive block and full relaxed recursive transformer."""

import jax
import jax.numpy as jnp
import pytest

from src.model.config import get_test_config
from src.model.recursive_block import RecursiveBlock
from src.model.relaxed_recursive_transformer import (
    RelaxedRecursiveTransformer,
    VanillaTransformer,
)


class TestRecursiveBlock:
    """Tests for the RecursiveBlock module."""

    def test_output_shape(self, rng_key, dummy_hidden, small_config):
        """RecursiveBlock output should match input shape."""
        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        block = RecursiveBlock(
            block_size=small_config.recursive.block_size,
            num_loops=small_config.recursive.num_loops,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
            max_seq_len=cfg.max_seq_len,
        )
        params = block.init(rng_key, dummy_hidden)
        output = block.apply(params, dummy_hidden)
        assert output.shape == dummy_hidden.shape

    def test_single_loop_equals_manual_layer_pass(self, rng_key, small_config):
        """With num_loops=1, RecursiveBlock output should exactly match manually
        applying each shared layer once in sequence."""
        from src.model.layers import TransformerBlock

        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        block_size = small_config.recursive.block_size
        x = jax.random.normal(rng_key, (1, 8, cfg.hidden_dim))

        block = RecursiveBlock(
            block_size=block_size,
            num_loops=1,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
            max_seq_len=cfg.max_seq_len,
        )

        params = block.init(rng_key, x)
        block_output = block.apply(params, x)

        h = x
        for i in range(block_size):
            layer = TransformerBlock(
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=head_dim,
                intermediate_dim=cfg.intermediate_dim,
                hidden_dim=cfg.hidden_dim,
                max_seq_len=cfg.max_seq_len,
                name=f"layer_{i}",
            )
            layer_params = {"params": params["params"][f"layer_{i}"]}
            h, _ = layer.apply(layer_params, h)

        assert jnp.allclose(block_output, h, atol=1e-5), (
            f"num_loops=1 output does not match manual single pass: "
            f"max_error={jnp.max(jnp.abs(block_output - h)):.6f}"
        )

    def test_multiple_loops_differ_from_single(self, rng_key, small_config):
        """With num_loops=2, output should differ from num_loops=1 (shared weights
        are re-applied, producing a different result)."""
        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        x = jax.random.normal(rng_key, (1, 8, cfg.hidden_dim))

        block_single = RecursiveBlock(
            block_size=small_config.recursive.block_size,
            num_loops=1,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
            max_seq_len=cfg.max_seq_len,
        )

        block_double = RecursiveBlock(
            block_size=small_config.recursive.block_size,
            num_loops=2,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
            max_seq_len=cfg.max_seq_len,
        )

        params = block_single.init(rng_key, x)
        output_single = block_single.apply(params, x)
        output_double = block_double.apply(params, x)

        assert not jnp.allclose(output_single, output_double, atol=1e-5)


class TestRelaxedRecursiveTransformer:
    """Tests for the full RelaxedRecursiveTransformer model."""

    def test_logits_shape(self, rng_key, dummy_input, small_config):
        """Model should produce logits of shape (batch, seq_len, vocab_size)."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        params = model.init(rng_key, dummy_input)
        logits = model.apply(params, dummy_input)
        batch, seq_len = dummy_input.shape
        assert logits.shape == (batch, seq_len, small_config.model.vocab_size)

    def test_no_lora_variant(self, rng_key, dummy_input, small_config):
        """Model with lora_rank=0 should work as pure recursive transformer."""
        from src.utils.config_utils import FullConfig, LoRAConfig

        no_lora_config = FullConfig(
            model=small_config.model,
            recursive=small_config.recursive,
            lora=LoRAConfig(rank=0, alpha=1, dropout=0.0),
            seed=42,
        )
        model = RelaxedRecursiveTransformer.from_config(no_lora_config)
        params = model.init(rng_key, dummy_input)
        logits = model.apply(params, dummy_input)
        batch, seq_len = dummy_input.shape
        assert logits.shape == (batch, seq_len, small_config.model.vocab_size)

    def test_recursive_has_fewer_params(self, rng_key, dummy_input, small_config):
        """Recursive model should have fewer parameters than vanilla with same depth."""
        from src.utils.config_utils import FullConfig, LoRAConfig

        no_lora_config = FullConfig(
            model=small_config.model,
            recursive=small_config.recursive,
            lora=LoRAConfig(rank=0, alpha=1, dropout=0.0),
            seed=42,
        )

        recursive_model = RelaxedRecursiveTransformer.from_config(no_lora_config)
        recursive_params = recursive_model.init(rng_key, dummy_input)
        recursive_count = sum(
            p.size for p in jax.tree.leaves(recursive_params)
        )

        vanilla_model = VanillaTransformer.from_config(small_config)
        vanilla_params = vanilla_model.init(rng_key, dummy_input)
        vanilla_count = sum(
            p.size for p in jax.tree.leaves(vanilla_params)
        )

        assert recursive_count < vanilla_count


class TestVanillaTransformer:
    """Tests for the baseline VanillaTransformer."""

    def test_logits_shape(self, rng_key, dummy_input, small_config):
        """Vanilla model should produce correctly shaped logits."""
        model = VanillaTransformer.from_config(small_config)
        params = model.init(rng_key, dummy_input)
        logits = model.apply(params, dummy_input)
        batch, seq_len = dummy_input.shape
        assert logits.shape == (batch, seq_len, small_config.model.vocab_size)
