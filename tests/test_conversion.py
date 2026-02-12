"""Tests for the conversion pipeline from vanilla to recursive models."""

import jax
import jax.numpy as jnp
import pytest

from src.conversion.convert_gemma import convert_random_to_recursive
from src.model.config import get_test_config
from src.model.relaxed_recursive_transformer import (
    RelaxedRecursiveTransformer,
    VanillaTransformer,
)
from src.utils.config_utils import FullConfig, LoRAConfig


class TestConversion:
    """Tests for vanilla-to-recursive model conversion."""

    def test_average_conversion_runs(self, small_config):
        """Converting a random 4-layer model to recursive should succeed."""
        vanilla_params, recursive_params, _ = convert_random_to_recursive(
            small_config, method="average"
        )
        assert vanilla_params is not None
        assert recursive_params is not None

    def test_recursive_produces_valid_logits(self, small_config):
        """Converted recursive model should produce valid logits."""
        _, recursive_params, config = convert_random_to_recursive(
            small_config, method="average"
        )
        model = RelaxedRecursiveTransformer.from_config(config)
        dummy_ids = jnp.ones((1, 4), dtype=jnp.int32)
        logits = model.apply(recursive_params, dummy_ids)
        assert logits.shape == (1, 4, config.model.vocab_size)
        assert jnp.all(jnp.isfinite(logits))

    def test_no_lora_conversion(self):
        """Conversion with lora_rank=0 should produce a pure recursive model."""
        config = FullConfig(
            model=get_test_config().model,
            recursive=get_test_config().recursive,
            lora=LoRAConfig(rank=0, alpha=1, dropout=0.0),
            seed=42,
        )
        vanilla_params, recursive_params, _ = convert_random_to_recursive(
            config, method="average"
        )
        model = RelaxedRecursiveTransformer.from_config(config)
        dummy_ids = jnp.ones((1, 4), dtype=jnp.int32)
        logits = model.apply(recursive_params, dummy_ids)
        assert logits.shape == (1, 4, config.model.vocab_size)


class TestFullRankLoRAReconstruction:
    """Tests that full-rank LoRA conversion reconstructs the original model."""

    def _make_symmetric_config(self):
        """Create a config where all projections have the same dimensions.

        Uses num_kv_heads == num_heads (no GQA) and intermediate_dim == hidden_dim
        so every projection is (hidden_dim, hidden_dim) and a single rank value
        can be truly full-rank for all 7 projections.
        """
        from src.utils.config_utils import (
            FullConfig,
            LoRAConfig,
            ModelConfig,
            RecursiveConfig,
        )

        hidden_dim = 64
        return FullConfig(
            model=ModelConfig(
                num_layers=4,
                hidden_dim=hidden_dim,
                num_heads=4,
                num_kv_heads=4,
                intermediate_dim=hidden_dim,
                vocab_size=128,
                max_seq_len=32,
            ),
            recursive=RecursiveConfig(num_loops=2, block_size=2),
            lora=LoRAConfig(rank=hidden_dim, alpha=hidden_dim, dropout=0.0),
            seed=42,
        )

    def test_full_rank_weight_reconstruction(self):
        """At full rank, shared + LoRA should exactly recover original projection weights."""
        from src.conversion.convert_gemma import extract_vanilla_layer_weights

        config = self._make_symmetric_config()
        vanilla_params, relaxed_params, _ = convert_random_to_recursive(
            config, method="average"
        )

        rp = relaxed_params["params"]
        block_size = config.recursive.block_size

        lora_name_map = {
            "query_proj": "query_lora",
            "key_proj": "key_lora",
            "value_proj": "value_lora",
            "output_proj": "output_lora",
            "gate_proj": "gate_lora",
            "up_proj": "up_lora",
            "down_proj": "down_lora",
        }

        for loop_idx in range(config.recursive.num_loops):
            for layer_idx in range(block_size):
                orig_idx = loop_idx * block_size + layer_idx
                orig_weights = extract_vanilla_layer_weights(vanilla_params, orig_idx)

                shared_key = f"shared_layer_{layer_idx}"
                lora_key = f"lora_loop_{loop_idx}_layer_{layer_idx}"

                attn_params = rp[shared_key]["attention"]
                ff_params = rp[shared_key]["feed_forward"]
                lora_params = rp[lora_key]

                for proj_name, lora_name in lora_name_map.items():
                    if proj_name in ["query_proj", "key_proj", "value_proj", "output_proj"]:
                        shared_kernel = attn_params[proj_name]["kernel"]
                    else:
                        shared_kernel = ff_params[proj_name]["kernel"]

                    lora_a = lora_params[lora_name]["lora_a"]
                    lora_b = lora_params[lora_name]["lora_b"]
                    reconstructed_kernel = shared_kernel + (lora_a.T @ lora_b)

                    original_kernel = orig_weights[proj_name].T

                    assert jnp.allclose(reconstructed_kernel, original_kernel, atol=1e-3), (
                        f"Weight mismatch at loop={loop_idx}, layer={layer_idx}, "
                        f"proj={proj_name}: max_error="
                        f"{jnp.max(jnp.abs(reconstructed_kernel - original_kernel)):.6f}"
                    )

    def test_full_rank_logits_close_to_vanilla(self):
        """At full rank, relaxed model logits should be much closer to vanilla than recursive."""
        config = self._make_symmetric_config()
        vanilla_params, relaxed_params, _ = convert_random_to_recursive(
            config, method="average"
        )

        no_lora_config = FullConfig(
            model=config.model,
            recursive=config.recursive,
            lora=LoRAConfig(rank=0, alpha=1, dropout=0.0),
            seed=config.seed,
        )
        _, recursive_params, _ = convert_random_to_recursive(
            no_lora_config, method="average"
        )

        dummy_ids = jnp.ones((1, 8), dtype=jnp.int32)

        vanilla_model = VanillaTransformer.from_config(config)
        vanilla_logits = vanilla_model.apply(vanilla_params, dummy_ids)

        relaxed_model = RelaxedRecursiveTransformer.from_config(config)
        relaxed_logits = relaxed_model.apply(relaxed_params, dummy_ids)

        recursive_model = RelaxedRecursiveTransformer.from_config(no_lora_config)
        recursive_logits = recursive_model.apply(recursive_params, dummy_ids)

        relaxed_error = jnp.mean((relaxed_logits - vanilla_logits) ** 2)
        recursive_error = jnp.mean((recursive_logits - vanilla_logits) ** 2)

        assert relaxed_error < recursive_error, (
            f"Full-rank relaxed (MSE={relaxed_error:.6f}) should be closer to "
            f"vanilla than pure recursive (MSE={recursive_error:.6f})"
        )


class TestCheckpointRoundTrip:
    """Tests for checkpoint save/load round-trip."""

    def test_save_and_load(self, small_config, tmp_path):
        """Checkpoint save/load should preserve parameters exactly."""
        from src.utils.checkpoint import load_checkpoint, save_checkpoint

        _, recursive_params, config = convert_random_to_recursive(
            small_config, method="average"
        )

        checkpoint_dir = str(tmp_path / "test_ckpt")
        save_checkpoint(recursive_params, checkpoint_dir, config)

        restored_params, restored_config = load_checkpoint(
            checkpoint_dir, recursive_params
        )

        original_leaves = jax.tree.leaves(recursive_params)
        restored_leaves = jax.tree.leaves(restored_params)

        assert len(original_leaves) == len(restored_leaves)
        for orig, restored in zip(original_leaves, restored_leaves):
            assert jnp.allclose(orig, restored, atol=1e-6)

        assert restored_config is not None
        assert restored_config.model.hidden_dim == config.model.hidden_dim
        assert restored_config.recursive.num_loops == config.recursive.num_loops
