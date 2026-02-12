"""Tests for early exit inference and depth-wise batching simulation."""

import jax
import jax.numpy as jnp
import pytest

from src.inference.depth_wise_batching import (
    format_batching_comparison,
    simulate_depth_wise_batching,
)
from src.inference.early_exit import EarlyExitStats, early_exit_generate
from src.model.config import get_test_config
from src.model.relaxed_recursive_transformer import RelaxedRecursiveTransformer


class TestEarlyExit:
    """Tests for early exit generation."""

    def test_full_recursion_output_shape(self, rng_key, small_config):
        """With threshold=0.0, generation should produce correct output shape."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        max_new = 4
        generated, stats = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=max_new,
            confidence_threshold=0.0,
        )

        assert generated.shape == (1, 8 + max_new)
        assert stats.total_tokens == max_new

    def test_threshold_zero_always_full_depth(self, rng_key, small_config):
        """With threshold=0.0, all tokens should exit at the final loop."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        max_new = 4
        _, stats = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=max_new,
            confidence_threshold=0.0,
        )

        assert all(d == small_config.recursive.num_loops - 1 for d in stats.exit_depths)
        assert stats.early_exit_rate == 0.0

    def test_threshold_zero_matches_normal_forward(self, rng_key, small_config):
        """With threshold=0.0, generated tokens should match a normal forward pass."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        generated_ee, _ = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=4,
            confidence_threshold=0.0,
        )

        generated_normal = input_ids
        for _ in range(4):
            logits = model.apply(params, generated_normal)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_normal = jnp.concatenate([generated_normal, next_token], axis=1)

        assert jnp.array_equal(generated_ee, generated_normal)

    def test_exit_stats_tracking(self, rng_key, small_config):
        """Exit statistics should be tracked correctly."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        max_new = 8
        _, stats = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=max_new,
            confidence_threshold=0.01,
        )

        assert len(stats.exit_depths) == max_new
        assert stats.total_tokens == max_new
        assert stats.num_loops == small_config.recursive.num_loops

    def test_generated_tokens_are_valid(self, rng_key, small_config):
        """Generated tokens should be within vocabulary range."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 4), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        generated, _ = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=4,
            confidence_threshold=0.01,
        )

        new_tokens = generated[:, 4:]
        assert jnp.all(new_tokens >= 0)
        assert jnp.all(new_tokens < small_config.model.vocab_size)

    def test_early_exit_uses_loop_logits(self, rng_key, small_config):
        """With low threshold, early exit should consider per-loop logits."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        _, stats = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=4,
            confidence_threshold=0.01,
        )

        assert all(0 <= d < small_config.recursive.num_loops for d in stats.exit_depths)

    def test_high_threshold_prevents_early_exit(self, rng_key, small_config):
        """With threshold near 1.0, early exit should rarely or never trigger."""
        model = RelaxedRecursiveTransformer.from_config(small_config)
        input_ids = jax.random.randint(rng_key, (1, 8), 0, small_config.model.vocab_size)
        params = model.init(rng_key, input_ids)

        _, stats = early_exit_generate(
            model,
            params,
            input_ids,
            small_config,
            max_new_tokens=4,
            confidence_threshold=0.9999,
        )

        assert stats.mean_exit_depth >= small_config.recursive.num_loops - 1 - 0.5


class TestEarlyExitStats:
    """Tests for the EarlyExitStats dataclass."""

    def test_mean_exit_depth(self):
        """Mean exit depth should be computed correctly."""
        stats = EarlyExitStats(exit_depths=[0, 1, 2, 3], total_tokens=4, num_loops=4)
        assert stats.mean_exit_depth == 1.5

    def test_early_exit_rate(self):
        """Early exit rate should count tokens exiting before final loop."""
        stats = EarlyExitStats(exit_depths=[0, 0, 1, 3], total_tokens=4, num_loops=4)
        assert stats.early_exit_rate == 0.75

    def test_empty_stats(self):
        """Empty stats should return zero values."""
        stats = EarlyExitStats()
        assert stats.mean_exit_depth == 0.0
        assert stats.early_exit_rate == 0.0


class TestDepthWiseBatching:
    """Tests for depth-wise batching simulation."""

    def test_simulation_runs(self):
        """Simulation should produce valid results."""
        stats = EarlyExitStats(
            exit_depths=[0, 1, 1, 2, 2, 2, 3, 3],
            total_tokens=8,
            num_loops=4,
        )
        result = simulate_depth_wise_batching(stats, batch_size=4, num_loops=4)
        assert result.theoretical_speedup >= 1.0
        assert sum(result.tokens_per_depth) == 8

    def test_all_full_depth_no_speedup(self):
        """When all tokens use full depth, speedup should be 1.0."""
        stats = EarlyExitStats(
            exit_depths=[3, 3, 3, 3],
            total_tokens=4,
            num_loops=4,
        )
        result = simulate_depth_wise_batching(stats, batch_size=2, num_loops=4)
        assert abs(result.theoretical_speedup - 1.0) < 0.01

    def test_format_comparison(self):
        """Formatting should produce a non-empty markdown table."""
        stats = EarlyExitStats(
            exit_depths=[0, 1, 2, 3],
            total_tokens=4,
            num_loops=4,
        )
        result = simulate_depth_wise_batching(stats, batch_size=2, num_loops=4)
        table = format_batching_comparison(result, num_loops=4)
        assert "Speedup" in table
        assert "Depth" in table
