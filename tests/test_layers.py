"""Tests for core transformer layer implementations."""

import jax
import jax.numpy as jnp
import pytest

from src.model.layers import (
    Attention,
    FeedForward,
    RMSNorm,
    TransformerBlock,
    apply_rotary_embedding,
    build_rope_cache,
)


class TestRMSNorm:
    """Tests for the RMSNorm module."""

    def test_output_shape(self, rng_key, dummy_hidden):
        """RMSNorm output should have the same shape as input."""
        norm = RMSNorm(dim=128)
        params = norm.init(rng_key, dummy_hidden)
        output = norm.apply(params, dummy_hidden)
        assert output.shape == dummy_hidden.shape

    def test_normalization_property(self, rng_key, dummy_hidden):
        """After RMSNorm, the RMS of each vector should be approximately 1."""
        norm = RMSNorm(dim=128)
        params = norm.init(rng_key, dummy_hidden)
        output = norm.apply(params, dummy_hidden)
        rms_values = jnp.sqrt(jnp.mean(jnp.square(output), axis=-1))
        assert jnp.allclose(rms_values, 1.0, atol=0.1)


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_cache_shape(self):
        """RoPE cache should have shape (seq_len, head_dim)."""
        cos, sin = build_rope_cache(seq_len=64, head_dim=32)
        assert cos.shape == (64, 32)
        assert sin.shape == (64, 32)

    def test_apply_preserves_shape(self):
        """Applying RoPE should not change the tensor shape."""
        batch, seq_len, num_heads, head_dim = 2, 16, 4, 32
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, num_heads, head_dim))
        cos, sin = build_rope_cache(seq_len, head_dim)
        output = apply_rotary_embedding(x, cos, sin)
        assert output.shape == x.shape


class TestAttention:
    """Tests for the multi-head attention module."""

    def test_output_shape_standard(self, rng_key, dummy_hidden, small_config):
        """Attention output should match input shape (batch, seq, hidden)."""
        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        attn = Attention(
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_heads,
            head_dim=head_dim,
            max_seq_len=cfg.max_seq_len,
        )
        params = attn.init(rng_key, dummy_hidden)
        output, _ = attn.apply(params, dummy_hidden)
        assert output.shape == dummy_hidden.shape

    def test_output_shape_gqa(self, rng_key, dummy_hidden, small_config):
        """Attention with GQA (num_kv_heads=1) should match input shape."""
        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        attn = Attention(
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=cfg.max_seq_len,
        )
        params = attn.init(rng_key, dummy_hidden)
        output, _ = attn.apply(params, dummy_hidden)
        assert output.shape == dummy_hidden.shape


class TestFeedForward:
    """Tests for the GeGLU feed-forward module."""

    def test_output_shape(self, rng_key, dummy_hidden, small_config):
        """FeedForward output should have shape (..., hidden_dim)."""
        cfg = small_config.model
        ffn = FeedForward(
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
        )
        params = ffn.init(rng_key, dummy_hidden)
        output = ffn.apply(params, dummy_hidden)
        assert output.shape == dummy_hidden.shape


class TestTransformerBlock:
    """Tests for the complete transformer block."""

    def test_output_shape(self, rng_key, dummy_hidden, small_config):
        """TransformerBlock output should match input shape."""
        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        block = TransformerBlock(
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
            rope_theta=cfg.rope_theta,
            max_seq_len=cfg.max_seq_len,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        params = block.init(rng_key, dummy_hidden)
        output, _ = block.apply(params, dummy_hidden)
        assert output.shape == dummy_hidden.shape

    def test_residual_connection(self, rng_key, small_config):
        """With zero-initialized weights, output should equal input (residual only)."""
        cfg = small_config.model
        head_dim = cfg.hidden_dim // cfg.num_heads
        block = TransformerBlock(
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=cfg.intermediate_dim,
            hidden_dim=cfg.hidden_dim,
        )
        x = jnp.ones((1, 4, cfg.hidden_dim))
        params = block.init(rng_key, x)
        zero_params = jax.tree.map(jnp.zeros_like, params)
        norm_param_path_attn = ("params", "attn_norm", "weight")
        norm_param_path_ffn = ("params", "ffn_norm", "weight")

        def set_nested(d, keys, val):
            """Recursively set a nested dictionary value."""
            if len(keys) == 1:
                d[keys[0]] = val
                return d
            d[keys[0]] = set_nested(dict(d[keys[0]]), keys[1:], val)
            return d

        zero_params = dict(zero_params)
        zero_params = set_nested(
            zero_params,
            norm_param_path_attn,
            jnp.ones(cfg.hidden_dim),
        )
        zero_params = set_nested(
            zero_params,
            norm_param_path_ffn,
            jnp.ones(cfg.hidden_dim),
        )

        output, _ = block.apply(zero_params, x)
        assert jnp.allclose(output, x, atol=1e-5)
