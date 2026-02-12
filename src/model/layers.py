"""Core transformer layer implementations in Flax.

Provides RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention
(GQA), GeGLU Feed-Forward, and a complete TransformerBlock combining pre-norm
attention and feed-forward with residual connections. Follows the Gemma
architecture as described in Bae et al. (2024).

All sub-modules use setup() instead of nn.compact so that their internal
projections (query_proj, gate_proj, etc.) are accessible as attributes. This
enables the relaxed recursive model to inject LoRA perturbations on top of
individual shared projections.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value and applies a learnable scale parameter.
    Used as pre-normalization in the Gemma architecture.

    Attributes:
        dim: Dimension of the input features to normalize.
        eps: Small constant for numerical stability.
    """

    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization to the input.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape as input.
        """
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normalized = x * jax.lax.rsqrt(variance + self.eps)
        return normalized * weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute sine and cosine tables for Rotary Position Embeddings.

    Args:
        seq_len: Maximum sequence length.
        head_dim: Dimension of each attention head (must be even).
        theta: Base frequency for the sinusoidal position encoding.

    Returns:
        Tuple of (cos_cache, sin_cache), each of shape (seq_len, head_dim).
    """
    freq_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (freq_indices / head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, inv_freq)
    cos_cache = jnp.cos(angles)
    sin_cache = jnp.sin(angles)
    cos_cache = jnp.concatenate([cos_cache, cos_cache], axis=-1)
    sin_cache = jnp.concatenate([sin_cache, sin_cache], axis=-1)
    return cos_cache, sin_cache


def apply_rotary_embedding(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """Apply rotary position embeddings to query or key tensors.

    Implements the rotation as described in Su et al. (2021), where pairs of
    dimensions are rotated by position-dependent angles.

    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim).
        cos: Cosine cache of shape (seq_len, head_dim).
        sin: Sine cache of shape (seq_len, head_dim).

    Returns:
        Tensor with rotary embeddings applied, same shape as input.
    """
    head_dim = x.shape[-1]
    x_first_half = x[..., : head_dim // 2]
    x_second_half = x[..., head_dim // 2 :]
    x_rotated = jnp.concatenate([-x_second_half, x_first_half], axis=-1)
    cos_expanded = cos[None, :, None, :]
    sin_expanded = sin[None, :, None, :]
    seq_len = x.shape[1]
    cos_expanded = cos_expanded[:, :seq_len, :, :]
    sin_expanded = sin_expanded[:, :seq_len, :, :]
    return x * cos_expanded + x_rotated * sin_expanded


class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention (GQA) support.

    Supports standard MHA (num_kv_heads == num_heads), GQA
    (num_kv_heads < num_heads), and Multi-Query Attention
    (num_kv_heads == 1). Includes optional KV cache for
    autoregressive generation.

    Sub-modules (query_proj, key_proj, value_proj, output_proj) are
    defined in setup() so they can be accessed as attributes for LoRA injection.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per attention head.
        rope_theta: Base frequency for rotary embeddings.
        max_seq_len: Maximum sequence length for RoPE cache.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    max_seq_len: int = 64

    def setup(self):
        """Initialize projection layers."""
        hidden_dim = self.num_heads * self.head_dim
        self.query_proj = nn.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="query_proj"
        )
        self.key_proj = nn.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, name="key_proj"
        )
        self.value_proj = nn.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, name="value_proj"
        )
        self.output_proj = nn.Dense(hidden_dim, use_bias=False, name="output_proj")

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        cache: Optional[dict] = None,
    ) -> tuple[jnp.ndarray, Optional[dict]]:
        """Compute multi-head attention with RoPE and optional KV caching.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mask: Optional attention mask of shape (batch, 1, seq_len, kv_len).
            cache: Optional dict with 'key' and 'value' tensors for
                autoregressive decoding.

        Returns:
            Tuple of (output, updated_cache) where output has shape
            (batch, seq_len, hidden_dim) and updated_cache is a dict or None.
        """
        batch_size, seq_len, hidden_dim = x.shape

        queries = self.query_proj(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        keys = self.key_proj(x).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        values = self.value_proj(x).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        cos, sin = build_rope_cache(self.max_seq_len, self.head_dim, self.rope_theta)

        if cache is not None:
            cache_len = cache["key"].shape[1]
            position_offset = cache_len
            cos_slice = cos[position_offset : position_offset + seq_len]
            sin_slice = sin[position_offset : position_offset + seq_len]
            queries = apply_rotary_embedding(queries, cos_slice, sin_slice)
            keys = apply_rotary_embedding(keys, cos_slice, sin_slice)
            keys = jnp.concatenate([cache["key"], keys], axis=1)
            values = jnp.concatenate([cache["value"], values], axis=1)
            updated_cache = {"key": keys, "value": values}
        else:
            cos_slice = cos[:seq_len]
            sin_slice = sin[:seq_len]
            queries = apply_rotary_embedding(queries, cos_slice, sin_slice)
            keys = apply_rotary_embedding(keys, cos_slice, sin_slice)
            updated_cache = None

        kv_group_size = self.num_heads // self.num_kv_heads
        if kv_group_size > 1:
            keys = jnp.repeat(keys, kv_group_size, axis=2)
            values = jnp.repeat(values, kv_group_size, axis=2)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attention_weights = jnp.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            attention_weights = jnp.where(mask, attention_weights, jnp.finfo(jnp.float32).min)

        attention_weights = jax.nn.softmax(attention_weights, axis=-1)

        attention_output = jnp.matmul(attention_weights, values)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )

        output = self.output_proj(attention_output)
        return output, updated_cache


class FeedForward(nn.Module):
    """GeGLU (Gated Linear Unit with GELU) Feed-Forward Network.

    Implements the gated FFN used in Gemma: gate and up projections are
    computed in parallel, combined via element-wise multiplication with
    GELU activation on the gate, then projected down.

    Sub-modules (gate_proj, up_proj, down_proj) are defined in setup()
    so they can be accessed as attributes for LoRA injection.

    Attributes:
        intermediate_dim: Dimension of the intermediate (expanded) representation.
        hidden_dim: Dimension of the input and output.
    """

    intermediate_dim: int
    hidden_dim: int

    def setup(self):
        """Initialize projection layers."""
        self.gate_proj = nn.Dense(
            self.intermediate_dim, use_bias=False, name="gate_proj"
        )
        self.up_proj = nn.Dense(
            self.intermediate_dim, use_bias=False, name="up_proj"
        )
        self.down_proj = nn.Dense(
            self.hidden_dim, use_bias=False, name="down_proj"
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply GeGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (..., hidden_dim).

        Returns:
            Output tensor of shape (..., hidden_dim).
        """
        gate = jax.nn.gelu(self.gate_proj(x), approximate=True)
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm attention and feed-forward.

    Applies pre-RMSNorm attention with residual connection, followed by
    pre-RMSNorm feed-forward with residual connection.

    Sub-modules (attn_norm, attention, ffn_norm, feed_forward) are defined
    in setup() so they can be accessed as attributes by the relaxed recursive
    model for injecting LoRA perturbations on individual projections.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads for GQA.
        head_dim: Dimension per attention head.
        intermediate_dim: Feed-forward intermediate dimension.
        hidden_dim: Model hidden dimension.
        rope_theta: Base frequency for rotary position embeddings.
        max_seq_len: Maximum sequence length.
        rms_norm_eps: Epsilon for RMSNorm.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    hidden_dim: int
    rope_theta: float = 10000.0
    max_seq_len: int = 64
    rms_norm_eps: float = 1e-6

    def setup(self):
        """Initialize sub-modules: norms, attention, and feed-forward."""
        self.attn_norm = RMSNorm(
            dim=self.hidden_dim, eps=self.rms_norm_eps, name="attn_norm"
        )
        self.attention = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            max_seq_len=self.max_seq_len,
            name="attention",
        )
        self.ffn_norm = RMSNorm(
            dim=self.hidden_dim, eps=self.rms_norm_eps, name="ffn_norm"
        )
        self.feed_forward = FeedForward(
            intermediate_dim=self.intermediate_dim,
            hidden_dim=self.hidden_dim,
            name="feed_forward",
        )

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        cache: Optional[dict] = None,
    ) -> tuple[jnp.ndarray, Optional[dict]]:
        """Apply one transformer block: pre-norm attention + pre-norm FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mask: Optional attention mask.
            cache: Optional KV cache dict for autoregressive decoding.

        Returns:
            Tuple of (output, updated_cache) where output has the same shape
            as input.
        """
        normed_for_attn = self.attn_norm(x)
        attn_output, updated_cache = self.attention(
            normed_for_attn, mask=mask, cache=cache
        )
        x = x + attn_output

        normed_for_ffn = self.ffn_norm(x)
        ffn_output = self.feed_forward(normed_for_ffn)
        x = x + ffn_output

        return x, updated_cache
