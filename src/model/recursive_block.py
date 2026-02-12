"""Recursive transformer block with parameter sharing across depth.

Implements the core recursive block from Bae et al. (2024) Section 3.1,
where a set of block_size unique transformer layers is applied num_loops
times to simulate a deeper network with shared parameters.
"""

from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from src.model.layers import TransformerBlock


class RecursiveBlock(nn.Module):
    """A block of transformer layers applied repeatedly via parameter sharing.

    Contains block_size unique TransformerBlock layers. During forward pass,
    these layers are applied num_loops times in sequence, effectively creating
    a (block_size * num_loops)-layer deep network with only block_size unique
    parameter sets.

    For example, with block_size=6 and num_loops=3, the model has 6 unique
    layers applied 3 times for 18 effective layers (matching Gemma 2B).

    Attributes:
        block_size: Number of unique transformer layers in the shared block.
        num_loops: Number of times the shared block is applied.
        num_heads: Number of query attention heads per layer.
        num_kv_heads: Number of key-value heads for GQA.
        head_dim: Dimension per attention head.
        intermediate_dim: Feed-forward intermediate dimension.
        hidden_dim: Model hidden dimension.
        rope_theta: Base frequency for rotary embeddings.
        max_seq_len: Maximum sequence length.
        rms_norm_eps: Epsilon for RMSNorm.
    """

    block_size: int
    num_loops: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    hidden_dim: int
    rope_theta: float = 10000.0
    max_seq_len: int = 64
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply the shared block of layers num_loops times.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mask: Optional causal attention mask.

        Returns:
            Output tensor of the same shape as input after all recursive passes.
        """
        layers = [
            TransformerBlock(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                rope_theta=self.rope_theta,
                max_seq_len=self.max_seq_len,
                rms_norm_eps=self.rms_norm_eps,
                name=f"layer_{i}",
            )
            for i in range(self.block_size)
        ]

        for loop_idx in range(self.num_loops):
            for layer in layers:
                x, _ = layer(x, mask=mask)

        return x
