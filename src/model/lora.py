"""Low-Rank Adaptation (LoRA) modules for depth-wise relaxation.

Implements LoRA layers that add trainable low-rank perturbations to frozen
base weights, following Hu et al. (2021). Includes SVD-based initialization
from residual matrices as described in Bae et al. (2024) Section 3.2, which
initializes LoRA matrices via truncated SVD of the difference between original
layer weights and shared (averaged) weights.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.model.layers import (
    Attention,
    FeedForward,
    RMSNorm,
    TransformerBlock,
    apply_rotary_embedding,
    build_rope_cache,
)

LORA_A_INIT_STDDEV = 0.01


class LoRALinear(nn.Module):
    """Dense layer augmented with a Low-Rank Adaptation.

    Computes output = dense(x) + (x @ A^T) @ B * (alpha / rank).
    A has shape (rank, in_features) and B has shape (rank, out_features).

    Attributes:
        out_features: Output dimension of the base dense layer.
        rank: Rank of the LoRA decomposition.
        alpha: Scaling numerator for LoRA output.
        dropout_rate: Dropout rate applied to LoRA input.
        use_bias: Whether the base dense layer uses bias.
        deterministic: Whether to disable dropout (True during inference).
    """

    out_features: int
    rank: int
    alpha: float = 16.0
    dropout_rate: float = 0.0
    use_bias: bool = False
    deterministic: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the base dense layer plus the LoRA perturbation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base_output = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
            name="base",
        )(x)

        if self.rank <= 0:
            return base_output

        in_features = x.shape[-1]
        scaling = self.alpha / self.rank

        lora_a = self.param(
            "lora_a",
            nn.initializers.normal(stddev=LORA_A_INIT_STDDEV),
            (self.rank, in_features),
        )
        lora_b = self.param(
            "lora_b",
            nn.initializers.zeros,
            (self.rank, self.out_features),
        )

        lora_input = x
        if self.dropout_rate > 0.0 and not self.deterministic:
            lora_input = nn.Dropout(rate=self.dropout_rate)(
                lora_input, deterministic=self.deterministic
            )

        lora_output = (lora_input @ lora_a.T) @ lora_b
        return base_output + lora_output * scaling


def init_lora_from_svd(
    original_weight: jnp.ndarray,
    shared_weight: jnp.ndarray,
    rank: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize LoRA matrices from the SVD of the residual between original and shared weights.

    Computes residual = original_weight - shared_weight, then performs truncated
    SVD to obtain A and B such that B^T @ A approximates the residual.

    At full rank (rank = min(rows, cols)), this exactly recovers the residual
    matrix within floating-point precision.

    Implements the initialization described in Equation 5 of Bae et al. (2024).

    Args:
        original_weight: Weight matrix from the original (non-shared) layer,
            shape (out_features, in_features).
        shared_weight: Weight matrix from the shared (averaged) layer,
            same shape as original_weight.
        rank: Number of singular values to retain.

    Returns:
        Tuple (A, B) where A has shape (rank, in_features) and B has shape
        (rank, out_features). The LoRA reconstruction is B^T @ A.
    """
    residual = original_weight - shared_weight
    u, s, vt = jnp.linalg.svd(residual, full_matrices=False)
    lora_b = (u[:, :rank] * s[:rank]).T
    lora_a = vt[:rank, :]
    return lora_a, lora_b


class LoRAAdapter(nn.Module):
    """Standalone LoRA adapter that adds a low-rank perturbation to an input.

    Unlike LoRALinear which wraps a Dense layer, this module only computes
    the LoRA delta: (x @ A^T) @ B * (alpha / rank). It is designed to be
    applied on top of a shared base layer's output to add depth-specific
    perturbations without duplicating the base weights.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the LoRA decomposition.
        alpha: Scaling numerator.
        dropout_rate: Dropout rate on LoRA input.
        deterministic: Whether to disable dropout.
    """

    in_features: int
    out_features: int
    rank: int
    alpha: float = 16.0
    dropout_rate: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the LoRA perturbation delta.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            LoRA delta tensor of shape (..., out_features).
        """
        if self.rank <= 0:
            return jnp.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype)

        scaling = self.alpha / self.rank

        lora_a = self.param(
            "lora_a",
            nn.initializers.normal(stddev=LORA_A_INIT_STDDEV),
            (self.rank, self.in_features),
        )
        lora_b = self.param(
            "lora_b",
            nn.initializers.zeros,
            (self.rank, self.out_features),
        )

        lora_input = x
        if self.dropout_rate > 0.0 and not self.deterministic:
            lora_input = nn.Dropout(rate=self.dropout_rate)(
                lora_input, deterministic=self.deterministic
            )

        return (lora_input @ lora_a.T) @ lora_b * scaling


class LoRALayerSet(nn.Module):
    """A complete set of LoRA adapters for one transformer layer at one depth.

    Contains adapters for all 7 projections (Q, K, V, O, gate, up, down)
    in a single transformer block. These are applied as additive perturbations
    on top of the shared base layer's projections.

    Attributes:
        hidden_dim: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
        intermediate_dim: FFN intermediate dimension.
        lora_rank: Rank for all LoRA adapters.
        lora_alpha: Scaling factor.
        lora_dropout: Dropout rate.
        deterministic: Whether to disable dropout.
    """

    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    lora_rank: int
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    deterministic: bool = True

    def setup(self):
        """Initialize all 7 LoRA adapters for this layer-depth pair."""
        self.query_lora = LoRAAdapter(
            in_features=self.hidden_dim,
            out_features=self.num_heads * self.head_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="query_lora",
        )
        self.key_lora = LoRAAdapter(
            in_features=self.hidden_dim,
            out_features=self.num_kv_heads * self.head_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="key_lora",
        )
        self.value_lora = LoRAAdapter(
            in_features=self.hidden_dim,
            out_features=self.num_kv_heads * self.head_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="value_lora",
        )
        self.output_lora = LoRAAdapter(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="output_lora",
        )
        self.gate_lora = LoRAAdapter(
            in_features=self.hidden_dim,
            out_features=self.intermediate_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="gate_lora",
        )
        self.up_lora = LoRAAdapter(
            in_features=self.hidden_dim,
            out_features=self.intermediate_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="up_lora",
        )
        self.down_lora = LoRAAdapter(
            in_features=self.intermediate_dim,
            out_features=self.hidden_dim,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            deterministic=self.deterministic,
            name="down_lora",
        )
