"""Full Relaxed Recursive Transformer model.

Implements the complete model from Bae et al. (2024), including token
embedding, recursive shared blocks with optional depth-wise LoRA relaxation,
final normalization, and a language model head with tied embeddings.

The model supports three operating modes:
- Vanilla: Standard transformer with unique layers (no sharing).
- Recursive: Shared block applied multiple times (pure layer tying).
- Relaxed Recursive: Shared block with per-loop LoRA adapters for depth-wise
  specialization, initialized via truncated SVD of residual matrices.

The relaxed path uses truly shared base TransformerBlock weights (the same
Flax modules re-applied each loop) combined with per-loop LoRA adapters that
are unique to each recursion depth. This ensures parameter sharing is real
and the LoRA overhead is the only additional cost.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.model.layers import (
    RMSNorm,
    TransformerBlock,
    apply_rotary_embedding,
    build_rope_cache,
)
from src.model.lora import LoRALayerSet
from src.utils.config_utils import FullConfig

EMBEDDING_INIT_STDDEV = 0.02


class RelaxedRecursiveTransformer(nn.Module):
    """Relaxed Recursive Transformer language model.

    Uses truly shared base TransformerBlock weights across all recursive loops.
    Each loop applies the same shared layers, but optionally adds per-loop
    LoRA perturbations for depth-wise specialization.

    When lora_rank=0, this is a pure Recursive Transformer with no relaxation.
    The number of effective layers equals block_size * num_loops.

    Attributes:
        vocab_size: Size of the token vocabulary.
        hidden_dim: Model hidden dimension and embedding dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads for GQA.
        head_dim: Dimension per attention head.
        intermediate_dim: Feed-forward intermediate dimension.
        block_size: Number of unique layers in the shared block.
        num_loops: Number of recursive applications of the shared block.
        lora_rank: Rank for depth-wise LoRA modules (0 disables LoRA).
        lora_alpha: Scaling factor for LoRA outputs.
        lora_dropout: Dropout rate for LoRA inputs.
        rope_theta: Base frequency for rotary embeddings.
        max_seq_len: Maximum sequence length.
        rms_norm_eps: Epsilon for RMSNorm.
    """

    vocab_size: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    block_size: int
    num_loops: int
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    rope_theta: float = 10000.0
    max_seq_len: int = 64
    rms_norm_eps: float = 1e-6

    def setup(self):
        """Initialize shared layers, per-loop LoRA adapters, and output layers."""
        self.embedding_table = self.param(
            "embedding",
            nn.initializers.normal(stddev=EMBEDDING_INIT_STDDEV),
            (self.vocab_size, self.hidden_dim),
        )

        self.shared_layers = [
            TransformerBlock(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                rope_theta=self.rope_theta,
                max_seq_len=self.max_seq_len,
                rms_norm_eps=self.rms_norm_eps,
                name=f"shared_layer_{layer_idx}",
            )
            for layer_idx in range(self.block_size)
        ]

        if self.lora_rank > 0:
            self.lora_sets = [
                [
                    LoRALayerSet(
                        hidden_dim=self.hidden_dim,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        intermediate_dim=self.intermediate_dim,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        name=f"lora_loop_{loop_idx}_layer_{layer_idx}",
                    )
                    for layer_idx in range(self.block_size)
                ]
                for loop_idx in range(self.num_loops)
            ]

        self.final_norm = RMSNorm(
            dim=self.hidden_dim, eps=self.rms_norm_eps, name="final_norm"
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        return_all_loop_logits: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Forward pass producing logits over the vocabulary.

        Args:
            input_ids: Integer token IDs of shape (batch, seq_len).
            mask: Optional causal mask. If None, a standard causal mask
                is constructed automatically.
            return_all_loop_logits: If True, also returns intermediate logits
                after each recursive loop (used for early exit inference).

        Returns:
            If return_all_loop_logits is False: logits of shape
                (batch, seq_len, vocab_size).
            If True: tuple of (final_logits, list_of_per_loop_logits).
        """
        batch_size, seq_len = input_ids.shape

        hidden_states = jnp.take(self.embedding_table, input_ids, axis=0)
        hidden_states = hidden_states * jnp.sqrt(jnp.float32(self.hidden_dim))

        if mask is None:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            causal_mask = causal_mask[None, None, :, :]
        else:
            causal_mask = mask

        all_loop_logits = []

        for loop_idx in range(self.num_loops):
            for layer_idx in range(self.block_size):
                if self.lora_rank > 0:
                    hidden_states = self._apply_shared_layer_with_lora(
                        hidden_states,
                        causal_mask,
                        layer_idx,
                        loop_idx,
                    )
                else:
                    hidden_states, _ = self.shared_layers[layer_idx](
                        hidden_states, mask=causal_mask
                    )

            if return_all_loop_logits:
                loop_normed = self.final_norm(hidden_states)
                loop_logits = loop_normed @ self.embedding_table.T
                all_loop_logits.append(loop_logits)

        hidden_states = self.final_norm(hidden_states)
        logits = hidden_states @ self.embedding_table.T

        if return_all_loop_logits:
            return logits, all_loop_logits
        return logits

    def embed(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Convert token IDs to scaled embeddings.

        Args:
            input_ids: Integer token IDs of shape (batch, seq_len).

        Returns:
            Embedding tensor of shape (batch, seq_len, hidden_dim).
        """
        hidden_states = jnp.take(self.embedding_table, input_ids, axis=0)
        return hidden_states * jnp.sqrt(jnp.float32(self.hidden_dim))

    def apply_loop(
        self,
        hidden_states: jnp.ndarray,
        mask: jnp.ndarray,
        loop_idx: int,
    ) -> jnp.ndarray:
        """Apply one recursive loop (all block_size layers) to hidden states.

        Runs the shared layers for a single loop iteration, optionally adding
        per-loop LoRA perturbations. This method enables true early exit by
        allowing the caller to run loops individually and stop early.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_dim).
            mask: Causal attention mask.
            loop_idx: Current recursive loop index.

        Returns:
            Updated hidden states of shape (batch, seq_len, hidden_dim).
        """
        for layer_idx in range(self.block_size):
            if self.lora_rank > 0:
                hidden_states = self._apply_shared_layer_with_lora(
                    hidden_states, mask, layer_idx, loop_idx,
                )
            else:
                hidden_states, _ = self.shared_layers[layer_idx](
                    hidden_states, mask=mask
                )
        return hidden_states

    def hidden_to_logits(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Apply final normalization and project to vocabulary logits.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_dim).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        normed = self.final_norm(hidden_states)
        return normed @ self.embedding_table.T

    def _apply_shared_layer_with_lora(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        layer_idx: int,
        loop_idx: int,
    ) -> jnp.ndarray:
        """Apply a shared transformer layer with LoRA perturbations.

        Runs the shared base layer's attention and FFN, then adds per-loop
        LoRA deltas to each projection's output. The base weights are identical
        across loops; only the LoRA adapters differ.

        Args:
            x: Input hidden states of shape (batch, seq_len, hidden_dim).
            mask: Causal attention mask.
            layer_idx: Index of the shared layer within the block.
            loop_idx: Current recursive loop index.

        Returns:
            Output hidden states of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape
        layer = self.shared_layers[layer_idx]
        lora_set = self.lora_sets[loop_idx][layer_idx]

        normed_for_attn = layer.attn_norm(x)

        attn = layer.attention
        base_queries = attn.query_proj(normed_for_attn)
        base_keys = attn.key_proj(normed_for_attn)
        base_values = attn.value_proj(normed_for_attn)

        queries = base_queries + lora_set.query_lora(normed_for_attn)
        keys = base_keys + lora_set.key_lora(normed_for_attn)
        values = base_values + lora_set.value_lora(normed_for_attn)

        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = build_rope_cache(self.max_seq_len, self.head_dim, self.rope_theta)
        cos_slice = cos[:seq_len]
        sin_slice = sin[:seq_len]
        queries = apply_rotary_embedding(queries, cos_slice, sin_slice)
        keys = apply_rotary_embedding(keys, cos_slice, sin_slice)

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
            attention_weights = jnp.where(
                mask, attention_weights, jnp.finfo(jnp.float32).min
            )

        attention_weights = jax.nn.softmax(attention_weights, axis=-1)
        attention_output = jnp.matmul(attention_weights, values)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        base_attn_out = attn.output_proj(attention_output)
        attn_out = base_attn_out + lora_set.output_lora(attention_output)
        x = x + attn_out

        normed_for_ffn = layer.ffn_norm(x)
        ff = layer.feed_forward

        base_gate = ff.gate_proj(normed_for_ffn)
        base_up = ff.up_proj(normed_for_ffn)
        gate = base_gate + lora_set.gate_lora(normed_for_ffn)
        up = base_up + lora_set.up_lora(normed_for_ffn)

        activated = jax.nn.gelu(gate, approximate=True) * up

        base_down = ff.down_proj(activated)
        down = base_down + lora_set.down_lora(activated)
        x = x + down

        return x

    @classmethod
    def from_config(cls, config: FullConfig) -> "RelaxedRecursiveTransformer":
        """Create a RelaxedRecursiveTransformer from a FullConfig.

        Args:
            config: Full configuration containing model, recursive, and LoRA settings.

        Returns:
            An uninitialized RelaxedRecursiveTransformer module.
        """
        head_dim = config.model.hidden_dim // config.model.num_heads
        return cls(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_kv_heads=config.model.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=config.model.intermediate_dim,
            block_size=config.recursive.block_size,
            num_loops=config.recursive.num_loops,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            rope_theta=config.model.rope_theta,
            max_seq_len=config.model.max_seq_len,
            rms_norm_eps=config.model.rms_norm_eps,
        )


class VanillaTransformer(nn.Module):
    """Standard (non-recursive) transformer for baseline comparison.

    Each layer has unique parameters with no sharing. Used as the reference
    model before conversion to recursive form.

    Attributes:
        vocab_size: Size of the token vocabulary.
        hidden_dim: Model hidden dimension.
        num_layers: Total number of unique decoder layers.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads for GQA.
        head_dim: Dimension per attention head.
        intermediate_dim: Feed-forward intermediate dimension.
        rope_theta: Base frequency for rotary embeddings.
        max_seq_len: Maximum sequence length.
        rms_norm_eps: Epsilon for RMSNorm.
    """

    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    rope_theta: float = 10000.0
    max_seq_len: int = 64
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Forward pass producing logits over the vocabulary.

        Args:
            input_ids: Integer token IDs of shape (batch, seq_len).
            mask: Optional causal mask.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape

        embedding_table = self.param(
            "embedding",
            nn.initializers.normal(stddev=EMBEDDING_INIT_STDDEV),
            (self.vocab_size, self.hidden_dim),
        )

        hidden_states = jnp.take(embedding_table, input_ids, axis=0)
        hidden_states = hidden_states * jnp.sqrt(jnp.float32(self.hidden_dim))

        if mask is None:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            causal_mask = causal_mask[None, None, :, :]
        else:
            causal_mask = mask

        for layer_idx in range(self.num_layers):
            layer = TransformerBlock(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                rope_theta=self.rope_theta,
                max_seq_len=self.max_seq_len,
                rms_norm_eps=self.rms_norm_eps,
                name=f"layer_{layer_idx}",
            )
            hidden_states, _ = layer(hidden_states, mask=causal_mask)

        final_norm = RMSNorm(
            dim=self.hidden_dim, eps=self.rms_norm_eps, name="final_norm"
        )
        hidden_states = final_norm(hidden_states)

        logits = hidden_states @ embedding_table.T

        return logits

    @classmethod
    def from_config(cls, config: FullConfig) -> "VanillaTransformer":
        """Create a VanillaTransformer from a FullConfig.

        Args:
            config: Full configuration containing model settings.

        Returns:
            An uninitialized VanillaTransformer module.
        """
        head_dim = config.model.hidden_dim // config.model.num_heads
        return cls(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            num_kv_heads=config.model.num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=config.model.intermediate_dim,
            rope_theta=config.model.rope_theta,
            max_seq_len=config.model.max_seq_len,
            rms_norm_eps=config.model.rms_norm_eps,
        )
