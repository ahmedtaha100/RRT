"""KV cache for efficient autoregressive generation in recursive transformers.

Provides a cache data structure that stores key and value tensors indexed by
both layer position and recursive loop depth, enabling efficient generation
without recomputing attention over previous positions.
"""

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp


@dataclass
class KVCache:
    """Key-Value cache for autoregressive generation.

    Stores cached key and value tensors indexed by (layer_idx, loop_idx)
    pairs. For a recursive transformer with block_size layers and num_loops
    recursive applications, this creates block_size * num_loops cache slots.

    Attributes:
        block_size: Number of unique layers in the shared block.
        num_loops: Number of recursive applications.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per attention head.
        cache: Internal storage mapping (layer_idx, loop_idx) to
            dict with 'key' and 'value' tensors.
    """

    block_size: int
    num_loops: int
    num_kv_heads: int
    head_dim: int
    cache: dict[tuple[int, int], dict[str, jnp.ndarray]] = field(
        default_factory=dict
    )

    def update(
        self,
        layer_idx: int,
        loop_idx: int,
        new_key: jnp.ndarray,
        new_value: jnp.ndarray,
    ) -> None:
        """Append new key and value tensors to the cache for a given position.

        Args:
            layer_idx: Index of the layer within the shared block.
            loop_idx: Index of the current recursive loop.
            new_key: New key tensor of shape (batch, new_seq_len, num_kv_heads, head_dim).
            new_value: New value tensor of same shape as new_key.
        """
        cache_key = (layer_idx, loop_idx)
        if cache_key in self.cache:
            existing = self.cache[cache_key]
            self.cache[cache_key] = {
                "key": jnp.concatenate([existing["key"], new_key], axis=1),
                "value": jnp.concatenate([existing["value"], new_value], axis=1),
            }
        else:
            self.cache[cache_key] = {"key": new_key, "value": new_value}

    def get(
        self,
        layer_idx: int,
        loop_idx: int,
    ) -> Optional[dict[str, jnp.ndarray]]:
        """Retrieve cached key and value tensors for a given position.

        Args:
            layer_idx: Index of the layer within the shared block.
            loop_idx: Index of the recursive loop.

        Returns:
            Dict with 'key' and 'value' tensors, or None if not cached.
        """
        return self.cache.get((layer_idx, loop_idx))

    def reset(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
