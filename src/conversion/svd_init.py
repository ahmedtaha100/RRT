"""SVD-based LoRA initialization for relaxed recursive transformers.

Computes truncated SVD of residual matrices (original - shared weights) to
initialize LoRA adapters, as described in Bae et al. (2024) Equation 5.
This allows the relaxed recursive model to approximate the original vanilla
model's behavior through low-rank depth-wise perturbations.
"""

import jax.numpy as jnp

from src.model.lora import init_lora_from_svd


PROJECTION_NAMES = [
    "query_proj",
    "key_proj",
    "value_proj",
    "output_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def compute_lora_init_for_layer(
    original_layer_weights: dict[str, jnp.ndarray],
    shared_layer_weights: dict[str, jnp.ndarray],
    rank: int,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    """Compute SVD-initialized LoRA matrices for all projections in a single layer.

    For each weight matrix (Q, K, V, O, gate, up, down projections), computes
    the residual between the original and shared weights, then performs truncated
    SVD to obtain LoRA A and B matrices.

    Args:
        original_layer_weights: Weight dict for the original (non-shared) layer.
            Keys are projection names (e.g., 'query_proj'), values are weight
            matrices in (out_features, in_features) format.
        shared_layer_weights: Weight dict for the shared (averaged) layer,
            same structure as original_layer_weights.
        rank: Rank for the truncated SVD decomposition.

    Returns:
        Dictionary mapping projection names to (A, B) tuples where A has shape
        (rank, in_features) and B has shape (rank, out_features).
    """
    lora_params = {}
    for proj_name in PROJECTION_NAMES:
        if proj_name in original_layer_weights and proj_name in shared_layer_weights:
            lora_a, lora_b = init_lora_from_svd(
                original_layer_weights[proj_name],
                shared_layer_weights[proj_name],
                rank=rank,
            )
            lora_params[proj_name] = (lora_a, lora_b)
    return lora_params


def compute_lora_init(
    original_layers: list[dict[str, jnp.ndarray]],
    shared_layers: list[dict[str, jnp.ndarray]],
    rank: int,
) -> list[dict[str, tuple[jnp.ndarray, jnp.ndarray]]]:
    """Compute SVD-initialized LoRA matrices for all layers across all loops.

    For each (original, shared) layer pair, computes the per-projection LoRA
    initialization via truncated SVD of the residual matrix.

    Args:
        original_layers: List of weight dicts for all original vanilla layers.
        shared_layers: List of weight dicts for the corresponding shared layers
            (the shared layer is repeated for each loop, so this list has the
            same length as original_layers).
        rank: Rank for the truncated SVD decomposition.

    Returns:
        List of dictionaries, one per layer, mapping projection names to
        (A, B) tuples.
    """
    all_lora_params = []
    for original, shared in zip(original_layers, shared_layers):
        layer_lora = compute_lora_init_for_layer(original, shared, rank)
        all_lora_params.append(layer_lora)
    return all_lora_params
