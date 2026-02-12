"""Efficiency analysis and parameter counting for model comparison.

Provides utilities for counting parameters, estimating FLOPs, and generating
comparison tables between vanilla, recursive, and relaxed recursive models.
"""

import jax
import jax.numpy as jnp

from src.utils.config_utils import FullConfig


def count_parameters(params: dict) -> dict[str, int]:
    """Count total, shared, and LoRA parameters in a model.

    Args:
        params: Model parameter dictionary.

    Returns:
        Dictionary with keys 'total', 'shared', 'lora', 'embedding',
        and 'overhead_percent'.
    """
    total = 0
    lora_count = 0
    embedding_count = 0

    leaves_with_path = list(_flatten_with_path(params))

    for path, leaf in leaves_with_path:
        size = leaf.size
        total += size
        path_str = "/".join(str(p) for p in path)
        if "lora_a" in path_str or "lora_b" in path_str:
            lora_count += size
        if "embedding" in path_str and "norm" not in path_str:
            embedding_count += size

    shared_count = total - lora_count - embedding_count

    if shared_count > 0:
        overhead_percent = (lora_count / shared_count) * 100
    else:
        overhead_percent = 0.0

    return {
        "total": total,
        "shared": shared_count,
        "lora": lora_count,
        "embedding": embedding_count,
        "overhead_percent": round(overhead_percent, 2),
    }


def _flatten_with_path(
    tree: dict,
    prefix: tuple = (),
) -> list[tuple[tuple, jnp.ndarray]]:
    """Recursively flatten a nested dict, tracking the key path.

    Args:
        tree: Nested dictionary to flatten.
        prefix: Current path prefix.

    Returns:
        List of (path_tuple, leaf_array) pairs.
    """
    results = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            results.extend(_flatten_with_path(value, prefix + (key,)))
    elif hasattr(tree, "shape"):
        results.append((prefix, tree))
    return results


def estimate_flops(config: FullConfig) -> dict[str, dict[str, float]]:
    """Estimate forward pass FLOPs for vanilla, recursive, and relaxed models.

    Computes approximate FLOPs based on matrix multiplication operations
    in attention and feed-forward layers.

    Args:
        config: Full model configuration.

    Returns:
        Dictionary mapping model variant names to dicts with 'attention_flops',
        'ffn_flops', 'total_flops', and 'relative' (normalized to vanilla).
    """
    hidden = config.model.hidden_dim
    inter = config.model.intermediate_dim
    seq = config.model.max_seq_len
    heads = config.model.num_heads
    kv_heads = config.model.num_kv_heads
    head_dim = hidden // heads
    num_layers = config.model.num_layers
    block_size = config.recursive.block_size
    num_loops = config.recursive.num_loops
    lora_rank = config.lora.rank

    qkvo_flops = 2 * seq * hidden * (heads * head_dim + 2 * kv_heads * head_dim + hidden)
    attn_score_flops = 2 * seq * seq * heads * head_dim
    attn_flops_per_layer = qkvo_flops + attn_score_flops

    ffn_flops_per_layer = 2 * seq * hidden * inter * 3

    flops_per_layer = attn_flops_per_layer + ffn_flops_per_layer

    vanilla_total = flops_per_layer * num_layers

    recursive_total = flops_per_layer * block_size * num_loops

    lora_flops_per_proj = 2 * seq * lora_rank * hidden * 2
    lora_flops_per_layer = lora_flops_per_proj * 7
    relaxed_total = recursive_total + lora_flops_per_layer * block_size * num_loops

    results = {
        "vanilla": {
            "attention_flops": attn_flops_per_layer * num_layers,
            "ffn_flops": ffn_flops_per_layer * num_layers,
            "total_flops": vanilla_total,
            "relative": 1.0,
        },
        "recursive": {
            "attention_flops": attn_flops_per_layer * block_size * num_loops,
            "ffn_flops": ffn_flops_per_layer * block_size * num_loops,
            "total_flops": recursive_total,
            "relative": recursive_total / vanilla_total if vanilla_total > 0 else 0.0,
        },
        "relaxed_recursive": {
            "attention_flops": attn_flops_per_layer * block_size * num_loops,
            "ffn_flops": ffn_flops_per_layer * block_size * num_loops,
            "lora_flops": lora_flops_per_layer * block_size * num_loops,
            "total_flops": relaxed_total,
            "relative": relaxed_total / vanilla_total if vanilla_total > 0 else 0.0,
        },
    }

    return results


def generate_comparison_table(
    configs: dict[str, FullConfig],
    params_dict: dict[str, dict],
) -> str:
    """Generate a markdown comparison table for multiple model variants.

    Args:
        configs: Dictionary mapping variant names to their FullConfig.
        params_dict: Dictionary mapping variant names to their parameters.

    Returns:
        Formatted markdown table string comparing parameters and FLOPs.
    """
    lines = [
        "| Model | Total Params | Shared Params | LoRA Params | LoRA Overhead % |",
        "|-------|-------------|---------------|-------------|-----------------|",
    ]

    for name, params in params_dict.items():
        counts = count_parameters(params)
        lines.append(
            f"| {name} | {counts['total']:,} | {counts['shared']:,} "
            f"| {counts['lora']:,} | {counts['overhead_percent']:.1f}% |"
        )

    return "\n".join(lines)
