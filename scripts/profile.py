"""Profiling script for comparing model variants.

Generates a comprehensive comparison of parameter counts, FLOPs, and
memory usage across vanilla, recursive, and relaxed recursive models.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Profile and compare model variants."""
    parser = argparse.ArgumentParser(
        description="Profile relaxed recursive transformer models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_small.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from src.conversion.convert_gemma import convert_random_to_recursive
    from src.evaluation.efficiency import count_parameters, estimate_flops
    from src.model.relaxed_recursive_transformer import (
        RelaxedRecursiveTransformer,
        VanillaTransformer,
    )
    from src.utils.config_utils import FullConfig, LoRAConfig, load_config

    config = load_config(args.config)
    rng = jax.random.PRNGKey(config.seed)
    dummy_ids = jnp.ones((1, config.model.max_seq_len), dtype=jnp.int32)

    print("=" * 70)
    print("Model Profiling — Relaxed Recursive Transformers")
    print("=" * 70)
    print(f"\nConfig: {args.config}")
    print(f"  Layers: {config.model.num_layers}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Heads: {config.model.num_heads} (KV: {config.model.num_kv_heads})")
    print(f"  Block size: {config.recursive.block_size}")
    print(f"  Num loops: {config.recursive.num_loops}")
    print(f"  LoRA rank: {config.lora.rank}")

    print("\n--- Vanilla Transformer ---")
    vanilla_model = VanillaTransformer.from_config(config)
    vanilla_params = vanilla_model.init(rng, dummy_ids)
    vanilla_counts = count_parameters(vanilla_params)

    start = time.perf_counter()
    vanilla_logits = vanilla_model.apply(vanilla_params, dummy_ids)
    vanilla_logits.block_until_ready()
    vanilla_time = time.perf_counter() - start
    print(f"  Parameters: {vanilla_counts['total']:,}")
    print(f"  Forward time: {vanilla_time * 1000:.1f} ms")

    print("\n--- Recursive Transformer (no LoRA) ---")
    no_lora_config = FullConfig(
        model=config.model,
        recursive=config.recursive,
        lora=LoRAConfig(rank=0, alpha=1, dropout=0.0),
        seed=config.seed,
    )
    _, recursive_params, _ = convert_random_to_recursive(
        no_lora_config, method="average"
    )
    recursive_model = RelaxedRecursiveTransformer.from_config(no_lora_config)
    recursive_counts = count_parameters(recursive_params)

    start = time.perf_counter()
    recursive_logits = recursive_model.apply(recursive_params, dummy_ids)
    recursive_logits.block_until_ready()
    recursive_time = time.perf_counter() - start
    print(f"  Parameters: {recursive_counts['total']:,}")
    print(f"  Forward time: {recursive_time * 1000:.1f} ms")
    print(
        f"  Parameter reduction: "
        f"{(1 - recursive_counts['total'] / vanilla_counts['total']) * 100:.1f}%"
    )

    print("\n--- Relaxed Recursive Transformer (with LoRA) ---")
    _, relaxed_params, _ = convert_random_to_recursive(config, method="average")
    relaxed_model = RelaxedRecursiveTransformer.from_config(config)
    relaxed_counts = count_parameters(relaxed_params)

    start = time.perf_counter()
    relaxed_logits = relaxed_model.apply(relaxed_params, dummy_ids)
    relaxed_logits.block_until_ready()
    relaxed_time = time.perf_counter() - start
    print(f"  Parameters: {relaxed_counts['total']:,}")
    print(f"  LoRA parameters: {relaxed_counts['lora']:,}")
    print(f"  LoRA overhead: {relaxed_counts['overhead_percent']:.1f}%")
    print(f"  Forward time: {relaxed_time * 1000:.1f} ms")

    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(
        f"\n  {'Model':<30} {'Params':>12} {'LoRA':>10} "
        f"{'Overhead':>10} {'Time (ms)':>10}"
    )
    print(f"  {'-' * 72}")
    print(
        f"  {'Vanilla':<30} {vanilla_counts['total']:>12,} "
        f"{'—':>10} {'—':>10} {vanilla_time * 1000:>10.1f}"
    )
    print(
        f"  {'Recursive':<30} {recursive_counts['total']:>12,} "
        f"{'0':>10} {'0.0%':>10} {recursive_time * 1000:>10.1f}"
    )
    print(
        f"  {'Relaxed Recursive':<30} {relaxed_counts['total']:>12,} "
        f"{relaxed_counts['lora']:>10,} "
        f"{relaxed_counts['overhead_percent']:>9.1f}% "
        f"{relaxed_time * 1000:>10.1f}"
    )

    print("\n" + "=" * 70)
    print("FLOPs Comparison")
    print("=" * 70)
    flops = estimate_flops(config)
    for variant, data in flops.items():
        print(
            f"  {variant:<30} {data['total_flops']:>18,.0f} "
            f"({data['relative']:.3f}x)"
        )

    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
