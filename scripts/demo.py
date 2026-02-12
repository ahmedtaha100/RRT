"""Demo script for Relaxed Recursive Transformers.

Creates a small random model (or loads a checkpoint), converts it to relaxed
recursive form, runs inference with text generation, and prints parameter
comparisons and exit statistics. Requires no downloads or external data —
works entirely on CPU.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from src.conversion.convert_gemma import convert_random_to_recursive
from src.evaluation.efficiency import count_parameters, estimate_flops
from src.inference.depth_wise_batching import (
    format_batching_comparison,
    simulate_depth_wise_batching,
)
from src.inference.early_exit import early_exit_generate
from src.model.config import get_test_config
from src.model.relaxed_recursive_transformer import (
    RelaxedRecursiveTransformer,
    VanillaTransformer,
)
from src.utils.config_utils import FullConfig, LoRAConfig

DEMO_NUM_GENERATE_TOKENS = 16
DEMO_PROMPT_FRACTION = 4
DEFAULT_TOKENIZER_NAME = "google/gemma-2b"


def _try_load_tokenizer(name: str = DEFAULT_TOKENIZER_NAME):
    """Attempt to load a HuggingFace tokenizer, returning None on failure."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return None


def _decode_tokens(token_ids, tokenizer=None):
    """Decode token IDs to text using a tokenizer or byte-level fallback."""
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if tokenizer is not None:
        try:
            return tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    return "".join(chr(t % 256) for t in ids)


def main():
    """Run the demo: create, convert, and evaluate a small test model."""
    parser = argparse.ArgumentParser(
        description="Demo for Relaxed Recursive Transformers."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a checkpoint directory to load instead of creating a random model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (used with --model if checkpoint has no config).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Relaxed Recursive Transformers — JAX/Flax Demo")
    print("=" * 60)

    if args.model:
        _run_checkpoint_demo(args)
        return

    _run_random_demo()


def _run_checkpoint_demo(args):
    """Run demo loading a saved checkpoint."""
    from src.utils.checkpoint import load_checkpoint
    from src.utils.config_utils import load_config

    print(f"\n[1/4] Loading checkpoint from {args.model}...")
    config_path = Path(args.model) / "config.json"
    if config_path.exists():
        import json
        from src.utils.config_utils import ModelConfig, RecursiveConfig

        with open(config_path) as f:
            cfg = json.load(f)
        config = FullConfig(
            model=ModelConfig(**cfg["model"]),
            recursive=RecursiveConfig(**cfg["recursive"]),
            lora=LoRAConfig(**cfg["lora"]),
            seed=cfg.get("seed", 42),
        )
    elif args.config:
        config = load_config(args.config)
    else:
        config = get_test_config()

    model = RelaxedRecursiveTransformer.from_config(config)
    dummy_ids = jnp.ones((1, DEMO_PROMPT_FRACTION), dtype=jnp.int32)
    template_params = model.init(jax.random.PRNGKey(0), dummy_ids)
    relaxed_params, _ = load_checkpoint(args.model, template_params)
    relaxed_counts = count_parameters(relaxed_params)
    print(f"  Loaded model: {relaxed_counts['total']:,} parameters")
    print(f"  LoRA parameters: {relaxed_counts['lora']:,}")

    rng = jax.random.PRNGKey(config.seed)
    prompt_len = config.model.max_seq_len // DEMO_PROMPT_FRACTION
    input_ids = jax.random.randint(rng, (1, prompt_len), 0, config.model.vocab_size)

    print(f"\n[2/4] Running forward pass...")
    logits = model.apply(relaxed_params, input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")

    print(f"\n[3/4] Text generation (argmax decoding)...")
    tokenizer = _try_load_tokenizer()
    generated, stats = early_exit_generate(
        model,
        relaxed_params,
        input_ids,
        config,
        max_new_tokens=DEMO_NUM_GENERATE_TOKENS,
        confidence_threshold=0.01,
    )
    new_tokens = generated[0, prompt_len:]
    decoded_text = _decode_tokens(new_tokens, tokenizer)
    print(f"  Prompt tokens: {input_ids[0, :8].tolist()}...")
    print(f"  Generated token IDs: {new_tokens.tolist()}")
    print(f"  Decoded text: {decoded_text!r}")
    if tokenizer is None:
        print(f"  (byte-level fallback — install transformers + log in for real decoding)")
    print(f"  Mean exit depth: {stats.mean_exit_depth:.2f} / {stats.num_loops}")
    print(f"  Early exit rate: {stats.early_exit_rate:.1%}")

    print(f"\n[4/4] Efficiency analysis...")
    print(f"  Parameters: {relaxed_counts['total']:,}")
    print(f"  LoRA overhead: {relaxed_counts['overhead_percent']:.1f}%")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def _run_random_demo():
    """Run demo creating a fresh random model."""
    config = get_test_config()
    rng = jax.random.PRNGKey(config.seed)
    seq_len = config.model.max_seq_len
    prompt_len = seq_len // DEMO_PROMPT_FRACTION

    print("\n[1/6] Creating vanilla transformer...")
    vanilla_model = VanillaTransformer.from_config(config)
    dummy_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    vanilla_params = vanilla_model.init(rng, dummy_ids)
    vanilla_counts = count_parameters(vanilla_params)
    print(f"  Vanilla model: {vanilla_counts['total']:,} parameters")

    print("\n[2/6] Converting to recursive (no LoRA)...")
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
    print(f"  Recursive model: {recursive_counts['total']:,} parameters")
    print(
        f"  Parameter reduction: "
        f"{(1 - recursive_counts['total'] / vanilla_counts['total']) * 100:.1f}%"
    )

    print("\n[3/6] Converting to relaxed recursive (with LoRA)...")
    _, relaxed_params, _ = convert_random_to_recursive(config, method="average")
    relaxed_model = RelaxedRecursiveTransformer.from_config(config)
    relaxed_counts = count_parameters(relaxed_params)
    print(f"  Relaxed recursive model: {relaxed_counts['total']:,} parameters")
    print(f"  LoRA parameters: {relaxed_counts['lora']:,}")
    print(f"  LoRA overhead: {relaxed_counts['overhead_percent']:.1f}%")

    print("\n[4/6] Running forward pass...")
    input_ids = jax.random.randint(rng, (1, seq_len), 0, config.model.vocab_size)
    logits = relaxed_model.apply(relaxed_params, input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Sample logits (first 8): {logits[0, 0, :8]}")

    print("\n[5/6] Text generation with early exit...")
    tokenizer = _try_load_tokenizer()
    prompt = input_ids[:, :prompt_len]
    generated, stats = early_exit_generate(
        relaxed_model,
        relaxed_params,
        prompt,
        config,
        max_new_tokens=DEMO_NUM_GENERATE_TOKENS,
        confidence_threshold=0.01,
    )
    new_tokens = generated[0, prompt_len:]
    decoded_text = _decode_tokens(new_tokens, tokenizer)
    print(f"  Prompt tokens ({prompt_len}): {prompt[0, :8].tolist()}...")
    print(f"  Generated token IDs ({DEMO_NUM_GENERATE_TOKENS}): {new_tokens.tolist()}")
    print(f"  Decoded text: {decoded_text!r}")
    if tokenizer is None:
        print(f"  (byte-level fallback — install transformers + log in for real decoding)")
    print(f"  Mean exit depth: {stats.mean_exit_depth:.2f} / {stats.num_loops}")
    print(f"  Early exit rate: {stats.early_exit_rate:.1%}")

    batching_result = simulate_depth_wise_batching(
        stats,
        batch_size=1,
        num_loops=config.recursive.num_loops,
    )

    print(f"\n[6/6] Summary tables...")

    print("\n" + "=" * 60)
    print("Parameter Comparison")
    print("=" * 60)
    print(f"  {'Model':<25} {'Params':>12} {'LoRA':>10} {'Overhead':>10}")
    print(f"  {'-' * 57}")
    print(
        f"  {'Vanilla':<25} {vanilla_counts['total']:>12,} "
        f"{'N/A':>10} {'N/A':>10}"
    )
    print(
        f"  {'Recursive':<25} {recursive_counts['total']:>12,} "
        f"{'0':>10} {'0.0%':>10}"
    )
    print(
        f"  {'Relaxed Recursive':<25} {relaxed_counts['total']:>12,} "
        f"{relaxed_counts['lora']:>10,} "
        f"{relaxed_counts['overhead_percent']:>9.1f}%"
    )

    print("\n" + "=" * 60)
    print("FLOPs Comparison")
    print("=" * 60)
    flops = estimate_flops(config)
    for variant, flop_data in flops.items():
        print(f"  {variant:<25} {flop_data['total_flops']:>15,.0f} ({flop_data['relative']:.2f}x)")

    print("\nDepth-wise Batching Simulation:")
    print(format_batching_comparison(batching_result, config.recursive.num_loops))

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
