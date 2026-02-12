"""Evaluation script for model quality and efficiency metrics.

Supports perplexity evaluation on WikiText-2 or synthetic data, and
parameter/FLOPs efficiency analysis.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Parse arguments and run requested evaluations."""
    parser = argparse.ArgumentParser(
        description="Evaluate a relaxed recursive transformer model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_small.yaml",
        help="Path to YAML configuration file (used if no checkpoint).",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="efficiency",
        help="Comma-separated evaluation types: 'perplexity', 'efficiency'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "wikitext2"],
        help="Dataset for perplexity evaluation: 'synthetic' or 'wikitext2'.",
    )
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from src.evaluation.efficiency import count_parameters, estimate_flops
    from src.model.config import get_test_config
    from src.model.relaxed_recursive_transformer import RelaxedRecursiveTransformer
    from src.utils.config_utils import load_config

    eval_types = [e.strip() for e in args.eval.split(",")]

    if args.checkpoint:
        from src.utils.checkpoint import load_checkpoint

        config_path = Path(args.checkpoint) / "config.json"
        if config_path.exists():
            import json
            from src.utils.config_utils import (
                FullConfig,
                LoRAConfig,
                ModelConfig,
                RecursiveConfig,
            )

            with open(config_path) as f:
                cfg = json.load(f)
            config = FullConfig(
                model=ModelConfig(**cfg["model"]),
                recursive=RecursiveConfig(**cfg["recursive"]),
                lora=LoRAConfig(**cfg["lora"]),
                seed=cfg.get("seed", 42),
            )
        else:
            config = load_config(args.config)

        model = RelaxedRecursiveTransformer.from_config(config)
        dummy_ids = jnp.ones((1, 4), dtype=jnp.int32)
        template_params = model.init(jax.random.PRNGKey(0), dummy_ids)
        params, _ = load_checkpoint(args.checkpoint, template_params)
    else:
        config = load_config(args.config)
        model = RelaxedRecursiveTransformer.from_config(config)
        dummy_ids = jnp.ones((1, 4), dtype=jnp.int32)
        params = model.init(jax.random.PRNGKey(config.seed), dummy_ids)

    if "efficiency" in eval_types:
        print("=" * 50)
        print("Efficiency Analysis")
        print("=" * 50)

        counts = count_parameters(params)
        print(f"\nParameter Counts:")
        print(f"  Total:     {counts['total']:>12,}")
        print(f"  Shared:    {counts['shared']:>12,}")
        print(f"  LoRA:      {counts['lora']:>12,}")
        print(f"  Embedding: {counts['embedding']:>12,}")
        print(f"  Overhead:  {counts['overhead_percent']:>11.1f}%")

        print(f"\nFLOPs Estimates:")
        flops = estimate_flops(config)
        for variant, data in flops.items():
            print(f"  {variant:<25} {data['total_flops']:>15,.0f} ({data['relative']:.2f}x)")

    if "perplexity" in eval_types:
        print("\n" + "=" * 50)
        print("Perplexity Evaluation")
        print("=" * 50)

        if args.dataset == "wikitext2":
            from src.evaluation.perplexity import evaluate_perplexity, load_wikitext2

            print("\n  Loading WikiText-2 test set...")
            texts = load_wikitext2(split="test")
            print(f"  Loaded {len(texts)} text samples.")

            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
                encode_fn = lambda text: tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                print("  Warning: Could not load Gemma tokenizer, using byte-level encoding.")
                encode_fn = lambda text: [b % config.model.vocab_size for b in text.encode("utf-8")]

            ppl = evaluate_perplexity(
                model.apply,
                params,
                texts,
                encode_fn,
                max_seq_len=config.model.max_seq_len,
            )
            print(f"\n  WikiText-2 perplexity: {ppl:.2f}")
        else:
            from src.evaluation.perplexity import evaluate_perplexity_simple

            rng = jax.random.PRNGKey(0)
            synthetic_tokens = jax.random.randint(rng, (512,), 0, config.model.vocab_size)

            ppl = evaluate_perplexity_simple(
                model.apply,
                params,
                synthetic_tokens,
                max_seq_len=config.model.max_seq_len,
            )
            print(f"\n  Perplexity on synthetic data: {ppl:.2f}")
            print("  (Note: meaningful perplexity requires real text data)")


if __name__ == "__main__":
    main()
