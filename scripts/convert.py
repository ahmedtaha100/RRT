"""Conversion script for creating relaxed recursive models.

Supports converting from random initialization (for testing) or from
pretrained HuggingFace models (e.g., Gemma 2B) to recursive and relaxed
recursive variants.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Parse arguments and run the conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert a transformer to relaxed recursive form."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="random",
        help="Model source: 'random' for test model, or HuggingFace model ID.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_small.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="relaxed",
        choices=["recursive", "relaxed"],
        help="Conversion method: 'recursive' or 'relaxed' (with LoRA).",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="average",
        choices=["average", "select_k"],
        help="Weight initialization strategy: 'average' or 'select_k'.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank for relaxed conversion.",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=None,
        help="Number of recursive loops (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/converted",
        help="Output directory for checkpoint.",
    )
    args = parser.parse_args()

    from src.utils.config_utils import load_config, LoRAConfig
    from src.utils.checkpoint import save_checkpoint

    config = load_config(args.config)

    if args.num_loops is not None:
        config.recursive.num_loops = args.num_loops
        config.recursive.block_size = config.model.num_layers // args.num_loops

    if args.method == "relaxed":
        config.lora.rank = args.lora_rank
        config.lora.alpha = args.lora_rank * 2
    else:
        config.lora.rank = 0

    if args.source == "random":
        from src.conversion.convert_gemma import convert_random_to_recursive

        print(f"Converting random model with config: {args.config}")
        print(f"  Method: {args.method}")
        print(f"  Init strategy: {args.init}")
        print(f"  LoRA rank: {config.lora.rank}")
        print(f"  Block size: {config.recursive.block_size}")
        print(f"  Num loops: {config.recursive.num_loops}")

        vanilla_params, recursive_params, config = convert_random_to_recursive(
            config, method=args.init
        )

        save_checkpoint(recursive_params, args.output, config)
        print(f"\nCheckpoint saved to: {args.output}")
    else:
        from src.conversion.convert_gemma import convert_gemma_to_recursive

        print(f"Converting {args.source} to relaxed recursive form...")
        convert_gemma_to_recursive(
            model_name=args.source,
            method=args.method,
            init_strategy=args.init,
            lora_rank=args.lora_rank,
            num_loops=config.recursive.num_loops,
            output_path=args.output,
        )
        print(f"\nCheckpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
