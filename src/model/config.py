"""Model configuration re-exports and factory functions.

Provides convenient access to configuration dataclasses and preset
configurations for testing and Gemma 2B scale experiments.
"""

from src.utils.config_utils import (
    FullConfig,
    LoRAConfig,
    ModelConfig,
    RecursiveConfig,
    load_config,
)


def get_test_config() -> FullConfig:
    """Return a small test configuration for CPU development and testing.

    Uses 4 layers, hidden_dim=128, 4 heads, 1 KV head, vocab=256,
    seq_len=64, block_size=2, num_loops=2, LoRA rank=8.
    """
    return FullConfig(
        model=ModelConfig(
            num_layers=4,
            hidden_dim=128,
            num_heads=4,
            num_kv_heads=1,
            intermediate_dim=512,
            vocab_size=256,
            max_seq_len=64,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
        ),
        recursive=RecursiveConfig(num_loops=2, block_size=2),
        lora=LoRAConfig(rank=8, alpha=16, dropout=0.0),
        seed=42,
    )


def get_gemma_2b_config(
    lora_rank: int = 64,
    num_loops: int = 3,
    block_size: int = 6,
) -> FullConfig:
    """Return a Gemma 2B configuration for the relaxed recursive transformer.

    Based on the architecture described in Bae et al. (2024): 18 decoder
    layers, hidden_dim=2048, 8 attention heads, 1 KV head (GQA), GeGLU
    feed-forward with intermediate_dim=16384, vocab_size=256128.

    Args:
        lora_rank: Rank for depth-wise LoRA modules. Set to 0 for pure recursive.
        num_loops: Number of recursive applications of the shared block.
        block_size: Number of unique layers in the shared block.
    """
    return FullConfig(
        model=ModelConfig(
            num_layers=18,
            hidden_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            intermediate_dim=16384,
            vocab_size=256128,
            max_seq_len=8192,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
        ),
        recursive=RecursiveConfig(num_loops=num_loops, block_size=block_size),
        lora=LoRAConfig(rank=lora_rank, alpha=lora_rank * 2, dropout=0.0),
        seed=42,
    )


__all__ = [
    "ModelConfig",
    "RecursiveConfig",
    "LoRAConfig",
    "FullConfig",
    "load_config",
    "get_test_config",
    "get_gemma_2b_config",
]
