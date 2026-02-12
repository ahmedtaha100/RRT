"""Configuration dataclasses and YAML loading for Relaxed Recursive Transformers.

Provides structured configuration for model architecture, recursive block settings,
LoRA parameters, and top-level experiment configuration. Supports loading from YAML
files. Preset factory functions (e.g., get_test_config, get_gemma_2b_config) are
defined in src.model.config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Transformer model architecture configuration.

    Attributes:
        num_layers: Total number of effective decoder layers.
        hidden_dim: Hidden dimension size for embeddings and residual stream.
        num_heads: Number of attention heads for queries.
        num_kv_heads: Number of key-value heads (for GQA/MQA).
        intermediate_dim: Intermediate dimension in the feed-forward network.
        vocab_size: Size of the token vocabulary.
        max_seq_len: Maximum sequence length supported.
        rope_theta: Base frequency for rotary position embeddings.
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
    """

    num_layers: int = 4
    hidden_dim: int = 128
    num_heads: int = 4
    num_kv_heads: int = 1
    intermediate_dim: int = 512
    vocab_size: int = 256
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6


@dataclass
class RecursiveConfig:
    """Configuration for recursive (layer-tying) block structure.

    Attributes:
        num_loops: Number of times the shared block is applied.
        block_size: Number of unique transformer layers in the shared block.
    """

    num_loops: int = 2
    block_size: int = 2


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation modules.

    Attributes:
        rank: Rank of the LoRA decomposition. Set to 0 to disable LoRA.
        alpha: Scaling factor for LoRA output (effective scale = alpha / rank).
        dropout: Dropout rate applied to LoRA input (0.0 disables dropout).
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0


@dataclass
class FullConfig:
    """Top-level configuration combining all sub-configurations.

    Attributes:
        model: Transformer architecture configuration.
        recursive: Recursive block configuration.
        lora: LoRA module configuration.
        seed: Random seed for reproducibility.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    recursive: RecursiveConfig = field(default_factory=RecursiveConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    seed: int = 42


def load_config(yaml_path: str | Path) -> FullConfig:
    """Load a FullConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A FullConfig instance populated from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw.get("model", {}))
    recursive_cfg = RecursiveConfig(**raw.get("recursive", {}))
    lora_cfg = LoRAConfig(**raw.get("lora", {}))
    seed = raw.get("seed", 42)

    return FullConfig(
        model=model_cfg,
        recursive=recursive_cfg,
        lora=lora_cfg,
        seed=seed,
    )
