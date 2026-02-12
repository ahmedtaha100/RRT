"""Checkpoint saving and loading utilities.

Provides save/load functions for model parameters and configuration using
orbax-checkpoint for robust, portable serialization of JAX pytrees.
"""

import json
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from src.utils.config_utils import FullConfig

PARAMS_SUBDIR = "params"


def save_checkpoint(
    params: dict,
    path: str | Path,
    config: Optional[FullConfig] = None,
) -> None:
    """Save model parameters and optional config to a checkpoint directory.

    Uses orbax PyTreeCheckpointer for reliable, framework-native serialization
    of JAX parameter trees.

    Args:
        params: Model parameter dictionary to save.
        path: Directory path for the checkpoint.
        config: Optional FullConfig to save alongside parameters.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    params_path = str(path / PARAMS_SUBDIR)
    checkpointer.save(params_path, params)

    if config is not None:
        config_dict = {
            "model": {
                "num_layers": config.model.num_layers,
                "hidden_dim": config.model.hidden_dim,
                "num_heads": config.model.num_heads,
                "num_kv_heads": config.model.num_kv_heads,
                "intermediate_dim": config.model.intermediate_dim,
                "vocab_size": config.model.vocab_size,
                "max_seq_len": config.model.max_seq_len,
                "rope_theta": config.model.rope_theta,
                "rms_norm_eps": config.model.rms_norm_eps,
            },
            "recursive": {
                "num_loops": config.recursive.num_loops,
                "block_size": config.recursive.block_size,
            },
            "lora": {
                "rank": config.lora.rank,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
            },
            "seed": config.seed,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)


def load_checkpoint(
    path: str | Path,
    params_template: Optional[dict] = None,
) -> tuple[dict, Optional[FullConfig]]:
    """Load model parameters and config from a checkpoint directory.

    Args:
        path: Directory path containing the checkpoint.
        params_template: Optional parameter template for shape/dtype hints.
            If provided, orbax uses it to restore with the correct structure.

    Returns:
        Tuple of (restored_params, config) where config may be None if
        no config.json was saved.
    """
    path = Path(path)

    checkpointer = ocp.PyTreeCheckpointer()
    params_path = str(path / PARAMS_SUBDIR)
    restored_params = checkpointer.restore(params_path, item=params_template)

    config = None
    config_path = path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        from src.utils.config_utils import (
            FullConfig,
            LoRAConfig,
            ModelConfig,
            RecursiveConfig,
        )
        config = FullConfig(
            model=ModelConfig(**config_dict["model"]),
            recursive=RecursiveConfig(**config_dict["recursive"]),
            lora=LoRAConfig(**config_dict["lora"]),
            seed=config_dict.get("seed", 42),
        )

    return restored_params, config
