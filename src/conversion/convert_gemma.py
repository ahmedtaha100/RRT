"""Conversion pipeline for pretrained models to relaxed recursive form.

Handles converting vanilla pretrained transformers (including Gemma 2B) into
recursive and relaxed recursive variants. Supports both real HuggingFace model
loading and random model generation for testing.
"""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from src.conversion.svd_init import compute_lora_init_for_layer, PROJECTION_NAMES
from src.conversion.weight_init import average_init, select_k_init
from src.model.relaxed_recursive_transformer import (
    RelaxedRecursiveTransformer,
    VanillaTransformer,
)
from src.utils.config_utils import FullConfig, LoRAConfig

GEMMA_2B_NUM_LAYERS = 18
INIT_DUMMY_SEQ_LEN = 4


def extract_vanilla_layer_weights(
    vanilla_params: dict,
    layer_idx: int,
) -> dict[str, jnp.ndarray]:
    """Extract projection weight matrices from a vanilla transformer layer.

    Extracts the kernel weights for all attention and feed-forward projections
    from a specific layer, transposing from Flax (in, out) format to
    (out, in) format for SVD initialization.

    Args:
        vanilla_params: Full parameter dictionary from a VanillaTransformer.
        layer_idx: Index of the layer to extract weights from.

    Returns:
        Dictionary mapping projection names to weight matrices in (out, in) format.
    """
    layer_params = vanilla_params["params"][f"layer_{layer_idx}"]
    weights = {}

    attn_params = layer_params["attention"]
    for proj_name in ["query_proj", "key_proj", "value_proj", "output_proj"]:
        kernel = attn_params[proj_name]["kernel"]
        weights[proj_name] = kernel.T

    ff_params = layer_params["feed_forward"]
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        kernel = ff_params[proj_name]["kernel"]
        weights[proj_name] = kernel.T

    return weights


def convert_random_to_recursive(
    config: FullConfig,
    method: str = "average",
    seed: Optional[int] = None,
) -> tuple[dict, dict, FullConfig]:
    """Convert a randomly initialized vanilla model to relaxed recursive form.

    Creates a random vanilla transformer, then converts it to a relaxed
    recursive model using the specified weight initialization method and
    SVD-based LoRA initialization.

    Args:
        config: Full configuration for the model.
        method: Weight initialization method ('average' or 'select_k').
        seed: Random seed for model initialization. Uses config.seed if None.

    Returns:
        Tuple of (vanilla_params, recursive_params, config) where
        vanilla_params are the original model weights, recursive_params
        are the converted relaxed recursive model weights, and config
        is the configuration used.
    """
    effective_seed = seed if seed is not None else config.seed
    rng = jax.random.PRNGKey(effective_seed)
    rng_init, rng_input = jax.random.split(rng)

    vanilla_model = VanillaTransformer.from_config(config)
    dummy_ids = jnp.ones((1, INIT_DUMMY_SEQ_LEN), dtype=jnp.int32)
    vanilla_params = vanilla_model.init(rng_init, dummy_ids)

    num_layers = config.model.num_layers
    block_size = config.recursive.block_size
    num_loops = config.recursive.num_loops
    lora_rank = config.lora.rank

    all_layer_weights = [
        extract_vanilla_layer_weights(vanilla_params, i)
        for i in range(num_layers)
    ]

    if method == "select_k":
        shared_block_weights = select_k_init(
            all_layer_weights, block_size, strategy="middle"
        )
    else:
        shared_block_weights = []
        for block_pos in range(block_size):
            tied_layers = [
                all_layer_weights[block_pos + loop * block_size]
                for loop in range(num_loops)
                if block_pos + loop * block_size < num_layers
            ]
            shared_block_weights.append(average_init(tied_layers))

    recursive_model = RelaxedRecursiveTransformer.from_config(config)
    recursive_params = recursive_model.init(rng_init, dummy_ids)

    recursive_params = _populate_shared_weights(
        recursive_params, shared_block_weights, config
    )

    if lora_rank > 0:
        recursive_params = _populate_lora_params(
            recursive_params,
            all_layer_weights,
            shared_block_weights,
            config,
        )

    return vanilla_params, recursive_params, config


def _populate_shared_weights(
    recursive_params: dict,
    shared_block_weights: list[dict[str, jnp.ndarray]],
    config: FullConfig,
) -> dict:
    """Populate the shared base layer weights in a recursive model.

    Sets the attention and feed-forward projection kernels in each shared
    layer to the averaged (or selected) weights from the vanilla model.

    Args:
        recursive_params: Initialized parameter dict for the recursive model.
        shared_block_weights: List of weight dicts for the shared block layers.
        config: Full configuration.

    Returns:
        Updated parameter dictionary with shared weights populated.
    """
    import copy
    params = copy.deepcopy(dict(recursive_params))
    params["params"] = dict(params["params"])

    for layer_idx in range(config.recursive.block_size):
        layer_key = f"shared_layer_{layer_idx}"
        if layer_key not in params["params"]:
            continue

        layer_params = dict(params["params"][layer_key])
        weights = shared_block_weights[layer_idx]

        attn_params = dict(layer_params["attention"])
        for proj_name in ["query_proj", "key_proj", "value_proj", "output_proj"]:
            proj_params = dict(attn_params[proj_name])
            proj_params["kernel"] = weights[proj_name].T
            attn_params[proj_name] = proj_params
        layer_params["attention"] = attn_params

        ff_params = dict(layer_params["feed_forward"])
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj_params = dict(ff_params[proj_name])
            proj_params["kernel"] = weights[proj_name].T
            ff_params[proj_name] = proj_params
        layer_params["feed_forward"] = ff_params

        params["params"][layer_key] = layer_params

    return params


LORA_NAME_MAP = {
    "query_proj": "query_lora",
    "key_proj": "key_lora",
    "value_proj": "value_lora",
    "output_proj": "output_lora",
    "gate_proj": "gate_lora",
    "up_proj": "up_lora",
    "down_proj": "down_lora",
}


def _populate_lora_params(
    recursive_params: dict,
    all_layer_weights: list[dict[str, jnp.ndarray]],
    shared_block_weights: list[dict[str, jnp.ndarray]],
    config: FullConfig,
) -> dict:
    """Populate per-loop LoRA adapters with SVD-initialized matrices.

    For each (loop, layer) pair, computes the residual between the original
    vanilla layer weights and the shared weights, then initializes LoRA A/B
    matrices via truncated SVD.

    Args:
        recursive_params: Parameter dict with shared weights already populated.
        all_layer_weights: Original vanilla layer weights.
        shared_block_weights: Averaged shared block weights.
        config: Full configuration.

    Returns:
        Updated parameter dictionary with LoRA adapters populated.
    """
    import copy
    params = copy.deepcopy(dict(recursive_params))
    params["params"] = dict(params["params"])

    block_size = config.recursive.block_size
    num_loops = config.recursive.num_loops
    lora_rank = config.lora.rank

    for loop_idx in range(num_loops):
        for layer_idx in range(block_size):
            lora_key = f"lora_loop_{loop_idx}_layer_{layer_idx}"
            if lora_key not in params["params"]:
                continue

            lora_set_params = dict(params["params"][lora_key])
            shared_weights = shared_block_weights[layer_idx]
            original_idx = loop_idx * block_size + layer_idx
            original_weights = all_layer_weights[original_idx]

            lora_init = compute_lora_init_for_layer(
                original_weights,
                shared_weights,
                rank=lora_rank,
            )

            for proj_name, lora_adapter_name in LORA_NAME_MAP.items():
                if lora_adapter_name in lora_set_params and proj_name in lora_init:
                    adapter_params = dict(lora_set_params[lora_adapter_name])
                    lora_a, lora_b = lora_init[proj_name]
                    adapter_params["lora_a"] = lora_a
                    adapter_params["lora_b"] = lora_b
                    lora_set_params[lora_adapter_name] = adapter_params

            params["params"][lora_key] = lora_set_params

    return params


def convert_gemma_to_recursive(
    model_name: str = "google/gemma-2b",
    method: str = "relaxed",
    init_strategy: str = "average",
    lora_rank: int = 64,
    num_loops: int = 3,
    output_path: Optional[str] = None,
) -> dict:
    """Convert a pretrained Gemma model to relaxed recursive form.

    First attempts to load using FlaxGemmaForCausalLM (Flax-native).
    If unavailable, falls back to PyTorch AutoModelForCausalLM and
    converts tensors to JAX arrays.

    Requires HuggingFace authentication for gated Gemma models.

    Args:
        model_name: HuggingFace model identifier (e.g., 'google/gemma-2b').
        method: Conversion method - 'recursive' for pure layer tying or
            'relaxed' for layer tying with LoRA relaxation.
        init_strategy: Weight initialization strategy ('average' or 'select_k').
        lora_rank: Rank for LoRA decomposition (only used when method='relaxed').
        num_loops: Number of recursive loop applications.
        output_path: Optional path to save the converted checkpoint.

    Returns:
        Converted model parameters.

    Raises:
        RuntimeError: If HuggingFace authentication fails or model cannot be loaded.
    """
    all_layer_weights = _load_gemma_weights(model_name)

    from src.model.config import get_gemma_2b_config

    effective_rank = lora_rank if method == "relaxed" else 0
    block_size = GEMMA_2B_NUM_LAYERS // num_loops
    config = get_gemma_2b_config(
        lora_rank=effective_rank,
        num_loops=num_loops,
        block_size=block_size,
    )

    if init_strategy == "select_k":
        shared_block_weights = select_k_init(
            all_layer_weights, block_size, strategy="middle"
        )
    else:
        shared_block_weights = []
        for block_pos in range(block_size):
            tied_layers = [
                all_layer_weights[block_pos + loop * block_size]
                for loop in range(num_loops)
                if block_pos + loop * block_size < GEMMA_2B_NUM_LAYERS
            ]
            shared_block_weights.append(average_init(tied_layers))

    recursive_model = RelaxedRecursiveTransformer.from_config(config)
    rng = jax.random.PRNGKey(config.seed)
    dummy_ids = jnp.ones((1, INIT_DUMMY_SEQ_LEN), dtype=jnp.int32)
    recursive_params = recursive_model.init(rng, dummy_ids)

    recursive_params = _populate_shared_weights(
        recursive_params, shared_block_weights, config
    )

    if method == "relaxed":
        recursive_params = _populate_lora_params(
            recursive_params,
            all_layer_weights,
            shared_block_weights,
            config,
        )

    if output_path is not None:
        from src.utils.checkpoint import save_checkpoint
        save_checkpoint(recursive_params, output_path, config)

    return recursive_params


def _load_gemma_weights(model_name: str) -> list[dict[str, jnp.ndarray]]:
    """Load Gemma weights, trying Flax-native loading first, then PyTorch fallback.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        List of weight dicts for all layers, in (out, in) format.

    Raises:
        RuntimeError: If model cannot be loaded.
    """
    try:
        from transformers import FlaxGemmaForCausalLM
        flax_model = FlaxGemmaForCausalLM.from_pretrained(model_name)
        return _extract_flax_gemma_weights(flax_model)
    except (ImportError, Exception):
        pass

    try:
        from transformers import AutoModelForCausalLM
        pt_model = AutoModelForCausalLM.from_pretrained(model_name)
        return _extract_pytorch_gemma_weights(pt_model)
    except ImportError as exc:
        raise RuntimeError(
            "transformers library is required for Gemma conversion. "
            "Install with: pip install transformers"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {model_name}. Gemma 2B requires authentication. "
            "Run `huggingface-cli login` and accept the license at "
            "https://huggingface.co/google/gemma-2b first."
        ) from exc


def _extract_flax_gemma_weights(flax_model) -> list[dict[str, jnp.ndarray]]:
    """Extract weights from a FlaxGemmaForCausalLM model.

    Args:
        flax_model: A loaded FlaxGemmaForCausalLM instance.

    Returns:
        List of weight dicts for all layers in (out, in) format.
    """
    params = flax_model.params
    all_layer_weights = []

    for layer_idx in range(GEMMA_2B_NUM_LAYERS):
        layer_params = params["model"]["layers"][str(layer_idx)]
        attn = layer_params["self_attn"]
        mlp = layer_params["mlp"]

        layer_w = {
            "query_proj": jnp.array(attn["q_proj"]["kernel"]).T,
            "key_proj": jnp.array(attn["k_proj"]["kernel"]).T,
            "value_proj": jnp.array(attn["v_proj"]["kernel"]).T,
            "output_proj": jnp.array(attn["o_proj"]["kernel"]).T,
            "gate_proj": jnp.array(mlp["gate_proj"]["kernel"]).T,
            "up_proj": jnp.array(mlp["up_proj"]["kernel"]).T,
            "down_proj": jnp.array(mlp["down_proj"]["kernel"]).T,
        }
        all_layer_weights.append(layer_w)

    return all_layer_weights


def _extract_pytorch_gemma_weights(pt_model) -> list[dict[str, jnp.ndarray]]:
    """Extract weights from a PyTorch AutoModelForCausalLM Gemma model.

    Converts PyTorch tensors to JAX arrays. PyTorch Linear weights are in
    (out, in) format, which matches our convention directly.

    Args:
        pt_model: A loaded AutoModelForCausalLM instance.

    Returns:
        List of weight dicts for all layers in (out, in) format.
    """
    state_dict = pt_model.state_dict()
    all_layer_weights = []

    for layer_idx in range(GEMMA_2B_NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}.self_attn"
        ff_prefix = f"model.layers.{layer_idx}.mlp"

        layer_w = {
            "query_proj": jnp.array(state_dict[f"{prefix}.q_proj.weight"].float().numpy()),
            "key_proj": jnp.array(state_dict[f"{prefix}.k_proj.weight"].float().numpy()),
            "value_proj": jnp.array(state_dict[f"{prefix}.v_proj.weight"].float().numpy()),
            "output_proj": jnp.array(state_dict[f"{prefix}.o_proj.weight"].float().numpy()),
            "gate_proj": jnp.array(state_dict[f"{ff_prefix}.gate_proj.weight"].float().numpy()),
            "up_proj": jnp.array(state_dict[f"{ff_prefix}.up_proj.weight"].float().numpy()),
            "down_proj": jnp.array(state_dict[f"{ff_prefix}.down_proj.weight"].float().numpy()),
        }
        all_layer_weights.append(layer_w)

    return all_layer_weights
