"""Weight initialization strategies for converting vanilla to recursive models.

Provides averaging and selection-based initialization for creating shared
block weights from the layers of a pretrained vanilla transformer, as
described in Bae et al. (2024) Section 3.1.
"""

import jax.numpy as jnp


def average_init(
    layer_weights_list: list[dict[str, jnp.ndarray]],
) -> dict[str, jnp.ndarray]:
    """Compute the element-wise mean of weights from a list of layers.

    Given weights from layers that will be tied together (e.g., layers
    [0, 6, 12] for the first position in a block_size=6, num_loops=3 setup),
    returns their element-wise average as the shared weight initialization.

    Args:
        layer_weights_list: List of weight dictionaries, one per layer.
            Each dictionary maps parameter names to JAX arrays.

    Returns:
        A single weight dictionary with the same structure, containing
        element-wise averaged values.
    """
    if len(layer_weights_list) == 1:
        return layer_weights_list[0]

    averaged = {}
    for key in layer_weights_list[0]:
        stacked = jnp.stack([lw[key] for lw in layer_weights_list])
        averaged[key] = jnp.mean(stacked, axis=0)
    return averaged


def select_k_init(
    all_layer_weights: list[dict[str, jnp.ndarray]],
    block_size: int,
    strategy: str = "middle",
) -> list[dict[str, jnp.ndarray]]:
    """Select block_size representative layers from the full set of layer weights.

    Instead of averaging, this strategy picks specific layers to use as the
    shared block. The 'middle' strategy selects layers from the middle third
    of the network, which tend to be more representative.

    Args:
        all_layer_weights: Weights for all original layers, ordered by depth.
        block_size: Number of layers to select for the shared block.
        strategy: Selection strategy. Currently supports 'middle' (selects
            from the middle third of the network) and 'uniform' (evenly
            spaced layers).

    Returns:
        List of block_size weight dictionaries for the selected layers.

    Raises:
        ValueError: If strategy is not recognized.
    """
    num_layers = len(all_layer_weights)

    if strategy == "middle":
        middle_start = num_layers // 3
        middle_end = 2 * num_layers // 3
        middle_range = middle_end - middle_start

        if middle_range <= block_size:
            indices = list(range(middle_start, middle_start + block_size))
        else:
            step = middle_range / block_size
            indices = [int(middle_start + i * step) for i in range(block_size)]

    elif strategy == "uniform":
        step = num_layers / block_size
        indices = [int(i * step) for i in range(block_size)]

    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    return [all_layer_weights[i] for i in indices]
