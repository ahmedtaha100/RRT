"""Early exit inference for recursive transformers.

Implements the early exit strategy from Bae et al. (2024) Section 4.1, where
generation can terminate a recursive loop early if the model's prediction has
converged (i.e., the argmax token hasn't changed between consecutive loops)
and the model is sufficiently confident in its prediction. This trades off a
small amount of quality for significant speedup.

The implementation uses the model's loop-by-loop methods (embed, apply_loop,
hidden_to_logits) to achieve real compute savings by skipping remaining loops
once convergence is detected, rather than computing all loops and discarding
later results.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from src.utils.config_utils import FullConfig


@dataclass
class EarlyExitStats:
    """Statistics from early exit generation.

    Attributes:
        exit_depths: List of exit loop indices for each generated token.
        total_tokens: Total number of tokens generated.
        num_loops: Maximum number of loops available.
    """

    exit_depths: list[int] = field(default_factory=list)
    total_tokens: int = 0
    num_loops: int = 1

    @property
    def mean_exit_depth(self) -> float:
        """Average exit depth across all generated tokens."""
        if not self.exit_depths:
            return 0.0
        return sum(self.exit_depths) / len(self.exit_depths)

    @property
    def early_exit_rate(self) -> float:
        """Fraction of tokens that exited before the final loop."""
        if not self.exit_depths:
            return 0.0
        early_exits = sum(1 for d in self.exit_depths if d < self.num_loops - 1)
        return early_exits / len(self.exit_depths)


def early_exit_generate(
    model,
    params: dict,
    input_ids: jnp.ndarray,
    config: FullConfig,
    max_new_tokens: int = 32,
    confidence_threshold: float = 0.0,
) -> tuple[jnp.ndarray, EarlyExitStats]:
    """Generate tokens with early exit based on prediction convergence and confidence.

    For each token position, runs the model loop-by-loop. Early exit is triggered
    when two conditions are met:
    1. Argmax convergence: the predicted token matches between consecutive loops.
    2. Confidence gate: the top-1 softmax probability exceeds confidence_threshold.

    When early exit triggers, remaining loops are skipped entirely, providing real
    compute savings proportional to the exit rate. Uses the model's embed(),
    apply_loop(), and hidden_to_logits() methods for loop-by-loop execution.

    Setting confidence_threshold=0.0 disables early exit entirely (always runs
    all loops via a single full forward pass). Higher thresholds require more
    model confidence to exit early:
    - 0.0: early exit disabled (always full depth)
    - 0.01: exit on convergence with minimal confidence requirement
    - 0.5: exit on convergence only if the top prediction has >= 50% probability
    - 0.9: exit on convergence only if very confident

    Args:
        model: A RelaxedRecursiveTransformer module instance. Must have embed(),
            apply_loop(), and hidden_to_logits() methods.
        params: Model parameters.
        input_ids: Prompt token IDs of shape (batch, prompt_len).
        config: Full model configuration.
        max_new_tokens: Maximum number of new tokens to generate.
        confidence_threshold: Minimum top-1 softmax probability required for
            early exit. Set to 0.0 to disable early exit entirely.

    Returns:
        Tuple of (generated_ids, stats) where generated_ids has shape
        (batch, prompt_len + max_new_tokens) and stats contains exit
        depth information.
    """
    num_loops = config.recursive.num_loops
    stats = EarlyExitStats(num_loops=num_loops)
    generated = input_ids

    for step in range(max_new_tokens):
        if confidence_threshold <= 0.0 or num_loops <= 1:
            logits = model.apply(params, generated)
            next_token_logits = logits[:, -1, :]
            stats.exit_depths.append(num_loops - 1)
        else:
            next_token_logits = _generate_step_with_early_exit(
                model, params, generated, num_loops,
                confidence_threshold, stats,
            )

        next_token = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        stats.total_tokens += 1
        generated = jnp.concatenate([generated, next_token], axis=1)

    return generated, stats


def _generate_step_with_early_exit(
    model,
    params: dict,
    generated: jnp.ndarray,
    num_loops: int,
    confidence_threshold: float,
    stats: EarlyExitStats,
) -> jnp.ndarray:
    """Run one generation step with true loop-by-loop early exit.

    Embeds the input, then runs recursive loops one at a time. After each loop,
    checks for argmax convergence and confidence. When both conditions are met,
    remaining loops are skipped (real compute savings).

    Args:
        model: The transformer module.
        params: Model parameters.
        generated: Current sequence of shape (batch, seq_len).
        num_loops: Total number of recursive loops.
        confidence_threshold: Minimum confidence for early exit.
        stats: Statistics tracker (exit_depths list is appended).

    Returns:
        Next-token logits of shape (batch, vocab_size).
    """
    seq_len = generated.shape[1]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    causal_mask = causal_mask[None, None, :, :]

    hidden_states = model.apply(params, generated, method='embed')

    exit_depth = num_loops - 1
    next_token_logits = None
    prev_prediction = None

    for loop_idx in range(num_loops):
        hidden_states = model.apply(
            params, hidden_states, causal_mask, loop_idx,
            method='apply_loop',
        )

        loop_logits = model.apply(
            params, hidden_states, method='hidden_to_logits',
        )
        curr_logits = loop_logits[:, -1, :]
        curr_prediction = jnp.argmax(curr_logits, axis=-1)

        if prev_prediction is not None:
            converged = jnp.all(prev_prediction == curr_prediction)
            if converged:
                curr_probs = jax.nn.softmax(curr_logits, axis=-1)
                top_prob = jnp.max(curr_probs, axis=-1)
                if jnp.all(top_prob >= confidence_threshold):
                    exit_depth = loop_idx
                    next_token_logits = curr_logits
                    break

        prev_prediction = curr_prediction

    if next_token_logits is None:
        next_token_logits = curr_logits

    stats.exit_depths.append(int(exit_depth))
    return next_token_logits
